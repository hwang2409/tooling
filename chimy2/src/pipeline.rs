//! Programmable vertex and fragment stages.
//!
//! Transparent geometry is rendered in a second pass. Its stable sort key is
//! the view-space z coordinate of each triangle centroid. Intersecting or
//! cyclically overlapping triangles are order-approximate by design.
//!
//! [`Pipeline::render`] can render into an integer supersampled target. The
//! default scale is one, so SSAA is off. A 2x target uses four times the color
//! and depth memory and increases raster work by about four times.

use crate::clip::{ClipVertex, clip_triangle_near, cull_backface};
use crate::fb::Framebuffer;
use crate::math::{Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::raster::{
    PixelRect, RasterState, ScreenVertex, rasterize_triangle_with_sampling_state,
    rasterize_triangle_with_state, triangle_pixel_rect, viewport_transform,
};
use std::marker::PhantomData;
use std::thread;

pub const TILE_SIZE: usize = 64;

/// Values passed from a vertex stage to a fragment stage.
///
/// This trait only interpolates values with core-computed weights. It does not
/// expose raster gradients, inverse `w`, or any other raster implementation
/// detail.
pub trait Varyings: Sized {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self;

    fn lerp(a: &Self, b: &Self, amount: f32) -> Self {
        Self::lerp3(a, b, b, Vec3::new(1.0 - amount, amount, 0.0))
    }
}

/// Varying data used by a texture sampler.
///
/// This is separate from [`Varyings`]. Implementations expose only texture
/// coordinates; the raster core computes their screen-space derivatives.
pub trait SamplingVaryings: Varyings {
    fn texture_coordinates(&self) -> crate::math::Vec2;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SampleDerivatives {
    pub ddx: crate::math::Vec2,
    pub ddy: crate::math::Vec2,
}

pub trait SampledFragmentStage<V: SamplingVaryings, Uniforms> {
    fn run_with_sampling(
        &self,
        varyings: &V,
        derivatives: &SampleDerivatives,
        uniforms: &Uniforms,
    ) -> u32;
}

impl Varyings for () {
    fn lerp3(_: &Self, _: &Self, _: &Self, _: Vec3) -> Self {}
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorVarying {
    pub color: Vec4,
}

impl ColorVarying {
    pub const fn new(color: Vec4) -> Self {
        Self { color }
    }
}

impl Varyings for ColorVarying {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        Self::new(a.color * weights.x + b.color * weights.y + c.color * weights.z)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VertexOutput<V> {
    pub clip_position: Vec4,
    pub varyings: V,
}

impl<V> VertexOutput<V> {
    pub const fn new(clip_position: Vec4, varyings: V) -> Self {
        Self {
            clip_position,
            varyings,
        }
    }
}

/// A vertex stage returns homogeneous clip coordinates. After near clipping,
/// the pipeline contract requires finite `clip_position.w > 0`; perspective
/// interpolation divides by this post-clip `w` before rasterization.
pub trait VertexStage<Vertex, Uniforms> {
    type Varyings: Varyings;

    fn run(&self, vertex: &Vertex, uniforms: &Uniforms) -> VertexOutput<Self::Varyings>;
}

/// A pure fragment stage. Its output depends only on the input varying and uniforms.
///
/// Stateful closures do not meet this contract because the stage must implement
/// `Fn`, not `FnMut`:
///
/// ```compile_fail
/// use chimy2::pipeline::fragment_stage;
///
/// let mut counter = 0;
/// let _stage = fragment_stage(move |_: &(), _: &()| {
///     counter += 1;
///     0
/// });
/// ```
pub trait FragmentStage<V: Varyings, Uniforms> {
    fn run(&self, varyings: &V, uniforms: &Uniforms) -> u32;
}

pub struct VertexFn<F, V> {
    function: F,
    marker: PhantomData<fn() -> V>,
}

pub fn vertex_stage<F, V>(function: F) -> VertexFn<F, V>
where
    V: Varyings,
{
    VertexFn {
        function,
        marker: PhantomData,
    }
}

impl<F, Vertex, Uniforms, V> VertexStage<Vertex, Uniforms> for VertexFn<F, V>
where
    F: Fn(&Vertex, &Uniforms) -> VertexOutput<V>,
    V: Varyings,
{
    type Varyings = V;

    fn run(&self, vertex: &Vertex, uniforms: &Uniforms) -> VertexOutput<V> {
        (self.function)(vertex, uniforms)
    }
}

pub struct FragmentFn<F, V> {
    function: F,
    marker: PhantomData<fn() -> V>,
}

pub fn fragment_stage<F, V>(function: F) -> FragmentFn<F, V>
where
    V: Varyings,
{
    FragmentFn {
        function,
        marker: PhantomData,
    }
}

impl<F, V, Uniforms> FragmentStage<V, Uniforms> for FragmentFn<F, V>
where
    F: Fn(&V, &Uniforms) -> u32,
    V: Varyings,
{
    fn run(&self, varyings: &V, uniforms: &Uniforms) -> u32 {
        (self.function)(varyings, uniforms)
    }
}

pub struct Pipeline<VS, FS> {
    pub vertex: VS,
    pub fragment: FS,
    thread_count: usize,
    ssaa_scale: usize,
}

impl<VS, FS> Pipeline<VS, FS> {
    pub fn new(vertex: VS, fragment: FS) -> Self {
        Self {
            vertex,
            fragment,
            thread_count: default_thread_count(),
            ssaa_scale: 1,
        }
    }

    /// Sets the number of tile workers used by later draw calls.
    ///
    /// The default is `CHIMY_THREADS` when set to a positive integer. If the
    /// variable is absent or invalid, the default is
    /// `std::thread::available_parallelism()`. A value of one selects the
    /// serial path exactly.
    pub fn set_thread_count(&mut self, count: usize) {
        self.thread_count = count.max(1);
    }

    pub const fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// Enables integer supersampling for [`Pipeline::render`]. A scale of one
    /// keeps rendering at the presented size. Two is the usual quality and
    /// cost tradeoff, using four times the color and depth memory.
    pub fn set_ssaa_scale(&mut self, scale: usize) {
        self.ssaa_scale = scale.max(1);
    }

    pub const fn ssaa_scale(&self) -> usize {
        self.ssaa_scale
    }

    /// Renders a frame through an optional supersampled internal target.
    ///
    /// The callback receives a frame queue. Opaque draws execute on submit;
    /// transparent draws flush globally, back-to-front, at callback end.
    /// Downsampling uses a linear-light box filter, then encodes once.
    pub fn render<'a, F>(&'a mut self, framebuffer: &mut Framebuffer, draw: F)
    where
        F: FnOnce(&mut RenderFrame<'a, VS, FS>, &mut Framebuffer),
    {
        if self.ssaa_scale <= 1 {
            let mut frame = RenderFrame::new(self);
            draw(&mut frame, framebuffer);
            frame.flush(framebuffer);
            return;
        }
        let width = framebuffer.width.saturating_mul(self.ssaa_scale);
        let height = framebuffer.height.saturating_mul(self.ssaa_scale);
        let mut internal = Framebuffer::new(width, height);
        let mut frame = RenderFrame::new(self);
        draw(&mut frame, &mut internal);
        frame.flush(&mut internal);
        internal.downsample_linear_into(framebuffer);
    }
}

fn default_thread_count() -> usize {
    std::env::var("CHIMY_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .or_else(|| thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
}

fn sort_triangles_back_to_front(triangles: &[[usize; 3]], sort_keys: &[f32]) -> Vec<[usize; 3]> {
    let mut indexed = triangles.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(left_index, _), (right_index, _)| {
        let left_key = sort_keys.get(*left_index).copied().unwrap_or(f32::INFINITY);
        let right_key = sort_keys
            .get(*right_index)
            .copied()
            .unwrap_or(f32::INFINITY);
        compare_depth_keys(left_key, right_key).then_with(|| left_index.cmp(right_index))
    });
    indexed.into_iter().map(|(_, triangle)| triangle).collect()
}

fn compare_depth_keys(left: f32, right: f32) -> std::cmp::Ordering {
    left.total_cmp(&right)
}

fn mesh_centroid_depths(mesh: &Mesh, model_view: crate::math::Mat4) -> Vec<f32> {
    mesh.triangles
        .iter()
        .map(|&[a, b, c]| {
            let centroid =
                (mesh.vertices[a].position + mesh.vertices[b].position + mesh.vertices[c].position)
                    / 3.0;
            let position = model_view * Vec4::new(centroid.x, centroid.y, centroid.z, 1.0);
            let z = position.z / position.w;
            if z.is_nan() { f32::INFINITY } else { z }
        })
        .collect()
}

type QueuedDraw<'a, FS> = Box<dyn FnOnce(&mut Framebuffer, &FS) + 'a>;
type RasterFn<V, FS, Uniforms> =
    fn(&mut Framebuffer, [ScreenVertex<V>; 3], &FS, &Uniforms, RasterState);

struct QueuedTransparent<'a, FS> {
    key: f32,
    submission_order: usize,
    draw: QueuedDraw<'a, FS>,
}

/// A frame records transparent draws until the callback ends.
///
/// Opaque draws execute when submitted. The frame then sorts every queued
/// transparent triangle by ascending view-space depth and renders them.
pub struct RenderFrame<'a, VS, FS> {
    pipeline: &'a mut Pipeline<VS, FS>,
    transparent: Vec<QueuedTransparent<'a, FS>>,
    next_submission_order: usize,
}

impl<'a, VS, FS> RenderFrame<'a, VS, FS> {
    fn new(pipeline: &'a mut Pipeline<VS, FS>) -> Self {
        Self {
            pipeline,
            transparent: Vec::new(),
            next_submission_order: 0,
        }
    }

    fn flush(&mut self, framebuffer: &mut Framebuffer) {
        self.transparent.sort_by(|left, right| {
            compare_depth_keys(left.key, right.key)
                .then_with(|| left.submission_order.cmp(&right.submission_order))
        });
        for command in self.transparent.drain(..) {
            (command.draw)(framebuffer, &self.pipeline.fragment);
        }
    }
}

impl<'a, VS, FS> RenderFrame<'a, VS, FS> {
    pub fn draw<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync + 'a,
    {
        self.pipeline
            .draw(framebuffer, vertices, triangles, uniforms);
    }

    pub fn draw_with_sampling<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync,
    {
        self.pipeline
            .draw_with_sampling(framebuffer, vertices, triangles, uniforms);
    }

    pub fn draw_mesh<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync,
    {
        self.pipeline.draw_mesh(framebuffer, mesh, uniforms);
    }

    pub fn draw_mesh_with_sampling<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync,
    {
        self.pipeline
            .draw_mesh_with_sampling(framebuffer, mesh, uniforms);
    }

    fn queue_transparent<Vertex, Uniforms>(
        &mut self,
        framebuffer: &Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        sort_keys: &[f32],
        uniforms: &'a Uniforms,
        rasterize: RasterFn<VS::Varyings, FS, Uniforms>,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync + 'a,
    {
        let prepared = prepare_triangles_with_keys(
            &self.pipeline.vertex,
            framebuffer,
            vertices,
            triangles,
            sort_keys,
            uniforms,
        );
        for (key, triangle) in prepared {
            let submission_order = self.next_submission_order;
            self.next_submission_order += 1;
            let draw = Box::new(move |framebuffer: &mut Framebuffer, fragment: &FS| {
                rasterize(
                    framebuffer,
                    triangle.vertices,
                    fragment,
                    uniforms,
                    RasterState::TRANSPARENT,
                );
            });
            self.transparent.push(QueuedTransparent {
                key,
                submission_order,
                draw,
            });
        }
    }

    pub fn draw_transparent<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        sort_keys: &[f32],
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync + 'a,
    {
        self.queue_transparent(
            framebuffer,
            vertices,
            triangles,
            sort_keys,
            uniforms,
            rasterize_plain_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_transparent_with_sampling<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        sort_keys: &[f32],
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync + 'a,
    {
        self.queue_transparent(
            framebuffer,
            vertices,
            triangles,
            sort_keys,
            uniforms,
            rasterize_sampled_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_mesh_transparent<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        model_view: crate::math::Mat4,
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync + 'a,
    {
        let sort_keys = mesh_centroid_depths(mesh, model_view);
        self.draw_transparent(
            framebuffer,
            &mesh.vertices,
            &mesh.triangles,
            &sort_keys,
            uniforms,
        );
    }

    pub fn draw_mesh_transparent_with_sampling<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        model_view: crate::math::Mat4,
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync + 'a,
    {
        let sort_keys = mesh_centroid_depths(mesh, model_view);
        self.draw_transparent_with_sampling(
            framebuffer,
            &mesh.vertices,
            &mesh.triangles,
            &sort_keys,
            uniforms,
        );
    }
}

impl<VS, FS> Pipeline<VS, FS> {
    /// Draws indexed triangles. Vertex outputs are viewport-transformed before
    /// rasterization, which keeps clipping and interpolation inside the core.
    pub fn draw<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync,
    {
        let prepared = if self.thread_count <= 1 {
            prepare_triangles(&self.vertex, framebuffer, vertices, triangles, uniforms)
        } else {
            prepare_triangles_parallel(
                &self.vertex,
                framebuffer,
                vertices,
                triangles,
                uniforms,
                self.thread_count,
            )
        };
        self.draw_prepared(
            framebuffer,
            prepared,
            uniforms,
            RasterState::OPAQUE,
            rasterize_plain_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    /// Draws a back-to-front transparent pass with depth testing enabled and
    /// depth writes disabled. `sort_keys` contains one view-space centroid z
    /// value per input triangle. Smaller z values are farther from a camera
    /// looking down -Z. Stable sorting uses the input index as a total-order
    /// tie breaker, including for non-finite keys.
    pub fn draw_transparent<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        sort_keys: &[f32],
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync,
    {
        let sorted = sort_triangles_back_to_front(triangles, sort_keys);
        let prepared = if self.thread_count <= 1 {
            prepare_triangles(&self.vertex, framebuffer, vertices, &sorted, uniforms)
        } else {
            prepare_triangles_parallel(
                &self.vertex,
                framebuffer,
                vertices,
                &sorted,
                uniforms,
                self.thread_count,
            )
        };
        self.draw_prepared(
            framebuffer,
            prepared,
            uniforms,
            RasterState::TRANSPARENT,
            rasterize_plain_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_mesh_transparent<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        model_view: crate::math::Mat4,
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync,
    {
        let sort_keys = mesh_centroid_depths(mesh, model_view);
        self.draw_transparent(
            framebuffer,
            &mesh.vertices,
            &mesh.triangles,
            &sort_keys,
            uniforms,
        );
    }

    /// Draws indexed triangles into the depth buffer without changing color.
    /// The preparation and raster kernel are shared with color draws.
    pub fn draw_depth<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync,
    {
        let prepared = if self.thread_count <= 1 {
            prepare_triangles(&self.vertex, framebuffer, vertices, triangles, uniforms)
        } else {
            prepare_triangles_parallel(
                &self.vertex,
                framebuffer,
                vertices,
                triangles,
                uniforms,
                self.thread_count,
            )
        };
        let rasterize = rasterize_depth_triangle::<VS::Varyings, FS, Uniforms>;
        if self.thread_count <= 1 || prepared.is_empty() {
            self.draw_serial(
                framebuffer,
                prepared,
                uniforms,
                RasterState::DEPTH_ONLY,
                rasterize,
            );
        } else {
            self.draw_parallel(
                framebuffer,
                &prepared,
                uniforms,
                RasterState::DEPTH_ONLY,
                rasterize,
            );
        }
    }

    pub fn draw_mesh_depth<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync,
    {
        self.draw_depth(framebuffer, mesh.vertices(), mesh.indices(), uniforms);
    }

    fn draw_serial<V, Uniforms>(
        &self,
        framebuffer: &mut Framebuffer,
        prepared: Vec<PreparedTriangle<V>>,
        uniforms: &Uniforms,
        state: RasterState,
        rasterize: fn(&mut Framebuffer, [ScreenVertex<V>; 3], &FS, &Uniforms, RasterState),
    ) where
        V: Varyings + Clone,
    {
        for triangle in prepared {
            rasterize(
                framebuffer,
                triangle.vertices,
                &self.fragment,
                uniforms,
                state,
            );
        }
    }

    fn draw_prepared<V, Uniforms>(
        &self,
        framebuffer: &mut Framebuffer,
        prepared: Vec<PreparedTriangle<V>>,
        uniforms: &Uniforms,
        state: RasterState,
        rasterize: fn(&mut Framebuffer, [ScreenVertex<V>; 3], &FS, &Uniforms, RasterState),
    ) where
        FS: Sync,
        Uniforms: Sync,
        V: Varyings + Clone + Send + Sync,
    {
        if self.thread_count <= 1 || prepared.is_empty() {
            self.draw_serial(framebuffer, prepared, uniforms, state, rasterize);
        } else {
            self.draw_parallel(framebuffer, &prepared, uniforms, state, rasterize);
        }
    }

    fn draw_parallel<V, Uniforms>(
        &self,
        framebuffer: &mut Framebuffer,
        prepared: &[PreparedTriangle<V>],
        uniforms: &Uniforms,
        state: RasterState,
        rasterize: fn(&mut Framebuffer, [ScreenVertex<V>; 3], &FS, &Uniforms, RasterState),
    ) where
        FS: Sync,
        Uniforms: Sync,
        V: Varyings + Clone + Send + Sync,
    {
        let tiles = make_tiles(framebuffer.width, framebuffer.height, prepared);
        if tiles.is_empty() {
            return;
        }
        let worker_count = self.thread_count.min(tiles.len()).max(1);
        let source = &*framebuffer;
        let tiles = &tiles;
        let results = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for worker_index in 0..worker_count {
                let fragment = &self.fragment;
                handles.push(scope.spawn(move || {
                    let mut results = Vec::new();
                    for tile_index in (worker_index..tiles.len()).step_by(worker_count) {
                        results.push(rasterize_tile(
                            &tiles[tile_index],
                            prepared,
                            source,
                            uniforms,
                            fragment,
                            state,
                            rasterize,
                        ));
                    }
                    results
                }));
            }
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("tile worker panicked"))
                .collect::<Vec<_>>()
        });

        for result in results {
            for row in 0..result.framebuffer.height {
                let destination_start = (result.y + row) * framebuffer.width + result.x;
                let destination_end = destination_start + result.framebuffer.width;
                let source_start = row * result.framebuffer.width;
                let source_end = source_start + result.framebuffer.width;
                framebuffer.color[destination_start..destination_end]
                    .copy_from_slice(&result.framebuffer.color[source_start..source_end]);
                framebuffer.depth[destination_start..destination_end]
                    .copy_from_slice(&result.framebuffer.depth[source_start..source_end]);
            }
        }
    }

    pub fn draw_with_sampling<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync,
    {
        let prepared = if self.thread_count <= 1 || triangles.len() < 2 {
            prepare_triangles(&self.vertex, framebuffer, vertices, triangles, uniforms)
        } else {
            prepare_triangles_parallel(
                &self.vertex,
                framebuffer,
                vertices,
                triangles,
                uniforms,
                self.thread_count,
            )
        };
        self.draw_prepared(
            framebuffer,
            prepared,
            uniforms,
            RasterState::OPAQUE,
            rasterize_sampled_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_transparent_with_sampling<Vertex, Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        sort_keys: &[f32],
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync,
    {
        let sorted = sort_triangles_back_to_front(triangles, sort_keys);
        let prepared = if self.thread_count <= 1 || sorted.len() < 2 {
            prepare_triangles(&self.vertex, framebuffer, vertices, &sorted, uniforms)
        } else {
            prepare_triangles_parallel(
                &self.vertex,
                framebuffer,
                vertices,
                &sorted,
                uniforms,
                self.thread_count,
            )
        };
        self.draw_prepared(
            framebuffer,
            prepared,
            uniforms,
            RasterState::TRANSPARENT,
            rasterize_sampled_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_mesh<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync,
    {
        self.draw(framebuffer, mesh.vertices(), mesh.indices(), uniforms);
    }

    pub fn draw_mesh_with_sampling<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync,
    {
        self.draw_with_sampling(framebuffer, mesh.vertices(), mesh.indices(), uniforms);
    }

    pub fn draw_mesh_transparent_with_sampling<Uniforms>(
        &mut self,
        framebuffer: &mut Framebuffer,
        mesh: &Mesh,
        model_view: crate::math::Mat4,
        uniforms: &Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync,
    {
        let sort_keys = mesh_centroid_depths(mesh, model_view);
        self.draw_transparent_with_sampling(
            framebuffer,
            &mesh.vertices,
            &mesh.triangles,
            &sort_keys,
            uniforms,
        );
    }
}

#[derive(Clone)]
struct PreparedTriangle<V> {
    vertices: [ScreenVertex<V>; 3],
    bounds: PixelRect,
}

struct Tile {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    triangles: Vec<usize>,
}

struct TileResult {
    x: usize,
    y: usize,
    framebuffer: Framebuffer,
}

fn make_tiles<V>(width: usize, height: usize, prepared: &[PreparedTriangle<V>]) -> Vec<Tile> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let tiles_x = width.div_ceil(TILE_SIZE);
    let tiles_y = height.div_ceil(TILE_SIZE);
    let mut bins: Vec<Vec<usize>> = (0..tiles_x * tiles_y).map(|_| Vec::new()).collect();
    for (triangle_index, triangle) in prepared.iter().enumerate() {
        let min_tile_x = triangle.bounds.min_x as usize / TILE_SIZE;
        let max_tile_x = triangle.bounds.max_x as usize / TILE_SIZE;
        let min_tile_y = triangle.bounds.min_y as usize / TILE_SIZE;
        let max_tile_y = triangle.bounds.max_y as usize / TILE_SIZE;
        for tile_y in min_tile_y..=max_tile_y {
            for tile_x in min_tile_x..=max_tile_x {
                bins[tile_y * tiles_x + tile_x].push(triangle_index);
            }
        }
    }

    bins.into_iter()
        .enumerate()
        .filter_map(|(index, triangles)| {
            if triangles.is_empty() {
                return None;
            }
            let tile_x = index % tiles_x;
            let tile_y = index / tiles_x;
            Some(Tile {
                x: tile_x * TILE_SIZE,
                y: tile_y * TILE_SIZE,
                width: (width - tile_x * TILE_SIZE).min(TILE_SIZE),
                height: (height - tile_y * TILE_SIZE).min(TILE_SIZE),
                triangles,
            })
        })
        .collect()
}

fn rasterize_tile<V, FS, Uniforms>(
    tile: &Tile,
    prepared: &[PreparedTriangle<V>],
    source: &Framebuffer,
    uniforms: &Uniforms,
    fragment: &FS,
    state: RasterState,
    rasterize: fn(&mut Framebuffer, [ScreenVertex<V>; 3], &FS, &Uniforms, RasterState),
) -> TileResult
where
    V: Varyings + Clone,
{
    let mut framebuffer = Framebuffer::new(tile.width, tile.height);
    for row in 0..tile.height {
        let source_start = (tile.y + row) * source.width + tile.x;
        let source_end = source_start + tile.width;
        let destination_start = row * tile.width;
        let destination_end = destination_start + tile.width;
        framebuffer.color[destination_start..destination_end]
            .copy_from_slice(&source.color[source_start..source_end]);
        framebuffer.depth[destination_start..destination_end]
            .copy_from_slice(&source.depth[source_start..source_end]);
    }

    for &triangle_index in &tile.triangles {
        let mut triangle = prepared[triangle_index].vertices.clone();
        for vertex in &mut triangle {
            vertex.position.x -= tile.x as f32;
            vertex.position.y -= tile.y as f32;
        }
        rasterize(&mut framebuffer, triangle, fragment, uniforms, state);
    }

    TileResult {
        x: tile.x,
        y: tile.y,
        framebuffer,
    }
}

fn rasterize_plain_triangle<V, FS, Uniforms>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    fragment: &FS,
    uniforms: &Uniforms,
    state: RasterState,
) where
    V: Varyings,
    FS: FragmentStage<V, Uniforms>,
{
    rasterize_triangle_with_state(framebuffer, vertices, state, |varyings| {
        fragment.run(&varyings, uniforms)
    });
}

fn rasterize_depth_triangle<V, FS, Uniforms>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    _: &FS,
    _: &Uniforms,
    state: RasterState,
) where
    V: Varyings,
{
    rasterize_triangle_with_state(framebuffer, vertices, state, |_| 0);
}

fn rasterize_sampled_triangle<V, FS, Uniforms>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    fragment: &FS,
    uniforms: &Uniforms,
    state: RasterState,
) where
    V: SamplingVaryings,
    FS: SampledFragmentStage<V, Uniforms>,
{
    rasterize_triangle_with_sampling_state(
        framebuffer,
        vertices,
        state,
        |varyings, derivatives| fragment.run_with_sampling(&varyings, &derivatives, uniforms),
    );
}

fn prepare_triangles<Vertex, Uniforms, VS>(
    vertex_stage: &VS,
    framebuffer: &Framebuffer,
    vertices: &[Vertex],
    triangles: &[[usize; 3]],
    uniforms: &Uniforms,
) -> Vec<PreparedTriangle<VS::Varyings>>
where
    VS: VertexStage<Vertex, Uniforms>,
    VS::Varyings: Clone,
{
    prepare_triangles_with_keys(
        vertex_stage,
        framebuffer,
        vertices,
        triangles,
        &[],
        uniforms,
    )
    .into_iter()
    .map(|(_, triangle)| triangle)
    .collect()
}

fn prepare_triangles_with_keys<Vertex, Uniforms, VS>(
    vertex_stage: &VS,
    framebuffer: &Framebuffer,
    vertices: &[Vertex],
    triangles: &[[usize; 3]],
    sort_keys: &[f32],
    uniforms: &Uniforms,
) -> Vec<(f32, PreparedTriangle<VS::Varyings>)>
where
    VS: VertexStage<Vertex, Uniforms>,
    VS::Varyings: Clone,
{
    let mut prepared = Vec::new();
    for (triangle_index, &[a, b, c]) in triangles.iter().enumerate() {
        let sort_key = sort_keys
            .get(triangle_index)
            .copied()
            .unwrap_or(f32::INFINITY);
        let Some(vertex_a) = vertices.get(a) else {
            continue;
        };
        let Some(vertex_b) = vertices.get(b) else {
            continue;
        };
        let Some(vertex_c) = vertices.get(c) else {
            continue;
        };
        let output_a = vertex_stage.run(vertex_a, uniforms);
        let output_b = vertex_stage.run(vertex_b, uniforms);
        let output_c = vertex_stage.run(vertex_c, uniforms);
        let clipped = clip_triangle_near([
            ClipVertex::new(output_a.clip_position, output_a.varyings),
            ClipVertex::new(output_b.clip_position, output_b.varyings),
            ClipVertex::new(output_c.clip_position, output_c.varyings),
        ]);
        for triangle in clipped.iter() {
            if cull_backface(triangle) {
                continue;
            }
            debug_assert!(
                triangle
                    .iter()
                    .all(|vertex| { vertex.position.w.is_finite() && vertex.position.w > 0.0 }),
                "post-clip vertex w must be finite and positive"
            );
            let Some(position_a) =
                viewport_transform(triangle[0].position, framebuffer.width, framebuffer.height)
            else {
                continue;
            };
            let Some(position_b) =
                viewport_transform(triangle[1].position, framebuffer.width, framebuffer.height)
            else {
                continue;
            };
            let Some(position_c) =
                viewport_transform(triangle[2].position, framebuffer.width, framebuffer.height)
            else {
                continue;
            };
            let vertices = [
                ScreenVertex::with_inverse_w(
                    position_a,
                    1.0 / triangle[0].position.w,
                    triangle[0].varyings.clone(),
                ),
                ScreenVertex::with_inverse_w(
                    position_b,
                    1.0 / triangle[1].position.w,
                    triangle[1].varyings.clone(),
                ),
                ScreenVertex::with_inverse_w(
                    position_c,
                    1.0 / triangle[2].position.w,
                    triangle[2].varyings.clone(),
                ),
            ];
            let bounds = triangle_pixel_rect(&vertices, framebuffer.width, framebuffer.height);
            if !bounds.is_empty() {
                prepared.push((sort_key, PreparedTriangle { vertices, bounds }));
            }
        }
    }
    prepared
}

fn prepare_triangles_parallel<Vertex, Uniforms, VS>(
    vertex_stage: &VS,
    framebuffer: &Framebuffer,
    vertices: &[Vertex],
    triangles: &[[usize; 3]],
    uniforms: &Uniforms,
    thread_count: usize,
) -> Vec<PreparedTriangle<VS::Varyings>>
where
    VS: VertexStage<Vertex, Uniforms> + Sync,
    Vertex: Sync,
    Uniforms: Sync,
    VS::Varyings: Clone + Send,
{
    if triangles.is_empty() {
        return Vec::new();
    }
    let chunk_count = thread_count.min(triangles.len()).max(1);
    let chunk_size = triangles.len().div_ceil(chunk_count);
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunk_count);
        for chunk in triangles.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                prepare_triangles(vertex_stage, framebuffer, vertices, chunk, uniforms)
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("front-end worker panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::argb8888;

    #[test]
    fn closure_stages_draw_a_triangle() {
        let mut pipeline = Pipeline::new(
            vertex_stage(|vertex: &Vec4, _: &()| VertexOutput::new(*vertex, ())),
            fragment_stage(|_: &(), _: &()| argb8888(255, 20, 40, 60)),
        );
        let mut framebuffer = Framebuffer::new(4, 4);
        pipeline.draw(
            &mut framebuffer,
            &[
                Vec4::new(-1.0, -1.0, 0.0, 1.0),
                Vec4::new(1.0, -1.0, 0.0, 1.0),
                Vec4::new(-1.0, 1.0, 0.0, 1.0),
            ],
            &[[0, 1, 2]],
            &(),
        );
        assert!(framebuffer.color.contains(&argb8888(255, 20, 40, 60)));
    }

    #[test]
    fn depth_draw_preserves_color() {
        let mut pipeline = Pipeline::new(
            vertex_stage(|vertex: &Vec4, _: &()| VertexOutput::new(*vertex, ())),
            (),
        );
        let mut framebuffer = Framebuffer::new(4, 4);
        framebuffer.clear(argb8888(255, 10, 20, 30));
        pipeline.draw_depth(
            &mut framebuffer,
            &[
                Vec4::new(-1.0, -1.0, 0.0, 1.0),
                Vec4::new(1.0, -1.0, 0.0, 1.0),
                Vec4::new(-1.0, 1.0, 0.0, 1.0),
            ],
            &[[0, 1, 2]],
            &(),
        );
        assert!(framebuffer.depth.iter().any(|&depth| depth < 1.0));
        assert!(
            framebuffer
                .color
                .iter()
                .all(|&color| color == argb8888(255, 10, 20, 30))
        );
    }

    #[test]
    fn color_varying_interpolates_components() {
        let result = ColorVarying::lerp3(
            &ColorVarying::new(Vec4::new(1.0, 0.0, 0.0, 1.0)),
            &ColorVarying::new(Vec4::new(0.0, 1.0, 0.0, 1.0)),
            &ColorVarying::new(Vec4::new(0.0, 0.0, 1.0, 1.0)),
            Vec3::new(0.25, 0.5, 0.25),
        );
        assert_eq!(result.color, Vec4::new(0.25, 0.5, 0.25, 1.0));
    }

    #[test]
    fn mesh_centroid_depth_uses_model_view() {
        let mesh = Mesh {
            vertices: vec![
                MeshVertex {
                    position: crate::math::Vec3::new(0.0, 0.0, 0.0),
                    texcoord: None,
                    normal: None,
                },
                MeshVertex {
                    position: crate::math::Vec3::new(1.0, 0.0, 0.0),
                    texcoord: None,
                    normal: None,
                },
                MeshVertex {
                    position: crate::math::Vec3::new(1.0, 1.0, 0.0),
                    texcoord: None,
                    normal: None,
                },
            ],
            triangles: vec![[0, 1, 2]],
        };
        let identity_key = mesh_centroid_depths(&mesh, crate::math::Mat4::IDENTITY)[0];
        let rotated_key = mesh_centroid_depths(
            &mesh,
            crate::math::Mat4::rotate(
                crate::math::Vec3::new(0.0, 1.0, 0.0),
                std::f32::consts::FRAC_PI_2,
            ),
        )[0];
        assert_ne!(identity_key, rotated_key);
    }
}
