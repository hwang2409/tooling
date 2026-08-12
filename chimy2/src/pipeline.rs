//! Programmable vertex and fragment stages.
//!
//! All color draws use one prepare, classify, schedule, and raster path.
//! Transparent draws are sorted by view-space centroid depth at frame flush.

use crate::camera::Camera;
use crate::clip::{ClipVertex, clip_triangle_near, cull_backface};
use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::postfx::{PostChain, sanitize_exposure};
use crate::raster::{
    FragmentColor, PixelRect, RasterState, ScreenVertex, rasterize_triangle_with_sampling_state,
    rasterize_triangle_with_state, triangle_pixel_rect, viewport_transform,
};
use crate::skybox::{CubeTexture, render_skybox_with_threads};
use std::marker::PhantomData;
use std::thread;

pub const TILE_SIZE: usize = 64;

pub trait Varyings: Sized {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self;

    fn lerp(a: &Self, b: &Self, amount: f32) -> Self {
        Self::lerp3(a, b, b, Vec3::new(1.0 - amount, amount, 0.0))
    }
}

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

    fn run_linear_with_sampling(
        &self,
        varyings: &V,
        derivatives: &SampleDerivatives,
        uniforms: &Uniforms,
    ) -> [f32; 4] {
        crate::fb::linear_rgba_from_argb8888(self.run_with_sampling(
            varyings,
            derivatives,
            uniforms,
        ))
    }

    /// Returns whether this material has fully opaque coverage.
    fn is_opaque(&self, _uniforms: &Uniforms) -> bool {
        true
    }

    /// Returns the matrix used to compute transparent mesh centroid keys.
    fn model_view(&self, _uniforms: &Uniforms) -> Option<Mat4> {
        None
    }
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
    pub view_position: Vec4,
    pub varyings: V,
}

impl<V> VertexOutput<V> {
    pub const fn new(clip_position: Vec4, varyings: V) -> Self {
        Self {
            clip_position,
            view_position: clip_position,
            varyings,
        }
    }

    pub const fn with_view_position(clip_position: Vec4, view_position: Vec4, varyings: V) -> Self {
        Self {
            clip_position,
            view_position,
            varyings,
        }
    }
}

pub trait VertexStage<Vertex, Uniforms> {
    type Varyings: Varyings;

    fn run(&self, vertex: &Vertex, uniforms: &Uniforms) -> VertexOutput<Self::Varyings>;
}

/// A pure fragment stage. The default material classification is opaque.
/// Shader implementations override it when alpha or texture coverage exists.
pub trait FragmentStage<V: Varyings, Uniforms> {
    fn run(&self, varyings: &V, uniforms: &Uniforms) -> u32;

    fn run_linear(&self, varyings: &V, uniforms: &Uniforms) -> [f32; 4] {
        crate::fb::linear_rgba_from_argb8888(self.run(varyings, uniforms))
    }

    fn is_opaque(&self, _uniforms: &Uniforms) -> bool {
        true
    }

    fn model_view(&self, _uniforms: &Uniforms) -> Option<Mat4> {
        None
    }
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
    post_chain: PostChain,
    hdr: bool,
    exposure: f32,
}

impl<VS, FS> Pipeline<VS, FS> {
    pub fn new(vertex: VS, fragment: FS) -> Self {
        Self {
            vertex,
            fragment,
            thread_count: default_thread_count(),
            ssaa_scale: 1,
            post_chain: PostChain::new(),
            hdr: false,
            exposure: 1.0,
        }
    }

    pub fn set_thread_count(&mut self, count: usize) {
        self.thread_count = count.max(1);
    }

    pub const fn thread_count(&self) -> usize {
        self.thread_count
    }

    pub fn set_ssaa_scale(&mut self, scale: usize) {
        self.ssaa_scale = scale.max(1);
    }

    pub const fn ssaa_scale(&self) -> usize {
        self.ssaa_scale
    }

    pub fn set_hdr(&mut self, hdr: bool) {
        self.hdr = hdr;
    }

    pub const fn hdr(&self) -> bool {
        self.hdr
    }

    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = sanitize_exposure(exposure);
    }

    pub const fn exposure(&self) -> f32 {
        self.exposure
    }

    /// Replaces the optional post chain. The chain runs after SSAA downsample.
    pub fn set_post_chain(&mut self, post_chain: PostChain) {
        self.post_chain = post_chain;
    }

    pub fn post_chain(&self) -> &PostChain {
        &self.post_chain
    }

    pub fn post_chain_mut(&mut self) -> &mut PostChain {
        &mut self.post_chain
    }

    pub fn clear_post_chain(&mut self) {
        self.post_chain.clear();
    }

    /// Renders one frame. The callback submits prepared draws to one queue.
    /// Flush renders every opaque draw first, then sorted transparent draws.
    pub fn render<'a, F>(&'a mut self, framebuffer: &mut Framebuffer, draw: F)
    where
        F: FnOnce(&mut RenderFrame<'a, VS, FS>, &mut Framebuffer),
    {
        if self.hdr {
            framebuffer.enable_hdr();
        }
        if self.ssaa_scale <= 1 {
            let mut frame = RenderFrame::new(self);
            draw(&mut frame, framebuffer);
            frame.flush(framebuffer);
            frame.apply_post_chain(framebuffer);
            return;
        }
        let width = framebuffer.width.saturating_mul(self.ssaa_scale);
        let height = framebuffer.height.saturating_mul(self.ssaa_scale);
        let mut internal = Framebuffer::new(width, height);
        if self.hdr {
            internal.enable_hdr();
        }
        let mut frame = RenderFrame::new(self);
        draw(&mut frame, &mut internal);
        frame.flush(&mut internal);
        internal.downsample_linear_into(framebuffer);
        frame.apply_post_chain(framebuffer);
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

fn compare_depth_keys(left: f32, right: f32) -> std::cmp::Ordering {
    left.total_cmp(&right)
}

fn mesh_centroid_depths(mesh: &Mesh, model_view: Mat4) -> Vec<f32> {
    mesh.indices()
        .iter()
        .map(|&[a, b, c]| {
            let centroid = (mesh.vertex(a).unwrap().position()
                + mesh.vertex(b).unwrap().position()
                + mesh.vertex(c).unwrap().position())
                / 3.0;
            let position = model_view * Vec4::new(centroid.x, centroid.y, centroid.z, 1.0);
            let z = position.z / position.w;
            if z.is_nan() { f32::INFINITY } else { z }
        })
        .collect()
}

type RasterFn<V, FS, Uniforms> =
    fn(&mut Framebuffer, [ScreenVertex<V>; 3], &FS, &Uniforms, RasterState);
type QueuedDraw<'a, FS> = Box<dyn FnOnce(&mut Framebuffer, &FS, usize) + 'a>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DrawClass {
    Opaque,
    Transparent,
}

struct QueuedCommand<'a, FS> {
    class: DrawClass,
    key: f32,
    submission_order: usize,
    draw: QueuedDraw<'a, FS>,
}

/// A frame queue stores classified, prepared draws until flush.
pub struct RenderFrame<'a, VS, FS> {
    pipeline: &'a mut Pipeline<VS, FS>,
    commands: Vec<QueuedCommand<'a, FS>>,
    next_submission_order: usize,
    skybox: Option<(&'a CubeTexture, Camera)>,
}

impl<'a, VS, FS> RenderFrame<'a, VS, FS> {
    fn new(pipeline: &'a mut Pipeline<VS, FS>) -> Self {
        Self {
            pipeline,
            commands: Vec::new(),
            next_submission_order: 0,
            skybox: None,
        }
    }

    fn queue_prepared<V, Uniforms>(
        &mut self,
        prepared: Vec<(f32, PreparedTriangle<V>)>,
        uniforms: &'a Uniforms,
        class: DrawClass,
        rasterize: RasterFn<V, FS, Uniforms>,
    ) where
        FS: Sync,
        Uniforms: Sync,
        V: Varyings + Clone + Send + Sync + 'a,
    {
        if class == DrawClass::Opaque {
            if prepared.is_empty() {
                return;
            }
            let prepared = prepared
                .into_iter()
                .map(|(_, triangle)| triangle)
                .collect::<Vec<_>>();
            let submission_order = self.next_submission_order;
            self.next_submission_order += 1;
            let draw = Box::new(
                move |framebuffer: &mut Framebuffer, fragment: &FS, threads| {
                    dispatch_prepared(
                        threads,
                        framebuffer,
                        &prepared,
                        uniforms,
                        fragment,
                        class.raster_state(),
                        rasterize,
                    );
                },
            );
            self.commands.push(QueuedCommand {
                class,
                key: 0.0,
                submission_order,
                draw,
            });
            return;
        }
        for (key, triangle) in prepared {
            let submission_order = self.next_submission_order;
            self.next_submission_order += 1;
            let triangle = vec![triangle];
            let draw = Box::new(
                move |framebuffer: &mut Framebuffer, fragment: &FS, threads| {
                    dispatch_prepared(
                        threads,
                        framebuffer,
                        &triangle,
                        uniforms,
                        fragment,
                        class.raster_state(),
                        rasterize,
                    );
                },
            );
            self.commands.push(QueuedCommand {
                class,
                key,
                submission_order,
                draw,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn queue<Vertex, Uniforms>(
        &mut self,
        framebuffer: &Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        sort_keys: Option<&[f32]>,
        uniforms: &'a Uniforms,
        class: DrawClass,
        rasterize: RasterFn<VS::Varyings, FS, Uniforms>,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync + 'a,
    {
        let prepared = prepare_triangles(
            &self.pipeline.vertex,
            framebuffer,
            vertices,
            triangles,
            sort_keys,
            uniforms,
        );
        self.queue_prepared(prepared, uniforms, class, rasterize);
    }

    fn flush(&mut self, framebuffer: &mut Framebuffer) {
        self.commands
            .sort_by(|left, right| match (left.class, right.class) {
                (DrawClass::Opaque, DrawClass::Transparent) => std::cmp::Ordering::Less,
                (DrawClass::Transparent, DrawClass::Opaque) => std::cmp::Ordering::Greater,
                (DrawClass::Opaque, DrawClass::Opaque) => {
                    left.submission_order.cmp(&right.submission_order)
                }
                (DrawClass::Transparent, DrawClass::Transparent) => {
                    compare_depth_keys(left.key, right.key)
                        .then_with(|| left.submission_order.cmp(&right.submission_order))
                }
            });
        let transparent_start = self
            .commands
            .iter()
            .position(|command| command.class == DrawClass::Transparent)
            .unwrap_or(self.commands.len());
        for command in self.commands.drain(..transparent_start) {
            (command.draw)(
                framebuffer,
                &self.pipeline.fragment,
                self.pipeline.thread_count,
            );
        }
        if let Some((cube, camera)) = self.skybox {
            render_skybox_with_threads(framebuffer, camera, cube, self.pipeline.thread_count);
        }
        for command in self.commands.drain(..) {
            (command.draw)(
                framebuffer,
                &self.pipeline.fragment,
                self.pipeline.thread_count,
            );
        }
    }

    fn apply_post_chain(&self, framebuffer: &mut Framebuffer) {
        self.pipeline.post_chain.apply(framebuffer);
    }
}

impl<'a, VS, FS> RenderFrame<'a, VS, FS> {
    /// Queues a view-rotation-only cube-map background for this frame.
    pub fn draw_skybox(&mut self, _: &Framebuffer, cube: &'a CubeTexture, camera: Camera) {
        self.skybox = Some((cube, camera));
    }

    pub fn draw<Vertex, Uniforms>(
        &mut self,
        framebuffer: &Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync + 'a,
    {
        let class = if self.pipeline.fragment.is_opaque(uniforms) {
            DrawClass::Opaque
        } else {
            DrawClass::Transparent
        };
        self.queue(
            framebuffer,
            vertices,
            triangles,
            None,
            uniforms,
            class,
            rasterize_plain_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_with_sampling<Vertex, Uniforms>(
        &mut self,
        framebuffer: &Framebuffer,
        vertices: &[Vertex],
        triangles: &[[usize; 3]],
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<Vertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Vertex: Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync + 'a,
    {
        let class = if self.pipeline.fragment.is_opaque(uniforms) {
            DrawClass::Opaque
        } else {
            DrawClass::Transparent
        };
        self.queue(
            framebuffer,
            vertices,
            triangles,
            None,
            uniforms,
            class,
            rasterize_sampled_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_mesh<Uniforms>(
        &mut self,
        framebuffer: &Framebuffer,
        mesh: &Mesh,
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: FragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: Clone + Send + Sync + 'a,
    {
        let model_view = self
            .pipeline
            .fragment
            .model_view(uniforms)
            .unwrap_or(Mat4::IDENTITY);
        let keys = mesh_centroid_depths(mesh, model_view);
        let class = if self.pipeline.fragment.is_opaque(uniforms) {
            DrawClass::Opaque
        } else {
            DrawClass::Transparent
        };
        self.queue(
            framebuffer,
            mesh.vertices(),
            mesh.indices(),
            Some(&keys),
            uniforms,
            class,
            rasterize_plain_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }

    pub fn draw_mesh_with_sampling<Uniforms>(
        &mut self,
        framebuffer: &Framebuffer,
        mesh: &Mesh,
        uniforms: &'a Uniforms,
    ) where
        VS: VertexStage<MeshVertex, Uniforms> + Sync,
        FS: SampledFragmentStage<VS::Varyings, Uniforms> + Sync,
        Uniforms: Sync,
        VS::Varyings: SamplingVaryings + Clone + Send + Sync + 'a,
    {
        let model_view = self
            .pipeline
            .fragment
            .model_view(uniforms)
            .unwrap_or(Mat4::IDENTITY);
        let keys = mesh_centroid_depths(mesh, model_view);
        let class = if self.pipeline.fragment.is_opaque(uniforms) {
            DrawClass::Opaque
        } else {
            DrawClass::Transparent
        };
        self.queue(
            framebuffer,
            mesh.vertices(),
            mesh.indices(),
            Some(&keys),
            uniforms,
            class,
            rasterize_sampled_triangle::<VS::Varyings, FS, Uniforms>,
        );
    }
}

impl<VS, FS> Pipeline<VS, FS> {
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
        let prepared = prepare_triangles(
            &self.vertex,
            framebuffer,
            vertices,
            triangles,
            None,
            uniforms,
        );
        let prepared = prepared
            .into_iter()
            .map(|(_, triangle)| triangle)
            .collect::<Vec<_>>();
        dispatch_prepared(
            self.thread_count,
            framebuffer,
            &prepared,
            uniforms,
            &self.fragment,
            RasterState::DEPTH_ONLY,
            rasterize_depth_triangle::<VS::Varyings, FS, Uniforms>,
        );
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

/// The one tile-aware dispatcher used by every queued draw.
fn dispatch_prepared<V, FS, Uniforms>(
    thread_count: usize,
    framebuffer: &mut Framebuffer,
    prepared: &[PreparedTriangle<V>],
    uniforms: &Uniforms,
    fragment: &FS,
    state: RasterState,
    rasterize: RasterFn<V, FS, Uniforms>,
) where
    V: Varyings + Clone + Send + Sync,
    FS: Sync,
    Uniforms: Sync,
{
    if prepared.is_empty() {
        return;
    }
    if thread_count <= 1 {
        for triangle in prepared {
            rasterize(
                framebuffer,
                triangle.vertices.clone(),
                fragment,
                uniforms,
                state,
            );
        }
        return;
    }
    let tiles = make_tiles(framebuffer.width, framebuffer.height, prepared);
    if tiles.is_empty() {
        return;
    }
    let worker_count = thread_count.min(tiles.len()).max(1);
    let source = &*framebuffer;
    let tiles = &tiles;
    let results = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            handles.push(scope.spawn(move || {
                (worker_index..tiles.len())
                    .step_by(worker_count)
                    .map(|tile_index| {
                        rasterize_tile(
                            &tiles[tile_index],
                            prepared,
                            source,
                            uniforms,
                            fragment,
                            state,
                            rasterize,
                        )
                    })
                    .collect::<Vec<_>>()
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
            if let (Some(destination), Some(source)) = (
                framebuffer.linear_pixels_mut(),
                result.framebuffer.linear_pixels(),
            ) {
                destination[destination_start..destination_end]
                    .copy_from_slice(&source[source_start..source_end]);
            }
            framebuffer.depth[destination_start..destination_end]
                .copy_from_slice(&result.framebuffer.depth[source_start..source_end]);
        }
    }
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
    rasterize: RasterFn<V, FS, Uniforms>,
) -> TileResult
where
    V: Varyings + Clone,
{
    let mut framebuffer = Framebuffer::new(tile.width, tile.height);
    if source.is_hdr() {
        framebuffer.enable_hdr();
    }
    for row in 0..tile.height {
        let source_start = (tile.y + row) * source.width + tile.x;
        let source_end = source_start + tile.width;
        let destination_start = row * tile.width;
        let destination_end = destination_start + tile.width;
        framebuffer.color[destination_start..destination_end]
            .copy_from_slice(&source.color[source_start..source_end]);
        if let (Some(destination), Some(source_linear)) =
            (framebuffer.linear_pixels_mut(), source.linear_pixels())
        {
            destination[destination_start..destination_end]
                .copy_from_slice(&source_linear[source_start..source_end]);
        }
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
    let hdr = framebuffer.is_hdr();
    rasterize_triangle_with_state(framebuffer, vertices, state, |varyings| {
        if hdr {
            FragmentColor::Linear(fragment.run_linear(&varyings, uniforms))
        } else {
            FragmentColor::Encoded(fragment.run(&varyings, uniforms))
        }
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
    let hdr = framebuffer.is_hdr();
    rasterize_triangle_with_sampling_state(
        framebuffer,
        vertices,
        state,
        |varyings, derivatives| {
            if hdr {
                FragmentColor::Linear(fragment.run_linear_with_sampling(
                    &varyings,
                    &derivatives,
                    uniforms,
                ))
            } else {
                FragmentColor::Encoded(fragment.run_with_sampling(
                    &varyings,
                    &derivatives,
                    uniforms,
                ))
            }
        },
    );
}

fn prepare_triangles<Vertex, Uniforms, VS>(
    vertex_stage: &VS,
    framebuffer: &Framebuffer,
    vertices: &[Vertex],
    triangles: &[[usize; 3]],
    sort_keys: Option<&[f32]>,
    uniforms: &Uniforms,
) -> Vec<(f32, PreparedTriangle<VS::Varyings>)>
where
    VS: VertexStage<Vertex, Uniforms>,
    VS::Varyings: Clone,
{
    let mut prepared = Vec::new();
    for (triangle_index, &[a, b, c]) in triangles.iter().enumerate() {
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
        let key = sort_keys
            .and_then(|keys| keys.get(triangle_index).copied())
            .unwrap_or_else(|| {
                centroid_view_depth([
                    output_a.view_position,
                    output_b.view_position,
                    output_c.view_position,
                ])
            });
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
                    .all(|vertex| vertex.position.w.is_finite() && vertex.position.w > 0.0)
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
                prepared.push((key, PreparedTriangle { vertices, bounds }));
            }
        }
    }
    prepared
}

fn centroid_view_depth(positions: [Vec4; 3]) -> f32 {
    let position = (positions[0] + positions[1] + positions[2]) / 3.0;
    let z = position.z / position.w;
    if z.is_nan() { f32::INFINITY } else { z }
}

impl DrawClass {
    const fn raster_state(self) -> RasterState {
        match self {
            Self::Opaque => RasterState::OPAQUE,
            Self::Transparent => RasterState::TRANSPARENT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::{argb8888, blend_argb8888_linear};

    #[derive(Clone, Copy)]
    struct RawVertex {
        clip_position: Vec4,
        view_position: Vec4,
        color: Vec4,
    }

    struct TransparentColor;

    impl FragmentStage<ColorVarying, ()> for TransparentColor {
        fn run(&self, varyings: &ColorVarying, _: &()) -> u32 {
            let color = varyings.color;
            argb8888(
                128,
                (color.x * 255.0).round() as u8,
                (color.y * 255.0).round() as u8,
                (color.z * 255.0).round() as u8,
            )
        }

        fn is_opaque(&self, _: &()) -> bool {
            false
        }
    }

    #[test]
    fn closure_stages_draw_a_triangle() {
        let mut pipeline = Pipeline::new(
            vertex_stage(|vertex: &Vec4, _: &()| VertexOutput::new(*vertex, ())),
            fragment_stage(|_: &(), _: &()| argb8888(255, 20, 40, 60)),
        );
        let mut framebuffer = Framebuffer::new(4, 4);
        let vertices = [
            Vec4::new(-1.0, -1.0, 0.0, 1.0),
            Vec4::new(1.0, -1.0, 0.0, 1.0),
            Vec4::new(-1.0, 1.0, 0.0, 1.0),
        ];
        pipeline.render(&mut framebuffer, |frame, target| {
            frame.draw(target, &vertices, &[[0, 1, 2]], &());
        });
        assert!(framebuffer.color.contains(&argb8888(255, 20, 40, 60)));
    }

    #[test]
    fn raw_transparent_draws_sort_by_view_space_depth() {
        let far_color = argb8888(128, 235, 70, 40);
        let near_color = argb8888(128, 40, 90, 235);
        let far = [
            RawVertex {
                clip_position: Vec4::new(-0.8 * 4.0, -0.8 * 4.0, 3.0, 4.0),
                view_position: Vec4::new(-0.8, -0.8, -4.0, 1.0),
                color: Vec4::new(235.0 / 255.0, 70.0 / 255.0, 40.0 / 255.0, 1.0),
            },
            RawVertex {
                clip_position: Vec4::new(0.8 * 4.0, -0.8 * 4.0, 3.0, 4.0),
                view_position: Vec4::new(0.8, -0.8, -4.0, 1.0),
                color: Vec4::new(235.0 / 255.0, 70.0 / 255.0, 40.0 / 255.0, 1.0),
            },
            RawVertex {
                clip_position: Vec4::new(0.0, 0.8 * 4.0, 3.0, 4.0),
                view_position: Vec4::new(0.0, 0.8, -4.0, 1.0),
                color: Vec4::new(235.0 / 255.0, 70.0 / 255.0, 40.0 / 255.0, 1.0),
            },
        ];
        let near = [
            RawVertex {
                clip_position: Vec4::new(-0.8, -0.8, -0.5, 1.0),
                view_position: Vec4::new(-0.8, -0.8, -1.0, 1.0),
                color: Vec4::new(40.0 / 255.0, 90.0 / 255.0, 235.0 / 255.0, 1.0),
            },
            RawVertex {
                clip_position: Vec4::new(0.8, -0.8, -0.5, 1.0),
                view_position: Vec4::new(0.8, -0.8, -1.0, 1.0),
                color: Vec4::new(40.0 / 255.0, 90.0 / 255.0, 235.0 / 255.0, 1.0),
            },
            RawVertex {
                clip_position: Vec4::new(0.0, 0.8, -0.5, 1.0),
                view_position: Vec4::new(0.0, 0.8, -1.0, 1.0),
                color: Vec4::new(40.0 / 255.0, 90.0 / 255.0, 235.0 / 255.0, 1.0),
            },
        ];
        let mut pipeline = Pipeline::new(
            vertex_stage(|vertex: &RawVertex, _: &()| {
                VertexOutput::with_view_position(
                    vertex.clip_position,
                    vertex.view_position,
                    ColorVarying::new(vertex.color),
                )
            }),
            TransparentColor,
        );
        let mut framebuffer = Framebuffer::new(16, 16);
        let background = argb8888(255, 12, 16, 24);
        framebuffer.clear(background);
        pipeline.render(&mut framebuffer, |frame, target| {
            frame.draw(target, &near, &[[0, 1, 2]], &());
            frame.draw(target, &far, &[[0, 1, 2]], &());
        });

        let far_over_background = blend_argb8888_linear(background, far_color);
        let expected = blend_argb8888_linear(far_over_background, near_color);
        assert_eq!(framebuffer.color[8 * 16 + 8], expected);
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
        let mesh = Mesh::new(
            vec![
                MeshVertex::new(Vec3::new(0.0, 0.0, 0.0), None, None),
                MeshVertex::new(Vec3::new(1.0, 0.0, 0.0), None, None),
                MeshVertex::new(Vec3::new(1.0, 1.0, 0.0), None, None),
            ],
            vec![[0, 1, 2]],
        );
        let identity_key = mesh_centroid_depths(&mesh, Mat4::IDENTITY)[0];
        let rotated_key = mesh_centroid_depths(
            &mesh,
            Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2),
        )[0];
        assert_ne!(identity_key, rotated_key);
    }
}
