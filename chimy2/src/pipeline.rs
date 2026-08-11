//! Programmable vertex and fragment stages.

use crate::clip::{ClipVertex, clip_triangle_near, cull_backface};
use crate::fb::Framebuffer;
use crate::math::{Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::raster::{
    PixelRect, ScreenVertex, rasterize_triangle, rasterize_triangle_with_sampling,
    triangle_pixel_rect, viewport_transform,
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
}

impl<VS, FS> Pipeline<VS, FS> {
    pub fn new(vertex: VS, fragment: FS) -> Self {
        Self {
            vertex,
            fragment,
            thread_count: default_thread_count(),
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
}

fn default_thread_count() -> usize {
    std::env::var("CHIMY_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .or_else(|| thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
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
        if self.thread_count <= 1 || prepared.is_empty() {
            self.draw_serial(framebuffer, prepared, uniforms);
        } else {
            self.draw_parallel(framebuffer, &prepared, uniforms);
        }
    }

    fn draw_serial<V, Uniforms>(
        &self,
        framebuffer: &mut Framebuffer,
        prepared: Vec<PreparedTriangle<V>>,
        uniforms: &Uniforms,
    ) where
        FS: FragmentStage<V, Uniforms>,
        V: Varyings + Clone,
    {
        for triangle in prepared {
            rasterize_triangle(framebuffer, triangle.vertices, |varyings| {
                self.fragment.run(&varyings, uniforms)
            });
        }
    }

    fn draw_serial_with_sampling<V, Uniforms>(
        &self,
        framebuffer: &mut Framebuffer,
        prepared: Vec<PreparedTriangle<V>>,
        uniforms: &Uniforms,
    ) where
        FS: SampledFragmentStage<V, Uniforms>,
        V: SamplingVaryings + Clone,
    {
        for triangle in prepared {
            rasterize_triangle_with_sampling(
                framebuffer,
                triangle.vertices,
                |varyings, derivatives| {
                    self.fragment
                        .run_with_sampling(&varyings, &derivatives, uniforms)
                },
            );
        }
    }

    fn draw_parallel<V, Uniforms>(
        &self,
        framebuffer: &mut Framebuffer,
        prepared: &[PreparedTriangle<V>],
        uniforms: &Uniforms,
    ) where
        FS: FragmentStage<V, Uniforms> + Sync,
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

    fn draw_parallel_with_sampling<V, Uniforms>(
        &self,
        framebuffer: &mut Framebuffer,
        prepared: &[PreparedTriangle<V>],
        uniforms: &Uniforms,
    ) where
        FS: SampledFragmentStage<V, Uniforms> + Sync,
        Uniforms: Sync,
        V: SamplingVaryings + Clone + Send + Sync,
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
                        results.push(rasterize_tile_with_sampling(
                            &tiles[tile_index],
                            prepared,
                            source,
                            uniforms,
                            fragment,
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
        if self.thread_count <= 1 || prepared.is_empty() {
            self.draw_serial_with_sampling(framebuffer, prepared, uniforms);
        } else {
            self.draw_parallel_with_sampling(framebuffer, &prepared, uniforms);
        }
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
        self.draw(framebuffer, &mesh.vertices, &mesh.triangles, uniforms);
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
        self.draw_with_sampling(framebuffer, &mesh.vertices, &mesh.triangles, uniforms);
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
) -> TileResult
where
    V: Varyings + Clone,
    FS: FragmentStage<V, Uniforms>,
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
        rasterize_triangle(&mut framebuffer, triangle, |varyings| {
            fragment.run(&varyings, uniforms)
        });
    }

    TileResult {
        x: tile.x,
        y: tile.y,
        framebuffer,
    }
}

fn rasterize_tile_with_sampling<V, FS, Uniforms>(
    tile: &Tile,
    prepared: &[PreparedTriangle<V>],
    source: &Framebuffer,
    uniforms: &Uniforms,
    fragment: &FS,
) -> TileResult
where
    V: SamplingVaryings + Clone,
    FS: SampledFragmentStage<V, Uniforms>,
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
        rasterize_triangle_with_sampling(&mut framebuffer, triangle, |varyings, derivatives| {
            fragment.run_with_sampling(&varyings, &derivatives, uniforms)
        });
    }

    TileResult {
        x: tile.x,
        y: tile.y,
        framebuffer,
    }
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
    let mut prepared = Vec::new();
    for &[a, b, c] in triangles {
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
                prepared.push(PreparedTriangle { vertices, bounds });
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
    fn color_varying_interpolates_components() {
        let result = ColorVarying::lerp3(
            &ColorVarying::new(Vec4::new(1.0, 0.0, 0.0, 1.0)),
            &ColorVarying::new(Vec4::new(0.0, 1.0, 0.0, 1.0)),
            &ColorVarying::new(Vec4::new(0.0, 0.0, 1.0, 1.0)),
            Vec3::new(0.25, 0.5, 0.25),
        );
        assert_eq!(result.color, Vec4::new(0.25, 0.5, 0.25, 1.0));
    }
}
