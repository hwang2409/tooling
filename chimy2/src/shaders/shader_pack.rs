//! Stylized shader materials built on the existing pipeline seam.

use super::linearize_color;
use crate::fb::argb8888_linear;
use crate::math::{Mat3, Mat4, Vec2, Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::pipeline::{FragmentStage, InstanceUniforms, Varyings, VertexOutput, VertexStage};

/// A mesh vertex with a per-triangle barycentric coordinate.
///
/// The existing vertex-stage seam receives a vertex, not a triangle corner.
/// [`expand_mesh_with_barycentrics`] therefore expands indexed triangles before
/// drawing. This keeps the edge varying in the shader-facing data path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShaderPackVertex {
    pub position: Vec3,
    pub texcoord: Option<Vec2>,
    pub normal: Option<Vec3>,
    pub barycentric: Vec3,
}

impl ShaderPackVertex {
    pub const fn new(vertex: MeshVertex, barycentric: Vec3) -> Self {
        Self {
            position: vertex.position(),
            texcoord: vertex.texcoord(),
            normal: vertex.normal(),
            barycentric,
        }
    }
}

/// Expands indexed mesh triangles so each shader vertex has an edge coordinate.
///
/// The returned indices are local to the returned vertex list. Invalid source
/// indices are skipped. The expansion is required for the toon and wireframe
/// edge tests because the core vertex stage has no triangle-corner argument.
pub fn expand_mesh_with_barycentrics(mesh: &Mesh) -> (Vec<ShaderPackVertex>, Vec<[usize; 3]>) {
    let mut vertices = Vec::with_capacity(mesh.indices().len() * 3);
    let mut triangles = Vec::with_capacity(mesh.indices().len());
    for &[a, b, c] in mesh.indices() {
        let Some(vertex_a) = mesh.vertex(a).copied() else {
            continue;
        };
        let Some(vertex_b) = mesh.vertex(b).copied() else {
            continue;
        };
        let Some(vertex_c) = mesh.vertex(c).copied() else {
            continue;
        };
        let base = vertices.len();
        vertices.extend([
            ShaderPackVertex::new(vertex_a, Vec3::new(1.0, 0.0, 0.0)),
            ShaderPackVertex::new(vertex_b, Vec3::new(0.0, 1.0, 0.0)),
            ShaderPackVertex::new(vertex_c, Vec3::new(0.0, 0.0, 1.0)),
        ]);
        triangles.push([base, base + 1, base + 2]);
    }
    (vertices, triangles)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ShaderTransform {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    transform: Mat4,
    normal_matrix: Mat3,
}

impl ShaderTransform {
    fn new(model: Mat4, view: Mat4, projection: Mat4) -> Self {
        Self {
            model,
            view,
            projection,
            transform: projection * view * model,
            normal_matrix: model.normal_matrix().unwrap_or_default(),
        }
    }

    const fn transform(self) -> Mat4 {
        self.transform
    }

    const fn model(self) -> Mat4 {
        self.model
    }

    const fn view(self) -> Mat4 {
        self.view
    }

    const fn projection(self) -> Mat4 {
        self.projection
    }

    const fn normal_matrix(self) -> Mat3 {
        self.normal_matrix
    }

    fn set_model(&mut self, model: Mat4) {
        self.model = model;
        self.rebuild();
    }

    fn set_view(&mut self, view: Mat4) {
        self.view = view;
        self.rebuild_transform();
    }

    fn set_projection(&mut self, projection: Mat4) {
        self.projection = projection;
        self.rebuild_transform();
    }

    fn rebuild(&mut self) {
        self.rebuild_transform();
        self.normal_matrix = self.model.normal_matrix().unwrap_or_default();
    }

    fn rebuild_transform(&mut self) {
        self.transform = self.projection * self.view * self.model;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShaderPackVaryings {
    pub world_position: Vec3,
    pub normal: Vec3,
    pub barycentric: Vec3,
    /// Clip-space x/y. Divide by [`Self::clip_w`] after interpolation.
    pub clip_xy: Vec2,
    /// Clip-space w carried beside x/y for post-interpolation division.
    pub clip_w: f32,
    /// Euclidean distance from the camera in view space.
    pub view_distance: f32,
}

impl Varyings for ShaderPackVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        Self {
            world_position: a.world_position * weights.x
                + b.world_position * weights.y
                + c.world_position * weights.z,
            normal: a.normal * weights.x + b.normal * weights.y + c.normal * weights.z,
            barycentric: a.barycentric * weights.x
                + b.barycentric * weights.y
                + c.barycentric * weights.z,
            clip_xy: a.clip_xy * weights.x + b.clip_xy * weights.y + c.clip_xy * weights.z,
            clip_w: a.clip_w * weights.x + b.clip_w * weights.y + c.clip_w * weights.z,
            view_distance: a.view_distance * weights.x
                + b.view_distance * weights.y
                + c.view_distance * weights.z,
        }
    }
}

fn shader_pack_vertex(
    vertex: &ShaderPackVertex,
    transform: ShaderTransform,
    snap_grid: Option<f32>,
) -> VertexOutput<ShaderPackVaryings> {
    let local = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
    let world4 = transform.model() * local;
    let world_position = Vec3::new(
        world4.x / world4.w,
        world4.y / world4.w,
        world4.z / world4.w,
    );
    let view4 =
        transform.view() * Vec4::new(world_position.x, world_position.y, world_position.z, 1.0);
    let mut clip_position = transform.projection() * view4;
    let mut ndc_position = Vec2::new(
        clip_position.x / clip_position.w,
        clip_position.y / clip_position.w,
    );
    if let Some(grid) = snap_grid.filter(|grid| grid.is_finite() && *grid > 0.0) {
        ndc_position.x = (ndc_position.x * grid).round() / grid;
        ndc_position.y = (ndc_position.y * grid).round() / grid;
        clip_position.x = ndc_position.x * clip_position.w;
        clip_position.y = ndc_position.y * clip_position.w;
    }
    let normal = transform.normal_matrix() * vertex.normal.unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    VertexOutput::new(
        clip_position,
        ShaderPackVaryings {
            world_position,
            normal,
            barycentric: vertex.barycentric,
            clip_xy: Vec2::new(clip_position.x, clip_position.y),
            clip_w: clip_position.w,
            view_distance: Vec3::new(view4.x, view4.y, view4.z).length(),
        },
    )
}

pub(crate) fn interpolated_screen_ndc(varyings: &ShaderPackVaryings) -> Vec2 {
    varyings.clip_xy / varyings.clip_w
}

/// Four toon bands use thresholds 0.25, 0.50, and 0.75.
pub fn toon_band(diffuse: f32) -> f32 {
    match diffuse.clamp(0.0, 1.0) {
        value if value < 0.25 => 0.15,
        value if value < 0.50 => 0.40,
        value if value < 0.75 => 0.70,
        _ => 1.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToonUniforms {
    transform: ShaderTransform,
    pub base_color: Vec3,
    pub light_direction: Vec3,
    pub ambient: f32,
    pub edge_threshold: f32,
    pub edge_darkening: f32,
}

impl ToonUniforms {
    pub fn new(
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        base_color: Vec3,
        light_direction: Vec3,
    ) -> Self {
        Self {
            transform: ShaderTransform::new(model, view, projection),
            base_color: linearize_color(base_color),
            light_direction,
            ambient: 0.08,
            edge_threshold: 0.08,
            edge_darkening: 0.18,
        }
    }

    pub const fn transform(self) -> Mat4 {
        self.transform.transform()
    }
    pub const fn model(&self) -> Mat4 {
        self.transform.model()
    }
    pub const fn view(&self) -> Mat4 {
        self.transform.view()
    }
    pub const fn projection(&self) -> Mat4 {
        self.transform.projection()
    }
    pub const fn normal_matrix(&self) -> Mat3 {
        self.transform.normal_matrix()
    }
    pub fn set_model(&mut self, model: Mat4) {
        self.transform.set_model(model);
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.transform.set_view(view);
    }
    pub fn set_projection(&mut self, projection: Mat4) {
        self.transform.set_projection(projection);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ToonShader;

impl VertexStage<ShaderPackVertex, ToonUniforms> for ToonShader {
    type Varyings = ShaderPackVaryings;

    fn run(
        &self,
        vertex: &ShaderPackVertex,
        uniforms: &ToonUniforms,
    ) -> VertexOutput<Self::Varyings> {
        shader_pack_vertex(vertex, uniforms.transform, None)
    }
}

impl FragmentStage<ShaderPackVaryings, ToonUniforms> for ToonShader {
    fn run(&self, varyings: &ShaderPackVaryings, uniforms: &ToonUniforms) -> u32 {
        Self::shade(varyings, uniforms)
    }
}

impl ToonShader {
    pub fn shade(varyings: &ShaderPackVaryings, uniforms: &ToonUniforms) -> u32 {
        let diffuse = varyings
            .normal
            .normalize()
            .dot(uniforms.light_direction.normalize())
            .max(0.0);
        let mut color = uniforms.base_color * (uniforms.ambient + toon_band(diffuse));
        if varyings
            .barycentric
            .x
            .min(varyings.barycentric.y)
            .min(varyings.barycentric.z)
            < uniforms.edge_threshold
        {
            color = color * uniforms.edge_darkening.clamp(0.0, 1.0);
        }
        argb8888_linear(1.0, [color.x, color.y, color.z])
    }
}

/// The PSX shader snaps post-divide vertices and applies ordered color crush.
/// The current kernel exposes only perspective-correct interpolation, so affine
/// UV texturing is intentionally omitted rather than forking the kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsxUniforms {
    transform: ShaderTransform,
    pub base_color: Vec3,
    pub vertex_snap_grid: f32,
    pub color_bits: u8,
    pub target_size: (u32, u32),
}

impl PsxUniforms {
    pub fn new(
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        base_color: Vec3,
        target_size: (u32, u32),
    ) -> Self {
        Self {
            transform: ShaderTransform::new(model, view, projection),
            base_color: linearize_color(base_color),
            vertex_snap_grid: 32.0,
            color_bits: 3,
            target_size,
        }
    }

    pub const fn transform(self) -> Mat4 {
        self.transform.transform()
    }
    pub const fn model(&self) -> Mat4 {
        self.transform.model()
    }
    pub const fn view(&self) -> Mat4 {
        self.transform.view()
    }
    pub const fn projection(&self) -> Mat4 {
        self.transform.projection()
    }
    pub fn set_model(&mut self, model: Mat4) {
        self.transform.set_model(model);
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.transform.set_view(view);
    }
    pub fn set_projection(&mut self, projection: Mat4) {
        self.transform.set_projection(projection);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PsxShader;

impl VertexStage<ShaderPackVertex, PsxUniforms> for PsxShader {
    type Varyings = ShaderPackVaryings;

    fn run(
        &self,
        vertex: &ShaderPackVertex,
        uniforms: &PsxUniforms,
    ) -> VertexOutput<Self::Varyings> {
        shader_pack_vertex(vertex, uniforms.transform, Some(uniforms.vertex_snap_grid))
    }
}

impl FragmentStage<ShaderPackVaryings, PsxUniforms> for PsxShader {
    fn run(&self, varyings: &ShaderPackVaryings, uniforms: &PsxUniforms) -> u32 {
        let ndc_position = interpolated_screen_ndc(varyings);
        let color = ordered_dither_linear(
            uniforms.base_color,
            ndc_position,
            uniforms.target_size,
            uniforms.color_bits,
        );
        argb8888_linear(1.0, [color.x, color.y, color.z])
    }
}

/// Standard 4x4 Bayer values use centered thresholds `(n + 0.5) / 16`.
pub const BAYER4: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

pub const fn bayer4_value(x: usize, y: usize) -> f32 {
    (BAYER4[y % 4][x % 4] as f32 + 0.5) / 16.0
}

fn dither_pixel(ndc: Vec2, target_size: (u32, u32)) -> (usize, usize) {
    let width = target_size.0.max(1) as f32;
    let height = target_size.1.max(1) as f32;
    let x = ((ndc.x * 0.5 + 0.5) * width).floor() as i32;
    let y = ((1.0 - (ndc.y * 0.5 + 0.5)) * height).floor() as i32;
    (
        x.clamp(0, width as i32 - 1) as usize,
        y.clamp(0, height as i32 - 1) as usize,
    )
}

fn quantize_channel(value: f32, threshold: f32, bits: u8) -> f32 {
    let bits = bits.clamp(1, 8);
    let levels = ((1u32 << bits) - 1) as f32;
    ((value + (threshold - 0.5) / levels).clamp(0.0, 1.0) * levels).round() / levels
}

/// Crushes linear RGB to `bits` per channel with ordered 4x4 Bayer dither.
pub fn ordered_dither_linear(
    color: Vec3,
    ndc_position: Vec2,
    target_size: (u32, u32),
    bits: u8,
) -> Vec3 {
    let (x, y) = dither_pixel(ndc_position, target_size);
    let threshold = bayer4_value(x, y);
    Vec3::new(
        quantize_channel(color.x, threshold, bits),
        quantize_channel(color.y, threshold, bits),
        quantize_channel(color.z, threshold, bits),
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitherUniforms {
    transform: ShaderTransform,
    pub base_color: Vec3,
    pub color_bits: u8,
    pub target_size: (u32, u32),
}

impl DitherUniforms {
    pub fn new(
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        base_color: Vec3,
        target_size: (u32, u32),
    ) -> Self {
        Self {
            transform: ShaderTransform::new(model, view, projection),
            base_color: linearize_color(base_color),
            color_bits: 3,
            target_size,
        }
    }

    pub const fn transform(self) -> Mat4 {
        self.transform.transform()
    }
    pub fn set_model(&mut self, model: Mat4) {
        self.transform.set_model(model);
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.transform.set_view(view);
    }
    pub fn set_projection(&mut self, projection: Mat4) {
        self.transform.set_projection(projection);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DitherShader;

impl VertexStage<ShaderPackVertex, DitherUniforms> for DitherShader {
    type Varyings = ShaderPackVaryings;

    fn run(
        &self,
        vertex: &ShaderPackVertex,
        uniforms: &DitherUniforms,
    ) -> VertexOutput<Self::Varyings> {
        shader_pack_vertex(vertex, uniforms.transform, None)
    }
}

impl FragmentStage<ShaderPackVaryings, DitherUniforms> for DitherShader {
    fn run(&self, varyings: &ShaderPackVaryings, uniforms: &DitherUniforms) -> u32 {
        let ndc_position = interpolated_screen_ndc(varyings);
        let color = ordered_dither_linear(
            uniforms.base_color,
            ndc_position,
            uniforms.target_size,
            uniforms.color_bits,
        );
        argb8888_linear(1.0, [color.x, color.y, color.z])
    }
}

/// Returns the linear fog amount for a documented linear start/end falloff.
pub fn linear_fog_factor(view_distance: f32, fog_start: f32, fog_end: f32) -> f32 {
    if fog_end <= fog_start {
        return f32::from(view_distance >= fog_end);
    }
    ((view_distance - fog_start) / (fog_end - fog_start)).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogUniforms {
    transform: ShaderTransform,
    pub base_color: Vec3,
    pub fog_color: Vec3,
    pub fog_start: f32,
    pub fog_end: f32,
}

impl FogUniforms {
    pub fn new(
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        base_color: Vec3,
        fog_color: Vec3,
        fog_start: f32,
        fog_end: f32,
    ) -> Self {
        Self {
            transform: ShaderTransform::new(model, view, projection),
            base_color: linearize_color(base_color),
            fog_color: linearize_color(fog_color),
            fog_start,
            fog_end,
        }
    }

    pub const fn transform(self) -> Mat4 {
        self.transform.transform()
    }
    pub fn set_model(&mut self, model: Mat4) {
        self.transform.set_model(model);
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.transform.set_view(view);
    }
    pub fn set_projection(&mut self, projection: Mat4) {
        self.transform.set_projection(projection);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FogShader;

impl VertexStage<ShaderPackVertex, FogUniforms> for FogShader {
    type Varyings = ShaderPackVaryings;

    fn run(
        &self,
        vertex: &ShaderPackVertex,
        uniforms: &FogUniforms,
    ) -> VertexOutput<Self::Varyings> {
        shader_pack_vertex(vertex, uniforms.transform, None)
    }
}

impl FragmentStage<ShaderPackVaryings, FogUniforms> for FogShader {
    fn run(&self, varyings: &ShaderPackVaryings, uniforms: &FogUniforms) -> u32 {
        let fog = linear_fog_factor(varyings.view_distance, uniforms.fog_start, uniforms.fog_end);
        let color = uniforms.base_color * (1.0 - fog) + uniforms.fog_color * fog;
        argb8888_linear(1.0, [color.x, color.y, color.z])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalsUniforms {
    transform: ShaderTransform,
}

impl NormalsUniforms {
    pub fn new(model: Mat4, view: Mat4, projection: Mat4) -> Self {
        Self {
            transform: ShaderTransform::new(model, view, projection),
        }
    }

    pub const fn transform(self) -> Mat4 {
        self.transform.transform()
    }
    pub fn set_model(&mut self, model: Mat4) {
        self.transform.set_model(model);
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.transform.set_view(view);
    }
    pub fn set_projection(&mut self, projection: Mat4) {
        self.transform.set_projection(projection);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NormalsShader;

impl VertexStage<ShaderPackVertex, NormalsUniforms> for NormalsShader {
    type Varyings = ShaderPackVaryings;

    fn run(
        &self,
        vertex: &ShaderPackVertex,
        uniforms: &NormalsUniforms,
    ) -> VertexOutput<Self::Varyings> {
        shader_pack_vertex(vertex, uniforms.transform, None)
    }
}

impl FragmentStage<ShaderPackVaryings, NormalsUniforms> for NormalsShader {
    fn run(&self, varyings: &ShaderPackVaryings, _: &NormalsUniforms) -> u32 {
        let normal = varyings.normal.normalize() * 0.5 + Vec3::new(0.5, 0.5, 0.5);
        argb8888_linear(1.0, [normal.x, normal.y, normal.z])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireframeUniforms {
    transform: ShaderTransform,
    pub base_color: Vec3,
    pub edge_color: Vec3,
    /// The varying-space width is stable across a triangle, not across pixels.
    pub edge_threshold: f32,
}

impl WireframeUniforms {
    pub fn new(
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        base_color: Vec3,
        edge_color: Vec3,
    ) -> Self {
        Self {
            transform: ShaderTransform::new(model, view, projection),
            base_color: linearize_color(base_color),
            edge_color: linearize_color(edge_color),
            edge_threshold: 0.045,
        }
    }

    pub const fn transform(self) -> Mat4 {
        self.transform.transform()
    }
    pub fn set_model(&mut self, model: Mat4) {
        self.transform.set_model(model);
    }
    pub fn set_view(&mut self, view: Mat4) {
        self.transform.set_view(view);
    }
    pub fn set_projection(&mut self, projection: Mat4) {
        self.transform.set_projection(projection);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WireframeShader;

impl VertexStage<ShaderPackVertex, WireframeUniforms> for WireframeShader {
    type Varyings = ShaderPackVaryings;

    fn run(
        &self,
        vertex: &ShaderPackVertex,
        uniforms: &WireframeUniforms,
    ) -> VertexOutput<Self::Varyings> {
        shader_pack_vertex(vertex, uniforms.transform, None)
    }
}

impl FragmentStage<ShaderPackVaryings, WireframeUniforms> for WireframeShader {
    fn run(&self, varyings: &ShaderPackVaryings, uniforms: &WireframeUniforms) -> u32 {
        let edge = varyings
            .barycentric
            .x
            .min(varyings.barycentric.y)
            .min(varyings.barycentric.z);
        let color = if edge < uniforms.edge_threshold {
            uniforms.edge_color
        } else {
            uniforms.base_color
        };
        argb8888_linear(1.0, [color.x, color.y, color.z])
    }
}

impl InstanceUniforms for ToonUniforms {
    fn set_instance_model(&mut self, model: Mat4) {
        self.set_model(model);
    }

    fn apply_instance_tint(&mut self, tint: Vec4) {
        self.base_color = self.base_color * Vec3::new(tint.x, tint.y, tint.z);
    }
}

impl InstanceUniforms for PsxUniforms {
    fn set_instance_model(&mut self, model: Mat4) {
        self.set_model(model);
    }

    fn apply_instance_tint(&mut self, tint: Vec4) {
        self.base_color = self.base_color * Vec3::new(tint.x, tint.y, tint.z);
    }
}

impl InstanceUniforms for DitherUniforms {
    fn set_instance_model(&mut self, model: Mat4) {
        self.set_model(model);
    }

    fn apply_instance_tint(&mut self, tint: Vec4) {
        self.base_color = self.base_color * Vec3::new(tint.x, tint.y, tint.z);
    }
}

impl InstanceUniforms for FogUniforms {
    fn set_instance_model(&mut self, model: Mat4) {
        self.set_model(model);
    }

    fn apply_instance_tint(&mut self, tint: Vec4) {
        let tint = Vec3::new(tint.x, tint.y, tint.z);
        self.base_color = self.base_color * tint;
        self.fog_color = self.fog_color * tint;
    }
}

impl InstanceUniforms for NormalsUniforms {
    fn set_instance_model(&mut self, model: Mat4) {
        self.set_model(model);
    }

    fn apply_instance_tint(&mut self, _: Vec4) {}
}

impl InstanceUniforms for WireframeUniforms {
    fn set_instance_model(&mut self, model: Mat4) {
        self.set_model(model);
    }

    fn apply_instance_tint(&mut self, tint: Vec4) {
        let tint = Vec3::new(tint.x, tint.y, tint.z);
        self.base_color = self.base_color * tint;
        self.edge_color = self.edge_color * tint;
    }
}

/// Descriptive aliases for callers that prefer the full material names.
pub type OrderedDitherShader = DitherShader;
pub type OrderedDitherUniforms = DitherUniforms;
pub type DepthFogShader = FogShader;
pub type DepthFogUniforms = FogUniforms;
pub type NormalsAsColorShader = NormalsShader;
pub type NormalsAsColorUniforms = NormalsUniforms;
