//! Flat and Blinn-Phong shaders built on the pipeline seam.
//!
//! Blinn-Phong lighting stays in world space: the vertex stage transforms
//! positions by `model`, and the fragment stage uses the world-space camera
//! and lights. Final color clamps to `[0, 1]` before 8-bit conversion.

use crate::fb::argb8888_linear;
use crate::image::{Texture, TextureDerivatives, srgb_to_linear};
use crate::math::{Mat3, Mat4, Vec3, Vec4};
use crate::mesh::MeshVertex;
use crate::pipeline::{
    FragmentStage, SampledFragmentStage, SamplingVaryings, Varyings, VertexOutput, VertexStage,
};
use crate::shadow::ShadowMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlatColorUniforms {
    pub transform: Mat4,
    pub color: u32,
}

impl FlatColorUniforms {
    pub const fn new(transform: Mat4, color: u32) -> Self {
        Self { transform, color }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FlatColorShader;

impl VertexStage<Vec4, FlatColorUniforms> for FlatColorShader {
    type Varyings = ();

    fn run(&self, vertex: &Vec4, uniforms: &FlatColorUniforms) -> VertexOutput<()> {
        VertexOutput::new(uniforms.transform * *vertex, ())
    }
}

impl FragmentStage<(), FlatColorUniforms> for FlatColorShader {
    fn run(&self, _: &(), uniforms: &FlatColorUniforms) -> u32 {
        uniforms.color
    }
}

/// Mesh transform uniforms with cache-safe matrix updates.
///
/// The source matrices are private. Use the accessors and setters instead:
///
/// ```compile_fail
/// use chimy2::math::Mat4;
/// use chimy2::shaders::MeshUniforms;
///
/// let mut uniforms = MeshUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, 0);
/// uniforms.model = Mat4::IDENTITY;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshUniforms {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    pub color: u32,
    transform: Mat4,
}

impl MeshUniforms {
    pub fn new(model: Mat4, view: Mat4, projection: Mat4, color: u32) -> Self {
        Self {
            model,
            view,
            projection,
            color,
            transform: projection * view * model,
        }
    }

    pub const fn transform(self) -> Mat4 {
        self.transform
    }

    pub const fn model(&self) -> Mat4 {
        self.model
    }

    pub const fn view(&self) -> Mat4 {
        self.view
    }

    pub const fn projection(&self) -> Mat4 {
        self.projection
    }

    pub fn set_model(&mut self, model: Mat4) {
        self.model = model;
        self.rebuild_transform();
    }

    pub fn set_view(&mut self, view: Mat4) {
        self.view = view;
        self.rebuild_transform();
    }

    pub fn set_projection(&mut self, projection: Mat4) {
        self.projection = projection;
        self.rebuild_transform();
    }

    fn rebuild_transform(&mut self) {
        self.transform = self.projection * self.view * self.model;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MeshShader;

impl VertexStage<MeshVertex, MeshUniforms> for MeshShader {
    type Varyings = ();

    fn run(&self, vertex: &MeshVertex, uniforms: &MeshUniforms) -> VertexOutput<()> {
        VertexOutput::new(
            uniforms.transform()
                * Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0),
            (),
        )
    }
}

impl FragmentStage<(), MeshUniforms> for MeshShader {
    fn run(&self, _: &(), uniforms: &MeshUniforms) -> u32 {
        uniforms.color
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFilter {
    Nearest,
    Bilinear,
    Trilinear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexturedUniforms<'a> {
    pub transform: Mat4,
    pub texture: &'a Texture,
    pub filter: TextureFilter,
}

impl<'a> TexturedUniforms<'a> {
    pub const fn new(transform: Mat4, texture: &'a Texture, filter: TextureFilter) -> Self {
        Self {
            transform,
            texture,
            filter,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexturedVaryings {
    pub texcoord: crate::math::Vec2,
}

impl Varyings for TexturedVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        Self {
            texcoord: a.texcoord * weights.x + b.texcoord * weights.y + c.texcoord * weights.z,
        }
    }
}

impl SamplingVaryings for TexturedVaryings {
    fn texture_coordinates(&self) -> crate::math::Vec2 {
        self.texcoord
    }
}

/// A textured shader must use [`Pipeline::draw_with_sampling`] or
/// [`Pipeline::draw_mesh_with_sampling`]. It has no plain fragment-stage
/// implementation, so a plain draw cannot silently lose its LOD derivatives.
///
/// ```compile_fail
/// use chimy2::fb::Framebuffer;
/// use chimy2::image::Texture;
/// use chimy2::math::Mat4;
/// use chimy2::mesh::Mesh;
/// use chimy2::pipeline::Pipeline;
/// use chimy2::shaders::{TextureFilter, TexturedShader, TexturedUniforms};
///
/// let texture = Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap();
/// let mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
/// let mut framebuffer = Framebuffer::new(4, 4);
/// let mut pipeline = Pipeline::new(TexturedShader, TexturedShader);
/// let uniforms = TexturedUniforms::new(Mat4::IDENTITY, &texture, TextureFilter::Trilinear);
/// pipeline.draw_mesh(&mut framebuffer, &mesh, &uniforms);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct TexturedShader;

impl<'a> VertexStage<MeshVertex, TexturedUniforms<'a>> for TexturedShader {
    type Varyings = TexturedVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &TexturedUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        let position = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        VertexOutput::new(
            uniforms.transform * position,
            TexturedVaryings {
                texcoord: vertex.texcoord.unwrap_or(crate::math::Vec2::ZERO),
            },
        )
    }
}

impl<'a> SampledFragmentStage<TexturedVaryings, TexturedUniforms<'a>> for TexturedShader {
    fn run_with_sampling(
        &self,
        varyings: &TexturedVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        uniforms: &TexturedUniforms<'a>,
    ) -> u32 {
        let pixel = sample_texture(
            uniforms.texture,
            varyings.texcoord,
            uniforms.filter,
            TextureDerivatives {
                ddx: derivatives.ddx,
                ddy: derivatives.ddy,
            },
        );
        argb8888_linear(pixel[3], [pixel[0], pixel[1], pixel[2]])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// A normalized direction from a surface point toward the light.
    pub direction: Vec3,
    pub color: Vec3,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3) -> Self {
        Self {
            direction,
            color: linearize_color(color),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub constant_attenuation: f32,
    pub linear_attenuation: f32,
    pub quadratic_attenuation: f32,
}

impl PointLight {
    pub fn new(
        position: Vec3,
        color: Vec3,
        constant_attenuation: f32,
        linear_attenuation: f32,
        quadratic_attenuation: f32,
    ) -> Self {
        Self {
            position,
            color: linearize_color(color),
            constant_attenuation,
            linear_attenuation,
            quadratic_attenuation,
        }
    }
}

/// Blinn-Phong uniforms with cache-safe matrix updates.
///
/// The source matrices are private. Use the accessors and setters instead:
///
/// ```compile_fail
/// use chimy2::math::{Mat4, Vec3};
/// use chimy2::shaders::{BlinnPhongUniforms, DirectionalLight, PointLight};
///
/// let mut uniforms = BlinnPhongUniforms::new(
///     Mat4::IDENTITY,
///     Mat4::IDENTITY,
///     Mat4::IDENTITY,
///     Vec3::ZERO,
///     Vec3::ZERO,
///     Vec3::ZERO,
///     8.0,
///     Vec3::ZERO,
///     DirectionalLight::new(Vec3::ZERO, Vec3::ZERO),
///     PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
/// );
/// uniforms.model = Mat4::IDENTITY;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct BlinnPhongUniforms {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    pub ambient_color: Vec3,
    pub diffuse_color: Vec3,
    pub specular_color: Vec3,
    pub shininess: f32,
    pub camera_position: Vec3,
    pub directional_light: DirectionalLight,
    pub point_light: PointLight,
    light_view_projection: Mat4,
    shadow_map: Option<ShadowMap>,
    shadow_constant_bias: f32,
    shadow_slope_bias: f32,
    transform: Mat4,
    normal_matrix: Mat3,
}

impl BlinnPhongUniforms {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        ambient_color: Vec3,
        diffuse_color: Vec3,
        specular_color: Vec3,
        shininess: f32,
        camera_position: Vec3,
        directional_light: DirectionalLight,
        point_light: PointLight,
    ) -> Self {
        Self {
            model,
            view,
            projection,
            ambient_color: linearize_color(ambient_color),
            diffuse_color: linearize_color(diffuse_color),
            specular_color: linearize_color(specular_color),
            shininess,
            camera_position,
            directional_light,
            point_light,
            light_view_projection: Mat4::IDENTITY,
            shadow_map: None,
            shadow_constant_bias: 0.002,
            shadow_slope_bias: 0.02,
            transform: projection * view * model,
            normal_matrix: model.normal_matrix().unwrap_or_default(),
        }
    }

    pub const fn transform(&self) -> Mat4 {
        self.transform
    }

    pub const fn normal_matrix(&self) -> Mat3 {
        self.normal_matrix
    }

    pub const fn model(&self) -> Mat4 {
        self.model
    }

    pub const fn view(&self) -> Mat4 {
        self.view
    }

    pub const fn projection(&self) -> Mat4 {
        self.projection
    }

    pub const fn light_view_projection(&self) -> Mat4 {
        self.light_view_projection
    }

    pub fn shadow_map(&self) -> Option<&ShadowMap> {
        self.shadow_map.as_ref()
    }

    pub const fn shadow_bias(&self) -> (f32, f32) {
        (self.shadow_constant_bias, self.shadow_slope_bias)
    }

    pub fn set_light_view_projection(&mut self, light_view_projection: Mat4) {
        self.light_view_projection = light_view_projection;
    }

    pub fn set_shadow_map(&mut self, shadow_map: Option<ShadowMap>) {
        self.shadow_map = shadow_map;
    }

    /// Sets `(constant, slope)` for the shadow bias.
    ///
    /// The default is `(0.002, 0.02)`. The compare uses
    /// `max(constant, slope * (1 - N dot L))`.
    pub fn set_shadow_bias(&mut self, constant: f32, slope: f32) {
        self.shadow_constant_bias = constant.max(0.0);
        self.shadow_slope_bias = slope.max(0.0);
    }

    pub fn set_model(&mut self, model: Mat4) {
        self.model = model;
        self.rebuild_caches();
    }

    pub fn set_view(&mut self, view: Mat4) {
        self.view = view;
        self.rebuild_transform();
    }

    pub fn set_projection(&mut self, projection: Mat4) {
        self.projection = projection;
        self.rebuild_transform();
    }

    fn rebuild_caches(&mut self) {
        self.rebuild_transform();
        self.normal_matrix = self.model.normal_matrix().unwrap_or_default();
    }

    fn rebuild_transform(&mut self) {
        self.transform = self.projection * self.view * self.model;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlinnPhongVaryings {
    pub world_position: Vec3,
    pub normal: Vec3,
    pub light_space_position: Vec4,
}

impl Varyings for BlinnPhongVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        Self {
            world_position: a.world_position * weights.x
                + b.world_position * weights.y
                + c.world_position * weights.z,
            normal: a.normal * weights.x + b.normal * weights.y + c.normal * weights.z,
            light_space_position: a.light_space_position * weights.x
                + b.light_space_position * weights.y
                + c.light_space_position * weights.z,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PreparedBlinnPhongVertex {
    clip_position: Vec4,
    world_position: Vec3,
    normal: Vec3,
    light_space_position: Vec4,
}

fn prepare_blinn_phong_vertex(
    vertex: &MeshVertex,
    uniforms: &BlinnPhongUniforms,
) -> PreparedBlinnPhongVertex {
    let local_position = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
    let world_position = uniforms.model() * local_position;
    let world_position = Vec3::new(
        world_position.x / world_position.w,
        world_position.y / world_position.w,
        world_position.z / world_position.w,
    );
    let normal = vertex.normal.unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let normal = (uniforms.normal_matrix() * normal).normalize();
    let light_space_position = uniforms.light_view_projection
        * Vec4::new(world_position.x, world_position.y, world_position.z, 1.0);
    PreparedBlinnPhongVertex {
        clip_position: uniforms.transform() * local_position,
        world_position,
        normal,
        light_space_position,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BlinnPhongShader;

impl VertexStage<MeshVertex, BlinnPhongUniforms> for BlinnPhongShader {
    type Varyings = BlinnPhongVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &BlinnPhongUniforms,
    ) -> VertexOutput<Self::Varyings> {
        let prepared = prepare_blinn_phong_vertex(vertex, uniforms);
        VertexOutput::new(
            prepared.clip_position,
            BlinnPhongVaryings {
                world_position: prepared.world_position,
                normal: prepared.normal,
                light_space_position: prepared.light_space_position,
            },
        )
    }
}

impl FragmentStage<BlinnPhongVaryings, BlinnPhongUniforms> for BlinnPhongShader {
    fn run(&self, varyings: &BlinnPhongVaryings, uniforms: &BlinnPhongUniforms) -> u32 {
        Self::shade(varyings, uniforms)
    }
}

impl BlinnPhongShader {
    /// Shades in world space. The normal is renormalized after interpolation.
    /// The point light uses constant + linear distance + quadratic distance².
    pub fn shade(varyings: &BlinnPhongVaryings, uniforms: &BlinnPhongUniforms) -> u32 {
        let lighted = evaluate_lighting(
            varyings.world_position,
            varyings.normal,
            varyings.light_space_position,
            uniforms,
            Vec3::new(1.0, 1.0, 1.0),
        );

        argb8888_linear(1.0, [lighted.x, lighted.y, lighted.z])
    }
}

fn evaluate_lighting(
    world_position: Vec3,
    interpolated_normal: Vec3,
    light_space_position: Vec4,
    uniforms: &BlinnPhongUniforms,
    albedo: Vec3,
) -> Vec3 {
    let normal = interpolated_normal.normalize();
    let view_direction = (uniforms.camera_position - world_position).normalize();
    let mut lighted = uniforms.ambient_color * albedo;

    let directional_visibility = uniforms.shadow_visibility(light_space_position, normal);
    lighted = lighted
        + evaluate_light(
            normal,
            view_direction,
            uniforms.directional_light.direction.normalize(),
            uniforms.directional_light.color,
            directional_visibility,
            uniforms,
            albedo,
        );

    let to_point = uniforms.point_light.position - world_position;
    let distance = to_point.length();
    let point_direction = to_point.normalize();
    let denominator = uniforms.point_light.constant_attenuation
        + uniforms.point_light.linear_attenuation * distance
        + uniforms.point_light.quadratic_attenuation * distance * distance;
    let attenuation = if denominator > 0.0 {
        1.0 / denominator
    } else {
        0.0
    };
    lighted = lighted
        + evaluate_light(
            normal,
            view_direction,
            point_direction,
            uniforms.point_light.color,
            attenuation,
            uniforms,
            albedo,
        );

    lighted
}

impl BlinnPhongUniforms {
    fn shadow_visibility(&self, light_space_position: Vec4, normal: Vec3) -> f32 {
        let Some(shadow_map) = self.shadow_map.as_ref() else {
            return 1.0;
        };
        if light_space_position.w <= 0.0 || !light_space_position.w.is_finite() {
            return 1.0;
        }
        let ndc = Vec3::new(
            light_space_position.x / light_space_position.w,
            light_space_position.y / light_space_position.w,
            light_space_position.z / light_space_position.w,
        );
        let uv = crate::math::Vec2::new((ndc.x + 1.0) * 0.5, (1.0 - ndc.y) * 0.5);
        let normal_dot_light = normal
            .normalize()
            .dot(self.directional_light.direction.normalize())
            .clamp(0.0, 1.0);
        let bias = self
            .shadow_constant_bias
            .max(self.shadow_slope_bias * (1.0 - normal_dot_light));
        shadow_map.visibility_3x3(uv, ndc.z, bias)
    }
}

fn evaluate_light(
    normal: Vec3,
    view_direction: Vec3,
    light_direction: Vec3,
    light_color: Vec3,
    attenuation: f32,
    uniforms: &BlinnPhongUniforms,
    albedo: Vec3,
) -> Vec3 {
    let diffuse = normal.dot(light_direction).max(0.0);
    let half_vector = (light_direction + view_direction).normalize();
    let specular = if diffuse > 0.0 {
        normal
            .dot(half_vector)
            .max(0.0)
            .powf(uniforms.shininess.max(0.0))
    } else {
        0.0
    };
    (uniforms.diffuse_color * albedo * diffuse + uniforms.specular_color * specular)
        * light_color
        * attenuation
}

#[derive(Clone, Debug, PartialEq)]
pub struct TexturedBlinnPhongUniforms<'a> {
    pub lighting: BlinnPhongUniforms,
    pub texture: &'a Texture,
    pub filter: TextureFilter,
}

impl<'a> TexturedBlinnPhongUniforms<'a> {
    pub const fn new(
        lighting: BlinnPhongUniforms,
        texture: &'a Texture,
        filter: TextureFilter,
    ) -> Self {
        Self {
            lighting,
            texture,
            filter,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TexturedBlinnPhongShader;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexturedBlinnPhongVaryings {
    pub world_position: Vec3,
    pub normal: Vec3,
    pub texcoord: crate::math::Vec2,
    pub light_space_position: Vec4,
}

impl Varyings for TexturedBlinnPhongVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        Self {
            world_position: a.world_position * weights.x
                + b.world_position * weights.y
                + c.world_position * weights.z,
            normal: a.normal * weights.x + b.normal * weights.y + c.normal * weights.z,
            texcoord: a.texcoord * weights.x + b.texcoord * weights.y + c.texcoord * weights.z,
            light_space_position: a.light_space_position * weights.x
                + b.light_space_position * weights.y
                + c.light_space_position * weights.z,
        }
    }
}

impl SamplingVaryings for TexturedBlinnPhongVaryings {
    fn texture_coordinates(&self) -> crate::math::Vec2 {
        self.texcoord
    }
}

impl<'a> VertexStage<MeshVertex, TexturedBlinnPhongUniforms<'a>> for TexturedBlinnPhongShader {
    type Varyings = TexturedBlinnPhongVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &TexturedBlinnPhongUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        let prepared = prepare_blinn_phong_vertex(vertex, &uniforms.lighting);
        VertexOutput::new(
            prepared.clip_position,
            TexturedBlinnPhongVaryings {
                world_position: prepared.world_position,
                normal: prepared.normal,
                texcoord: vertex.texcoord.unwrap_or(crate::math::Vec2::ZERO),
                light_space_position: prepared.light_space_position,
            },
        )
    }
}

impl<'a> SampledFragmentStage<TexturedBlinnPhongVaryings, TexturedBlinnPhongUniforms<'a>>
    for TexturedBlinnPhongShader
{
    fn run_with_sampling(
        &self,
        varyings: &TexturedBlinnPhongVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        uniforms: &TexturedBlinnPhongUniforms<'a>,
    ) -> u32 {
        let pixel = sample_texture(
            uniforms.texture,
            varyings.texcoord,
            uniforms.filter,
            TextureDerivatives {
                ddx: derivatives.ddx,
                ddy: derivatives.ddy,
            },
        );
        let albedo = Vec3::new(pixel[0], pixel[1], pixel[2]);
        let lighted = evaluate_lighting(
            varyings.world_position,
            varyings.normal,
            varyings.light_space_position,
            &uniforms.lighting,
            albedo,
        );

        argb8888_linear(pixel[3], [lighted.x, lighted.y, lighted.z])
    }
}

fn sample_texture(
    texture: &Texture,
    uv: crate::math::Vec2,
    filter: TextureFilter,
    derivatives: TextureDerivatives,
) -> [f32; 4] {
    match filter {
        TextureFilter::Nearest => texture.sample_linear_nearest(uv),
        TextureFilter::Bilinear => texture.sample_linear_bilinear(uv),
        TextureFilter::Trilinear => {
            texture.sample_linear_trilinear_with_derivatives(uv, derivatives)
        }
    }
}

fn linearize_color(color: Vec3) -> Vec3 {
    Vec3::new(
        srgb_to_linear(color.x),
        srgb_to_linear(color.y),
        srgb_to_linear(color.z),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniforms() -> BlinnPhongUniforms {
        BlinnPhongUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            8.0,
            Vec3::new(0.0, 0.0, 1.0),
            DirectionalLight::new(
                Vec3::new(0.0, 1.0, 1.0).normalize(),
                Vec3::new(1.0, 1.0, 1.0),
            ),
            PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
        )
    }

    #[test]
    fn blinn_phong_single_pixel_uses_half_vector() {
        let varyings = BlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        // The hand calculation is H = normalize((0, 1, 1) + (0, 0, 1)),
        // so N dot H = 0.9238795. With shininess 8, specular is 0.53079,
        // which rounds to 135 in each 8-bit channel.
        assert_eq!(BlinnPhongShader::shade(&varyings, &uniforms()), 0xffc1c1c1);
    }

    #[test]
    fn blinn_phong_skips_specular_on_backside() {
        let varyings = BlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        let mut uniforms = uniforms();
        uniforms.shininess = 2.0;
        // L = (0, 4/5, -3/5), V = (0, 0, 1), and N = (0, 0, 1).
        // N dot L = -3/5, but H = normalize(L + V) has N dot H = 1/sqrt(5).
        // The backside gate therefore makes the hand-computed result zero.
        uniforms.directional_light.direction = Vec3::new(0.0, 0.8, -0.6);
        assert_eq!(BlinnPhongShader::shade(&varyings, &uniforms), 0xff000000);
    }

    #[test]
    fn mesh_uniform_set_model_rebuilds_transform_cache() {
        let mut uniforms = MeshUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, 0);
        let model = Mat4::translate(Vec3::new(1.0, 2.0, 3.0));
        uniforms.set_model(model);

        assert_eq!(uniforms.model(), model);
        assert_eq!(
            uniforms.transform(),
            Mat4::IDENTITY * Mat4::IDENTITY * model
        );
    }

    #[test]
    fn mesh_uniform_set_view_rebuilds_transform_cache() {
        let mut uniforms = MeshUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, 0);
        let view = Mat4::scale(Vec3::new(2.0, 3.0, 4.0));
        uniforms.set_view(view);

        assert_eq!(uniforms.view(), view);
        assert_eq!(uniforms.transform(), Mat4::IDENTITY * view * Mat4::IDENTITY);
    }

    #[test]
    fn mesh_uniform_set_projection_rebuilds_transform_cache() {
        let mut uniforms = MeshUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, 0);
        let projection = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.25);
        uniforms.set_projection(projection);

        assert_eq!(uniforms.projection(), projection);
        assert_eq!(
            uniforms.transform(),
            projection * Mat4::IDENTITY * Mat4::IDENTITY
        );
    }

    #[test]
    fn blinn_phong_uniform_set_model_rebuilds_transform_and_normal_caches() {
        let mut uniforms = uniforms();
        let model =
            Mat4::translate(Vec3::new(1.0, 2.0, 3.0)) * Mat4::scale(Vec3::new(2.0, 3.0, 4.0));
        uniforms.set_model(model);

        assert_eq!(uniforms.model(), model);
        assert_eq!(
            uniforms.transform(),
            Mat4::IDENTITY * Mat4::IDENTITY * model
        );
        assert_eq!(uniforms.normal_matrix(), model.normal_matrix().unwrap());
    }

    #[test]
    fn blinn_phong_uniform_set_view_rebuilds_transform_cache() {
        let mut uniforms = uniforms();
        let view = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.25);
        uniforms.set_view(view);

        assert_eq!(uniforms.view(), view);
        assert_eq!(uniforms.transform(), Mat4::IDENTITY * view * Mat4::IDENTITY);
        assert_eq!(uniforms.normal_matrix(), Mat3::IDENTITY);
    }

    #[test]
    fn blinn_phong_uniform_set_projection_rebuilds_transform_cache() {
        let mut uniforms = uniforms();
        let projection = Mat4::scale(Vec3::new(0.5, 0.75, 1.25));
        uniforms.set_projection(projection);

        assert_eq!(uniforms.projection(), projection);
        assert_eq!(
            uniforms.transform(),
            projection * Mat4::IDENTITY * Mat4::IDENTITY
        );
        assert_eq!(uniforms.normal_matrix(), Mat3::IDENTITY);
    }

    #[test]
    fn blinn_phong_shadow_mutators_update_immediately() {
        let mut uniforms = uniforms();
        let light_view_projection = Mat4::translate(Vec3::new(2.0, 3.0, 4.0));
        uniforms.set_light_view_projection(light_view_projection);
        uniforms.set_shadow_bias(0.01, 0.04);

        assert_eq!(uniforms.light_view_projection(), light_view_projection);
        assert_eq!(uniforms.shadow_bias(), (0.01, 0.04));
        assert!(uniforms.shadow_map().is_none());
    }
}
