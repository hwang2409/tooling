//! Flat and Blinn-Phong shaders built on the pipeline seam.
//!
//! Blinn-Phong lighting stays in world space: the vertex stage transforms
//! positions by `model`, and the fragment stage uses the world-space camera
//! and lights. Final color clamps to `[0, 1]` before 8-bit conversion.

use crate::fb::argb8888_linear;
use crate::image::{ColorSpace, Texture, TextureDerivatives, srgb_to_linear};
use crate::math::{Mat3, Mat4, Vec2, Vec3, Vec4};
use crate::mesh::MeshVertex;
use crate::pipeline::{
    FragmentStage, SampledFragmentStage, SamplingVaryings, Varyings, VertexOutput, VertexStage,
};
use crate::shadow::{ShadowMap, ShadowState};
use crate::skybox::CubeTexture;

pub mod shader_pack;
pub use shader_pack::{
    BAYER4, DepthFogShader, DepthFogUniforms, DitherShader, DitherUniforms, FogShader, FogUniforms,
    NormalsAsColorShader, NormalsAsColorUniforms, NormalsShader, NormalsUniforms,
    OrderedDitherShader, OrderedDitherUniforms, PsxShader, PsxUniforms, ShaderPackVaryings,
    ShaderPackVertex, ToonShader, ToonUniforms, WireframeShader, WireframeUniforms, bayer4_value,
    expand_mesh_with_barycentrics, linear_fog_factor, ordered_dither_linear, toon_band,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlatColorUniforms {
    pub transform: Mat4,
    pub color: u32,
    pub alpha: f32,
}

impl FlatColorUniforms {
    pub const fn new(transform: Mat4, color: u32) -> Self {
        Self {
            transform,
            color,
            alpha: 1.0,
        }
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
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
        multiply_alpha(uniforms.color, uniforms.alpha)
    }

    fn is_opaque(&self, uniforms: &FlatColorUniforms) -> bool {
        uniforms.alpha == 1.0 && color_alpha_is_one(uniforms.color)
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
    pub alpha: f32,
    transform: Mat4,
}

impl MeshUniforms {
    pub fn new(model: Mat4, view: Mat4, projection: Mat4, color: u32) -> Self {
        Self {
            model,
            view,
            projection,
            color,
            alpha: 1.0,
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

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
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
                * Vec4::new(
                    vertex.position().x,
                    vertex.position().y,
                    vertex.position().z,
                    1.0,
                ),
            (),
        )
    }
}

impl FragmentStage<(), MeshUniforms> for MeshShader {
    fn run(&self, _: &(), uniforms: &MeshUniforms) -> u32 {
        multiply_alpha(uniforms.color, uniforms.alpha)
    }

    fn is_opaque(&self, uniforms: &MeshUniforms) -> bool {
        uniforms.alpha == 1.0 && color_alpha_is_one(uniforms.color)
    }

    fn model_view(&self, uniforms: &MeshUniforms) -> Option<Mat4> {
        Some(uniforms.view() * uniforms.model())
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
    pub alpha: f32,
}

impl<'a> TexturedUniforms<'a> {
    pub const fn new(transform: Mat4, texture: &'a Texture, filter: TextureFilter) -> Self {
        Self {
            transform,
            texture,
            filter,
            alpha: 1.0,
        }
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
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

/// A textured shader must use `RenderFrame::draw_with_sampling` or
/// `RenderFrame::draw_mesh_with_sampling`. It has no plain fragment-stage
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
/// pipeline.render(&mut framebuffer, |frame, target| {
///     frame.draw_mesh(target, &mesh, &uniforms);
/// });
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
        let position = Vec4::new(
            vertex.position().x,
            vertex.position().y,
            vertex.position().z,
            1.0,
        );
        VertexOutput::new(
            uniforms.transform * position,
            TexturedVaryings {
                texcoord: vertex.texcoord().unwrap_or(crate::math::Vec2::ZERO),
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
        argb8888_linear(pixel[3] * uniforms.alpha, [pixel[0], pixel[1], pixel[2]])
    }

    fn is_opaque(&self, uniforms: &TexturedUniforms<'a>) -> bool {
        uniforms.alpha == 1.0 && texture_has_no_alpha(uniforms.texture)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DirectionalLight {
    /// A normalized direction from a surface point toward the light.
    pub direction: Vec3,
    /// Linear color. Constructors and uniform mutators clamp it to nonnegative values.
    pub color: Vec3,
}

/// Maximum number of directional lights stored in one lighting uniform.
pub const MAX_DIRECTIONAL_LIGHTS: usize = 8;

/// Maximum number of point lights stored in one lighting uniform.
pub const MAX_POINT_LIGHTS: usize = 8;

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3) -> Self {
        Self {
            direction,
            color: linearize_color(nonnegative_color(color)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PointLight {
    pub position: Vec3,
    /// Linear color. Constructors and uniform mutators clamp it to nonnegative values.
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
            color: linearize_color(nonnegative_color(color)),
            constant_attenuation: constant_attenuation.max(0.0),
            linear_attenuation: linear_attenuation.max(0.0),
            quadratic_attenuation: quadratic_attenuation.max(0.0),
        }
    }
}

fn nonnegative_color(color: Vec3) -> Vec3 {
    Vec3::new(color.x.max(0.0), color.y.max(0.0), color.z.max(0.0))
}

fn sanitize_directional_light(mut light: DirectionalLight) -> DirectionalLight {
    light.color = nonnegative_color(light.color);
    light
}

fn sanitize_point_light(mut light: PointLight) -> PointLight {
    light.color = nonnegative_color(light.color);
    light.constant_attenuation = light.constant_attenuation.max(0.0);
    light.linear_attenuation = light.linear_attenuation.max(0.0);
    light.quadratic_attenuation = light.quadratic_attenuation.max(0.0);
    light
}

/// Blinn-Phong uniforms with cache-safe matrix updates and fixed light arrays.
///
/// The arrays keep uniform clones cheap and preserve deterministic submission
/// order. Directional light zero owns the single optional shadow map. Other
/// lights are always unshadowed.
/// Material colors are private and use the same clamp-at-zero boundary policy.
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
    ambient_color: Vec3,
    diffuse_color: Vec3,
    specular_color: Vec3,
    pub shininess: f32,
    pub camera_position: Vec3,
    pub alpha: f32,
    directional_lights: [DirectionalLight; MAX_DIRECTIONAL_LIGHTS],
    directional_light_count: usize,
    point_lights: [PointLight; MAX_POINT_LIGHTS],
    point_light_count: usize,
    shadow_state: Option<ShadowState>,
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
            ambient_color: linearize_color(nonnegative_color(ambient_color)),
            diffuse_color: linearize_color(nonnegative_color(diffuse_color)),
            specular_color: linearize_color(nonnegative_color(specular_color)),
            shininess,
            camera_position,
            alpha: 1.0,
            directional_lights: std::array::from_fn(|index| {
                if index == 0 {
                    sanitize_directional_light(directional_light)
                } else {
                    DirectionalLight::default()
                }
            }),
            directional_light_count: 1,
            point_lights: std::array::from_fn(|index| {
                if index == 0 {
                    sanitize_point_light(point_light)
                } else {
                    PointLight::default()
                }
            }),
            point_light_count: 1,
            shadow_state: None,
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

    /// Returns the material ambient color in linear space.
    pub const fn ambient_color(&self) -> Vec3 {
        self.ambient_color
    }

    /// Returns the material diffuse color in linear space.
    pub const fn diffuse_color(&self) -> Vec3 {
        self.diffuse_color
    }

    /// Returns the material specular color in linear space.
    pub const fn specular_color(&self) -> Vec3 {
        self.specular_color
    }

    /// Sets the authored sRGB ambient color and clamps it to nonnegative values.
    pub fn set_ambient_color(&mut self, color: Vec3) {
        self.ambient_color = linearize_color(nonnegative_color(color));
    }

    /// Sets the authored sRGB diffuse color and clamps it to nonnegative values.
    pub fn set_diffuse_color(&mut self, color: Vec3) {
        self.diffuse_color = linearize_color(nonnegative_color(color));
    }

    /// Sets the authored sRGB specular color and clamps it to nonnegative values.
    pub fn set_specular_color(&mut self, color: Vec3) {
        self.specular_color = linearize_color(nonnegative_color(color));
    }

    /// Returns the first directional light, or a zero light when the array is empty.
    pub const fn directional_light(&self) -> DirectionalLight {
        self.directional_lights[0]
    }

    /// Returns the first point light, or a zero light when the array is empty.
    pub const fn point_light(&self) -> PointLight {
        self.point_lights[0]
    }

    /// Returns the active directional lights in array order.
    pub fn directional_lights(&self) -> &[DirectionalLight] {
        &self.directional_lights[..self.directional_light_count]
    }

    /// Returns the active point lights in array order.
    pub fn point_lights(&self) -> &[PointLight] {
        &self.point_lights[..self.point_light_count]
    }

    pub const fn directional_light_count(&self) -> usize {
        self.directional_light_count
    }

    pub const fn point_light_count(&self) -> usize {
        self.point_light_count
    }

    /// Appends a directional light and returns its array index.
    pub fn add_directional_light(
        &mut self,
        light: DirectionalLight,
    ) -> Result<usize, &'static str> {
        if self.directional_light_count == MAX_DIRECTIONAL_LIGHTS {
            return Err("directional light capacity reached");
        }
        let index = self.directional_light_count;
        self.directional_lights[index] = sanitize_directional_light(light);
        self.directional_light_count += 1;
        Ok(index)
    }

    /// Appends a point light and returns its array index.
    pub fn add_point_light(&mut self, light: PointLight) -> Result<usize, &'static str> {
        if self.point_light_count == MAX_POINT_LIGHTS {
            return Err("point light capacity reached");
        }
        let index = self.point_light_count;
        self.point_lights[index] = sanitize_point_light(light);
        self.point_light_count += 1;
        Ok(index)
    }

    /// Replaces a directional light. Index zero owns the optional shadow map.
    pub fn set_directional_light(
        &mut self,
        index: usize,
        light: DirectionalLight,
    ) -> Result<(), &'static str> {
        let Some(slot) = self.directional_lights.get_mut(index) else {
            return Err("directional light index out of bounds");
        };
        if index >= self.directional_light_count {
            return Err("directional light index out of bounds");
        }
        *slot = sanitize_directional_light(light);
        if index == 0 {
            self.shadow_state = None;
        }
        Ok(())
    }

    /// Replaces a point light at an existing array index.
    pub fn set_point_light(&mut self, index: usize, light: PointLight) -> Result<(), &'static str> {
        if index >= self.point_light_count {
            return Err("point light index out of bounds");
        }
        self.point_lights[index] = sanitize_point_light(light);
        Ok(())
    }

    /// Removes a directional light and compacts later entries.
    pub fn remove_directional_light(&mut self, index: usize) -> Option<DirectionalLight> {
        if index >= self.directional_light_count {
            return None;
        }
        let removed = self.directional_lights[index];
        self.directional_lights[index..self.directional_light_count].rotate_left(1);
        self.directional_light_count -= 1;
        self.directional_lights[self.directional_light_count] = DirectionalLight::default();
        if index == 0 {
            self.shadow_state = None;
        }
        Some(removed)
    }

    /// Removes a point light and compacts later entries.
    pub fn remove_point_light(&mut self, index: usize) -> Option<PointLight> {
        if index >= self.point_light_count {
            return None;
        }
        let removed = self.point_lights[index];
        self.point_lights[index..self.point_light_count].rotate_left(1);
        self.point_light_count -= 1;
        self.point_lights[self.point_light_count] = PointLight::default();
        Some(removed)
    }

    pub fn clear_directional_lights(&mut self) {
        self.directional_lights[..self.directional_light_count].fill(DirectionalLight::default());
        self.directional_light_count = 0;
        self.shadow_state = None;
    }

    pub fn clear_point_lights(&mut self) {
        self.point_lights[..self.point_light_count].fill(PointLight::default());
        self.point_light_count = 0;
    }

    pub const fn light_view_projection(&self) -> Mat4 {
        match &self.shadow_state {
            Some(state) => state.light_view_projection(),
            None => Mat4::IDENTITY,
        }
    }

    pub fn shadow_map(&self) -> Option<&ShadowMap> {
        self.shadow_state.as_ref().map(ShadowState::shadow_map)
    }

    pub const fn shadow_bias(&self) -> (f32, f32) {
        match &self.shadow_state {
            Some(state) => state.bias(),
            None => (0.002, 0.02),
        }
    }

    /// Replaces directional light zero and its complete shadow state as one update.
    ///
    /// The default bias is `(0.002, 0.02)`. The shadow compare uses
    /// `max(constant, slope * (1 - N dot L))`.
    pub fn set_directional_shadow(
        &mut self,
        directional_light: DirectionalLight,
        shadow_state: Option<ShadowState>,
    ) {
        if self.directional_light_count == 0 {
            self.directional_lights[0] = sanitize_directional_light(directional_light);
            self.directional_light_count = 1;
        } else {
            self.directional_lights[0] = sanitize_directional_light(directional_light);
        }
        self.shadow_state = shadow_state;
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

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
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
    let local_position = Vec4::new(
        vertex.position().x,
        vertex.position().y,
        vertex.position().z,
        1.0,
    );
    let world_position = uniforms.model() * local_position;
    let world_position = Vec3::new(
        world_position.x / world_position.w,
        world_position.y / world_position.w,
        world_position.z / world_position.w,
    );
    let normal = vertex.normal().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let normal = (uniforms.normal_matrix() * normal).normalize();
    let light_space_position = uniforms.light_view_projection()
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

    fn is_opaque(&self, uniforms: &BlinnPhongUniforms) -> bool {
        uniforms.alpha == 1.0
    }

    fn model_view(&self, uniforms: &BlinnPhongUniforms) -> Option<Mat4> {
        Some(uniforms.view() * uniforms.model())
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

        argb8888_linear(uniforms.alpha, [lighted.x, lighted.y, lighted.z])
    }
}

/// Blinn-Phong lighting with a linear-space cube-map reflection mix.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentBlinnPhongUniforms<'a> {
    pub lighting: BlinnPhongUniforms,
    pub environment: &'a CubeTexture,
    reflectivity: f32,
}

impl<'a> EnvironmentBlinnPhongUniforms<'a> {
    pub fn new(
        lighting: BlinnPhongUniforms,
        environment: &'a CubeTexture,
        reflectivity: f32,
    ) -> Self {
        Self {
            lighting,
            environment,
            reflectivity: sanitize_unit(reflectivity),
        }
    }

    pub const fn reflectivity(&self) -> f32 {
        self.reflectivity
    }

    pub fn set_reflectivity(&mut self, reflectivity: f32) {
        self.reflectivity = sanitize_unit(reflectivity);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EnvironmentBlinnPhongShader;

impl<'a> VertexStage<MeshVertex, EnvironmentBlinnPhongUniforms<'a>>
    for EnvironmentBlinnPhongShader
{
    type Varyings = BlinnPhongVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &EnvironmentBlinnPhongUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        let prepared = prepare_blinn_phong_vertex(vertex, &uniforms.lighting);
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

impl<'a> FragmentStage<BlinnPhongVaryings, EnvironmentBlinnPhongUniforms<'a>>
    for EnvironmentBlinnPhongShader
{
    fn run(
        &self,
        varyings: &BlinnPhongVaryings,
        uniforms: &EnvironmentBlinnPhongUniforms<'a>,
    ) -> u32 {
        let lit = evaluate_lighting(
            varyings.world_position,
            varyings.normal,
            varyings.light_space_position,
            &uniforms.lighting,
            Vec3::new(1.0, 1.0, 1.0),
        );
        let normal = varyings.normal.normalize();
        let view_direction =
            (uniforms.lighting.camera_position - varyings.world_position).normalize();
        let reflected_direction = reflection_vector(view_direction, normal);
        let environment = uniforms.environment.sample(reflected_direction);
        let reflectivity = uniforms.reflectivity;
        let color = lit * (1.0 - reflectivity)
            + Vec3::new(environment[0], environment[1], environment[2]) * reflectivity;
        argb8888_linear(uniforms.lighting.alpha, [color.x, color.y, color.z])
    }

    fn is_opaque(&self, uniforms: &EnvironmentBlinnPhongUniforms<'a>) -> bool {
        uniforms.lighting.alpha == 1.0
    }

    fn model_view(&self, uniforms: &EnvironmentBlinnPhongUniforms<'a>) -> Option<Mat4> {
        Some(uniforms.lighting.view() * uniforms.lighting.model())
    }
}

/// Reflects the incoming camera ray around a surface normal.
///
/// `view_direction` points from the surface to the camera. The incident ray
/// is its negation, so the returned direction points toward the environment.
pub fn reflection_vector(view_direction: Vec3, normal: Vec3) -> Vec3 {
    let incident = -view_direction.normalize();
    let normal = normal.normalize();
    (incident - normal * (2.0 * incident.dot(normal))).normalize()
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
    // The renderer has one shadow map, and it belongs to directional light zero.
    for (index, light) in uniforms.directional_lights().iter().enumerate() {
        let visibility = if index == 0 {
            directional_visibility
        } else {
            1.0
        };
        lighted = lighted
            + evaluate_light(
                normal,
                view_direction,
                light.direction.normalize(),
                light.color,
                visibility,
                uniforms,
                albedo,
            );
    }

    for light in uniforms.point_lights() {
        let to_point = light.position - world_position;
        let distance = to_point.length();
        let point_direction = to_point.normalize();
        let denominator = light.constant_attenuation
            + light.linear_attenuation * distance
            + light.quadratic_attenuation * distance * distance;
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
                light.color,
                attenuation,
                uniforms,
                albedo,
            );
    }

    lighted
}

impl BlinnPhongUniforms {
    fn shadow_visibility(&self, light_space_position: Vec4, normal: Vec3) -> f32 {
        let Some(shadow_state) = self.shadow_state.as_ref() else {
            return 1.0;
        };
        let shadow_map = shadow_state.shadow_map();
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
            .dot(self.directional_light().direction.normalize())
            .clamp(0.0, 1.0);
        let (constant_bias, slope_bias) = shadow_state.bias();
        let bias = constant_bias.max(slope_bias * (1.0 - normal_dot_light));
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
    pub alpha: f32,
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
            alpha: 1.0,
        }
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
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
                texcoord: vertex.texcoord().unwrap_or(crate::math::Vec2::ZERO),
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

        argb8888_linear(
            pixel[3] * uniforms.alpha * uniforms.lighting.alpha,
            [lighted.x, lighted.y, lighted.z],
        )
    }

    fn is_opaque(&self, uniforms: &TexturedBlinnPhongUniforms<'a>) -> bool {
        uniforms.alpha == 1.0
            && uniforms.lighting.alpha == 1.0
            && texture_has_no_alpha(uniforms.texture)
    }

    fn model_view(&self, uniforms: &TexturedBlinnPhongUniforms<'a>) -> Option<Mat4> {
        Some(uniforms.lighting.view() * uniforms.lighting.model())
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

/// Uniforms for a tangent-space normal map layered on the textured lighting path.
/// The normal map must use [`crate::image::ColorSpace::Linear`].
#[derive(Clone, Debug, PartialEq)]
pub struct NormalMappedBlinnPhongUniforms<'a> {
    pub lighting: BlinnPhongUniforms,
    pub texture: &'a Texture,
    pub normal_map: &'a Texture,
    pub filter: TextureFilter,
}

impl<'a> NormalMappedBlinnPhongUniforms<'a> {
    pub fn new(
        lighting: BlinnPhongUniforms,
        texture: &'a Texture,
        normal_map: &'a Texture,
        filter: TextureFilter,
    ) -> Result<Self, &'static str> {
        if normal_map.color_space() != ColorSpace::Linear {
            return Err("normal maps must use ColorSpace::Linear");
        }
        Ok(Self {
            lighting,
            texture,
            normal_map,
            filter,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NormalMappedBlinnPhongShader;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalMappedBlinnPhongVaryings {
    pub world_position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec4,
    pub texcoord: Vec2,
    pub light_space_position: Vec4,
}

impl Varyings for NormalMappedBlinnPhongVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        Self {
            world_position: a.world_position * weights.x
                + b.world_position * weights.y
                + c.world_position * weights.z,
            normal: a.normal * weights.x + b.normal * weights.y + c.normal * weights.z,
            tangent: a.tangent * weights.x + b.tangent * weights.y + c.tangent * weights.z,
            texcoord: a.texcoord * weights.x + b.texcoord * weights.y + c.texcoord * weights.z,
            light_space_position: a.light_space_position * weights.x
                + b.light_space_position * weights.y
                + c.light_space_position * weights.z,
        }
    }
}

impl SamplingVaryings for NormalMappedBlinnPhongVaryings {
    fn texture_coordinates(&self) -> Vec2 {
        self.texcoord
    }
}

impl<'a> VertexStage<MeshVertex, NormalMappedBlinnPhongUniforms<'a>>
    for NormalMappedBlinnPhongShader
{
    type Varyings = NormalMappedBlinnPhongVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &NormalMappedBlinnPhongUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        let prepared = prepare_blinn_phong_vertex(vertex, &uniforms.lighting);
        let tangent = vertex
            .tangent()
            .map_or(Vec4::new(0.0, 0.0, 0.0, 0.0), |tangent| {
                let transformed =
                    uniforms.lighting.model() * Vec4::new(tangent.x, tangent.y, tangent.z, 0.0);
                Vec4::new(
                    transformed.x,
                    transformed.y,
                    transformed.z,
                    tangent.w * model_handedness(uniforms.lighting.model()),
                )
            });
        VertexOutput::new(
            prepared.clip_position,
            NormalMappedBlinnPhongVaryings {
                world_position: prepared.world_position,
                normal: prepared.normal,
                tangent,
                texcoord: vertex.texcoord().unwrap_or(Vec2::ZERO),
                light_space_position: prepared.light_space_position,
            },
        )
    }
}

fn model_handedness(model: Mat4) -> f32 {
    let x = Vec3::new(model.get(0, 0), model.get(1, 0), model.get(2, 0));
    let y = Vec3::new(model.get(0, 1), model.get(1, 1), model.get(2, 1));
    let z = Vec3::new(model.get(0, 2), model.get(1, 2), model.get(2, 2));
    if x.dot(y.cross(z)) < 0.0 { -1.0 } else { 1.0 }
}

impl<'a> SampledFragmentStage<NormalMappedBlinnPhongVaryings, NormalMappedBlinnPhongUniforms<'a>>
    for NormalMappedBlinnPhongShader
{
    fn run_with_sampling(
        &self,
        varyings: &NormalMappedBlinnPhongVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        uniforms: &NormalMappedBlinnPhongUniforms<'a>,
    ) -> u32 {
        let derivatives = TextureDerivatives {
            ddx: derivatives.ddx,
            ddy: derivatives.ddy,
        };
        let albedo_pixel = sample_texture(
            uniforms.texture,
            varyings.texcoord,
            uniforms.filter,
            derivatives,
        );
        let normal_pixel = sample_texture(
            uniforms.normal_map,
            varyings.texcoord,
            uniforms.filter,
            derivatives,
        );
        let normal = tangent_space_normal(varyings, normal_pixel);
        let albedo = Vec3::new(albedo_pixel[0], albedo_pixel[1], albedo_pixel[2]);
        let lighted = evaluate_lighting(
            varyings.world_position,
            normal,
            varyings.light_space_position,
            &uniforms.lighting,
            albedo,
        );
        argb8888_linear(
            albedo_pixel[3] * uniforms.lighting.alpha,
            [lighted.x, lighted.y, lighted.z],
        )
    }

    fn is_opaque(&self, uniforms: &NormalMappedBlinnPhongUniforms<'a>) -> bool {
        uniforms.lighting.alpha == 1.0 && texture_has_no_alpha(uniforms.texture)
    }

    fn model_view(&self, uniforms: &NormalMappedBlinnPhongUniforms<'a>) -> Option<Mat4> {
        Some(uniforms.lighting.view() * uniforms.lighting.model())
    }
}

fn tangent_space_normal(varyings: &NormalMappedBlinnPhongVaryings, pixel: [f32; 4]) -> Vec3 {
    let sampled = remap_normal_sample(Vec3::new(pixel[0], pixel[1], pixel[2]));
    let normal = varyings.normal.normalize();
    let tangent = Vec3::new(varyings.tangent.x, varyings.tangent.y, varyings.tangent.z);
    if tangent.length() == 0.0 {
        return normal;
    }
    let tangent = (tangent - normal * normal.dot(tangent)).normalize();
    if tangent.length() == 0.0 {
        return normal;
    }
    let sign = if varyings.tangent.w < 0.0 { -1.0 } else { 1.0 };
    let bitangent = normal.cross(tangent) * sign;
    (tangent * sampled.x + bitangent * sampled.y + normal * sampled.z).normalize()
}

fn remap_normal_sample(sample: Vec3) -> Vec3 {
    sample * 2.0 - Vec3::new(1.0, 1.0, 1.0)
}

fn linearize_color(color: Vec3) -> Vec3 {
    Vec3::new(
        srgb_to_linear(color.x),
        srgb_to_linear(color.y),
        srgb_to_linear(color.z),
    )
}

fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn multiply_alpha(color: u32, alpha: f32) -> u32 {
    let [source_alpha, red, green, blue] = color.to_be_bytes();
    crate::fb::argb8888(
        (f32::from(source_alpha) / 255.0 * alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
        red,
        green,
        blue,
    )
}

fn color_alpha_is_one(color: u32) -> bool {
    color.to_be_bytes()[0] == 255
}

fn texture_has_no_alpha(texture: &Texture) -> bool {
    texture.pixels().iter().all(|pixel| pixel[3] == 255)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::{Framebuffer, argb8888};
    use crate::math::Vec2;

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
    fn reflection_vector_matches_hand_computed_normal_incidence() {
        let reflected = reflection_vector(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(reflected, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn reflection_parameter_is_sanitized_at_uniform_boundary() {
        let cube = crate::skybox::CubeTexture::new(std::array::from_fn(|_| {
            Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap()
        }))
        .unwrap();
        let mut uniforms = EnvironmentBlinnPhongUniforms::new(uniforms(), &cube, f32::NAN);
        assert_eq!(uniforms.reflectivity(), 0.0);
        uniforms.set_reflectivity(2.0);
        assert_eq!(uniforms.reflectivity(), 1.0);
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
        uniforms.set_directional_shadow(
            DirectionalLight::new(Vec3::new(0.0, 0.8, -0.6), Vec3::new(1.0, 1.0, 1.0)),
            None,
        );
        assert_eq!(BlinnPhongShader::shade(&varyings, &uniforms), 0xff000000);
    }

    #[test]
    fn normal_map_remap_is_hand_computed() {
        let remapped = remap_normal_sample(Vec3::new(0.75, 0.5, 1.0));
        assert_eq!(remapped, Vec3::new(0.5, 0.0, 1.0));
    }

    #[test]
    fn normal_map_without_tangent_falls_back_to_geometric_normal() {
        let varyings = NormalMappedBlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 2.0),
            tangent: Vec4::new(0.0, 0.0, 0.0, 0.0),
            texcoord: Vec2::ZERO,
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        assert_eq!(
            tangent_space_normal(&varyings, [1.0, 0.0, 0.0, 1.0]),
            Vec3::new(0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn interpolated_tbn_is_orthonormal_after_renormalize() {
        let a = NormalMappedBlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
            texcoord: Vec2::ZERO,
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        let b = NormalMappedBlinnPhongVaryings {
            tangent: Vec4::new(0.0, 1.0, 0.0, 1.0),
            ..a
        };
        let c = NormalMappedBlinnPhongVaryings {
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
            ..a
        };
        let varyings =
            NormalMappedBlinnPhongVaryings::lerp3(&a, &b, &c, Vec3::new(0.25, 0.5, 0.25));
        let transformed = tangent_space_normal(&varyings, [1.0, 1.0, 1.0, 1.0]);
        let expected_y = (2.0_f32 / 3.0).sqrt();
        let expected_z = (1.0_f32 / 3.0).sqrt();
        assert!((transformed.x).abs() < 1e-6);
        assert!((transformed.y - expected_y).abs() < 1e-6);
        assert!((transformed.z - expected_z).abs() < 1e-6);
        assert!((transformed.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn negative_determinant_model_flips_tangent_handedness() {
        let mesh = crate::mesh::Mesh::parse(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
             vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
             f 1/1 2/2 3/3 4/4\n",
        )
        .unwrap();
        let albedo = Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap();
        let normal_map =
            Texture::new_with_color_space(1, 1, vec![[128, 128, 255, 255]], ColorSpace::Linear)
                .unwrap();
        let lighting = BlinnPhongUniforms::new(
            Mat4::scale(Vec3::new(1.0, 1.0, -1.0)),
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            8.0,
            Vec3::new(0.0, 0.0, 1.0),
            DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
            PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
        );
        let uniforms = NormalMappedBlinnPhongUniforms::new(
            lighting,
            &albedo,
            &normal_map,
            TextureFilter::Nearest,
        )
        .unwrap();
        let output = NormalMappedBlinnPhongShader.run(mesh.vertex(0).unwrap(), &uniforms);
        assert_eq!(output.varyings.tangent.w, -1.0);
        let reconstructed_bitangent = output.varyings.normal.cross(Vec3::new(
            output.varyings.tangent.x,
            output.varyings.tangent.y,
            output.varyings.tangent.z,
        )) * output.varyings.tangent.w;
        assert!((reconstructed_bitangent.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normal_map_uniform_rejects_srgb_normal_texture() {
        let albedo = Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap();
        let normal_map = Texture::new(1, 1, vec![[128, 128, 255, 255]]).unwrap();
        assert!(
            NormalMappedBlinnPhongUniforms::new(
                uniforms(),
                &albedo,
                &normal_map,
                TextureFilter::Nearest,
            )
            .is_err()
        );
    }

    #[test]
    fn normal_mapped_lighting_alpha_controls_output_and_classification() {
        let albedo = Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap();
        let normal_map =
            Texture::new_with_color_space(1, 1, vec![[128, 128, 255, 255]], ColorSpace::Linear)
                .unwrap();
        let mut lighting = uniforms();
        lighting.set_alpha(0.5);
        let uniforms = NormalMappedBlinnPhongUniforms::new(
            lighting,
            &albedo,
            &normal_map,
            TextureFilter::Nearest,
        )
        .unwrap();
        let varyings = NormalMappedBlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
            texcoord: Vec2::ZERO,
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        let output = NormalMappedBlinnPhongShader.run_with_sampling(
            &varyings,
            &crate::pipeline::SampleDerivatives::default(),
            &uniforms,
        );
        assert_eq!(output.to_be_bytes()[0], 128);
        assert!(!NormalMappedBlinnPhongShader.is_opaque(&uniforms));
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
    fn blinn_phong_light_mutators_rebuild_array_state_immediately() {
        let mut uniforms = uniforms();
        uniforms.clear_directional_lights();
        uniforms.clear_point_lights();
        assert_eq!(uniforms.directional_light_count(), 0);
        assert_eq!(uniforms.point_light_count(), 0);

        let directional = DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0));
        let point = PointLight::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.8, 0.4, 0.2),
            1.0,
            0.1,
            0.01,
        );
        assert_eq!(uniforms.add_directional_light(directional), Ok(0));
        assert_eq!(uniforms.add_point_light(point), Ok(0));
        assert_eq!(uniforms.directional_lights(), &[directional]);
        assert_eq!(uniforms.point_lights(), &[point]);

        let moved_directional =
            DirectionalLight::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.2, 0.3, 0.4));
        let moved_point = PointLight::new(
            Vec3::new(-1.0, -2.0, -3.0),
            Vec3::new(0.4, 0.5, 0.6),
            2.0,
            0.2,
            0.02,
        );
        assert_eq!(uniforms.set_directional_light(0, moved_directional), Ok(()));
        assert_eq!(uniforms.set_point_light(0, moved_point), Ok(()));
        assert_eq!(uniforms.directional_light(), moved_directional);
        assert_eq!(uniforms.point_light(), moved_point);
        assert_eq!(
            uniforms.remove_directional_light(0),
            Some(moved_directional)
        );
        assert_eq!(uniforms.remove_point_light(0), Some(moved_point));
        assert_eq!(uniforms.directional_light_count(), 0);
        assert_eq!(uniforms.point_light_count(), 0);
    }

    #[test]
    fn light_uniform_boundary_clamps_negative_colors_and_intensities() {
        let directional =
            DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(-1.0, 0.5, -0.25));
        assert_eq!(directional.color.x, 0.0);
        assert_eq!(directional.color.z, 0.0);
        assert!(directional.color.y > 0.0);

        let point = PointLight::new(Vec3::ZERO, Vec3::new(-1.0, 0.5, -0.25), -1.0, -0.5, -0.25);
        assert_eq!(point.color.x, 0.0);
        assert_eq!(point.color.z, 0.0);
        assert_eq!(point.constant_attenuation, 0.0);
        assert_eq!(point.linear_attenuation, 0.0);
        assert_eq!(point.quadratic_attenuation, 0.0);

        let mut uniforms = uniforms();
        uniforms.clear_directional_lights();
        uniforms.clear_point_lights();
        let public_directional = DirectionalLight {
            color: Vec3::new(-1.0, 2.0, -3.0),
            ..DirectionalLight::default()
        };
        let public_point = PointLight {
            color: Vec3::new(-1.0, 2.0, -3.0),
            constant_attenuation: -1.0,
            linear_attenuation: -2.0,
            quadratic_attenuation: -3.0,
            ..PointLight::default()
        };
        uniforms.add_directional_light(public_directional).unwrap();
        uniforms.add_point_light(public_point).unwrap();
        assert_eq!(
            uniforms.directional_lights()[0].color,
            Vec3::new(0.0, 2.0, 0.0)
        );
        assert_eq!(uniforms.point_lights()[0].color, Vec3::new(0.0, 2.0, 0.0));
        assert_eq!(uniforms.point_lights()[0].constant_attenuation, 0.0);
        assert_eq!(uniforms.point_lights()[0].linear_attenuation, 0.0);
        assert_eq!(uniforms.point_lights()[0].quadratic_attenuation, 0.0);
    }

    #[test]
    fn material_uniform_boundary_clamps_negative_colors() {
        let mut uniforms = BlinnPhongUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(-1.0, 0.5, -0.25),
            Vec3::new(-0.75, 0.4, -0.1),
            Vec3::new(-0.5, 0.3, -0.05),
            8.0,
            Vec3::ZERO,
            DirectionalLight::default(),
            PointLight::default(),
        );
        assert_eq!(uniforms.ambient_color().x, 0.0);
        assert_eq!(uniforms.ambient_color().z, 0.0);
        assert_eq!(uniforms.diffuse_color().x, 0.0);
        assert_eq!(uniforms.diffuse_color().z, 0.0);
        assert_eq!(uniforms.specular_color().x, 0.0);
        assert_eq!(uniforms.specular_color().z, 0.0);

        uniforms.set_ambient_color(Vec3::new(-1.0, 0.2, -1.0));
        uniforms.set_diffuse_color(Vec3::new(-1.0, 0.3, -1.0));
        uniforms.set_specular_color(Vec3::new(-1.0, 0.4, -1.0));
        assert_eq!(uniforms.ambient_color().x, 0.0);
        assert_eq!(uniforms.diffuse_color().x, 0.0);
        assert_eq!(uniforms.specular_color().x, 0.0);
    }

    #[test]
    fn point_light_removal_compacts_order_and_changes_production_shading() {
        let mut base = uniforms();
        base.set_ambient_color(Vec3::ZERO);
        base.set_diffuse_color(Vec3::new(1.0, 1.0, 1.0));
        base.set_specular_color(Vec3::ZERO);
        base.clear_directional_lights();
        base.clear_point_lights();
        let first = PointLight::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            0.0,
            0.0,
        );
        let middle = PointLight::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0,
            0.0,
            0.0,
        );
        let last = PointLight::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            4.0,
            0.0,
            0.0,
        );
        base.add_point_light(first).unwrap();
        base.add_point_light(middle).unwrap();
        base.add_point_light(last).unwrap();
        let varyings = BlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        let original_shading = BlinnPhongShader::shade(&varyings, &base);

        let mut without_first = base.clone();
        assert_eq!(without_first.remove_point_light(0), Some(first));
        assert_eq!(without_first.point_lights(), &[middle, last]);
        let first_removed_shading = BlinnPhongShader::shade(&varyings, &without_first);

        let mut without_middle = base;
        assert_eq!(without_middle.remove_point_light(1), Some(middle));
        assert_eq!(without_middle.point_lights(), &[first, last]);
        let middle_removed_shading = BlinnPhongShader::shade(&varyings, &without_middle);

        assert_ne!(first_removed_shading, original_shading);
        assert_ne!(middle_removed_shading, original_shading);
        assert_ne!(first_removed_shading, middle_removed_shading);
    }

    #[test]
    fn blinn_phong_two_directional_lights_accumulate_in_linear_space() {
        let varyings = BlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        let mut uniforms = BlinnPhongUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::ZERO,
            8.0,
            Vec3::new(0.0, 0.0, 1.0),
            DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.5, 0.0, 0.0)),
            PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
        );
        uniforms.clear_directional_lights();
        uniforms.clear_point_lights();
        uniforms
            .add_directional_light(DirectionalLight::new(
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.5, 0.0, 0.0),
            ))
            .unwrap();
        uniforms
            .add_directional_light(DirectionalLight::new(
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.5, 0.0),
            ))
            .unwrap();

        let expected = argb8888_linear(1.0, [srgb_to_linear(0.5), srgb_to_linear(0.5), 0.0]);
        assert_eq!(BlinnPhongShader::shade(&varyings, &uniforms), expected);
    }

    #[test]
    fn blinn_phong_zero_lights_keep_ambient_only() {
        let mut uniforms = uniforms();
        uniforms.set_ambient_color(Vec3::new(0.25, 0.5, 0.75));
        uniforms.clear_directional_lights();
        uniforms.clear_point_lights();
        let varyings = BlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        let expected = argb8888_linear(
            1.0,
            [
                srgb_to_linear(0.25),
                srgb_to_linear(0.5),
                srgb_to_linear(0.75),
            ],
        );
        assert_eq!(BlinnPhongShader::shade(&varyings, &uniforms), expected);
    }

    #[test]
    fn blinn_phong_many_lights_clamp_after_accumulation() {
        let mut uniforms = uniforms();
        uniforms.set_ambient_color(Vec3::ZERO);
        uniforms.set_diffuse_color(Vec3::new(1.0, 1.0, 1.0));
        uniforms.set_specular_color(Vec3::ZERO);
        uniforms.clear_directional_lights();
        uniforms.clear_point_lights();
        for _ in 0..6 {
            uniforms
                .add_point_light(PointLight::new(
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    6.0,
                    0.0,
                    0.0,
                ))
                .unwrap();
        }
        let varyings = BlinnPhongVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        assert_eq!(BlinnPhongShader::shade(&varyings, &uniforms), 0xffffffff);
        assert_ne!(
            BlinnPhongShader::shade(&varyings, &{
                let mut fewer = uniforms.clone();
                fewer.remove_point_light(0);
                fewer
            },),
            0xffffffff,
            "dropping one point light must fail the accumulation gate"
        );
    }

    fn pack_varyings(normal: Vec3, barycentric: Vec3, distance: f32) -> ShaderPackVaryings {
        ShaderPackVaryings {
            world_position: Vec3::ZERO,
            normal,
            barycentric,
            clip_xy: Vec2::ZERO,
            clip_w: 1.0,
            view_distance: distance,
        }
    }

    fn pack_vertex(position: Vec3) -> ShaderPackVertex {
        ShaderPackVertex::new(
            MeshVertex::new(position, None, Some(Vec3::new(0.0, 0.0, 1.0))),
            Vec3::new(1.0, 0.0, 0.0),
        )
    }

    #[test]
    fn shader_pack_toon_bands_and_edge_gate_are_mutation_gates() {
        assert_eq!(toon_band(0.1), 0.15);
        assert_eq!(toon_band(0.4), 0.40);
        assert_eq!(toon_band(0.6), 0.70);
        assert_eq!(toon_band(0.9), 1.0);

        let uniforms = ToonUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.8, 0.4, 0.2),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let interior = pack_varyings(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.33, 0.34, 0.33), 1.0);
        let edge = pack_varyings(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.01, 0.49, 0.50), 1.0);
        assert_ne!(
            ToonShader::shade(&interior, &uniforms),
            ToonShader::shade(&edge, &uniforms)
        );
        assert_ne!(
            toon_band(0.6),
            0.6,
            "removing quantization must fail this gate"
        );
    }

    #[test]
    fn shader_pack_psx_snap_and_dither_are_mutation_gates() {
        let vertex = pack_vertex(Vec3::new(0.37, -0.21, 0.0));
        let mut uniforms = PsxUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.2, 0.4, 0.8),
            (4, 4),
        );
        uniforms.vertex_snap_grid = 4.0;
        let snapped = <PsxShader as VertexStage<ShaderPackVertex, PsxUniforms>>::run(
            &PsxShader, &vertex, &uniforms,
        );
        uniforms.vertex_snap_grid = f32::INFINITY;
        let unsnapped = <PsxShader as VertexStage<ShaderPackVertex, PsxUniforms>>::run(
            &PsxShader, &vertex, &uniforms,
        );
        assert_ne!(snapped.clip_position, unsnapped.clip_position);

        assert_eq!(BAYER4[0], [0, 8, 2, 10]);
        assert_eq!(bayer4_value(0, 0), 0.5 / 16.0);
        assert_eq!(bayer4_value(3, 3), 5.5 / 16.0);
        let color = Vec3::new(0.5, 0.5, 0.5);
        let first = ordered_dither_linear(color, Vec2::new(-0.75, 0.75), (4, 4), 2);
        let second = ordered_dither_linear(color, Vec2::new(-0.25, 0.75), (4, 4), 2);
        assert_ne!(first, second, "identity matrix must fail this dither gate");
    }

    #[test]
    fn shader_pack_dither_reconstructs_screen_position_after_interpolation() {
        let make_vertex = |ndc_x: f32, clip_w: f32| {
            let mut vertex =
                pack_varyings(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.33, 0.34, 0.33), 1.0);
            vertex.clip_xy = Vec2::new(ndc_x * clip_w, 0.0);
            vertex.clip_w = clip_w;
            vertex
        };
        let a = make_vertex(-0.84375, 1.0);
        let b = make_vertex(0.0, 2.0);
        let c = make_vertex(0.09375, 4.0);
        // Screen barycentrics are equal here. The core therefore supplies
        // corrected weights (4/7, 2/7, 1/7) for w=(1, 2, 4).
        let interpolated =
            ShaderPackVaryings::lerp3(&a, &b, &c, Vec3::new(4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0));
        let reconstructed_ndc = shader_pack::interpolated_screen_ndc(&interpolated);
        let pixel_x = ((reconstructed_ndc.x * 0.5 + 0.5) * 64.0).floor() as i32;
        assert!((reconstructed_ndc.x + 0.25).abs() < 1e-6);
        assert_eq!(pixel_x, 24);

        // The old perspective-correct NDC varying lands on pixel 17.
        let old_ndc: f32 = (-0.84375 * 4.0 + 0.0 * 2.0 + 0.09375) / 7.0;
        let old_pixel_x = ((old_ndc * 0.5 + 0.5) * 64.0).floor() as i32;
        assert_eq!(
            old_pixel_x, 17,
            "reverting to the old varying must fail this gate"
        );
    }

    fn render_dither_scene_at_depth(camera_depth: f32, clip_ws: [f32; 3]) -> Framebuffer {
        let camera = crate::camera::Camera::new(
            Vec3::new(0.0, 0.0, camera_depth),
            crate::math::Quat::IDENTITY,
            1.0,
            64.0 / 48.0,
            0.1,
            100.0,
        );
        let projection = camera.projection_matrix();
        let ndc = [
            Vec2::new(-0.84375, -0.55),
            Vec2::new(0.0, -0.55),
            Vec2::new(0.09375, 0.75),
        ];
        let vertices: [ShaderPackVertex; 3] = std::array::from_fn(|index| {
            let ndc = ndc[index];
            let clip_w = clip_ws[index];
            ShaderPackVertex::new(
                MeshVertex::new(
                    Vec3::new(
                        ndc.x * clip_w / projection.get(0, 0),
                        ndc.y * clip_w / projection.get(1, 1),
                        -clip_w,
                    ),
                    None,
                    Some(Vec3::new(0.0, 0.0, 1.0)),
                ),
                match index {
                    0 => Vec3::new(1.0, 0.0, 0.0),
                    1 => Vec3::new(0.0, 1.0, 0.0),
                    _ => Vec3::new(0.0, 0.0, 1.0),
                },
            )
        });
        let uniforms = DitherUniforms::new(
            Mat4::translate(Vec3::new(0.0, 0.0, camera_depth)),
            camera.view_matrix(),
            projection,
            Vec3::new(0.5, 0.5, 0.5),
            (64, 48),
        );
        let mut framebuffer = Framebuffer::new(64, 48);
        framebuffer.clear(argb8888(255, 8, 10, 16));
        let mut pipeline = crate::pipeline::Pipeline::new(DitherShader, DitherShader);
        pipeline.set_thread_count(1);
        pipeline.render(&mut framebuffer, |frame, target| {
            frame.draw(target, &vertices, &[[0, 1, 2]], &uniforms);
        });
        framebuffer
    }

    #[test]
    fn shader_pack_dither_depth_skew_is_screen_stationary_through_pipeline() {
        // The projected triangle stays fixed while its per-vertex w values change.
        let near = render_dither_scene_at_depth(4.0, [1.0, 2.0, 4.0]);
        let far = render_dither_scene_at_depth(8.0, [2.0, 3.0, 8.0]);
        assert!(near.color.iter().any(|&pixel| pixel != 0xff080a10));
        assert_eq!(near.color, far.color);
    }

    #[test]
    fn shader_pack_fog_falloff_is_hand_computed_and_mutation_gated() {
        assert_eq!(linear_fog_factor(2.0, 2.0, 6.0), 0.0);
        assert_eq!(linear_fog_factor(4.0, 2.0, 6.0), 0.5);
        assert_eq!(linear_fog_factor(8.0, 2.0, 6.0), 1.0);
        assert_ne!(
            linear_fog_factor(4.0, 2.0, 6.0),
            0.0,
            "zero fog density must fail this gate"
        );
    }

    #[test]
    fn shader_pack_normals_remap_is_mutation_gated() {
        let varyings = pack_varyings(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.33, 0.34, 0.33), 1.0);
        let uniforms = NormalsUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY);
        let remapped = <NormalsShader as FragmentStage<ShaderPackVaryings, NormalsUniforms>>::run(
            &NormalsShader,
            &varyings,
            &uniforms,
        );
        let without_remap = argb8888_linear(1.0, [0.0, 0.0, 1.0]);
        assert_ne!(
            remapped, without_remap,
            "dropping normal remap must fail this gate"
        );
    }

    #[test]
    fn shader_pack_wireframe_threshold_is_mutation_gated() {
        let uniforms = WireframeUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(1.0, 0.8, 0.1),
        );
        let edge = pack_varyings(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.5, 0.5), 1.0);
        let interior = pack_varyings(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.3, 0.4, 0.3), 1.0);
        let edge_color = <WireframeShader as FragmentStage<
            ShaderPackVaryings,
            WireframeUniforms,
        >>::run(&WireframeShader, &edge, &uniforms);
        let interior_color = <WireframeShader as FragmentStage<
            ShaderPackVaryings,
            WireframeUniforms,
        >>::run(&WireframeShader, &interior, &uniforms);
        assert_ne!(edge_color, interior_color);
        let mut disabled = uniforms;
        disabled.edge_threshold = 0.0;
        assert_eq!(
            <WireframeShader as FragmentStage<ShaderPackVaryings, WireframeUniforms>>::run(
                &WireframeShader,
                &edge,
                &disabled,
            ),
            interior_color,
            "zero edge threshold must fail this wireframe gate"
        );
    }

    #[test]
    fn shader_pack_uniform_setters_rebuild_all_caches() {
        macro_rules! assert_setters {
            ($uniforms:expr) => {{
                let mut uniforms = $uniforms;
                let model = Mat4::translate(Vec3::new(1.0, 2.0, 3.0));
                uniforms.set_model(model);
                assert_eq!(
                    uniforms.transform(),
                    Mat4::IDENTITY * Mat4::IDENTITY * model
                );

                let mut uniforms = $uniforms;
                let view = Mat4::scale(Vec3::new(2.0, 3.0, 4.0));
                uniforms.set_view(view);
                assert_eq!(uniforms.transform(), Mat4::IDENTITY * view * Mat4::IDENTITY);

                let mut uniforms = $uniforms;
                let projection = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.25);
                uniforms.set_projection(projection);
                assert_eq!(
                    uniforms.transform(),
                    projection * Mat4::IDENTITY * Mat4::IDENTITY
                );
            }};
        }

        assert_setters!(ToonUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.8, 0.4, 0.2),
            Vec3::new(0.0, 0.0, 1.0),
        ));
        assert_setters!(PsxUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.2, 0.4, 0.8),
            (64, 48),
        ));
        assert_setters!(DitherUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.8, 0.4, 0.2),
            (64, 48),
        ));
        assert_setters!(FogUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.2, 0.4, 0.8),
            Vec3::new(0.1, 0.1, 0.1),
            2.0,
            6.0,
        ));
        assert_setters!(NormalsUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
        ));
        assert_setters!(WireframeUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(1.0, 0.8, 0.1),
        ));
    }

    #[test]
    fn blinn_phong_shadow_state_updates_atomically() {
        let mut uniforms = uniforms();
        let shadow_map = ShadowMap::from_depth(1, 1, vec![0.0]).unwrap();
        let first_matrix = Mat4::translate(Vec3::new(2.0, 3.0, 4.0));
        let first_state = ShadowState::new(first_matrix, shadow_map.clone());
        uniforms.set_directional_shadow(
            DirectionalLight::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            Some(first_state),
        );

        let moved_matrix = Mat4::translate(Vec3::new(-2.0, 1.0, 5.0));
        let mut moved_state = ShadowState::new(moved_matrix, shadow_map);
        moved_state.set_bias(0.01, 0.04);
        let moved_light = DirectionalLight::new(
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(1.0, 1.0, 1.0),
        );
        uniforms.set_directional_shadow(moved_light, Some(moved_state));

        assert_eq!(uniforms.directional_light(), moved_light);
        assert_eq!(uniforms.light_view_projection(), moved_matrix);
        assert_eq!(uniforms.shadow_bias(), (0.01, 0.04));
        assert!(uniforms.shadow_map().is_some());
    }
}
