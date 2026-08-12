//! Cook-Torrance GGX shaders.
//!
//! This family shares the existing world-space vertex, light, shadow, texture,
//! and tangent seams with Blinn-Phong. LDR calls encode at the write boundary;
//! HDR calls return unclamped linear lighting to the post chain.

use super::*;
use std::f32::consts::PI;

/// Disney's roughness convention: the microfacet alpha is roughness squared.
pub const GGX_MIN_ROUGHNESS: f32 = 0.045;
const GGX_EPSILON: f32 = 1.0e-6;

/// Sanitized Cook-Torrance material and the shared scene lighting state.
///
/// Base color is authored in sRGB for [`Self::new`]. glTF uses
/// [`Self::new_with_linear_base_color`] because its factor is already linear.
/// Roughness is clamped to 0.045 so the GGX distribution cannot become
/// singular. Metallic is clamped to `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CookTorranceUniforms {
    pub lighting: BlinnPhongUniforms,
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
}

impl CookTorranceUniforms {
    pub fn new(
        lighting: BlinnPhongUniforms,
        base_color: Vec3,
        metallic: f32,
        roughness: f32,
    ) -> Self {
        Self::new_with_linear_base_color(
            lighting,
            linearize_color(sanitize_base_color(base_color)),
            metallic,
            roughness,
        )
    }

    pub fn new_with_linear_base_color(
        lighting: BlinnPhongUniforms,
        base_color: Vec3,
        metallic: f32,
        roughness: f32,
    ) -> Self {
        Self {
            lighting,
            base_color: sanitize_base_color(base_color),
            metallic: sanitize_unit(metallic),
            roughness: sanitize_roughness(roughness),
        }
    }

    pub const fn base_color(&self) -> Vec3 {
        self.base_color
    }

    pub const fn metallic(&self) -> f32 {
        self.metallic
    }

    pub const fn roughness(&self) -> f32 {
        self.roughness
    }

    /// Sets an authored sRGB base color and rebuilds the stored linear value.
    pub fn set_base_color(&mut self, base_color: Vec3) {
        self.base_color = linearize_color(sanitize_base_color(base_color));
    }

    pub fn set_base_color_linear(&mut self, base_color: Vec3) {
        self.base_color = sanitize_base_color(base_color);
    }

    pub fn set_metallic(&mut self, metallic: f32) {
        self.metallic = sanitize_unit(metallic);
    }

    pub fn set_roughness(&mut self, roughness: f32) {
        self.roughness = sanitize_roughness(roughness);
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.lighting.set_alpha(alpha);
    }
}

/// The GGX shader reuses the proven world-space varying layout.
pub type CookTorranceVaryings = BlinnPhongVaryings;
pub type GgxVaryings = BlinnPhongVaryings;

#[derive(Clone, Copy, Debug, Default)]
pub struct CookTorranceShader;

impl VertexStage<MeshVertex, CookTorranceUniforms> for CookTorranceShader {
    type Varyings = CookTorranceVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &CookTorranceUniforms,
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

impl FragmentStage<CookTorranceVaryings, CookTorranceUniforms> for CookTorranceShader {
    fn run(&self, varyings: &CookTorranceVaryings, uniforms: &CookTorranceUniforms) -> u32 {
        Self::shade(varyings, uniforms)
    }

    fn run_linear(
        &self,
        varyings: &CookTorranceVaryings,
        uniforms: &CookTorranceUniforms,
    ) -> [f32; 4] {
        let lighted = evaluate_ggx_lighting(
            varyings.world_position,
            varyings.normal,
            varyings.light_space_position,
            uniforms,
            Vec3::new(1.0, 1.0, 1.0),
        );
        [uniforms.lighting.alpha, lighted.x, lighted.y, lighted.z]
    }

    fn is_opaque(&self, uniforms: &CookTorranceUniforms) -> bool {
        uniforms.lighting.alpha == 1.0
    }

    fn model_view(&self, uniforms: &CookTorranceUniforms) -> Option<Mat4> {
        Some(uniforms.lighting.view() * uniforms.lighting.model())
    }
}

impl CookTorranceShader {
    /// Evaluates one fragment and encodes the accumulated linear result once.
    /// Without HDR, bright specular values clamp at 1.0 in this encode.
    pub fn shade(varyings: &CookTorranceVaryings, uniforms: &CookTorranceUniforms) -> u32 {
        let lighted = evaluate_ggx_lighting(
            varyings.world_position,
            varyings.normal,
            varyings.light_space_position,
            uniforms,
            Vec3::new(1.0, 1.0, 1.0),
        );
        argb8888_linear(uniforms.lighting.alpha, [lighted.x, lighted.y, lighted.z])
    }
}

/// GGX uniforms with precomputed diffuse and split-sum image lighting.
#[derive(Clone, Debug, PartialEq)]
pub struct IblCookTorranceUniforms<'a> {
    pub lighting: CookTorranceUniforms,
    pub ibl: &'a IblMaps,
}

impl<'a> IblCookTorranceUniforms<'a> {
    pub fn new(
        lighting: BlinnPhongUniforms,
        ibl: &'a IblMaps,
        base_color: Vec3,
        metallic: f32,
        roughness: f32,
    ) -> Self {
        Self {
            lighting: CookTorranceUniforms::new(lighting, base_color, metallic, roughness),
            ibl,
        }
    }

    pub fn new_with_linear_base_color(
        lighting: BlinnPhongUniforms,
        ibl: &'a IblMaps,
        base_color: Vec3,
        metallic: f32,
        roughness: f32,
    ) -> Self {
        Self {
            lighting: CookTorranceUniforms::new_with_linear_base_color(
                lighting, base_color, metallic, roughness,
            ),
            ibl,
        }
    }

    pub const fn base_color(&self) -> Vec3 {
        self.lighting.base_color()
    }

    pub const fn metallic(&self) -> f32 {
        self.lighting.metallic()
    }

    pub const fn roughness(&self) -> f32 {
        self.lighting.roughness()
    }
}

/// GGX shader variant that adds irradiance and split-sum specular IBL.
#[derive(Clone, Copy, Debug, Default)]
pub struct IblCookTorranceShader;

impl<'a> VertexStage<MeshVertex, IblCookTorranceUniforms<'a>> for IblCookTorranceShader {
    type Varyings = CookTorranceVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &IblCookTorranceUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        let prepared = prepare_blinn_phong_vertex(vertex, &uniforms.lighting.lighting);
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

impl<'a> FragmentStage<CookTorranceVaryings, IblCookTorranceUniforms<'a>>
    for IblCookTorranceShader
{
    fn run(&self, varyings: &CookTorranceVaryings, uniforms: &IblCookTorranceUniforms<'a>) -> u32 {
        let linear = self.run_linear(varyings, uniforms);
        argb8888_linear(linear[0], [linear[1], linear[2], linear[3]])
    }

    fn run_linear(
        &self,
        varyings: &CookTorranceVaryings,
        uniforms: &IblCookTorranceUniforms<'a>,
    ) -> [f32; 4] {
        let lighted = evaluate_ibl_ggx_lighting(
            varyings.world_position,
            varyings.normal,
            varyings.light_space_position,
            uniforms,
            Vec3::new(1.0, 1.0, 1.0),
        );
        [
            uniforms.lighting.lighting.alpha,
            lighted.x,
            lighted.y,
            lighted.z,
        ]
    }

    fn is_opaque(&self, uniforms: &IblCookTorranceUniforms<'a>) -> bool {
        uniforms.lighting.lighting.alpha == 1.0
    }

    fn model_view(&self, uniforms: &IblCookTorranceUniforms<'a>) -> Option<Mat4> {
        Some(uniforms.lighting.lighting.view() * uniforms.lighting.lighting.model())
    }
}

/// GGX / Trowbridge-Reitz normal distribution.
///
/// `alpha = roughness^2`, following the Disney convention.
pub fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let n_dot_h = n_dot_h.clamp(0.0, 1.0);
    let alpha = sanitize_roughness(roughness).powi(2);
    let alpha_squared = alpha * alpha;
    // Compute the inner term in this stable order to avoid cancellation at
    // N dot H = 1. The roughness floor keeps it positive. Clamping the inner
    // term, rather than pi times its square, preserves sharp highlights.
    let n_dot_h_squared = n_dot_h * n_dot_h;
    let denominator = ((1.0 - n_dot_h_squared) + n_dot_h_squared * alpha_squared).max(GGX_EPSILON);
    alpha_squared / (PI * denominator * denominator)
}

/// Schlick's approximation to the Smith geometry term.
///
/// For direct lighting, `k = alpha / 2`, where `alpha = roughness^2`.
pub fn smith_schlick_ggx(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let n_dot_l = n_dot_l.clamp(0.0, 1.0);
    let n_dot_v = n_dot_v.clamp(0.0, 1.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return 0.0;
    }
    let alpha = sanitize_roughness(roughness).powi(2);
    let k = alpha * 0.5;
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k).max(GGX_EPSILON);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k).max(GGX_EPSILON);
    g1_l * g1_v
}

/// Schlick Fresnel with metallic interpolation of dielectric F0 and base color.
pub fn schlick_fresnel(h_dot_v: f32, base_color: Vec3, metallic: f32) -> Vec3 {
    let metallic = sanitize_unit(metallic);
    let base_color = sanitize_base_color(base_color);
    let dielectric_f0 = Vec3::new(0.04, 0.04, 0.04);
    let f0 = dielectric_f0 * (1.0 - metallic) + base_color * metallic;
    let one_minus_cosine = 1.0 - h_dot_v.clamp(0.0, 1.0);
    let factor = one_minus_cosine.powi(5);
    f0 + (Vec3::new(1.0, 1.0, 1.0) - f0) * factor
}

/// The production Cook-Torrance BRDF core used by the fragment shader.
///
/// The diffuse term is Lambert scaled by `kd = (1 - F) * (1 - metallic)`.
/// The specular denominator is `4 (N dot L) (N dot V)` with an epsilon guard.
pub fn cook_torrance_brdf(
    normal: Vec3,
    light_direction: Vec3,
    view_direction: Vec3,
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
) -> Vec3 {
    let normal = normal.normalize();
    let light_direction = light_direction.normalize();
    let view_direction = view_direction.normalize();
    let n_dot_l = normal.dot(light_direction).clamp(0.0, 1.0);
    let n_dot_v = normal.dot(view_direction).clamp(0.0, 1.0);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return Vec3::ZERO;
    }
    let half_vector = (light_direction + view_direction).normalize();
    if half_vector.length() == 0.0 {
        return Vec3::ZERO;
    }
    let n_dot_h = normal.dot(half_vector).clamp(0.0, 1.0);
    let h_dot_v = half_vector.dot(view_direction).clamp(0.0, 1.0);
    let fresnel = schlick_fresnel(h_dot_v, base_color, metallic);
    let distribution = ggx_distribution(n_dot_h, roughness);
    let geometry = smith_schlick_ggx(n_dot_l, n_dot_v, roughness);
    let denominator = (4.0 * n_dot_l * n_dot_v).max(GGX_EPSILON);
    let specular = fresnel * (distribution * geometry / denominator);
    let kd = (Vec3::new(1.0, 1.0, 1.0) - fresnel) * (1.0 - sanitize_unit(metallic));
    let diffuse = sanitize_base_color(base_color) * kd * (1.0 / PI);
    diffuse + specular
}

/// Alias with the shorter family name used by demos and callers.
pub fn ggx_brdf(
    normal: Vec3,
    light_direction: Vec3,
    view_direction: Vec3,
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
) -> Vec3 {
    cook_torrance_brdf(
        normal,
        light_direction,
        view_direction,
        base_color,
        metallic,
        roughness,
    )
}

fn evaluate_ggx_lighting(
    world_position: Vec3,
    interpolated_normal: Vec3,
    light_space_position: Vec4,
    uniforms: &CookTorranceUniforms,
    albedo: Vec3,
) -> Vec3 {
    let base_color = sanitize_base_color(uniforms.base_color * albedo);
    let ambient = uniforms.lighting.ambient_color() * base_color;
    ambient
        + evaluate_direct_ggx_lighting(
            world_position,
            interpolated_normal,
            light_space_position,
            &uniforms.lighting,
            base_color,
            uniforms.metallic,
            uniforms.roughness,
        )
}

fn evaluate_direct_ggx_lighting(
    world_position: Vec3,
    interpolated_normal: Vec3,
    light_space_position: Vec4,
    lighting: &BlinnPhongUniforms,
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
) -> Vec3 {
    let normal = interpolated_normal.normalize();
    let view_direction = (lighting.camera_position - world_position).normalize();
    let mut lighted = Vec3::ZERO;
    let directional_visibility = lighting.shadow_visibility(light_space_position, normal);

    for (index, light) in lighting.directional_lights().iter().enumerate() {
        let visibility = if index == 0 {
            directional_visibility
        } else {
            1.0
        };
        let brdf = cook_torrance_brdf(
            normal,
            light.direction,
            view_direction,
            base_color,
            metallic,
            roughness,
        );
        lighted = lighted
            + brdf * normal.dot(light.direction.normalize()).max(0.0) * light.color * visibility;
    }

    for light in lighting.point_lights() {
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
        let brdf = cook_torrance_brdf(
            normal,
            point_direction,
            view_direction,
            base_color,
            metallic,
            roughness,
        );
        lighted = lighted + brdf * normal.dot(point_direction).max(0.0) * light.color * attenuation;
    }

    lighted
}

fn evaluate_ibl_ggx_lighting(
    world_position: Vec3,
    interpolated_normal: Vec3,
    light_space_position: Vec4,
    uniforms: &IblCookTorranceUniforms<'_>,
    albedo: Vec3,
) -> Vec3 {
    let normal = interpolated_normal.normalize();
    let view_direction = (uniforms.lighting.lighting.camera_position - world_position).normalize();
    let base_color = sanitize_base_color(uniforms.lighting.base_color * albedo);
    let n_dot_v = normal.dot(view_direction).clamp(0.0, 1.0);
    let fresnel = schlick_fresnel(n_dot_v, base_color, uniforms.lighting.metallic);
    let kd = (Vec3::new(1.0, 1.0, 1.0) - fresnel) * (1.0 - uniforms.lighting.metallic);
    let irradiance = uniforms.ibl.irradiance.sample(normal);
    let diffuse = irradiance * base_color * kd;

    let reflected = reflection_vector(view_direction, normal);
    let prefiltered = uniforms
        .ibl
        .prefiltered
        .sample(reflected, uniforms.lighting.roughness);
    let environment_brdf = uniforms
        .ibl
        .brdf_lut
        .sample(n_dot_v, uniforms.lighting.roughness);
    let f0 = Vec3::new(0.04, 0.04, 0.04) * (1.0 - uniforms.lighting.metallic)
        + base_color * uniforms.lighting.metallic;
    let specular =
        prefiltered * (f0 * environment_brdf.x + Vec3::new(1.0, 1.0, 1.0) * environment_brdf.y);
    diffuse
        + specular
        + evaluate_direct_ggx_lighting(
            world_position,
            interpolated_normal,
            light_space_position,
            &uniforms.lighting.lighting,
            base_color,
            uniforms.lighting.metallic,
            uniforms.lighting.roughness,
        )
}

#[derive(Clone, Debug, PartialEq)]
pub struct TexturedCookTorranceUniforms<'a> {
    pub lighting: CookTorranceUniforms,
    pub texture: &'a Texture,
    pub filter: TextureFilter,
    pub alpha: f32,
}

impl<'a> TexturedCookTorranceUniforms<'a> {
    pub const fn new(
        lighting: CookTorranceUniforms,
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
pub struct TexturedCookTorranceShader;

impl<'a> VertexStage<MeshVertex, TexturedCookTorranceUniforms<'a>> for TexturedCookTorranceShader {
    type Varyings = TexturedBlinnPhongVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &TexturedCookTorranceUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        let prepared = prepare_blinn_phong_vertex(vertex, &uniforms.lighting.lighting);
        VertexOutput::new(
            prepared.clip_position,
            TexturedBlinnPhongVaryings {
                world_position: prepared.world_position,
                normal: prepared.normal,
                texcoord: vertex.texcoord().unwrap_or(Vec2::ZERO),
                light_space_position: prepared.light_space_position,
            },
        )
    }
}

impl<'a> SampledFragmentStage<TexturedBlinnPhongVaryings, TexturedCookTorranceUniforms<'a>>
    for TexturedCookTorranceShader
{
    fn run_with_sampling(
        &self,
        varyings: &TexturedBlinnPhongVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        uniforms: &TexturedCookTorranceUniforms<'a>,
    ) -> u32 {
        let linear = self.run_linear_with_sampling(varyings, derivatives, uniforms);
        argb8888_linear(linear[0], [linear[1], linear[2], linear[3]])
    }

    fn run_linear_with_sampling(
        &self,
        varyings: &TexturedBlinnPhongVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        uniforms: &TexturedCookTorranceUniforms<'a>,
    ) -> [f32; 4] {
        let pixel = sample_texture(
            uniforms.texture,
            varyings.texcoord,
            uniforms.filter,
            TextureDerivatives {
                ddx: derivatives.ddx,
                ddy: derivatives.ddy,
            },
        );
        let lighted = evaluate_ggx_lighting(
            varyings.world_position,
            varyings.normal,
            varyings.light_space_position,
            &uniforms.lighting,
            Vec3::new(pixel[0], pixel[1], pixel[2]),
        );
        [
            pixel[3] * uniforms.alpha * uniforms.lighting.lighting.alpha,
            lighted.x,
            lighted.y,
            lighted.z,
        ]
    }

    fn is_opaque(&self, uniforms: &TexturedCookTorranceUniforms<'a>) -> bool {
        uniforms.alpha == 1.0
            && uniforms.lighting.lighting.alpha == 1.0
            && texture_has_no_alpha(uniforms.texture)
    }

    fn model_view(&self, uniforms: &TexturedCookTorranceUniforms<'a>) -> Option<Mat4> {
        Some(uniforms.lighting.lighting.view() * uniforms.lighting.lighting.model())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalMappedCookTorranceUniforms<'a> {
    pub lighting: CookTorranceUniforms,
    pub texture: &'a Texture,
    pub normal_map: &'a Texture,
    pub filter: TextureFilter,
}

impl<'a> NormalMappedCookTorranceUniforms<'a> {
    pub fn new(
        lighting: CookTorranceUniforms,
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
pub struct NormalMappedCookTorranceShader;

impl<'a> VertexStage<MeshVertex, NormalMappedCookTorranceUniforms<'a>>
    for NormalMappedCookTorranceShader
{
    type Varyings = NormalMappedBlinnPhongVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &NormalMappedCookTorranceUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        let prepared = prepare_blinn_phong_vertex(vertex, &uniforms.lighting.lighting);
        let tangent = vertex
            .tangent()
            .map_or(Vec4::new(0.0, 0.0, 0.0, 0.0), |tangent| {
                let transformed = uniforms.lighting.lighting.model()
                    * Vec4::new(tangent.x, tangent.y, tangent.z, 0.0);
                Vec4::new(
                    transformed.x,
                    transformed.y,
                    transformed.z,
                    tangent.w * model_handedness(uniforms.lighting.lighting.model()),
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

impl<'a> SampledFragmentStage<NormalMappedBlinnPhongVaryings, NormalMappedCookTorranceUniforms<'a>>
    for NormalMappedCookTorranceShader
{
    fn run_with_sampling(
        &self,
        varyings: &NormalMappedBlinnPhongVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        uniforms: &NormalMappedCookTorranceUniforms<'a>,
    ) -> u32 {
        let linear = self.run_linear_with_sampling(varyings, derivatives, uniforms);
        argb8888_linear(linear[0], [linear[1], linear[2], linear[3]])
    }

    fn run_linear_with_sampling(
        &self,
        varyings: &NormalMappedBlinnPhongVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        uniforms: &NormalMappedCookTorranceUniforms<'a>,
    ) -> [f32; 4] {
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
        let lighted = evaluate_ggx_lighting(
            varyings.world_position,
            normal,
            varyings.light_space_position,
            &uniforms.lighting,
            Vec3::new(albedo_pixel[0], albedo_pixel[1], albedo_pixel[2]),
        );
        [
            albedo_pixel[3] * uniforms.lighting.lighting.alpha,
            lighted.x,
            lighted.y,
            lighted.z,
        ]
    }

    fn is_opaque(&self, uniforms: &NormalMappedCookTorranceUniforms<'a>) -> bool {
        uniforms.lighting.lighting.alpha == 1.0 && texture_has_no_alpha(uniforms.texture)
    }

    fn model_view(&self, uniforms: &NormalMappedCookTorranceUniforms<'a>) -> Option<Mat4> {
        Some(uniforms.lighting.lighting.view() * uniforms.lighting.lighting.model())
    }
}

fn sanitize_base_color(color: Vec3) -> Vec3 {
    Vec3::new(
        sanitize_unit(color.x),
        sanitize_unit(color.y),
        sanitize_unit(color.z),
    )
}

fn sanitize_roughness(roughness: f32) -> f32 {
    sanitize_unit(roughness).max(GGX_MIN_ROUGHNESS)
}

pub type GgxUniforms = CookTorranceUniforms;
pub type GgxShader = CookTorranceShader;
pub type TexturedGgxUniforms<'a> = TexturedCookTorranceUniforms<'a>;
pub type TexturedGgxShader = TexturedCookTorranceShader;
pub type NormalMappedGgxUniforms<'a> = NormalMappedCookTorranceUniforms<'a>;
pub type NormalMappedGgxShader = NormalMappedCookTorranceShader;
pub type CookTorranceGGXUniforms = CookTorranceUniforms;
pub type CookTorranceGGXShader = CookTorranceShader;
pub type IblGgxUniforms<'a> = IblCookTorranceUniforms<'a>;
pub type IblGgxShader = IblCookTorranceShader;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }

    #[test]
    fn ggx_distribution_roughness_half_at_normal_incidence_is_hand_computed() {
        // alpha = roughness^2 = 0.5^2 = 0.25; alpha^2 = 0.0625.
        // At N dot H = 1, D = 0.0625 / (pi * 0.0625^2) = 5.092958.
        assert_close(ggx_distribution(1.0, 0.5), 5.092958);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn ggx_distribution_keeps_valid_low_roughness_values() {
        // The 0.045 floor gives D = 1 / (pi * roughness^4) = 77624.71875.
        assert_close(ggx_distribution(1.0, 0.045), 77624.71875);
        // At roughness 0.1, D = 1 / (pi * 0.1^4) = 3183.0986.
        assert!((ggx_distribution(1.0, 0.1) - 3183.0986).abs() < 0.001);
    }

    #[test]
    fn fresnel_f0_lerps_from_dielectric_to_metal() {
        let base_color = Vec3::new(0.8, 0.2, 0.1);
        assert_eq!(
            schlick_fresnel(1.0, base_color, 0.0),
            Vec3::new(0.04, 0.04, 0.04)
        );
        let half = schlick_fresnel(1.0, base_color, 0.5);
        assert_close(half.x, 0.42);
        assert_close(half.y, 0.12);
        assert_close(half.z, 0.07);
        assert_eq!(schlick_fresnel(1.0, base_color, 1.0), base_color);
    }

    #[test]
    fn smith_geometry_is_hand_computed_for_known_configuration() {
        // roughness = 0.5 gives alpha = 0.25 and k = alpha / 2 = 0.125.
        // G1(0.5) = 0.5 / (0.5 * 0.875 + 0.125) = 0.8888889.
        // G1(0.8) = 0.8 / (0.8 * 0.875 + 0.125) = 0.969697.
        assert_close(smith_schlick_ggx(0.5, 0.8, 0.5), 0.8619529);
    }

    #[test]
    fn full_brdf_matches_hand_computed_normal_incidence() {
        // N = L = V = (0, 0, 1), baseColor = 0.8, metallic = 0, roughness = 0.5.
        // D = 0.0625 / (pi * 0.0625^2), G = 1, F = 0.04, and the specular
        // term is 0.04 * D / (4 * 1 * 1) = 0.05092958.
        // kd = (1 - 0.04) * (1 - 0) = 0.96, so diffuse is 0.8 * 0.96 / pi.
        // The complete BRDF is 0.24446199 + 0.05092958 = 0.29539157.
        let result = cook_torrance_brdf(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.8, 0.8, 0.8),
            0.0,
            0.5,
        );
        assert_close(result.x, 0.29539157);
        assert_close(result.y, 0.29539157);
        assert_close(result.z, 0.29539157);
    }

    #[test]
    fn kd_energy_conservation_is_bounded() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let base_color = Vec3::new(0.8, 0.8, 0.8);
        let dielectric = cook_torrance_brdf(normal, normal, normal, base_color, 0.0, 0.5);
        let metal = cook_torrance_brdf(normal, normal, normal, base_color, 1.0, 0.5);
        // Production BRDF arithmetic: D = 5.092958, G = 1, F = 0.04.
        // Dielectric = (0.96 * 0.8) / pi + (0.04 * D / 4) = 0.2953916.
        assert_close(dielectric.x, 0.2953916);
        // Metallic kd is zero. Metal = 0.8 * D / 4 = 1.0185916.
        // If production kd drops (1 - metallic), this assertion fails.
        assert_close(metal.x, 1.0185916);
        assert!(dielectric.x.is_finite() && metal.x.is_finite());
    }

    #[test]
    fn material_boundaries_rebuild_sanitized_values_immediately() {
        let lighting = BlinnPhongUniforms::new_with_linear_colors(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            Vec3::ZERO,
            DirectionalLight::default(),
            PointLight::default(),
        );
        let mut uniforms = CookTorranceUniforms::new_with_linear_base_color(
            lighting,
            Vec3::new(-1.0, 0.5, 2.0),
            f32::NAN,
            0.0,
        );
        assert_eq!(uniforms.base_color(), Vec3::new(0.0, 0.5, 1.0));
        assert_eq!(uniforms.metallic(), 0.0);
        assert_eq!(uniforms.roughness(), GGX_MIN_ROUGHNESS);
        uniforms.set_metallic(2.0);
        uniforms.set_roughness(-1.0);
        assert_eq!(uniforms.metallic(), 1.0);
        assert_eq!(uniforms.roughness(), GGX_MIN_ROUGHNESS);
    }
}
