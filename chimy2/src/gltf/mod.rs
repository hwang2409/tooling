//! glTF 2.0 loading, animation sampling, and deterministic CPU deformation.
//!
//! Supported features are documented by SUPPORTED_SUBSET. Unsupported
//! features return an error. The loader uses no serialization crate because
//! glTF is parsed by the local JSON parser.

use crate::fb::{Framebuffer, argb8888};
use crate::image::{ColorSpace, Texture};
use crate::json::{self, Value};
use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::pipeline::{
    FragmentStage, Pipeline, SampledFragmentStage, SamplingVaryings, Varyings, VertexOutput,
    VertexStage,
};
use crate::shaders::{
    BlinnPhongUniforms, CookTorranceShader, CookTorranceUniforms, CookTorranceVaryings,
    DirectionalLight, NormalMappedBlinnPhongVaryings, NormalMappedCookTorranceShader,
    NormalMappedCookTorranceUniforms, PointLight, TextureFilter, TexturedBlinnPhongVaryings,
    TexturedCookTorranceShader, TexturedCookTorranceUniforms,
};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const SUPPORTED_SUBSET: &str = "glTF JSON (.gltf), external .bin buffers and base64 binary data URIs; TRIANGLES primitives with POSITION, optional NORMAL, TEXCOORD_0, JOINTS_0, WEIGHTS_0, morph POSITION/NORMAL targets, and indices; semantic-typed FLOAT vertex and animation accessors; SCALAR/VEC2/VEC3/VEC4/MAT4 accessors with component types 5120-5126 and byteStride; node TRS or matrix hierarchy; PBR baseColorFactor/baseColorTexture and normalTexture; OPAQUE and BLEND alpha modes; skins with inverseBindMatrices; translation, rotation, scale, and weights animations with LINEAR and STEP. Unsupported primitive modes, sparse accessors, cubic animation, MASK alpha, GLB, and unsupported image formats return errors. Morph targets are not combined with LodMeshes.";

const WEIGHT_TOLERANCE: f32 = 1.0e-5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GltfError {
    pub offset: Option<usize>,
    pub message: String,
}

impl GltfError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            offset: None,
            message: message.into(),
        }
    }

    fn at(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset: Some(offset),
            message: message.into(),
        }
    }
}

impl Display for GltfError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(offset) = self.offset {
            write!(formatter, "byte {offset}: {}", self.message)
        } else {
            write!(formatter, "{}", self.message)
        }
    }
}

impl std::error::Error for GltfError {}

impl From<json::Error> for GltfError {
    fn from(error: json::Error) -> Self {
        Self::at(error.offset, error.message)
    }
}

#[derive(Clone, Debug)]
pub struct GltfAsset {
    pub meshes: Vec<GltfMesh>,
    pub nodes: Vec<GltfNode>,
    pub scenes: Vec<GltfScene>,
    pub skins: Vec<GltfSkin>,
    pub animations: Vec<GltfAnimation>,
    pub materials: Vec<GltfMaterial>,
    pub default_scene: usize,
}

#[derive(Clone, Debug)]
pub struct GltfMesh {
    pub name: String,
    pub primitives: Vec<GltfPrimitive>,
    /// Default morph weights from the glTF mesh, one per primitive target.
    pub weights: Vec<f32>,
}

/// Per-vertex glTF morph deltas.
///
/// This follows glTF 2.0 section 3.7.3, "Morph Targets":
/// <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_morph_targets>.
/// POSITION and NORMAL target attributes contain deltas, not replacement
/// values. Sparse accessors are outside this renderer subset and are rejected
/// by the accessor parser.
#[derive(Clone, Debug, PartialEq)]
pub struct MorphTarget {
    pub position_deltas: Vec<Vec3>,
    pub normal_deltas: Option<Vec<Vec3>>,
}

#[derive(Clone, Debug)]
pub struct GltfPrimitive {
    pub mesh: Mesh,
    pub material: Option<usize>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub morph_targets: Vec<MorphTarget>,
    morph_weights: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct GltfNode {
    pub name: String,
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub matrix: Option<Mat4>,
    pub translation: Vec3,
    pub rotation: [f32; 4],
    pub scale: Vec3,
}

#[derive(Clone, Debug)]
pub struct GltfScene {
    pub name: String,
    pub nodes: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct GltfSkin {
    pub name: String,
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
    pub skeleton: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct GltfAnimation {
    pub name: String,
    pub samplers: Vec<GltfAnimationSampler>,
    pub channels: Vec<GltfAnimationChannel>,
    pub duration: f32,
}

#[derive(Clone, Debug)]
pub struct GltfAnimationSampler {
    pub input: Vec<f32>,
    pub output: Vec<[f32; 4]>,
    pub output_components: usize,
    pub interpolation: Interpolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationPath {
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(Clone, Debug)]
pub struct GltfAnimationChannel {
    pub sampler: usize,
    pub node: usize,
    pub path: AnimationPath,
}

#[derive(Clone, Debug)]
pub struct GltfMaterial {
    pub name: String,
    pub base_color_factor: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub albedo_texture: Option<Texture>,
    pub normal_map_texture: Option<Texture>,
    pub alpha_mode: GltfAlphaMode,
    pub alpha_cutoff: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GltfAlphaMode {
    Opaque,
    Blend,
    Mask,
}

#[derive(Clone, Copy)]
struct GltfMaterialParameters {
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
    alpha: f32,
}

const DEFAULT_GLTF_MATERIAL: GltfMaterialParameters = GltfMaterialParameters {
    base_color: Vec3::new(1.0, 1.0, 1.0),
    metallic: 1.0,
    roughness: 1.0,
    alpha: 1.0,
};

impl GltfMaterial {
    /// Returns glTF PBR factors without converting them to Blinn-Phong.
    /// Base color and scalar factors remain linear at the GGX boundary.
    fn ggx_parameters(&self) -> GltfMaterialParameters {
        GltfMaterialParameters {
            base_color: Vec3::new(
                self.base_color_factor.x,
                self.base_color_factor.y,
                self.base_color_factor.z,
            ),
            metallic: self.metallic_factor,
            roughness: self.roughness_factor,
            alpha: match self.alpha_mode {
                GltfAlphaMode::Opaque => 1.0,
                GltfAlphaMode::Blend | GltfAlphaMode::Mask => {
                    self.base_color_factor.w.clamp(0.0, 1.0)
                }
            },
        }
    }

    fn render_parameters(&self) -> GltfMaterialParameters {
        self.ggx_parameters()
    }
}

enum GltfUniformKind<'a> {
    Plain(CookTorranceUniforms),
    Textured(TexturedCookTorranceUniforms<'a>),
    NormalMapped(NormalMappedCookTorranceUniforms<'a>),
}

struct GltfUniforms<'a> {
    kind: GltfUniformKind<'a>,
    alpha_mode: GltfAlphaMode,
}

#[derive(Clone)]
enum GltfVaryings {
    Plain(CookTorranceVaryings),
    Textured(TexturedBlinnPhongVaryings),
    NormalMapped(NormalMappedBlinnPhongVaryings),
}

#[derive(Clone, Copy, Debug, Default)]
struct GltfShader;

impl Varyings for GltfVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        match (a, b, c) {
            (Self::Plain(a), Self::Plain(b), Self::Plain(c)) => {
                Self::Plain(CookTorranceVaryings::lerp3(a, b, c, weights))
            }
            (Self::Textured(a), Self::Textured(b), Self::Textured(c)) => {
                Self::Textured(TexturedBlinnPhongVaryings::lerp3(a, b, c, weights))
            }
            (Self::NormalMapped(a), Self::NormalMapped(b), Self::NormalMapped(c)) => {
                Self::NormalMapped(NormalMappedBlinnPhongVaryings::lerp3(a, b, c, weights))
            }
            _ => panic!("glTF draws must use matching varying types"),
        }
    }
}

impl SamplingVaryings for GltfVaryings {
    fn texture_coordinates(&self) -> Vec2 {
        match self {
            Self::Plain(_) => Vec2::ZERO,
            Self::Textured(varyings) => varyings.texture_coordinates(),
            Self::NormalMapped(varyings) => varyings.texture_coordinates(),
        }
    }
}

impl<'a> VertexStage<MeshVertex, GltfUniforms<'a>> for GltfShader {
    type Varyings = GltfVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &GltfUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        match &uniforms.kind {
            GltfUniformKind::Plain(uniforms) => {
                let output = VertexStage::run(&CookTorranceShader, vertex, uniforms);
                VertexOutput::new(output.clip_position, GltfVaryings::Plain(output.varyings))
            }
            GltfUniformKind::Textured(uniforms) => {
                let output = TexturedCookTorranceShader.run(vertex, uniforms);
                VertexOutput::new(
                    output.clip_position,
                    GltfVaryings::Textured(output.varyings),
                )
            }
            GltfUniformKind::NormalMapped(uniforms) => {
                let output = NormalMappedCookTorranceShader.run(vertex, uniforms);
                VertexOutput::new(
                    output.clip_position,
                    GltfVaryings::NormalMapped(output.varyings),
                )
            }
        }
    }
}

impl<'a> SampledFragmentStage<GltfVaryings, GltfUniforms<'a>> for GltfShader {
    fn run_with_sampling(
        &self,
        varyings: &GltfVaryings,
        derivatives: &crate::pipeline::SampleDerivatives,
        gltf_uniforms: &GltfUniforms<'a>,
    ) -> u32 {
        match (&gltf_uniforms.kind, varyings) {
            (GltfUniformKind::Plain(uniforms), GltfVaryings::Plain(varyings)) => {
                FragmentStage::run(&CookTorranceShader, varyings, uniforms)
            }
            (GltfUniformKind::Textured(uniforms), GltfVaryings::Textured(varyings)) => {
                let pixel =
                    TexturedCookTorranceShader.run_with_sampling(varyings, derivatives, uniforms);
                if gltf_uniforms.alpha_mode == GltfAlphaMode::Opaque {
                    force_opaque_alpha(pixel)
                } else {
                    pixel
                }
            }
            (GltfUniformKind::NormalMapped(uniforms), GltfVaryings::NormalMapped(varyings)) => {
                let pixel = NormalMappedCookTorranceShader.run_with_sampling(
                    varyings,
                    derivatives,
                    uniforms,
                );
                if gltf_uniforms.alpha_mode == GltfAlphaMode::Opaque {
                    force_opaque_alpha(pixel)
                } else {
                    pixel
                }
            }
            _ => panic!("glTF draws must use matching varying types"),
        }
    }

    fn is_opaque(&self, uniforms: &GltfUniforms<'a>) -> bool {
        if uniforms.alpha_mode == GltfAlphaMode::Opaque {
            return true;
        }
        if uniforms.alpha_mode == GltfAlphaMode::Blend {
            return false;
        }
        match &uniforms.kind {
            GltfUniformKind::Plain(uniforms) => CookTorranceShader.is_opaque(uniforms),
            GltfUniformKind::Textured(uniforms) => TexturedCookTorranceShader.is_opaque(uniforms),
            GltfUniformKind::NormalMapped(uniforms) => {
                NormalMappedCookTorranceShader.is_opaque(uniforms)
            }
        }
    }

    fn model_view(&self, uniforms: &GltfUniforms<'a>) -> Option<Mat4> {
        match &uniforms.kind {
            GltfUniformKind::Plain(uniforms) => CookTorranceShader.model_view(uniforms),
            GltfUniformKind::Textured(uniforms) => TexturedCookTorranceShader.model_view(uniforms),
            GltfUniformKind::NormalMapped(uniforms) => {
                NormalMappedCookTorranceShader.model_view(uniforms)
            }
        }
    }

    fn culling_transform(&self, uniforms: &GltfUniforms<'a>) -> Option<Mat4> {
        match &uniforms.kind {
            GltfUniformKind::Plain(uniforms) => CookTorranceShader.culling_transform(uniforms),
            GltfUniformKind::Textured(uniforms) => {
                TexturedCookTorranceShader.culling_transform(uniforms)
            }
            GltfUniformKind::NormalMapped(uniforms) => {
                NormalMappedCookTorranceShader.culling_transform(uniforms)
            }
        }
    }
}

fn force_opaque_alpha(pixel: u32) -> u32 {
    let [_, red, green, blue] = pixel.to_be_bytes();
    argb8888(255, red, green, blue)
}

/// Submits glTF draws through the shared material-aware renderer.
pub fn submit_gltf_draws(
    framebuffer: &mut Framebuffer,
    asset: &GltfAsset,
    draws: &[GltfDraw],
    view: Mat4,
    projection: Mat4,
    camera_position: Vec3,
) -> Result<(), GltfError> {
    let directional = DirectionalLight::new(
        Vec3::new(-0.4, -0.8, -0.6).normalize(),
        Vec3::new(1.0, 0.95, 0.9),
    );
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let mut uniforms = Vec::with_capacity(draws.len());
    for draw in draws {
        let material = draw
            .material
            .map(|index| {
                asset
                    .materials
                    .get(index)
                    .ok_or_else(|| GltfError::new("draw material index is out of range"))
            })
            .transpose()?;
        let parameters = material
            .map(GltfMaterial::render_parameters)
            .unwrap_or(DEFAULT_GLTF_MATERIAL);
        let albedo = material.and_then(|material| material.albedo_texture.as_ref());
        let normal_map = material.and_then(|material| material.normal_map_texture.as_ref());
        let lighting = make_gltf_lighting(
            draw.model,
            view,
            projection,
            parameters,
            camera_position,
            directional,
            point,
        );
        let kind = if let (Some(albedo), Some(normal_map)) = (albedo, normal_map) {
            GltfUniformKind::NormalMapped(
                NormalMappedCookTorranceUniforms::new(
                    CookTorranceUniforms::new_with_linear_base_color(
                        lighting,
                        parameters.base_color,
                        parameters.metallic,
                        parameters.roughness,
                    ),
                    albedo,
                    normal_map,
                    TextureFilter::Bilinear,
                )
                .map_err(GltfError::new)?,
            )
        } else if let Some(albedo) = albedo {
            GltfUniformKind::Textured(TexturedCookTorranceUniforms::new(
                CookTorranceUniforms::new_with_linear_base_color(
                    lighting,
                    parameters.base_color,
                    parameters.metallic,
                    parameters.roughness,
                ),
                albedo,
                TextureFilter::Bilinear,
            ))
        } else {
            GltfUniformKind::Plain(CookTorranceUniforms::new_with_linear_base_color(
                lighting,
                parameters.base_color,
                parameters.metallic,
                parameters.roughness,
            ))
        };
        uniforms.push(GltfUniforms {
            kind,
            alpha_mode: material
                .map(|material| material.alpha_mode)
                .unwrap_or(GltfAlphaMode::Opaque),
        });
    }
    let mut pipeline = Pipeline::new(GltfShader, GltfShader);
    pipeline.render(framebuffer, |frame, target| {
        for (draw, uniforms) in draws.iter().zip(&uniforms) {
            frame.draw_mesh_with_sampling(target, &draw.mesh, uniforms);
        }
    });
    Ok(())
}

fn make_gltf_lighting(
    model: Mat4,
    view: Mat4,
    projection: Mat4,
    parameters: GltfMaterialParameters,
    camera_position: Vec3,
    directional: DirectionalLight,
    point: PointLight,
) -> BlinnPhongUniforms {
    let mut lighting = BlinnPhongUniforms::new_with_linear_colors(
        model,
        view,
        projection,
        Vec3::new(0.1, 0.1, 0.1),
        Vec3::ZERO,
        Vec3::ZERO,
        0.0,
        camera_position,
        directional,
        point,
    );
    lighting.set_alpha(parameters.alpha);
    lighting
}

#[derive(Clone, Debug)]
pub struct GltfDraw {
    pub mesh: Mesh,
    pub model: Mat4,
    pub material: Option<usize>,
}

// focused loader modules. these files are included in one private namespace so
// their internal helpers keep the original visibility and behavior.
include!("document.rs");
include!("hierarchy.rs");
include!("animation.rs");
include!("buffers.rs");
include!("model.rs");
include!("skin.rs");
include!("animation_parse.rs");
include!("materials.rs");
include!("json.rs");

#[cfg(test)]
include!("tests.rs");
