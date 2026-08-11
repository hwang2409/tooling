//! Shared submission path for meshes with Wavefront material groups.

use crate::fb::Framebuffer;
use crate::image::Texture;
use crate::material::Material;
use crate::math::{Mat4, Vec2, Vec3};
use crate::mesh::{Mesh, MeshVertex};
use crate::pipeline::{
    FragmentStage, Pipeline, SampleDerivatives, SampledFragmentStage, SamplingVaryings, Varyings,
    VertexOutput, VertexStage,
};
use crate::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, BlinnPhongVaryings, NormalMappedBlinnPhongShader,
    NormalMappedBlinnPhongUniforms, NormalMappedBlinnPhongVaryings, TextureFilter,
    TexturedBlinnPhongShader, TexturedBlinnPhongUniforms, TexturedBlinnPhongVaryings,
};

/// A mesh and its parsed material for one draw in a material-group frame.
pub struct MaterialGroup<'a> {
    mesh: &'a Mesh,
    material: &'a Material,
    albedo_override: Option<&'a Texture>,
}

impl<'a> MaterialGroup<'a> {
    pub fn new(mesh: &'a Mesh, material: &'a Material) -> Self {
        Self {
            mesh,
            material,
            albedo_override: None,
        }
    }

    /// Uses a replacement albedo while keeping the material's other maps.
    pub fn with_albedo_override(mut self, texture: &'a Texture) -> Self {
        self.albedo_override = Some(texture);
        self
    }
}

enum MaterialUniforms<'a> {
    Plain(BlinnPhongUniforms),
    Textured(TexturedBlinnPhongUniforms<'a>),
    NormalMapped(NormalMappedBlinnPhongUniforms<'a>),
}

#[derive(Clone)]
enum MaterialVaryings {
    Plain(BlinnPhongVaryings),
    Textured(TexturedBlinnPhongVaryings),
    NormalMapped(NormalMappedBlinnPhongVaryings),
}

#[derive(Clone, Copy, Debug, Default)]
struct MaterialShader;

impl Varyings for MaterialVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        match (a, b, c) {
            (Self::Plain(a), Self::Plain(b), Self::Plain(c)) => {
                Self::Plain(BlinnPhongVaryings::lerp3(a, b, c, weights))
            }
            (Self::Textured(a), Self::Textured(b), Self::Textured(c)) => {
                Self::Textured(TexturedBlinnPhongVaryings::lerp3(a, b, c, weights))
            }
            (Self::NormalMapped(a), Self::NormalMapped(b), Self::NormalMapped(c)) => {
                Self::NormalMapped(NormalMappedBlinnPhongVaryings::lerp3(a, b, c, weights))
            }
            _ => panic!("material groups must use matching varying types"),
        }
    }
}

impl SamplingVaryings for MaterialVaryings {
    fn texture_coordinates(&self) -> Vec2 {
        match self {
            Self::Plain(_) => Vec2::ZERO,
            Self::Textured(varyings) => varyings.texture_coordinates(),
            Self::NormalMapped(varyings) => varyings.texture_coordinates(),
        }
    }
}

impl<'a> VertexStage<MeshVertex, MaterialUniforms<'a>> for MaterialShader {
    type Varyings = MaterialVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &MaterialUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        match uniforms {
            MaterialUniforms::Plain(uniforms) => {
                let output = VertexStage::run(&BlinnPhongShader, vertex, uniforms);
                VertexOutput::new(
                    output.clip_position,
                    MaterialVaryings::Plain(output.varyings),
                )
            }
            MaterialUniforms::Textured(uniforms) => {
                let output = TexturedBlinnPhongShader.run(vertex, uniforms);
                VertexOutput::new(
                    output.clip_position,
                    MaterialVaryings::Textured(output.varyings),
                )
            }
            MaterialUniforms::NormalMapped(uniforms) => {
                let output = NormalMappedBlinnPhongShader.run(vertex, uniforms);
                VertexOutput::new(
                    output.clip_position,
                    MaterialVaryings::NormalMapped(output.varyings),
                )
            }
        }
    }
}

impl<'a> SampledFragmentStage<MaterialVaryings, MaterialUniforms<'a>> for MaterialShader {
    fn run_with_sampling(
        &self,
        varyings: &MaterialVaryings,
        derivatives: &SampleDerivatives,
        uniforms: &MaterialUniforms<'a>,
    ) -> u32 {
        match (varyings, uniforms) {
            (MaterialVaryings::Plain(varyings), MaterialUniforms::Plain(uniforms)) => {
                FragmentStage::run(&BlinnPhongShader, varyings, uniforms)
            }
            (MaterialVaryings::Textured(varyings), MaterialUniforms::Textured(uniforms)) => {
                TexturedBlinnPhongShader.run_with_sampling(varyings, derivatives, uniforms)
            }
            (
                MaterialVaryings::NormalMapped(varyings),
                MaterialUniforms::NormalMapped(uniforms),
            ) => NormalMappedBlinnPhongShader.run_with_sampling(varyings, derivatives, uniforms),
            _ => panic!("material groups must use matching varying types"),
        }
    }

    fn is_opaque(&self, uniforms: &MaterialUniforms<'a>) -> bool {
        match uniforms {
            MaterialUniforms::Plain(uniforms) => BlinnPhongShader.is_opaque(uniforms),
            MaterialUniforms::Textured(uniforms) => TexturedBlinnPhongShader.is_opaque(uniforms),
            MaterialUniforms::NormalMapped(uniforms) => {
                NormalMappedBlinnPhongShader.is_opaque(uniforms)
            }
        }
    }

    fn model_view(&self, uniforms: &MaterialUniforms<'a>) -> Option<Mat4> {
        match uniforms {
            MaterialUniforms::Plain(uniforms) => BlinnPhongShader.model_view(uniforms),
            MaterialUniforms::Textured(uniforms) => TexturedBlinnPhongShader.model_view(uniforms),
            MaterialUniforms::NormalMapped(uniforms) => {
                NormalMappedBlinnPhongShader.model_view(uniforms)
            }
        }
    }
}

/// Queues all material groups, then flushes their opaque and transparent draws once.
pub fn submit_material_groups(
    framebuffer: &mut Framebuffer,
    groups: &[MaterialGroup<'_>],
    base_lighting: &BlinnPhongUniforms,
) {
    let mut uniforms = Vec::with_capacity(groups.len());
    for group in groups {
        let mut lighting = base_lighting.clone();
        group.material.apply_to(&mut lighting);
        let albedo = group
            .albedo_override
            .or_else(|| group.material.albedo_texture());
        if let (Some(texture), Some(normal_map)) = (albedo, group.material.normal_map_texture()) {
            uniforms.push(MaterialUniforms::NormalMapped(
                NormalMappedBlinnPhongUniforms::new(
                    lighting,
                    texture,
                    normal_map,
                    TextureFilter::Bilinear,
                )
                .expect("MTL loader assigns linear color space to normal maps"),
            ));
        } else if let Some(texture) = albedo {
            uniforms.push(MaterialUniforms::Textured(TexturedBlinnPhongUniforms::new(
                lighting,
                texture,
                TextureFilter::Bilinear,
            )));
        } else {
            uniforms.push(MaterialUniforms::Plain(lighting));
        }
    }

    let mut pipeline = Pipeline::new(MaterialShader, MaterialShader);
    pipeline.render(framebuffer, |frame, target| {
        for (group, uniforms) in groups.iter().zip(uniforms.iter()) {
            frame.draw_mesh_with_sampling(target, group.mesh, uniforms);
        }
    });
}
