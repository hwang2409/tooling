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
#[cfg(test)]
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[cfg(test)]
static MATERIAL_VERTEX_RUNS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static MATERIAL_TEST_LOCK: Mutex<()> = Mutex::new(());

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
        #[cfg(test)]
        MATERIAL_VERTEX_RUNS.fetch_add(1, Ordering::Relaxed);
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

    fn culling_transform(&self, uniforms: &MaterialUniforms<'a>) -> Option<Mat4> {
        match uniforms {
            MaterialUniforms::Plain(uniforms) => BlinnPhongShader.culling_transform(uniforms),
            MaterialUniforms::Textured(uniforms) => {
                TexturedBlinnPhongShader.culling_transform(uniforms)
            }
            MaterialUniforms::NormalMapped(uniforms) => {
                NormalMappedBlinnPhongShader.culling_transform(uniforms)
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
    submit_material_groups_with_culling(framebuffer, groups, base_lighting, true);
}

/// Queues material groups with an explicit mesh-culling mode.
pub fn submit_material_groups_with_culling(
    framebuffer: &mut Framebuffer,
    groups: &[MaterialGroup<'_>],
    base_lighting: &BlinnPhongUniforms,
    culling_enabled: bool,
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
    pipeline.set_culling_enabled(culling_enabled);
    pipeline.render(framebuffer, |frame, target| {
        for (group, uniforms) in groups.iter().zip(uniforms.iter()) {
            frame.draw_mesh_with_sampling(target, group.mesh, uniforms);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MaterialLibrary;
    use crate::shaders::DirectionalLight;

    fn mesh() -> Mesh {
        Mesh::new(
            vec![
                MeshVertex::new(Vec3::new(-0.4, -0.4, 0.0), None, None),
                MeshVertex::new(Vec3::new(0.4, -0.4, 0.0), None, None),
                MeshVertex::new(Vec3::new(-0.4, 0.4, 0.0), None, None),
            ],
            vec![[0, 1, 2]],
        )
    }

    fn material() -> Material {
        MaterialLibrary::parse("newmtl test\nKd 1 1 1\n")
            .unwrap()
            .get("test")
            .unwrap()
            .clone()
    }

    fn lighting(model: Mat4) -> BlinnPhongUniforms {
        BlinnPhongUniforms::new(
            model,
            Mat4::IDENTITY,
            Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 10.0),
            Vec3::new(0.02, 0.02, 0.02),
            Vec3::new(0.8, 0.8, 0.8),
            Vec3::new(0.2, 0.2, 0.2),
            16.0,
            Vec3::new(0.0, 0.0, 5.0),
            DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
            crate::shaders::PointLight::default(),
        )
    }

    #[test]
    fn offscreen_mtl_group_is_rejected_before_vertex_stage() {
        let _lock = MATERIAL_TEST_LOCK.lock().unwrap();
        let mesh = mesh();
        let material = material();
        let group = MaterialGroup::new(&mesh, &material);
        let groups = [group];
        let mut framebuffer = Framebuffer::new(32, 32);
        MATERIAL_VERTEX_RUNS.store(0, Ordering::Relaxed);
        submit_material_groups_with_culling(
            &mut framebuffer,
            &groups,
            &lighting(Mat4::translate(Vec3::new(3.0, 0.0, -3.0))),
            true,
        );
        assert_eq!(MATERIAL_VERTEX_RUNS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn visible_mtl_group_is_byte_identical_with_culling_on_and_off() {
        let _lock = MATERIAL_TEST_LOCK.lock().unwrap();
        let mesh = mesh();
        let material = material();
        let group = MaterialGroup::new(&mesh, &material);
        let groups = [group];
        let base = lighting(Mat4::translate(Vec3::new(0.0, 0.0, -3.0)));
        let render = |culling_enabled| {
            let mut framebuffer = Framebuffer::new(32, 32);
            submit_material_groups_with_culling(&mut framebuffer, &groups, &base, culling_enabled);
            framebuffer
        };
        let culled = render(true);
        let unculled = render(false);
        assert_eq!(culled.color, unculled.color);
        assert_eq!(culled.depth, unculled.depth);
    }
}
