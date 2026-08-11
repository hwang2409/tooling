//! Headless-friendly demo for OBJ material groups and MTL texture maps.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::Vec2;
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::{FragmentStage, Pipeline, SampledFragmentStage, SamplingVaryings, Varyings};
use chimy2::pipeline::{SampleDerivatives, VertexOutput, VertexStage};
use chimy2::present::InputState;
use chimy2::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, BlinnPhongVaryings, DirectionalLight,
    NormalMappedBlinnPhongShader, NormalMappedBlinnPhongUniforms, NormalMappedBlinnPhongVaryings,
    PointLight, TextureFilter, TexturedBlinnPhongShader, TexturedBlinnPhongUniforms,
    TexturedBlinnPhongVaryings,
};

enum MtlUniforms<'a> {
    Plain(BlinnPhongUniforms),
    Textured(TexturedBlinnPhongUniforms<'a>),
    NormalMapped(NormalMappedBlinnPhongUniforms<'a>),
}

#[derive(Clone)]
enum MtlVaryings {
    Plain(BlinnPhongVaryings),
    Textured(TexturedBlinnPhongVaryings),
    NormalMapped(NormalMappedBlinnPhongVaryings),
}

#[derive(Clone, Copy, Debug, Default)]
struct MtlShader;

impl Varyings for MtlVaryings {
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

impl SamplingVaryings for MtlVaryings {
    fn texture_coordinates(&self) -> Vec2 {
        match self {
            Self::Plain(_) => Vec2::ZERO,
            Self::Textured(varyings) => varyings.texture_coordinates(),
            Self::NormalMapped(varyings) => varyings.texture_coordinates(),
        }
    }
}

impl<'a> VertexStage<chimy2::mesh::MeshVertex, MtlUniforms<'a>> for MtlShader {
    type Varyings = MtlVaryings;

    fn run(
        &self,
        vertex: &chimy2::mesh::MeshVertex,
        uniforms: &MtlUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        match uniforms {
            MtlUniforms::Plain(uniforms) => {
                let output = VertexStage::run(&BlinnPhongShader, vertex, uniforms);
                VertexOutput::new(output.clip_position, MtlVaryings::Plain(output.varyings))
            }
            MtlUniforms::Textured(uniforms) => {
                let output = TexturedBlinnPhongShader.run(vertex, uniforms);
                VertexOutput::new(output.clip_position, MtlVaryings::Textured(output.varyings))
            }
            MtlUniforms::NormalMapped(uniforms) => {
                let output = NormalMappedBlinnPhongShader.run(vertex, uniforms);
                VertexOutput::new(
                    output.clip_position,
                    MtlVaryings::NormalMapped(output.varyings),
                )
            }
        }
    }
}

impl<'a> SampledFragmentStage<MtlVaryings, MtlUniforms<'a>> for MtlShader {
    fn run_with_sampling(
        &self,
        varyings: &MtlVaryings,
        derivatives: &SampleDerivatives,
        uniforms: &MtlUniforms<'a>,
    ) -> u32 {
        match (varyings, uniforms) {
            (MtlVaryings::Plain(varyings), MtlUniforms::Plain(uniforms)) => {
                FragmentStage::run(&BlinnPhongShader, varyings, uniforms)
            }
            (MtlVaryings::Textured(varyings), MtlUniforms::Textured(uniforms)) => {
                TexturedBlinnPhongShader.run_with_sampling(varyings, derivatives, uniforms)
            }
            (MtlVaryings::NormalMapped(varyings), MtlUniforms::NormalMapped(uniforms)) => {
                NormalMappedBlinnPhongShader.run_with_sampling(varyings, derivatives, uniforms)
            }
            _ => panic!("material groups must use matching varying types"),
        }
    }

    fn is_opaque(&self, uniforms: &MtlUniforms<'a>) -> bool {
        match uniforms {
            MtlUniforms::Plain(uniforms) => BlinnPhongShader.is_opaque(uniforms),
            MtlUniforms::Textured(uniforms) => TexturedBlinnPhongShader.is_opaque(uniforms),
            MtlUniforms::NormalMapped(uniforms) => NormalMappedBlinnPhongShader.is_opaque(uniforms),
        }
    }

    fn model_view(&self, uniforms: &MtlUniforms<'a>) -> Option<Mat4> {
        match uniforms {
            MtlUniforms::Plain(uniforms) => BlinnPhongShader.model_view(uniforms),
            MtlUniforms::Textured(uniforms) => TexturedBlinnPhongShader.model_view(uniforms),
            MtlUniforms::NormalMapped(uniforms) => {
                NormalMappedBlinnPhongShader.model_view(uniforms)
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let path = format!("{}/assets/multi_material.obj", env!("CARGO_MANIFEST_DIR"));
    let asset = Mesh::load_with_materials(path)?;
    let mesh = asset.mesh;
    let materials = asset.materials;
    let group_meshes = mesh
        .material_groups()
        .iter()
        .map(|group| {
            (
                group.material_name().to_string(),
                mesh.submesh(group.triangle_range())
                    .expect("validated material group range"),
            )
        })
        .collect::<Vec<_>>();
    let orbit = OrbitController::new(Vec3::ZERO, 5.5, 0.0, 0.0);

    run_demo(
        "chimy2 mtl materials",
        800,
        600,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = orbit.camera(0.9, aspect, 0.1, 100.0);
            let model = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.35);
            framebuffer.clear(argb8888(255, 10, 14, 22));

            let mut uniforms = Vec::with_capacity(group_meshes.len());
            for (material_name, _) in &group_meshes {
                let Some(material) = materials.get(material_name) else {
                    continue;
                };
                let mut lighting = BlinnPhongUniforms::new(
                    model,
                    camera.view_matrix(),
                    camera.projection_matrix(),
                    Vec3::new(0.03, 0.03, 0.03),
                    Vec3::new(0.8, 0.8, 0.8),
                    Vec3::new(0.3, 0.3, 0.3),
                    16.0,
                    camera.position,
                    DirectionalLight::new(
                        Vec3::new(-0.3, 0.4, 1.0).normalize(),
                        Vec3::new(1.0, 0.95, 0.9),
                    ),
                    PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
                );
                material.apply_to(&mut lighting);
                if let (Some(texture), Some(normal_map)) =
                    (material.albedo_texture(), material.normal_map_texture())
                {
                    uniforms.push(MtlUniforms::NormalMapped(
                        NormalMappedBlinnPhongUniforms::new(
                            lighting,
                            texture,
                            normal_map,
                            TextureFilter::Bilinear,
                        )
                        .expect("MTL loader assigns linear color space to normal maps"),
                    ));
                } else if let Some(texture) = material.albedo_texture() {
                    uniforms.push(MtlUniforms::Textured(TexturedBlinnPhongUniforms::new(
                        lighting,
                        texture,
                        TextureFilter::Bilinear,
                    )));
                } else {
                    uniforms.push(MtlUniforms::Plain(lighting));
                }
            }

            let mut pipeline = Pipeline::new(MtlShader, MtlShader);
            pipeline.render(framebuffer, |frame, target| {
                for ((_, group_mesh), uniforms) in group_meshes.iter().zip(uniforms.iter()) {
                    frame.draw_mesh_with_sampling(target, group_mesh, uniforms);
                }
            });
        },
    )
}
