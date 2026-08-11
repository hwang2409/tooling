//! Headless-friendly demo for OBJ material groups and MTL texture maps.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, NormalMappedBlinnPhongShader,
    NormalMappedBlinnPhongUniforms, PointLight, TextureFilter, TexturedBlinnPhongShader,
    TexturedBlinnPhongUniforms,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let path = format!("{}/assets/multi_material.obj", env!("CARGO_MANIFEST_DIR"));
    let asset = Mesh::load_with_materials(path)?;
    let mesh = asset.mesh;
    let materials = asset.materials;
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

            for group in mesh.material_groups() {
                let Some(material) = materials.get(group.material_name()) else {
                    continue;
                };
                let Some(group_mesh) = mesh.submesh(group.triangle_range()) else {
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
                    let uniforms = NormalMappedBlinnPhongUniforms::new(
                        lighting,
                        texture,
                        normal_map,
                        TextureFilter::Bilinear,
                    )
                    .expect("MTL loader assigns linear color space to normal maps");
                    let mut pipeline =
                        Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
                    pipeline.draw_mesh_with_sampling(framebuffer, &group_mesh, &uniforms);
                } else if let Some(texture) = material.albedo_texture() {
                    let uniforms =
                        TexturedBlinnPhongUniforms::new(lighting, texture, TextureFilter::Bilinear);
                    let mut pipeline =
                        Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
                    pipeline.draw_mesh_with_sampling(framebuffer, &group_mesh, &uniforms);
                } else {
                    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
                    pipeline.draw_mesh(framebuffer, &group_mesh, &lighting);
                }
            }
        },
    )
}
