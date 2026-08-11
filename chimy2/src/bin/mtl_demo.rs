//! Headless-friendly demo for OBJ material groups and MTL texture maps.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::Mesh;
use chimy2::mtl_render::{MaterialGroup, submit_material_groups};
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongUniforms, DirectionalLight, PointLight};

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

            let base_lighting = BlinnPhongUniforms::new(
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
            let groups = group_meshes
                .iter()
                .filter_map(|(material_name, group_mesh)| {
                    materials
                        .get(material_name)
                        .map(|material| MaterialGroup::new(group_mesh, material))
                })
                .collect::<Vec<_>>();
            submit_material_groups(framebuffer, &groups, &base_lighting);
        },
    )
}
