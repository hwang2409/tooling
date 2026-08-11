//! Reflective mesh under a procedural cube-map sky.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo, uv_sphere};
use chimy2::fb::Framebuffer;
use chimy2::image::Texture;
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{
    BlinnPhongUniforms, DirectionalLight, EnvironmentBlinnPhongShader,
    EnvironmentBlinnPhongUniforms, PointLight,
};
use chimy2::skybox::CubeTexture;
use std::path::Path;

fn load_skybox() -> Result<CubeTexture, Box<dyn std::error::Error>> {
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let faces = ["px", "nx", "py", "ny", "pz", "nz"]
        .map(|name| Texture::load(assets.join(format!("skybox_{name}.qoi"))))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CubeTexture::new(faces.try_into().expect("six cube faces"))?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let skybox = load_skybox()?;
    let mesh = uv_sphere(1.3, 24, 48);

    run_demo(
        "chimy2 skybox",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = OrbitController::new(Vec3::ZERO, 4.4, elapsed * 0.28, 0.18)
                .camera(1.0, aspect, 0.1, 100.0);
            let lighting = BlinnPhongUniforms::new(
                Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.4),
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.03, 0.04, 0.06),
                Vec3::new(0.3, 0.35, 0.42),
                Vec3::new(0.9, 0.95, 1.0),
                48.0,
                camera.position,
                DirectionalLight::new(
                    Vec3::new(-0.35, 0.75, 1.0).normalize(),
                    Vec3::new(1.0, 0.92, 0.82),
                ),
                PointLight::new(
                    Vec3::new(2.0, 1.5, 3.0),
                    Vec3::new(0.6, 0.7, 1.0),
                    1.0,
                    0.08,
                    0.02,
                ),
            );
            let uniforms = EnvironmentBlinnPhongUniforms::new(lighting, &skybox, 0.72);
            let mut pipeline =
                Pipeline::new(EnvironmentBlinnPhongShader, EnvironmentBlinnPhongShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw_skybox(target, &skybox, camera);
                frame.draw_mesh(target, &mesh, &uniforms);
            });
        },
    )
}
