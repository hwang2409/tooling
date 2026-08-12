//! Cook-Torrance GGX roughness and metallic grid under the skybox.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::Texture;
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{CookTorranceShader, CookTorranceUniforms, DirectionalLight, PointLight};
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
    let hdr = args.hdr;
    let skybox = load_skybox()?;
    let sphere = uv_sphere(0.72, 24, 48);

    run_demo(
        "chimy2 cook-torrance ggx",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = OrbitController::new(Vec3::ZERO, 8.0, elapsed * 0.18, 0.18)
                .camera(1.0, aspect, 0.1, 100.0);
            let light_direction = Vec3::new(-0.45, 0.8, 0.9).normalize();
            let point_position = Vec3::new(
                (elapsed * 0.7).cos() * 3.0,
                2.5,
                (elapsed * 0.7).sin() * 3.0,
            );

            framebuffer.clear(argb8888(255, 8, 10, 16));
            let mut uniforms = Vec::with_capacity(12);
            for row in 0..3 {
                for column in 0..4 {
                    let x = (column as f32 - 1.5) * 1.65;
                    let y = (1.0 - row as f32) * 1.65;
                    let model = Mat4::translate(Vec3::new(x, y, 0.0))
                        * Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.25);
                    uniforms.push(CookTorranceUniforms::new_with_linear_base_color(
                        chimy2::shaders::BlinnPhongUniforms::new_with_linear_colors(
                            model,
                            camera.view_matrix(),
                            camera.projection_matrix(),
                            Vec3::new(0.025, 0.03, 0.05),
                            Vec3::ZERO,
                            Vec3::ZERO,
                            0.0,
                            camera.position,
                            DirectionalLight::new(light_direction, Vec3::new(1.0, 0.95, 0.9)),
                            PointLight::new(
                                point_position,
                                Vec3::new(0.8, 0.9, 1.0),
                                1.0,
                                0.08,
                                0.02,
                            ),
                        ),
                        Vec3::new(0.82, 0.16, 0.06),
                        column as f32 / 3.0,
                        [0.1, 0.5, 0.9][row],
                    ));
                }
            }
            let mut pipeline = Pipeline::new(CookTorranceShader, CookTorranceShader);
            pipeline.set_hdr(hdr);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw_skybox(target, &skybox, camera);
                for uniform in &uniforms {
                    frame.draw_mesh(target, &sphere, uniform);
                }
            });
        },
    )
}
