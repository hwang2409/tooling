//! Several colored moving lights on a deterministic software-rendered sphere.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let mesh = uv_sphere(1.35, 32, 64);

    run_demo(
        "chimy2 multiple lights",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = OrbitController::new(Vec3::ZERO, 4.6, 0.35 + elapsed * 0.22, 0.24)
                .camera(1.0, aspect, 0.1, 100.0);
            let model = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.2);
            let mut uniforms = BlinnPhongUniforms::new(
                model,
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.035, 0.04, 0.055),
                Vec3::new(0.72, 0.58, 0.42),
                Vec3::new(0.7, 0.7, 0.7),
                48.0,
                camera.position,
                DirectionalLight::new(
                    Vec3::new(-0.4, 0.8, 1.0).normalize(),
                    Vec3::new(0.85, 0.9, 1.0),
                ),
                PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
            );
            uniforms
                .add_point_light(PointLight::new(
                    Vec3::new(
                        (elapsed * 0.8).cos() * 2.6,
                        1.4,
                        (elapsed * 0.8).sin() * 2.6,
                    ),
                    Vec3::new(1.0, 0.18, 0.08),
                    1.0,
                    0.12,
                    0.025,
                ))
                .expect("point-light capacity");
            uniforms
                .add_point_light(PointLight::new(
                    Vec3::new(
                        (elapsed * 0.65 + 2.1).cos() * 2.3,
                        0.6,
                        (elapsed * 0.65 + 2.1).sin() * 2.3,
                    ),
                    Vec3::new(0.12, 0.38, 1.0),
                    1.0,
                    0.1,
                    0.02,
                ))
                .expect("point-light capacity");
            uniforms
                .add_point_light(PointLight::new(
                    Vec3::new(
                        (elapsed * 0.5 + 4.2).cos() * 2.0,
                        -1.1,
                        (elapsed * 0.5 + 4.2).sin() * 2.0,
                    ),
                    Vec3::new(0.15, 1.0, 0.3),
                    1.0,
                    0.14,
                    0.03,
                ))
                .expect("point-light capacity");

            framebuffer.clear(argb8888(255, 8, 10, 16));
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw_mesh(target, &mesh, &uniforms);
            });
        },
    )
}
