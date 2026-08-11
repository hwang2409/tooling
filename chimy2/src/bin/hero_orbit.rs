//! Hero scene: a slowly orbiting, textured, blinn-phong lit sphere.
//!
//! Camera drifts around the subject. A warm directional key light sweeps its
//! direction with elapsed time, and a cool point light circles the sphere.
//! All motion is deterministic — the same elapsed time produces the same
//! frame, which is what makes the `--screenshot` path reproducible.

use chimy2::camera::{Camera, OrbitController};
use chimy2::demo::{DemoArgs, banded_texture, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::WrapMode;
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{
    BlinnPhongUniforms, DirectionalLight, PointLight, TextureFilter, TexturedBlinnPhongShader,
    TexturedBlinnPhongUniforms,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let mesh = uv_sphere(1.35, 48, 96);
    let texture = banded_texture(256).with_wrap_mode(WrapMode::Repeat);

    run_demo(
        "chimy2 hero orbit",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let orbit_angle = 0.35 + elapsed * 0.25;
            let orbit = OrbitController::new(Vec3::ZERO, 4.4, orbit_angle, 0.28);
            let camera: Camera = orbit.camera(1.05, aspect, 0.1, 100.0);

            let sun_angle = elapsed * 0.4;
            let sun_direction = Vec3::new(sun_angle.cos() * 0.6, 0.85, sun_angle.sin() * 0.6);
            let point_light_position = Vec3::new(
                (elapsed * 0.9).cos() * 2.2,
                1.5 + (elapsed * 0.7).sin() * 0.25,
                (elapsed * 0.9).sin() * 2.2,
            );

            let model = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.18);
            let lighting = BlinnPhongUniforms::new(
                model,
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.22, 0.24, 0.32),
                Vec3::new(1.05, 1.05, 1.05),
                Vec3::new(0.85, 0.90, 0.95),
                72.0,
                camera.position,
                DirectionalLight::new(sun_direction.normalize(), Vec3::new(1.05, 1.02, 0.92)),
                PointLight::new(
                    point_light_position,
                    Vec3::new(0.35, 0.55, 1.05),
                    1.0,
                    0.06,
                    0.015,
                ),
            );
            let uniforms =
                TexturedBlinnPhongUniforms::new(lighting, &texture, TextureFilter::Trilinear);

            framebuffer.clear(argb8888(255, 8, 10, 16));
            let mut pipeline = Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw_mesh_with_sampling(target, &mesh, &uniforms);
            });
        },
    )
}
