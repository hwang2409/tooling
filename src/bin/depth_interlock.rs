//! Depth-buffer showcase: three orthogonal bars pierce a central sphere.
//!
//! Every triangle draws through the same shader over the same framebuffer, and
//! the z-buffer resolves per-pixel occlusion. A painter's-algorithm renderer
//! cannot reproduce this scene because there is no submission order that keeps
//! the intersecting surfaces visually correct.

use chimy2::camera::{Camera, OrbitController};
use chimy2::demo::{DemoArgs, cube_with_uvs, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};

struct Body {
    mesh: chimy2::mesh::Mesh,
    model: Mat4,
    color: Vec3,
    shininess: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let sphere = uv_sphere(0.95, 32, 64);
    let bar = cube_with_uvs(1.0);

    run_demo(
        "chimy2 depth interlock",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let orbit = OrbitController::new(Vec3::ZERO, 6.0, 0.6 + elapsed * 0.18, 0.35);
            let camera: Camera = orbit.camera(1.0, aspect, 0.1, 100.0);

            let spin = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.35)
                * Mat4::rotate(Vec3::new(1.0, 0.0, 0.0), elapsed * 0.18);

            let bar_scale_x = Mat4::scale(Vec3::new(2.6, 0.28, 0.28));
            let bar_scale_y = Mat4::scale(Vec3::new(0.28, 2.6, 0.28));
            let bar_scale_z = Mat4::scale(Vec3::new(0.28, 0.28, 2.6));
            let bodies = [
                Body {
                    mesh: sphere.clone(),
                    model: spin,
                    color: Vec3::new(0.86, 0.82, 0.72),
                    shininess: 64.0,
                },
                Body {
                    mesh: bar.clone(),
                    model: spin * bar_scale_x,
                    color: Vec3::new(0.92, 0.30, 0.24),
                    shininess: 96.0,
                },
                Body {
                    mesh: bar.clone(),
                    model: spin * bar_scale_y,
                    color: Vec3::new(0.30, 0.85, 0.75),
                    shininess: 96.0,
                },
                Body {
                    mesh: bar.clone(),
                    model: spin * bar_scale_z,
                    color: Vec3::new(0.42, 0.55, 0.98),
                    shininess: 96.0,
                },
            ];

            framebuffer.clear(argb8888(255, 10, 12, 18));
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            for body in &bodies {
                let uniforms = BlinnPhongUniforms::new(
                    body.model,
                    camera.view_matrix(),
                    camera.projection_matrix(),
                    Vec3::new(0.10, 0.12, 0.16),
                    body.color,
                    Vec3::new(0.75, 0.78, 0.82),
                    body.shininess,
                    camera.position,
                    DirectionalLight::new(
                        Vec3::new(0.35, 0.85, 0.55).normalize(),
                        Vec3::new(1.05, 1.00, 0.90),
                    ),
                    PointLight::new(
                        Vec3::new(-3.0, 2.5, 3.5),
                        Vec3::new(0.55, 0.68, 1.05),
                        1.0,
                        0.08,
                        0.02,
                    ),
                );
                pipeline.draw_mesh(framebuffer, &body.mesh, &uniforms);
            }
        },
    )
}
