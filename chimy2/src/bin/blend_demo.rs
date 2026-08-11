//! Alpha-blending and SSAA showcase.
//!
//! Transparent triangles use a stable back-to-front view-space centroid sort.
//! Intersecting or cyclically overlapping triangles remain an order
//! approximation because a per-triangle sort cannot solve those cycles.

use chimy2::demo::{DemoArgs, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use std::f32::consts::PI;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env().map_err(std::io::Error::other)?;
    let ssaa = args.ssaa;
    let sphere = uv_sphere(1.25, 24, 36);
    run_demo(
        "chimy2 alpha blending and ssaa",
        WIDTH,
        HEIGHT,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera_position = Vec3::new(0.0, 1.2, 6.0);
            let view = Mat4::look_at(
                camera_position,
                Vec3::new(0.0, 0.3, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            let projection = Mat4::perspective(PI / 4.0, aspect, 0.1, 30.0);
            let light = DirectionalLight::new(
                Vec3::new(0.5, 0.9, 1.0).normalize(),
                Vec3::new(1.0, 0.95, 0.85),
            );
            let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
            let sphere_uniforms = BlinnPhongUniforms::new(
                Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.25),
                view,
                projection,
                Vec3::new(0.04, 0.04, 0.05),
                Vec3::new(0.25, 0.45, 0.8),
                Vec3::new(0.6, 0.7, 0.9),
                32.0,
                camera_position,
                light,
                point,
            );
            let mut quad_uniforms = BlinnPhongUniforms::new(
                Mat4::IDENTITY,
                view,
                projection,
                Vec3::new(0.08, 0.03, 0.02),
                Vec3::new(0.95, 0.25, 0.08),
                Vec3::ZERO,
                8.0,
                camera_position,
                light,
                point,
            );
            quad_uniforms.set_alpha(0.45);
            let quads = orbiting_quads(elapsed);

            framebuffer.clear(argb8888(255, 10, 14, 24));
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            pipeline.set_ssaa_scale(if ssaa { 2 } else { 1 });
            pipeline.render(framebuffer, |frame, target| {
                target.clear(argb8888(255, 10, 14, 24));
                frame.draw_mesh(target, &sphere, &sphere_uniforms);
                frame.draw_mesh(target, &quads, &quad_uniforms);
            });
        },
    )
}

fn orbiting_quads(elapsed: f32) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for (index, center) in [Vec3::new(-1.25, 0.6, 0.45), Vec3::new(1.25, 0.85, -0.15)]
        .into_iter()
        .enumerate()
    {
        let angle = elapsed * (0.65 + index as f32 * 0.2) + index as f32;
        let rotation = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), angle);
        let base = vertices.len();
        for point in [
            Vec3::new(-0.8, -0.7, 0.0),
            Vec3::new(0.8, -0.7, 0.0),
            Vec3::new(0.8, 0.7, 0.0),
            Vec3::new(-0.8, 0.7, 0.0),
        ] {
            let position = rotation * chimy2::math::Vec4::new(point.x, point.y, point.z, 1.0);
            vertices.push(MeshVertex::new(
                Vec3::new(
                    position.x + center.x,
                    position.y + center.y,
                    position.z + center.z,
                ),
                None,
                Some(Vec3::new(0.0, 0.0, 1.0)),
            ));
        }
        triangles.push([base, base + 1, base + 2]);
        triangles.push([base, base + 2, base + 3]);
    }
    Mesh::new(vertices, triangles)
}
