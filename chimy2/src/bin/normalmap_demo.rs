//! Tangent-space normal mapping demo with a deterministic orbiting light.

use chimy2::camera::Camera;
use chimy2::demo::{DemoArgs, run_demo};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::{ColorSpace, Texture, WrapMode};
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{
    BlinnPhongUniforms, DirectionalLight, NormalMappedBlinnPhongShader,
    NormalMappedBlinnPhongUniforms, PointLight, TextureFilter,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let mesh = Mesh::parse(
        "v -1.5 -1.0 0\nv 1.5 -1.0 0\nv 1.5 1.0 0\nv -1.5 1.0 0\n\
         vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
         f 1/1 2/2 3/3 4/4\n",
    )?;
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let albedo = Texture::load(assets.join("gradient.qoi"))?.with_wrap_mode(WrapMode::Repeat);
    let normal_map =
        Texture::load_with_color_space(assets.join("normal_bump.qoi"), ColorSpace::Linear)?
            .with_wrap_mode(WrapMode::Repeat);

    run_demo(
        "chimy2 normal mapping",
        800,
        600,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = Camera::new(
                Vec3::new(0.0, 0.0, 4.5),
                chimy2::math::Quat::IDENTITY,
                0.9,
                aspect,
                0.1,
                100.0,
            );
            let angle = elapsed * 1.2;
            let light_direction = Vec3::new(angle.cos() * 0.8, 0.45, angle.sin() * 0.8).normalize();
            let lighting = BlinnPhongUniforms::new(
                Mat4::IDENTITY,
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.04, 0.04, 0.04),
                Vec3::new(0.9, 0.9, 0.9),
                Vec3::new(0.8, 0.8, 0.8),
                32.0,
                camera.position,
                DirectionalLight::new(light_direction, Vec3::new(1.0, 0.95, 0.9)),
                PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
            );
            let uniforms = NormalMappedBlinnPhongUniforms::new(
                lighting,
                &albedo,
                &normal_map,
                TextureFilter::Bilinear,
            );
            framebuffer.clear(argb8888(255, 10, 14, 22));
            let mut pipeline =
                Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
            pipeline.draw_mesh_with_sampling(framebuffer, &mesh, &uniforms);
        },
    )
}
