use chimy2::demo::run_demo;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pcss_shadow_scene::build_pcss_shadow_scene;
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use chimy2::shadow::ShadowState;
use std::f32::consts::PI;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 640;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = chimy2::demo::DemoArgs::from_env().map_err(std::io::Error::other)?;
    run_demo(
        "chimy2 percentage-closer soft shadows",
        WIDTH,
        HEIGHT,
        args,
        draw,
    )
}

fn draw(framebuffer: &mut Framebuffer, _: f32, _: &InputState) {
    let scene = build_pcss_shadow_scene();
    let view = Mat4::look_at(
        scene.camera_position,
        scene.target,
        Vec3::new(0.0, 1.0, 0.0),
    );
    let projection = Mat4::perspective(
        PI / 4.0,
        framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32,
        0.1,
        40.0,
    );
    let directional = DirectionalLight::new(scene.light_direction, Vec3::new(1.0, 0.93, 0.82));
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let make_uniforms = |model: Mat4, diffuse: Vec3| {
        let mut uniforms = BlinnPhongUniforms::new(
            model,
            view,
            projection,
            Vec3::new(0.02, 0.025, 0.035),
            diffuse,
            Vec3::new(0.24, 0.24, 0.24),
            28.0,
            scene.camera_position,
            directional,
            point,
        );
        let mut shadow = ShadowState::new(scene.light_view_projection, scene.shadow_map.clone());
        shadow.set_bias(0.002, 0.025);
        shadow.set_light_size(1.4);
        uniforms.set_directional_shadow(directional, Some(shadow));
        uniforms
    };
    let ground_uniforms = make_uniforms(Mat4::IDENTITY, Vec3::new(0.52, 0.55, 0.58));
    let pole_uniforms = make_uniforms(scene.pole_model, Vec3::new(0.72, 0.22, 0.07));
    let slab_uniforms = make_uniforms(scene.slab_model, Vec3::new(0.12, 0.38, 0.72));
    framebuffer.clear(argb8888(255, 11, 15, 24));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(framebuffer, |frame, target| {
        frame.draw_mesh(target, &scene.ground, &ground_uniforms);
        frame.draw_mesh(target, &scene.pole, &pole_uniforms);
        frame.draw_mesh(target, &scene.slab, &slab_uniforms);
    });
}
