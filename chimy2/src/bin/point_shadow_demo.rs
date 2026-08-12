use chimy2::demo::run_demo;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::point_shadow_scene::build_point_shadow_scene;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use chimy2::shadow::{CubeShadowState, render_cube_shadow_map};
use std::f32::consts::PI;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const SHADOW_SIZE: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = chimy2::demo::DemoArgs::from_env().map_err(std::io::Error::other)?;
    run_demo("chimy2 point-light cube shadows", WIDTH, HEIGHT, args, draw)
}

fn draw(framebuffer: &mut Framebuffer, _: f32, _: &InputState) {
    let scene = build_point_shadow_scene();
    let mut shadow_meshes = vec![(&scene.floor, Mat4::IDENTITY)];
    shadow_meshes.extend(scene.wall_models.iter().map(|&model| (&scene.wall, model)));
    shadow_meshes.extend(
        scene
            .occluder_models
            .iter()
            .map(|&model| (&scene.occluder, model)),
    );
    let shadow_map =
        render_cube_shadow_map(scene.light_position, 0.1, 14.0, SHADOW_SIZE, &shadow_meshes)
            .expect("point shadow faces have valid dimensions");
    let mut shadow_state = CubeShadowState::new(scene.light_position, shadow_map);
    shadow_state.set_bias(0.0015, 0.018);

    framebuffer.clear(argb8888(255, 8, 10, 16));
    let camera_position = Vec3::new(7.0, 4.8, 8.0);
    let target = Vec3::new(0.0, 1.8, 0.0);
    let view = Mat4::look_at(camera_position, target, Vec3::new(0.0, 1.0, 0.0));
    let projection = Mat4::perspective(
        PI / 4.0,
        framebuffer.width as f32 / framebuffer.height.max(1) as f32,
        0.1,
        30.0,
    );
    let point = PointLight::new(
        scene.light_position,
        Vec3::new(1.0, 0.68, 0.3),
        1.0,
        0.08,
        0.025,
    );
    let directional = DirectionalLight::new(Vec3::ZERO, Vec3::ZERO);
    let make_uniforms = |model: Mat4, color: Vec3| {
        let mut uniforms = BlinnPhongUniforms::new(
            model,
            view,
            projection,
            Vec3::new(0.015, 0.018, 0.025),
            color,
            Vec3::new(0.3, 0.3, 0.3),
            28.0,
            camera_position,
            directional,
            point,
        );
        uniforms
            .set_point_light_shadow(0, Some(shadow_state.clone()))
            .expect("point light zero exists");
        uniforms
    };
    let floor_uniforms = make_uniforms(Mat4::IDENTITY, Vec3::new(0.5, 0.52, 0.56));
    let wall_uniforms = scene
        .wall_models
        .map(|model| make_uniforms(model, Vec3::new(0.28, 0.32, 0.42)));
    let occluder_uniforms = scene
        .occluder_models
        .map(|model| make_uniforms(model, Vec3::new(0.75, 0.18, 0.06)));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(framebuffer, |frame, target| {
        frame.draw_mesh(target, &scene.floor, &floor_uniforms);
        for uniforms in &wall_uniforms {
            frame.draw_mesh(target, &scene.wall, uniforms);
        }
        for uniforms in &occluder_uniforms {
            frame.draw_mesh(target, &scene.occluder, uniforms);
        }
    });
}
