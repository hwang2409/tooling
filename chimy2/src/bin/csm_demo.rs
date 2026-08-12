use chimy2::cascaded_shadow_scene::build_cascaded_shadow_scene;
use chimy2::csm::{CascadeShadowConfig, render_cascade_shadow_maps_with_config};
use chimy2::demo::run_demo;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 640;
const SHADOW_SIZE: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = chimy2::demo::DemoArgs::from_env()?;
    run_demo(
        "chimy2 cascaded directional shadows",
        WIDTH,
        HEIGHT,
        args,
        draw,
    )
}

fn draw(framebuffer: &mut Framebuffer, _: f32, _: &InputState) {
    let scene = build_cascaded_shadow_scene(
        framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32,
    );
    let mut shadow_meshes = vec![(&scene.ground, Mat4::IDENTITY)];
    shadow_meshes.extend(
        scene
            .object_models
            .iter()
            .map(|&model| (&scene.object, model)),
    );
    let cascades = render_cascade_shadow_maps_with_config(
        scene.camera,
        scene.light_direction,
        CascadeShadowConfig::default(),
        SHADOW_SIZE,
        &shadow_meshes,
    )
    .expect("cascaded shadow maps have valid dimensions");
    let directional = DirectionalLight::new(scene.light_direction, Vec3::new(1.0, 0.93, 0.82));
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let make_uniforms = |model: Mat4, color: Vec3| {
        let mut uniforms = BlinnPhongUniforms::new(
            model,
            scene.camera.view_matrix(),
            scene.camera.projection_matrix(),
            Vec3::new(0.02, 0.025, 0.035),
            color,
            Vec3::new(0.25, 0.25, 0.25),
            24.0,
            scene.camera.position,
            directional,
            point,
        );
        uniforms.set_directional_cascaded_shadow(directional, cascades.clone());
        uniforms
    };
    let ground_uniforms = make_uniforms(Mat4::IDENTITY, Vec3::new(0.52, 0.55, 0.58));
    let object_uniforms = scene
        .object_models
        .map(|model| make_uniforms(model, Vec3::new(0.78, 0.24, 0.07)));
    framebuffer.clear(argb8888(255, 11, 15, 24));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(framebuffer, |frame, target| {
        frame.draw_mesh(target, &scene.ground, &ground_uniforms);
        for uniforms in &object_uniforms {
            frame.draw_mesh(target, &scene.object, uniforms);
        }
    });
}
