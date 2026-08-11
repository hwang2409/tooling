use chimy2::demo::{plane_xz, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use chimy2::shadow::{
    ShadowDepthShader, ShadowDepthUniforms, ShadowMap, ShadowState, directional_light_view,
};
use std::f32::consts::PI;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const SHADOW_SIZE: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = chimy2::demo::DemoArgs::from_env().map_err(std::io::Error::other)?;
    run_demo("chimy2 directional shadows", WIDTH, HEIGHT, args, draw)
}

fn draw(framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState) {
    let ground = plane_xz(10.0, 10.0, 8, 1.0);
    let caster = uv_sphere(1.0, 16, 24);
    let caster_model = Mat4::translate(Vec3::new(0.0, 1.15, 0.0));
    let target = Vec3::new(0.0, 0.8, 0.0);
    let direction = Vec3::new(
        (elapsed * 0.7).cos() * 0.75,
        1.1,
        (elapsed * 0.7).sin() * 0.75,
    )
    .normalize();
    let light_view_projection = Mat4::orthographic(-6.0, 6.0, -6.0, 6.0, 1.0, 20.0)
        * directional_light_view(direction, target, 8.0, Vec3::new(0.0, 1.0, 0.0));

    let mut shadow_target = Framebuffer::new(SHADOW_SIZE, SHADOW_SIZE);
    shadow_target.clear(0);
    let mut depth_pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    let ground_depth = ShadowDepthUniforms::new(Mat4::IDENTITY, light_view_projection);
    depth_pipeline.draw_mesh_depth(&mut shadow_target, &ground, &ground_depth);
    let caster_depth = ShadowDepthUniforms::new(caster_model, light_view_projection);
    depth_pipeline.draw_mesh_depth(&mut shadow_target, &caster, &caster_depth);
    let shadow_map = ShadowMap::from_framebuffer(&shadow_target).expect("shadow target has size");

    framebuffer.clear(argb8888(255, 14, 19, 30));
    let camera_position = Vec3::new(6.5, 4.5, 7.0);
    let view = Mat4::look_at(camera_position, target, Vec3::new(0.0, 1.0, 0.0));
    let projection = Mat4::perspective(
        PI / 4.0,
        framebuffer.width as f32 / framebuffer.height.max(1) as f32,
        0.1,
        30.0,
    );
    let directional = DirectionalLight::new(direction, Vec3::new(1.0, 0.93, 0.82));
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);

    let mut ground_uniforms = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        view,
        projection,
        Vec3::new(0.025, 0.025, 0.025),
        Vec3::new(0.52, 0.55, 0.58),
        Vec3::new(0.15, 0.15, 0.15),
        24.0,
        camera_position,
        directional,
        point,
    );
    let mut ground_shadow = ShadowState::new(light_view_projection, shadow_map.clone());
    ground_shadow.set_bias(0.002, 0.025);
    ground_uniforms.set_directional_shadow(directional, Some(ground_shadow));

    let mut caster_uniforms = BlinnPhongUniforms::new(
        caster_model,
        view,
        projection,
        Vec3::new(0.025, 0.025, 0.025),
        Vec3::new(0.72, 0.25, 0.08),
        Vec3::new(0.85, 0.85, 0.85),
        32.0,
        camera_position,
        directional,
        point,
    );
    let mut caster_shadow = ShadowState::new(light_view_projection, shadow_map);
    caster_shadow.set_bias(0.002, 0.025);
    caster_uniforms.set_directional_shadow(directional, Some(caster_shadow));

    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(framebuffer, |frame, target| {
        frame.draw_mesh(target, &ground, &ground_uniforms);
        frame.draw_mesh(target, &caster, &caster_uniforms);
    });
}
