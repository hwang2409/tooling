use chimy2::demo::{cube_with_uvs, plane_xz};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use chimy2::shadow::{
    ShadowDepthShader, ShadowDepthUniforms, ShadowMap, ShadowState, directional_light_view,
};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join(format!("{name}.ppm"))
}

fn assert_golden(name: &str, framebuffer: &Framebuffer) {
    let path = golden_path(name);
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).expect("write golden");
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    let expected = fs::read(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

fn render_shadow_scene(light_direction: Vec3, constant_bias: f32, slope_bias: f32) -> Framebuffer {
    let ground = plane_xz(8.0, 8.0, 8, 1.0);
    let caster = cube_with_uvs(0.8);
    let caster_model = Mat4::translate(Vec3::new(0.0, 0.8, 0.0));
    let target = Vec3::new(0.0, 0.7, 0.0);
    let light_direction = light_direction.normalize();
    let light_view_projection = Mat4::orthographic(-5.0, 5.0, -5.0, 5.0, 1.0, 20.0)
        * directional_light_view(light_direction, target, 8.0, Vec3::new(0.0, 1.0, 0.0));

    let mut shadow_target = Framebuffer::new(128, 128);
    shadow_target.clear(0);
    let mut depth_pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    depth_pipeline.draw_mesh_depth(
        &mut shadow_target,
        &ground,
        &ShadowDepthUniforms::new(Mat4::IDENTITY, light_view_projection),
    );
    depth_pipeline.draw_mesh_depth(
        &mut shadow_target,
        &caster,
        &ShadowDepthUniforms::new(caster_model, light_view_projection),
    );
    let shadow_map = ShadowMap::from_framebuffer(&shadow_target).expect("shadow target has size");

    let camera_position = Vec3::new(5.2, 3.6, 6.0);
    let view = Mat4::look_at(camera_position, target, Vec3::new(0.0, 1.0, 0.0));
    let projection = Mat4::perspective(0.78, WIDTH as f32 / HEIGHT as f32, 0.1, 30.0);
    let directional = DirectionalLight::new(light_direction, Vec3::new(1.0, 0.95, 0.85));
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let make_uniforms = |model: Mat4, diffuse: Vec3| {
        let mut uniforms = BlinnPhongUniforms::new(
            model,
            view,
            projection,
            Vec3::new(0.025, 0.025, 0.025),
            diffuse,
            Vec3::new(0.3, 0.3, 0.3),
            24.0,
            camera_position,
            directional,
            point,
        );
        let mut shadow_state = ShadowState::new(light_view_projection, shadow_map.clone());
        shadow_state.set_bias(constant_bias, slope_bias);
        uniforms.set_directional_shadow(directional, Some(shadow_state));
        uniforms
    };

    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    let ground_uniforms = make_uniforms(Mat4::IDENTITY, Vec3::new(0.58, 0.6, 0.62));
    let caster_uniforms = make_uniforms(caster_model, Vec3::new(0.78, 0.25, 0.08));
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &ground, &ground_uniforms);
        frame.draw_mesh(target, &caster, &caster_uniforms);
    });
    framebuffer
}

#[test]
fn caster_and_ground_shadow_golden() {
    assert_golden(
        "m11-directional-shadow",
        &render_shadow_scene(Vec3::new(0.7, 1.0, 0.35), 0.002, 0.025),
    );
}

#[test]
fn moving_light_changes_shadow_direction_golden() {
    let original = render_shadow_scene(Vec3::new(0.7, 1.0, 0.35), 0.002, 0.025);
    let moved = render_shadow_scene(Vec3::new(-0.8, 1.0, -0.2), 0.002, 0.025);
    assert_ne!(original.color, moved.color);
    assert_golden("m11-directional-shadow-moved-light", &moved);
}

#[test]
fn grazing_angle_bias_golden() {
    let biased = render_shadow_scene(Vec3::new(0.12, 0.3, 1.0), 0.008, 0.08);
    let unbiassed = render_shadow_scene(Vec3::new(0.12, 0.3, 1.0), 0.0, 0.0);
    assert_ne!(biased.color, unbiassed.color);
    assert_golden("m11-directional-shadow-grazing-bias", &biased);
}
