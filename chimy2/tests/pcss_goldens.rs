use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pcss_shadow_scene::build_pcss_shadow_scene;
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use chimy2::shadow::ShadowState;
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

fn render_pcss_scene() -> Framebuffer {
    let scene = build_pcss_shadow_scene();
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let view = Mat4::look_at(
        scene.camera_position,
        scene.target,
        Vec3::new(0.0, 1.0, 0.0),
    );
    let projection = Mat4::perspective(0.78, WIDTH as f32 / HEIGHT as f32, 0.1, 40.0);
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
        shadow.set_light_size(0.85);
        uniforms.set_directional_shadow(directional, Some(shadow));
        uniforms
    };
    let ground_uniforms = make_uniforms(Mat4::IDENTITY, Vec3::new(0.52, 0.55, 0.58));
    let pole_uniforms = make_uniforms(scene.pole_model, Vec3::new(0.72, 0.22, 0.07));
    let slab_uniforms = make_uniforms(scene.slab_model, Vec3::new(0.12, 0.38, 0.72));
    framebuffer.clear(argb8888(255, 11, 15, 24));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &scene.ground, &ground_uniforms);
        frame.draw_mesh(target, &scene.pole, &pole_uniforms);
        frame.draw_mesh(target, &scene.slab, &slab_uniforms);
    });
    framebuffer
}

#[test]
fn pcss_pole_and_slab_golden() {
    assert_golden("pcss-pole-slab", &render_pcss_scene());
}

#[test]
fn pcss_scene_is_byte_deterministic() {
    assert_eq!(render_pcss_scene().color, render_pcss_scene().color);
}
