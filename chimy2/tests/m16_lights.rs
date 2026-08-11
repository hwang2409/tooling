use chimy2::camera::Camera;
use chimy2::demo::uv_sphere;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use std::f32::consts::FRAC_PI_3;
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

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

fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 5.4),
        Quat::IDENTITY,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    )
}

fn uniforms(camera: Camera) -> BlinnPhongUniforms {
    let mut uniforms = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.025, 0.03, 0.04),
        Vec3::new(0.72, 0.58, 0.42),
        Vec3::new(0.35, 0.35, 0.35),
        32.0,
        camera.position,
        DirectionalLight::new(Vec3::ZERO, Vec3::ZERO),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    );
    uniforms.clear_directional_lights();
    uniforms.clear_point_lights();
    uniforms
}

fn render(uniforms: BlinnPhongUniforms) -> Framebuffer {
    let mesh = uv_sphere(1.4, 24, 48);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 8, 10, 16));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.draw_mesh(&mut framebuffer, &mesh, &uniforms);
    framebuffer
}

#[test]
fn two_directional_lights_from_different_sides_golden() {
    let camera = camera();
    let mut uniforms = uniforms(camera);
    uniforms
        .add_directional_light(DirectionalLight::new(
            Vec3::new(-0.65, 0.2, 1.0).normalize(),
            Vec3::new(1.0, 0.25, 0.12),
        ))
        .unwrap();
    uniforms
        .add_directional_light(DirectionalLight::new(
            Vec3::new(0.65, 0.15, 1.0).normalize(),
            Vec3::new(0.12, 0.35, 1.0),
        ))
        .unwrap();
    assert_golden("m16-two-directional-lights", &render(uniforms));
}

#[test]
fn three_point_lights_have_distinct_colors_and_falloffs_golden() {
    let camera = camera();
    let mut uniforms = uniforms(camera);
    uniforms
        .add_point_light(PointLight::new(
            Vec3::new(-2.4, 1.6, 2.5),
            Vec3::new(1.0, 0.1, 0.08),
            1.0,
            0.05,
            0.02,
        ))
        .unwrap();
    uniforms
        .add_point_light(PointLight::new(
            Vec3::new(2.1, 0.3, 2.0),
            Vec3::new(0.08, 0.35, 1.0),
            1.0,
            0.12,
            0.04,
        ))
        .unwrap();
    uniforms
        .add_point_light(PointLight::new(
            Vec3::new(0.0, -2.3, 1.0),
            Vec3::new(0.12, 1.0, 0.22),
            1.0,
            0.2,
            0.08,
        ))
        .unwrap();
    assert_golden("m16-three-point-lights", &render(uniforms));
}

#[test]
fn many_lights_clamp_after_accumulation_golden() {
    let camera = camera();
    let mut uniforms = uniforms(camera);
    for (index, color) in [
        Vec3::new(1.0, 0.15, 0.1),
        Vec3::new(0.1, 1.0, 0.15),
        Vec3::new(0.12, 0.2, 1.0),
        Vec3::new(1.0, 0.75, 0.1),
        Vec3::new(0.7, 0.1, 1.0),
        Vec3::new(0.1, 1.0, 0.9),
    ]
    .into_iter()
    .enumerate()
    {
        let angle = index as f32 * FRAC_PI_3;
        uniforms
            .add_point_light(PointLight::new(
                Vec3::new(angle.cos() * 2.8, angle.sin() * 1.4, 2.4),
                color,
                1.0,
                0.02,
                0.01,
            ))
            .unwrap();
    }
    assert_golden("m16-many-lights", &render(uniforms));
}
