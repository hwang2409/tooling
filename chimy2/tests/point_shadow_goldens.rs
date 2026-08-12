use chimy2::demo::{cube_with_uvs, plane_xz};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use chimy2::shadow::{CubeShadowState, render_cube_shadow_map};
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

fn render_point_shadow_scene() -> Framebuffer {
    let floor = plane_xz(5.0, 5.0, 8, 1.0);
    let cube = cube_with_uvs(0.7);
    let cube_model = Mat4::translate(Vec3::new(0.0, 0.7, 0.0));
    let light_position = Vec3::new(-1.2, 3.0, 1.0);
    let shadow_map = render_cube_shadow_map(
        light_position,
        0.1,
        10.0,
        96,
        &[(&floor, Mat4::IDENTITY), (&cube, cube_model)],
    )
    .unwrap();
    let mut shadow_state = CubeShadowState::new(light_position, shadow_map);
    shadow_state.set_bias(0.0015, 0.018);

    let camera_position = Vec3::new(5.0, 3.8, 5.8);
    let target = Vec3::new(0.0, 0.65, 0.0);
    let view = Mat4::look_at(camera_position, target, Vec3::new(0.0, 1.0, 0.0));
    let projection = Mat4::perspective(0.78, WIDTH as f32 / HEIGHT as f32, 0.1, 20.0);
    let directional = DirectionalLight::new(Vec3::ZERO, Vec3::ZERO);
    let point = PointLight::new(light_position, Vec3::new(1.0, 0.55, 0.22), 1.0, 0.06, 0.02);
    let make_uniforms = |model: Mat4, color: Vec3| {
        let mut uniforms = BlinnPhongUniforms::new(
            model,
            view,
            projection,
            Vec3::new(0.018, 0.02, 0.028),
            color,
            Vec3::new(0.28, 0.28, 0.28),
            24.0,
            camera_position,
            directional,
            point,
        );
        uniforms
            .set_point_light_shadow(0, Some(shadow_state.clone()))
            .unwrap();
        uniforms
    };
    let floor_uniforms = make_uniforms(Mat4::IDENTITY, Vec3::new(0.48, 0.52, 0.58));
    let cube_uniforms = make_uniforms(cube_model, Vec3::new(0.72, 0.16, 0.04));
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 8, 10, 16));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &floor, &floor_uniforms);
        frame.draw_mesh(target, &cube, &cube_uniforms);
    });
    framebuffer
}

#[test]
fn point_shadow_demo_golden() {
    assert_golden("m25-point-shadow", &render_point_shadow_scene());
}
