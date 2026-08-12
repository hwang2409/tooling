use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::point_shadow_scene::build_point_shadow_scene;
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
    let scene = build_point_shadow_scene();
    let light_position = scene.light_position;
    let mut shadow_meshes = vec![(&scene.floor, Mat4::IDENTITY)];
    shadow_meshes.extend(scene.wall_models.iter().map(|&model| (&scene.wall, model)));
    shadow_meshes.extend(
        scene
            .occluder_models
            .iter()
            .map(|&model| (&scene.occluder, model)),
    );
    let shadow_map = render_cube_shadow_map(light_position, 0.1, 14.0, 96, &shadow_meshes).unwrap();
    let mut shadow_state = CubeShadowState::new(light_position, shadow_map);
    shadow_state.set_bias(0.0015, 0.018);

    let camera_position = Vec3::new(7.0, 4.8, 8.0);
    let target = Vec3::new(0.0, 1.8, 0.0);
    let view = Mat4::look_at(camera_position, target, Vec3::new(0.0, 1.0, 0.0));
    let projection = Mat4::perspective(
        std::f32::consts::PI / 4.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        30.0,
    );
    let directional = DirectionalLight::new(Vec3::ZERO, Vec3::ZERO);
    let point = PointLight::new(light_position, Vec3::new(1.0, 0.68, 0.3), 1.0, 0.08, 0.025);
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
    let floor_uniforms = make_uniforms(Mat4::IDENTITY, Vec3::new(0.5, 0.52, 0.56));
    let wall_uniforms = scene
        .wall_models
        .map(|model| make_uniforms(model, Vec3::new(0.28, 0.32, 0.42)));
    let occluder_uniforms = scene
        .occluder_models
        .map(|model| make_uniforms(model, Vec3::new(0.75, 0.18, 0.06)));
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 8, 10, 16));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &scene.floor, &floor_uniforms);
        for uniforms in &wall_uniforms {
            frame.draw_mesh(target, &scene.wall, uniforms);
        }
        for uniforms in &occluder_uniforms {
            frame.draw_mesh(target, &scene.occluder, uniforms);
        }
    });
    framebuffer
}

#[test]
fn point_shadow_demo_golden() {
    assert_golden("m25-point-shadow", &render_point_shadow_scene());
}
