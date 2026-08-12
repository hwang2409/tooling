use chimy2::cascaded_shadow_scene::build_cascaded_shadow_scene;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use chimy2::shadow::render_cascade_shadow_maps;
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

fn render_csm_scene() -> Framebuffer {
    let scene = build_cascaded_shadow_scene(WIDTH as f32 / HEIGHT as f32);
    let mut shadow_meshes = vec![(&scene.ground, Mat4::IDENTITY)];
    shadow_meshes.extend(
        scene
            .object_models
            .iter()
            .map(|&model| (&scene.object, model)),
    );
    let cascades =
        render_cascade_shadow_maps(scene.camera, scene.light_direction, 3, 96, &shadow_meshes)
            .unwrap();
    let directional = DirectionalLight::new(scene.light_direction, Vec3::new(1.0, 0.93, 0.82));
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let mut ground_uniforms = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        scene.camera.view_matrix(),
        scene.camera.projection_matrix(),
        Vec3::new(0.02, 0.025, 0.035),
        Vec3::new(0.52, 0.55, 0.58),
        Vec3::new(0.25, 0.25, 0.25),
        24.0,
        scene.camera.position,
        directional,
        point,
    );
    ground_uniforms.set_directional_cascaded_shadow(directional, cascades.clone());
    let object_uniforms = scene.object_models.map(|model| {
        let mut uniforms = BlinnPhongUniforms::new(
            model,
            scene.camera.view_matrix(),
            scene.camera.projection_matrix(),
            Vec3::new(0.02, 0.025, 0.035),
            Vec3::new(0.78, 0.24, 0.07),
            Vec3::new(0.25, 0.25, 0.25),
            24.0,
            scene.camera.position,
            directional,
            point,
        );
        uniforms.set_directional_cascaded_shadow(directional, cascades.clone());
        uniforms
    });
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 11, 15, 24));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &scene.ground, &ground_uniforms);
        for uniforms in &object_uniforms {
            frame.draw_mesh(target, &scene.object, uniforms);
        }
    });
    framebuffer
}

#[test]
fn cascaded_shadow_demo_golden() {
    assert_golden("csm-directional-shadow", &render_csm_scene());
}
