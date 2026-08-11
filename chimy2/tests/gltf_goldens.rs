use chimy2::camera::OrbitController;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::gltf::GltfAsset;
use chimy2::image::Texture;
use chimy2::math::Vec3;
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{MeshShader, MeshUniforms};
use std::path::Path;

fn render(animation: Option<usize>, time: f32) -> Framebuffer {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/arm.gltf");
    let asset = GltfAsset::load(path).unwrap();
    let draws = asset
        .scene_draws(asset.default_scene, animation, time)
        .unwrap();
    let mut orbit = OrbitController::new(Vec3::new(0.5, 0.5, 0.0), 3.5, 0.0, 0.0);
    orbit.step(0.2 / 60.0, 0.0, 0.0);
    let camera = orbit.camera(1.0, 1.0, 0.1, 100.0);
    let mut framebuffer = Framebuffer::new(64, 64);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    let uniforms: Vec<_> = draws
        .iter()
        .map(|draw| {
            MeshUniforms::new(
                draw.model,
                camera.view_matrix(),
                camera.projection_matrix(),
                argb8888(255, 220, 160, 70),
            )
        })
        .collect();
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        for (draw, uniforms) in draws.iter().zip(&uniforms) {
            frame.draw_mesh(target, &draw.mesh, uniforms);
        }
    });
    framebuffer
}

fn assert_golden(name: &str, animation: Option<usize>, time: f32) {
    let framebuffer = render(animation, time);
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name);
    let bytes = std::fs::read(path).unwrap();
    let golden = Texture::from_ppm(&bytes).unwrap();
    assert_eq!((golden.width(), golden.height()), (64, 64));
    for (index, pixel) in golden.pixels().iter().enumerate() {
        let [_, red, green, blue] = framebuffer.color[index].to_be_bytes();
        assert_eq!(*pixel, [red, green, blue, 255], "pixel {index}");
    }
}

#[test]
fn gltf_static_mesh_golden() {
    assert_golden("gltf-static.ppm", None, 0.0);
}

#[test]
fn gltf_arm_bind_pose_golden() {
    assert_golden("gltf-arm-t0.ppm", Some(0), 0.0);
}

#[test]
fn gltf_arm_animated_pose_golden() {
    assert_golden("gltf-arm-t1.ppm", Some(0), 1.0);
}
