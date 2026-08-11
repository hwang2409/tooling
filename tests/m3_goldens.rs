use chimy2::camera::Camera;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{MeshShader, MeshUniforms};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 32;
const HEIGHT: usize = 24;

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
        panic!(
            "regenerated {}, rerun without GOLDEN_REGEN to compare",
            path.display()
        );
    }
    let expected = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing golden {}: {error}; run GOLDEN_REGEN=1 cargo test --test m3_goldens",
            path.display()
        )
    });
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

fn draw_mesh(mesh: &Mesh, camera: Camera, model: Mat4, color: u32) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    let uniforms = MeshUniforms::new(
        model,
        camera.view_matrix(),
        camera.projection_matrix(),
        color,
    );
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.draw_mesh(&mut framebuffer, mesh, &uniforms);
    framebuffer
}

fn draw_identity_mesh(mesh: &Mesh, color: u32) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    let uniforms = MeshUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, color);
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.draw_mesh(&mut framebuffer, mesh, &uniforms);
    framebuffer
}

#[test]
fn cube_mesh_fixed_camera_golden() {
    let mesh = Mesh::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cube.obj")).unwrap();
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    let framebuffer = draw_mesh(
        &mesh,
        camera,
        Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.35),
        argb8888(255, 225, 160, 70),
    );
    assert_golden("m3-cube-camera", &framebuffer);
}

#[test]
fn near_plane_clipping_golden() {
    let mesh = Mesh::parse("v -0.8 -0.7 -0.25\nv 0.8 -0.7 -0.7\nv 0 0.8 -0.7\nf 1 2 3\n").unwrap();
    let camera = Camera::new(
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.5,
        10.0,
    );
    let framebuffer = draw_mesh(&mesh, camera, Mat4::IDENTITY, argb8888(255, 80, 200, 120));
    assert!(framebuffer.color.iter().any(|&pixel| pixel != 0xff0c1018));
    assert_golden("m3-near-plane-clipping", &framebuffer);
}

#[test]
fn backface_culling_golden() {
    let front = Mesh::parse("v -0.8 -0.7 0\nv 0.8 -0.7 0\nv 0 0.8 0\nf 1 2 3\n").unwrap();
    let back = Mesh::parse("v -0.8 -0.7 -0.1\nv 0 0.8 -0.1\nv 0.8 -0.7 -0.1\nf 1 2 3\n").unwrap();
    let mut framebuffer = draw_identity_mesh(&front, argb8888(255, 225, 60, 60));
    let uniforms = MeshUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        argb8888(255, 60, 80, 225),
    );
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.draw_mesh(&mut framebuffer, &back, &uniforms);
    assert!(framebuffer.color.iter().all(|&pixel| pixel != 0xff3c50e1));
    assert_golden("m3-backface-culling", &framebuffer);
}
