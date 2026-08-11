use chimy2::fb::{Framebuffer, argb8888, blend_argb8888_linear};
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{FlatColorShader, FlatColorUniforms, MeshShader, MeshUniforms};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 130;
const HEIGHT: usize = 12;

fn quad_mesh(z: f32) -> Mesh {
    Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(-0.85, -0.8, z), None, None),
            MeshVertex::new(Vec3::new(0.85, -0.8, z), None, None),
            MeshVertex::new(Vec3::new(0.85, 0.8, z), None, None),
            MeshVertex::new(Vec3::new(-0.85, 0.8, z), None, None),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )
}

fn transparent_scene(thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 64, 64, 64));
    let far_mesh = quad_mesh(-0.3);
    let near_mesh = quad_mesh(0.3);
    let mut far = MeshUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        argb8888(255, 235, 70, 40),
    );
    far.set_alpha(0.5);
    let mut near = MeshUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        argb8888(255, 40, 90, 235),
    );
    near.set_alpha(0.5);

    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.set_thread_count(thread_count);
    pipeline.render(&mut framebuffer, |frame, target| {
        target.clear(argb8888(255, 64, 64, 64));
        frame.draw_mesh(target, &near_mesh, &near);
        frame.draw_mesh(target, &far_mesh, &far);
    });
    framebuffer
}

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
    assert_eq!(fs::read(&path).expect("read golden"), actual, "{name}");
}

#[test]
fn transparent_overlap_is_sorted_and_serial_parallel_identical() {
    let serial = transparent_scene(1);
    for thread_count in [2, 4, 8] {
        assert_eq!(serial, transparent_scene(thread_count));
    }
    let center = serial.color[HEIGHT / 2 * WIDTH + WIDTH / 2];
    let far_over_background =
        blend_argb8888_linear(argb8888(255, 64, 64, 64), argb8888(128, 235, 70, 40));
    let expected = blend_argb8888_linear(far_over_background, argb8888(128, 40, 90, 235));
    assert_eq!(center, expected);
    for x in [16, 64, 113] {
        assert_eq!(serial.color[HEIGHT / 2 * WIDTH + x], expected);
    }
    assert_golden("m12-transparent-overlap", &serial);
}

#[test]
fn alpha_one_draw_is_opaque_and_writes_depth() {
    let mesh = quad_mesh(-0.3);
    let uniforms = MeshUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        argb8888(255, 235, 70, 40),
    );
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.set_thread_count(4);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &mesh, &uniforms);
    });
    assert!(framebuffer.depth.iter().any(|&depth| depth < 1.0));
}

fn render_edge(scale: usize, thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(12, 12);
    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    pipeline.set_ssaa_scale(scale);
    pipeline.set_thread_count(thread_count);
    let uniforms = FlatColorUniforms::new(Mat4::IDENTITY, argb8888(255, 240, 240, 240));
    pipeline.render(&mut framebuffer, |pipeline, target| {
        target.clear(argb8888(255, 20, 20, 20));
        pipeline.draw(
            target,
            &[
                Vec4::new(-1.0, -1.0, 0.0, 1.0),
                Vec4::new(0.85, -1.0, 0.0, 1.0),
                Vec4::new(-1.0, 0.85, 0.0, 1.0),
            ],
            &[[0, 1, 2]],
            &uniforms,
        );
    });
    framebuffer
}

#[test]
fn ssaa_changes_edges_and_serial_parallel_is_identical() {
    let aliased = render_edge(1, 1);
    let smooth = render_edge(2, 1);
    assert_eq!(smooth, render_edge(2, 4));
    assert_ne!(aliased.color, smooth.color);
    assert_golden("m12-ssaa-edge", &smooth);
}
