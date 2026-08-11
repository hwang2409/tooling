use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec4};
use chimy2::pipeline::Pipeline;
use chimy2::raster::{ScreenVertex, rasterize_triangle};
use chimy2::shaders::{FlatColorShader, FlatColorUniforms};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 8;
const HEIGHT: usize = 8;

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
            "missing golden {}: {error}; run GOLDEN_REGEN=1 cargo test",
            path.display()
        )
    });
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

#[test]
fn ci_does_not_enable_golden_regeneration() {
    if std::env::var_os("CI").is_some() {
        assert!(
            std::env::var_os("GOLDEN_REGEN").is_none(),
            "GOLDEN_REGEN must not be set in CI"
        );
    }
}

fn render(vertices: &[Vec4], triangles: &[[usize; 3]], colors_and_z: &[(u32, f32)]) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    for (triangle, &(color, z)) in triangles.iter().zip(colors_and_z) {
        let transformed: Vec<Vec4> = triangle
            .iter()
            .map(|&index| {
                let mut vertex = vertices[index];
                vertex.z = z;
                vertex
            })
            .collect();
        pipeline.draw(
            &mut framebuffer,
            &transformed,
            &[[0, 1, 2]],
            &FlatColorUniforms::new(Mat4::IDENTITY, color),
        );
    }
    framebuffer
}

#[test]
fn single_triangle_golden() {
    let framebuffer = render(
        &[
            Vec4::new(-0.75, -0.7, 0.0, 1.0),
            Vec4::new(0.75, -0.7, 0.0, 1.0),
            Vec4::new(0.0, 0.75, 0.0, 1.0),
        ],
        &[[0, 1, 2]],
        &[(argb8888(255, 225, 60, 60), 0.0)],
    );
    assert_golden("single-triangle", &framebuffer);
}

#[test]
fn overlapping_triangles_use_depth_not_painter_order() {
    let vertices = [
        Vec4::new(-0.85, -0.8, 0.0, 1.0),
        Vec4::new(0.85, -0.8, 0.0, 1.0),
        Vec4::new(0.0, 0.85, 0.0, 1.0),
    ];
    let near = render(
        &vertices,
        &[[0, 1, 2], [0, 1, 2]],
        &[
            (argb8888(255, 235, 70, 70), -0.5),
            (argb8888(255, 70, 90, 235), 0.5),
        ],
    );
    let painter_order = render(
        &vertices,
        &[[0, 1, 2], [0, 1, 2]],
        &[
            (argb8888(255, 235, 70, 70), 0.5),
            (argb8888(255, 70, 90, 235), -0.5),
        ],
    );
    assert_ne!(near.color, painter_order.color);
    assert_golden("overlapping-triangles-depth", &near);
}

#[test]
fn shared_edge_golden_has_exactly_one_coverage_per_pixel() {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let mut calls = 0;
    let red = [
        ScreenVertex::new(chimy2::math::Vec3::new(0.0, 0.0, 0.0), ()),
        ScreenVertex::new(chimy2::math::Vec3::new(WIDTH as f32, 0.0, 0.0), ()),
        ScreenVertex::new(chimy2::math::Vec3::new(0.0, HEIGHT as f32, 0.0), ()),
    ];
    let blue = [
        ScreenVertex::new(chimy2::math::Vec3::new(WIDTH as f32, 0.0, -0.1), ()),
        ScreenVertex::new(
            chimy2::math::Vec3::new(WIDTH as f32, HEIGHT as f32, -0.1),
            (),
        ),
        ScreenVertex::new(chimy2::math::Vec3::new(0.0, HEIGHT as f32, -0.1), ()),
    ];
    rasterize_triangle(&mut framebuffer, red, |_| {
        calls += 1;
        argb8888(255, 235, 70, 70)
    });
    rasterize_triangle(&mut framebuffer, blue, |_| {
        calls += 1;
        argb8888(255, 70, 90, 235)
    });
    assert_eq!(calls, WIDTH * HEIGHT);
    assert!(framebuffer.depth.iter().all(|&depth| depth < 1.0));
    assert_golden("shared-edge", &framebuffer);
}
