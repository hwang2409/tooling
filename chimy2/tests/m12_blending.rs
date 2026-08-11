use chimy2::fb::{Framebuffer, argb8888, blend_argb8888_linear};
use chimy2::math::{Mat4, Vec4};
use chimy2::pipeline::{ColorVarying, Pipeline, VertexOutput, fragment_stage, vertex_stage};
use chimy2::shaders::{FlatColorShader, FlatColorUniforms};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 16;
const HEIGHT: usize = 12;

fn quad_vertices(z: f32) -> [Vec4; 4] {
    [
        Vec4::new(-0.85, -0.8, z, 1.0),
        Vec4::new(0.85, -0.8, z, 1.0),
        Vec4::new(0.85, 0.8, z, 1.0),
        Vec4::new(-0.85, 0.8, z, 1.0),
    ]
}

fn transparent_scene(thread_count: usize, reverse_sort: bool, depth_write: bool) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 64, 64, 64));
    let far_vertices = quad_vertices(0.3);
    let near_vertices = quad_vertices(-0.3);
    let triangles = [[0, 1, 2]];
    let mut far = FlatColorUniforms::new(Mat4::IDENTITY, argb8888(255, 235, 70, 40));
    far.set_alpha(0.5);
    let mut near = FlatColorUniforms::new(Mat4::IDENTITY, argb8888(255, 40, 90, 235));
    near.set_alpha(0.5);

    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    pipeline.set_thread_count(thread_count);
    let draw_far = |pipeline: &mut Pipeline<FlatColorShader, FlatColorShader>,
                    framebuffer: &mut Framebuffer| {
        if depth_write {
            pipeline.draw(framebuffer, &far_vertices, &triangles, &far);
        } else {
            pipeline.draw_transparent(framebuffer, &far_vertices, &triangles, &[0.3], &far);
        }
    };
    let draw_near = |pipeline: &mut Pipeline<FlatColorShader, FlatColorShader>,
                     framebuffer: &mut Framebuffer| {
        if depth_write {
            pipeline.draw(framebuffer, &near_vertices, &triangles, &near);
        } else {
            pipeline.draw_transparent(framebuffer, &near_vertices, &triangles, &[-0.3], &near);
        }
    };
    if reverse_sort {
        draw_near(&mut pipeline, &mut framebuffer);
        draw_far(&mut pipeline, &mut framebuffer);
    } else {
        draw_far(&mut pipeline, &mut framebuffer);
        draw_near(&mut pipeline, &mut framebuffer);
    }
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

fn sorted_batch(reverse: bool, thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 64, 64, 64));
    let vertices = [
        (
            Vec4::new(-0.85, -0.8, 0.0, 1.0),
            Vec4::new(0.9, 0.1, 0.05, 0.5),
        ),
        (
            Vec4::new(0.85, -0.8, 0.0, 1.0),
            Vec4::new(0.9, 0.1, 0.05, 0.5),
        ),
        (
            Vec4::new(0.0, 0.8, 0.0, 1.0),
            Vec4::new(0.9, 0.1, 0.05, 0.5),
        ),
        (
            Vec4::new(-0.85, -0.8, 0.0, 1.0),
            Vec4::new(0.05, 0.2, 0.9, 0.5),
        ),
        (
            Vec4::new(0.85, -0.8, 0.0, 1.0),
            Vec4::new(0.05, 0.2, 0.9, 0.5),
        ),
        (
            Vec4::new(0.0, 0.8, 0.0, 1.0),
            Vec4::new(0.05, 0.2, 0.9, 0.5),
        ),
    ];
    let mut pipeline = Pipeline::new(
        vertex_stage(|vertex: &(Vec4, Vec4), _: &()| {
            VertexOutput::new(vertex.0, ColorVarying::new(vertex.1))
        }),
        fragment_stage(|varyings: &ColorVarying, _: &()| {
            let color = varyings.color;
            chimy2::fb::argb8888_linear(color.w, [color.x, color.y, color.z])
        }),
    );
    pipeline.set_thread_count(thread_count);
    pipeline.draw_transparent(
        &mut framebuffer,
        &vertices,
        &[[0, 1, 2], [3, 4, 5]],
        if reverse { &[-0.3, 0.3] } else { &[0.3, -0.3] },
        &(),
    );
    framebuffer
}

#[test]
fn transparent_overlap_is_sorted_and_serial_parallel_identical() {
    assert_eq!(sorted_batch(false, 1), sorted_batch(false, 4));
    assert_ne!(sorted_batch(false, 1), sorted_batch(true, 1));
    let serial = transparent_scene(1, false, false);
    let parallel = transparent_scene(4, false, false);
    assert_eq!(serial, parallel);
    assert_ne!(serial, transparent_scene(1, true, false));
    assert_ne!(serial, transparent_scene(1, false, true));
    let center = serial.color[HEIGHT / 2 * WIDTH + WIDTH / 2];
    let far_over_background =
        blend_argb8888_linear(argb8888(255, 64, 64, 64), argb8888(128, 235, 70, 40));
    let expected = blend_argb8888_linear(far_over_background, argb8888(128, 40, 90, 235));
    assert_eq!(center, expected);
    assert_golden("m12-transparent-overlap", &serial);
}

fn render_edge(scale: usize, thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(12, 12);
    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    pipeline.set_ssaa_scale(scale);
    pipeline.set_thread_count(thread_count);
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
            &FlatColorUniforms::new(Mat4::IDENTITY, argb8888(255, 240, 240, 240)),
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
