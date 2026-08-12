use chimy2::demo::{build_morph_scene, render_morph_scene};
use chimy2::fb::Framebuffer;
use chimy2::gltf::{MorphTarget, blend_morph_targets};
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{MeshShader, MeshUniforms};
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

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join("m34-morph-targets.ppm")
}

#[test]
fn fixed_morph_scene_is_deterministic_and_matches_golden() {
    let scene = build_morph_scene(WIDTH as f32 / HEIGHT as f32);
    let mut first = Framebuffer::new(WIDTH, HEIGHT);
    render_morph_scene(&mut first, &scene, [0.35, 0.7]);
    let mut second = Framebuffer::new(WIDTH, HEIGHT);
    render_morph_scene(&mut second, &scene, [0.35, 0.7]);
    assert_eq!(first.color, second.color);
    let path = golden_path();
    let actual = ppm(&first);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).expect("write golden");
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    assert_eq!(fs::read(&path).expect("read golden"), actual);
}

#[test]
fn morphed_bounds_keep_culling_on_and_off_identical() {
    let base = Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(1.2, -0.4, -2.0), None, None),
            MeshVertex::new(Vec3::new(1.4, -0.4, -2.0), None, None),
            MeshVertex::new(Vec3::new(1.3, 0.4, -2.0), None, None),
        ],
        vec![[0, 1, 2]],
    );
    let target = MorphTarget::new(
        vec![
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
        ],
        None,
    )
    .unwrap();
    let morphed = blend_morph_targets(&base, &[target], &[1.0]).unwrap();
    let projection = Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 5.0);
    let uniforms = MeshUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, projection, 0xffe07030);
    let mut culling_on = Framebuffer::new(32, 32);
    let mut pipeline_on = Pipeline::new(MeshShader, MeshShader);
    pipeline_on.render(&mut culling_on, |frame, target| {
        frame.draw_mesh(target, &morphed, &uniforms);
    });
    let mut culling_off = Framebuffer::new(32, 32);
    let mut pipeline_off = Pipeline::new(MeshShader, MeshShader);
    pipeline_off.set_culling_enabled(false);
    pipeline_off.render(&mut culling_off, |frame, target| {
        frame.draw_mesh(target, &morphed, &uniforms);
    });
    assert_eq!(culling_on.color, culling_off.color);
    assert!(culling_on.color.iter().any(|&pixel| pixel != 0));
}
