use chimy2::camera::OrbitController;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::gltf::{GltfAlphaMode, GltfAsset, GltfDraw, GltfMaterial, submit_gltf_draws};
use chimy2::image::Texture;
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::mesh::{Mesh, MeshVertex};
use std::fs;
use std::path::Path;

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

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
    submit_gltf_draws(
        &mut framebuffer,
        &asset,
        &draws,
        camera.view_matrix(),
        camera.projection_matrix(),
        camera.position,
    )
    .unwrap();
    framebuffer
}

fn assert_golden(name: &str, animation: Option<usize>, time: f32) {
    let framebuffer = render(animation, time);
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name);
    let actual = ppm(&framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, actual).unwrap();
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
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

fn render_base_color(factor: Vec4) -> Framebuffer {
    let mesh = Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(-0.8, -0.8, 0.0), None, None),
            MeshVertex::new(Vec3::new(0.8, -0.8, 0.0), None, None),
            MeshVertex::new(Vec3::new(0.0, 0.8, 0.0), None, None),
        ],
        vec![[0, 1, 2]],
    );
    let asset = GltfAsset {
        meshes: Vec::new(),
        nodes: Vec::new(),
        scenes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        materials: vec![GltfMaterial {
            name: String::new(),
            base_color_factor: factor,
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            albedo_texture: None,
            normal_map_texture: None,
            alpha_mode: GltfAlphaMode::Opaque,
            alpha_cutoff: 0.5,
        }],
        default_scene: 0,
    };
    let draw = GltfDraw {
        mesh,
        model: Mat4::IDENTITY,
        material: Some(0),
    };
    let mut framebuffer = Framebuffer::new(32, 32);
    submit_gltf_draws(
        &mut framebuffer,
        &asset,
        &[draw],
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Vec3::new(0.0, 0.0, 1.0),
    )
    .unwrap();
    framebuffer
}

#[test]
fn gltf_material_base_color_reaches_rendering() {
    let red = render_base_color(Vec4::new(1.0, 0.0, 0.0, 1.0));
    let blue = render_base_color(Vec4::new(0.0, 0.0, 1.0, 1.0));
    assert_ne!(red.color, blue.color);
}
