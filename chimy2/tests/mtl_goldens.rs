use chimy2::camera::Camera;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::{ColorSpace, Texture};
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::mesh::{Mesh, ObjAsset};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, NormalMappedBlinnPhongShader,
    NormalMappedBlinnPhongUniforms, PointLight, TextureFilter, TexturedBlinnPhongShader,
    TexturedBlinnPhongUniforms,
};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;

fn asset() -> ObjAsset {
    Mesh::load_with_materials(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/multi_material.obj"),
    )
    .unwrap()
}

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/mtl-multi-material.ppm")
}

fn assert_golden(framebuffer: &Framebuffer) {
    let path = golden_path();
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).unwrap();
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    assert_eq!(fs::read(&path).unwrap(), actual, "{}", path.display());
}

fn render(swap_diffuse: bool, wrong_albedo_space: bool, flatten_groups: bool) -> Framebuffer {
    let asset = asset();
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let wrong_albedo = wrong_albedo_space.then(|| {
        Texture::load_with_color_space(assets.join("checker.qoi"), ColorSpace::Linear).unwrap()
    });
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        0.9,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 10, 14, 22));

    for group in asset.mesh.material_groups() {
        let selected_name = if flatten_groups {
            "warm"
        } else {
            group.material_name()
        };
        let source = asset.materials.get(selected_name).unwrap();
        let mut material = source.clone();
        if swap_diffuse {
            material.diffuse = if selected_name == "warm" {
                Vec3::new(1.0, 1.0, 1.0)
            } else {
                Vec3::new(0.85, 0.12, 0.05)
            };
        }
        let group_mesh = asset.mesh.submesh(group.triangle_range()).unwrap();
        let mut lighting = BlinnPhongUniforms::new(
            Mat4::IDENTITY,
            camera.view_matrix(),
            camera.projection_matrix(),
            Vec3::new(0.03, 0.03, 0.03),
            Vec3::new(0.8, 0.8, 0.8),
            Vec3::new(0.3, 0.3, 0.3),
            16.0,
            camera.position,
            DirectionalLight::new(
                Vec3::new(-0.3, 0.4, 1.0).normalize(),
                Vec3::new(1.0, 0.95, 0.9),
            ),
            PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
        );
        material.apply_to(&mut lighting);
        if let Some(texture) = material.albedo_texture() {
            let texture = wrong_albedo.as_ref().unwrap_or(texture);
            if let Some(normal_map) = material.normal_map_texture() {
                let uniforms = NormalMappedBlinnPhongUniforms::new(
                    lighting,
                    texture,
                    normal_map,
                    TextureFilter::Bilinear,
                )
                .unwrap();
                let mut pipeline =
                    Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
                pipeline.draw_mesh_with_sampling(&mut framebuffer, &group_mesh, &uniforms);
            } else {
                let uniforms =
                    TexturedBlinnPhongUniforms::new(lighting, texture, TextureFilter::Bilinear);
                let mut pipeline =
                    Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
                pipeline.draw_mesh_with_sampling(&mut framebuffer, &group_mesh, &uniforms);
            }
        } else {
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            pipeline.draw_mesh(&mut framebuffer, &group_mesh, &lighting);
        }
    }
    framebuffer
}

#[test]
fn multi_material_scene_golden() {
    assert_golden(&render(false, false, false));
}

#[test]
fn material_mutation_gates_change_the_production_render() {
    let baseline = render(false, false, false);
    assert_ne!(baseline.color, render(true, false, false).color);
    assert_ne!(baseline.color, render(false, true, false).color);
    assert_ne!(baseline.color, render(false, false, true).color);
}
