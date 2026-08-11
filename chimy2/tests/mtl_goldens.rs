use chimy2::camera::Camera;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::{ColorSpace, Texture};
use chimy2::material::{Material, MaterialLibrary};
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

fn render_material_groups(
    groups: &[(Mesh, Material)],
    wrong_albedo: Option<&Texture>,
) -> Framebuffer {
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        0.9,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    let fallback_albedo = Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap();
    let fallback_normal =
        Texture::new_with_color_space(1, 1, vec![[128, 128, 255, 255]], ColorSpace::Linear)
            .unwrap();
    let mut uniforms = Vec::with_capacity(groups.len());

    for (_, material) in groups {
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
        let texture = match (material.albedo_texture(), wrong_albedo) {
            (Some(_), Some(texture)) => texture,
            (Some(texture), None) => texture,
            (None, _) => &fallback_albedo,
        };
        let normal_map = material.normal_map_texture().unwrap_or(&fallback_normal);
        uniforms.push(
            NormalMappedBlinnPhongUniforms::new(
                lighting,
                texture,
                normal_map,
                TextureFilter::Bilinear,
            )
            .unwrap(),
        );
    }

    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 10, 14, 22));
    let mut pipeline = Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        for ((group_mesh, _), uniforms) in groups.iter().zip(uniforms.iter()) {
            frame.draw_mesh_with_sampling(target, group_mesh, uniforms);
        }
    });
    framebuffer
}

fn render(swap_diffuse: bool, wrong_albedo_space: bool, flatten_groups: bool) -> Framebuffer {
    let asset = asset();
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let wrong_albedo = wrong_albedo_space.then(|| {
        Texture::load_with_color_space(assets.join("checker.qoi"), ColorSpace::Linear).unwrap()
    });
    let mut groups = Vec::new();
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
        groups.push((group_mesh, material));
    }
    render_material_groups(&groups, wrong_albedo.as_ref())
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

fn alpha_mesh() -> Mesh {
    Mesh::parse(
        "v -0.8 -0.8 0\nv 0.8 -0.8 0\nv 0.8 0.8 0\nv -0.8 0.8 0\n\
         vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
         f 1/1 2/2 3/3 4/4\n",
    )
    .unwrap()
}

fn alpha_material() -> Material {
    MaterialLibrary::parse("newmtl alpha\nKa 1 1 1\nKd 1 1 1\nd 0.5\n")
        .unwrap()
        .get("alpha")
        .unwrap()
        .clone()
}

fn alpha_lighting(material: &Material) -> BlinnPhongUniforms {
    let mut uniforms = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::ZERO,
        0.0,
        Vec3::new(0.0, 0.0, 2.0),
        DirectionalLight::new(Vec3::ZERO, Vec3::ZERO),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    );
    material.apply_to(&mut uniforms);
    uniforms
}

fn rendered_alpha(framebuffer: &Framebuffer) -> u8 {
    framebuffer.color[framebuffer.height / 2 * framebuffer.width + framebuffer.width / 2]
        .to_be_bytes()[0]
}

#[test]
fn material_alpha_reaches_plain_textured_and_normal_mapped_shaders() {
    let material = alpha_material();
    let mesh = alpha_mesh();
    let albedo = Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap();
    let normal_map =
        Texture::new_with_color_space(1, 1, vec![[128, 128, 255, 255]], ColorSpace::Linear)
            .unwrap();

    let mut plain_framebuffer = Framebuffer::new(16, 16);
    let plain_uniforms = alpha_lighting(&material);
    let mut plain_pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    plain_pipeline.render(&mut plain_framebuffer, |frame, target| {
        frame.draw_mesh(target, &mesh, &plain_uniforms);
    });

    let mut textured_framebuffer = Framebuffer::new(16, 16);
    let textured_uniforms =
        TexturedBlinnPhongUniforms::new(alpha_lighting(&material), &albedo, TextureFilter::Nearest);
    let mut textured_pipeline = Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
    textured_pipeline.render(&mut textured_framebuffer, |frame, target| {
        frame.draw_mesh_with_sampling(target, &mesh, &textured_uniforms);
    });

    let mut normal_framebuffer = Framebuffer::new(16, 16);
    let normal_uniforms = NormalMappedBlinnPhongUniforms::new(
        alpha_lighting(&material),
        &albedo,
        &normal_map,
        TextureFilter::Nearest,
    )
    .unwrap();
    let mut normal_pipeline =
        Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
    normal_pipeline.render(&mut normal_framebuffer, |frame, target| {
        frame.draw_mesh_with_sampling(target, &mesh, &normal_uniforms);
    });

    assert_eq!(rendered_alpha(&plain_framebuffer), 128);
    assert_eq!(rendered_alpha(&textured_framebuffer), 128);
    assert_eq!(rendered_alpha(&normal_framebuffer), 128);
}

#[test]
fn overlapping_transparent_material_groups_use_one_global_flush() {
    let far = MaterialLibrary::parse("newmtl far\nKa 0.03 0.01 0.01\nKd 0.85 0.12 0.05\nd 0.5\n")
        .unwrap()
        .get("far")
        .unwrap()
        .clone();
    let near = MaterialLibrary::parse("newmtl near\nKa 0.02 0.02 0.02\nKd 0.05 0.35 0.85\nd 0.5\n")
        .unwrap()
        .get("near")
        .unwrap()
        .clone();
    let mut groups = vec![
        (
            Mesh::new(
                vec![
                    chimy2::mesh::MeshVertex::new(Vec3::new(-0.8, -0.8, -0.3), None, None),
                    chimy2::mesh::MeshVertex::new(Vec3::new(0.8, -0.8, -0.3), None, None),
                    chimy2::mesh::MeshVertex::new(Vec3::new(0.8, 0.8, -0.3), None, None),
                    chimy2::mesh::MeshVertex::new(Vec3::new(-0.8, 0.8, -0.3), None, None),
                ],
                vec![[0, 1, 2], [0, 2, 3]],
            ),
            far,
        ),
        (
            Mesh::new(
                vec![
                    chimy2::mesh::MeshVertex::new(Vec3::new(-0.8, -0.8, 0.3), None, None),
                    chimy2::mesh::MeshVertex::new(Vec3::new(0.8, -0.8, 0.3), None, None),
                    chimy2::mesh::MeshVertex::new(Vec3::new(0.8, 0.8, 0.3), None, None),
                    chimy2::mesh::MeshVertex::new(Vec3::new(-0.8, 0.8, 0.3), None, None),
                ],
                vec![[0, 1, 2], [0, 2, 3]],
            ),
            near,
        ),
    ];
    groups.reverse();
    let framebuffer = render_material_groups(&groups, None);
    let center = framebuffer.color[HEIGHT / 2 * WIDTH + WIDTH / 2];
    // One global classified queue renders the far material before the near one.
    assert_eq!(center, 0xff6f3c89);
}
