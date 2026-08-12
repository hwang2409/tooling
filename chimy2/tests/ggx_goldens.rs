use chimy2::camera::Camera;
use chimy2::demo::{cube_with_uvs, plane_xz, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::gltf::{GltfAlphaMode, GltfAsset, GltfDraw, GltfMaterial, submit_gltf_draws};
use chimy2::image::{ColorSpace, Texture, WrapMode};
use chimy2::math::{Mat4, Quat, Vec3, Vec4};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{
    BlinnPhongUniforms, CookTorranceShader, CookTorranceUniforms, DirectionalLight,
    NormalMappedCookTorranceShader, NormalMappedCookTorranceUniforms, PointLight, TextureFilter,
};
use chimy2::shadow::{
    ShadowDepthShader, ShadowDepthUniforms, ShadowMap, ShadowState, directional_light_view,
};
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
        .join("tests/goldens")
        .join(format!("{name}.ppm"))
}

fn assert_golden(name: &str, framebuffer: &Framebuffer) {
    let path = golden_path(name);
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).unwrap();
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    assert_eq!(fs::read(path).unwrap(), actual, "{name}");
}

fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 7.0),
        Quat::IDENTITY,
        0.9,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        30.0,
    )
}

fn ggx_uniforms(
    model: Mat4,
    camera: &Camera,
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
) -> CookTorranceUniforms {
    let lighting = BlinnPhongUniforms::new_with_linear_colors(
        model,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.025, 0.03, 0.05),
        Vec3::ZERO,
        Vec3::ZERO,
        0.0,
        camera.position,
        DirectionalLight::new(
            Vec3::new(-0.45, 0.75, 1.0).normalize(),
            Vec3::new(1.0, 0.95, 0.9),
        ),
        PointLight::new(
            Vec3::new(2.0, 2.5, 3.0),
            Vec3::new(0.35, 0.5, 0.8),
            1.0,
            0.08,
            0.02,
        ),
    );
    CookTorranceUniforms::new_with_linear_base_color(lighting, base_color, metallic, roughness)
}

fn render_spheres(values: &[(f32, f32)]) -> Framebuffer {
    let camera = camera();
    let sphere = uv_sphere(0.82, 24, 48);
    let uniforms: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(index, &(metallic, roughness))| {
            let x = (index as f32 - (values.len() as f32 - 1.0) * 0.5) * 1.8;
            ggx_uniforms(
                Mat4::translate(Vec3::new(x, 0.0, 0.0)),
                &camera,
                Vec3::new(0.82, 0.16, 0.06),
                metallic,
                roughness,
            )
        })
        .collect();
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 8, 10, 16));
    let mut pipeline = Pipeline::new(CookTorranceShader, CookTorranceShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        for uniform in &uniforms {
            frame.draw_mesh(target, &sphere, uniform);
        }
    });
    framebuffer
}

#[test]
fn ggx_roughness_sweep_golden() {
    assert_golden(
        "ggx-roughness-sweep",
        &render_spheres(&[(0.0, 0.1), (0.0, 0.5), (0.0, 0.9)]),
    );
}

#[test]
fn ggx_metal_dielectric_pair_golden() {
    assert_golden(
        "ggx-metal-dielectric",
        &render_spheres(&[(0.0, 0.35), (1.0, 0.35)]),
    );
}

#[test]
fn gltf_metallic_roughness_golden() {
    let material = GltfMaterial {
        name: "ggx-test".to_string(),
        base_color_factor: Vec4::new(0.82, 0.16, 0.06, 1.0),
        metallic_factor: 0.85,
        roughness_factor: 0.25,
        albedo_texture: None,
        normal_map_texture: None,
        alpha_mode: GltfAlphaMode::Opaque,
        alpha_cutoff: 0.5,
    };
    let mesh = Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(-1.2, -1.0, 0.0), None, None),
            MeshVertex::new(Vec3::new(1.2, -1.0, 0.0), None, None),
            MeshVertex::new(Vec3::new(0.0, 1.2, 0.0), None, None),
        ],
        vec![[0, 1, 2]],
    );
    let asset = GltfAsset {
        meshes: Vec::new(),
        nodes: Vec::new(),
        scenes: Vec::new(),
        skins: Vec::new(),
        animations: Vec::new(),
        materials: vec![material],
        default_scene: 0,
    };
    let camera = camera();
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 8, 10, 16));
    submit_gltf_draws(
        &mut framebuffer,
        &asset,
        &[GltfDraw {
            mesh,
            model: Mat4::IDENTITY,
            material: Some(0),
        }],
        camera.view_matrix(),
        camera.projection_matrix(),
        camera.position,
    )
    .unwrap();
    assert_golden("gltf-ggx-metallic-roughness", &framebuffer);
}

#[test]
fn ggx_normal_map_shadow_integration_golden() {
    let mut caster = cube_with_uvs(0.8);
    caster.generate_tangents();
    let ground = plane_xz(8.0, 8.0, 8, 1.0);
    let caster_model = Mat4::translate(Vec3::new(0.0, 0.8, 0.0));
    let target = Vec3::new(0.0, 0.7, 0.0);
    let light_direction = Vec3::new(0.7, 1.0, 0.35).normalize();
    let light_view_projection = Mat4::orthographic(-5.0, 5.0, -5.0, 5.0, 1.0, 20.0)
        * directional_light_view(light_direction, target, 8.0, Vec3::new(0.0, 1.0, 0.0));
    let mut shadow_target = Framebuffer::new(128, 128);
    shadow_target.clear(0);
    let mut depth_pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    depth_pipeline.draw_mesh_depth(
        &mut shadow_target,
        &ground,
        &ShadowDepthUniforms::new(Mat4::IDENTITY, light_view_projection),
    );
    depth_pipeline.draw_mesh_depth(
        &mut shadow_target,
        &caster,
        &ShadowDepthUniforms::new(caster_model, light_view_projection),
    );
    let shadow_map = ShadowMap::from_framebuffer(&shadow_target).unwrap();
    let camera_position = Vec3::new(5.2, 3.6, 6.0);
    let view = Mat4::look_at(camera_position, target, Vec3::new(0.0, 1.0, 0.0));
    let projection = Mat4::perspective(0.78, WIDTH as f32 / HEIGHT as f32, 0.1, 30.0);
    let directional = DirectionalLight::new(light_direction, Vec3::new(1.0, 0.95, 0.85));
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let mut lighting = CookTorranceUniforms::new_with_linear_base_color(
        BlinnPhongUniforms::new_with_linear_colors(
            caster_model,
            view,
            projection,
            Vec3::new(0.025, 0.025, 0.025),
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
            camera_position,
            directional,
            point,
        ),
        Vec3::new(0.78, 0.25, 0.08),
        0.15,
        0.45,
    );
    lighting.lighting.set_directional_shadow(
        directional,
        Some(ShadowState::new(light_view_projection, shadow_map)),
    );
    let albedo = Texture::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/gradient.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat);
    let normal_map = Texture::load_with_color_space(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/normal_bump.qoi"),
        ColorSpace::Linear,
    )
    .unwrap()
    .with_wrap_mode(WrapMode::Repeat);
    let uniforms = NormalMappedCookTorranceUniforms::new(
        lighting,
        &albedo,
        &normal_map,
        TextureFilter::Bilinear,
    )
    .unwrap();
    let mut ground_lighting = uniforms.lighting.clone();
    ground_lighting.lighting.set_model(Mat4::IDENTITY);
    let ground_uniforms = NormalMappedCookTorranceUniforms::new(
        ground_lighting,
        &albedo,
        &normal_map,
        TextureFilter::Bilinear,
    )
    .unwrap();
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    let mut pipeline = Pipeline::new(
        NormalMappedCookTorranceShader,
        NormalMappedCookTorranceShader,
    );
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh_with_sampling(target, &ground, &ground_uniforms);
        frame.draw_mesh_with_sampling(target, &caster, &uniforms);
    });
    assert_golden("ggx-normal-map-shadow", &framebuffer);
}
