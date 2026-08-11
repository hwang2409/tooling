use chimy2::camera::Camera;
use chimy2::demo::{cube_with_uvs, plane_xz};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::{ColorSpace, Texture, WrapMode};
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{
    BlinnPhongUniforms, DirectionalLight, NormalMappedBlinnPhongShader,
    NormalMappedBlinnPhongUniforms, PointLight, TextureFilter,
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

fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn quad() -> Mesh {
    Mesh::parse(
        "v -1.5 -1.0 0\nv 1.5 -1.0 0\nv 1.5 1.0 0\nv -1.5 1.0 0\n\
         vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
         f 1/1 2/2 3/3 4/4\n",
    )
    .unwrap()
}

fn textures() -> (Texture, Texture) {
    let albedo = Texture::load(assets().join("gradient.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat);
    let normal_map =
        Texture::load_with_color_space(assets().join("normal_bump.qoi"), ColorSpace::Linear)
            .unwrap()
            .with_wrap_mode(WrapMode::Repeat);
    (albedo, normal_map)
}

fn lighting(camera: &Camera, light_direction: Vec3) -> BlinnPhongUniforms {
    BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.035, 0.035, 0.035),
        Vec3::new(0.9, 0.9, 0.9),
        Vec3::new(0.7, 0.7, 0.7),
        32.0,
        camera.position,
        DirectionalLight::new(light_direction.normalize(), Vec3::new(1.0, 0.95, 0.9)),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    )
}

fn render_quad(light_direction: Vec3, bump: bool) -> Framebuffer {
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 4.5),
        Quat::IDENTITY,
        0.9,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    let albedo = Texture::load(assets().join("gradient.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat);
    let normal_map = if bump {
        Texture::load_with_color_space(assets().join("normal_bump.qoi"), ColorSpace::Linear)
            .unwrap()
            .with_wrap_mode(WrapMode::Repeat)
    } else {
        Texture::new_with_color_space(1, 1, vec![[128, 128, 255, 255]], ColorSpace::Linear).unwrap()
    };
    let uniforms = NormalMappedBlinnPhongUniforms::new(
        lighting(&camera, light_direction),
        &albedo,
        &normal_map,
        TextureFilter::Bilinear,
    );
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    let mut pipeline = Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
    pipeline.draw_mesh_with_sampling(&mut framebuffer, &quad(), &uniforms);
    framebuffer
}

#[test]
fn normal_mapped_quad_has_visible_relief_golden() {
    let normal_mapped = render_quad(Vec3::new(0.8, 0.35, 1.0), true);
    let geometric = render_quad(Vec3::new(0.8, 0.35, 1.0), false);
    assert_ne!(normal_mapped.color, geometric.color);
    assert_golden("m12-normal-map-relief", &normal_mapped);
}

#[test]
fn moving_normal_map_light_changes_shading_golden() {
    let original = render_quad(Vec3::new(0.8, 0.35, 1.0), true);
    let moved = render_quad(Vec3::new(-0.8, 0.35, 1.0), true);
    assert_ne!(original.color, moved.color);
    assert_golden("m12-normal-map-moved-light", &moved);
}

#[test]
fn normal_mapped_mesh_uses_directional_shadows_golden() {
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
    let mut lighting = BlinnPhongUniforms::new(
        caster_model,
        view,
        projection,
        Vec3::new(0.025, 0.025, 0.025),
        Vec3::new(0.78, 0.25, 0.08),
        Vec3::new(0.3, 0.3, 0.3),
        24.0,
        camera_position,
        directional,
        point,
    );
    lighting.set_directional_shadow(
        directional,
        Some(ShadowState::new(light_view_projection, shadow_map)),
    );
    let (albedo, normal_map) = textures();
    let uniforms = NormalMappedBlinnPhongUniforms::new(
        lighting,
        &albedo,
        &normal_map,
        TextureFilter::Bilinear,
    );
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    let mut pipeline = Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
    let mut ground_lighting = uniforms.lighting.clone();
    ground_lighting.set_model(Mat4::IDENTITY);
    let ground_uniforms = NormalMappedBlinnPhongUniforms::new(
        ground_lighting,
        &albedo,
        &normal_map,
        TextureFilter::Bilinear,
    );
    pipeline.draw_mesh_with_sampling(&mut framebuffer, &ground, &ground_uniforms);
    pipeline.draw_mesh_with_sampling(&mut framebuffer, &caster, &uniforms);
    assert_golden("m12-normal-map-shadow", &framebuffer);
}
