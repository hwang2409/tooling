use chimy2::camera::{Camera, OrbitController};
use chimy2::demo::uv_sphere;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::Texture;
use chimy2::math::{Mat4, Quat, Vec3, Vec4};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{
    BlinnPhongUniforms, DirectionalLight, EnvironmentBlinnPhongShader,
    EnvironmentBlinnPhongUniforms, FlatColorShader, FlatColorUniforms, MeshShader, MeshUniforms,
    PointLight,
};
use chimy2::skybox::CubeTexture;
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 32;
const HEIGHT: usize = 24;

fn cube() -> CubeTexture {
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let faces = ["px", "nx", "py", "ny", "pz", "nz"]
        .map(|name| Texture::load(assets.join(format!("skybox_{name}.qoi"))))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    CubeTexture::new(faces.try_into().unwrap()).unwrap()
}

fn camera(orientation: Quat) -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 4.0),
        orientation,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    )
}

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

fn assert_golden(name: &str, framebuffer: &Framebuffer) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(format!("{name}.ppm"));
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, actual).unwrap();
        panic!("regenerated {}", path.display());
    }
    assert_eq!(fs::read(&path).unwrap(), actual, "golden mismatch: {name}");
}

fn sky_frame(camera: Camera) -> Framebuffer {
    let cube = cube();
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 1, 2, 3));
    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &cube, camera);
    });
    framebuffer
}

#[test]
fn skybox_orientation_a_golden() {
    assert_golden("skybox-orientation-a", &sky_frame(camera(Quat::IDENTITY)));
}

#[test]
fn skybox_orientation_b_golden() {
    let orientation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.1)
        * Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -0.35);
    assert_golden("skybox-orientation-b", &sky_frame(camera(orientation)));
}

#[test]
fn opaque_mesh_occludes_skybox_golden() {
    let cube = cube();
    let camera = camera(Quat::IDENTITY);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 1, 2, 3));
    let vertices = [
        Vec4::new(-0.7, -0.6, 0.0, 1.0),
        Vec4::new(0.7, -0.6, 0.0, 1.0),
        Vec4::new(0.0, 0.7, 0.0, 1.0),
    ];
    let uniforms = FlatColorUniforms::new(Mat4::IDENTITY, argb8888(255, 245, 245, 245));
    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &cube, camera);
        frame.draw(target, &vertices, &[[0, 1, 2]], &uniforms);
    });
    assert!(framebuffer.depth.iter().any(|&depth| depth < 1.0));
    assert_golden("skybox-opaque-occlusion", &framebuffer);
}

#[test]
fn opaque_sky_transparent_overlap_golden() {
    let cube = cube();
    let camera = camera(Quat::IDENTITY);
    let opaque = Mesh::parse("v -0.55 -0.5 0\nv 0.55 -0.5 0\nv 0 0.6 0\nf 1 2 3\n").unwrap();
    let transparent =
        Mesh::parse("v -0.9 -0.8 0.5\nv 0.9 -0.8 0.5\nv 0 0.85 0.5\nf 1 2 3\n").unwrap();
    let opaque_uniforms = MeshUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        argb8888(255, 245, 245, 245),
    );
    let mut transparent_uniforms = MeshUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        argb8888(255, 40, 90, 235),
    );
    transparent_uniforms.set_alpha(0.5);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 1, 2, 3));
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &cube, camera);
        frame.draw_mesh(target, &opaque, &opaque_uniforms);
        frame.draw_mesh(target, &transparent, &transparent_uniforms);
    });
    assert_golden("skybox-mixed-transparent", &framebuffer);
}

#[test]
fn reflective_mesh_under_skybox_golden() {
    let cube = cube();
    let camera = OrbitController::new(Vec3::ZERO, 4.4, 0.4, 0.18).camera(
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    let lighting = BlinnPhongUniforms::new(
        Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.2),
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.03, 0.04, 0.06),
        Vec3::new(0.3, 0.35, 0.42),
        Vec3::new(0.9, 0.95, 1.0),
        48.0,
        camera.position,
        DirectionalLight::new(
            Vec3::new(-0.35, 0.75, 1.0).normalize(),
            Vec3::new(1.0, 0.92, 0.82),
        ),
        PointLight::new(
            Vec3::new(2.0, 1.5, 3.0),
            Vec3::new(0.6, 0.7, 1.0),
            1.0,
            0.08,
            0.02,
        ),
    );
    let uniforms = EnvironmentBlinnPhongUniforms::new(lighting, &cube, 0.72);
    let mesh = uv_sphere(1.3, 16, 32);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 1, 2, 3));
    let mut pipeline = Pipeline::new(EnvironmentBlinnPhongShader, EnvironmentBlinnPhongShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &cube, camera);
        frame.draw_mesh(target, &mesh, &uniforms);
    });
    assert_golden("skybox-reflective-mesh", &framebuffer);
}
