use chimy2::camera::Camera;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::{Texture, WrapMode};
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{
    BlinnPhongUniforms, DirectionalLight, PointLight, TextureFilter, TexturedBlinnPhongShader,
    TexturedBlinnPhongUniforms, TexturedShader, TexturedUniforms,
};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

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

fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    )
}

fn foreshortened_quad() -> Mesh {
    Mesh::parse(
        "v -1.5 -1 0\nv 1.5 -1 -2\nv 1.5 1 -2\nv -1.5 1 0\n\
         vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
         f 1/1 2/2 3/3 4/4\n",
    )
    .unwrap()
}

fn background() -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    framebuffer
}

#[test]
fn perspective_textured_quad_golden() {
    let texture = Texture::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/checker.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat);
    let camera = camera();
    let mut framebuffer = background();
    let uniforms =
        TexturedUniforms::new(camera.view_projection(), &texture, TextureFilter::Nearest);
    let mut pipeline = Pipeline::new(TexturedShader, TexturedShader);
    let mesh = foreshortened_quad();
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh_with_sampling(target, &mesh, &uniforms);
    });
    assert_golden("m5-perspective-textured-quad", &framebuffer);
}

#[test]
fn bilinear_and_nearest_have_distinct_goldens() {
    let texture = Texture::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/gradient.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::ClampToEdge);
    let camera = camera();
    let mesh = foreshortened_quad();
    let mut nearest = background();
    let mut bilinear = background();
    let mut nearest_pipeline = Pipeline::new(TexturedShader, TexturedShader);
    let nearest_uniforms =
        TexturedUniforms::new(camera.view_projection(), &texture, TextureFilter::Nearest);
    nearest_pipeline.render(&mut nearest, |frame, target| {
        frame.draw_mesh_with_sampling(target, &mesh, &nearest_uniforms);
    });
    let mut bilinear_pipeline = Pipeline::new(TexturedShader, TexturedShader);
    let bilinear_uniforms =
        TexturedUniforms::new(camera.view_projection(), &texture, TextureFilter::Bilinear);
    bilinear_pipeline.render(&mut bilinear, |frame, target| {
        frame.draw_mesh_with_sampling(target, &mesh, &bilinear_uniforms);
    });
    assert_ne!(nearest.color, bilinear.color);
    assert_golden("m5-nearest-textured-quad", &nearest);
    assert_golden("m5-bilinear-textured-quad", &bilinear);
}

#[test]
fn textured_lit_mesh_golden() {
    let texture = Texture::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/checker.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat);
    let camera = camera();
    let lighting = BlinnPhongUniforms::new(
        Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.25),
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.04, 0.04, 0.04),
        Vec3::new(0.9, 0.9, 0.9),
        Vec3::new(0.7, 0.7, 0.7),
        24.0,
        camera.position,
        DirectionalLight::new(
            Vec3::new(-0.4, 0.7, 1.0).normalize(),
            Vec3::new(1.0, 0.95, 0.9),
        ),
        PointLight::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 0.5, 0.3),
            1.0,
            0.08,
            0.02,
        ),
    );
    let mut framebuffer = background();
    let uniforms = TexturedBlinnPhongUniforms::new(lighting, &texture, TextureFilter::Bilinear);
    let mut pipeline = Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
    let mesh = foreshortened_quad();
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh_with_sampling(target, &mesh, &uniforms);
    });
    assert_golden("m5-textured-lit-mesh", &framebuffer);
}
