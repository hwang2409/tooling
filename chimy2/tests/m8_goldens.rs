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

fn checkerboard() -> Texture {
    let mut pixels = Vec::with_capacity(32 * 32);
    for y in 0..32 {
        for x in 0..32 {
            let value = if (x + y) % 2 == 0 { 245 } else { 10 };
            pixels.push([value, value, value, 255]);
        }
    }
    Texture::new(32, 32, pixels)
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat)
}

#[test]
fn minified_checkerboard_aliased_and_trilinear_goldens() {
    let texture = checkerboard();
    let transform = camera().view_projection();
    let mesh = foreshortened_quad();
    let mut aliased = background();
    let mut trilinear = background();
    let mut aliased_pipeline = Pipeline::new(TexturedShader, TexturedShader);
    aliased_pipeline.draw_mesh(
        &mut aliased,
        &mesh,
        &TexturedUniforms::new(transform, &texture, TextureFilter::Bilinear),
    );
    let mut trilinear_pipeline = Pipeline::new(TexturedShader, TexturedShader);
    trilinear_pipeline.draw_mesh(
        &mut trilinear,
        &mesh,
        &TexturedUniforms::new(transform, &texture, TextureFilter::Trilinear),
    );
    assert_ne!(aliased.color, trilinear.color);
    assert_golden("m8-checker-aliased", &aliased);
    assert_golden("m8-checker-trilinear", &trilinear);
}

#[test]
fn srgb_gradient_lighting_golden() {
    let texture = Texture::new(
        4,
        4,
        (0..4)
            .flat_map(|y| (0..4).map(move |x| [(x * 85) as u8, (y * 85) as u8, 128, 255]))
            .collect(),
    )
    .unwrap()
    .with_wrap_mode(WrapMode::ClampToEdge);
    let camera = camera();
    let lighting = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.03, 0.03, 0.03),
        Vec3::new(0.8, 0.8, 0.8),
        Vec3::ZERO,
        8.0,
        camera.position,
        DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    );
    let mut framebuffer = background();
    let uniforms = TexturedBlinnPhongUniforms::new(lighting, &texture, TextureFilter::Bilinear);
    let mut pipeline = Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
    pipeline.draw_mesh(&mut framebuffer, &foreshortened_quad(), &uniforms);
    assert_golden("m8-srgb-gradient-lit", &framebuffer);
}
