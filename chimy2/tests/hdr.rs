use chimy2::camera::Camera;
use chimy2::demo::uv_sphere;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::postfx::{AcesTonemapPass, BloomPass, PostChain};
use chimy2::shaders::{
    BlinnPhongUniforms, CookTorranceShader, CookTorranceUniforms, DirectionalLight, PointLight,
};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(format!("{name}.ppm"))
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
    let path = golden_path(name);
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, actual).unwrap();
        return;
    }
    assert_eq!(fs::read(path).unwrap(), actual, "{name}");
}

fn render(hdr: bool, exposure: f32, bloom: bool) -> Framebuffer {
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        0.85,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        20.0,
    );
    let sphere = uv_sphere(1.1, 24, 48);
    let lighting = BlinnPhongUniforms::new_with_linear_colors(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.02, 0.02, 0.02),
        Vec3::ZERO,
        Vec3::ZERO,
        0.0,
        camera.position,
        DirectionalLight {
            direction: Vec3::new(-0.35, 0.7, 1.0).normalize(),
            color: Vec3::new(5.0, 4.5, 4.0),
        },
        PointLight::default(),
    );
    let uniforms = CookTorranceUniforms::new_with_linear_base_color(
        lighting,
        Vec3::new(0.9, 0.18, 0.04),
        0.85,
        0.1,
    );
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 5, 7, 12));
    let mut pipeline = Pipeline::new(CookTorranceShader, CookTorranceShader);
    pipeline.set_hdr(hdr);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &sphere, &uniforms);
    });
    if hdr {
        let mut chain = PostChain::new();
        if bloom {
            chain.push(BloomPass);
        }
        chain.push(AcesTonemapPass::new(exposure));
        chain.apply(&mut framebuffer);
    }
    framebuffer
}

#[test]
fn hdr_ggx_highlight_rolls_off_with_ldr_contrast() {
    let hdr = render(true, 1.0, false);
    let ldr = render(false, 1.0, false);
    assert_ne!(hdr.color, ldr.color);
    assert_golden("m18-hdr-ggx-highlight", &hdr);
    assert_golden("m18-ldr-ggx-highlight", &ldr);
}

#[test]
fn hdr_exposure_sweep_has_distinct_goldens() {
    let half = render(true, 0.5, false);
    let double = render(true, 2.0, false);
    assert_ne!(half.color, double.color);
    assert_golden("m18-hdr-exposure-half", &half);
    assert_golden("m18-hdr-exposure-double", &double);
}

#[test]
fn hdr_bloom_uses_a_threshold_above_one() {
    let without_bloom = render(true, 1.0, false);
    let with_bloom = render(true, 1.0, true);
    assert_ne!(without_bloom.color, with_bloom.color);
    assert_golden("m18-hdr-bloom", &with_bloom);
}
