use chimy2::demo::DemoArgs;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec4};
use chimy2::pipeline::Pipeline;
use chimy2::postfx::PostChain;
use chimy2::shaders::{FlatColorShader, FlatColorUniforms};
use std::fs;
use std::path::PathBuf;

fn fixture() -> Framebuffer {
    let width = 16;
    let height = 12;
    let mut framebuffer = Framebuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let edge = x >= 8;
            let red = if edge { 235 } else { 18 + x as u8 * 4 };
            let green = if edge {
                42 + y as u8 * 3
            } else {
                24 + y as u8 * 4
            };
            let blue = if edge {
                18 + y as u8 * 2
            } else {
                48 + x as u8 * 3
            };
            framebuffer.color[y * width + x] = argb8888(255, red, green, blue);
        }
    }
    for y in 4..8 {
        for x in 6..10 {
            framebuffer.color[y * width + x] = if (x + y) % 2 == 0 {
                argb8888(255, 255, 235, 180)
            } else {
                argb8888(255, 245, 95, 24)
            };
        }
    }
    framebuffer
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
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
        fs::write(&path, &actual).expect("write postfx golden");
    }
    let expected = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing golden {}: {error}; run GOLDEN_REGEN=1 cargo test --test postfx",
            path.display()
        )
    });
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

fn production_chain(bloom: bool, fxaa: bool, vignette: bool) -> PostChain {
    DemoArgs {
        bloom,
        fxaa,
        vignette,
        ..DemoArgs::default()
    }
    .post_chain()
}

#[test]
fn bloom_golden() {
    let mut framebuffer = fixture();
    production_chain(true, false, false).apply(&mut framebuffer);
    assert_golden("m17-postfx-bloom", &framebuffer);
}

#[test]
fn fxaa_golden() {
    let mut framebuffer = fixture();
    production_chain(false, true, false).apply(&mut framebuffer);
    assert_golden("m17-postfx-fxaa", &framebuffer);
}

#[test]
fn vignette_golden() {
    let mut framebuffer = fixture();
    production_chain(false, false, true).apply(&mut framebuffer);
    assert_golden("m17-postfx-vignette", &framebuffer);
}

#[test]
fn full_chain_golden() {
    let mut framebuffer = fixture();
    production_chain(true, true, true).apply(&mut framebuffer);
    assert_golden("m17-postfx-full-chain", &framebuffer);
}

#[test]
fn ssaa_pipeline_post_chain_golden() {
    let mut framebuffer = Framebuffer::new(12, 12);
    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    pipeline.set_ssaa_scale(2);
    pipeline.set_post_chain(
        DemoArgs {
            vignette: true,
            ..DemoArgs::default()
        }
        .post_chain(),
    );
    let uniforms = FlatColorUniforms::new(Mat4::IDENTITY, argb8888(255, 240, 240, 240));
    pipeline.render(&mut framebuffer, |frame, target| {
        target.clear(argb8888(255, 20, 20, 20));
        frame.draw(
            target,
            &[
                Vec4::new(-1.0, -1.0, 0.0, 1.0),
                Vec4::new(0.85, -1.0, 0.0, 1.0),
                Vec4::new(-1.0, 0.85, 0.0, 1.0),
            ],
            &[[0, 1, 2]],
            &uniforms,
        );
    });
    assert_golden("m17-postfx-ssaa-chain", &framebuffer);
}
