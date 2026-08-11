use chimy2::fb::{Framebuffer, argb8888};
use chimy2::postfx::{BloomPass, FxaaPass, PostChain, VignettePass};
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

#[test]
fn bloom_golden() {
    let mut framebuffer = fixture();
    PostChain::new()
        .with_pass(BloomPass)
        .apply(&mut framebuffer);
    assert_golden("m17-postfx-bloom", &framebuffer);
}

#[test]
fn fxaa_golden() {
    let mut framebuffer = fixture();
    PostChain::new().with_pass(FxaaPass).apply(&mut framebuffer);
    assert_golden("m17-postfx-fxaa", &framebuffer);
}

#[test]
fn vignette_golden() {
    let mut framebuffer = fixture();
    PostChain::new()
        .with_pass(VignettePass)
        .apply(&mut framebuffer);
    assert_golden("m17-postfx-vignette", &framebuffer);
}

#[test]
fn full_chain_golden() {
    let mut framebuffer = fixture();
    let chain = PostChain::new()
        .with_pass(BloomPass)
        .with_pass(FxaaPass)
        .with_pass(VignettePass);
    chain.apply(&mut framebuffer);
    assert_golden("m17-postfx-full-chain", &framebuffer);
}

#[test]
fn full_chain_order_is_observable() {
    let mut forward = fixture();
    PostChain::new()
        .with_pass(BloomPass)
        .with_pass(FxaaPass)
        .with_pass(VignettePass)
        .apply(&mut forward);
    let mut swapped = fixture();
    PostChain::new()
        .with_pass(VignettePass)
        .with_pass(BloomPass)
        .with_pass(FxaaPass)
        .apply(&mut swapped);
    assert_ne!(forward.color, swapped.color);
}
