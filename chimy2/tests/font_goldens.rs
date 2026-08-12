use chimy2::fb::Framebuffer;
use chimy2::font::draw_ascii_hud;
use std::fs;
use std::path::{Path, PathBuf};

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
        fs::write(&path, &actual).expect("write golden");
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    assert_eq!(fs::read(&path).expect("read golden"), actual, "{name}");
}

#[test]
fn full_ascii_hud_golden() {
    let mut framebuffer = Framebuffer::new(512, 96);
    framebuffer.clear(0xff050812);
    draw_ascii_hud(&mut framebuffer, 42);
    assert_golden("font-full-ascii-hud", &framebuffer);
}
