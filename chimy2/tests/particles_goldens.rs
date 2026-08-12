use chimy2::demo::{build_fountain_scene, render_fountain_scene};
use chimy2::fb::Framebuffer;
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
        .join("tests")
        .join("goldens")
        .join(format!("{name}.ppm"))
}

#[test]
fn fountain_fixed_step_golden() {
    let mut framebuffer = Framebuffer::new(96, 64);
    let mut scene = build_fountain_scene(96.0 / 64.0);
    for _ in 0..45 {
        render_fountain_scene(&mut framebuffer, &mut scene);
    }
    let actual = ppm(&framebuffer);
    let path = golden_path("particles-fountain");
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, actual).expect("write golden");
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    let expected = fs::read(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}
