use chimy2::demo::{build_lod_scene, render_lod_scene};
use chimy2::fb::Framebuffer;
use std::collections::BTreeSet;
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

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join("lod-rings.ppm")
}

#[test]
fn lod_rings_golden() {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let scene = build_lod_scene(WIDTH as f32 / HEIGHT as f32);
    let counts = scene
        .mesh
        .levels()
        .iter()
        .map(|mesh| mesh.indices().len())
        .collect::<Vec<_>>();
    assert!(
        counts.windows(2).all(|window| window[0] > window[1]),
        "LOD counts: {counts:?}"
    );
    let selected_levels = scene
        .models
        .iter()
        .map(|model| {
            scene
                .mesh
                .select(
                    scene.camera.view_matrix(),
                    scene.projection,
                    *model,
                    WIDTH,
                    HEIGHT,
                )
                .level()
        })
        .collect::<Vec<_>>();
    let selected_set = selected_levels.iter().copied().collect::<BTreeSet<_>>();
    assert!(
        selected_set.len() >= 3,
        "expected at least three selected LOD levels, got {selected_levels:?}"
    );
    render_lod_scene(&mut framebuffer, &scene);
    let path = golden_path();
    let actual = ppm(&framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).expect("write golden");
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    let expected = fs::read(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}
