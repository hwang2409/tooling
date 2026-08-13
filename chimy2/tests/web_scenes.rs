//! Contract test: every scene JSON shipped in the browser gallery must parse.
//!
//! The strict scene loader rejects unknown keys, malformed shapes, and any type
//! mismatch — so this test catches typos, orphan fields, and drift between the
//! docs and the shipped scenes forever, without touching any golden images.
//! Rendering is not exercised here: the browser build cannot read files off the
//! filesystem, and the native renderer already has focused golden tests for
//! every rendering feature the scenes reference.

use chimy2::scene::Scene;
use std::path::{Path, PathBuf};

fn scene_files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("web")
        .join("scenes");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("web/scenes/ directory must exist alongside the browser demo")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".scene.json"))
        })
        .collect();
    files.sort();
    files
}

#[test]
fn every_web_scene_parses_strictly() {
    let files = scene_files();
    assert!(
        !files.is_empty(),
        "web/scenes/ must contain at least one *.scene.json"
    );
    for path in files {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        Scene::from_str(&source)
            .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    }
}

#[test]
fn every_web_scene_has_a_gallery_screenshot() {
    let gallery = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("web")
        .join("gallery");
    for scene in scene_files() {
        let stem = scene
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".scene.json"))
            .unwrap_or_else(|| panic!("unexpected scene filename: {}", scene.display()));
        let thumbnail = gallery.join(format!("{stem}.png"));
        assert!(
            thumbnail.is_file(),
            "missing gallery screenshot for {}: expected {}",
            scene.display(),
            thumbnail.display()
        );
    }
}
