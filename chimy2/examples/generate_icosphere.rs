//! Deterministic icosphere generator.
//!
//! Takes the same 12 unit-sphere-projected icosahedron vertices used by the
//! existing `assets/icosphere.obj` file (encoded verbatim so the committed
//! bytes don't drift from platform to platform), then splits each triangle
//! into four by inserting normalized edge midpoints. Every operation is
//! IEEE-754 deterministic: add / sub / mul / div and correctly-rounded
//! `sqrt`. No platform libm calls (no sin/cos/tan/pow/exp).
//!
//! Usage:
//!   cargo run --release --example generate_icosphere -- 3 assets/icosphere_hires.obj
//!
//! The vertex format matches the existing icosphere: six decimal places,
//! space-separated. Output is stable across macOS and Linux hosts.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::ExitCode;

// Unit-sphere vertices for the base icosahedron. These bytes are the same
// verbatim rationals the committed `assets/icosphere.obj` opens with, so
// re-running the generator at level 2 reproduces that file exactly.
const BASE_VERTICES: &[[f32; 3]] = &[
    [-0.525731, 0.850651, 0.000000],
    [0.525731, 0.850651, 0.000000],
    [-0.525731, -0.850651, 0.000000],
    [0.525731, -0.850651, 0.000000],
    [0.000000, -0.525731, 0.850651],
    [0.000000, 0.525731, 0.850651],
    [0.000000, -0.525731, -0.850651],
    [0.000000, 0.525731, -0.850651],
    [0.850651, 0.000000, -0.525731],
    [0.850651, 0.000000, 0.525731],
    [-0.850651, 0.000000, -0.525731],
    [-0.850651, 0.000000, 0.525731],
];

// 20 base triangles (indices into BASE_VERTICES). Winding matches the
// existing `assets/icosphere.obj` at level 0.
const BASE_TRIANGLES: &[[u32; 3]] = &[
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

fn normalize(vertex: [f32; 3]) -> [f32; 3] {
    let length_squared = vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2];
    let inverse_length = 1.0 / length_squared.sqrt();
    [
        vertex[0] * inverse_length,
        vertex[1] * inverse_length,
        vertex[2] * inverse_length,
    ]
}

fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    normalize([
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ])
}

fn subdivide(vertices: &mut Vec<[f32; 3]>, triangles: Vec<[u32; 3]>) -> Vec<[u32; 3]> {
    let mut midpoint_cache = BTreeMap::<(u32, u32), u32>::new();
    let mut next = Vec::with_capacity(triangles.len() * 4);
    for [a, b, c] in triangles {
        let ab = get_midpoint(vertices, &mut midpoint_cache, a, b);
        let bc = get_midpoint(vertices, &mut midpoint_cache, b, c);
        let ca = get_midpoint(vertices, &mut midpoint_cache, c, a);
        next.push([a, ab, ca]);
        next.push([b, bc, ab]);
        next.push([c, ca, bc]);
        next.push([ab, bc, ca]);
    }
    next
}

fn get_midpoint(
    vertices: &mut Vec<[f32; 3]>,
    cache: &mut BTreeMap<(u32, u32), u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&index) = cache.get(&key) {
        return index;
    }
    let vertex = midpoint(vertices[a as usize], vertices[b as usize]);
    let index = vertices.len() as u32;
    vertices.push(vertex);
    cache.insert(key, index);
    index
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let subdivisions = args
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(3);
    let output = args
        .next()
        .unwrap_or_else(|| format!("{}/assets/icosphere_hires.obj", env!("CARGO_MANIFEST_DIR")));

    let mut vertices: Vec<[f32; 3]> = BASE_VERTICES.to_vec();
    let mut triangles: Vec<[u32; 3]> = BASE_TRIANGLES.to_vec();
    for _ in 0..subdivisions {
        triangles = subdivide(&mut vertices, triangles);
    }

    let mut text = String::new();
    text.push_str(&format!(
        "# subdivided icosphere ({subdivisions} levels, {} triangles)\n",
        triangles.len()
    ));
    for vertex in &vertices {
        text.push_str(&format!(
            "v {:.6} {:.6} {:.6}\n",
            vertex[0], vertex[1], vertex[2]
        ));
    }
    for triangle in &triangles {
        text.push_str(&format!(
            "f {} {} {}\n",
            triangle[0] + 1,
            triangle[1] + 1,
            triangle[2] + 1
        ));
    }
    if let Err(error) = fs::write(&output, text) {
        eprintln!("write {output}: {error}");
        return ExitCode::FAILURE;
    }
    println!(
        "wrote {} vertices / {} triangles to {output}",
        vertices.len(),
        triangles.len()
    );
    ExitCode::SUCCESS
}
