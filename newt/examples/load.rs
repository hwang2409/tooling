//! Tier-5 demo: load a scene from a `newt` model file, step it, and render
//! a wireframe of the final state via chimy2. Dispatches by file extension:
//! `.json` uses [`newt::model::load_from_path`] and `.xml` uses
//! [`newt::mjcf::load_mjcf_path`] (v1 tier 7). Works with any of the packaged
//! models (`models/pendulum.json|xml`, `models/arm.json|xml`,
//! `models/stack.json|xml`, `models/biped-simple.xml`) or an external file
//! passed via `--model`.
//!
//! Free bodies are drawn as unit-cube wireframes scaled by 2·half_extents;
//! trees are drawn as rods between each hinge link's parent joint anchor
//! and the link's COM. Sites (if any) are marked with a small cross.
//!
//! Run:
//! ```text
//! cargo run --release --example load -- --model models/arm.json --frames 900 --out /tmp/arm.ppm
//! cargo run --release --example load -- --model models/biped-simple.xml --frames 400 --out /tmp/biped.ppm
//! ```

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::geom::{GeomAttach, GeomShape};
use newt::math::{Quat, Vec3};
use newt::mjcf::load_mjcf_path;
use newt::model::{Scene, load_from_path};
use newt::tree::forward_kinematics;

use std::path::PathBuf;

struct Args {
    model: PathBuf,
    frames: usize,
    out: PathBuf,
    size: (usize, usize),
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("models/arm.json"),
        frames: 600,
        out: PathBuf::from("newt-load.ppm"),
        size: (640, 360),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => a.model = PathBuf::from(it.next().unwrap()),
            "--frames" => a.frames = it.next().unwrap().parse().unwrap(),
            "--out" => a.out = PathBuf::from(it.next().unwrap()),
            "--size" => {
                let s = it.next().unwrap();
                let (w, h) = s.split_once('x').expect("--size WxH");
                a.size = (w.parse().unwrap(), h.parse().unwrap());
            }
            _ => panic!("unknown arg: {arg}"),
        }
    }
    a
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let ext = args
        .model
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("json");
    let mut scene = match ext {
        "xml" | "mjcf" => {
            load_mjcf_path(&args.model).unwrap_or_else(|e| panic!("load {:?}: {e}", args.model))
        }
        _ => load_from_path(&args.model).unwrap_or_else(|e| panic!("load {:?}: {e}", args.model)),
    };
    for _ in 0..args.frames {
        scene.world.step();
    }

    let (width, height) = args.size;
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));

    // Frame the scene using a bounding sphere over every renderable point
    // so a scene we've never seen still fits the camera view.
    let (center, radius) = scene_extent(&scene);
    let camera = build_camera(center, radius, width, height);
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));
    draw_bodies(&mut fb, &scene.world, camera, width, height);
    draw_trees(&mut fb, &scene.world, camera, width, height);
    draw_sites(&mut fb, &scene, camera, width, height);

    write_ppm(&args.out, &fb)?;
    println!(
        "wrote {} ({}x{}) after {} steps of {}",
        args.out.display(),
        width,
        height,
        args.frames,
        args.model.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// scene extent + camera framing
// ---------------------------------------------------------------------------

fn scene_extent(scene: &Scene) -> (Vec3, f32) {
    let mut pts: Vec<Vec3> = Vec::new();
    for b in &scene.world.bodies {
        pts.push(b.position);
    }
    for t in &scene.world.trees {
        let poses = forward_kinematics(t);
        for (p, _) in &poses {
            pts.push(*p);
        }
    }
    for i in 0..scene.sites.len() {
        pts.push(scene.site_pose_by_index(i).0);
    }
    if pts.is_empty() {
        return (Vec3::ZERO, 2.0);
    }
    let mut center = Vec3::ZERO;
    for p in &pts {
        center += *p;
    }
    center *= 1.0 / pts.len() as f32;
    let mut r_sq: f32 = 0.5;
    for p in &pts {
        let d = *p - center;
        let dd = d.dot(d);
        if dd > r_sq {
            r_sq = dd;
        }
    }
    (center, r_sq.sqrt() + 0.6)
}

fn build_camera(center: Vec3, radius: f32, width: usize, height: usize) -> Mat4 {
    let dist = radius * 3.2;
    Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.05,
        200.0,
    ) * Mat4::look_at(
        CVec3::new(
            center.x + dist,
            center.y - dist * 0.6,
            center.z + dist * 0.4,
        ),
        CVec3::new(center.x, center.y, center.z),
        CVec3::new(0.0, 0.0, 1.0),
    )
}

// ---------------------------------------------------------------------------
// drawing helpers
// ---------------------------------------------------------------------------

fn project(camera: Mat4, world_pt: Vec3, width: usize, height: usize) -> Option<(i32, i32)> {
    let clip = camera * Vec4::new(world_pt.x, world_pt.y, world_pt.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let ndc_z = clip.z / clip.w;
    if !(-1.0..=1.0).contains(&ndc_z) {
        return None;
    }
    let sx = (ndc_x * 0.5 + 0.5) * (width as f32);
    let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * (height as f32);
    Some((sx as i32, sy as i32))
}

fn draw_line(fb: &mut Framebuffer, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: u32) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let w = fb.width as i32;
    let h = fb.height as i32;
    loop {
        if x0 >= 0 && y0 >= 0 && x0 < w && y0 < h {
            fb.put_pixel(x0 as usize, y0 as usize, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn draw_ground_grid(fb: &mut Framebuffer, camera: Mat4, width: usize, height: usize, color: u32) {
    let span = 3.0;
    let step = 0.5;
    let n = (2.0 * span / step) as i32;
    for i in 0..=n {
        let t = -span + (i as f32) * step;
        for (a, b) in [
            (Vec3::new(-span, t, 0.0), Vec3::new(span, t, 0.0)),
            (Vec3::new(t, -span, 0.0), Vec3::new(t, span, 0.0)),
        ] {
            if let (Some(p0), Some(p1)) = (
                project(camera, a, width, height),
                project(camera, b, width, height),
            ) {
                draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
            }
        }
    }
}

const BOX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

fn box_corners_world(center: Vec3, ori: Quat, half: Vec3) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    let signs = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];
    for (i, &(sx, sy, sz)) in signs.iter().enumerate() {
        let local = Vec3::new(sx * half.x, sy * half.y, sz * half.z);
        out[i] = center + ori.rotate(local);
    }
    out
}

/// Draw every free body: if it has an attached box geom, draw that as a
/// wireframe; otherwise draw a small marker cross at the body position.
fn draw_bodies(
    fb: &mut Framebuffer,
    world: &newt::world::World,
    camera: Mat4,
    width: usize,
    height: usize,
) {
    let colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 240, 100, 140),
        argb8888(0xff, 180, 240, 120),
    ];
    for (bi, body) in world.bodies.iter().enumerate() {
        let color = colors[bi % colors.len()];
        let mut drew = false;
        for g in &world.geoms {
            if !matches!(g.attachment(), GeomAttach::Body(i) if i == bi) {
                continue;
            }
            if let GeomShape::Box { half_extents } = g.shape {
                let center_world = body.position + body.orientation.rotate(g.local_offset);
                let ori_world = body.orientation * g.local_orientation;
                let corners = box_corners_world(center_world, ori_world, half_extents);
                let proj: [Option<(i32, i32)>; 8] =
                    std::array::from_fn(|i| project(camera, corners[i], width, height));
                for &(a, b) in BOX_EDGES.iter() {
                    if let (Some(p0), Some(p1)) = (proj[a], proj[b]) {
                        draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
                    }
                }
                drew = true;
            }
        }
        if !drew {
            // Fallback: small cross at the body position.
            if let Some((x, y)) = project(camera, body.position, width, height) {
                draw_line(fb, x - 5, y, x + 5, y, color);
                draw_line(fb, x, y - 5, x, y + 5, color);
            }
        }
    }
}

/// Draw every tree link as a rod between its parent's joint anchor and the
/// link's COM. Skip the root link (nothing to connect to).
fn draw_trees(
    fb: &mut Framebuffer,
    world: &newt::world::World,
    camera: Mat4,
    width: usize,
    height: usize,
) {
    let rod_colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 220, 120, 220),
        argb8888(0xff, 240, 100, 140),
        argb8888(0xff, 180, 240, 120),
    ];
    for tree in &world.trees {
        let poses = forward_kinematics(tree);
        for i in 1..tree.links.len() {
            let link = &tree.links[i];
            let parent = link.parent.unwrap();
            let (parent_pos, parent_ori) = poses[parent];
            let anchor_world = parent_pos + parent_ori.rotate(link.joint_offset_in_parent.0);
            let com_world = poses[i].0;
            let color = rod_colors[i % rod_colors.len()];
            if let (Some(p0), Some(p1)) = (
                project(camera, anchor_world, width, height),
                project(camera, com_world, width, height),
            ) {
                draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
            }
        }
    }
}

fn draw_sites(fb: &mut Framebuffer, scene: &Scene, camera: Mat4, width: usize, height: usize) {
    for i in 0..scene.sites.len() {
        let (pos, _) = scene.site_pose_by_index(i);
        if let Some((x, y)) = project(camera, pos, width, height) {
            let s = 4;
            draw_line(fb, x - s, y, x + s, y, argb8888(0xff, 240, 240, 240));
            draw_line(fb, x, y - s, x, y + s, argb8888(0xff, 240, 240, 240));
        }
    }
}
