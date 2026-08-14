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
    balance: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("models/arm.json"),
        frames: 600,
        out: PathBuf::from("newt-load.ppm"),
        size: (640, 360),
        balance: false,
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
            // Apply the source biped's torso balance controller each
            // step — pass this for biped-simple.xml so the render
            // shows a standing biped (joint PD alone can't stabilize
            // the inverted-pendulum, see docs/mjcf.md).
            "--balance" => a.balance = true,
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
    let stand_target_z = scene
        .world
        .trees
        .first()
        .map(|t| forward_kinematics(t)[0].0.z)
        .unwrap_or(0.0);
    for _ in 0..args.frames {
        if args.balance {
            apply_stand_balance(&mut scene.world, stand_target_z);
        }
        scene.world.step();
    }

    let (width, height) = args.size;
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));

    // Frame the scene using a bounding sphere over every renderable point
    // so a scene we've never seen still fits the camera view.
    let (center, radius) = scene_extent(&scene);
    let camera = build_camera(center, radius, width, height);
    draw_ground_grid(
        &mut fb,
        camera,
        center,
        radius,
        width,
        height,
        argb8888(0xff, 40, 45, 55),
    );
    draw_bodies(&mut fb, &scene.world, camera, width, height);
    draw_trees(&mut fb, &scene.world, camera, width, height);
    draw_link_geoms(&mut fb, &scene.world, camera, width, height);
    draw_sites(&mut fb, &scene, camera, width, height);

    write_ppm(&args.out, &fb)?;
    let assist_label = if args.balance {
        "WITH source-style torso balance assist"
    } else {
        "with pure joint PD (NO external assist)"
    };
    println!(
        "wrote {} ({}x{}) after {} steps of {} — {}",
        args.out.display(),
        width,
        height,
        args.frames,
        args.model.display(),
        assist_label,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// scene extent + camera framing
// ---------------------------------------------------------------------------

fn scene_extent(scene: &Scene) -> (Vec3, f32) {
    // Focus the camera on the biped-like actor: for scenes that have a
    // tree, center on the root link and pick a radius that fits the
    // tree's own span (independent of any drift). For scenes with only
    // free bodies (stack, projectile…), fall back to a bounding-sphere
    // over the bodies. This keeps the biped in frame even after it
    // slides across the ground while still framing multi-body demos.
    if let Some(tree) = scene.world.trees.first() {
        let poses = forward_kinematics(tree);
        let root = poses[0].0;
        let mut r_sq: f32 = 0.0;
        for (p, _) in &poses {
            let d = *p - root;
            let dd = d.dot(d);
            if dd > r_sq {
                r_sq = dd;
            }
        }
        let radius = (r_sq.sqrt() + 0.6).max(1.5);
        return (root, radius);
    }
    let mut pts: Vec<Vec3> = Vec::new();
    for b in &scene.world.bodies {
        pts.push(b.position);
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

fn draw_ground_grid(
    fb: &mut Framebuffer,
    camera: Mat4,
    center: Vec3,
    radius: f32,
    width: usize,
    height: usize,
    color: u32,
) {
    // Grid centered under the camera focus so the biped stays over
    // visible floor even when it drifts across the world.
    let span = (radius * 2.0).max(3.0);
    let step = 0.5;
    let n = (2.0 * span / step) as i32;
    let cx = center.x;
    let cy = center.y;
    for i in 0..=n {
        let t = -span + (i as f32) * step;
        for (a, b) in [
            (
                Vec3::new(cx - span, cy + t, 0.0),
                Vec3::new(cx + span, cy + t, 0.0),
            ),
            (
                Vec3::new(cx + t, cy - span, 0.0),
                Vec3::new(cx + t, cy + span, 0.0),
            ),
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

/// Draw every geom attached to a tree link as a wireframe: boxes get
/// their 8-corner cube edges, capsules and cylinders get their axis
/// segment plus cross-hairs at the two endpoints. Non-tree-attached
/// geoms are handled by `draw_bodies`; static planes are represented
/// by the ground grid.
fn draw_link_geoms(
    fb: &mut Framebuffer,
    world: &newt::world::World,
    camera: Mat4,
    width: usize,
    height: usize,
) {
    let color = argb8888(0xff, 200, 220, 240);
    for tree in &world.trees {
        let poses = forward_kinematics(tree);
        for g in &world.geoms {
            let GeomAttach::Link(t_idx, l_idx) = g.attachment() else {
                continue;
            };
            let this_tree = std::ptr::from_ref(tree) as usize;
            let matching_tree = std::ptr::from_ref(&world.trees[t_idx]) as usize;
            if this_tree != matching_tree {
                continue;
            }
            let (link_pos, link_ori) = poses[l_idx];
            let center = link_pos + link_ori.rotate(g.local_offset);
            let ori_world = link_ori * g.local_orientation;
            match g.shape {
                GeomShape::Box { half_extents } => {
                    let corners = box_corners_world(center, ori_world, half_extents);
                    let proj: [Option<(i32, i32)>; 8] =
                        std::array::from_fn(|i| project(camera, corners[i], width, height));
                    for &(a, b) in BOX_EDGES.iter() {
                        if let (Some(p0), Some(p1)) = (proj[a], proj[b]) {
                            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
                        }
                    }
                }
                GeomShape::Capsule {
                    radius,
                    half_height,
                }
                | GeomShape::Cylinder {
                    radius,
                    half_height,
                } => {
                    let axis = ori_world.rotate(Vec3::new(0.0, 0.0, half_height));
                    let a = center - axis;
                    let b = center + axis;
                    if let (Some(pa), Some(pb)) = (
                        project(camera, a, width, height),
                        project(camera, b, width, height),
                    ) {
                        draw_line(fb, pa.0, pa.1, pb.0, pb.1, color);
                        let r = radius.max(0.01);
                        // Rough end caps: cross-hairs of width `2r` in the
                        // link's XY plane (screen-projected).
                        for &(p, _) in &[(a, ()), (b, ())] {
                            let ex = ori_world.rotate(Vec3::new(r, 0.0, 0.0));
                            let ey = ori_world.rotate(Vec3::new(0.0, r, 0.0));
                            if let (Some(p0), Some(p1)) = (
                                project(camera, p - ex, width, height),
                                project(camera, p + ex, width, height),
                            ) {
                                draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
                            }
                            if let (Some(p0), Some(p1)) = (
                                project(camera, p - ey, width, height),
                                project(camera, p + ey, width, height),
                            ) {
                                draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
                            }
                        }
                    }
                }
                GeomShape::Sphere { radius } => {
                    let r = radius;
                    let ex = ori_world.rotate(Vec3::new(r, 0.0, 0.0));
                    let ey = ori_world.rotate(Vec3::new(0.0, r, 0.0));
                    let ez = ori_world.rotate(Vec3::new(0.0, 0.0, r));
                    for e in [ex, ey, ez] {
                        if let (Some(p0), Some(p1)) = (
                            project(camera, center - e, width, height),
                            project(camera, center + e, width, height),
                        ) {
                            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
                        }
                    }
                }
                _ => {
                    if let Some((x, y)) = project(camera, center, width, height) {
                        draw_line(fb, x - 3, y, x + 3, y, color);
                        draw_line(fb, x, y - 3, x, y + 3, color);
                    }
                }
            }
        }
    }
}

/// Source-mirror torso balance controller — same 6-component wrench
/// the tests apply, kept in sync with
/// `tests/mjcf_load.rs::apply_source_balance_wrench` and with
/// `~/me/fun/biped/biped/mujoco_biped.py::_apply_balance_controller`
/// at `assist_scale=1.0`. See `newt/docs/mjcf.md` for the derivation
/// and frame-conversion note. Only invoked when the caller passes
/// `--balance`; the plain-PD demo leaves this off so the render
/// matches actual pure-PD behavior (biped falls).
fn apply_stand_balance(world: &mut newt::world::World, target_z: f32) {
    if world.trees.is_empty() {
        return;
    }
    let (torso_pos, torso_ori) = forward_kinematics(&world.trees[0])[0];
    let up_world = torso_ori.rotate(Vec3::new(0.0, 0.0, 1.0));
    let omega_body = Vec3::new(
        world.trees[0].qdot.first().copied().unwrap_or(0.0),
        world.trees[0].qdot.get(1).copied().unwrap_or(0.0),
        world.trees[0].qdot.get(2).copied().unwrap_or(0.0),
    );
    let v_body = Vec3::new(
        world.trees[0].qdot.get(3).copied().unwrap_or(0.0),
        world.trees[0].qdot.get(4).copied().unwrap_or(0.0),
        world.trees[0].qdot.get(5).copied().unwrap_or(0.0),
    );
    let omega_world = torso_ori.rotate(omega_body);
    let v_world = torso_ori.rotate(v_body);
    let force_x = (42.0 * -torso_pos.x + 82.0 * -v_world.x).clamp(-75.0, 75.0);
    let force_y = (90.0 * -torso_pos.y - 35.0 * v_world.y).clamp(-35.0, 35.0);
    let force_z = (240.0 * (target_z - torso_pos.z) - 70.0 * v_world.z).clamp(-90.0, 260.0);
    let torque_x = (135.0 * up_world.y - 24.0 * omega_world.x).clamp(-95.0, 95.0);
    let torque_y = (-135.0 * up_world.x - 24.0 * omega_world.y).clamp(-95.0, 95.0);
    let torque_z = (-12.0 * omega_world.z).clamp(-28.0, 28.0);
    world.trees[0].applied_wrenches[0] = (
        Vec3::new(force_x, force_y, force_z),
        Vec3::new(torque_x, torque_y, torque_z),
    );
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
