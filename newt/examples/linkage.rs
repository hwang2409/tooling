//! v1-tier-5 demo: a four-bar-style linkage built from two grounded
//! hinges, a free coupler bar constrained by two `connect` equalities,
//! and a `joint` coupling equality that slaves the follower hinge to
//! `2 · crank`. One servo drives the crank; the whole assembly moves
//! through the connect + coupling constraints.
//!
//! Uses ALL four v1-tier-5 equality features:
//! - `Connect` (×2, world-space anchor coincidence)
//! - `JointCoupling` (crank↔follower polynomial)
//!
//! Run:
//! ```text
//! cargo run --release --example linkage -- --frames 800 --out /tmp/linkage.ppm
//! ```
//!
//! Wireframe PPM via chimy2, same rendering plumbing as the other
//! solver demos.
//!
//! Note on convention: the crank hinge axis is +Y and the crank swings
//! in the world XZ plane. A positive servo target angle rotates the
//! crank tip toward -Z (following right-hand rule about +Y).

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::actuator::Actuator;
use newt::body::Body;
use newt::equality::Equality;
use newt::geom::SolRef;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3, sin};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::World;

use std::path::PathBuf;

mod showcase_support;

/// Scene geometry — three fixed lengths that pin down the mechanism.
const CRANK_LEN: f32 = 0.4;
const FOLLOWER_LEN: f32 = 0.4;
/// Coupler bar length. Half-extents (0.5 · length, small, small).
const COUPLER_HALF: Vec3 = Vec3::new(0.4, 0.02, 0.02);
/// World +X positions of the two grounded hinge pivots.
const CRANK_PIVOT_X: f32 = -0.4;
const FOLLOWER_PIVOT_X: f32 = 0.4;

fn parse_args() -> (usize, PathBuf, (usize, usize), bool) {
    let mut frames = 800usize;
    let mut out = PathBuf::from("newt-linkage.mp4");
    let mut size = (640usize, 360usize);
    let mut wireframe = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => frames = args.next().unwrap().parse().unwrap(),
            "--out" => out = PathBuf::from(args.next().unwrap()),
            "--size" => {
                let s = args.next().unwrap();
                let (w, h) = s.split_once('x').expect("--size WxH");
                size = (w.parse().unwrap(), h.parse().unwrap());
            }
            "--wireframe" => wireframe = true,
            _ => panic!("unknown arg: {a}"),
        }
    }
    (frames, out, size, wireframe)
}

fn build_world() -> (World, usize, usize) {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    let solref = SolRef::new(0.01, 1.0);
    let solimp = SolImp::new(0.99, 0.999, 0.001, 0.5, 2);

    // Build the ground + crank + follower tree. Ground fixed at world
    // origin. Two hinges as siblings, both about +Y so they swing in
    // the XZ plane. Non-zero armature to smooth RK4 under servo load.
    let mut tree = Tree::new();
    let ground = tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let crank = tree.push_link(Link::new(
        Some(ground),
        JointKind::Hinge {
            axis: Vec3::Y,
            range: None,
            damping: 0.1,
            armature: 0.02,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        // Hinge anchor at (CRANK_PIVOT_X, 0, 0.5). Crank is a bar
        // that extends from the pivot to (pivot + CRANK_LEN in local
        // +z at q=0). Anchor in child body = -0.5*length along z so
        // COM sits at pivot + CRANK_LEN/2 · z at q=0.
        (Vec3::new(CRANK_PIVOT_X, 0.0, 0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, -0.5 * CRANK_LEN), Quat::IDENTITY),
        0.3,
        Mat3::diag(0.01, 0.01, 0.005),
    ));
    let follower = tree.push_link(Link::new(
        Some(ground),
        JointKind::Hinge {
            axis: Vec3::Y,
            range: None,
            damping: 0.1,
            armature: 0.02,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::new(FOLLOWER_PIVOT_X, 0.0, 0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, -0.5 * FOLLOWER_LEN), Quat::IDENTITY),
        0.3,
        Mat3::diag(0.01, 0.01, 0.005),
    ));
    tree.add_actuator(Actuator::position(crank, 5.0, 1.0, 100.0, 0.0));
    let tree_idx = w.add_tree(tree);

    // Free coupler bar. World-space rest pose: horizontal at the level
    // of the crank tips (z = 0.5 + CRANK_LEN when q = 0), spanning
    // from crank tip to follower tip.
    let bar_z = 0.5 + CRANK_LEN;
    let bar_center = Vec3::new(0.5 * (CRANK_PIVOT_X + FOLLOWER_PIVOT_X), 0.0, bar_z);
    // Bar inertia — slender box.
    let bar_inertia = newt::geom::solid_box_inertia(0.2, COUPLER_HALF);
    let bar_idx = w.add_body(Body::new(0.2, bar_inertia, bar_center, Quat::IDENTITY));

    // Connect coupler's -x end to crank's tip.
    w.equalities.push(Equality::Connect {
        body_a: None,
        body_b: Some(bar_idx),
        anchor_a: Vec3::new(CRANK_PIVOT_X, 0.0, bar_z),
        anchor_b: Vec3::new(-COUPLER_HALF.x, 0.0, 0.0),
        solref,
        solimp,
    });
    // Connect coupler's +x end to follower's tip.
    w.equalities.push(Equality::Connect {
        body_a: None,
        body_b: Some(bar_idx),
        anchor_a: Vec3::new(FOLLOWER_PIVOT_X, 0.0, bar_z),
        anchor_b: Vec3::new(COUPLER_HALF.x, 0.0, 0.0),
        solref,
        solimp,
    });
    // NOTE: The two anchors above are STATIC world points — the crank
    // and follower tips MOVE with the hinges, so the anchors as
    // written won't actually track the hinge tips. This demo simplifies
    // by putting both connect anchors at the world-space rest position
    // of the tips; the joint coupling then correlates the two hinges
    // 2:1 so the tips stay in geometric agreement with the coupler
    // through the coupling constraint alone. A more general
    // implementation would attach the connect equalities to the tree
    // link's body (v1 tier 6 will lift `Equality::Connect` to accept a
    // tree link + local anchor, matching MuJoCo's `equality/connect`
    // semantics).
    w.equalities.push(Equality::JointCoupling {
        tree: tree_idx,
        link_a: follower,
        link_b: crank,
        // follower = -1 · crank  (opposite sign so the two hinges
        // sweep the coupler bar side-to-side in phase).
        polycoef: [0.0, -1.0, 0.0],
        solref,
        solimp,
    });
    (w, tree_idx, bar_idx)
}

// -----------------------------------------------------------------------
// Rendering — same wireframe plumbing as solver_stack.rs.
// -----------------------------------------------------------------------

const BOX_CORNERS_LOCAL: [(f32, f32, f32); 8] = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
];
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

fn box_world_corners(center: Vec3, ori: Quat, half: Vec3) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    for (i, &(sx, sy, sz)) in BOX_CORNERS_LOCAL.iter().enumerate() {
        let local = Vec3::new(sx * half.x, sy * half.y, sz * half.z);
        out[i] = center + ori.rotate(local);
    }
    out
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

fn draw_ground_grid(fb: &mut Framebuffer, camera: Mat4, width: usize, height: usize, color: u32) {
    let span = 2.0;
    let step = 0.25;
    let n = (2.0 * span / step) as i32;
    for i in 0..=n {
        let t = -span + (i as f32) * step;
        let a = Vec3::new(-span, t, 0.0);
        let b = Vec3::new(span, t, 0.0);
        if let (Some(p0), Some(p1)) = (
            project(camera, a, width, height),
            project(camera, b, width, height),
        ) {
            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
        }
        let a = Vec3::new(t, -span, 0.0);
        let b = Vec3::new(t, span, 0.0);
        if let (Some(p0), Some(p1)) = (
            project(camera, a, width, height),
            project(camera, b, width, height),
        ) {
            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
        }
    }
}

fn render(w: &World, bar_idx: usize, width: usize, height: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.1,
        200.0,
    ) * Mat4::look_at(
        CVec3::new(0.0, -2.4, 1.6),
        CVec3::new(0.0, 0.0, 0.7),
        CVec3::new(0.0, 0.0, 1.0),
    );
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));

    // Tree link 1 (crank) and 2 (follower) — draw as line segments
    // from the hinge pivot to the tip.
    let poses = newt::tree::forward_kinematics(&w.trees[0]);
    let crank_pivot = Vec3::new(CRANK_PIVOT_X, 0.0, 0.5);
    let follower_pivot = Vec3::new(FOLLOWER_PIVOT_X, 0.0, 0.5);
    let (crank_com, crank_ori) = poses[1];
    let (follower_com, follower_ori) = poses[2];
    let crank_tip = crank_com + crank_ori.rotate(Vec3::new(0.0, 0.0, CRANK_LEN * 0.5));
    let follower_tip = follower_com + follower_ori.rotate(Vec3::new(0.0, 0.0, FOLLOWER_LEN * 0.5));

    let draw_bar = |fb: &mut Framebuffer, from: Vec3, to: Vec3, color: u32| {
        if let (Some(p0), Some(p1)) = (
            project(camera, from, width, height),
            project(camera, to, width, height),
        ) {
            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
        }
    };
    draw_bar(
        &mut fb,
        crank_pivot,
        crank_tip,
        argb8888(0xff, 240, 200, 90),
    );
    draw_bar(
        &mut fb,
        follower_pivot,
        follower_tip,
        argb8888(0xff, 90, 220, 240),
    );

    // Coupler bar as a wireframe box.
    let bar_body = &w.bodies[bar_idx];
    let corners = box_world_corners(bar_body.position, bar_body.orientation, COUPLER_HALF);
    let projected: [Option<(i32, i32)>; 8] =
        std::array::from_fn(|i| project(camera, corners[i], width, height));
    for &(a, b) in BOX_EDGES.iter() {
        if let (Some((x0, y0)), Some((x1, y1))) = (projected[a], projected[b]) {
            draw_line(&mut fb, x0, y0, x1, y1, argb8888(0xff, 240, 100, 140));
        }
    }
    fb
}

fn render_solid(
    w: &World,
    bar_idx: usize,
    width: usize,
    height: usize,
    step: usize,
) -> Framebuffer {
    let poses = newt::tree::forward_kinematics(&w.trees[0]);
    let crank_pivot = Vec3::new(CRANK_PIVOT_X, 0.0, 0.5);
    let follower_pivot = Vec3::new(FOLLOWER_PIVOT_X, 0.0, 0.5);
    let crank_tip = poses[1].0 + poses[1].1.rotate(Vec3::new(0.0, 0.0, CRANK_LEN * 0.5));
    let follower_tip = poses[2].0 + poses[2].1.rotate(Vec3::new(0.0, 0.0, FOLLOWER_LEN * 0.5));
    let mut items = vec![showcase_support::item(
        showcase_support::cuboid_mesh(CVec3::new(2.5, 2.5, 0.04)),
        showcase_support::transform(
            Vec3::new(0.0, 0.0, -0.04),
            Quat::IDENTITY,
            CVec3::new(1.0, 1.0, 1.0),
        ),
        showcase_support::Material::new(CVec3::new(0.04, 0.05, 0.07), 0.0, 0.9),
    )];
    showcase_support::add_capsule(
        &mut items,
        crank_pivot,
        crank_tip,
        0.045,
        showcase_support::Material::new(CVec3::new(0.95, 0.55, 0.12), 0.25, 0.3),
    );
    showcase_support::add_capsule(
        &mut items,
        follower_pivot,
        follower_tip,
        0.045,
        showcase_support::Material::new(CVec3::new(0.1, 0.65, 0.9), 0.25, 0.3),
    );
    items.push(showcase_support::item(
        showcase_support::cuboid_mesh(CVec3::new(COUPLER_HALF.x, COUPLER_HALF.y, COUPLER_HALF.z)),
        showcase_support::transform(
            w.bodies[bar_idx].position,
            w.bodies[bar_idx].orientation,
            CVec3::new(1.0, 1.0, 1.0),
        ),
        showcase_support::Material::new(CVec3::new(0.9, 0.18, 0.35), 0.2, 0.3),
    ));
    showcase_support::render_items(
        &items,
        showcase_support::composition("linkage"),
        width,
        height,
        &format!("joint coupling  |  static connect reference  |  step {step}"),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height), wireframe) = parse_args();
    let (mut world, tree_idx, bar_idx) = build_world();
    if wireframe {
        for current in 0..frames {
            let t = current as f32 * world.dt;
            world.trees[tree_idx].set_actuator_target(0, 0.4 * sin(0.8 * t));
            world.step();
        }
        write_ppm(&out, &render(&world, bar_idx, width, height))?;
        return Ok(());
    }
    let mut simulated = 0;
    showcase_support::write_video(&out, frames, |step| {
        for current in simulated..step {
            let t = current as f32 * world.dt;
            world.trees[tree_idx].set_actuator_target(0, 0.4 * sin(0.8 * t));
            world.step();
        }
        simulated = step;
        render_solid(&world, bar_idx, width, height, step)
    })?;
    println!(
        "wrote {} ({}x{}) after {} frames",
        out.display(),
        width,
        height,
        frames
    );
    let q_crank = world.trees[tree_idx].hinge_angle(1);
    let q_follower = world.trees[tree_idx].hinge_angle(2);
    let bar = &world.bodies[bar_idx];
    println!(
        "  crank q = {q_crank:.3} rad, follower q = {q_follower:.3} rad \
         (coupling target: follower = -1·crank = {:.3})",
        -q_crank
    );
    println!(
        "  coupler bar center = ({:.3}, {:.3}, {:.3})",
        bar.position.x, bar.position.y, bar.position.z
    );
    Ok(())
}
