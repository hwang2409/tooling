//! Tier 4 demo: a 3-link commanded arm follows a three-waypoint target
//! sequence (reach up, reach sideways, settle) driven by PD position
//! servos. Wireframe PPM via chimy2's Framebuffer + Mat4 helpers, same
//! plumbing as `pendulum.rs`.
//!
//! Deterministic — hand-written trig in newt, fixed dt, fixed waypoint
//! schedule. Every parameter is compile-time visible so the render
//! reproduces bit-for-bit at the same `--frames` value across platforms.
//!
//! Run:
//! ```text
//! cargo run --release --example arm -- --frames 1800 --out /tmp/arm.ppm --size 640x360
//! sips -s format png /tmp/arm.ppm --out /tmp/arm.png
//! ```

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::actuator::PdServo;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, forward_kinematics, rk4_step};

use std::path::PathBuf;

fn parse_args() -> (usize, PathBuf, (usize, usize)) {
    let mut frames = 1800usize;
    let mut out = PathBuf::from("newt-arm.ppm");
    let mut size = (640usize, 360usize);
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
            _ => panic!("unknown arg: {a}"),
        }
    }
    (frames, out, size)
}

const L1: f32 = 0.5;
const L2: f32 = 0.5;
const L3: f32 = 0.5;
const M1: f32 = 1.0;
const M2: f32 = 0.8;
const M3: f32 = 0.6;

/// Three waypoints for the arm's shoulder / elbow / wrist joints (rad).
/// Chosen so each pose is legibly distinct at demo scale:
///   • "reach up"       — arm curls upward into +y-then-+z
///   • "reach sideways" — arm extends along +y and mildly folds
///   • "settle"         — back to hanging straight down
const WAYPOINTS: [[f32; 3]; 3] = [
    [1.8, -1.0, -0.5], // reach up
    [1.2, -0.4, 0.0],  // reach sideways
    [0.0, 0.0, 0.0],   // settle
];

fn build_arm() -> Tree {
    let mut tree = Tree::new();
    // Root: fixed anchor above the "ground" so the swing volume stays in
    // the camera frustum.
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, 1.6), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let lens = [L1, L2, L3];
    let masses = [M1, M2, M3];
    for i in 0..3 {
        let l = lens[i];
        let m = masses[i];
        let i_perp = (1.0 / 12.0) * m * l * l;
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, 0.0, -lens[i - 1] * 0.5)
        };
        tree.push_link(Link::new(
            Some(i),
            JointKind::hinge(Vec3::X),
            (parent_anchor, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
            m,
            Mat3::diag(i_perp, i_perp, 1e-6),
        ));
    }
    tree
}

fn attach_servos(tree: &mut Tree) -> [usize; 3] {
    // Reflected inertia estimates: m·L² for the point-mass-on-rod pattern.
    // Force clamp scales with the link's mass — biped-style bounds.
    let s1 = PdServo::from_dampratio(
        1,
        /*kp*/ 200.0,
        /*ζ*/ 1.0,
        M1 * L1 * L1,
        /*clamp*/ 60.0,
    );
    let s2 = PdServo::from_dampratio(2, 150.0, 1.0, M2 * L2 * L2, 40.0);
    let s3 = PdServo::from_dampratio(3, 100.0, 1.0, M3 * L3 * L3, 30.0);
    [
        tree.add_actuator(s1),
        tree.add_actuator(s2),
        tree.add_actuator(s3),
    ]
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

fn rod_endpoints(poses: &[(Vec3, Quat)], i: usize, l: f32) -> (Vec3, Vec3) {
    let (com, ori) = poses[i];
    let top = com + ori.rotate(Vec3::new(0.0, 0.0, l * 0.5));
    let bot = com + ori.rotate(Vec3::new(0.0, 0.0, -l * 0.5));
    (top, bot)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height)) = parse_args();
    let mut tree = build_arm();
    let servos = attach_servos(&mut tree);

    let dt = 0.005f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    // Split frames evenly across the three waypoints.
    let per_phase = frames / 3;

    // Trail: tip position sampled every frame.
    let mut trail: Vec<Vec3> = Vec::with_capacity(frames);
    let lens = [L1, L2, L3];
    let mut frame = 0usize;
    for (phase, targets) in WAYPOINTS.iter().enumerate() {
        for (i, &t) in targets.iter().enumerate() {
            tree.set_actuator_target(servos[i], t);
        }
        let stop = if phase + 1 == WAYPOINTS.len() {
            frames
        } else {
            (phase + 1) * per_phase
        };
        while frame < stop {
            rk4_step(&mut tree, g, dt, |_| vec![(Vec3::ZERO, Vec3::ZERO); 4]);
            let poses = forward_kinematics(&tree);
            let (_top, bot) = rod_endpoints(&poses, 3, lens[2]);
            trail.push(bot);
            frame += 1;
        }
    }

    // Render final frame: rods + trail.
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.05,
        100.0,
    ) * Mat4::look_at(
        // Camera on +x, looking at (0, 0, 1) — the y-z plane is the arm's
        // swing plane, y → image-right, z → image-up.
        CVec3::new(4.8, 0.0, 1.1),
        CVec3::new(0.0, 0.0, 1.0),
        CVec3::new(0.0, 0.0, 1.0),
    );

    // Trail — colour ramps by frame index so the temporal path reads at a
    // glance (cool early → warm late).
    for (i, w) in trail.windows(2).enumerate() {
        let t = i as f32 / trail.len().max(1) as f32;
        let r = (60.0 + 190.0 * t) as u8;
        let g = (60.0 + 60.0 * t) as u8;
        let b = (200.0 - 120.0 * t) as u8;
        if let (Some(p0), Some(p1)) = (
            project(camera, w[0], width, height),
            project(camera, w[1], width, height),
        ) {
            draw_line(&mut fb, p0.0, p0.1, p1.0, p1.1, argb8888(0xff, r, g, b));
        }
    }

    // Base pivot marker (small cross) so the anchor position is obvious.
    let poses = forward_kinematics(&tree);
    let base = poses[0].0; // fixed root COM (= anchor for link 1).
    if let Some((x, y)) = project(camera, base, width, height) {
        let s = 5;
        draw_line(&mut fb, x - s, y, x + s, y, argb8888(0xff, 220, 220, 220));
        draw_line(&mut fb, x, y - s, x, y + s, argb8888(0xff, 220, 220, 220));
    }

    // Rods at final frame — three distinct colours so the joint chain is
    // easy to trace.
    let colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 220, 120, 220),
    ];
    for (i, &l) in [L1, L2, L3].iter().enumerate() {
        let (top, bot) = rod_endpoints(&poses, i + 1, l);
        if let (Some(a), Some(b)) = (
            project(camera, top, width, height),
            project(camera, bot, width, height),
        ) {
            draw_line(&mut fb, a.0, a.1, b.0, b.1, colors[i]);
        }
    }

    write_ppm(&out, &fb)?;
    println!(
        "wrote {} ({}x{}) — final q = ({:.3}, {:.3}, {:.3})",
        out.display(),
        width,
        height,
        tree.hinge_angle(1),
        tree.hinge_angle(2),
        tree.hinge_angle(3),
    );
    Ok(())
}
