//! Tier 4 demo, tier-5 upgrade: a 3-link commanded arm follows a
//! three-waypoint target sequence (reach up → reach sideways → settle)
//! driven by PD position servos. Wireframe PPM via chimy2, same rendering
//! plumbing as `pendulum.rs`.
//!
//! # What changed in tier 5
//!
//! The kinematic tree used to be hand-built in this file. Now the arm
//! model — link masses, inertias, joint axes, servo gains, force clamps —
//! all lives in [`newt/models/arm.json`](../models/arm.json) and is loaded
//! through [`newt::model::load_from_path`]. This demo is now the reference
//! usage of the tier-5 loader. Actuators are looked up by NAME
//! (`shoulder_servo` / `elbow_servo` / `wrist_servo`) rather than by
//! index, and the tip position that draws the trail comes from the
//! `tip` [`newt::model::Site`] defined in the same JSON.
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

use newt::math::{Quat, Vec3};
use newt::model::{Scene, load_from_path};
use newt::tree::{forward_kinematics, rk4_step};

use std::path::PathBuf;

fn parse_args() -> (usize, PathBuf, (usize, usize), PathBuf) {
    let mut frames = 1800usize;
    let mut out = PathBuf::from("newt-arm.ppm");
    let mut size = (640usize, 360usize);
    let mut model = PathBuf::from("models/arm.json");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => frames = args.next().unwrap().parse().unwrap(),
            "--out" => out = PathBuf::from(args.next().unwrap()),
            "--model" => model = PathBuf::from(args.next().unwrap()),
            "--size" => {
                let s = args.next().unwrap();
                let (w, h) = s.split_once('x').expect("--size WxH");
                size = (w.parse().unwrap(), h.parse().unwrap());
            }
            _ => panic!("unknown arg: {a}"),
        }
    }
    (frames, out, size, model)
}

/// The three-waypoint reach sequence (shoulder / elbow / wrist, radians).
///   * reach up
///   * reach sideways
///   * settle
const WAYPOINTS: [[f32; 3]; 3] = [[1.8, -1.0, -0.5], [1.2, -0.4, 0.0], [0.0, 0.0, 0.0]];

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

fn rod_endpoints(poses: &[(Vec3, Quat)], link_idx: usize, l: f32) -> (Vec3, Vec3) {
    let (com, ori) = poses[link_idx];
    let top = com + ori.rotate(Vec3::new(0.0, 0.0, l * 0.5));
    let bot = com + ori.rotate(Vec3::new(0.0, 0.0, -l * 0.5));
    (top, bot)
}

fn tip_world(scene: &Scene) -> Vec3 {
    scene
        .site_pose("tip")
        .expect("arm.json must define a tip site")
        .0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height), model_path) = parse_args();

    let mut scene = load_from_path(&model_path)
        .unwrap_or_else(|e| panic!("load {}: {e}", model_path.display()));
    let arm_idx = *scene
        .trees_by_name
        .get("arm")
        .expect("model must define a tree named \"arm\"");
    let servo_ids: [(usize, usize); 3] =
        ["shoulder_servo", "elbow_servo", "wrist_servo"].map(|n| {
            *scene
                .actuators_by_name
                .get(n)
                .unwrap_or_else(|| panic!("actuator {n} missing from model"))
        });

    // Rod length is model-defined: each hinge child has
    // joint_offset_in_child.translation = (0, 0, L/2). Read L back so the
    // rendering stays honest to the model.
    let rod_lens: [f32; 3] = {
        let tree = &scene.world.trees[arm_idx];
        [1, 2, 3].map(|i| tree.links[i].joint_offset_in_child.0.z * 2.0)
    };

    let dt = scene.world.dt;
    let g = scene.world.gravity;
    let per_phase = frames / 3;

    let mut trail: Vec<Vec3> = Vec::with_capacity(frames);
    let mut frame = 0usize;
    for (phase, targets) in WAYPOINTS.iter().enumerate() {
        for (i, &t) in targets.iter().enumerate() {
            let (t_idx, a_idx) = servo_ids[i];
            assert_eq!(t_idx, arm_idx);
            scene.world.trees[t_idx].set_actuator_target(a_idx, t);
        }
        let stop = if phase + 1 == WAYPOINTS.len() {
            frames
        } else {
            (phase + 1) * per_phase
        };
        while frame < stop {
            rk4_step(&mut scene.world.trees[arm_idx], g, dt, |_| {
                vec![(Vec3::ZERO, Vec3::ZERO); 4]
            });
            trail.push(tip_world(&scene));
            frame += 1;
        }
    }

    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.05,
        100.0,
    ) * Mat4::look_at(
        // Camera on +x, looking at (0, 0, 1) — the y-z plane is the arm's
        // swing plane, y → image-right, z → image-up. Matches the tier-4
        // camera exactly.
        CVec3::new(4.8, 0.0, 1.1),
        CVec3::new(0.0, 0.0, 1.0),
        CVec3::new(0.0, 0.0, 1.0),
    );

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

    let poses = forward_kinematics(&scene.world.trees[arm_idx]);
    let base = poses[0].0;
    if let Some((x, y)) = project(camera, base, width, height) {
        let s = 5;
        draw_line(&mut fb, x - s, y, x + s, y, argb8888(0xff, 220, 220, 220));
        draw_line(&mut fb, x, y - s, x, y + s, argb8888(0xff, 220, 220, 220));
    }

    let colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 220, 120, 220),
    ];
    for (i, &l) in rod_lens.iter().enumerate() {
        let (top, bot) = rod_endpoints(&poses, i + 1, l);
        if let (Some(a), Some(b)) = (
            project(camera, top, width, height),
            project(camera, bot, width, height),
        ) {
            draw_line(&mut fb, a.0, a.1, b.0, b.1, colors[i]);
        }
    }

    write_ppm(&out, &fb)?;
    let tree = &scene.world.trees[arm_idx];
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
