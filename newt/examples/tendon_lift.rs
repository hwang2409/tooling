//! v2 tier 3 demo: a motor-driven spatial tendon lifts a hanging box up
//! and around a fixed sphere obstacle. Draws the tendon path (straight
//! segment + arc) alongside the box and sphere.
//!
//! Run:
//! ```text
//! cargo run --release --example tendon_lift -- --frames 900 --out /tmp/tendon_lift.ppm
//! ```

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::actuator::Actuator;
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{SolverConfig, SolverMode};
use newt::tendon::{SpatialTendonSite, Tendon, WrapSphere, tendon_kinematics};
use newt::tree::{Link, Tree, forward_kinematics};
use newt::world::World;

use std::path::PathBuf;

const SPHERE_CENTER: Vec3 = Vec3::new(0.5, 0.0, -0.4);
const SPHERE_RADIUS: f32 = 0.3;
const ANCHOR_WORLD: Vec3 = Vec3::new(0.0, 0.0, 0.0);
const BOX_SIZE: f32 = 0.15;
const REST_LENGTH: f32 = 1.5;

fn parse_args() -> (usize, PathBuf, (usize, usize)) {
    let mut frames = 900usize;
    let mut out = PathBuf::from("newt-tendon-lift.ppm");
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

fn build_world() -> World {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        ..SolverConfig::DEFAULT
    };
    let mut tree = Tree::new();
    // Root at world origin (holds the anchor + wrap sphere by proxy —
    // they're world-static in the tendon; the fixed root is just here
    // to give the box a parent).
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Box on a slide axis (0, 0, -1) — positive q lowers the box.
    // Body sits at initial position (1.0, 0, -0.9) below the sphere.
    tree.push_link(Link::new(
        Some(0),
        JointKind::Slide {
            axis: Vec3::new(0.0, 0.0, -1.0),
            range: None,
            damping: 4.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(1.0, 0.0, -0.9), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    // Spatial tendon: anchor → sphere wrap → box site.
    let mut cable = Tendon::spatial(
        vec![
            SpatialTendonSite {
                link: Some(0),
                position_local: ANCHOR_WORLD,
            },
            SpatialTendonSite {
                link: Some(1),
                position_local: Vec3::ZERO,
            },
        ],
        vec![Some(WrapSphere {
            link: Some(0),
            center_local: SPHERE_CENTER,
            radius: SPHERE_RADIUS,
            side_hint_world: None,
        })],
    );
    cable.springlength = Some(REST_LENGTH);
    cable.stiffness = 60.0;
    cable.damping = 6.0;
    tree.add_tendon(cable);
    // Motor drives the tendon: negative ctrl shortens the tendon
    // (pulls the box UP toward the anchor).
    let motor = Actuator::motor(0, 1.0, 0.0).on_tendon(0);
    let aid = tree.add_actuator(motor);
    tree.set_actuator_target(aid, -40.0);
    world.add_tree(tree);
    world
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

fn draw_box(
    fb: &mut Framebuffer,
    camera: Mat4,
    w: usize,
    h: usize,
    center: Vec3,
    size: f32,
    color: u32,
) {
    let s = size;
    let verts = [
        center + Vec3::new(-s, -s, -s),
        center + Vec3::new(s, -s, -s),
        center + Vec3::new(s, s, -s),
        center + Vec3::new(-s, s, -s),
        center + Vec3::new(-s, -s, s),
        center + Vec3::new(s, -s, s),
        center + Vec3::new(s, s, s),
        center + Vec3::new(-s, s, s),
    ];
    let edges = [
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
    for &(a, b) in &edges {
        if let (Some(p0), Some(p1)) = (
            project(camera, verts[a], w, h),
            project(camera, verts[b], w, h),
        ) {
            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
        }
    }
}

fn draw_sphere_wire(
    fb: &mut Framebuffer,
    camera: Mat4,
    w: usize,
    h: usize,
    center: Vec3,
    radius: f32,
    color: u32,
) {
    // Wire sphere as three orthogonal circles.
    for axis in 0..3 {
        let mut prev: Option<(i32, i32)> = None;
        for i in 0..=32 {
            let t = i as f32 * (2.0 * std::f32::consts::PI / 32.0);
            let (c, s) = (newt::math::cos(t), newt::math::sin(t));
            let p = match axis {
                0 => center + Vec3::new(0.0, c * radius, s * radius),
                1 => center + Vec3::new(c * radius, 0.0, s * radius),
                _ => center + Vec3::new(c * radius, s * radius, 0.0),
            };
            let cur = project(camera, p, w, h);
            if let (Some(pa), Some(pb)) = (prev, cur) {
                draw_line(fb, pa.0, pa.1, pb.0, pb.1, color);
            }
            prev = cur;
        }
    }
}

fn draw_tendon_path(
    fb: &mut Framebuffer,
    camera: Mat4,
    w: usize,
    h: usize,
    world: &World,
    color: u32,
) {
    // Reconstruct the tendon world-path from current state. Anchor at
    // ANCHOR_WORLD, box site at box world position, wrap around
    // SPHERE_CENTER with SPHERE_RADIUS. If wrap engaged, draw
    // anchor→T_A, arc T_A→T_B, T_B→box; else straight.
    let poses = forward_kinematics(&world.trees[0]);
    let box_com = poses[1].0;
    let a = ANCHOR_WORLD;
    let b = box_com;
    let c = SPHERE_CENTER;
    let d_ac = a - c;
    let d_bc = b - c;
    let r = SPHERE_RADIUS;
    let ab = b - a;
    let ab_len2 = ab.length_squared();
    if ab_len2 == 0.0 {
        return;
    }
    let t_closest = (-(a - c)).dot(ab) / ab_len2;
    let p_closest = a + ab * t_closest;
    let perp_sq = (p_closest - c).length_squared();
    let inside = (0.0..=1.0).contains(&t_closest);
    let d_a2 = d_ac.length_squared();
    let d_b2 = d_bc.length_squared();
    if !inside || perp_sq >= r * r || d_a2 <= r * r || d_b2 <= r * r {
        if let (Some(pa), Some(pb)) = (project(camera, a, w, h), project(camera, b, w, h)) {
            draw_line(fb, pa.0, pa.1, pb.0, pb.1, color);
        }
        return;
    }
    // Wrap engaged — approximate the arc by sampling along the plane.
    let d_a = d_a2.sqrt();
    let d_b = d_b2.sqrt();
    let x_a_hat = d_ac / d_a;
    let cb_perp = d_bc - x_a_hat * (d_bc.dot(x_a_hat));
    let cb_perp_len = cb_perp.length();
    if cb_perp_len == 0.0 {
        return;
    }
    let y_a_hat = cb_perp / cb_perp_len;
    let t_a = (d_a2 - r * r).sqrt();
    let t_a_point = c + x_a_hat * (r * r / d_a) + y_a_hat * (r * t_a / d_a);
    let x_b_hat = d_bc / d_b;
    let ca_perp = d_ac - x_b_hat * (d_ac.dot(x_b_hat));
    let ca_perp_len = ca_perp.length();
    if ca_perp_len == 0.0 {
        return;
    }
    let y_b_hat = ca_perp / ca_perp_len;
    let t_b = (d_b2 - r * r).sqrt();
    let t_b_point = c + x_b_hat * (r * r / d_b) + y_b_hat * (r * t_b / d_b);
    // Straight A → T_A.
    if let (Some(p0), Some(p1)) = (project(camera, a, w, h), project(camera, t_a_point, w, h)) {
        draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
    }
    // Arc T_A → T_B on the sphere: 16-segment polyline.
    let mut prev = project(camera, t_a_point, w, h);
    for i in 1..=16 {
        let t = i as f32 / 16.0;
        // Slerp between T_A and T_B on the sphere: normalize + interpolate
        // + normalize back is a cheap approximation.
        let u = (t_a_point - c).normalize();
        let v = (t_b_point - c).normalize();
        let mid = u * (1.0 - t) + v * t;
        let p = c + mid.normalize() * r;
        let cur = project(camera, p, w, h);
        if let (Some(a), Some(b)) = (prev, cur) {
            draw_line(fb, a.0, a.1, b.0, b.1, color);
        }
        prev = cur;
    }
    // Straight T_B → B.
    if let (Some(p0), Some(p1)) = (project(camera, t_b_point, w, h), project(camera, b, w, h)) {
        draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height)) = parse_args();
    let mut world = build_world();
    for _ in 0..frames {
        world.step();
    }
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.05,
        100.0,
    ) * Mat4::look_at(
        CVec3::new(3.0, -3.5, 1.2),
        CVec3::new(0.5, 0.0, -0.5),
        CVec3::new(0.0, 0.0, 1.0),
    );
    // Sphere obstacle.
    draw_sphere_wire(
        &mut fb,
        camera,
        width,
        height,
        SPHERE_CENTER,
        SPHERE_RADIUS,
        argb8888(0xff, 90, 90, 140),
    );
    // Anchor point (small crosshair).
    if let Some(p) = project(camera, ANCHOR_WORLD, width, height) {
        for d in -4..=4 {
            let color = argb8888(0xff, 240, 240, 240);
            if p.0 + d >= 0 && p.0 + d < width as i32 && p.1 >= 0 && p.1 < height as i32 {
                fb.put_pixel((p.0 + d) as usize, p.1 as usize, color);
            }
            if p.1 + d >= 0 && p.1 + d < height as i32 && p.0 >= 0 && p.0 < width as i32 {
                fb.put_pixel(p.0 as usize, (p.1 + d) as usize, color);
            }
        }
    }
    // Box.
    let poses = forward_kinematics(&world.trees[0]);
    let box_com = poses[1].0;
    draw_box(
        &mut fb,
        camera,
        width,
        height,
        box_com,
        BOX_SIZE,
        argb8888(0xff, 240, 200, 90),
    );
    // Tendon path.
    draw_tendon_path(
        &mut fb,
        camera,
        width,
        height,
        &world,
        argb8888(0xff, 90, 220, 240),
    );
    write_ppm(&out, &fb)?;
    let kin = tendon_kinematics(&world.trees[0].tendons[0], &world.trees[0], &poses);
    println!(
        "wrote {} ({}x{}) — final slide={:.3} m tendon length={:.3} m",
        out.display(),
        width,
        height,
        world.trees[0].slide_position(1),
        kin.length,
    );
    Ok(())
}
