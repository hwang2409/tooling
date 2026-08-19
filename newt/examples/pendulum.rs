//! Tier 3 demo: double pendulum with a fading trail traced by the tip of
//! rod 2. Wireframe PPM via chimy2's Framebuffer + Mat4 helpers, same
//! plumbing as `tumble.rs` / `stack.rs`.
//!
//! Run:
//! ```text
//! cargo run --release --example pendulum -- --frames 900 --wireframe --out /tmp/pendulum.ppm
//! ```

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, forward_kinematics, rk4_step};

use std::path::PathBuf;

mod showcase_support;

fn parse_args() -> (usize, PathBuf, (usize, usize), bool) {
    let mut frames = 900usize;
    let default_out = PathBuf::from("newt-pendulum.mp4");
    let mut out = default_out.clone();
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
    if wireframe && out == default_out {
        out.set_extension("ppm");
    }
    (frames, out, size, wireframe)
}

const L1: f32 = 0.9;
const L2: f32 = 0.6;
const M1: f32 = 1.3;
const M2: f32 = 0.7;

fn build_tree() -> Tree {
    let mut tree = Tree::new();
    // Pivot at world (0, 0, 1.5) so the pendulum swings above the y=0 plane
    // and stays in the camera frustum.
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, 1.5), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let i1 = (1.0 / 12.0) * M1 * L1 * L1;
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, L1 * 0.5), Quat::IDENTITY),
        M1,
        Mat3::diag(i1, i1, 1e-6),
    ));
    let i2 = (1.0 / 12.0) * M2 * L2 * L2;
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.0, 0.0, -L1 * 0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, L2 * 0.5), Quat::IDENTITY),
        M2,
        Mat3::diag(i2, i2, 1e-6),
    ));
    // Moderate starting angles: the chaotic regime kicks in but the tip
    // draws a rich curve rather than winding up in fast rotations.
    tree.set_hinge_angle(1, 1.4);
    tree.set_hinge_angle(2, -0.8);
    tree
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

fn rod_endpoints(tree: &Tree, poses: &[(Vec3, Quat)], i: usize, l: f32) -> (Vec3, Vec3) {
    let (com, ori) = poses[i];
    let top = com + ori.rotate(Vec3::new(0.0, 0.0, l * 0.5));
    let bot = com + ori.rotate(Vec3::new(0.0, 0.0, -l * 0.5));
    let _ = tree; // unused; kept for future variants
    (top, bot)
}

fn render_frame(tree: &Tree, trail: &[Vec3], width: usize, height: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.05,
        100.0,
    ) * Mat4::look_at(
        // Camera on +x, looking at origin — the y-z plane is the pendulum's
        // swing plane, so y→image-right and z→image-up.
        CVec3::new(4.5, 0.0, 1.0),
        CVec3::new(0.0, 0.0, 0.6),
        CVec3::new(0.0, 0.0, 1.0),
    );

    // Trail — fade from cool to warm along the run.
    for (i, w) in trail.windows(2).enumerate() {
        let t = i as f32 / trail.len().max(1) as f32;
        let r = (60.0 + 190.0 * t) as u32;
        let g = (60.0 + 60.0 * t) as u32;
        let b = (200.0 - 120.0 * t) as u32;
        if let (Some(p0), Some(p1)) = (
            project(camera, w[0], width, height),
            project(camera, w[1], width, height),
        ) {
            draw_line(
                &mut fb,
                p0.0,
                p0.1,
                p1.0,
                p1.1,
                argb8888(0xff, r as u8, g as u8, b as u8),
            );
        }
    }

    // Rods at final frame.
    let poses = forward_kinematics(tree);
    let colors = [argb8888(0xff, 240, 200, 90), argb8888(0xff, 90, 220, 240)];
    for (i, &l) in [L1, L2].iter().enumerate() {
        let (top, bot) = rod_endpoints(tree, &poses, i + 1, l);
        if let (Some(a), Some(b)) = (
            project(camera, top, width, height),
            project(camera, bot, width, height),
        ) {
            draw_line(&mut fb, a.0, a.1, b.0, b.1, colors[i]);
        }
    }

    fb
}

fn render_solid(
    tree: &Tree,
    trail: &[Vec3],
    width: usize,
    height: usize,
    step: usize,
) -> Framebuffer {
    let poses = forward_kinematics(tree);
    let mut items = vec![showcase_support::item(
        showcase_support::cuboid_mesh(chimy2::math::Vec3::new(3.0, 3.0, 0.04)),
        showcase_support::transform(
            Vec3::new(0.0, 0.0, -0.04),
            Quat::IDENTITY,
            chimy2::math::Vec3::new(1.0, 1.0, 1.0),
        ),
        showcase_support::Material::new(chimy2::math::Vec3::new(0.04, 0.05, 0.07), 0.0, 0.9),
    )];
    for (index, &length) in [L1, L2].iter().enumerate() {
        let (top, bottom) = rod_endpoints(tree, &poses, index + 1, length);
        showcase_support::add_capsule(
            &mut items,
            top,
            bottom,
            0.055,
            showcase_support::Material::new(
                if index == 0 {
                    chimy2::math::Vec3::new(0.95, 0.55, 0.12)
                } else {
                    chimy2::math::Vec3::new(0.1, 0.65, 0.9)
                },
                0.2,
                0.3,
            ),
        );
    }
    for point in trail.iter().step_by(20) {
        showcase_support::add_marker(
            &mut items,
            *point,
            0.012,
            showcase_support::Material::new(chimy2::math::Vec3::new(0.9, 0.2, 0.35), 0.1, 0.4),
        );
    }
    showcase_support::render_items(
        &items,
        showcase_support::composition("pendulum"),
        width,
        height,
        &format!("double pendulum  |  step {step}"),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height), wireframe) = parse_args();
    let mut tree = build_tree();
    let dt = 0.005f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    let mut trail = Vec::with_capacity(frames);
    if wireframe {
        for _ in 0..frames {
            rk4_step(&mut tree, g, dt, |_| vec![(Vec3::ZERO, Vec3::ZERO); 3]);
            let poses = forward_kinematics(&tree);
            trail.push(rod_endpoints(&tree, &poses, 2, L2).1);
        }
        write_ppm(&out, &render_frame(&tree, &trail, width, height))?;
        return Ok(());
    }
    let mut simulated = 0;
    showcase_support::write_video(&out, frames, |step| {
        for _ in simulated..step {
            rk4_step(&mut tree, g, dt, |_| vec![(Vec3::ZERO, Vec3::ZERO); 3]);
            let poses = forward_kinematics(&tree);
            trail.push(rod_endpoints(&tree, &poses, 2, L2).1);
        }
        simulated = step;
        render_solid(&tree, &trail, width, height, step)
    })?;
    println!(
        "wrote {} ({}x{}) — final θ1={:.3} θ2={:.3}",
        out.display(),
        width,
        height,
        tree.hinge_angle(1),
        tree.hinge_angle(2)
    );
    Ok(())
}
