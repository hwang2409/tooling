//! v1 tier-1 demo: cart on a horizontal slide + pole on a hinge, with
//! two actuator modes — the visual pin that slide and hinge joints
//! cooperate through ABA AND that the v2 tier-2 actuator flavors wire
//! into that path correctly.
//!
//! - default (position mode): a PD position servo on the SLIDE holds
//!   the cart near x = 0 while the pole swings freely.
//! - `--velocity`: a VELOCITY actuator on the SLIDE tracks a
//!   sinusoidal velocity profile so the cart wiggles left-right and
//!   drags the pole through Coriolis coupling. A wire-through bug on
//!   the velocity flavor (wrong bias sign, gear ignored, act-vs-ctrl
//!   swap) shows up as either a runaway cart or a dead one.
//!
//! Run:
//! ```text
//! cargo run --release --example cartpole -- --frames 900 --out /tmp/cartpole.mp4
//! cargo run --release --example cartpole -- --velocity --frames 900 --out /tmp/cartpole_vel.mp4
//! cargo run --release --example cartpole -- --frames 900 --still /tmp/cartpole.ppm
//! ```
//!
//! Rendering: solid shaded chimy2 meshes. The camera sits on +Y and looks at
//! the swing plane.

mod showcase_support;

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::actuator::Actuator;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, forward_kinematics, rk4_step};

use std::path::PathBuf;

const CART_M: f32 = 1.5;
const POLE_M: f32 = 0.5;
const POLE_L: f32 = 0.9;
const H: f32 = 1.1; // cart height above the ground plane
const G: f32 = 9.81;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Position,
    Velocity,
}

struct Args {
    frames: usize,
    out: PathBuf,
    size: (usize, usize),
    mode: Mode,
    frames_dir: Option<PathBuf>,
    still: Option<PathBuf>,
    wireframe: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        frames: 900,
        out: PathBuf::from("newt-cartpole.mp4"),
        size: (640, 360),
        mode: Mode::Position,
        frames_dir: None,
        still: None,
        wireframe: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--frames" => a.frames = it.next().unwrap().parse().unwrap(),
            "--out" => a.out = PathBuf::from(it.next().unwrap()),
            "--size" => {
                let s = it.next().unwrap();
                let (w, h) = s.split_once('x').expect("--size WxH");
                a.size = (w.parse().unwrap(), h.parse().unwrap());
            }
            "--velocity" => a.mode = Mode::Velocity,
            "--position" => a.mode = Mode::Position,
            "--frames-dir" => a.frames_dir = Some(PathBuf::from(it.next().unwrap())),
            "--still" => a.still = Some(PathBuf::from(it.next().unwrap())),
            "--wireframe" => a.wireframe = true,
            _ => panic!("unknown arg: {arg}"),
        }
    }
    a
}

fn build_cartpole(mode: Mode) -> (Tree, usize) {
    let mut tree = Tree::new();
    // Fixed root at (0, 0, H).
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, H), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Cart: slide along +X. Small-but-positive inertia; cart never rotates.
    tree.push_link(Link::new(
        Some(0),
        JointKind::slide(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        CART_M,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    // Pole: uniform rod along child +Z, hinge about +Y (swings in x-z).
    let i_perp = (1.0 / 12.0) * POLE_M * POLE_L * POLE_L;
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::Y),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, POLE_L * 0.5), Quat::IDENTITY),
        POLE_M,
        Mat3::diag(i_perp, i_perp, 1e-6),
    ));
    let actuator_idx = match mode {
        Mode::Position => {
            // PD position servo on the SLIDE: hold cart at x = 0.
            // Reflected inertia estimate = CART_M + POLE_M so the pole's
            // worst-case contribution is folded into the damping choice.
            tree.add_actuator(Actuator::position_from_dampratio(
                1,
                60.0,            // kp — N per m of error
                1.0,             // critical damping
                CART_M + POLE_M, // reflected inertia estimate (kg)
                30.0,            // force clamp (N)
            ))
        }
        Mode::Velocity => {
            // Velocity actuator on the SLIDE: kv=25 N per m/s of error,
            // clamped to ±30 N. The main-loop rewrites `ctrl` each step
            // to a sinusoidal target rate — the actuator tracks it.
            tree.add_actuator(Actuator::velocity(1, /*kv*/ 25.0, /*clamp*/ 30.0))
        }
    };
    // Start pole tilted so the demo has motion out of the gate.
    tree.set_hinge_angle(2, 0.45);
    (tree, actuator_idx)
}

fn mode_str(mode: Mode) -> &'static str {
    match mode {
        Mode::Position => "position",
        Mode::Velocity => "velocity",
    }
}

fn render_cartpole_frame(
    tree: &Tree,
    width: usize,
    height: usize,
    step: usize,
    mode: Mode,
) -> Framebuffer {
    let poses = forward_kinematics(tree);
    let (cart_com, _) = poses[1];
    let (pole_com, pole_ori) = poses[2];
    let pole_tip = pole_com + pole_ori.rotate(Vec3::new(0.0, 0.0, -POLE_L * 0.5));
    let mut items = vec![showcase_support::item(
        showcase_support::cuboid_mesh(chimy2::math::Vec3::new(3.0, 2.0, 0.04)),
        chimy2::math::Mat4::translate(chimy2::math::Vec3::new(0.0, 0.0, -0.04)),
        showcase_support::Material::new(chimy2::math::Vec3::new(0.04, 0.05, 0.07), 0.0, 0.9),
    )];
    items.push(showcase_support::item(
        showcase_support::cuboid_mesh(chimy2::math::Vec3::new(0.28, 0.22, 0.09)),
        chimy2::math::Mat4::translate(showcase_support::to_cvec(cart_com)),
        showcase_support::Material::new(chimy2::math::Vec3::new(0.12, 0.48, 0.78), 0.45, 0.25),
    ));
    showcase_support::add_capsule(
        &mut items,
        cart_com,
        pole_tip,
        0.045,
        showcase_support::Material::new(chimy2::math::Vec3::new(0.92, 0.42, 0.08), 0.5, 0.2),
    );
    showcase_support::add_marker(
        &mut items,
        cart_com,
        0.08,
        showcase_support::Material::new(chimy2::math::Vec3::new(0.95, 0.75, 0.16), 0.35, 0.18),
    );
    showcase_support::render_items(
        &items,
        showcase_support::composition("cartpole"),
        width,
        height,
        &format!("cartpole  |  step {step}  |  {} mode", mode_str(mode)),
    )
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let (frames, out, (width, height), mode) = (args.frames, args.out, args.size, args.mode);
    let (mut tree, actuator_idx) = build_cartpole(mode);
    let dt = 0.005f32;
    let g_vec = Vec3::new(0.0, 0.0, -G);

    // Trail: pole tip (world) across the whole run.
    let mut trail: Vec<Vec3> = Vec::with_capacity(frames);
    // Cart-position samples across the run (draws a translucent slide track).
    let mut cart_trail: Vec<Vec3> = Vec::with_capacity(frames);
    let video_mode = !args.wireframe && args.still.is_none() && args.frames_dir.is_none();
    let mut video = video_mode
        .then(|| showcase_support::VideoWriter::new(&out))
        .transpose()?;
    for step in 0..frames {
        if let Mode::Velocity = mode {
            // Track a sinusoidal velocity profile: 0.6·sin(2π·t/1.5s) m/s
            // — one full cycle every 1.5 s. Deterministic + libm-free:
            // use the crate's math::sin, not std::f32::sin.
            let t = step as f32 * dt;
            let phase = t / 1.5;
            let target_vel = 0.6 * newt::math::sin(2.0 * std::f32::consts::PI * phase);
            tree.set_actuator_target(actuator_idx, target_vel);
        }
        rk4_step(&mut tree, g_vec, dt, |_| vec![(Vec3::ZERO, Vec3::ZERO); 3]);
        let poses = forward_kinematics(&tree);
        let (cart_com, _) = poses[1];
        let (pole_com, pole_ori) = poses[2];
        let pole_tip = pole_com + pole_ori.rotate(Vec3::new(0.0, 0.0, -POLE_L * 0.5));
        trail.push(pole_tip);
        cart_trail.push(cart_com);
        if let Some(writer) = video.as_mut()
            && ((step + 1) % showcase_support::SIM_STEPS_PER_VIDEO_FRAME == 0 || step + 1 == frames)
        {
            writer.push(&render_cartpole_frame(&tree, width, height, step + 1, mode))?;
        }
    }

    if let Some(writer) = video {
        writer.finish()?;
        println!(
            "wrote {} ({}x{}, {} fps)",
            out.display(),
            width,
            height,
            showcase_support::VIDEO_FPS
        );
        return Ok(());
    }

    if !args.wireframe {
        let path = args
            .still
            .or_else(|| {
                args.frames_dir
                    .map(|directory| directory.join("frame-00.ppm"))
            })
            .unwrap_or(out.clone());
        write_ppm(
            path,
            &render_cartpole_frame(&tree, width, height, frames, mode),
        )?;
        return Ok(());
    }

    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.05,
        100.0,
    ) * Mat4::look_at(
        // Camera on +Y looking at the cart's rest height.
        CVec3::new(0.0, 4.5, H),
        CVec3::new(0.0, 0.0, H),
        CVec3::new(0.0, 0.0, 1.0),
    );

    // Ground: a horizontal line at z = 0 spanning [-2, 2] on x.
    let g_col = argb8888(0xff, 90, 90, 110);
    if let (Some(a), Some(b)) = (
        project(camera, Vec3::new(-2.0, 0.0, 0.0), width, height),
        project(camera, Vec3::new(2.0, 0.0, 0.0), width, height),
    ) {
        draw_line(&mut fb, a.0, a.1, b.0, b.1, g_col);
    }

    // Slide track: gray line at cart height, spanning the range the cart
    // visited.
    if let (Some(min_x), Some(max_x)) = (
        cart_trail
            .iter()
            .map(|p| p.x)
            .fold(None, |m: Option<f32>, x| Some(m.map_or(x, |v| v.min(x)))),
        cart_trail
            .iter()
            .map(|p| p.x)
            .fold(None, |m: Option<f32>, x| Some(m.map_or(x, |v| v.max(x)))),
    ) {
        let track_lo = min_x - 0.15;
        let track_hi = max_x + 0.15;
        if let (Some(a), Some(b)) = (
            project(camera, Vec3::new(track_lo, 0.0, H), width, height),
            project(camera, Vec3::new(track_hi, 0.0, H), width, height),
        ) {
            draw_line(&mut fb, a.0, a.1, b.0, b.1, argb8888(0xff, 70, 80, 110));
        }
    }

    // Pole tip trail — cool → warm along the run.
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

    // Final frame: draw cart (small rectangle in x-z plane, centered at
    // cart COM) + pole (line from cart to pole tip).
    let poses = forward_kinematics(&tree);
    let (cart_com, _) = poses[1];
    let cart_col = argb8888(0xff, 220, 220, 220);
    let cart_half_x = 0.14f32;
    let cart_half_z = 0.06f32;
    let cart_corners = [
        Vec3::new(cart_com.x - cart_half_x, 0.0, cart_com.z - cart_half_z),
        Vec3::new(cart_com.x + cart_half_x, 0.0, cart_com.z - cart_half_z),
        Vec3::new(cart_com.x + cart_half_x, 0.0, cart_com.z + cart_half_z),
        Vec3::new(cart_com.x - cart_half_x, 0.0, cart_com.z + cart_half_z),
    ];
    for i in 0..4 {
        let a = cart_corners[i];
        let b = cart_corners[(i + 1) % 4];
        if let (Some(pa), Some(pb)) = (
            project(camera, a, width, height),
            project(camera, b, width, height),
        ) {
            draw_line(&mut fb, pa.0, pa.1, pb.0, pb.1, cart_col);
        }
    }
    // Pole: line from cart COM (pivot) to pole tip.
    let (pole_com, pole_ori) = poses[2];
    let pole_tip = pole_com + pole_ori.rotate(Vec3::new(0.0, 0.0, -POLE_L * 0.5));
    let pole_col = argb8888(0xff, 240, 200, 90);
    if let (Some(pa), Some(pb)) = (
        project(camera, cart_com, width, height),
        project(camera, pole_tip, width, height),
    ) {
        draw_line(&mut fb, pa.0, pa.1, pb.0, pb.1, pole_col);
    }
    // Pivot marker (small cross at the cart's COM).
    if let Some((x, y)) = project(camera, cart_com, width, height) {
        let s = 4;
        draw_line(&mut fb, x - s, y, x + s, y, argb8888(0xff, 220, 220, 220));
        draw_line(&mut fb, x, y - s, x, y + s, argb8888(0xff, 220, 220, 220));
    }

    write_ppm(&out, &fb)?;
    let mode_str = match mode {
        Mode::Position => "position",
        Mode::Velocity => "velocity",
    };
    println!(
        "wrote {} ({}x{}, {} mode) — final cart x = {:.3} m, cart_v = {:.3} m/s, pole θ = {:.3} rad",
        out.display(),
        width,
        height,
        mode_str,
        tree.slide_position(1),
        tree.slide_rate(1),
        tree.hinge_angle(2)
    );
    Ok(())
}
