//! v1-tier-4 demo: 6-box tower plus a heavy box dropped on top, solved
//! with the MuJoCo soft-constraint model + PGS. Shows the classical
//! solver-vs-penalty win — a stack that would jitter apart under a
//! penalty spring survives the impact under PGS.
//!
//! Uses a heavy BOX (not sphere) for the dropped mass because box-sphere
//! narrow-phase is not yet implemented; the visual "heavy object impacts
//! stack" story is the same.
//!
//! Run:
//! ```text
//! cargo run --release --example solver_stack -- --frames 800 --out /tmp/solver_stack.mp4
//! cargo run --release --example solver_stack -- --frames 800 --still /tmp/solver_stack.ppm
//! ```
//!
//! Solid shaded meshes use the shared showcase adapter.

mod showcase_support;

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::body::Body;
use newt::geom::{Geom, SolRef};
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::world::World;

use std::path::PathBuf;

struct Args {
    frames: usize,
    out: PathBuf,
    size: (usize, usize),
    frames_dir: Option<PathBuf>,
    still: Option<PathBuf>,
    wireframe: bool,
}

fn parse_args() -> Args {
    let mut frames = 800usize;
    let mut out = PathBuf::from("newt-solver-stack.mp4");
    let mut size = (640usize, 360usize);
    let mut frames_dir = None;
    let mut still = None;
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
            "--frames-dir" => frames_dir = Some(PathBuf::from(args.next().unwrap())),
            "--still" => still = Some(PathBuf::from(args.next().unwrap())),
            "--wireframe" => wireframe = true,
            _ => panic!("unknown arg: {a}"),
        }
    }
    Args {
        frames,
        out,
        size,
        frames_dir,
        still,
        wireframe,
    }
}

const HALF: Vec3 = Vec3::new(0.3, 0.3, 0.3);
const HEAVY_HALF: Vec3 = Vec3::new(0.35, 0.35, 0.35);

fn build_world() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };
    // Softer solref for RK4-ZOH stability under the coupled multi-body
    // regime (see docs/solver.md).
    let solref = SolRef::new(0.05, 1.5);

    // Static ground.
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8);
    plane.solref = solref;
    world.add_geom(plane);

    // 6-box tower, symmetry-broken (0.01 m x-offsets so no accidental
    // symmetry masks bugs — see docs/solver.md).
    for k in 0..6 {
        let pos = Vec3::new(0.005 * k as f32, 0.0, 0.3 + 0.6 * k as f32);
        let idx = world.add_body(Body::solid_box(1.0, HALF, pos, Quat::IDENTITY));
        let mut g = Geom::r#box(idx, HALF, Vec3::ZERO, Quat::IDENTITY, 0.8);
        g.solref = solref;
        world.add_geom(g);
    }
    // Heavy box (5 kg) dropped from above the tower.
    let drop_pos = Vec3::new(0.0, 0.0, 5.0);
    let h_idx = world.add_body(Body::solid_box(5.0, HEAVY_HALF, drop_pos, Quat::IDENTITY));
    let mut heavy_geom = Geom::r#box(h_idx, HEAVY_HALF, Vec3::ZERO, Quat::IDENTITY, 0.8);
    heavy_geom.solref = solref;
    world.add_geom(heavy_geom);
    world
}

// Box wireframe: 8 corners, 12 edges.
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

fn box_world_corners(body: &Body, half: Vec3) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    for (i, &(sx, sy, sz)) in BOX_CORNERS_LOCAL.iter().enumerate() {
        let local = Vec3::new(sx * half.x, sy * half.y, sz * half.z);
        out[i] = body.position + body.orientation.rotate(local);
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
    let span = 3.0;
    let step = 0.5;
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

fn render(world: &World, width: usize, height: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.1,
        200.0,
    ) * Mat4::look_at(
        CVec3::new(4.5, -5.5, 3.5),
        CVec3::new(0.0, 0.0, 1.8),
        CVec3::new(0.0, 0.0, 1.0),
    );
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));
    let box_colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 240, 100, 140),
        argb8888(0xff, 120, 240, 130),
        argb8888(0xff, 240, 160, 60),
        argb8888(0xff, 180, 120, 240),
    ];
    // 6 tower boxes then 1 heavy dropper.
    for (idx, body) in world.bodies.iter().enumerate().take(6) {
        let corners = box_world_corners(body, HALF);
        let projected: [Option<(i32, i32)>; 8] =
            std::array::from_fn(|i| project(camera, corners[i], width, height));
        for &(a, b) in BOX_EDGES.iter() {
            if let (Some((x0, y0)), Some((x1, y1))) = (projected[a], projected[b]) {
                draw_line(&mut fb, x0, y0, x1, y1, box_colors[idx]);
            }
        }
    }
    if let Some(heavy) = world.bodies.get(6) {
        let corners = box_world_corners(heavy, HEAVY_HALF);
        let projected: [Option<(i32, i32)>; 8] =
            std::array::from_fn(|i| project(camera, corners[i], width, height));
        for &(a, b) in BOX_EDGES.iter() {
            if let (Some((x0, y0)), Some((x1, y1))) = (projected[a], projected[b]) {
                draw_line(&mut fb, x0, y0, x1, y1, argb8888(0xff, 255, 255, 255));
            }
        }
    }
    fb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let (frames, out, (width, height)) = (args.frames, args.out, args.size);
    let mut world = build_world();
    let still_path = args.still.or_else(|| {
        args.frames_dir
            .map(|directory| directory.join("frame-00.ppm"))
    });
    if args.wireframe || still_path.is_some() {
        for _ in 0..frames {
            world.step();
        }
        let path = still_path.unwrap_or(out.clone());
        if args.wireframe {
            let fb = render(&world, width, height);
            write_ppm(path, &fb)?;
        } else {
            let items = showcase_support::world_items(&world);
            showcase_support::write_frame(
                &items,
                showcase_support::composition("stack"),
                width,
                height,
                &format!("solver stack  |  step {frames}"),
                path,
            )?;
        }
    } else {
        let mut simulated = 0;
        showcase_support::write_video(&out, frames, |step| {
            for _ in simulated..step {
                world.step();
            }
            simulated = step;
            let items = showcase_support::world_items(&world);
            showcase_support::render_items(
                &items,
                showcase_support::composition("stack"),
                width,
                height,
                &format!("solver stack  |  step {step}"),
            )
        })?;
    }
    println!(
        "wrote {} ({}x{}, {:.2}x simulation speed) — final positions:",
        out.display(),
        width,
        height,
        showcase_support::video_speed_factor(world.dt)
    );
    for (i, b) in world.bodies.iter().enumerate() {
        if i < 6 {
            println!("  box{i}: z = {:.4}", b.position.z);
        } else {
            println!("  heavy_box: pos = {:?}", b.position);
        }
    }
    Ok(())
}
