//! Tier 2 demo: three boxes dropped and settling into a vertical stack on
//! a static plane. Exercises box-plane (bottom box, 4 corner contacts) and
//! box-box (upper boxes, vertex-vs-face) contact paths under gravity.
//! Wireframe PPM via chimy2, same rendering plumbing as `tumble.rs`.
//!
//! Run:
//! ```text
//! cargo run --release --example stack -- --frames 800 --out /tmp/stack.ppm
//! ```

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::world::World;

use std::path::PathBuf;

fn parse_args() -> (usize, PathBuf, (usize, usize)) {
    let mut frames = 800usize;
    let mut out = PathBuf::from("newt-stack.ppm");
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

const HALF: Vec3 = Vec3::new(0.35, 0.35, 0.35);

fn build_world() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.6));

    // Drop heights are staggered so the boxes arrive in sequence rather
    // than crashing together mid-air. Perfect vertical alignment on X/Y —
    // box_box's face-along-direction pick handles the corner-on-corner
    // degeneracy that a naive nearest-face rule would trip on.
    let drops = [
        (Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.7), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 2.9), Quat::IDENTITY),
    ];
    for &(pos, ori) in &drops {
        let idx = world.add_body(Body::solid_box(1.0, HALF, pos, ori));
        world.add_geom(Geom::r#box(idx, HALF, Vec3::ZERO, Quat::IDENTITY, 0.6));
    }
    world
}

// Box wireframe: 8 corners, 12 edges — same shape as tier-1 tumble.rs.
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
        CVec3::new(3.5, -4.5, 2.5),
        CVec3::new(0.0, 0.0, 1.2),
        CVec3::new(0.0, 0.0, 1.0),
    );
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));
    let colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 240, 100, 140),
    ];
    for (idx, body) in world.bodies.iter().enumerate() {
        let corners = box_world_corners(body, HALF);
        let projected: [Option<(i32, i32)>; 8] =
            std::array::from_fn(|i| project(camera, corners[i], width, height));
        for &(a, b) in BOX_EDGES.iter() {
            if let (Some((x0, y0)), Some((x1, y1))) = (projected[a], projected[b]) {
                draw_line(&mut fb, x0, y0, x1, y1, colors[idx]);
            }
        }
    }
    fb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height)) = parse_args();
    let mut world = build_world();
    for _ in 0..frames {
        world.step();
    }
    let fb = render(&world, width, height);
    write_ppm(&out, &fb)?;
    println!(
        "wrote {} ({}x{}) — final box positions: {:?}",
        out.display(),
        width,
        height,
        world.bodies.iter().map(|b| b.position).collect::<Vec<_>>()
    );
    Ok(())
}
