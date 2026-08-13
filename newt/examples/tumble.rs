//! Tier 1 demo: three tumbling boxes under gravity, rendered as a wireframe
//! PPM via chimy2's Framebuffer + Mat4 helpers.
//!
//! Run:
//! ```text
//! cargo run --release --example tumble -- --frames 200 --out /tmp/tumble.ppm
//! ```
//!
//! Deliberately minimal: no lighting, no rasterizer — just per-body world→view
//! →clip→screen projection and Bresenham line draws for the twelve box edges.
//! The point is to prove newt's state can drive chimy2, not to exercise the
//! full renderer (later tiers do that).

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::body::Body;
use newt::math::{Quat, Vec3};
use newt::world::World;

use std::path::PathBuf;

fn parse_args() -> (usize, PathBuf, (usize, usize)) {
    let mut frames = 200usize;
    let mut out = PathBuf::from("newt-tumble.ppm");
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

fn build_world() -> (World, [Vec3; 3]) {
    // Same three-body scene as the golden trajectory test, so a reader can
    // eyeball the demo against the golden file offsets.
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    let half_extents = [
        Vec3::new(0.4, 0.3, 0.2),
        Vec3::new(0.3, 0.3, 0.3),
        Vec3::new(0.5, 0.2, 0.1),
    ];

    let mut b0 = Body::solid_box(
        1.0,
        half_extents[0],
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
    );
    b0.linear_velocity = Vec3::new(1.0, 0.5, 3.0);
    b0.angular_velocity_body = Vec3::new(1.0, 2.0, 0.5);

    let mut b1 = Body::solid_box(
        2.0,
        half_extents[1],
        Vec3::new(1.5, -1.0, 4.0),
        Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3),
    );
    b1.linear_velocity = Vec3::new(-0.5, 1.5, 2.0);
    b1.angular_velocity_body = Vec3::new(0.2, 3.5, 0.1);

    let mut b2 = Body::solid_box(
        0.5,
        half_extents[2],
        Vec3::new(-1.0, 2.0, 6.0),
        Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0), 0.7),
    );
    b2.linear_velocity = Vec3::new(2.0, -1.0, 0.5);
    b2.angular_velocity_body = Vec3::new(0.05, 0.1, 4.0);

    world.add_body(b0);
    world.add_body(b1);
    world.add_body(b2);
    (world, half_extents)
}

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

fn world_corners(body: &Body, half_extents: Vec3) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    for (i, &(sx, sy, sz)) in BOX_CORNERS_LOCAL.iter().enumerate() {
        let local = Vec3::new(
            sx * half_extents.x,
            sy * half_extents.y,
            sz * half_extents.z,
        );
        out[i] = body.position + body.orientation.rotate(local);
    }
    out
}

fn project(camera_matrix: Mat4, world_pt: Vec3, width: usize, height: usize) -> Option<(i32, i32)> {
    let clip = camera_matrix * Vec4::new(world_pt.x, world_pt.y, world_pt.z, 1.0);
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
    loop {
        if x0 >= 0 && y0 >= 0 {
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

fn render(world: &World, half_extents: &[Vec3; 3], width: usize, height: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));

    let camera_matrix = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.1,
        200.0,
    ) * Mat4::look_at(
        CVec3::new(6.0, -8.0, 6.0),
        CVec3::new(0.0, 0.0, 3.0),
        CVec3::new(0.0, 0.0, 1.0),
    );

    let colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 240, 100, 140),
    ];

    for (idx, body) in world.bodies.iter().enumerate() {
        let corners = world_corners(body, half_extents[idx]);
        let projected: [Option<(i32, i32)>; 8] =
            std::array::from_fn(|i| project(camera_matrix, corners[i], width, height));
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
    let (mut world, half_extents) = build_world();

    for _ in 0..frames {
        world.step();
    }

    let fb = render(&world, &half_extents, width, height);
    write_ppm(&out, &fb)?;
    println!("wrote {} ({}x{})", out.display(), width, height);
    Ok(())
}
