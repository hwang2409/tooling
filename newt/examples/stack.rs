//! Tier 2 demo: three spheres dropped and settling into a vertical stack on
//! a static plane. Renders a wireframe PPM through chimy2 the same way
//! `tumble.rs` does.
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
use newt::math::{Quat, Vec3, cos, sin};
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

const RADIUS: f32 = 0.3;

fn build_world() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.6));
    for &h in &[0.5f32, 1.3, 2.1] {
        let idx = world.add_body(Body::solid_sphere(
            1.0,
            RADIUS,
            Vec3::new(0.0, 0.0, h),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::sphere(idx, RADIUS, Vec3::ZERO, 0.6));
    }
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

/// Draw three great circles (xy, xz, yz) through the sphere center to sketch
/// its silhouette without pulling in a real mesh.
fn draw_sphere_wireframe(
    fb: &mut Framebuffer,
    camera: Mat4,
    width: usize,
    height: usize,
    center: Vec3,
    radius: f32,
    color: u32,
) {
    const SEGMENTS: usize = 24;
    let planes: [(Vec3, Vec3); 3] = [(Vec3::X, Vec3::Y), (Vec3::X, Vec3::Z), (Vec3::Y, Vec3::Z)];
    for &(u, v) in &planes {
        let mut prev: Option<(i32, i32)> = None;
        for i in 0..=SEGMENTS {
            let theta = (i as f32) / (SEGMENTS as f32) * (2.0 * newt::math::PI);
            let p = center + u * (radius * cos(theta)) + v * (radius * sin(theta));
            let cur = project(camera, p, width, height);
            if let (Some(a), Some(b)) = (prev, cur) {
                draw_line(fb, a.0, a.1, b.0, b.1, color);
            }
            prev = cur;
        }
    }
}

fn draw_ground_grid(fb: &mut Framebuffer, camera: Mat4, width: usize, height: usize, color: u32) {
    let span = 3.0;
    let step = 0.5;
    let n = (2.0 * span / step) as i32;
    for i in 0..=n {
        let t = -span + (i as f32) * step;
        // Line parallel to X:
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
        CVec3::new(0.0, 0.0, 0.8),
        CVec3::new(0.0, 0.0, 1.0),
    );
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));
    let colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 90, 220, 240),
        argb8888(0xff, 240, 100, 140),
    ];
    for (i, body) in world.bodies.iter().enumerate() {
        draw_sphere_wireframe(
            &mut fb,
            camera,
            width,
            height,
            body.position,
            RADIUS,
            colors[i],
        );
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
        "wrote {} ({}x{}) — final body z positions: {:?}",
        out.display(),
        width,
        height,
        world
            .bodies
            .iter()
            .map(|b| b.position.z)
            .collect::<Vec<_>>()
    );
    Ok(())
}
