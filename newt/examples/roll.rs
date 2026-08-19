//! Tier 2 demo: two spheres launched at each other across a plane. Rolling
//! friction slows them, then they collide and separate. Wireframe PPM via
//! chimy2.
//!
//! Run:
//! ```text
//! cargo run --release --example roll -- --frames 400 --out /tmp/roll.ppm
//! ```

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3, cos, sin};
use newt::world::World;

use std::path::PathBuf;

mod showcase_support;

fn parse_args() -> (usize, PathBuf, (usize, usize), bool) {
    let mut frames = 400usize;
    let mut out = PathBuf::from("newt-roll.mp4");
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

const RADIUS: f32 = 0.35;

fn build_world() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.9));

    let a = world.add_body(Body::solid_sphere(
        1.0,
        RADIUS,
        Vec3::new(-3.0, 0.0, RADIUS + 0.01),
        Quat::IDENTITY,
    ));
    world.bodies[a].linear_velocity = Vec3::new(4.5, 0.0, 0.0);
    world.add_geom(Geom::sphere(a, RADIUS, Vec3::ZERO, 0.9));

    let b = world.add_body(Body::solid_sphere(
        1.0,
        RADIUS,
        Vec3::new(3.0, 0.0, RADIUS + 0.01),
        Quat::IDENTITY,
    ));
    world.bodies[b].linear_velocity = Vec3::new(-3.5, 0.0, 0.0);
    world.add_geom(Geom::sphere(b, RADIUS, Vec3::ZERO, 0.9));

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

fn draw_sphere(
    fb: &mut Framebuffer,
    camera: Mat4,
    width: usize,
    height: usize,
    body: &Body,
    radius: f32,
    color: u32,
) {
    const SEGMENTS: usize = 24;
    // Draw three great circles rotated by the body orientation so the spin is
    // visible in the demo.
    let planes: [(Vec3, Vec3); 3] = [(Vec3::X, Vec3::Y), (Vec3::X, Vec3::Z), (Vec3::Y, Vec3::Z)];
    for &(u_local, v_local) in &planes {
        let u = body.orientation.rotate(u_local);
        let v = body.orientation.rotate(v_local);
        let mut prev: Option<(i32, i32)> = None;
        for i in 0..=SEGMENTS {
            let theta = (i as f32) / (SEGMENTS as f32) * (2.0 * newt::math::PI);
            let p = body.position + u * (radius * cos(theta)) + v * (radius * sin(theta));
            let cur = project(camera, p, width, height);
            if let (Some(a), Some(b)) = (prev, cur) {
                draw_line(fb, a.0, a.1, b.0, b.1, color);
            }
            prev = cur;
        }
    }
}

fn draw_ground_grid(fb: &mut Framebuffer, camera: Mat4, width: usize, height: usize, color: u32) {
    let span = 5.0;
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
        CVec3::new(0.0, -8.0, 3.0),
        CVec3::new(0.0, 0.0, 0.2),
        CVec3::new(0.0, 0.0, 1.0),
    );
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));
    let colors = [argb8888(0xff, 240, 200, 90), argb8888(0xff, 90, 220, 240)];
    for (i, body) in world.bodies.iter().enumerate() {
        draw_sphere(&mut fb, camera, width, height, body, RADIUS, colors[i]);
    }
    fb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height), wireframe) = parse_args();
    let mut world = build_world();
    if wireframe {
        for _ in 0..frames {
            world.step();
        }
        write_ppm(&out, &render(&world, width, height))?;
        return Ok(());
    }
    let mut simulated = 0;
    showcase_support::write_video(&out, frames, |step| {
        for _ in simulated..step {
            world.step();
        }
        simulated = step;
        let items = showcase_support::world_items(&world);
        showcase_support::render_items(
            &items,
            showcase_support::composition("features"),
            width,
            height,
            &format!("rolling friction  |  step {step}"),
        )
    })?;
    println!(
        "wrote {} ({}x{}) — final body x positions: {:?}",
        out.display(),
        width,
        height,
        world
            .bodies
            .iter()
            .map(|b| b.position.x)
            .collect::<Vec<_>>()
    );
    Ok(())
}
