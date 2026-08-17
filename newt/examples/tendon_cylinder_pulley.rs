//! Cylinder-wrap and pulley-branch demo.
//!
//! Run with `cargo run --release --example tendon_cylinder_pulley`.
//! The default output is an mp4 file.

mod showcase_support;

use chimy2::fb::{Framebuffer, argb8888};
use newt::actuator::Actuator;
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::tendon::{
    SpatialSegment, SpatialTendonBranch, SpatialTendonSite, SpatialWrap, Tendon, WrapCylinder,
};
use newt::tree::{Link, Tree, forward_kinematics};
use newt::world::World;
use std::path::PathBuf;

fn site(link: Option<usize>, position_local: Vec3) -> SpatialTendonSite {
    SpatialTendonSite {
        link,
        position_local,
    }
}

fn build_world() -> World {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::Slide {
            axis: Vec3::Z,
            range: None,
            damping: 2.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(1.4, 0.2, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let wrap = SpatialWrap::Cylinder(WrapCylinder {
        link: None,
        center_local: Vec3::ZERO,
        axis_local: Vec3::Z,
        radius: 0.4,
        sidesite: Some(site(None, Vec3::new(0.0, 1.0, 0.0))),
    });
    let tendon = Tendon::spatial_branches(vec![
        SpatialTendonBranch {
            sites: vec![
                site(None, Vec3::new(-1.8, 0.2, 0.0)),
                site(Some(1), Vec3::ZERO),
            ],
            segments: vec![SpatialSegment { wrap: Some(wrap) }],
            divisor: 1.0,
        },
        SpatialTendonBranch {
            sites: vec![
                site(None, Vec3::new(-1.8, -0.8, 0.0)),
                site(Some(1), Vec3::ZERO),
            ],
            segments: vec![SpatialSegment { wrap: None }],
            divisor: 2.0,
        },
    ]);
    tree.add_tendon(tendon);
    let actuator = tree.add_actuator(Actuator::motor(0, 1.0, 0.0).on_tendon(0));
    tree.set_actuator_target(actuator, -12.0);
    tree.set_slide_position(1, 0.3);
    world.add_tree(tree);
    world
}

fn project(p: Vec3, width: usize, height: usize) -> (i32, i32) {
    let scale = (width.min(height) as f32) * 0.18;
    (
        (width as f32 * 0.5 + p.x * scale) as i32,
        (height as f32 * 0.62 - p.y * scale) as i32,
    )
}

fn line(fb: &mut Framebuffer, a: (i32, i32), b: (i32, i32), color: u32) {
    let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = a.0 + ((b.0 - a.0) as f32 * t) as i32;
        let y = a.1 + ((b.1 - a.1) as f32 * t) as i32;
        if x >= 0 && y >= 0 && x < fb.width as i32 && y < fb.height as i32 {
            fb.put_pixel(x as usize, y as usize, color);
        }
    }
}

fn frame(world: &World, width: usize, height: usize, step: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let center = project(Vec3::ZERO, width, height);
    let radius = (width.min(height) as f32 * 0.18 * 0.4) as i32;
    let mut previous = center;
    for i in 0..=96 {
        let a = newt::math::TAU * i as f32 / 96.0;
        let p = (
            center.0 + (radius as f32 * newt::math::cos(a)) as i32,
            center.1 - (radius as f32 * newt::math::sin(a)) as i32,
        );
        if i > 0 {
            line(&mut fb, previous, p, argb8888(0xff, 80, 100, 180));
        }
        previous = p;
    }
    let poses = forward_kinematics(&world.trees[0]);
    let mass = poses[1].0;
    line(
        &mut fb,
        project(Vec3::new(-1.8, 0.2, 0.0), width, height),
        project(mass, width, height),
        argb8888(0xff, 70, 220, 240),
    );
    line(
        &mut fb,
        project(Vec3::new(-1.8, -0.8, 0.0), width, height),
        project(mass, width, height),
        argb8888(0xff, 240, 180, 80),
    );
    let m = project(mass, width, height);
    for dx in -7..=7 {
        for dy in -7..=7 {
            if dx * dx + dy * dy < 40 {
                let x = m.0 + dx;
                let y = m.1 + dy;
                if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
                    fb.put_pixel(x as usize, y as usize, argb8888(0xff, 245, 150, 70));
                }
            }
        }
    }
    let _ = step;
    fb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut frames = 600usize;
    let mut out = PathBuf::from("newt-tendon-cylinder-pulley.mp4");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => frames = args.next().unwrap().parse()?,
            "--out" => out = PathBuf::from(args.next().unwrap()),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let mut world = build_world();
    let mut simulated = 0;
    showcase_support::write_video(&out, frames, |step| {
        for _ in simulated..step {
            world.step();
        }
        simulated = step;
        frame(&world, 640, 360, step)
    })?;
    println!("wrote {}", out.display());
    Ok(())
}
