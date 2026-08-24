//! MuJoCo-style heightfield demo: a sphere and a box tumble over terrain.
//!
//! The default output is a full-length MP4 from the shared system-ffmpeg
//! pipeline. The printed speed is simulation speed, not a playback claim.

mod showcase_support;

use newt::body::Body;
use newt::geom::{Geom, HeightField};
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::world::World;
use std::path::PathBuf;

fn build_world() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };
    let hfield = HeightField {
        nrow: 7,
        ncol: 7,
        size: [2.0, 2.0, 0.8, 0.2],
        data: vec![
            0.05, 0.10, 0.18, 0.12, 0.25, 0.20, 0.30, 0.12, 0.18, 0.28, 0.22, 0.35, 0.30, 0.42,
            0.20, 0.30, 0.40, 0.34, 0.48, 0.42, 0.52, 0.15, 0.24, 0.32, 0.45, 0.38, 0.55, 0.46,
            0.28, 0.36, 0.48, 0.42, 0.58, 0.50, 0.62, 0.22, 0.34, 0.43, 0.52, 0.47, 0.64, 0.56,
            0.30, 0.40, 0.50, 0.58, 0.55, 0.70, 0.62,
        ],
    };
    let hfield_id = world.add_hfield(hfield);
    world.add_geom(Geom::static_hfield(
        hfield_id,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));

    let sphere_id = world.add_body(Body::solid_sphere(
        0.8,
        0.18,
        Vec3::new(-1.3, -1.1, 1.5),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(sphere_id, 0.18, Vec3::ZERO, 0.8));

    let box_half = Vec3::new(0.28, 0.22, 0.20);
    let box_id = world.add_body(Body::solid_box(
        1.2,
        box_half,
        Vec3::new(1.0, -0.5, 1.8),
        Quat::from_axis_angle(Vec3::Y, 0.35),
    ));
    world.add_geom(Geom::r#box(
        box_id,
        box_half,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));
    // Sphere-box is outside this ticket's collision matrix. Keep both
    // bodies paired with the terrain while excluding that unrelated pair.
    world.pair_list = Some(vec![(0, 1), (0, 2)]);
    world
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut frames = 1800usize;
    let mut out = PathBuf::from("newt-hfield-demo.mp4");
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
        let items = showcase_support::world_items(&world);
        showcase_support::render_items(
            &items,
            showcase_support::composition("hfield"),
            800,
            480,
            &format!("heightfield terrain  |  step {step}"),
        )
    })?;
    println!(
        "wrote {} ({} video frames, {:.2}x simulation speed)",
        out.display(),
        frames.div_ceil(showcase_support::SIM_STEPS_PER_VIDEO_FRAME),
        showcase_support::video_speed_factor(world.dt),
    );
    Ok(())
}
