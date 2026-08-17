//! Muscle-driven pendulum showcase.
//!
//! Run with `cargo run --release --example muscle_pendulum -- --frames 1800
//! --out muscle-pendulum.mp4`. The video uses the standard chimy2 pipeline.

mod showcase_support;

use chimy2::fb::Framebuffer;
use chimy2::math::Vec3 as CVec3;
use newt::actuator::Actuator;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, forward_kinematics, rk4_step};
use std::path::PathBuf;

fn args() -> (usize, PathBuf, (usize, usize)) {
    let mut frames = 1800;
    let mut output = PathBuf::from("newt-muscle-pendulum.mp4");
    let mut size = (640, 360);
    let mut values = std::env::args().skip(1);
    while let Some(value) = values.next() {
        match value.as_str() {
            "--frames" => frames = values.next().unwrap().parse().unwrap(),
            "--out" => output = PathBuf::from(values.next().unwrap()),
            "--size" => {
                let dimensions = values.next().unwrap();
                let (width, height) = dimensions.split_once('x').unwrap();
                size = (width.parse().unwrap(), height.parse().unwrap());
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    (frames, output, size)
}

fn build_tree() -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, 1.5), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.45), Quat::IDENTITY),
        1.0,
        Mat3::diag(0.07, 0.07, 0.001),
    ));
    let params = [0.75, 1.05, 1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2];
    let actuator = Actuator::muscle(
        1,
        params,
        params,
        [-1.0, 1.0],
        1.0,
        -1.0,
        [0.02, 0.06, 0.1],
        Some((0.0, 1.0)),
        None,
    );
    tree.add_actuator(actuator);
    tree.set_hinge_angle(1, 0.65);
    tree
}

fn frame(tree: &Tree, width: usize, height: usize, step: usize) -> Framebuffer {
    let poses = forward_kinematics(tree);
    let (pivot, _) = poses[0];
    let (center, orientation) = poses[1];
    let top = center + orientation.rotate(Vec3::new(0.0, 0.0, 0.45));
    let bottom = center + orientation.rotate(Vec3::new(0.0, 0.0, -0.45));
    let mut items = vec![showcase_support::item(
        showcase_support::cuboid_mesh(CVec3::new(5.0, 5.0, 0.04)),
        showcase_support::transform(
            Vec3::new(0.0, 0.0, -0.04),
            Quat::IDENTITY,
            CVec3::new(1.0, 1.0, 1.0),
        ),
        showcase_support::Material::new(CVec3::new(0.04, 0.05, 0.07), 0.0, 0.9),
    )];
    showcase_support::add_capsule(
        &mut items,
        top,
        bottom,
        0.07,
        showcase_support::Material::new(CVec3::new(0.12, 0.45, 0.85), 0.2, 0.25),
    );
    showcase_support::add_marker(
        &mut items,
        pivot,
        0.12,
        showcase_support::Material::new(CVec3::new(0.95, 0.65, 0.12), 0.4, 0.2),
    );
    showcase_support::render_items(
        &items,
        showcase_support::composition("muscle"),
        width,
        height,
        &format!(
            "muscle pendulum  step {step:04}  activation {:.3}",
            tree.actuators[0].act
        ),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, output, (width, height)) = args();
    let mut tree = build_tree();
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut writer = showcase_support::VideoWriter::new(&output)?;
    writer.push(&frame(&tree, width, height, 0))?;
    let mut simulated = 0;
    let mut min_activation = tree.actuators[0].act;
    let mut max_activation = min_activation;
    while simulated < frames {
        let next = simulated + showcase_support::SIM_STEPS_PER_VIDEO_FRAME.min(frames - simulated);
        for step in simulated..next {
            tree.actuators[0].ctrl = if step < frames / 3 {
                0.0
            } else if step < 2 * frames / 3 {
                1.0
            } else {
                0.0
            };
            rk4_step(&mut tree, gravity, 0.005, |_| {
                vec![(Vec3::ZERO, Vec3::ZERO); 2]
            });
            min_activation = min_activation.min(tree.actuators[0].act);
            max_activation = max_activation.max(tree.actuators[0].act);
        }
        writer.push(&frame(&tree, width, height, next))?;
        simulated = next;
    }
    writer.finish()?;
    println!(
        "wrote {} ({frames} simulation steps, activation {min_activation:.6}..{max_activation:.6})",
        output.display()
    );
    Ok(())
}
