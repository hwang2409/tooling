//! Focused videos for the newt feature surface.
//!
//! The scenes share the showcase renderer and keep the physics setup small.
//! Run with `--scene joints|equalities|geoms|sensors|solvers|integrators`.

mod showcase_support;

use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Vec3 as CVec3};
use newt::actuator::Actuator;
use newt::body::Body;
use newt::equality::Equality;
use newt::geom::{Geom, SolRef, solid_cylinder_inertia, solid_ellipsoid_inertia};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::model::load_from_path;
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::tree::{Link, Tree, forward_kinematics};
use newt::world::{Integrator, World};
use std::path::PathBuf;

struct Args {
    scene: String,
    frames: usize,
    out: PathBuf,
    size: (usize, usize),
}

fn args() -> Args {
    let mut scene = String::from("joints");
    let mut frames = 900;
    let mut out = PathBuf::from("newt-features.mp4");
    let mut size = (800, 480);
    let mut values = std::env::args().skip(1);
    while let Some(value) = values.next() {
        match value.as_str() {
            "--scene" => scene = values.next().unwrap(),
            "--frames" => frames = values.next().unwrap().parse().unwrap(),
            "--out" => out = values.next().unwrap().into(),
            "--size" => {
                let dimensions = values.next().unwrap();
                let (width, height) = dimensions.split_once('x').unwrap();
                size = (width.parse().unwrap(), height.parse().unwrap());
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    Args {
        scene,
        frames,
        out,
        size,
    }
}

fn solver() -> SolverConfig {
    SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 35,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    }
}

fn root(position: Vec3) -> Link {
    Link::new(
        None,
        JointKind::Fixed,
        (position, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    )
}

fn link(parent: usize, joint: JointKind, offset: Vec3) -> Link {
    Link::new(
        Some(parent),
        joint,
        (Vec3::ZERO, Quat::IDENTITY),
        (offset, Quat::IDENTITY),
        0.8,
        Mat3::diag(0.04, 0.04, 0.01),
    )
}

fn build_joints() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = solver();
    let plane = world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8));

    let mut hinge = Tree::new();
    hinge.push_link(root(Vec3::new(-1.5, 0.0, 0.0)));
    hinge.push_link(link(
        0,
        JointKind::hinge(Vec3::Y),
        Vec3::new(0.0, 0.0, 0.65),
    ));
    hinge.set_hinge_angle(1, 0.7);
    let hinge_id = world.add_tree(hinge);
    let hinge_geom = world.add_geom(Geom::capsule_on_link(
        hinge_id,
        1,
        0.12,
        0.52,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));

    let mut ball = Tree::new();
    ball.push_link(root(Vec3::new(0.0, 0.0, 0.0)));
    ball.push_link(link(0, JointKind::ball(), Vec3::new(0.0, 0.0, 0.65)));
    ball.set_ball_orientation(1, Quat::from_axis_angle(Vec3::Y, 0.55));
    ball.set_ball_omega(1, Vec3::new(0.0, 1.4, 0.6));
    let ball_id = world.add_tree(ball);
    let ball_geom = world.add_geom(Geom::sphere_on_link(ball_id, 1, 0.18, Vec3::ZERO, 0.8));

    let mut slide = Tree::new();
    slide.push_link(root(Vec3::new(1.5, 0.0, 0.0)));
    slide.push_link(link(
        0,
        JointKind::slide(Vec3::Z),
        Vec3::new(0.0, 0.0, 0.65),
    ));
    slide.set_slide_position(1, 0.25);
    slide.set_slide_rate(1, -0.2);
    let slide_id = world.add_tree(slide);
    let slide_geom = world.add_geom(Geom::box_on_link(
        slide_id,
        1,
        Vec3::new(0.2, 0.2, 0.28),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));

    world.pair_list = Some(vec![
        (plane, hinge_geom),
        (plane, ball_geom),
        (plane, slide_geom),
    ]);
    world
}

fn build_equalities() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::ZERO;
    world.solver = solver();
    let solref = SolRef::new(0.01, 1.0);
    let solimp = SolImp::new(0.99, 0.999, 0.001, 0.5, 2);
    let sphere = |position| Body::solid_sphere(1.0, 0.13, position, Quat::IDENTITY);

    let connect_a = world.add_body(sphere(Vec3::new(-1.7, 0.0, 1.35)));
    let connect_b = world.add_body(sphere(Vec3::new(-1.7, 0.0, 0.85)));
    world.add_geom(Geom::sphere(connect_a, 0.13, Vec3::ZERO, 0.8));
    world.add_geom(Geom::sphere(connect_b, 0.13, Vec3::ZERO, 0.8));
    world.equalities.push(Equality::Connect {
        body_a: None,
        body_b: Some(connect_a),
        anchor_a: Vec3::new(-1.7, 0.0, 1.48),
        anchor_b: Vec3::new(0.0, 0.0, 0.13),
        solref,
        solimp,
    });
    world.equalities.push(Equality::Connect {
        body_a: Some(connect_a),
        body_b: Some(connect_b),
        anchor_a: Vec3::new(0.0, 0.0, -0.13),
        anchor_b: Vec3::new(0.0, 0.0, 0.13),
        solref,
        solimp,
    });
    world.bodies[connect_b].linear_velocity = Vec3::new(0.0, 0.35, 0.0);

    let weld_a = world.add_body(sphere(Vec3::new(-0.25, 0.0, 1.2)));
    let weld_b = world.add_body(sphere(Vec3::new(0.25, 0.0, 1.2)));
    world.add_geom(Geom::sphere(weld_a, 0.13, Vec3::ZERO, 0.8));
    world.add_geom(Geom::sphere(weld_b, 0.13, Vec3::ZERO, 0.8));
    world.equalities.push(Equality::Weld {
        body_a: Some(weld_a),
        body_b: Some(weld_b),
        anchor_a: Vec3::new(0.25, 0.0, 0.0),
        anchor_b: Vec3::new(-0.25, 0.0, 0.0),
        relative_orientation: Quat::IDENTITY,
        solref,
        solimp,
    });
    world.bodies[weld_b].angular_velocity_body = Vec3::new(0.0, 1.2, 0.0);

    let distance_a = world.add_body(sphere(Vec3::new(1.0, 0.0, 1.2)));
    let distance_b = world.add_body(sphere(Vec3::new(2.0, 0.0, 1.2)));
    world.add_geom(Geom::sphere(distance_a, 0.13, Vec3::ZERO, 0.8));
    world.add_geom(Geom::sphere(distance_b, 0.13, Vec3::ZERO, 0.8));
    world.bodies[distance_a].linear_velocity = Vec3::new(0.0, 0.4, 0.0);
    world.bodies[distance_b].linear_velocity = Vec3::new(0.0, -0.4, 0.0);
    world.equalities.push(Equality::Distance {
        body_a: Some(distance_a),
        body_b: Some(distance_b),
        anchor_a: Vec3::ZERO,
        anchor_b: Vec3::ZERO,
        distance: 1.0,
        solref,
        solimp,
    });
    world.pair_list = Some(Vec::new());
    world
}

fn build_geoms() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = solver();
    let plane = world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8));
    let capsule = world.add_body(Body::new(
        1.0,
        newt::geom::solid_capsule_inertia(1.0, 0.16, 0.34),
        Vec3::new(-1.2, 0.0, 1.6),
        Quat::from_axis_angle(Vec3::Y, 0.3),
    ));
    let capsule_geom = world.add_geom(Geom::capsule(
        capsule,
        0.16,
        0.34,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));
    let cylinder = world.add_body(Body::new(
        1.0,
        solid_cylinder_inertia(1.0, 0.28, 0.38),
        Vec3::new(0.0, 0.0, 1.5),
        Quat::from_axis_angle(Vec3::X, 0.25),
    ));
    let cylinder_geom = world.add_geom(Geom::cylinder(
        cylinder,
        0.28,
        0.38,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));
    let ellipsoid = world.add_body(Body::new(
        1.0,
        solid_ellipsoid_inertia(1.0, Vec3::new(0.38, 0.25, 0.5)),
        Vec3::new(1.2, 0.0, 1.7),
        Quat::from_axis_angle(Vec3::Y, -0.3),
    ));
    let ellipsoid_geom = world.add_geom(Geom::ellipsoid(
        ellipsoid,
        Vec3::new(0.38, 0.25, 0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));
    world.pair_list = Some(vec![
        (plane, capsule_geom),
        (plane, cylinder_geom),
        (plane, ellipsoid_geom),
    ]);
    world
}

fn shifted(mut items: Vec<showcase_support::Item>, x: f32) -> Vec<showcase_support::Item> {
    for value in &mut items {
        value.model = Mat4::translate(CVec3::new(x, 0.0, 0.0)) * value.model;
    }
    items
}

fn stack_world(mode: SolverMode, integrator: Integrator) -> World {
    let mut world = load_from_path("models/stack.json")
        .expect("stack model")
        .world;
    world.solver.mode = mode;
    world.integrator = integrator;
    world
}

fn integrator_world(integrator: Integrator) -> World {
    let mut world = World::new();
    world.dt = 0.01;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = solver();
    world.integrator = integrator;
    let plane = world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8));
    let mut tree = Tree::new();
    tree.push_link(root(Vec3::new(0.0, 0.0, 1.5)));
    let hinge = tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 5.0,
            armature: 0.02,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY),
        0.8,
        Mat3::diag(0.04, 0.04, 0.01),
    ));
    tree.add_actuator(Actuator::velocity(hinge, 160.0, 100.0));
    let tree_id = world.add_tree(tree);
    let capsule = world.add_geom(Geom::capsule_on_link(
        tree_id,
        hinge,
        0.12,
        0.38,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.8,
    ));
    world.pair_list = Some(vec![(plane, capsule)]);
    world.trees[tree_id].set_hinge_angle(hinge, 0.8);
    world.trees[tree_id].set_actuator_target(0, 1.4);
    world
}

fn integrator_target(step: usize) -> f32 {
    1.4 * newt::math::sin(step as f32 * 0.15)
}

fn drive_integrators(left: &mut World, right: &mut World, step: usize) {
    let target = integrator_target(step);
    left.trees[0].set_actuator_target(0, target);
    right.trees[0].set_actuator_target(0, target);
    left.step();
    right.step();
}

fn assert_integrator_trajectories_differ() {
    let mut euler = integrator_world(Integrator::Euler);
    let mut implicit = integrator_world(Integrator::ImplicitFast);
    let mut max_position_delta: f32 = 0.0;
    for step in 0..600 {
        drive_integrators(&mut euler, &mut implicit, step);
        max_position_delta =
            max_position_delta.max((euler.trees[0].q[0] - implicit.trees[0].q[0]).abs());
    }
    let position_delta = (euler.trees[0].q[0] - implicit.trees[0].q[0]).abs();
    let velocity_delta = (euler.trees[0].qdot[0] - implicit.trees[0].qdot[0]).abs();
    assert!(
        position_delta + velocity_delta > 1.0e-6,
        "Euler and ImplicitFast trajectories must differ"
    );
    assert!(
        max_position_delta > 0.4,
        "integrator trajectories must separate visibly: {max_position_delta} rad"
    );
}

fn render_compare(
    left: &World,
    right: &World,
    width: usize,
    height: usize,
    hud: &str,
) -> Framebuffer {
    let mut items = shifted(showcase_support::world_items(left), -2.2);
    items.extend(shifted(showcase_support::world_items(right), 2.2));
    showcase_support::render_items(
        &items,
        showcase_support::composition("compare"),
        width,
        height,
        hud,
    )
}

fn run_joints(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut world = build_joints();
    let (width, height) = args.size;
    let mut simulated = 0;
    showcase_support::write_video(&args.out, args.frames, |step| {
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
            &format!("joints  hinge | ball | slide  |  step {step}"),
        )
    })
}

fn run_equalities(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut world = build_equalities();
    let (width, height) = args.size;
    let mut simulated = 0;
    showcase_support::write_video(&args.out, args.frames, |step| {
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
            &format!("equalities  connect | weld | distance  |  step {step}"),
        )
    })
}

fn run_geoms(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut world = build_geoms();
    let (width, height) = args.size;
    let mut simulated = 0;
    showcase_support::write_video(&args.out, args.frames, |step| {
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
            &format!("contacts  capsule | cylinder | ellipsoid  |  step {step}"),
        )
    })
}

fn run_sensors(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut scene = load_from_path("models/arm.json").expect("arm model");
    let arm = *scene.trees_by_name.get("arm").expect("arm tree");
    let servos = ["shoulder_servo", "elbow_servo", "wrist_servo"]
        .map(|name| *scene.actuators_by_name.get(name).expect("arm actuator"));
    let (width, height) = args.size;
    let mut simulated = 0;
    showcase_support::write_video(&args.out, args.frames, |step| {
        for current in simulated..step {
            let phase = current as f32 * scene.world.dt;
            for (i, &(_, actuator)) in servos.iter().enumerate() {
                scene.world.trees[arm].set_actuator_target(
                    actuator,
                    [0.45, -0.65, 0.35][i] * newt::math::sin(phase + i as f32),
                );
            }
            scene.world.step();
        }
        simulated = step;
        let q = scene
            .world
            .sensor(*scene.sensors_by_name.get("shoulder_q").unwrap())
            .unwrap()[0];
        let speed = scene
            .world
            .sensor(*scene.sensors_by_name.get("tip_velocimeter").unwrap())
            .unwrap();
        let accel = scene
            .world
            .sensor(*scene.sensors_by_name.get("tip_imu_accel").unwrap())
            .unwrap();
        let poses = forward_kinematics(&scene.world.trees[arm]);
        let mut items = vec![showcase_support::item(
            showcase_support::cuboid_mesh(CVec3::new(3.0, 3.0, 0.04)),
            showcase_support::transform(
                Vec3::new(0.0, 0.0, -0.04),
                Quat::IDENTITY,
                CVec3::new(1.0, 1.0, 1.0),
            ),
            showcase_support::Material::new(CVec3::new(0.04, 0.05, 0.07), 0.0, 0.9),
        )];
        for link in 1..poses.len() {
            if let Some(parent) = scene.world.trees[arm].links[link].parent {
                showcase_support::add_capsule(
                    &mut items,
                    poses[parent].0,
                    poses[link].0,
                    0.06,
                    showcase_support::Material::new(
                        CVec3::new(0.12 + link as f32 * 0.1, 0.4, 0.75),
                        0.35,
                        0.3,
                    ),
                );
            }
        }
        showcase_support::render_items(
            &items,
            showcase_support::composition("arm"),
            width,
            height,
            &format!(
                "sensors  q {:.2}  vel ({:.2},{:.2},{:.2})  accel ({:.1},{:.1},{:.1})",
                q, speed[0], speed[1], speed[2], accel[0], accel[1], accel[2]
            ),
        )
    })
}

fn run_compare(args: &Args, integrators: bool) -> Result<(), Box<dyn std::error::Error>> {
    let (mut left, mut right) = if integrators {
        assert_integrator_trajectories_differ();
        (
            integrator_world(Integrator::Euler),
            integrator_world(Integrator::ImplicitFast),
        )
    } else {
        (
            stack_world(SolverMode::Pgs, Integrator::Rk4),
            stack_world(SolverMode::Newton, Integrator::Rk4),
        )
    };
    let (width, height) = args.size;
    let mut simulated = 0;
    showcase_support::write_video(&args.out, args.frames, |step| {
        for current in simulated..step {
            if integrators {
                drive_integrators(&mut left, &mut right, current);
            } else {
                left.step();
                right.step();
            }
        }
        simulated = step;
        let hud = if integrators {
            format!(
                "integrators  Euler damped servo       ImplicitFast damped servo  |  step {step}"
            )
        } else {
            format!("solvers  PGS                              Newton  |  step {step}")
        };
        render_compare(&left, &right, width, height, &hud)
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = args();
    match args.scene.as_str() {
        "joints" => run_joints(&args),
        "equalities" => run_equalities(&args),
        "geoms" => run_geoms(&args),
        "sensors" => run_sensors(&args),
        "solvers" => run_compare(&args, false),
        "integrators" => run_compare(&args, true),
        other => Err(format!("unknown scene: {other}").into()),
    }
}
