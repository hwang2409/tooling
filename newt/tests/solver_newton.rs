//! Newton solver anchors: load-time selection, cross-solver agreement, and
//! deterministic trajectories.

use newt::body::Body;
use newt::equality::Equality;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::mjcf::load_mjcf_str;
use newt::model::load_str;
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::world::World;

fn resting_box(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::new(0.01, 0.0, 0.6),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.5,
    ));
    world
}

fn stack(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0));
    let half = Vec3::splat(0.25);
    for (index, z) in [0.25, 0.75, 1.25].into_iter().enumerate() {
        let body = world.add_body(Body::solid_box(
            1.0,
            half,
            Vec3::new(0.01 * index as f32, 0.0, z),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::r#box(body, half, Vec3::ZERO, Quat::IDENTITY, 1.0));
    }
    world
}

fn incline(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let tilt = 22.0f32 * std::f32::consts::PI / 180.0;
    let orientation = Quat::from_axis_angle(Vec3::Y, tilt);
    world.add_geom(Geom::static_plane(
        Vec3::ZERO,
        orientation.rotate(Vec3::Z),
        1.0,
    ));
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.2),
        Vec3::new(0.02, 0.0, 0.3),
        orientation,
    ));
    let mut geom = Geom::r#box(body, Vec3::splat(0.2), Vec3::ZERO, Quat::IDENTITY, 1.0);
    geom.friction = 0.8;
    world.add_geom(geom);
    world
}

fn equality_linkage(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.02, 0.0, 1.0),
        Quat::IDENTITY,
    ));
    world.equalities.push(Equality::Connect {
        body_a: None,
        body_b: Some(body),
        anchor_a: Vec3::new(0.0, 0.0, 1.1),
        anchor_b: Vec3::new(0.0, 0.0, 0.1),
        solref: newt::geom::SolRef::new(0.03, 1.0),
        solimp: newt::solver::SolImp::DEFAULT,
    });
    world
}

#[test]
fn json_selects_newton_and_rejects_elliptic_newton() {
    let scene = load_str(
        r#"{
            "version":"1",
            "solver":{"mode":"newton","cone":"pyramidal","iterations":12}
        }"#,
    )
    .unwrap();
    assert_eq!(scene.world.solver.mode, SolverMode::Newton);
    assert_eq!(scene.world.solver.iterations, 12);

    let error = load_str(
        r#"{
            "version":"1",
            "solver":{"mode":"newton","cone":"elliptic"}
        }"#,
    )
    .unwrap_err();
    assert!(error.message.contains("elliptic"));
    assert!(error.message.contains("pyramidal"));
}

#[test]
fn mjcf_selects_newton_and_rejects_elliptic_newton() {
    let scene = load_mjcf_str(
        r#"<mujoco>
            <option solver="Newton" cone="pyramidal" iterations="12"/>
            <worldbody/>
        </mujoco>"#,
    )
    .unwrap();
    assert_eq!(scene.world.solver.mode, SolverMode::Newton);
    assert_eq!(scene.world.solver.iterations, 12);

    let error = load_mjcf_str(
        r#"<mujoco>
            <option solver="Newton" cone="elliptic"/>
            <worldbody/>
        </mujoco>"#,
    )
    .unwrap_err();
    assert!(error.message.contains("elliptic"));
}

#[test]
fn newton_and_pgs_agree_on_a_resting_contact() {
    let mut pgs = resting_box(SolverMode::Pgs);
    let mut newton = resting_box(SolverMode::Newton);
    for _ in 0..120 {
        pgs.step();
        newton.step();
    }
    let p = pgs.bodies[0];
    let n = newton.bodies[0];
    assert!((p.position.z - n.position.z).abs() < 2e-3);
    assert!((p.linear_velocity.z - n.linear_velocity.z).abs() < 2e-2);
    assert!((p.position.x - n.position.x).abs() < 2e-3);
    assert!(n.position.z > 0.45 && n.position.z < 0.52);
}

#[test]
fn newton_trajectory_is_bit_deterministic() {
    let mut left = resting_box(SolverMode::Newton);
    let mut right = resting_box(SolverMode::Newton);
    for _ in 0..160 {
        left.step();
        right.step();
    }
    assert_eq!(left.bodies, right.bodies);
}

#[test]
fn newton_agrees_with_pgs_across_contact_limit_and_equality_scenes() {
    for (name, mut pgs, mut newton, steps) in [
        (
            "stack",
            stack(SolverMode::Pgs),
            stack(SolverMode::Newton),
            120,
        ),
        (
            "incline",
            incline(SolverMode::Pgs),
            incline(SolverMode::Newton),
            120,
        ),
        (
            "equality_linkage",
            equality_linkage(SolverMode::Pgs),
            equality_linkage(SolverMode::Newton),
            120,
        ),
    ] {
        for _ in 0..steps {
            pgs.step();
            newton.step();
        }
        for (pgs_body, newton_body) in pgs.bodies.iter().zip(&newton.bodies) {
            assert!(
                (pgs_body.position - newton_body.position).length() < 3e-2,
                "{name}"
            );
            assert!(
                (pgs_body.linear_velocity - newton_body.linear_velocity).length() < 2e-1,
                "{name} velocity"
            );
        }
    }

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/references/joint_limit_swing.xml"
    );
    let source = std::fs::read_to_string(path).unwrap();
    let mut pgs = newt::mjcf::load_mjcf_str(&source).unwrap().world;
    let mut newton = newt::mjcf::load_mjcf_str(&source).unwrap().world;
    pgs.solver.mode = SolverMode::Pgs;
    newton.solver.mode = SolverMode::Newton;
    for _ in 0..120 {
        pgs.step();
        newton.step();
    }
    assert!((pgs.trees[0].q[0] - newton.trees[0].q[0]).abs() < 2e-2);
    assert!((pgs.trees[0].qdot[0] - newton.trees[0].qdot[0]).abs() < 2e-1);
}
