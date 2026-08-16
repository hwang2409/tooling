//! Newton solver anchors: load-time selection, cross-solver agreement, and
//! deterministic trajectories.

use newt::body::Body;
use newt::contact::Contact;
use newt::equality::Equality;
use newt::geom::{Geom, SolRef};
use newt::math::{Quat, Vec3};
use newt::mjcf::load_mjcf_str;
use newt::model::load_str;
use newt::solver::{
    ConeKind, SolverConfig, SolverMode, solve_free_bodies_newton_diag,
    solve_free_bodies_newton_trace,
};
use newt::world::World;
use std::panic::{AssertUnwindSafe, catch_unwind};

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

fn condim_four_contact(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8);
    plane.condim = 4;
    plane.torsional_friction = 0.2;
    world.add_geom(plane);
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.3,
        Vec3::new(0.01, -0.008, 0.295),
        Quat::IDENTITY,
    ));
    // The geom offset creates a contact lever arm from the body COM. The
    // tilted spin axis exercises angular and linear coupling in condim 4.
    world.bodies[body].angular_velocity_body = Vec3::new(0.15, -0.1, 3.0);
    let mut sphere = Geom::sphere(body, 0.3, Vec3::new(0.03, -0.02, 0.0), 0.8);
    sphere.condim = 4;
    sphere.torsional_friction = 0.2;
    world.add_geom(sphere);
    world
}

fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "non-string panic".to_string()
    }
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
    assert_eq!(scene.world.solver.cone, ConeKind::Pyramidal);

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
    assert_eq!(scene.world.solver.cone, ConeKind::Pyramidal);

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
fn programmatic_newton_elliptic_is_rejected_at_step_time() {
    let mut world = resting_box(SolverMode::Newton);
    world.solver.cone = ConeKind::Elliptic;
    let panic = catch_unwind(AssertUnwindSafe(|| world.step())).unwrap_err();
    let message = panic_text(panic);
    assert_eq!(
        message,
        SolverConfig {
            mode: SolverMode::Newton,
            iterations: 30,
            cone: ConeKind::Elliptic,
        }
        .validate()
        .unwrap_err()
    );
}

#[test]
fn contact_force_writeback_keeps_input_indices_when_a_gap_skips_a_row() {
    let body = Body::solid_sphere(1.0, 0.5, Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY);
    let geoms = vec![
        Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8).with_solref(SolRef::new(0.0223, 1.0)),
        Geom::sphere(0, 0.5, Vec3::ZERO, 0.8).with_solref(SolRef::new(0.0223, 1.0)),
    ];
    let contacts = vec![
        Contact {
            geom_a: 1,
            geom_b: 0,
            position_world: Vec3::ZERO,
            normal_world: Vec3::Z,
            penetration: 0.0,
            friction: 0.8,
            gap: 0.05,
        },
        Contact {
            geom_a: 1,
            geom_b: 0,
            position_world: Vec3::ZERO,
            normal_world: Vec3::Z,
            penetration: 0.02,
            friction: 0.8,
            gap: 0.0,
        },
    ];
    let (_, forces) = solve_free_bodies_newton_diag(
        &[body],
        &geoms,
        &contacts,
        &[],
        Vec3::new(0.0, 0.0, -9.81),
        0.005,
        ConeKind::Pyramidal,
        30,
    );
    assert_eq!(forces.len(), 2);
    assert_eq!(forces[0], 0.0);
    assert!((forces[1] - 85.68).abs() < 0.2, "forces={forces:?}");
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
        let mut max_position_delta: f32 = 0.0;
        let mut max_velocity_delta: f32 = 0.0;
        for (pgs_body, newton_body) in pgs.bodies.iter().zip(&newton.bodies) {
            max_position_delta =
                max_position_delta.max((pgs_body.position - newton_body.position).length());
            max_velocity_delta = max_velocity_delta
                .max((pgs_body.linear_velocity - newton_body.linear_velocity).length());
        }
        println!(
            "{name} cross-solver maxima: position={max_position_delta:.9e} velocity={max_velocity_delta:.9e}"
        );
        let (position_bound, velocity_bound) = match name {
            "stack" => (1.5e-3, 4.0e-2),
            "incline" => (1.2e-2, 3.5e-2),
            "equality_linkage" => (1.0e-6, 1.0e-5),
            _ => unreachable!("unknown cross-solver scene {name}"),
        };
        assert!(max_position_delta < position_bound, "{name} position");
        assert!(max_velocity_delta < velocity_bound, "{name} velocity");
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
    let joint_position_delta = (pgs.trees[0].q[0] - newton.trees[0].q[0]).abs();
    let joint_velocity_delta = (pgs.trees[0].qdot[0] - newton.trees[0].qdot[0]).abs();
    println!(
        "joint_limit_swing cross-solver maxima: position={joint_position_delta:.9e} velocity={joint_velocity_delta:.9e}"
    );
    assert!(joint_position_delta < 1e-5);
    assert!(joint_velocity_delta < 1e-4);
}

#[test]
fn live_stack_newton_cost_trace_is_monotone() {
    let mut world = stack(SolverMode::Newton);
    for body in &mut world.bodies {
        body.position.z -= 0.01;
    }
    let contacts = world.detect_contacts();
    let costs = solve_free_bodies_newton_trace(
        &world.bodies,
        &world.geoms,
        &contacts,
        &world.equalities,
        world.gravity,
        world.dt,
        world.solver.cone,
        world.solver.iterations,
    );
    assert!(costs.len() > 1, "expected active stack rows: {contacts:?}");
    for pair in costs.windows(2) {
        assert!(pair[1] <= pair[0] + 1e-6, "cost increased: {pair:?}");
    }
    let newton_iterations = costs.len() - 1;
    println!(
        "live stack convergence: threshold=1e-7 scaled, PGS sweeps=30, Newton iterations={newton_iterations}"
    );
    assert!(newton_iterations <= world.solver.iterations as usize);
}

#[test]
fn newton_and_pgs_agree_on_condim_four_torsional_contact() {
    let mut pgs = condim_four_contact(SolverMode::Pgs);
    let mut newton = condim_four_contact(SolverMode::Newton);
    let mut max_position_delta: f32 = 0.0;
    let mut max_velocity_delta: f32 = 0.0;
    let mut max_spin_delta: f32 = 0.0;
    for _ in 0..120 {
        pgs.step();
        newton.step();
        let p = pgs.bodies[0];
        let n = newton.bodies[0];
        max_position_delta = max_position_delta.max((p.position - n.position).length());
        max_velocity_delta =
            max_velocity_delta.max((p.linear_velocity - n.linear_velocity).length());
        max_spin_delta =
            max_spin_delta.max((p.angular_velocity_body - n.angular_velocity_body).length());
    }
    let p = pgs.bodies[0];
    let n = newton.bodies[0];
    assert!(
        p.position.x.is_finite()
            && p.position.y.is_finite()
            && p.position.z.is_finite()
            && n.position.x.is_finite()
            && n.position.y.is_finite()
            && n.position.z.is_finite()
    );
    assert!(
        p.angular_velocity_body.x.is_finite()
            && p.angular_velocity_body.y.is_finite()
            && p.angular_velocity_body.z.is_finite()
            && n.angular_velocity_body.x.is_finite()
            && n.angular_velocity_body.y.is_finite()
            && n.angular_velocity_body.z.is_finite()
    );
    // Measured maxima are 5.744356895e-4 m, 1.912438497e-2 m/s, and
    // 1.578792334e-1 rad/s. These bounds add modest headroom.
    const MAX_POSITION_DELTA: f32 = 1.0e-3;
    const MAX_VELOCITY_DELTA: f32 = 3.0e-2;
    const MAX_SPIN_DELTA: f32 = 2.0e-1;
    println!(
        "condim4 cross-solver maxima: position={max_position_delta:.9e} velocity={max_velocity_delta:.9e} spin={max_spin_delta:.9e}"
    );
    assert!(max_position_delta < MAX_POSITION_DELTA);
    assert!(max_velocity_delta < MAX_VELOCITY_DELTA);
    assert!(max_spin_delta < MAX_SPIN_DELTA);
}
