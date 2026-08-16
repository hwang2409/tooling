use newt::actuator::Actuator;
use newt::body::Body;
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::model::load_str;
use newt::tree::{Link, Tree, aba, forward_kinematics};
use newt::world::{Integrator, World};

fn damped_hinge(damping: f32) -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::Z,
            range: None,
            damping,
            armature: 0.0,
            limit: JointLimit::new(0.0, 0.0),
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    tree.set_hinge_rate(1, 1.0);
    tree
}

#[test]
fn euler_uses_new_velocity_for_position() {
    let mut world = World::new();
    world.integrator = Integrator::Euler;
    world.dt = 0.1;
    world.gravity = Vec3::new(0.0, 0.0, -10.0);
    world.add_body(Body::principal_axis(
        1.0,
        1.0,
        1.0,
        1.0,
        Vec3::ZERO,
        Quat::IDENTITY,
    ));

    world.step();

    assert_eq!(world.bodies[0].linear_velocity.z, -1.0);
    assert_eq!(world.bodies[0].position.z, -0.1);
}

#[test]
fn implicit_joint_damping_stays_stable_when_explicit_probe_diverges() {
    let dt = 0.005;
    let mut explicit = damped_hinge(1_000.0);
    for _ in 0..5 {
        let poses = forward_kinematics(&explicit);
        let qddot = aba(
            &explicit,
            &poses,
            Vec3::ZERO,
            &vec![(Vec3::ZERO, Vec3::ZERO); 2],
        );
        let slot = explicit.v_offset[1];
        explicit.qdot[slot] += qddot[slot] * dt;
    }
    assert!(explicit.qdot[explicit.v_offset[1]].abs() > 100.0);

    let mut world = World::new();
    world.integrator = Integrator::Euler;
    world.dt = dt;
    world.gravity = Vec3::ZERO;
    world.add_tree(damped_hinge(1_000.0));
    for _ in 0..5 {
        world.step();
    }
    let rate = world.trees[0].qdot[world.trees[0].v_offset[1]];
    assert!(rate.is_finite());
    assert!(rate.abs() < 1.0);
}

#[test]
fn implicitfast_folds_velocity_actuator_derivative() {
    let dt = 0.005;
    let mut euler_tree = damped_hinge(0.0);
    euler_tree.add_actuator(Actuator::velocity(1, 1_000.0, 0.0));

    let mut euler = World::new();
    euler.integrator = Integrator::Euler;
    euler.dt = dt;
    euler.gravity = Vec3::ZERO;
    euler.add_tree(euler_tree);
    for _ in 0..5 {
        euler.step();
    }
    let explicit_rate = euler.trees[0].qdot[euler.trees[0].v_offset[1]];
    assert!(explicit_rate.abs() > 100.0);

    let mut implicit_tree = damped_hinge(0.0);
    implicit_tree.add_actuator(Actuator::velocity(1, 1_000.0, 0.0));
    let mut implicit = World::new();
    implicit.integrator = Integrator::ImplicitFast;
    implicit.dt = dt;
    implicit.gravity = Vec3::ZERO;
    implicit.add_tree(implicit_tree);
    for _ in 0..5 {
        implicit.step();
    }
    let implicit_rate = implicit.trees[0].qdot[implicit.trees[0].v_offset[1]];
    assert!(implicit_rate.is_finite());
    assert!(implicit_rate.abs() < 1.0);
}

#[test]
fn rk4_is_the_default_and_explicit_selection_matches_it() {
    let body = Body::principal_axis(
        1.0,
        1.0,
        2.0,
        3.0,
        Vec3::new(0.0, 0.0, 1.0),
        Quat::from_axis_angle(Vec3::new(1.0, 0.3, -0.2), 0.4),
    );
    let mut default_world = World::new();
    default_world.add_body(body);
    let mut explicit_world = World::new();
    explicit_world.integrator = Integrator::Rk4;
    explicit_world.add_body(body);
    for _ in 0..20 {
        default_world.step();
        explicit_world.step();
    }
    assert_eq!(default_world.bodies, explicit_world.bodies);
}

#[test]
fn json_integrator_selection_and_validation() {
    assert_eq!(
        load_str(r#"{"integrator":"Euler"}"#)
            .unwrap()
            .world
            .integrator,
        Integrator::Euler
    );
    assert_eq!(
        load_str(r#"{"integrator":"implicitfast"}"#)
            .unwrap()
            .world
            .integrator,
        Integrator::ImplicitFast
    );
    let error = load_str(r#"{"integrator":"newton"}"#).unwrap_err();
    assert!(
        error
            .message
            .contains("expected RK4 | Euler | implicitfast")
    );
}

#[test]
fn euler_tumbling_energy_is_not_conservative() {
    let mut euler = World::new();
    euler.integrator = Integrator::Euler;
    euler.gravity = Vec3::ZERO;
    let mut body = Body::principal_axis(1.0, 1.0, 2.0, 3.0, Vec3::ZERO, Quat::IDENTITY);
    body.angular_velocity_body = Vec3::new(0.1, 8.0, 0.1);
    let initial = body.kinetic_energy();
    euler.add_body(body);
    for _ in 0..2_000 {
        euler.step();
    }

    let mut rk4 = World::new();
    rk4.gravity = Vec3::ZERO;
    rk4.add_body(body);
    for _ in 0..2_000 {
        rk4.step();
    }
    let euler_energy = euler.bodies[0].kinetic_energy();
    let rk4_energy = rk4.bodies[0].kinetic_energy();
    println!("tumbling energy: initial={initial:.9} euler={euler_energy:.9} rk4={rk4_energy:.9}");
    assert!(euler_energy > initial);
    assert!(euler_energy - initial > 0.1);
    assert!((rk4_energy - initial).abs() < 0.01);
}
