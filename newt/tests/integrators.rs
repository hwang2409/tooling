use std::fs;

use newt::actuator::Actuator;
use newt::body::Body;
use newt::dynamics::{bias_forces, cholesky, cholesky_solve, mass_matrix};
use newt::joint::{JointKind, JointLimit};
use newt::json::{self, Value};
use newt::math::{Mat3, Quat, Vec3};
use newt::mjcf::{load_mjcf_path, load_mjcf_str};
use newt::model::load_str;
use newt::tendon::{FixedTendonJoint, Tendon, tendon_kinematics};
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

fn damped_free_root(damping: f32) -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        2.0,
        Mat3::IDENTITY,
    ));
    tree.links[0].free_damping = damping;
    tree.qdot[3] = 3.0;
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
fn euler_free_root_damping_uses_hand_derived_implicit_rate() {
    let dt = 0.1;
    let mut world = World::new();
    world.integrator = Integrator::Euler;
    world.dt = dt;
    world.gravity = Vec3::ZERO;
    world.add_tree(damped_free_root(4.0));

    world.step();

    // m vdot = -d v_new gives v_new = m/(m + dt*d) * v_old.
    let tree = &world.trees[0];
    assert!((tree.qdot[3] - 2.5).abs() < 1e-6, "qvel = {}", tree.qdot[3]);
    assert!((tree.q[0] - 0.25).abs() < 1e-6, "qpos = {}", tree.q[0]);
    assert!(((tree.qdot[3] - 3.0) / dt + 5.0).abs() < 1e-5);
}

#[test]
fn free_root_damping_loads_from_json_and_mjcf() {
    let json = r#"{
        "trees": [{
            "name": "t",
            "links": [{
                "name": "root",
                "joint": {"kind": "free", "damping": 2.5},
                "mass": 1,
                "inertia": {"kind": "diag", "values": [1, 1, 1]}
            }]
        }]
    }"#;
    let scene = load_str(json).expect("json free-root damping should load");
    assert_eq!(scene.world.trees[0].links[0].free_damping, 2.5);

    let mjcf = r#"<mujoco><worldbody>
        <body name="root">
            <inertial mass="1" diaginertia="1 1 1"/>
            <freejoint damping="3.5"/>
            <body name="child" pos="0 0 -1">
                <joint type="hinge" axis="1 0 0"/>
                <inertial mass="1" diaginertia="1 1 1"/>
            </body>
        </body>
    </worldbody></mujoco>"#;
    let scene = load_mjcf_str(mjcf).expect("mjcf free-root damping should load");
    assert_eq!(scene.world.trees[0].links[0].free_damping, 3.5);
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
fn implicitfast_folds_muscle_velocity_derivative() {
    let mut tree = damped_hinge(0.0);
    tree.q[0] = 0.2;
    let actuator = tree.add_actuator(Actuator::muscle(
        1,
        [0.75, 1.05, 100.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2],
        [0.75, 1.05, 100.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2],
        [0.0, 1.0],
        1.0,
        1.0,
        [0.01, 0.04, 0.0],
        None,
        None,
    ));
    tree.actuators[actuator].act = 1.0;
    tree.qdot[0] = 0.5;
    let dt = 0.005;
    let damping = tree.actuators[actuator].velocity_damping(tree.q[0], tree.qdot[0]);
    assert!(damping > 0.0, "muscle FV curve must add velocity damping");
    let explicit = tree.mass_matrix()[0];
    let implicit = tree.implicit_mass_matrix(dt, true)[0];
    assert!((implicit - explicit - dt * damping).abs() < 1e-5);
}

#[test]
fn implicitfast_tendon_velocity_derivative_stays_explicit() {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    for _ in 0..2 {
        let parent = tree.links.len() - 1;
        tree.push_link(Link::new(
            Some(parent),
            JointKind::hinge(Vec3::Z),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::IDENTITY,
        ));
    }
    let tendon = tree.add_tendon(Tendon::fixed(vec![
        FixedTendonJoint { link: 1, coef: 1.0 },
        FixedTendonJoint { link: 2, coef: 2.0 },
    ]));
    let actuator = tree.add_actuator(Actuator::velocity(0, 100.0, 0.0).on_tendon(tendon));
    tree.set_actuator_target(actuator, 0.0);
    tree.qdot[tree.v_offset[1]] = 1.0;
    tree.qdot[tree.v_offset[2]] = -0.25;

    let poses = forward_kinematics(&tree);
    let kin = tendon_kinematics(&tree.tendons[tendon], &tree, &poses);
    let force = tree.actuators[actuator].torque(kin.length, kin.velocity);
    let mut rhs = vec![0.0; tree.nv()];
    for (slot, &coef) in kin.jacobian.iter().enumerate() {
        rhs[slot] += coef * force;
    }
    let mass = mass_matrix(&tree);
    let bias = bias_forces(&tree, Vec3::ZERO);
    for (r, value) in rhs.iter_mut().enumerate() {
        *value -= bias[r];
    }
    let factor = cholesky(&mass, tree.nv()).expect("dense mass matrix should be positive definite");
    let dense_qacc = cholesky_solve(&factor, tree.nv(), &rhs);

    let mut world = World::new();
    world.integrator = Integrator::ImplicitFast;
    world.dt = 0.005;
    world.gravity = Vec3::ZERO;
    world.add_tree(tree);
    let before = world.trees[0].qdot.clone();
    world.step();
    let observed_qacc: Vec<f32> = world.trees[0]
        .qdot
        .iter()
        .zip(before.iter())
        .map(|(&after, &before)| (after - before) / world.dt)
        .collect();

    for slot in [world.trees[0].v_offset[1], world.trees[0].v_offset[2]] {
        assert!(
            (observed_qacc[slot] - dense_qacc[slot]).abs() < 1e-4,
            "tendon actuator qacc slot {slot}: observed {}, dense {}",
            observed_qacc[slot],
            dense_qacc[slot]
        );
    }
}

#[test]
fn implicitfast_muscle_tendon_velocity_term_stays_explicit() {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    for _ in 0..2 {
        let parent = tree.links.len() - 1;
        tree.push_link(Link::new(
            Some(parent),
            JointKind::hinge(Vec3::Z),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::IDENTITY,
        ));
    }
    let tendon = tree.add_tendon(Tendon::fixed(vec![
        FixedTendonJoint { link: 1, coef: 1.0 },
        FixedTendonJoint { link: 2, coef: 2.0 },
    ]));
    let actuator = tree.add_actuator(
        Actuator::muscle(
            0,
            [0.75, 1.05, 1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2],
            [0.75, 1.05, 1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2],
            [0.0, 2.0],
            1.0,
            1.0,
            [0.01, 0.04, 0.0],
            None,
            None,
        )
        .on_tendon(tendon),
    );
    tree.actuators[actuator].act = 1.0;
    tree.qdot[tree.v_offset[1]] = 0.5;
    tree.qdot[tree.v_offset[2]] = -0.25;

    let poses = forward_kinematics(&tree);
    let kin = tendon_kinematics(&tree.tendons[tendon], &tree, &poses);
    let force = tree.actuators[actuator].torque(kin.length, kin.velocity);
    let mut rhs = vec![0.0; tree.nv()];
    for (slot, &coef) in kin.jacobian.iter().enumerate() {
        rhs[slot] += coef * force;
    }
    let mass = mass_matrix(&tree);
    let bias = bias_forces(&tree, Vec3::ZERO);
    for (slot, value) in rhs.iter_mut().enumerate() {
        *value -= bias[slot];
    }
    let factor = cholesky(&mass, tree.nv()).expect("dense mass matrix should be positive definite");
    let dense_qacc = cholesky_solve(&factor, tree.nv(), &rhs);

    let mut world = World::new();
    world.integrator = Integrator::ImplicitFast;
    world.dt = 0.005;
    world.gravity = Vec3::ZERO;
    world.add_tree(tree);
    let before = world.trees[0].qdot.clone();
    world.step();
    for slot in [world.trees[0].v_offset[1], world.trees[0].v_offset[2]] {
        let observed = (world.trees[0].qdot[slot] - before[slot]) / world.dt;
        assert!(
            (observed - dense_qacc[slot]).abs() < 1e-4,
            "muscle tendon qacc slot {slot}: observed {observed}, dense {}",
            dense_qacc[slot]
        );
    }
}

#[test]
fn implicitfast_joint_muscle_matches_mujoco_capture() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/references");
    let fixture = fs::read_to_string(root.join("muscle_implicitfast_joint.json"))
        .expect("implicitfast fixture readable");
    let document = json::parse(&fixture).expect("implicitfast fixture valid");
    let rows = match field(&document, "rows") {
        Value::Array(rows) => rows,
        _ => panic!("rows must be an array"),
    };
    let mut scene = load_mjcf_path(root.join("muscle_implicitfast_joint.xml"))
        .expect("implicitfast joint scene loads");
    let actuator = scene.actuators_by_name.values().next().unwrap().1;
    for row in rows {
        scene.world.trees[0].actuators[actuator].ctrl = 1.0;
        scene.world.step();
        let qpos = first_number(field(row, "qpos"), "qpos");
        let qvel = first_number(field(row, "qvel"), "qvel");
        let act = first_number(field(row, "act"), "act");
        assert!((scene.world.trees[0].q[0] - qpos).abs() < 3e-6);
        assert!((scene.world.trees[0].qdot[0] - qvel).abs() < 5e-6);
        assert!((scene.world.trees[0].actuators[actuator].act - act).abs() < 2e-6);
    }
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Object(fields) = value else {
        panic!("expected object for {name}")
    };
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
        .unwrap()
}

fn first_number(value: &Value, name: &str) -> f32 {
    let Value::Array(values) = value else {
        panic!("{name} must be an array")
    };
    let Value::Number(number) = values.first().unwrap() else {
        panic!("{name}[0] must be a number")
    };
    *number as f32
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
