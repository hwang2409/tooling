//! Tree-contact solver anchors.
//!
//! These scenes keep the tree contact path separate from the legacy penalty
//! path. The force checks use the touch sensor, which reads the solved normal
//! impulse after the original contact-index mapping.

use newt::body::Body;
use newt::geom::Geom;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::sensor::{Sensor, SensorKind};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::{Integrator, World};

fn free_sphere_tree(z: f32) -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(0.0, 0.0, z), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.1, 0.1, 0.1),
    ));
    tree
}

fn resting_tree_world(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.0));
    let tree = world.add_tree(free_sphere_tree(0.6));
    let sphere = world.add_geom(Geom::sphere_on_link(tree, 0, 0.5, Vec3::ZERO, 0.0));
    world
        .add_sensor(Sensor {
            name: "tree-touch".into(),
            kind: SensorKind::Touch { geom: sphere },
        })
        .expect("touch sensor should validate");
    world
}

fn asymmetric_tree_world(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.7));
    let tree = world.add_tree(free_sphere_tree(0.72));
    let orientation = Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.2).normalize(), 0.35);
    world.trees[tree].q[0] = 0.11;
    world.trees[tree].q[1] = -0.08;
    world.trees[tree].q[3..7].copy_from_slice(&[
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]);
    world.trees[tree]
        .qdot
        .copy_from_slice(&[0.7, -0.4, 0.3, 0.2, -0.1, -0.3]);
    world.add_geom(Geom::box_on_link(
        tree,
        0,
        Vec3::new(0.22, 0.17, 0.19),
        Vec3::new(0.08, -0.06, 0.0),
        Quat::from_axis_angle(Vec3::Y, 0.2),
        0.7,
    ));
    world
}

#[test]
fn tree_contact_solver_supports_weight_with_pgs_and_newton() {
    for mode in [SolverMode::Pgs, SolverMode::Newton] {
        let mut world = resting_tree_world(mode);
        for _ in 0..400 {
            world.step();
        }
        let force = world.sensor(0).expect("touch reading")[0];
        let z = world.trees[0].q[2];
        let vz = world.trees[0].qdot[3 + 2];
        assert!((force - 9.81).abs() < 0.15, "{mode:?} force={force}");
        assert!((z - 0.5).abs() < 0.02, "{mode:?} z={z}");
        assert!(vz.abs() < 0.05, "{mode:?} vz={vz}");
        assert!(
            world.trees[0]
                .qfrc_applied
                .iter()
                .all(|force| *force == 0.0),
            "{mode:?} solver impulse leaked into persistent qfrc_applied"
        );
    }
}

#[test]
fn tree_contact_solver_supports_implicitfast_pipeline() {
    for mode in [SolverMode::Pgs, SolverMode::Newton] {
        let mut world = resting_tree_world(mode);
        world.integrator = Integrator::ImplicitFast;
        for _ in 0..400 {
            world.step();
        }
        let force = world.sensor(0).expect("touch reading")[0];
        assert!((force - 9.81).abs() < 0.15, "{mode:?} force={force}");
        assert!(world.trees[0].qdot[5].abs() < 0.05, "{mode:?} vz");
    }
}

#[test]
fn tree_contact_cross_solver_agreement_stays_tight() {
    let mut pgs = asymmetric_tree_world(SolverMode::Pgs);
    let mut newton = asymmetric_tree_world(SolverMode::Newton);
    let mut max_q = 0.0f32;
    let mut max_qdot = 0.0f32;
    for _ in 0..100 {
        pgs.step();
        newton.step();
        for (a, b) in pgs.trees[0].q.iter().zip(&newton.trees[0].q) {
            max_q = max_q.max((a - b).abs());
        }
        for (a, b) in pgs.trees[0].qdot.iter().zip(&newton.trees[0].qdot) {
            max_qdot = max_qdot.max((a - b).abs());
        }
    }
    println!("tree contact pgs/newton max q={max_q:.6e} qdot={max_qdot:.6e}");
    assert!(max_q < 1.0e-6, "max tree q delta={max_q}");
    assert!(max_qdot < 2.0e-6, "max tree qdot delta={max_qdot}");
}

#[test]
fn tree_contact_touch_keeps_force_free_contact_index_alignment() {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    let tree = world.add_tree(free_sphere_tree(0.45));
    let upper_plane = world.add_geom(Geom::static_plane(Vec3::new(0.0, 0.0, 0.97), Vec3::Z, 0.0));
    let lower_plane = world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.0));
    let upper_sphere = world.add_geom(Geom::sphere_on_link(
        tree,
        0,
        0.5,
        Vec3::new(0.0, 0.0, 1.0),
        0.0,
    ));
    let lower_sphere = world.add_geom(Geom::sphere_on_link(tree, 0, 0.5, Vec3::ZERO, 0.0));
    world.geoms[upper_sphere].gap = 0.1;
    world.pair_list = Some(vec![
        (upper_plane, upper_sphere),
        (lower_plane, lower_sphere),
    ]);
    world
        .add_sensor(Sensor {
            name: "lower-touch".into(),
            kind: SensorKind::Touch { geom: lower_sphere },
        })
        .expect("touch sensor should validate");
    world.evaluate_sensors(&[(upper_plane, upper_sphere), (lower_plane, lower_sphere)]);
    let force = world.sensor(0).expect("touch reading")[0];
    // The upper contact is inside its force-free gap and occupies index 0.
    // The active lower contact is index 1. A compact-row indexing mutant
    // reports zero here.
    assert!(force > 1.0, "lower contact force={force}");
}

#[test]
fn tree_body_contact_transfers_equal_and_opposite_impulses() {
    for mode in [SolverMode::Pgs, SolverMode::Newton] {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        world.solver = SolverConfig {
            mode,
            iterations: 30,
            cone: ConeKind::Pyramidal,
        };
        let tree = world.add_tree(free_sphere_tree(0.0));
        world.trees[tree].q[0] = -0.4;
        let body = world.add_body(Body::solid_sphere(
            2.0,
            0.5,
            Vec3::new(0.4, 0.0, 0.0),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::sphere_on_link(tree, 0, 0.5, Vec3::ZERO, 0.0));
        world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.0));
        world.step();
        let tree_momentum = Vec3::new(
            world.trees[tree].qdot[3],
            world.trees[tree].qdot[4],
            world.trees[tree].qdot[5],
        );
        let body_momentum = world.bodies[body].linear_velocity * world.bodies[body].mass;
        let total_momentum = tree_momentum + body_momentum;
        assert!(
            total_momentum.x.abs() < 1e-4,
            "{mode:?} total px={}",
            total_momentum.x
        );
        assert!(
            total_momentum.y.abs() < 1e-4,
            "{mode:?} total py={}",
            total_momentum.y
        );
        assert!(
            total_momentum.z.abs() < 1e-4,
            "{mode:?} total pz={}",
            total_momentum.z
        );
    }
}

#[test]
fn tree_tree_contact_transfers_equal_and_opposite_impulses() {
    for mode in [SolverMode::Pgs, SolverMode::Newton] {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        world.solver = SolverConfig {
            mode,
            iterations: 30,
            cone: ConeKind::Pyramidal,
        };
        let left = world.add_tree(free_sphere_tree(0.0));
        let right = world.add_tree(free_sphere_tree(0.0));
        world.trees[left].q[0] = -0.4;
        world.trees[right].q[0] = 0.4;
        world.trees[left].qdot[3] = 1.0;
        world.trees[right].qdot[3] = -1.0;
        world.add_geom(Geom::sphere_on_link(left, 0, 0.5, Vec3::ZERO, 0.0));
        world.add_geom(Geom::sphere_on_link(right, 0, 0.5, Vec3::ZERO, 0.0));
        world.step();
        let total_px = world.trees[left].qdot[3] + world.trees[right].qdot[3];
        assert!(total_px.abs() < 1.0e-4, "{mode:?} total px={total_px}");
    }
}

#[test]
fn penalty_mode_keeps_legacy_free_root_force_behavior() {
    let mut with_force = World::new();
    let mut without_force = World::new();
    let tree_with_force = with_force.add_tree(free_sphere_tree(1.0));
    let tree_without_force = without_force.add_tree(free_sphere_tree(1.0));
    with_force.trees[tree_with_force].qfrc_applied[5] = 100.0;
    with_force.step();
    without_force.step();
    assert_eq!(
        with_force.trees[tree_with_force].q,
        without_force.trees[tree_without_force].q
    );
    assert_eq!(
        with_force.trees[tree_with_force].qdot,
        without_force.trees[tree_without_force].qdot
    );
}
