//! Tree-contact solver anchors.
//!
//! These scenes keep the tree contact path separate from the legacy penalty
//! path. The force checks use the touch sensor, which reads the solved normal
//! impulse after the original contact-index mapping.

use newt::body::Body;
use newt::contact::{Contact, narrow_phase, narrow_phase_solver};
use newt::geom::{Geom, geom_world_pose};
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

fn free_root_contact_row(tree: &Tree, point: Vec3, sign: f32) -> Vec<f32> {
    let root = Vec3::new(tree.q[0], tree.q[1], tree.q[2]);
    tree.point_jacobian(0, point - root)
        .translational
        .into_iter()
        .map(|column| column.dot(Vec3::X) * sign)
        .collect()
}

fn hand_free_root_response(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(slot, (a, b))| {
            let inverse_mass = if slot < 3 { 10.0 } else { 1.0 };
            a * b * inverse_mass
        })
        .sum()
}

fn resting_tree_world(mode: SolverMode) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations: 40,
        pgs_tolerance: 0.0,
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
    asymmetric_tree_world_with_iterations(mode, 40)
}

fn asymmetric_tree_world_with_iterations(mode: SolverMode, iterations: u32) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode,
        iterations,
        pgs_tolerance: 0.0,
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

fn tilted_tree_box_box_world(mode: SolverMode) -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::ZERO;
    world.solver = SolverConfig {
        mode,
        iterations: 40,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };

    let half = Vec3::splat(0.5);
    let mut tree = free_sphere_tree(1.48);
    let tree_orientation = Quat::from_axis_angle(Vec3::X, 0.005);
    tree.q[3..7].copy_from_slice(&[
        tree_orientation.x,
        tree_orientation.y,
        tree_orientation.z,
        tree_orientation.w,
    ]);
    let tree_index = world.add_tree(tree);
    world.add_geom(Geom::box_on_link(
        tree_index,
        0,
        half,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));

    let body = world.add_body(Body::solid_box(
        1.0,
        half,
        Vec3::new(0.0, 0.0, 0.5),
        Quat::from_axis_angle(Vec3::X, 0.01),
    ));
    world.add_geom(Geom::r#box(body, half, Vec3::ZERO, Quat::IDENTITY, 0.0));
    world.set_solver_phase_capture(true);
    world
}

#[test]
fn tree_contact_cross_solver_iteration_probe() {
    let mut pgs = asymmetric_tree_world_with_iterations(SolverMode::Pgs, 400);
    let mut newton = asymmetric_tree_world_with_iterations(SolverMode::Newton, 40);
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
    println!("tree contact pgs(400)/newton(40) max q={max_q:.6e} qdot={max_qdot:.6e}");
}

#[test]
fn tilted_tree_box_box_onset_uses_solver_manifold_for_tree_rows() {
    let probe_world = tilted_tree_box_box_world(SolverMode::Pgs);
    let tree_geom = &probe_world.geoms[0];
    let body_geom = &probe_world.geoms[1];
    let tree_pose = geom_world_pose(
        tree_geom,
        Vec3::new(0.0, 0.0, 1.48),
        Quat::from_axis_angle(Vec3::X, 0.005),
    );
    let body_pose = geom_world_pose(
        body_geom,
        Vec3::new(0.0, 0.0, 0.5),
        Quat::from_axis_angle(Vec3::X, 0.01),
    );
    assert_eq!(
        narrow_phase(0, tree_geom, &tree_pose, 1, body_geom, &body_pose, &[]).len,
        4,
        "legacy tree manifold mutant baseline"
    );
    assert_eq!(
        narrow_phase_solver(0, tree_geom, &tree_pose, 1, body_geom, &body_pose, &[]).len,
        1,
        "tilted box-box solver manifold"
    );
    for mode in [SolverMode::Pgs, SolverMode::Newton] {
        let mut world = tilted_tree_box_box_world(mode);
        world.step();
        let phase = world.solver_phase_diagnostics().unwrap();
        assert_eq!(phase.contacts.len(), 1, "{mode:?} solver contact count");
        assert_eq!(phase.row_to_contact.len(), 4, "{mode:?} solver row count");
        let contact = phase.contacts[0];
        assert!((contact.position_world.x - 0.5).abs() < 1.0e-5);
        assert!((contact.position_world.y - 0.4949751).abs() < 1.0e-5);
        assert!((contact.position_world.z - 1.004975).abs() < 1.0e-5);
        assert!((contact.penetration - 0.02250594).abs() < 2.0e-5);
        assert!(phase.tree_qfrc[0][5] > 1.0, "{mode:?} normal response");
        assert!(world.trees[0].qdot[5] > 0.01, "{mode:?} tree response");
        assert!(
            world.bodies[0].linear_velocity.z < 0.0,
            "{mode:?} body response"
        );
    }
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
fn solver_modes_route_mocap_contacts_into_shared_rows() {
    for mode in [SolverMode::Pgs, SolverMode::Newton] {
        let mut world = World::new();
        world.dt = 0.005;
        world.gravity = Vec3::ZERO;
        world.solver = SolverConfig {
            mode,
            iterations: 40,
            pgs_tolerance: 0.0,
            cone: ConeKind::Pyramidal,
        };

        let mut platform = free_sphere_tree(0.25);
        platform.q[0] = -0.4;
        platform.set_mocap(0, true);
        platform.set_mocap_velocity(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        let platform_tree = world.add_tree(platform);
        world.add_geom(Geom::sphere_on_link(platform_tree, 0, 0.3, Vec3::ZERO, 0.8));
        let body = world.add_body(Body::solid_sphere(
            1.0,
            0.2,
            Vec3::new(0.0, 0.0, 0.25),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::sphere(body, 0.2, Vec3::ZERO, 0.0));

        for step in 0..100 {
            world.set_mocap_pose(
                platform_tree,
                Vec3::new(-0.4 + step as f32 * world.dt, 0.0, 0.25),
                Quat::IDENTITY,
            );
            world.step();
        }
        assert!(
            world.bodies[body].position.x > 0.05,
            "{mode:?} mocap platform failed to push sphere: x={}",
            world.bodies[body].position.x
        );
    }
}

#[test]
fn implicit_tree_contact_response_matches_hand_matrix() {
    for implicit_fast in [false, true] {
        let mut no_damping_force = None;
        for damping in [0.0, 100.0] {
            let mut world = World::new();
            world.dt = 0.01;
            world.gravity = Vec3::ZERO;
            world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.0));
            let tree = world.add_tree(free_sphere_tree(0.4));
            world.trees[tree].links[0].free_damping = damping;
            world.add_geom(Geom::sphere_on_link(tree, 0, 0.5, Vec3::ZERO, 0.0));
            let contacts = world.detect_contacts();
            let solution = newt::solver::solve_tree_contacts(
                &world.bodies,
                &world.trees,
                &world.geoms,
                &contacts,
                world.gravity,
                world.dt,
                ConeKind::Pyramidal,
                40,
                false,
                Some(implicit_fast),
            );
            let response = world.trees[tree].implicit_mass_matrix(world.dt, implicit_fast);
            let expected = 1.0 / (1.0 + world.dt * damping);
            let observed = response[5 * world.trees[tree].nv() + 5];
            assert!((1.0 / observed - expected).abs() < 1.0e-5);
            let assembled = solution
                .contact_response
                .first()
                .copied()
                .expect("plane contact must assemble one response row");
            assert!((assembled - expected).abs() < 1.0e-5);
            let contact_force = solution.tree_qfrc[tree][5];
            assert!(contact_force > 0.0);
            if damping == 0.0 {
                no_damping_force = Some(contact_force);
            } else {
                let no_damping_force = no_damping_force.expect("baseline response");
                assert!(
                    (no_damping_force / contact_force - expected).abs() < 1.0e-5,
                    "implicit_fast={implicit_fast} contact response={}; expected {}",
                    no_damping_force / contact_force,
                    expected
                );
            }
        }
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
    // Shared MuJoCo diagApprox/Rpy regularization changes this finite-solver
    // residual. A 10x PGS iteration probe measured the same gap, so this is
    // not iteration starvation. Bounds keep measured headroom explicit.
    // Measured maxima: 1.365253e-2 q and 4.467820e-1 qdot.
    assert!(max_q < 2.0e-2, "max tree q delta={max_q}");
    assert!(max_qdot < 5.0e-1, "max tree qdot delta={max_qdot}");
}

#[test]
fn tree_contact_touch_keeps_force_free_contact_index_alignment() {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        pgs_tolerance: 0.0,
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
            pgs_tolerance: 0.0,
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
            pgs_tolerance: 0.0,
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
fn tree_tree_two_contact_response_includes_the_opposite_tree() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let left = world.add_tree(free_sphere_tree(0.0));
    let right = world.add_tree(free_sphere_tree(0.0));
    world.trees[left].q[0] = -0.4;
    world.trees[right].q[0] = 0.4;
    let left_a = world.add_geom(Geom::sphere_on_link(
        left,
        0,
        0.2,
        Vec3::new(0.0, 0.05, 0.0),
        0.0,
    ));
    let left_b = world.add_geom(Geom::sphere_on_link(
        left,
        0,
        0.2,
        Vec3::new(0.0, -0.05, 0.0),
        0.0,
    ));
    let right_a = world.add_geom(Geom::sphere_on_link(
        right,
        0,
        0.2,
        Vec3::new(0.0, 0.05, 0.0),
        0.0,
    ));
    let right_b = world.add_geom(Geom::sphere_on_link(
        right,
        0,
        0.2,
        Vec3::new(0.0, -0.05, 0.0),
        0.0,
    ));
    for geom in [left_a, left_b, right_a, right_b] {
        world.geoms[geom].condim = 1;
    }
    let points = [Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.0, -0.05, 0.0)];
    let contacts = points
        .iter()
        .zip([(left_a, right_a), (left_b, right_b)])
        .map(|(&point, (geom_a, geom_b))| Contact {
            geom_a,
            geom_b,
            position_world: point,
            normal_world: Vec3::X,
            penetration: 0.1,
            friction: 0.0,
            gap: 0.0,
        })
        .collect::<Vec<_>>();
    let solution = newt::solver::solve_tree_contacts(
        &world.bodies,
        &world.trees,
        &world.geoms,
        &contacts,
        world.gravity,
        0.005,
        ConeKind::Pyramidal,
        40,
        true,
        None,
    );
    assert!(
        solution.tree_qfrc[right][3].abs() > 1.0e-3,
        "opposite tree received no contact acceleration"
    );

    let left_rows = points.map(|point| free_root_contact_row(&world.trees[left], point, 1.0));
    let right_rows = points.map(|point| free_root_contact_row(&world.trees[right], point, -1.0));
    for row in 0..2 {
        for column in 0..2 {
            let hand = hand_free_root_response(&left_rows[row], &left_rows[column])
                + hand_free_root_response(&right_rows[row], &right_rows[column]);
            let measured = solution.contact_response[row * 2 + column];
            assert!(
                (measured - hand).abs() < 1.0e-5,
                "response[{row},{column}] measured={measured} hand={hand}"
            );
        }
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
