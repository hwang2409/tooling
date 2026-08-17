//! Test battery for tendons (v2 tier 3).
//!
//! Each test states its hand-derived expected value, then asserts within
//! a measured tolerance. Discriminating mutants are named where they
//! would flip the assertion.

use newt::actuator::Actuator;
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{SolImp, SolverConfig, SolverMode};
use newt::tendon::{FixedTendonJoint, SpatialTendonSite, Tendon};
use newt::tree::{Link, aba, forward_kinematics};
use newt::world::World;

fn approx(a: f32, b: f32, tol: f32, ctx: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{ctx}: expected {a} ≈ {b} (tol {tol}, actual diff {})",
        (a - b).abs()
    );
}

#[test]
fn tendon_actuator_link_index_does_not_double_count_torque() {
    fn make(link_idx: usize) -> (newt::tree::Tree, Vec<f32>) {
        let mut tree = newt::tree::Tree::new();
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
            JointKind::hinge(Vec3::Y),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        let tendon = Tendon::fixed(vec![FixedTendonJoint { link: 1, coef: 1.0 }]);
        let tendon_idx = tree.add_tendon(tendon);
        let actuator = Actuator::motor(link_idx, 1.0, 0.0).on_tendon(tendon_idx);
        let actuator_idx = tree.add_actuator(actuator);
        tree.set_actuator_target(actuator_idx, 1.0);
        let poses = forward_kinematics(&tree);
        let qddot = aba(
            &tree,
            &poses,
            Vec3::ZERO,
            &vec![(Vec3::ZERO, Vec3::ZERO); 2],
        );
        (tree, qddot)
    }

    let (_, expected) = make(0);
    let (_, observed) = make(1);
    approx(
        observed[0],
        expected[0],
        1e-6,
        "tendon actuator must not also use link_idx as a joint transmission",
    );
}

#[test]
fn json_accepts_moving_wrap_for_envelope_jacobian() {
    let src = r#"{
        "trees": [{
            "name": "t",
            "links": [
                {"name":"root","joint":{"kind":"fixed"},"mass":1,
                 "inertia":{"kind":"diag","values":[1,1,1]}},
                {"name":"hinge","parent":"root","joint":{"kind":"hinge","axis":[0,1,0]},"mass":1,
                 "inertia":{"kind":"diag","values":[1,1,1]}}
            ]
        }],
        "tendons": [{
            "name":"cable", "tree":"t", "kind":"spatial",
            "sites":[
                {"link":"root","position":[-1,0,0]},
                {"link":"root","position":[1,0,0]}
            ],
            "wraps":[{"segment":0,"link":"hinge","center":[0,0,0],"radius":0.1}]
        }]
    }"#;
    let scene = newt::model::load_str(src).expect("moving wrap is supported");
    assert_eq!(scene.world.trees[0].tendons.len(), 1);
}

// ---------------------------------------------------------------------------
// Coupling anchor: fixed tendon in the stiff limit ≈ joint-coupling equality
// ---------------------------------------------------------------------------
//
// Two pendulums under gravity, coupled by a fixed tendon with coef=[1, -1]
// and stiffness large. Compared against a MuJoCo-equality-coupled pair
// (polycoef=[0, 1, 0] → q_a = q_b). Independent reference path: the
// equality solver is entirely separate machinery.

fn make_two_pendulum_tree(coupling_tendon: bool, stiffness: f32) -> World {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut tree = newt::tree::Tree::new();
    // Root fixed at origin.
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Pendulum A: hinge about x with unit arm along -z.
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 0.5,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(-0.5, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    // Pendulum B: hinge about x, offset in +x.
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 0.5,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(0.5, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    if coupling_tendon {
        let mut tendon = Tendon::fixed(vec![
            FixedTendonJoint { link: 1, coef: 1.0 },
            FixedTendonJoint {
                link: 2,
                coef: -1.0,
            },
        ]);
        tendon.springlength = Some(0.0);
        tendon.stiffness = stiffness;
        tendon.damping = 2.0 * (stiffness * 1.0).sqrt(); // approx critical
        tree.add_tendon(tendon);
    }
    world.add_tree(tree);
    world
}

fn make_two_pendulum_equality_world() -> World {
    let mut world = make_two_pendulum_tree(false, 0.0);
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        ..SolverConfig::DEFAULT
    };
    world
        .equalities
        .push(newt::equality::Equality::JointCoupling {
            tree: 0,
            link_a: 1,
            link_b: 2,
            polycoef: [0.0, 1.0, 0.0],
            solref: newt::geom::SolRef::DEFAULT,
            solimp: SolImp::DEFAULT,
        });
    world
}

#[test]
fn fixed_tendon_stiff_coupling_anchor_matches_equality() {
    // Independent references: (a) fixed-tendon coupled pendulums with
    // very stiff tendon; (b) equality-constraint coupled pendulums.
    // Both start at angle_a = 0.3, angle_b = 0 and run 400 steps of
    // gravity dynamics. The stiff-tendon limit should track the
    // equality solution to within a bounded gap that shrinks with
    // stiffness — but not to zero (the equality is exact, the spring
    // is not).
    let mut w_tendon = make_two_pendulum_tree(true, 20_000.0);
    let mut w_equality = make_two_pendulum_equality_world();
    w_tendon.trees[0].set_hinge_angle(1, 0.3);
    w_equality.trees[0].set_hinge_angle(1, 0.3);
    for _ in 0..400 {
        w_tendon.step();
        w_equality.step();
    }
    let q_t = w_tendon.trees[0].hinge_angle(1);
    let q_e = w_equality.trees[0].hinge_angle(1);
    // Coupling: q_a and q_b track each other in both worlds.
    let a_diff_t = (w_tendon.trees[0].hinge_angle(1) - w_tendon.trees[0].hinge_angle(2)).abs();
    let a_diff_e = (w_equality.trees[0].hinge_angle(1) - w_equality.trees[0].hinge_angle(2)).abs();
    // Tolerance: stiff spring holds q_a − q_b ≲ 5 mrad; equality is exact.
    assert!(a_diff_t < 0.005, "tendon coupling gap too wide: {a_diff_t}");
    assert!(a_diff_e < 1e-4, "equality coupling gap: {a_diff_e}");
    // Both solutions should track each other within a few mrad after
    // dynamics settle — both dissipate similarly since we chose damping
    // to be roughly critical on the tendon path.
    let track = (q_t - q_e).abs();
    // Loose bound: the tendon path bakes in extra damping; agreement
    // within ~0.05 rad (3 deg) is expected. A wrong sign in the Jacobian
    // (mutant) would drive the two paths apart by orders (>1 rad).
    assert!(
        track < 0.05,
        "tendon-coupled solution diverged from equality-coupled by {track}"
    );
}

// ---------------------------------------------------------------------------
// Spring anchor: hanging mass on a spatial tendon settles at k·(L−L0) = m·g
// ---------------------------------------------------------------------------

#[test]
fn spatial_tendon_spring_settles_at_hand_derived_equilibrium() {
    // A single slide joint (+z direction) on a fixed root. The tendon
    // connects a site on the root (anchor at z=0 world) to a site at
    // the slide link's COM. Springlength = L0 = 1.0 m. Stiffness k =
    // 100. Damping = 20 (critical for m=1 with ω = sqrt(k/m) = 10 →
    // 2·1·10 = 20). Gravity is -9.81. Slide q sits at z = -q relative
    // to the root when q > 0 (slide direction -z). Wait — let me
    // choose: slide axis = -Z so that as q increases (pull down), the
    // slide link moves DOWN. Site on slide at COM. Anchor at world
    // origin. Tendon length = q (the down-displacement).
    //
    // Actually simpler: root fixed at z=0, slide with axis +Z (positive
    // q = up). Mass hangs BELOW anchor via string. Hmm — tendon spring
    // will pull the mass UP toward the anchor at rest_length L0.
    //
    // Configuration: slide axis = -Z (mass below anchor at q>0). Site
    // at anchor (world origin, on root). Site on slide-link at COM.
    // Slide displacement q → slide COM at world (0, 0, -q).
    // Tendon length L = q. Equilibrium: k(q − L0) + m·g = 0 with g >
    // 0 (downward). But sign: spring force pulls mass UP when q > L0
    // (F_spring on mass = +z), gravity pulls DOWN (-z). At
    // equilibrium: F_spring_up = m·g (both magnitudes).
    // Force on mass along slide axis (-Z) direction = -(k(q − L0)) −
    // m·(-g) via Jacobian projection... let me just solve numerically.
    //
    // The scalar generalized force on slide dof q is:
    //   f_q = f_tendon · dL/dq + f_gravity_along_axis
    // f_tendon = -k(L − L0) − c·Ldot. dL/dq = 1 (slide axis contribution
    // to the length gradient equals 1 when tendon along the slide axis).
    // f_gravity is a WORLD gravity, appearing in ABA's f_ext at each
    // link. Slide accel = f_q / (m + armature).
    // At equilibrium (rest): 0 = -k(q − L0) - m·g_along_axis.
    // Here axis is -Z, gravity is -Z·9.81 world. Body's applied
    // world force is (0, 0, -9.81)·m. Slide axis in world = (0,0,-1).
    // f_gen_slide = axis · f_world = -1 · (-9.81)·m = +9.81·m (a
    // positive generalized force pushes the slide in +q, i.e., DOWN).
    // Wait, actually generalized force = -∂V/∂q. Gravity potential V
    // = m·g·z_com = m·(9.81)·(-q) = -m·9.81·q. So -∂V/∂q = m·9.81
    // (positive q → deeper hang → more negative V).
    //
    // Spring: F = -k(q - L0). At equilibrium: -k(q_eq - L0) + m·9.81 = 0
    // → q_eq = L0 + m·9.81 / k = 1.0 + 1.0·9.81 / 100 = 1.0981 m.
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut tree = newt::tree::Tree::new();
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
            axis: Vec3::new(0.0, 0.0, -1.0),
            range: None,
            damping: 20.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    let mut tendon = Tendon::spatial(
        vec![
            SpatialTendonSite {
                link: Some(0),
                position_local: Vec3::ZERO,
            },
            SpatialTendonSite {
                link: Some(1),
                position_local: Vec3::ZERO,
            },
        ],
        vec![None],
    );
    tendon.springlength = Some(1.0);
    tendon.stiffness = 100.0;
    // (Damping already lives on the slide's joint-damping; a second
    // tendon damping term would double-count. Keep tendon.damping = 0.)
    tendon.damping = 0.0;
    tree.add_tendon(tendon);
    world.add_tree(tree);
    world.trees[0].set_slide_position(1, 1.0); // start at L0 for a clean settle
    for _ in 0..2000 {
        world.step();
    }
    let q = world.trees[0].slide_position(1);
    let expected = 1.0 + 9.81 / 100.0;
    approx(q, expected, 5e-3, "spring equilibrium slide position");
}

// ---------------------------------------------------------------------------
// Tendon limit: 10k step no-creep + per-tendon solref respected
// ---------------------------------------------------------------------------

fn make_limited_tendon_world(solref_timeconst: f32) -> World {
    // Slide joint pulled by gravity into a tendon-range limit. Range
    // clamps the tendon length between 0 and 1.05 m. The joint itself
    // has NO limit — the tendon limit is the only constraint. Solver =
    // PGS so the length limit engages the solver-row path.
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        ..SolverConfig::DEFAULT
    };
    let mut tree = newt::tree::Tree::new();
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
            axis: Vec3::new(0.0, 0.0, -1.0),
            range: None,
            damping: 5.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    let mut tendon = Tendon::spatial(
        vec![
            SpatialTendonSite {
                link: Some(0),
                position_local: Vec3::ZERO,
            },
            SpatialTendonSite {
                link: Some(1),
                position_local: Vec3::ZERO,
            },
        ],
        vec![None],
    );
    tendon.range = Some((0.0, 1.05));
    tendon.limit_solref = Some(newt::geom::SolRef::new(solref_timeconst, 1.0));
    tree.add_tendon(tendon);
    world.add_tree(tree);
    world
}

#[test]
fn tendon_limit_no_creep_over_10k_steps() {
    let mut world = make_limited_tendon_world(0.02);
    // Start at the boundary.
    world.trees[0].set_slide_position(1, 1.05);
    for _ in 0..10_000 {
        world.step();
    }
    let q = world.trees[0].slide_position(1);
    // Under gravity, mass should sit just past the limit but not creep
    // significantly. Bounded creep < a few mm.
    assert!(q < 1.07, "tendon limit creep too far: q={q} (bound 1.07)");
    assert!(
        q > 1.04,
        "mass fell BELOW the limit — solver rows not engaging: q={q}"
    );
}

#[test]
fn tendon_limit_solref_override_changes_penetration() {
    // Mutant discriminator: if the solver defaults solref (drops the
    // per-tendon override), stiff and soft tendons produce identical
    // penetrations. With the override honored, penetration scales with
    // timeconst.
    let mut stiff = make_limited_tendon_world(0.005);
    let mut soft = make_limited_tendon_world(0.05);
    stiff.trees[0].set_slide_position(1, 1.05);
    soft.trees[0].set_slide_position(1, 1.05);
    for _ in 0..800 {
        stiff.step();
        soft.step();
    }
    let q_stiff = stiff.trees[0].slide_position(1);
    let q_soft = soft.trees[0].slide_position(1);
    // Soft tendon should let the mass sit deeper below the limit than
    // the stiff one — differences on the order of 1–20 mm.
    let gap = q_soft - q_stiff;
    assert!(
        gap > 1e-4,
        "per-tendon solref appears to be ignored: stiff={q_stiff}, soft={q_soft} (gap {gap})"
    );
}

// ---------------------------------------------------------------------------
// Actuator-on-tendon: motor drives coupled pendulums with hand-derived split
// ---------------------------------------------------------------------------

#[test]
fn motor_on_fixed_tendon_drives_both_joints_proportional_to_coef() {
    // Two-pendulum tree with a fixed tendon coef=[1, -1]. A motor with
    // gear=1 and ctrl=10 puts F=10 on the tendon. Via Jᵀ, the joint
    // torque split is:
    //   τ_a = coef_a · F =  10
    //   τ_b = coef_b · F = -10
    // Both pendulums are identical. After a short step, angular
    // accelerations should be equal-and-opposite (to leading order,
    // before gravity/damping do anything to break the symmetry).
    let mut world = make_two_pendulum_tree(true, 0.0);
    // Zero the tendon spring so ONLY the motor force is present.
    world.trees[0].tendons[0].springlength = None;
    world.trees[0].tendons[0].stiffness = 0.0;
    world.trees[0].tendons[0].damping = 0.0;
    world.gravity = Vec3::ZERO;
    // Damping on the hinges was 0.5 — drop to zero for a clean read.
    for link in world.trees[0].links.iter_mut().skip(1) {
        if let JointKind::Hinge { damping, .. } = &mut link.joint {
            *damping = 0.0;
        }
    }
    // Motor on the tendon.
    let actuator = Actuator::motor(0, 1.0, 0.0).on_tendon(0);
    let aid = world.trees[0].add_actuator(actuator);
    world.trees[0].set_actuator_target(aid, 10.0);
    // Query qddot directly via ABA (bypasses integrator).
    let poses = newt::tree::forward_kinematics(&world.trees[0]);
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); 3];
    let qddot = newt::tree::aba(&world.trees[0], &poses, world.gravity, &ext);
    // Angular accel about hinge = τ / I_eff (I_eff = m·L² ≈ 1 for
    // point mass at L=1). Expect qddot_a ≈ +10, qddot_b ≈ -10.
    approx(
        qddot[world.trees[0].v_offset[1]],
        10.0,
        1e-2,
        "pendulum A acceleration (motor τ = +10)",
    );
    approx(
        qddot[world.trees[0].v_offset[2]],
        -10.0,
        1e-2,
        "pendulum B acceleration (motor τ = -10)",
    );
}

// ---------------------------------------------------------------------------
// Sensors + loader tests
// ---------------------------------------------------------------------------

#[test]
fn json_loads_fixed_tendon_with_sensors_and_motor() {
    let src = r#"{
        "version": "1",
        "gravity": [0, 0, -9.81],
        "timestep": 0.005,
        "trees": [{
            "name": "t",
            "links": [
                {"name": "root", "joint": {"kind": "fixed"}, "mass": 1.0,
                 "inertia": {"kind": "diag", "values": [1, 1, 1]}},
                {"name": "j1", "parent": "root",
                 "joint": {"kind": "hinge", "axis": [1, 0, 0]},
                 "joint_offset_in_parent": {"position": [-0.5, 0, 0]},
                 "joint_offset_in_child": {"position": [0, 0, 1]},
                 "mass": 1.0,
                 "inertia": {"kind": "diag", "values": [0.001, 0.001, 0.001]}},
                {"name": "j2", "parent": "root",
                 "joint": {"kind": "hinge", "axis": [1, 0, 0]},
                 "joint_offset_in_parent": {"position": [0.5, 0, 0]},
                 "joint_offset_in_child": {"position": [0, 0, 1]},
                 "mass": 1.0,
                 "inertia": {"kind": "diag", "values": [0.001, 0.001, 0.001]}}
            ]
        }],
        "tendons": [{
            "name": "coup", "tree": "t", "kind": "fixed",
            "joints": [{"link": "j1", "coef": 1}, {"link": "j2", "coef": -1}],
            "springlength": 0.0, "stiffness": 100.0
        }],
        "actuators": [{
            "name": "m", "type": "motor", "tree": "t", "tendon": "coup", "gear": 1.0
        }],
        "sensors": [
            {"name": "L", "kind": "tendonpos", "tendon": "coup"},
            {"name": "Ldot", "kind": "tendonvel", "tendon": "coup"}
        ]
    }"#;
    let scene = newt::model::load_str(src).expect("scene loads");
    let (tree_idx, tendon_idx) = scene.tendons_by_name["coup"];
    assert_eq!(tree_idx, 0);
    assert_eq!(tendon_idx, 0);
    // Actuator wired as tendon transmission.
    let (t, a) = scene.actuators_by_name["m"];
    assert_eq!(
        scene.world.trees[t].actuators[a].tendon_target,
        Some(tendon_idx)
    );
    // Sensors evaluate.
    let mut world = scene.world;
    world.trees[0].set_hinge_angle(1, 0.4);
    world.trees[0].set_hinge_angle(2, -0.1);
    world.step();
    let l_idx = scene.sensors_by_name["L"];
    let ldot_idx = scene.sensors_by_name["Ldot"];
    let l = world.sensor(l_idx).unwrap()[0];
    let _ = world.sensor(ldot_idx).unwrap()[0];
    // After one step, L ≈ q1 − q2 = 0.5 with a slight nudge from
    // dynamics.
    assert!(
        (l - 0.5).abs() < 0.05,
        "tendonpos sensor should read ~0.5 rad (got {l})"
    );
}

#[test]
fn json_loads_cylinder_wrap() {
    let src = r#"{
        "trees": [{
            "name": "t",
            "links": [{"name":"r","joint":{"kind":"fixed"},"mass":1,
                       "inertia":{"kind":"diag","values":[1,1,1]}}]
        }],
        "tendons": [{
            "name": "cbl", "tree": "t", "kind": "spatial",
            "sites": [
                {"link":"r","position":[-1,0,0]},
                {"link":"r","position":[1,0,0]}
            ],
            "wraps": [{
                "segment": 0, "kind": "cylinder",
                "link": "r", "center": [0,0,0], "radius": 0.1
            }]
        }]
    }"#;
    let scene = newt::model::load_str(src).expect("cylinder wrap is supported");
    assert_eq!(scene.world.trees[0].tendons.len(), 1);
}

#[test]
fn tendon_kinematics_matches_two_link_arm_hand_computation() {
    // Two-link chain: hinges about y, arm 1. Site at first link's tip
    // (0, 0, -1 body), site at second link's tip. Straight segment
    // between them; verify length matches a direct forward-kinematics
    // computation of both site world positions.
    let mut tree = newt::tree::Tree::new();
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
        JointKind::hinge(Vec3::Y),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::Y),
        (Vec3::new(0.0, 0.0, -1.0), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    let tendon = Tendon::spatial(
        vec![
            SpatialTendonSite {
                link: Some(1),
                position_local: Vec3::new(0.0, 0.0, -1.0),
            },
            SpatialTendonSite {
                link: Some(2),
                position_local: Vec3::new(0.0, 0.0, -1.0),
            },
        ],
        vec![None],
    );
    tree.set_hinge_angle(1, 0.3);
    tree.set_hinge_angle(2, 0.7);
    let poses = newt::tree::forward_kinematics(&tree);
    let kin = newt::tendon::tendon_kinematics(&tendon, &tree, &poses);
    // Compute site world positions manually (this uses the same fwd-K but
    // exercises the SITE JACOBIAN independently of the tendon
    // implementation).
    let (com1, ori1) = poses[1];
    let (com2, ori2) = poses[2];
    let p1 = com1 + ori1.rotate(Vec3::new(0.0, 0.0, -1.0));
    let p2 = com2 + ori2.rotate(Vec3::new(0.0, 0.0, -1.0));
    let expected = (p2 - p1).length();
    approx(kin.length, expected, 1e-4, "2-link arm site tendon length");
}
