//! Constraint-based joint-limit anchor: pendulum released above its range.
//!
//! Solver-mode joint limits replace the tier-3 penalty limit torque with a
//! per-DOF constraint solved by the same PGS as contacts. The anchor: a
//! hinge with `range = (-π/4, +π/4)` released at π/3 (above range). The
//! joint must settle at or just inside the upper limit and NOT drift
//! outward over 10 000 steps.

use newt::body::Body;
use newt::geom::Geom;
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::World;

fn pendulum_with_range(range: (f32, f32)) -> World {
    let mut w = World::new();
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    // No contacts — just a tree with one hinge and range limits.
    let mut tree = Tree::new();
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
        JointKind::Hinge {
            axis: Vec3::X,
            range: Some(range),
            damping: 0.5,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    w.add_tree(tree);
    w
}

#[test]
fn solver_hinge_limit_holds_pendulum_at_upper_bound() {
    let range = (
        -std::f32::consts::FRAC_PI_4,
        std::f32::consts::FRAC_PI_4, // +45°
    );
    let mut w = pendulum_with_range(range);
    // Release well ABOVE the upper limit — π/3 = 60°. Under gravity, the
    // pendulum should be pulled BACK INSIDE the range and settle. Skip
    // the initial swing (2000 steps @ 5ms = 10s) then measure the max
    // angle over the following 2000 steps — that's the true "does the
    // limit hold?" question.
    w.trees[0].set_hinge_angle(1, std::f32::consts::FRAC_PI_3);
    for _ in 0..4000 {
        w.step();
    }
    let mut max_angle_after_settle = f32::NEG_INFINITY;
    for _ in 0..2000 {
        w.step();
        let a = w.trees[0].hinge_angle(1);
        if a > max_angle_after_settle {
            max_angle_after_settle = a;
        }
    }
    // After settling, the angle must be at or just past the limit — small
    // penetration proportional to solimp.width (~1 mm ≈ 0.001 rad).
    assert!(
        max_angle_after_settle < range.1 + 0.05,
        "solver limit failed: post-settle max angle {max_angle_after_settle} exceeded {} + 0.05",
        range.1
    );
}

#[test]
fn solver_hinge_limit_no_creep_over_10k_steps() {
    let range = (-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4);
    let mut w = pendulum_with_range(range);
    // Start AT the upper limit and rest there under gravity that pushes
    // OUT of range (rotate pendulum so gravity torques it +q direction).
    // We simulate an "against the limit" scenario by starting at +π/4 and
    // applying a persistent torque outward via qfrc_applied.
    w.trees[0].set_hinge_angle(1, std::f32::consts::FRAC_PI_4);
    // Persistent outward torque = 5 N·m. This would push q past +π/4 if
    // the limit weren't holding it.
    let slot = w.trees[0].v_offset[1];
    w.trees[0].qfrc_applied[slot] = 5.0;
    let mut max_penetration = 0.0f32;
    for _ in 0..10_000 {
        w.step();
        let over = w.trees[0].hinge_angle(1) - range.1;
        if over > max_penetration {
            max_penetration = over;
        }
    }
    // Penetration must stay bounded (no runaway creep).
    assert!(
        max_penetration < 0.05,
        "solver limit creep: max penetration {max_penetration} rad exceeded 0.05 over 10k steps"
    );
}

// Suppress the "unused import" warnings by referencing them in a trivial
// test.
#[test]
fn imports_are_wired() {
    let _ = Body::solid_sphere(1.0, 0.5, Vec3::ZERO, Quat::IDENTITY);
    let _ = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5);
}
