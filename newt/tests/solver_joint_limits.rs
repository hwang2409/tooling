//! Constraint-based joint-limit anchor: pendulum released above its range.
//!
//! Solver-mode joint limits replace the tier-3 penalty limit torque with a
//! per-DOF constraint solved by the same PGS as contacts. The anchor: a
//! hinge with `range = (-π/4, +π/4)` released at π/3 (above range). The
//! joint must settle at or just inside the upper limit and NOT drift
//! outward over 10 000 steps.
//!
//! # Belt-and-braces: structural penalty neutralization
//!
//! Round-2 review caught a double-enforcement bug: without the mode gate
//! in [`crate::tree::aba`] (added in this same round), the tier-3
//! penalty limit force STACKED on top of the PGS limit, and both anchors
//! passed even with the PGS limit no-op'd or sign-flipped. To keep this
//! test discriminating for the PGS pathway even if a future refactor
//! regressed the mode gate, every pendulum here is constructed with a
//! [`JointLimit`] that has ZERO stiffness AND zero damping — so
//! [`crate::tree::joint_limit_scalar_force`] returns exactly zero and
//! any surviving penalty contribution would be structurally neutralized.
//! The PGS limit is then the ONLY force holding the joint at its range.

use newt::body::Body;
use newt::geom::Geom;
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::World;

/// Zero-everything limit config so any surviving penalty contribution
/// evaluates to zero — leaves the PGS constraint as the sole limit
/// authority under test. See the module docs.
const NEUTRALIZED_LIMIT: JointLimit = JointLimit {
    stiffness: 0.0,
    damping: 0.0,
    solref: None,
    solimp: None,
};

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
            limit: NEUTRALIZED_LIMIT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    w.add_tree(tree);
    w
}

/// Discrimination requirement (round-2 review): every anchor here must
/// FAIL with either the reviewer's mutant of the PGS accumulate loop
/// (no-op) OR a sign-flip mutant. Since the penalty term is already
/// neutralized to zero (see [`NEUTRALIZED_LIMIT`]), the only way to
/// FAIL these anchors is to break the PGS impulse pathway itself.
///
/// Anchor design: apply a persistent outward torque > `m·g·L`
/// (gravity's max restoring torque) so gravity CANNOT hold the joint
/// inside the range — only the PGS constraint can. Any regression to
/// the PGS pathway lets the joint runaway past the boundary.
///
/// With `m=1, g=9.81, L=1`, gravity max torque is `9.81 N·m` at
/// `q=π/2`. Persistent torque `12 N·m` exceeds this everywhere, so
/// under a no-op or sign-flipped PGS the pendulum accelerates through
/// the range and spins without bound.
const OUTWARD_TORQUE_NM: f32 = 12.0;

#[test]
fn solver_hinge_limit_holds_pendulum_released_above_range() {
    let range = (
        -std::f32::consts::FRAC_PI_4,
        std::f32::consts::FRAC_PI_4, // +45°
    );
    let mut w = pendulum_with_range(range);
    // Release ABOVE the upper limit and pin gravity aside by applying a
    // persistent outward torque bigger than gravity's max restoring
    // torque (see OUTWARD_TORQUE_NM). Only PGS can hold the joint
    // inside the range.
    w.trees[0].set_hinge_angle(1, std::f32::consts::FRAC_PI_3);
    let slot = w.trees[0].v_offset[1];
    w.trees[0].qfrc_applied[slot] = OUTWARD_TORQUE_NM;
    // Under PGS the joint drops from π/3 to just past the limit and
    // holds. Under a broken PGS the outward torque wins over gravity
    // and the joint accelerates outward. Measure the peak angle across
    // the ENTIRE run, not post-settle — a broken PGS may find a lower
    // "quasi-steady" but blow through the boundary in transients.
    let mut peak_angle = w.trees[0].hinge_angle(1);
    for _ in 0..6000 {
        w.step();
        let a = w.trees[0].hinge_angle(1);
        if a > peak_angle {
            peak_angle = a;
        }
    }
    // Tolerance: `OUTWARD_TORQUE_NM = 12 N·m` versus a critically-damped
    // limit spring's transient overshoot allows some ballistic
    // penetration on the first swing. Empirically PGS caps the peak at
    // ~π/3 (release angle) or just below; a sign-flipped or no-op PGS
    // sends the peak to > 100 rad within a few hundred steps.
    assert!(
        peak_angle < std::f32::consts::FRAC_PI_3 + 0.2,
        "solver limit failed: peak angle {peak_angle} exceeded release+0.2"
    );
}

#[test]
fn solver_hinge_limit_no_creep_over_10k_steps() {
    let range = (-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4);
    let mut w = pendulum_with_range(range);
    // Start AT the upper limit with a persistent outward torque bigger
    // than gravity's max restoring torque. Only PGS can bound the
    // penetration; without it, the joint accelerates through the range
    // and monotonically spins.
    w.trees[0].set_hinge_angle(1, std::f32::consts::FRAC_PI_4);
    let slot = w.trees[0].v_offset[1];
    w.trees[0].qfrc_applied[slot] = OUTWARD_TORQUE_NM;
    let mut max_penetration = 0.0f32;
    for _ in 0..10_000 {
        w.step();
        let over = w.trees[0].hinge_angle(1) - range.1;
        if over > max_penetration {
            max_penetration = over;
        }
    }
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

/// Per-joint SolRef override discrimination: a very soft SolRef
/// (timeconst = 0.2s) settles the limit LOWER on the boundary than a
/// stiff one (default 0.02s). We measure penetration under both and
/// assert the soft-override lets more through — confirming that
/// `JointLimit::solref` actually threads into `solve_tree_limits`.
#[test]
fn solver_hinge_limit_solref_override_changes_penetration() {
    use newt::geom::SolRef;
    let range = (-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4);

    fn build_with(solref_override: Option<SolRef>, range: (f32, f32)) -> World {
        let mut w = World::new();
        w.solver = SolverConfig {
            mode: SolverMode::Pgs,
            iterations: 30,
            cone: ConeKind::Pyramidal,
        };
        let mut tree = Tree::new();
        tree.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        let mut limit = NEUTRALIZED_LIMIT;
        limit.solref = solref_override;
        tree.push_link(Link::new(
            Some(0),
            JointKind::Hinge {
                axis: Vec3::X,
                range: Some(range),
                damping: 0.5,
                armature: 0.0,
                limit,
            },
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
            1.0,
            Mat3::diag(0.01, 0.01, 0.01),
        ));
        w.add_tree(tree);
        w.trees[0].set_hinge_angle(1, std::f32::consts::FRAC_PI_4);
        let slot = w.trees[0].v_offset[1];
        w.trees[0].qfrc_applied[slot] = 12.0;
        w
    }

    // Default SolRef (stiff, 0.02s) — small steady penetration.
    let mut stiff = build_with(None, range);
    let mut soft = build_with(Some(SolRef::new(0.2, 1.0)), range);

    for _ in 0..3000 {
        stiff.step();
        soft.step();
    }
    let stiff_pen = stiff.trees[0].hinge_angle(1) - range.1;
    let soft_pen = soft.trees[0].hinge_angle(1) - range.1;
    // Both must stay bounded (limit still holds) but the soft-override
    // must let strictly more through — proof that solref threads in.
    assert!(stiff_pen < 0.05, "stiff limit blew out: pen={stiff_pen}");
    assert!(soft_pen < 0.5, "soft limit runaway: pen={soft_pen}");
    assert!(
        soft_pen > stiff_pen + 0.001,
        "solref override did not change penetration: stiff={stiff_pen}, soft={soft_pen}"
    );
}
