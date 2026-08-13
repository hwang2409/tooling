//! Anchors for the three hinge features that ship in this tier:
//!
//! - **damping**: a damped pendulum's successive swing peaks strictly
//!   decrease. Missing damping code or the wrong sign would let peaks stay
//!   constant or grow.
//! - **armature**: a hinge with large armature and a fixed torque
//!   accelerates by exactly the hand-computed ratio slower than the same
//!   torque on a hinge without armature.
//! - **range limits**: a pendulum released above its lower limit settles
//!   inside/near the limit; no energy is *gained* across limit bounces
//!   (the penalty spring only removes energy through its damper).

use newt::joint::{HingeLimit, JointKind};
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, aba, forward_kinematics, rk4_step};

fn build_hinge_link(_axis: Vec3, l: f32, m: f32, joint: JointKind, initial_angle: f32) -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let i_perp = (1.0 / 12.0) * m * l * l;
    tree.push_link(Link::new(
        Some(0),
        joint,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
        m,
        Mat3::diag(i_perp, i_perp, 1e-6),
    ));
    tree.set_hinge_angle(1, initial_angle);
    tree
}

#[test]
fn damped_pendulum_peaks_are_strictly_decreasing() {
    // A pendulum with joint damping = 0.4 N·m·s/rad, uniform rod. Peaks
    // (positive maxima of the swing) must decrease monotonically.
    let joint = JointKind::Hinge {
        axis: Vec3::X,
        range: None,
        damping: 0.4,
        armature: 0.0,
        limit: HingeLimit::DEFAULT,
    };
    let mut tree = build_hinge_link(Vec3::X, 1.0, 1.0, joint, 0.8);
    let dt = 0.001f32;
    let steps = 8_000usize;
    let mut peaks: Vec<f32> = Vec::new();
    let mut prev = tree.hinge_angle(1);
    let mut prev_dir = 0.0f32;
    for _ in 0..steps {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -9.81), dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 2]
        });
        let cur = tree.hinge_angle(1);
        let dir = cur - prev;
        // Peak detected when direction flips from positive to negative and
        // the current value is positive (upper turnaround). Records the
        // previous sample which was the local max.
        if prev_dir > 0.0 && dir <= 0.0 && prev > 0.0 {
            peaks.push(prev);
        }
        prev_dir = dir;
        prev = cur;
    }
    assert!(
        peaks.len() >= 3,
        "expected at least 3 positive peaks; got {}: {peaks:?}",
        peaks.len()
    );
    for window in peaks.windows(2) {
        assert!(
            window[1] < window[0],
            "peaks not strictly decreasing: {peaks:?}"
        );
    }
    // Additional check: the *first* peak is strictly less than the release
    // angle (0.8) — verifies damping fired even during the first quarter
    // period, not only across the equilibrium sweep.
    assert!(
        peaks[0] < 0.8,
        "first peak {} not below release angle 0.8 — damping likely inactive",
        peaks[0]
    );
}

#[test]
fn armature_scales_static_angular_acceleration_by_hand_ratio() {
    // Apply a fixed joint torque τ via qfrc_applied. At q=0, gravity torque
    // is zero, so all the acceleration comes from τ. Compare the initial α
    // for the same rod with and without armature.
    let l = 1.0f32;
    let m = 1.0f32;
    let armature = 0.5f32;
    let plain_joint = JointKind::Hinge {
        axis: Vec3::X,
        range: None,
        damping: 0.0,
        armature: 0.0,
        limit: HingeLimit::DEFAULT,
    };
    let armed_joint = JointKind::Hinge {
        axis: Vec3::X,
        range: None,
        damping: 0.0,
        armature,
        limit: HingeLimit::DEFAULT,
    };
    let mut plain = build_hinge_link(Vec3::X, l, m, plain_joint, 0.0);
    let mut armed = build_hinge_link(Vec3::X, l, m, armed_joint, 0.0);
    // Apply the same joint torque to both.
    let tau = 3.0f32;
    plain.qfrc_applied[plain.v_offset[1]] = tau;
    armed.qfrc_applied[armed.v_offset[1]] = tau;
    // Zero gravity so only τ matters — pure D q̈ = τ.
    let g = Vec3::ZERO;
    let poses_plain = forward_kinematics(&plain);
    let poses_armed = forward_kinematics(&armed);
    let a_plain = aba(&plain, &poses_plain, g, &vec![(Vec3::ZERO, Vec3::ZERO); 2]);
    let a_armed = aba(&armed, &poses_armed, g, &vec![(Vec3::ZERO, Vec3::ZERO); 2]);
    // Effective inertia about the pivot for a uniform rod = (1/3) m L².
    let d_plain = (1.0 / 3.0) * m * l * l;
    let alpha_plain_expected = tau / d_plain;
    let alpha_armed_expected = tau / (d_plain + armature);
    let ratio_expected = alpha_armed_expected / alpha_plain_expected;
    let ratio_measured = a_armed[0] / a_plain[0];
    assert!(
        (a_plain[0] - alpha_plain_expected).abs() < 1e-5,
        "plain hinge α mismatch: {} vs {alpha_plain_expected}",
        a_plain[0]
    );
    assert!(
        (ratio_measured - ratio_expected).abs() < 1e-5,
        "armature ratio {} vs expected {}",
        ratio_measured,
        ratio_expected
    );
}

#[test]
fn hinge_range_limit_confines_release_from_outside() {
    // Pendulum released ABOVE its lower limit; the penalty spring should
    // pull it into the allowed range and it should stay there (or oscillate
    // narrowly around the limit) — never wander back through the limit by
    // more than a small violation.
    let lo = -0.5f32;
    let hi = 0.5f32;
    let joint = JointKind::Hinge {
        axis: Vec3::X,
        range: Some((lo, hi)),
        damping: 0.05,
        armature: 0.0,
        limit: HingeLimit::new(2000.0, 40.0),
    };
    // Release ABOVE the upper limit so the spring pulls in the negative
    // direction plus gravity swings it further into the well.
    let mut tree = build_hinge_link(Vec3::X, 1.0, 1.0, joint, hi + 0.3);
    let dt = 0.001f32;
    // Run 6 s.
    for _ in 0..6_000 {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -9.81), dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 2]
        });
    }
    let final_angle = tree.hinge_angle(1);
    assert!(
        final_angle >= lo - 0.05 && final_angle <= hi + 0.05,
        "final angle {final_angle} outside limits [{lo}, {hi}]"
    );
    // Also verify no gain: peak violation over the run must decrease over
    // time. Re-run and record.
    let mut tree = build_hinge_link(
        Vec3::X,
        1.0,
        1.0,
        JointKind::Hinge {
            axis: Vec3::X,
            range: Some((lo, hi)),
            damping: 0.05,
            armature: 0.0,
            limit: HingeLimit::new(2000.0, 40.0),
        },
        hi + 0.3,
    );
    let mut window_max_early: f32 = 0.0;
    let mut window_max_late: f32 = 0.0;
    for step in 0..6_000usize {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -9.81), dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 2]
        });
        let violation_hi = (tree.hinge_angle(1) - hi).max(0.0);
        let violation_lo = (lo - tree.hinge_angle(1)).max(0.0);
        let v = violation_hi.max(violation_lo);
        if step < 1500 {
            window_max_early = window_max_early.max(v);
        }
        if step > 4500 {
            window_max_late = window_max_late.max(v);
        }
    }
    assert!(
        window_max_late <= window_max_early,
        "limit violation grew from early={window_max_early} to late={window_max_late}"
    );
}
