//! Analytic forward-dynamics derivative anchors.
//!
//! These tests use a one-link pendulum so each expected derivative comes
//! from the closed-form torque balance, not from a finite difference of newt.

use newt::actuator::Actuator;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree};

fn pendulum(damping: f32) -> Tree {
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
            range: None,
            damping,
            armature: 0.0,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.7,
        Mat3::diag(0.02, 0.03, 0.04),
    ));
    tree
}

fn zero_wrenches(tree: &Tree) -> Vec<(Vec3, Vec3)> {
    vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()]
}

#[test]
fn pendulum_position_derivative_matches_closed_form_torque_balance() {
    let mut tree = pendulum(0.0);
    let angle = 0.31;
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    tree.set_hinge_angle(1, angle);

    let derivatives = tree.derivatives(gravity, &zero_wrenches(&tree));
    let mass_about_hinge = 0.02 + 1.7;
    let expected = -1.7 * 9.81 * angle.cos() / mass_about_hinge;

    assert_eq!(derivatives.qacc_q.len(), 1);
    assert!(
        (derivatives.qacc_q[0] - expected).abs() < 2.0e-3,
        "analytic pendulum derivative {} != hand value {}",
        derivatives.qacc_q[0],
        expected
    );
}

#[test]
fn damping_derivative_comes_from_rne_velocity_bias() {
    let damping = 0.73;
    let mut tree = pendulum(damping);
    tree.set_hinge_rate(1, 0.4);
    let derivatives = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    let mass_about_hinge = 0.02 + 1.7;
    let expected = -damping / mass_about_hinge;

    assert_eq!(derivatives.qacc_qvel.len(), 1);
    assert!(
        (derivatives.qacc_qvel[0] - expected).abs() < 2.0e-5,
        "qvel derivative {} != hand value {}",
        derivatives.qacc_qvel[0],
        expected
    );
}

#[test]
fn motor_control_derivative_uses_the_actuator_gain_path() {
    let mut tree = pendulum(0.0);
    let gear = 2.4;
    tree.add_actuator(Actuator::motor(1, gear, 0.0));
    let derivatives = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    let mass_about_hinge = 0.02 + 1.7;
    let expected = gear / mass_about_hinge;

    assert_eq!(derivatives.qacc_ctrl.len(), 1);
    assert!(
        (derivatives.qacc_ctrl[0] - expected).abs() < 2.0e-5,
        "ctrl derivative {} != hand value {}",
        derivatives.qacc_ctrl[0],
        expected
    );
}
