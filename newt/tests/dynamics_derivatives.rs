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

#[test]
fn free_root_uses_six_tangent_columns() {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        2.0,
        Mat3::diag(1.0, 1.2, 1.4),
    ));
    let orientation = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.4);
    tree.set_free_root_pose(Vec3::ZERO, orientation);
    let force = Vec3::new(3.0, -2.0, 5.0);
    let ext = vec![(force, Vec3::ZERO)];
    let derivatives = tree.derivatives(Vec3::ZERO, &ext);
    assert_eq!(derivatives.qacc_q.len(), 36);

    let force_body = orientation.inverse_rotate(force) / 2.0;
    for column in 0..3 {
        for row in 0..6 {
            assert!(derivatives.qacc_q[row * 6 + column].abs() < 1.0e-6);
        }
    }
    for column in 0..3 {
        let expected = -[Vec3::X, Vec3::Y, Vec3::Z][column].cross(force_body);
        assert!((derivatives.qacc_q[3 * 6 + 3 + column] - expected.x).abs() < 2.0e-5);
        assert!((derivatives.qacc_q[4 * 6 + 3 + column] - expected.y).abs() < 2.0e-5);
        assert!((derivatives.qacc_q[5 * 6 + 3 + column] - expected.z).abs() < 2.0e-5);
    }
}

#[test]
fn ball_position_derivative_has_three_tangent_columns() {
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
        JointKind::Ball {
            damping: 0.0,
            armature: 0.0,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(0.2, 0.3, 0.4),
    ));
    tree.set_ball_orientation(1, Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.4));
    let derivatives = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert_eq!(derivatives.qacc_q.len(), 9);
    assert!(derivatives.qacc_q.iter().all(|value| value.abs() < 1.0e-6));
}

#[test]
fn saturated_velocity_actuator_has_zero_velocity_derivative() {
    let mut tree = pendulum(0.0);
    tree.add_actuator(Actuator::velocity(1, 10.0, 1.0));
    tree.set_hinge_rate(1, -1.0);
    let derivatives = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert!(derivatives.qacc_qvel[0].abs() < 1.0e-6);
}

#[test]
fn limit_position_derivative_uses_the_active_spring() {
    let mut tree = pendulum(0.0);
    tree.links[1].joint = JointKind::Hinge {
        axis: Vec3::X,
        range: Some((0.0, 1.0)),
        damping: 0.0,
        armature: 0.0,
        limit: newt::joint::JointLimit::new(20.0, 0.0),
    };
    tree.set_hinge_angle(1, 1.2);
    let derivatives = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    let mass_about_hinge = 0.02 + 1.7;
    assert!((derivatives.qacc_q[0] + 20.0 / mass_about_hinge).abs() < 2.0e-5);
}

#[test]
fn mixed_slide_and_hinge_velocity_anchor_is_hand_derived() {
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
        JointKind::Slide {
            axis: Vec3::X,
            range: None,
            damping: 0.6,
            armature: 0.0,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        2.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 0.9,
            armature: 0.0,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.7,
        Mat3::diag(0.02, 0.03, 0.04),
    ));
    tree.set_slide_rate(1, 0.0);
    tree.set_hinge_rate(2, 0.0);
    let derivatives = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert!((derivatives.qacc_qvel[0] + 0.6 / 2.0).abs() < 2.0e-5);
    assert!((derivatives.qacc_qvel[3] + 0.9 / 1.72).abs() < 2.0e-5);
    assert!(derivatives.qacc_qvel[1].abs() < 1.0e-6);
    assert!(derivatives.qacc_qvel[2].abs() < 1.0e-6);
}
