//! Analytic forward-dynamics derivative anchors.
//!
//! Hand-derived anchors and saved MuJoCo transitionFD data cover the smooth
//! derivative paths. No expected value comes from a Newt finite difference.

use newt::actuator::Actuator;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tendon::{
    SpatialSegment, SpatialTendonBranch, SpatialTendonSite, SpatialWrap, Tendon, WrapCylinder,
    WrapSphere,
};
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

fn fixture_values(fixture: &str) -> [f32; 4] {
    let qacc = json_array(fixture, "qacc")[0];
    let timestep = json_number(fixture, "timestep");
    let a = json_array(fixture, "A");
    let b = json_array(fixture, "B");
    [
        qacc,
        a[2] / timestep,
        (a[3] - 1.0) / timestep,
        b[1] / timestep,
    ]
}

fn json_number(fixture: &str, key: &str) -> f32 {
    let start = fixture.find(&format!("\"{key}\"")).expect("fixture key") + key.len() + 2;
    fixture[start..]
        .split(|ch: char| {
            !ch.is_ascii_digit() && ch != '.' && ch != '-' && ch != '+' && ch != 'e' && ch != 'E'
        })
        .find(|value| !value.is_empty())
        .expect("fixture number")
        .parse()
        .expect("fixture value is a float")
}

fn json_array(fixture: &str, key: &str) -> Vec<f32> {
    let key_start = fixture.find(&format!("\"{key}\"")).expect("fixture key");
    let array_start = fixture[key_start..].find('[').expect("fixture array") + key_start;
    let mut depth = 0;
    let mut array_end = array_start;
    for (offset, ch) in fixture[array_start..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    array_end = array_start + offset;
                    break;
                }
            }
            _ => {}
        }
    }
    fixture[array_start + 1..array_end]
        .split(|ch: char| ch == '[' || ch == ']' || ch == ',' || ch.is_ascii_whitespace())
        .filter(|value| !value.is_empty())
        .map(|value| value.parse().expect("fixture array value is a float"))
        .collect()
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
fn sibling_ball_joint_does_not_break_tendon_derivatives() {
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
            damping: 0.0,
            armature: 0.0,
            limit: newt::joint::JointLimit::DEFAULT,
        },
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
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.add_tendon(Tendon::spatial(
        vec![
            SpatialTendonSite {
                link: None,
                position_local: Vec3::new(-1.0, 0.0, 0.0),
            },
            SpatialTendonSite {
                link: Some(1),
                position_local: Vec3::new(1.0, 0.0, 0.0),
            },
        ],
        vec![None],
    ));

    let derivatives = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert_eq!(derivatives.qacc_q.len(), 16);
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

#[test]
fn pendulum_matches_saved_mujoco_transition_fd_fixture() {
    let expected = fixture_values(include_str!("fixtures/mujoco_pendulum_transition_fd.json"));
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
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, -1.0), Quat::IDENTITY),
        1.7,
        Mat3::diag(0.02, 0.03, 0.04),
    ));
    tree.set_hinge_angle(1, 0.31);
    tree.set_hinge_rate(1, 0.4);
    tree.add_actuator(Actuator::motor(1, 2.4, 0.0));
    tree.set_actuator_target(0, 0.2);

    let actual = tree.derivatives(Vec3::new(0.0, 0.0, -9.81), &zero_wrenches(&tree));
    assert!((actual.qacc[0] - expected[0]).abs() < 2.0e-5);
    assert!((actual.qacc_q[0] - expected[1]).abs() < 2.0e-5);
    assert!((actual.qacc_qvel[0] - expected[2]).abs() < 2.0e-5);
    assert!((actual.qacc_ctrl[0] - expected[3]).abs() < 2.0e-5);
}

#[test]
fn slide_matches_saved_mujoco_transition_fd_fixture() {
    let expected = fixture_values(include_str!("fixtures/mujoco_slide_transition_fd.json"));
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
    tree.set_slide_position(1, 0.17);
    tree.set_slide_rate(1, 0.2);
    tree.add_actuator(Actuator::motor(1, 1.3, 0.0));
    tree.set_actuator_target(0, 0.2);

    let actual = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert!((actual.qacc[0] - expected[0]).abs() < 2.0e-5);
    assert!((actual.qacc_q[0] - expected[1]).abs() < 2.0e-5);
    // MuJoCo's Euler damping regularization shifts these two entries by
    // about 1e-3 from Newt's explicit damping path.
    assert!((actual.qacc_qvel[0] - expected[2]).abs() < 2.0e-3);
    assert!((actual.qacc_ctrl[0] - expected[3]).abs() < 2.0e-3);
}

#[test]
fn spatial_tendon_position_derivative_matches_hand_spring() {
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
        JointKind::slide(Vec3::Z),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
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
    tendon.springlength = Some(0.4);
    tendon.stiffness = 100.0;
    tree.add_tendon(tendon);
    tree.set_slide_position(1, 0.5);

    let actual = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert!((actual.qacc_q[0] + 100.0).abs() < 2.0e-4);
}

fn wrapped_slide_tree(wrap: SpatialWrap, springlength: f32) -> Tree {
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
        JointKind::slide(Vec3::Z),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    let mut tendon = Tendon::spatial_branches(vec![SpatialTendonBranch {
        sites: vec![
            SpatialTendonSite {
                link: None,
                position_local: Vec3::new(-2.0, 0.2, 0.0),
            },
            SpatialTendonSite {
                link: None,
                position_local: Vec3::new(2.0, 0.2, 0.0),
            },
        ],
        segments: vec![SpatialSegment { wrap: Some(wrap) }],
        divisor: 1.0,
    }]);
    tendon.springlength = Some(springlength);
    tendon.stiffness = 1.0;
    tree.add_tendon(tendon);
    tree
}

fn symmetric_wrap_length() -> f32 {
    let d_squared = 4.04;
    let tangent = (d_squared - 0.25).sqrt();
    let cosine = -3.96 / d_squared;
    let gamma = newt::math::atan2((1.0 - cosine * cosine).sqrt(), cosine);
    let phi = (tangent / d_squared.sqrt()).asin();
    2.0 * tangent + 0.5 * (gamma - 2.0 * phi)
}

#[test]
fn cylinder_wrap_position_derivative_matches_hand_second_derivative() {
    let length = symmetric_wrap_length();
    let wrap = SpatialWrap::Cylinder(WrapCylinder {
        link: Some(1),
        center_local: Vec3::ZERO,
        axis_local: Vec3::Z,
        radius: 0.5,
        sidesite: None,
    });
    let tree = wrapped_slide_tree(wrap, length - 1.0);
    let actual = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));

    // A symmetric axial lift has L(q)'' = 1/L(0). The spring force is -1,
    // and the slide mass is 1, so qacc_q = -1/L(0).
    let expected = -1.0 / length;
    assert!((actual.qacc_q[0] - expected).abs() < 2.0e-3);
}

#[test]
fn sphere_wrap_position_derivative_matches_hand_second_derivative() {
    let length = symmetric_wrap_length();
    let wrap = SpatialWrap::Sphere(WrapSphere {
        link: Some(1),
        center_local: Vec3::ZERO,
        radius: 0.5,
        side_hint_world: None,
    });
    let tree = wrapped_slide_tree(wrap, length - 1.0);
    let actual = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));

    let d_squared: f32 = 4.04;
    let tangent = (d_squared - 0.25).sqrt();
    let cosine = -3.96 / d_squared;
    let sine = (1.0 - cosine * cosine).sqrt();
    let gamma_second = -16.0 / (d_squared * d_squared * sine);
    let phi_second = 0.5 / (d_squared * tangent);
    let length_second = 2.0 / tangent + 0.5 * (gamma_second - 2.0 * phi_second);
    let expected = -length_second;
    assert!((actual.qacc_q[0] - expected).abs() < 2.0e-3);
}

#[test]
fn muscle_position_derivative_matches_hand_force_length_curve() {
    let mut tree = pendulum(0.0);
    tree.links[1].joint = JointKind::hinge(Vec3::X);
    tree.links[1].joint_offset_in_child = (Vec3::ZERO, Quat::IDENTITY);
    let params = [0.75, 1.05, 100.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2];
    let actuator = tree.add_actuator(Actuator::muscle(
        1,
        params,
        params,
        [0.0, 1.0],
        1.0,
        1.0,
        [0.01, 0.04, 0.0],
        None,
        None,
    ));
    tree.actuators[actuator].act = 1.0;
    tree.set_hinge_angle(1, 0.8);

    // l = 0.75 + 0.3q = 0.99, fl' = (1-l)/(1-0.75)^2 = 0.16.
    // The active force derivative is -100 * 0.16 * 0.3 = -4.8.
    let actual = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert!((actual.qacc_q[0] + 4.8 / 0.02).abs() < 2.0e-3);
}

#[test]
fn muscle_position_derivative_uses_distinct_bias_parameters() {
    let mut tree = pendulum(0.0);
    tree.links[1].joint = JointKind::hinge(Vec3::X);
    tree.links[1].joint_offset_in_child = (Vec3::ZERO, Quat::IDENTITY);
    let gain = [0.75, 1.05, 100.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2];
    let bias = [0.9, 1.1, 50.0, 100.0, 0.5, 1.6, 1.5, 2.0, 1.2];
    let actuator = tree.add_actuator(Actuator::muscle(
        1,
        gain,
        bias,
        [0.0, 1.0],
        1.0,
        1.0,
        [0.01, 0.04, 0.0],
        None,
        None,
    ));
    tree.actuators[actuator].act = 1.0;
    tree.set_hinge_angle(1, 0.8);

    // Gain contributes -4.8. Bias has l=1.06, b=1.3, and contributes
    // -50*2*(0.06/0.3)*0.2/0.3 = -13.333333.
    let expected = -(4.8 + 13.333333) / 0.02;
    let actual = tree.derivatives(Vec3::ZERO, &zero_wrenches(&tree));
    assert!((actual.qacc_q[0] - expected).abs() < 2.0e-3);
}

#[test]
fn rotated_fixed_child_wrench_position_derivative_matches_hand_torque() {
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
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    tree.push_link(Link::new(
        Some(1),
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(0.1, 0.1, 0.1),
    ));
    tree.links[2].joint_offset_in_parent.1 =
        Quat::from_axis_angle(Vec3::X, core::f32::consts::FRAC_PI_2);
    let angle = 0.4;
    tree.set_hinge_angle(1, angle);
    let mut external = zero_wrenches(&tree);
    external[2].0 = Vec3::new(0.0, 0.0, 3.0);

    let actual = tree.derivatives(Vec3::ZERO, &external);
    let expected = -3.0 * angle.sin() / 2.1;
    assert!((actual.qacc_q[0] - expected).abs() < 2.0e-4);
}
