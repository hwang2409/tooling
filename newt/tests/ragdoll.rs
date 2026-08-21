use newt::joint::JointKind;
use newt::math::Vec3;
use newt::ragdoll::{BoneSpec, RagdollBuildError, RagdollBuilder, RagdollJointSpec};
use newt::solver::{SolverConfig, SolverMode};
use newt::tree::forward_kinematics;
use newt::world::World;

fn bone(name: &str, parent: Option<&str>, joint: RagdollJointSpec) -> BoneSpec {
    BoneSpec {
        name: name.into(),
        parent: parent.map(str::to_owned),
        length: 0.5,
        half_extents: Vec3::new(0.1, 0.25, 0.1),
        mass: 1.0,
        joint,
    }
}

fn fixed_bone(name: &str, parent: Option<&str>) -> BoneSpec {
    bone(name, parent, RagdollJointSpec::Fixed)
}

fn hinge_bone(name: &str, parent: Option<&str>) -> BoneSpec {
    bone(
        name,
        parent,
        RagdollJointSpec::Hinge {
            axis: Vec3::Z,
            limit: None,
        },
    )
}

fn x_hinge_bone(name: &str, parent: Option<&str>) -> BoneSpec {
    bone(
        name,
        parent,
        RagdollJointSpec::Hinge {
            axis: Vec3::X,
            limit: None,
        },
    )
}

#[test]
fn ragdoll_smoke_biped() {
    let mut builder = RagdollBuilder::new();
    builder
        .add_bone(fixed_bone("pelvis", None))
        .add_bone(fixed_bone("torso", Some("pelvis")))
        .add_bone(fixed_bone("head", Some("torso")))
        .add_bone(hinge_bone("upper_arm_l", Some("torso")))
        .add_bone(hinge_bone("upper_arm_r", Some("torso")))
        .add_bone(hinge_bone("forearm_l", Some("upper_arm_l")))
        .add_bone(hinge_bone("forearm_r", Some("upper_arm_r")))
        .add_bone(hinge_bone("upper_leg_l", Some("pelvis")))
        .add_bone(hinge_bone("upper_leg_r", Some("pelvis")))
        .add_bone(hinge_bone("lower_leg_l", Some("upper_leg_l")))
        .add_bone(hinge_bone("lower_leg_r", Some("upper_leg_r")))
        .add_bone(hinge_bone("foot_l", Some("lower_leg_l")));

    let mut world = World::new();
    let handles = builder.build(&mut world).unwrap();
    assert_eq!(handles.bodies.len(), 12);
    assert_eq!(handles.joints.len(), 11);
    assert_eq!(world.trees.len(), 1);
    assert_eq!(world.geoms.len(), 12);

    let tree = &world.trees[0];
    for (child_name, parent_name) in [
        ("torso", "pelvis"),
        ("head", "torso"),
        ("upper_arm_l", "torso"),
        ("upper_arm_r", "torso"),
        ("forearm_l", "upper_arm_l"),
        ("forearm_r", "upper_arm_r"),
        ("upper_leg_l", "pelvis"),
        ("upper_leg_r", "pelvis"),
        ("lower_leg_l", "upper_leg_l"),
        ("lower_leg_r", "upper_leg_r"),
        ("foot_l", "lower_leg_l"),
    ] {
        let child_id = handles.by_name[child_name];
        let parent_id = handles.by_name[parent_name];
        assert_eq!(tree.links[child_id].parent, Some(parent_id));
        assert!(handles.joints.contains(&child_id));
    }
}

#[test]
fn ragdoll_lookup_by_name() {
    let mut builder = RagdollBuilder::new();
    builder
        .add_bone(fixed_bone("root", None))
        .add_bone(hinge_bone("middle", Some("root")))
        .add_bone(hinge_bone("tip", Some("middle")));
    let mut world = World::new();
    let handles = builder.build(&mut world).unwrap();

    assert_eq!(handles.by_name["root"], handles.bodies[0]);
    assert_eq!(handles.by_name["middle"], handles.bodies[1]);
    assert_eq!(handles.by_name["tip"], handles.bodies[2]);
}

#[test]
fn ragdoll_gravity_drop_stable() {
    let mut builder = RagdollBuilder::new();
    builder
        .pin_root()
        .add_bone(fixed_bone("root", None))
        .add_bone(x_hinge_bone("link", Some("root")))
        .add_bone(x_hinge_bone("tip", Some("link")));
    let mut world = World::new();
    let handles = builder.build(&mut world).unwrap();

    let initial_angle = world.trees[0].hinge_angle(handles.by_name["link"]);
    for _ in 0..100 {
        world.step();
    }
    let tree = &world.trees[0];
    for (position, _) in forward_kinematics(tree) {
        assert!(position.x.is_finite());
        assert!(position.y.is_finite());
        assert!(position.z.is_finite());
        assert!(position.length() < 100.0);
    }
    let final_angle = tree.hinge_angle(handles.by_name["link"]);
    assert!((final_angle - initial_angle).abs() > 1e-4);
    for link_id in handles.bodies {
        let offset = tree.v_offset[link_id];
        let nv = tree.links[link_id].joint.nv();
        assert!(
            tree.qdot[offset..offset + nv]
                .iter()
                .all(|v| v.abs() < 100.0)
        );
    }
}

#[test]
fn ragdoll_hinge_limit_respected() {
    let mut builder = RagdollBuilder::new();
    builder
        .pin_root()
        .add_bone(fixed_bone("root", None))
        .add_bone(bone(
            "arm",
            Some("root"),
            RagdollJointSpec::Hinge {
                axis: Vec3::Z,
                limit: Some((0.0, core::f32::consts::FRAC_PI_2)),
            },
        ));
    let mut world = World::new();
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        ..SolverConfig::DEFAULT
    };
    let handles = builder.build(&mut world).unwrap();
    let arm = handles.by_name["arm"];
    {
        let tree = &mut world.trees[0];
        assert!(matches!(
            tree.links[arm].joint,
            JointKind::Hinge {
                range: Some((0.0, high)),
                ..
            } if (high - core::f32::consts::FRAC_PI_2).abs() < 1e-6
        ));
        tree.qfrc_applied[tree.v_offset[arm]] = 12.0;
    }
    for _ in 0..200 {
        world.step();
    }
    assert!(
        world.trees[0].hinge_angle(arm) <= core::f32::consts::FRAC_PI_2 + 0.2,
        "angle was {}",
        world.trees[0].hinge_angle(arm)
    );
}

#[test]
fn ragdoll_missing_parent_errors() {
    let mut builder = RagdollBuilder::new();
    builder.add_bone(fixed_bone("child", Some("nonexistent")));
    let mut world = World::new();
    assert_eq!(
        builder.build(&mut world),
        Err(RagdollBuildError::UnknownParent("nonexistent".into()))
    );
}

#[test]
fn ragdoll_cycle_errors() {
    let mut builder = RagdollBuilder::new();
    builder
        .add_bone(fixed_bone("a", Some("b")))
        .add_bone(fixed_bone("b", Some("a")));
    let mut world = World::new();
    assert_eq!(builder.build(&mut world), Err(RagdollBuildError::Cycle));
}

#[test]
fn ragdoll_root_hinge_errors() {
    let mut builder = RagdollBuilder::new();
    builder.add_bone(hinge_bone("root", None));
    let mut world = World::new();
    assert_eq!(
        builder.build(&mut world),
        Err(RagdollBuildError::UnsupportedRootJoint("root".into()))
    );
}

#[test]
fn ragdoll_root_ball_limit_errors() {
    let mut builder = RagdollBuilder::new();
    builder.add_bone(bone(
        "root",
        None,
        RagdollJointSpec::Ball {
            swing_limit: Some(0.5),
            twist_limit: None,
        },
    ));
    let mut world = World::new();
    assert_eq!(
        builder.build(&mut world),
        Err(RagdollBuildError::UnsupportedRootJoint("root".into()))
    );
}

#[test]
fn ragdoll_root_six_dof_errors() {
    let mut builder = RagdollBuilder::new();
    builder.add_bone(bone(
        "root",
        None,
        RagdollJointSpec::SixDof {
            linear_limits: [None, None, None],
            angular_limits: [None, None, None],
        },
    ));
    let mut world = World::new();
    assert_eq!(
        builder.build(&mut world),
        Err(RagdollBuildError::UnsupportedRootJoint("root".into()))
    );
}

#[test]
fn ragdoll_invalid_hinge_limits_error() {
    for limit in [(1.0, 1.0), (f32::INFINITY, 1.0), (f32::NAN, 1.0)] {
        let mut builder = RagdollBuilder::new();
        builder.add_bone(fixed_bone("root", None)).add_bone(bone(
            "hinge",
            Some("root"),
            RagdollJointSpec::Hinge {
                axis: Vec3::X,
                limit: Some(limit),
            },
        ));
        let mut world = World::new();
        assert!(matches!(
            builder.build(&mut world),
            Err(RagdollBuildError::InvalidJointLimit { bone, .. }) if bone == "hinge"
        ));
    }
}

#[test]
fn ragdoll_invalid_ball_limits_error() {
    for (swing_limit, twist_limit) in [
        (Some(0.0), None),
        (Some(f32::INFINITY), None),
        (Some(f32::NAN), None),
        (None, Some(-1.0)),
    ] {
        let mut builder = RagdollBuilder::new();
        builder.add_bone(fixed_bone("root", None)).add_bone(bone(
            "ball",
            Some("root"),
            RagdollJointSpec::Ball {
                swing_limit,
                twist_limit,
            },
        ));
        let mut world = World::new();
        assert!(matches!(
            builder.build(&mut world),
            Err(RagdollBuildError::InvalidJointLimit { bone, .. }) if bone == "ball"
        ));
    }
}

#[test]
fn ragdoll_invalid_six_dof_limits_error() {
    for (linear_limits, angular_limits) in [
        ([Some((1.0, 1.0)), None, None], [None, None, None]),
        ([Some((f32::INFINITY, 1.0)), None, None], [None, None, None]),
        ([None, None, None], [Some((f32::NAN, 1.0)), None, None]),
    ] {
        let mut builder = RagdollBuilder::new();
        builder.add_bone(fixed_bone("root", None)).add_bone(bone(
            "six",
            Some("root"),
            RagdollJointSpec::SixDof {
                linear_limits,
                angular_limits,
            },
        ));
        let mut world = World::new();
        assert!(matches!(
            builder.build(&mut world),
            Err(RagdollBuildError::InvalidJointLimit { bone, .. }) if bone == "six"
        ));
    }
}
