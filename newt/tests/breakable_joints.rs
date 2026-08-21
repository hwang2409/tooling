use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::SolverMode;
use newt::tree::{Link, Tree};
use newt::world::World;

fn world_with_threshold(threshold: Option<f32>) -> World {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(
        Link::new(
            Some(0),
            JointKind::hinge(Vec3::X),
            (Vec3::new(0.5, 0.0, 0.0), Quat::IDENTITY),
            (Vec3::new(-0.5, 0.0, 0.0), Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        )
        .with_break_threshold(threshold),
    );
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.add_tree(tree);
    world
}

#[test]
fn breakable_joint_no_threshold_never_breaks() {
    let mut world = world_with_threshold(None);
    world.trees[0].set_joint_torque_clamped(1, 10_000.0, 0.0);
    for _ in 0..100 {
        world.step();
    }
    assert!(!world.trees[0].links[1].is_broken());
}

#[test]
fn breakable_joint_below_threshold_holds() {
    let mut world = world_with_threshold(Some(100.0));
    world.trees[0].set_joint_torque_clamped(1, 10.0, 0.0);
    for _ in 0..100 {
        world.step();
    }
    assert!(!world.trees[0].links[1].is_broken());
}

#[test]
fn breakable_joint_above_threshold_breaks_next_step() {
    let mut world = world_with_threshold(Some(0.01));
    world.trees[0].set_joint_torque_clamped(1, 10.0, 0.0);
    world.trees[0].set_link_wrench(1, Vec3::new(20.0, 0.0, 0.0), Vec3::ZERO);
    world.step();
    assert!(world.trees[0].links[1].is_broken());
    let joint_id = world.joint_id(0, 1);
    assert_eq!(world.broken_this_step(), &[joint_id]);

    world.trees[0].set_joint_torque_clamped(1, 0.0, 0.0);
    let parent_before = world.trees[0].link_pose(0).0;
    let child_before = world.trees[0].link_pose(1).0;
    world.step();
    let parent_after = world.trees[0].link_pose(0).0;
    let child_after = world.trees[1].link_pose(0).0;
    let parent_velocity = Vec3::new(
        world.trees[0].qdot[3],
        world.trees[0].qdot[4],
        world.trees[0].qdot[5],
    );
    let child_velocity = Vec3::new(
        world.trees[1].qdot[3],
        world.trees[1].qdot[4],
        world.trees[1].qdot[5],
    );
    let relative_motion = (child_after - child_before) - (parent_after - parent_before);
    assert!(
        (child_after - child_before).length() > 1.0e-5,
        "detached child did not move: before={child_before:?} after={child_after:?}"
    );
    assert!(
        child_velocity.length() > 1.0e-5,
        "detached child has no independent velocity: {child_velocity:?}"
    );
    assert!(
        parent_velocity.length() > 1.0e-5,
        "parent body did not move: {parent_velocity:?}"
    );
    assert!(
        (child_velocity - parent_velocity).length() > 1.0e-5,
        "child and parent velocities stayed coupled: parent={parent_velocity:?} child={child_velocity:?}"
    );
    assert!(
        relative_motion.length() > 1.0e-5,
        "child and parent positions stayed coupled: parent={parent_after:?} child={child_after:?}"
    );
    assert!(world.broken_this_step().is_empty());
}

#[test]
fn breakable_joint_broken_this_step_reported() {
    let mut world = world_with_threshold(Some(0.01));
    world.trees[0].set_joint_torque_clamped(1, 10.0, 0.0);
    world.step();
    assert_eq!(world.broken_this_step(), &[world.joint_id(0, 1)]);
    world.step();
    assert!(world.broken_this_step().is_empty());
}

#[test]
fn breakable_joint_manual_break() {
    let mut world = world_with_threshold(None);
    let joint_id = world.joint_id(0, 1);
    world.break_joint(joint_id);
    assert!(world.broken_this_step().is_empty());
    world.step();
    assert!(world.trees[0].links[1].is_broken());
    assert_eq!(world.broken_this_step(), &[joint_id]);
}

#[test]
fn breakable_joint_reset_restores() {
    let mut world = world_with_threshold(None);
    world.solver.mode = SolverMode::Pgs;
    world.trees[0].links[1].joint = JointKind::Hinge {
        axis: Vec3::X,
        range: Some((-0.1, 0.1)),
        damping: 0.0,
        armature: 0.0,
        limit: newt::joint::JointLimit::DEFAULT,
    };
    world.trees[0].set_hinge_angle(1, 1.0);
    let joint_id = world.joint_id(0, 1);
    world.break_joint(joint_id);
    world.step();

    world.reset_joint(joint_id);
    world.trees[0].set_hinge_angle(1, 1.0);
    world.trees[0].set_hinge_rate(1, 0.0);
    world.step();
    let reset_rate = world.trees[0].hinge_rate(1);
    assert!(!world.trees[0].links[1].is_broken());
    assert_ne!(reset_rate, 0.0);
}
