use newt::actuator::Actuator;
use newt::equality::Equality;
use newt::geom::{Geom, SolRef};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::sensor::{Sensor, SensorAttach, SensorKind, SiteFrame};
use newt::solver::SolImp;
use newt::tendon::{FixedTendonJoint, Tendon};
use newt::tree::{Link, Tree};
use newt::world::{World, WorldJointId};

fn link(parent: Option<usize>, joint: JointKind, x: f32) -> Link {
    Link::new(
        parent,
        joint,
        (Vec3::new(x, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    )
}

fn world_with_references() -> World {
    let mut tree = Tree::new();
    tree.push_link(link(None, JointKind::Fixed, 0.0));
    tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 1.0));
    tree.push_link(link(Some(1), JointKind::hinge(Vec3::Z), 1.0));
    tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), -1.0));
    tree.add_tendon(Tendon::fixed(vec![FixedTendonJoint { link: 2, coef: 1.0 }]));
    tree.add_actuator(Actuator::motor(3, 1.0, 0.0));
    tree.add_actuator(Actuator::motor(0, 1.0, 0.0).on_tendon(0));

    let mut world = World::new();
    world.add_tree(tree);
    world.add_geom(Geom::sphere_on_link(0, 2, 0.1, Vec3::ZERO, 0.5));
    world.add_geom(Geom::sphere_on_link(0, 3, 0.1, Vec3::ZERO, 0.5));
    world
        .add_sensor(Sensor {
            name: "parent_joint".into(),
            kind: SensorKind::JointPos { tree: 0, link: 3 },
        })
        .unwrap();
    world
        .add_sensor(Sensor {
            name: "child_joint".into(),
            kind: SensorKind::JointPos { tree: 0, link: 2 },
        })
        .unwrap();
    world
        .add_sensor(Sensor {
            name: "child_site".into(),
            kind: SensorKind::FramePos(SiteFrame {
                attach: SensorAttach::Link(0, 2),
                local_offset: Vec3::ZERO,
                local_orientation: Quat::IDENTITY,
            }),
        })
        .unwrap();
    world
        .add_sensor(Sensor {
            name: "child_tendon".into(),
            kind: SensorKind::TendonPos { tree: 0, tendon: 0 },
        })
        .unwrap();
    world
        .add_sensor(Sensor {
            name: "child_touch".into(),
            kind: SensorKind::Touch { geom: 0 },
        })
        .unwrap();
    world.equalities.push(Equality::JointCoupling {
        tree: 0,
        link_a: 2,
        link_b: 2,
        polycoef: [0.0, 1.0, 0.0],
        solref: SolRef::DEFAULT,
        solimp: SolImp::DEFAULT,
    });
    world
        .add_keyframe(
            "saved",
            vec![0.25, -0.5, 0.75],
            vec![0.1, -0.2, 0.3],
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        )
        .unwrap();
    world
}

#[test]
fn detach_remaps_all_indexed_world_state_atomically() {
    let mut world = world_with_references();
    let saved_sensor_id = 1;
    let report = world.detach_subtree(world.joint_id(0, 1)).unwrap();

    assert_eq!(
        report.remap_link(WorldJointId {
            tree_id: 0,
            link_id: 3
        }),
        Ok(WorldJointId {
            tree_id: 0,
            link_id: 1
        })
    );
    assert_eq!(
        report.remap_link(WorldJointId {
            tree_id: 0,
            link_id: 2
        }),
        Ok(WorldJointId {
            tree_id: 1,
            link_id: 1
        })
    );
    assert_eq!(report.remap_tendon(0, 0), Ok((1, 0)));
    assert_eq!(report.remap_actuator(0, 0), Ok((0, 0)));
    assert_eq!(report.remap_actuator(0, 1), Ok((1, 0)));

    assert_eq!(world.trees.len(), 2);
    assert_eq!(world.trees[0].links.len(), 2);
    assert_eq!(world.trees[1].links.len(), 2);
    assert_eq!(world.geoms[0].link, Some((1, 1)));
    assert_eq!(world.geoms[1].link, Some((0, 1)));
    assert_eq!(world.equalities[0].tree_index(), Some(1));
    assert_eq!(world.trees[1].tendons.len(), 1);
    assert_eq!(world.trees[1].actuators.len(), 1);
    assert_eq!(world.trees[0].actuators.len(), 1);

    // The sensor id and flat data layout stay stable. This used to panic at
    // sensor.rs:406 after a split removed a sensor entry without rebuilding offsets.
    assert!(world.sensor(saved_sensor_id).is_some());
    assert_eq!(
        world.sensors.sensors[1].kind,
        SensorKind::JointPos { tree: 1, link: 1 }
    );
    assert_eq!(
        world.sensors.sensors[2].kind,
        SensorKind::FramePos(SiteFrame {
            attach: SensorAttach::Link(1, 1),
            local_offset: Vec3::ZERO,
            local_orientation: Quat::IDENTITY,
        })
    );
    assert_eq!(
        world.sensors.sensors[3].kind,
        SensorKind::TendonPos { tree: 1, tendon: 0 }
    );

    // The child root changes from a hinge to a free joint, so the keyframe
    // must be rebuilt in the new dense q/qdot layout. This used to panic at
    // world.rs:514 when reset sliced the old dimensions.
    world.reset_to_keyframe("saved").unwrap();
    assert_eq!(
        world.keyframes[0].q.len(),
        world.trees.iter().map(Tree::nq).sum()
    );
    assert_eq!(
        world.keyframes[0].qdot.len(),
        world.trees.iter().map(Tree::nv).sum()
    );
    world.step();
    assert!(world.sensor(saved_sensor_id).is_some());
}

#[test]
fn cross_split_references_return_an_error_without_mutation() {
    let mut world = world_with_references();
    world.equalities.push(Equality::JointCoupling {
        tree: 0,
        link_a: 1,
        link_b: 3,
        polycoef: [0.0, 1.0, 0.0],
        solref: SolRef::DEFAULT,
        solimp: SolImp::DEFAULT,
    });
    let result = world.detach_subtree(WorldJointId {
        tree_id: 0,
        link_id: 1,
    });
    let error = result.unwrap_err();
    assert!(error.0.contains("cross the subtree split"));
    assert_eq!(world.trees.len(), 1);
    assert_eq!(world.sensors.sensors.len(), 5);
}
