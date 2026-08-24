use newt::actuator::Actuator;
use newt::equality::Equality;
use newt::geom::{Geom, SolRef};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::model::{Scene, Site, SiteAttach};
use newt::sensor::{Sensor, SensorAttach, SensorKind, SiteFrame};
use newt::solver::SolImp;
use newt::tendon::{FixedTendonJoint, Tendon};
use newt::tree::{Link, Tree};
use newt::world::World;
use std::collections::HashMap;

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

fn world_with_handle_tree() -> World {
    let mut tree = Tree::new();
    tree.push_link(link(None, JointKind::Fixed, 0.0));
    tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 1.0));
    tree.push_link(link(Some(1), JointKind::hinge(Vec3::Z), 1.0));
    tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), -1.0));

    let mut world = World::new();
    world.add_tree(tree);
    world
}

fn world_with_mocap_handle_tree() -> World {
    let mut tree = Tree::new();
    tree.push_link(link(None, JointKind::Fixed, 0.0));
    tree.set_mocap(0, true);
    tree.push_link(link(Some(0), JointKind::Fixed, 1.0));

    let mut world = World::new();
    world.add_tree(tree);
    world
}

#[test]
fn detach_remaps_all_indexed_world_state_atomically() {
    let mut world = world_with_references();
    let parent_sibling = world.joint_id(0, 3);
    let detached_child = world.joint_id(0, 2);
    let saved_sensor_id = 1;
    let report = world.detach_subtree(world.joint_id(0, 1)).unwrap();

    assert_eq!(report.remap_link(parent_sibling), Ok(world.joint_id(0, 1)));
    assert_eq!(report.remap_link(detached_child), Ok(world.joint_id(1, 1)));
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
    let result = world.detach_subtree(world.joint_id(0, 1));
    let error = result.unwrap_err();
    assert!(error.0.contains("cross the subtree split"));
    assert_eq!(world.trees.len(), 1);
    assert_eq!(world.sensors.sensors.len(), 5);
}

#[test]
fn keyframes_append_detached_state_after_later_trees() {
    let mut source = Tree::new();
    source.push_link(link(None, JointKind::Fixed, 0.0));
    source.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 1.0));

    let mut later = Tree::new();
    later.push_link(link(None, JointKind::Fixed, 0.0));
    later.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 2.0));

    let mut world = World::new();
    world.add_tree(source);
    world.add_tree(later);
    world
        .add_keyframe("saved", vec![1.25, 2.5], vec![3.5, 4.5], vec![], vec![])
        .unwrap();

    world.detach_subtree(world.joint_id(0, 1)).unwrap();

    assert_eq!(world.keyframes[0].q[0], 2.5);
    assert_eq!(world.keyframes[0].qdot[0], 4.5);
    assert_eq!(world.keyframes[0].q.len(), 8);
    assert_eq!(world.keyframes[0].qdot.len(), 7);
}

#[test]
fn detached_root_joint_consumers_are_explicitly_invalidated() {
    let mut world = world_with_references();
    world
        .add_sensor(Sensor {
            name: "detached_root".into(),
            kind: SensorKind::JointPos { tree: 0, link: 1 },
        })
        .unwrap();

    let result = world.detach_subtree(world.joint_id(0, 1));
    assert!(result.unwrap_err().0.contains("detached root joint"));
    assert_eq!(world.trees.len(), 1);
    assert_eq!(world.sensors.sensors.len(), 6);

    let mut world = world_with_references();
    world.equalities.push(Equality::JointCoupling {
        tree: 0,
        link_a: 1,
        link_b: 2,
        polycoef: [0.0, 1.0, 0.0],
        solref: SolRef::DEFAULT,
        solimp: SolImp::DEFAULT,
    });
    let result = world.detach_subtree(world.joint_id(0, 1));
    assert!(result.unwrap_err().0.contains("detached root joint"));
    assert_eq!(world.trees.len(), 1);
}

#[test]
fn direct_tree_population_has_atomic_workspace_setup() {
    let mut world = World::new();
    let mut tree = Tree::new();
    tree.push_link(link(None, JointKind::Fixed, 0.0));
    tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 1.0));
    world.trees.push(tree);

    let link = world.joint_id(0, 1);
    world.detach_subtree(link).unwrap();
    assert_eq!(world.trees.len(), 2);
    assert_eq!(world.trees[0].links.len(), 1);
    assert_eq!(world.trees[1].links.len(), 1);
}

#[test]
fn scene_detach_remaps_sites_and_name_maps_atomically() {
    let mut world = World::new();
    let mut tree = Tree::new();
    tree.push_link(link(None, JointKind::Fixed, 0.0));
    tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 1.0));
    world.add_tree(tree);
    let mut other_tree = Tree::new();
    other_tree.push_link(link(None, JointKind::Fixed, 5.0));
    world.add_tree(other_tree);

    let mut links = HashMap::new();
    links.insert("root".into(), 0);
    links.insert("arm".into(), 1);
    let mut other_links = HashMap::new();
    other_links.insert("other_root".into(), 0);
    let mut trees_by_name = HashMap::new();
    trees_by_name.insert("body".into(), 0);
    trees_by_name.insert("other_body".into(), 1);
    let mut sites_by_name = HashMap::new();
    sites_by_name.insert("tip".into(), 0);
    sites_by_name.insert("other_tip".into(), 1);
    let mut scene = Scene {
        world,
        bodies_by_name: HashMap::new(),
        trees_by_name,
        links_by_name: vec![links, other_links],
        geoms_by_name: HashMap::new(),
        sites: vec![
            Site {
                name: "tip".into(),
                attach: SiteAttach::Link { tree: 0, link: 1 },
                local_offset: Vec3::ZERO,
                local_orientation: Quat::IDENTITY,
            },
            Site {
                name: "other_tip".into(),
                attach: SiteAttach::Link { tree: 1, link: 0 },
                local_offset: Vec3::ZERO,
                local_orientation: Quat::IDENTITY,
            },
        ],
        sites_by_name,
        actuators_by_name: HashMap::new(),
        sensors_by_name: HashMap::new(),
        tendons_by_name: HashMap::new(),
    };

    scene.detach_subtree(scene.world.joint_id(0, 1)).unwrap();
    assert_eq!(scene.sites[0].attach, SiteAttach::Link { tree: 2, link: 0 });
    assert_eq!(scene.links_by_name[2]["arm"], 0);
    assert_eq!(scene.links_by_name[1]["other_root"], 0);
    assert!(scene.site_pose("tip").is_some());
    assert!(scene.site_pose("other_tip").is_some());
}

#[test]
fn cloned_world_rejects_source_handle() {
    let source = world_with_handle_tree();
    let source_handle = source.joint_id(0, 1);
    let mut clone = source.clone();

    let error = clone.detach_subtree(source_handle).unwrap_err();
    assert!(error.0.contains("belongs to another world"));
}

#[test]
fn cloned_world_is_logically_equal_but_structural_edits_are_not() {
    let world = world_with_handle_tree();
    let mut clone = world.clone();

    assert_eq!(world, clone);

    clone.trees[0].links[1].mass += 1.0;
    assert_ne!(world, clone);
}

#[test]
fn replacing_a_link_with_its_clone_rejects_the_old_handle() {
    let mut world = world_with_handle_tree();
    let stale = world.joint_id(0, 1);
    world.trees[0].links[1] = world.trees[0].links[1].clone();

    let error = world.detach_subtree(stale).unwrap_err();
    assert!(error.0.contains("stale link handle"));
}

#[test]
fn value_edits_preserve_live_handles() {
    let mut world = world_with_handle_tree();
    let mass_handle = world.joint_id(0, 1);
    world.trees[0].links[1].mass = 2.0;
    world.detach_subtree(mass_handle).unwrap();

    let mut damping_world = world_with_handle_tree();
    let damping_handle = damping_world.joint_id(0, 1);
    if let JointKind::Hinge { damping, .. } = &mut damping_world.trees[0].links[1].joint {
        *damping = 2.5;
    }
    damping_world.detach_subtree(damping_handle).unwrap();

    let mut mocap_world = world_with_mocap_handle_tree();
    let pose_handle = mocap_world.joint_id(0, 1);
    mocap_world.set_mocap_pose(0, Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
    mocap_world.detach_subtree(pose_handle).unwrap();
}

#[test]
fn remapped_unaffected_handle_stays_valid() {
    let mut world = world_with_handle_tree();
    let mut other_tree = Tree::new();
    other_tree.push_link(link(None, JointKind::Fixed, 5.0));
    other_tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 1.0));
    world.add_tree(other_tree);
    let unaffected = world.joint_id(1, 1);

    let report = world.detach_subtree(world.joint_id(0, 1)).unwrap();
    let remapped = report.remap_link(unaffected).unwrap();
    assert_eq!(remapped.tree_id, 1);
    assert_eq!(remapped.link_id, unaffected.link_id);
    world.detach_subtree(remapped).unwrap();
}

#[test]
fn stale_link_handle_cannot_detach_a_compacted_sibling() {
    let mut world = world_with_references();
    let stale = world.joint_id(0, 1);
    world.detach_subtree(stale).unwrap();

    let result = world.detach_subtree(stale);
    assert!(result.unwrap_err().0.contains("stale link handle"));
    assert_eq!(world.trees[0].links.len(), 2);
}

#[test]
fn sequential_detaches_reject_stale_handles_in_reports() {
    let mut world = world_with_handle_tree();
    let stale = world.joint_id(0, 1);
    world.detach_subtree(stale).unwrap();

    let second = world.joint_id(0, 1);
    let second_report = world.detach_subtree(second).unwrap();

    let error = second_report.remap_link(stale).unwrap_err();
    assert!(error.0.contains("stale link handle"));
    assert!(
        world
            .detach_subtree(stale)
            .unwrap_err()
            .0
            .contains("stale link handle")
    );
}

#[test]
fn stale_handle_cannot_alias_after_interleaved_tree_addition() {
    let mut world = world_with_references();
    let stale = world.joint_id(0, 1);
    let survivor = world.joint_id(0, 3);
    world.detach_subtree(stale).unwrap();

    let mut added = Tree::new();
    added.push_link(link(None, JointKind::Fixed, 10.0));
    world.add_tree(added);

    assert_ne!(world.joint_id(0, 1), survivor);
    assert!(
        world
            .detach_subtree(stale)
            .unwrap_err()
            .0
            .contains("stale link handle")
    );
}

#[test]
fn direct_link_addition_keeps_existing_handle_live() {
    let mut world = world_with_handle_tree();
    let survivor = world.joint_id(0, 3);
    world.trees[0].push_link(link(Some(0), JointKind::hinge(Vec3::Z), 2.0));

    world.detach_subtree(survivor).unwrap();
}

#[test]
fn direct_tree_addition_keeps_existing_handle_live() {
    let mut world = world_with_handle_tree();
    let survivor = world.joint_id(0, 3);
    let mut added = Tree::new();
    added.push_link(link(None, JointKind::Fixed, 10.0));
    world.trees.push(added);

    world.detach_subtree(survivor).unwrap();
}

#[test]
fn replacing_equal_length_tree_rejects_old_handle() {
    let mut world = world_with_handle_tree();
    let stale = world.joint_id(0, 1);
    let mut replacement = Tree::new();
    replacement.push_link(link(None, JointKind::Fixed, 10.0));
    replacement.push_link(link(Some(0), JointKind::hinge(Vec3::X), 11.0));
    replacement.push_link(link(Some(1), JointKind::hinge(Vec3::Y), 12.0));
    replacement.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 13.0));
    world.trees[0] = replacement;

    let error = world.detach_subtree(stale).unwrap_err();
    assert!(error.0.contains("stale link handle"));
    assert_eq!(world.trees.len(), 1);
    assert_eq!(world.trees[0].links.len(), 4);
    world.detach_subtree(world.joint_id(0, 1)).unwrap();
}

#[test]
fn direct_tree_bootstrap_rejects_old_handle_after_equal_length_replacement() {
    let mut world = World::new();
    let mut tree = Tree::new();
    tree.push_link(link(None, JointKind::Fixed, 0.0));
    tree.push_link(link(Some(0), JointKind::hinge(Vec3::Z), 1.0));
    world.trees.push(tree);
    let stale = world.joint_id(0, 1);

    let mut replacement = Tree::new();
    replacement.push_link(link(None, JointKind::Fixed, 10.0));
    replacement.push_link(link(Some(0), JointKind::hinge(Vec3::X), 11.0));
    world.trees[0] = replacement;

    let error = world.detach_subtree(stale).unwrap_err();
    assert!(error.0.contains("stale link handle"));
}

#[test]
fn handle_from_another_world_is_rejected() {
    let first = world_with_handle_tree();
    let mut second = world_with_handle_tree();
    let foreign = first.joint_id(0, 1);

    let error = second.detach_subtree(foreign).unwrap_err();
    assert!(error.0.contains("belongs to another world"));
    assert_eq!(second.trees.len(), 1);
}

#[test]
fn detach_clears_last_solver_phase_diagnostics() {
    let mut world = world_with_references();
    world.set_solver_phase_capture(true);
    world.capture_solver_phase();
    assert!(world.solver_phase_diagnostics().is_some());

    world.detach_subtree(world.joint_id(0, 1)).unwrap();
    assert!(world.solver_phase_diagnostics().is_none());
}
