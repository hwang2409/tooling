use newt::actuator::{Actuator, BiasType, DynType, GainType};
use newt::body::Body;
use newt::geom::Geom;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::sensor::{Sensor, SensorAttach, SensorKind, SiteFrame};
use newt::tree::{Link, Tree, forward_kinematics};
use newt::world::World;

fn hinge_tree() -> Tree {
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
        JointKind::hinge(Vec3::Z),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(-1.0, 0.0, 0.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree
}

#[test]
fn link_jacobian_matches_two_link_hand_derivation() {
    let mut tree = hinge_tree();
    tree.set_hinge_angle(1, 0.0);
    let jacobian = tree.link_jacobian(1);
    assert_eq!(jacobian.nv(), 1);
    assert_eq!(jacobian.rotational[0], Vec3::Z);
    assert!((jacobian.translational[0] - Vec3::Y).length() < 1e-6);
}

#[test]
fn inverse_dynamics_at_uses_explicit_state_without_mutating_tree() {
    let tree = hinge_tree();
    let q = vec![0.25];
    let qdot = vec![0.4];
    let qddot = vec![0.7];
    let tau = tree.inverse_dynamics_at(
        &q,
        &qdot,
        &qddot,
        Vec3::new(0.0, 0.0, -9.81),
        &vec![(Vec3::ZERO, Vec3::ZERO); 2],
    );
    assert_eq!(tau.len(), 1);
    assert_eq!(tree.q, vec![0.0]);
}

#[test]
fn keyframe_reset_then_step_matches_fresh_state() {
    let mut world = World::new();
    let mut tree = hinge_tree();
    let actuator = tree.add_actuator(Actuator::general(
        1,
        GainType::Fixed,
        [1.0, 0.0, 0.0],
        BiasType::None,
        [0.0, 0.0, 0.0],
        1.0,
        DynType::Filter,
        [0.1],
        None,
        None,
    ));
    world.add_tree(tree);
    world
        .add_keyframe("pose", vec![0.35], vec![-0.2], vec![0.25], vec![0.6])
        .unwrap();

    let mut fresh = world.clone();
    fresh.keyframes.clear();
    fresh.trees[0].q[0] = 0.35;
    fresh.trees[0].qdot[0] = -0.2;
    fresh.trees[0].actuators[actuator].act = 0.25;
    fresh.trees[0].actuators[actuator].ctrl = 0.6;

    world.reset_to_keyframe("pose").unwrap();
    for _ in 0..5 {
        world.step();
        fresh.step();
    }
    assert_eq!(world.trees[0].q, fresh.trees[0].q);
    assert_eq!(world.trees[0].qdot, fresh.trees[0].qdot);
    assert_eq!(world.trees[0].actuators, fresh.trees[0].actuators);
}

#[test]
fn rangefinder_and_magnetometer_use_site_frame() {
    let mut world = World::new();
    world.magnetic_field = Vec3::new(0.0, -0.5, 0.0);
    let body = world.add_body(Body::principal_axis(
        1.0,
        1.0,
        1.0,
        1.0,
        Vec3::new(0.0, 0.0, -1.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.0));
    let site = SiteFrame {
        attach: SensorAttach::Body(body),
        local_offset: Vec3::ZERO,
        local_orientation: Quat::IDENTITY,
    };
    let range = world
        .add_sensor(Sensor {
            name: "range".into(),
            kind: SensorKind::Rangefinder(site),
        })
        .unwrap();
    let magnetometer = world
        .add_sensor(Sensor {
            name: "mag".into(),
            kind: SensorKind::Magnetometer(site),
        })
        .unwrap();
    world.evaluate_sensors(&[]);
    assert!((world.sensor(range).unwrap()[0] - 1.0).abs() < 1e-6);
    assert_eq!(world.sensor(magnetometer).unwrap(), &[0.0, -0.5, 0.0]);
}

#[test]
fn mocap_root_pose_is_not_integrated_under_force() {
    let mut world = World::new();
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.set_mocap(0, true);
    tree.set_link_wrench(0, Vec3::new(100.0, 0.0, 0.0), Vec3::ZERO);
    world.add_tree(tree);
    world.set_mocap_pose(0, Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY);
    world.step();
    assert_eq!(world.tree_link_pose(0, 0).0, Vec3::new(2.0, 0.0, 0.0));
}

#[test]
fn mocap_root_restore_does_not_freeze_descendant_slots() {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut tree = hinge_tree();
    tree.links[0].mocap = true;
    world.add_tree(tree);
    let initial = world.trees[0].q[0];
    for _ in 0..10 {
        world.step();
    }
    assert!((world.trees[0].q[0] - initial).abs() > 1e-5);
}

#[test]
fn mocap_platform_velocity_drags_sphere() {
    let mut world = World::new();
    world.dt = 0.002;
    let mut platform = Tree::new();
    platform.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        10.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    platform.set_mocap(0, true);
    world.add_tree(platform);
    let mut platform_geom = Geom::cylinder(0, 2.0, 0.1, Vec3::ZERO, Quat::IDENTITY, 1.0);
    platform_geom.body = None;
    platform_geom.link = Some((0, 0));
    world.add_geom(platform_geom);
    let sphere = world.add_body(Body::principal_axis(
        1.0,
        1.0,
        1.0,
        1.0,
        Vec3::new(0.0, 0.0, 0.7),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(sphere, 0.5, Vec3::ZERO, 1.0));
    for step in 0..200 {
        let x = step as f32 * world.dt;
        world.trees[0].set_mocap_pose(Vec3::new(x, 0.0, 0.0), Quat::IDENTITY);
        world.trees[0].set_mocap_velocity(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        world.step();
    }
    assert!(world.bodies[sphere].position.x > 0.05);
    assert!((world.tree_link_pose(0, 0).0.x - 0.398).abs() < 1e-6);
}

#[test]
fn mixed_tree_point_jacobian_matches_central_difference() {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::Z),
        (Vec3::new(0.2, 0.1, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.4), Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(Link::new(
        Some(1),
        JointKind::slide(Vec3::Y),
        (Vec3::new(0.3, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(Link::new(
        Some(2),
        JointKind::Ball {
            damping: 0.0,
            armature: 0.0,
        },
        (Vec3::new(0.0, 0.2, 0.0), Quat::IDENTITY),
        (Vec3::new(0.1, 0.0, 0.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let root_q = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.5).normalize(), 0.4);
    tree.q[0..3].copy_from_slice(&[0.3, -0.2, 0.7]);
    tree.q[3..7].copy_from_slice(&[root_q.x, root_q.y, root_q.z, root_q.w]);
    tree.q[tree.q_offset[1]] = 0.25;
    tree.q[tree.q_offset[2]] = -0.15;
    let ball = Quat::from_axis_angle(Vec3::new(0.4, 1.0, -0.2).normalize(), -0.3);
    let ball_off = tree.q_offset[3];
    tree.q[ball_off..ball_off + 4].copy_from_slice(&[ball.x, ball.y, ball.z, ball.w]);

    let point_local = Vec3::new(0.2, -0.1, 0.3);
    let jacobian = tree.point_jacobian(3, point_local);
    let eps = 1.0e-4;
    let point = |state: &Tree| {
        let (position, orientation) = forward_kinematics(state)[3];
        position + orientation.rotate(point_local)
    };
    for column in 0..tree.nv() {
        let mut plus = tree.clone();
        let mut minus = tree.clone();
        if column < 3 {
            for (state, sign) in [(&mut plus, 1.0f32), (&mut minus, -1.0f32)] {
                let root = Quat::new(state.q[3], state.q[4], state.q[5], state.q[6]);
                let delta = Quat::from_axis_angle([Vec3::X, Vec3::Y, Vec3::Z][column], sign * eps);
                let next = root * delta;
                state.q[3..7].copy_from_slice(&[next.x, next.y, next.z, next.w]);
            }
        } else if column < 6 {
            let root = Quat::new(tree.q[3], tree.q[4], tree.q[5], tree.q[6]);
            let direction = root.rotate([Vec3::X, Vec3::Y, Vec3::Z][column - 3]);
            plus.q[0] += direction.x * eps;
            plus.q[1] += direction.y * eps;
            plus.q[2] += direction.z * eps;
            minus.q[0] -= direction.x * eps;
            minus.q[1] -= direction.y * eps;
            minus.q[2] -= direction.z * eps;
        } else {
            let mut link = None;
            for i in 1..tree.links.len() {
                let start = tree.v_offset[i];
                let width = tree.links[i].joint.nv();
                if (start..start + width).contains(&column) {
                    link = Some(i);
                    break;
                }
            }
            let link = link.expect("joint column");
            let local = column - tree.v_offset[link];
            match tree.links[link].joint {
                JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                    plus.q[tree.q_offset[link]] += eps;
                    minus.q[tree.q_offset[link]] -= eps;
                }
                JointKind::Ball { .. } => {
                    let off = tree.q_offset[link];
                    let current = Quat::new(
                        tree.q[off],
                        tree.q[off + 1],
                        tree.q[off + 2],
                        tree.q[off + 3],
                    );
                    let axis = [Vec3::X, Vec3::Y, Vec3::Z][local];
                    let p = current * Quat::from_axis_angle(axis, eps);
                    let m = current * Quat::from_axis_angle(axis, -eps);
                    plus.q[off..off + 4].copy_from_slice(&[p.x, p.y, p.z, p.w]);
                    minus.q[off..off + 4].copy_from_slice(&[m.x, m.y, m.z, m.w]);
                }
                _ => unreachable!(),
            }
        }
        let derivative = (point(&plus) - point(&minus)) / (2.0 * eps);
        assert!(
            (derivative - jacobian.translational[column]).length() < 2.0e-3,
            "column {column}: finite difference {derivative:?}, analytic {:?}",
            jacobian.translational[column]
        );
    }
}

#[test]
fn rangefinder_hits_each_supported_shape() {
    let cases = [
        ("plane", 2.0),
        ("sphere", 1.5),
        ("box", 1.5),
        ("capsule", 1.0),
        ("cylinder", 1.5),
        ("ellipsoid", 1.5),
    ];
    for (kind, expected) in cases {
        let mut world = World::new();
        let probe = world.add_body(Body::principal_axis(
            1.0,
            1.0,
            1.0,
            1.0,
            Vec3::ZERO,
            Quat::IDENTITY,
        ));
        let target = world.add_body(Body::principal_axis(
            1.0,
            1.0,
            1.0,
            1.0,
            Vec3::new(0.0, 0.0, 2.0),
            Quat::IDENTITY,
        ));
        match kind {
            "plane" => world.add_geom(Geom::static_plane(Vec3::new(0.0, 0.0, 2.0), Vec3::Z, 0.0)),
            "sphere" => world.add_geom(Geom::sphere(target, 0.5, Vec3::ZERO, 0.0)),
            "box" => world.add_geom(Geom::r#box(
                target,
                Vec3::splat(0.5),
                Vec3::ZERO,
                Quat::IDENTITY,
                0.0,
            )),
            "capsule" => world.add_geom(Geom::capsule(
                target,
                0.5,
                0.5,
                Vec3::ZERO,
                Quat::IDENTITY,
                0.0,
            )),
            "cylinder" => world.add_geom(Geom::cylinder(
                target,
                0.5,
                0.5,
                Vec3::ZERO,
                Quat::IDENTITY,
                0.0,
            )),
            "ellipsoid" => world.add_geom(Geom::ellipsoid(
                target,
                Vec3::splat(0.5),
                Vec3::ZERO,
                Quat::IDENTITY,
                0.0,
            )),
            _ => unreachable!(),
        };
        let site = SiteFrame {
            attach: SensorAttach::Body(probe),
            local_offset: Vec3::ZERO,
            local_orientation: Quat::IDENTITY,
        };
        let sensor = world
            .add_sensor(Sensor {
                name: kind.into(),
                kind: SensorKind::Rangefinder(site),
            })
            .unwrap();
        world.evaluate_sensors(&[]);
        assert!(
            (world.sensor(sensor).unwrap()[0] - expected).abs() < 1.0e-5,
            "{kind}: got {}, expected {expected}",
            world.sensor(sensor).unwrap()[0]
        );
    }
    let mut empty = World::new();
    let probe = empty.add_body(Body::principal_axis(
        1.0,
        1.0,
        1.0,
        1.0,
        Vec3::ZERO,
        Quat::IDENTITY,
    ));
    let sensor = empty
        .add_sensor(Sensor {
            name: "empty".into(),
            kind: SensorKind::Rangefinder(SiteFrame {
                attach: SensorAttach::Body(probe),
                local_offset: Vec3::ZERO,
                local_orientation: Quat::IDENTITY,
            }),
        })
        .unwrap();
    empty.evaluate_sensors(&[]);
    assert_eq!(empty.sensor(sensor).unwrap(), &[-1.0]);
}

#[test]
fn subtreecom_matches_hand_sum() {
    let mut world = World::new();
    world.add_tree(hinge_tree());
    let site = Sensor {
        name: "com".into(),
        kind: SensorKind::SubtreeCom { tree: 0, link: 0 },
    };
    let sensor = world.add_sensor(site).unwrap();
    world.evaluate_sensors(&[]);
    assert_eq!(world.sensor(sensor).unwrap(), &[0.5, 0.0, 0.0]);
}
