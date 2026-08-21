use newt::body::Body;
use newt::broadphase::{Aabb, Ray};
use newt::geom::{ConvexMesh, Geom, GeomPose, HeightField};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree};
use newt::world::{RayHit, ShapeDesc, World};

fn pose(x: f32, y: f32, z: f32) -> GeomPose {
    GeomPose {
        position: Vec3::new(x, y, z),
        orientation: Quat::IDENTITY,
    }
}

fn serialize_hits(hits: &[RayHit]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(hits.len() * 32);
    for hit in hits {
        bytes.extend_from_slice(&(hit.geom_id as u64).to_le_bytes());
        match hit.body_id {
            Some(body_id) => {
                bytes.push(1);
                bytes.extend_from_slice(&(body_id as u64).to_le_bytes());
            }
            None => bytes.push(0),
        }
        for value in [
            hit.t,
            hit.point_world.x,
            hit.point_world.y,
            hit.point_world.z,
            hit.normal_world.x,
            hit.normal_world.y,
            hit.normal_world.z,
        ] {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    bytes
}

#[test]
fn raycast_nearest_all_miss_max_dist_and_mask() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    for (x, group) in [(3.0, 0x1), (2.0, 0x2), (1.0, 0x4)] {
        let body = world.add_body(Body::solid_sphere(
            1.0,
            0.5,
            Vec3::new(x, 0.0, 0.0),
            Quat::IDENTITY,
        ));
        world.add_geom(
            Geom::sphere(body, 0.5, Vec3::ZERO, 0.0).with_collision_filter(group, u32::MAX),
        );
    }
    let ray = Ray {
        origin: Vec3::ZERO,
        direction: Vec3::X,
    };
    let nearest = world.raycast(ray, 10.0, u32::MAX).expect("ray should hit");
    assert_eq!(nearest.geom_id, 2);
    assert!((nearest.t - 0.5).abs() < 1.0e-5);
    assert_eq!(world.raycast(ray, 0.4, u32::MAX), None);
    assert_eq!(world.raycast(ray, 10.0, 0x2).unwrap().geom_id, 1);
    let all = world.raycast_all(ray, 10.0, u32::MAX);
    assert_eq!(all.len(), 3);
    assert!(all[0].t < all[1].t && all[1].t < all[2].t);
    assert_eq!(all[0].geom_id, 2);
    assert_ne!(all[0].geom_id, 0);
    assert_eq!(world.raycast_all(ray, 10.0, 0x6).len(), 2);
    assert_eq!(world.raycast_all(ray, 10.0, 0).len(), 0);
    let first_bytes = serialize_hits(&world.raycast_all(ray, 10.0, u32::MAX));
    let second_bytes = serialize_hits(&world.raycast_all(ray, 10.0, u32::MAX));
    assert_eq!(first_bytes, second_bytes);
    assert_eq!(
        world.raycast(
            Ray {
                origin: Vec3::ZERO,
                direction: Vec3::Y
            },
            10.0,
            u32::MAX
        ),
        None
    );
}

#[test]
fn body_pose_mutation_refreshes_query_proxy() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(2.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.0));
    let ray = Ray {
        origin: Vec3::ZERO,
        direction: Vec3::X,
    };
    assert!(world.raycast(ray, 5.0, u32::MAX).is_some());

    world.set_body_pose(body, Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY);

    let hit = world
        .raycast(
            Ray {
                origin: Vec3::new(9.0, 0.0, 0.0),
                direction: Vec3::X,
            },
            5.0,
            u32::MAX,
        )
        .expect("moved body should hit");
    assert_eq!(hit.geom_id, 0);
    assert!((hit.t - 0.5).abs() < 1.0e-5);
}

fn free_tree_query_world() -> (World, usize) {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let tree_id = world.add_tree(tree);
    let mut geom = Geom::sphere(0, 0.5, Vec3::ZERO, 0.0);
    geom.body = None;
    geom.link = Some((tree_id, 0));
    world.add_geom(geom);
    (world, tree_id)
}

fn ray_hit_at_five(world: &World) -> RayHit {
    world
        .raycast(
            Ray {
                origin: Vec3::new(9.0, 0.0, 0.0),
                direction: Vec3::X,
            },
            5.0,
            u32::MAX,
        )
        .expect("moved tree geometry should hit")
}

#[test]
fn keyframe_application_refreshes_query_proxy() {
    let (mut world, _) = free_tree_query_world();
    world
        .add_keyframe(
            "moved",
            vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0; 6],
            vec![],
            vec![],
        )
        .unwrap();
    assert_eq!(
        world.raycast(
            Ray {
                origin: Vec3::new(9.0, 0.0, 0.0),
                direction: Vec3::X
            },
            5.0,
            u32::MAX
        ),
        None
    );

    world.reset_to_keyframe("moved").unwrap();

    assert_eq!(ray_hit_at_five(&world).geom_id, 0);
}

#[test]
fn qpos_application_refreshes_query_proxy() {
    let (mut world, _) = free_tree_query_world();
    assert_eq!(
        world.raycast(
            Ray {
                origin: Vec3::new(9.0, 0.0, 0.0),
                direction: Vec3::X
            },
            5.0,
            u32::MAX
        ),
        None
    );

    world.apply_mujoco_qpos(&[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

    assert_eq!(ray_hit_at_five(&world).geom_id, 0);
}

#[test]
fn direct_tree_pose_mutation_refreshes_query_proxy() {
    let (mut world, tree_id) = free_tree_query_world();
    assert_eq!(
        world.raycast(
            Ray {
                origin: Vec3::new(9.0, 0.0, 0.0),
                direction: Vec3::X
            },
            5.0,
            u32::MAX
        ),
        None
    );

    world.trees[tree_id].set_free_root_pose(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY);

    assert_eq!(ray_hit_at_five(&world).geom_id, 0);
}

#[test]
fn overlap_queries_filter_and_sort_geom_ids() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    for (x, group) in [(0.5, 0x1), (1.5, 0x2), (-1.5, 0x4), (-0.5, 0x8)] {
        let body = world.add_body(Body::solid_sphere(
            1.0,
            0.25,
            Vec3::new(x, 0.0, 0.0),
            Quat::IDENTITY,
        ));
        world.add_geom(
            Geom::sphere(body, 0.25, Vec3::ZERO, 0.0).with_collision_filter(group, u32::MAX),
        );
    }
    let mut raw_sphere = Vec::new();
    world.raw_query_aabb_candidates(
        Aabb::from_center_extents(Vec3::ZERO, Vec3::splat(2.0)),
        |geom_id| raw_sphere.push(geom_id),
    );
    assert_eq!(raw_sphere, vec![2, 3, 0, 1]);
    assert_ne!(raw_sphere, vec![0, 1, 2, 3]);
    let mut raw_box = Vec::new();
    world.raw_query_aabb_candidates(
        Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
        |geom_id| raw_box.push(geom_id),
    );
    assert_eq!(raw_box, vec![3, 0]);
    assert_ne!(raw_box, vec![0, 3]);
    assert_eq!(
        world.overlap_sphere(Vec3::ZERO, 2.0, u32::MAX),
        vec![0, 1, 2, 3]
    );
    assert_eq!(world.overlap_sphere(Vec3::ZERO, 2.0, 0x6), vec![1, 2]);
    assert_eq!(
        world.overlap_box(
            Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            u32::MAX
        ),
        vec![0, 3]
    );
    assert!(world.overlap_box(Aabb::UNBOUNDED, 0).is_empty());
}

#[test]
fn sphere_shape_cast_reports_analytic_toi() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let target = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        target,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world
        .shape_cast(
            ShapeDesc::Sphere { radius: 0.5 },
            pose(0.0, 0.0, 0.0),
            pose(10.0, 0.0, 0.0),
            u32::MAX,
        )
        .expect("sphere should hit box");
    assert_eq!(hit.geom_id, 0);
    assert!((hit.t - 0.4).abs() < 2.0e-4, "toi was {}", hit.t);
    assert!(hit.normal_world.x < -0.9);
}

#[test]
fn box_shape_cast_reports_hit() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let target = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        target,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world
        .shape_cast(
            ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
            pose(0.0, 0.0, 0.0),
            pose(10.0, 0.0, 0.0),
            u32::MAX,
        )
        .expect("box should hit box");
    assert_eq!(hit.geom_id, 0);
    assert!(hit.t < 0.5);
    assert!(hit.normal_world.x < -0.9);
}

#[test]
fn shape_cast_layer_mask_skips_closer_geom() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    for (x, group) in [(3.0, 0x1), (6.0, 0x2)] {
        let body = world.add_body(Body::solid_box(
            1.0,
            Vec3::splat(0.5),
            Vec3::new(x, 0.0, 0.0),
            Quat::IDENTITY,
        ));
        world.add_geom(
            Geom::r#box(body, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY, 0.0)
                .with_collision_filter(group, u32::MAX),
        );
    }
    let hit = world
        .shape_cast(
            ShapeDesc::Sphere { radius: 0.25 },
            pose(0.0, 0.0, 0.0),
            pose(10.0, 0.0, 0.0),
            0x2,
        )
        .expect("unmasked box should be hit");
    assert_eq!(hit.geom_id, 1);
}

#[test]
fn capsule_shape_cast_parallel_to_wall_misses() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let wall = world.add_body(Body::solid_box(
        1.0,
        Vec3::new(0.25, 2.0, 2.0),
        Vec3::new(0.0, 3.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        wall,
        Vec3::new(0.25, 2.0, 2.0),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    assert_eq!(
        world.shape_cast(
            ShapeDesc::Capsule {
                radius: 0.25,
                half_height: 0.5
            },
            pose(-2.0, 0.0, 0.0),
            pose(2.0, 0.0, 0.0),
            u32::MAX
        ),
        None
    );
}

#[test]
fn query_does_not_change_simulation_state() {
    let mut queried = World::new();
    queried.gravity = Vec3::new(0.0, 0.0, -1.0);
    let body = queried.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(0.0, 0.0, 2.0),
        Quat::IDENTITY,
    ));
    queried.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.0));
    let mut control = queried.clone();
    queried.raycast(
        Ray {
            origin: Vec3::ZERO,
            direction: Vec3::Z,
        },
        10.0,
        u32::MAX,
    );
    queried.shape_cast(
        ShapeDesc::Sphere { radius: 0.25 },
        pose(-1.0, 0.0, 2.0),
        pose(1.0, 0.0, 2.0),
        u32::MAX,
    );
    queried.overlap_sphere(Vec3::ZERO, 4.0, u32::MAX);
    queried.step();
    control.step();
    assert_eq!(queried.bodies, control.bodies);
}

#[test]
fn convex_mesh_ray_uses_mesh_asset() {
    let mut world = World::new();
    let mesh_id = world.add_mesh(ConvexMesh {
        vertices: vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.0, 0.5, -0.5),
            Vec3::new(0.0, 0.0, 0.5),
        ],
        faces: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    });
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(2.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::mesh(body, mesh_id, Vec3::ZERO, Quat::IDENTITY, 0.0));
    let hit = world.raycast(
        Ray {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        },
        10.0,
        u32::MAX,
    );
    assert!(hit.is_some());
}

#[test]
fn convex_mesh_shape_cast_reuses_ccd_sweep() {
    let mut world = World::new();
    let mesh = ConvexMesh {
        vertices: vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.0, 0.5, -0.5),
            Vec3::new(0.0, 0.0, 0.5),
        ],
        faces: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    };
    let mesh_id = world.add_mesh(mesh);
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::mesh(body, mesh_id, Vec3::ZERO, Quat::IDENTITY, 0.0));
    let hit = world
        .shape_cast(
            ShapeDesc::ConvexMesh { mesh_id },
            pose(0.0, 0.0, 0.0),
            pose(10.0, 0.0, 0.0),
            u32::MAX,
        )
        .expect("convex mesh should hit convex mesh");
    assert_eq!(hit.geom_id, 0);
    assert!(hit.t > 0.3 && hit.t < 0.6);
}

#[test]
fn raycast_cylinder_hits_curved_surface() {
    let mut world = World::new();
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(3.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::cylinder(
        body,
        0.5,
        1.0,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world
        .raycast(
            Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            },
            10.0,
            u32::MAX,
        )
        .expect("cylinder should be hit");
    assert!((hit.t - 2.5).abs() < 1.0e-5);
    assert!(hit.normal_world.x < -0.99);
}

#[test]
fn raycast_ellipsoid_hits_surface() {
    let mut world = World::new();
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(3.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::ellipsoid(
        body,
        Vec3::new(0.5, 1.0, 1.0),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world
        .raycast(
            Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            },
            10.0,
            u32::MAX,
        )
        .expect("ellipsoid should be hit");
    assert!((hit.t - 2.5).abs() < 1.0e-5);
    assert!(hit.normal_world.x < -0.99);
}

#[test]
fn raycast_capsule_hits_barrel() {
    let mut world = World::new();
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(3.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::capsule(
        body,
        0.5,
        1.0,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world
        .raycast(
            Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            },
            10.0,
            u32::MAX,
        )
        .expect("capsule barrel should be hit");
    assert!((hit.t - 2.5).abs() < 1.0e-5);
    assert!(hit.normal_world.x < -0.99);
}

#[test]
fn raycast_plane_and_heightfield_return_normals() {
    let mut world = World::new();
    world.add_geom(Geom::static_plane(Vec3::new(3.0, 0.0, 0.0), -Vec3::X, 0.0));
    let plane_hit = world
        .raycast(
            Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            },
            10.0,
            u32::MAX,
        )
        .expect("plane should be hit");
    assert!((plane_hit.t - 3.0).abs() < 1.0e-5);
    assert!(plane_hit.normal_world.x < -0.99);

    let hfield_id = world.add_hfield(HeightField {
        nrow: 2,
        ncol: 2,
        size: [1.0, 1.0, 1.0, 1.0],
        data: vec![0.0; 4],
    });
    world.add_geom(Geom::static_hfield(
        hfield_id,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hfield_hit = world
        .raycast(
            Ray {
                origin: Vec3::new(0.0, 0.0, 1.0),
                direction: -Vec3::Z,
            },
            10.0,
            u32::MAX,
        )
        .expect("heightfield should be hit");
    assert!((hfield_hit.t - 1.0).abs() < 1.0e-5);
    assert!(hfield_hit.normal_world.z > 0.99);
}

#[test]
fn raycast_box_from_inside_returns_exit_normal() {
    let mut world = World::new();
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world
        .raycast(
            Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            },
            10.0,
            u32::MAX,
        )
        .expect("inside ray should exit the box");
    assert!(hit.t > 0.0);
    assert!(hit.normal_world.length_squared() > 0.99);
    assert!(hit.normal_world.x > 0.99);
    assert_eq!(
        world.raycast(
            Ray {
                origin: Vec3::ZERO,
                direction: Vec3::X,
            },
            0.4,
            u32::MAX,
        ),
        None
    );
}

#[test]
fn shape_cast_finds_subsample_window() {
    let mut world = World::new();
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.005,
        Vec3::new(5.03, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(body, 0.005, Vec3::ZERO, 0.0));
    let hit = world
        .shape_cast(
            ShapeDesc::Sphere { radius: 0.005 },
            pose(0.0, 0.0, 0.0),
            pose(10.0, 0.0, 0.0),
            u32::MAX,
        )
        .expect("thin crossing must not tunnel");
    assert!((hit.t - 0.502).abs() < 1.0e-4, "toi was {}", hit.t);
}

#[test]
fn shape_cast_capsule_barrel_hit_reports_toi() {
    let mut world = World::new();
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::capsule(
        body,
        0.5,
        1.0,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world
        .shape_cast(
            ShapeDesc::Sphere { radius: 0.25 },
            pose(0.0, 0.0, 0.0),
            pose(10.0, 0.0, 0.0),
            u32::MAX,
        )
        .expect("sphere should hit the capsule barrel");
    assert!((hit.t - 0.425).abs() < 1.0e-4, "toi was {}", hit.t);
}

#[test]
fn shape_cast_rotation_is_part_of_the_sweep() {
    let mut world = World::new();
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.1),
        Vec3::new(0.9, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.1),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    let hit = world.shape_cast(
        ShapeDesc::Capsule {
            radius: 0.1,
            half_height: 1.0,
        },
        pose(0.0, 0.0, 0.0),
        GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::from_axis_angle(Vec3::Y, core::f32::consts::FRAC_PI_2),
        },
        u32::MAX,
    );
    assert!(hit.is_some(), "rotation-driven collision was missed");
}
