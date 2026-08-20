use newt::body::Body;
use newt::broadphase::{Aabb, Ray};
use newt::geom::{ConvexMesh, Geom, GeomPose};
use newt::math::{Quat, Vec3};
use newt::world::{ShapeDesc, World};

fn pose(x: f32, y: f32, z: f32) -> GeomPose {
    GeomPose {
        position: Vec3::new(x, y, z),
        orientation: Quat::IDENTITY,
    }
}

#[test]
fn raycast_nearest_all_miss_max_dist_and_mask() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    for (x, group) in [(1.0, 0x1), (2.0, 0x2), (3.0, 0x4)] {
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
    assert_eq!(nearest.geom_id, 0);
    assert!((nearest.t - 0.5).abs() < 1.0e-5);
    assert_eq!(world.raycast(ray, 0.4, u32::MAX), None);
    assert_eq!(world.raycast(ray, 10.0, 0x2).unwrap().geom_id, 1);
    assert_eq!(world.raycast_all(ray, 10.0, 0x6).len(), 2);
    assert_eq!(world.raycast_all(ray, 10.0, 0).len(), 0);
    assert_eq!(
        world.raycast_all(ray, 10.0, u32::MAX),
        world.raycast_all(ray, 10.0, u32::MAX)
    );
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
fn overlap_queries_filter_and_sort_geom_ids() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    for (x, group) in [(-1.5, 0x1), (-0.5, 0x2), (0.5, 0x4), (1.5, 0x8)] {
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
    assert_eq!(
        world.overlap_sphere(Vec3::ZERO, 2.0, u32::MAX),
        vec![0, 1, 2, 3]
    );
    assert_eq!(world.overlap_sphere(Vec3::ZERO, 2.0, 0x6), vec![1, 2]);
    assert_eq!(
        world.overlap_box(
            Aabb::new(Vec3::new(-2.0, -1.0, -1.0), Vec3::new(0.0, 1.0, 1.0)),
            u32::MAX
        ),
        vec![0, 1]
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
