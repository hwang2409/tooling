use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::mjcf::load_mjcf_str;
use newt::model::load_str;
use newt::world::World;
use std::panic::{self, AssertUnwindSafe};

fn pair_world(a_group: u32, a_mask: u32, b_group: u32, b_mask: u32) -> World {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let body_a = world.add_body(Body::solid_sphere(1.0, 0.5, Vec3::ZERO, Quat::IDENTITY));
    let body_b = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(0.75, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(
        Geom::sphere(body_a, 0.5, Vec3::ZERO, 0.5).with_collision_filter(a_group, a_mask),
    );
    world.add_geom(
        Geom::sphere(body_b, 0.5, Vec3::ZERO, 0.5).with_collision_filter(b_group, b_mask),
    );
    world
}

#[test]
fn symmetric_filter_emits_a_manifold_in_either_id_order() {
    let mut first = pair_world(0x01, 0x02, 0x02, 0x01);
    let mut second = pair_world(0x02, 0x01, 0x01, 0x02);

    assert_eq!(first.broadphase_pair_count(), 1);
    assert_eq!(second.broadphase_pair_count(), 1);
    assert!(!first.detect_contacts().is_empty());
    assert!(!second.detect_contacts().is_empty());
}

#[test]
fn asymmetric_filter_rejects_candidate() {
    let mut world = pair_world(0x01, 0x02, 0x04, 0x01);

    assert_eq!(world.broadphase_pair_count(), 0);
    assert!(world.detect_contacts().is_empty());
}

#[test]
fn filter_requires_both_directions() {
    let mut world = pair_world(0x01, 0x02, 0x02, 0x04);

    assert_eq!(world.broadphase_pair_count(), 0);
    assert!(world.detect_contacts().is_empty());
}

#[test]
fn default_filter_matches_all_overlapping_pairs() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    for x in [0.0, 0.5, 0.8] {
        let body = world.add_body(Body::solid_sphere(
            1.0,
            0.5,
            Vec3::new(x, 0.0, 0.0),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.5));
    }

    assert_eq!(world.broadphase_pair_count(), 3);
    assert_eq!(world.detect_contacts().len(), 3);
}

#[test]
fn manual_pair_list_bypasses_filter() {
    let mut world = pair_world(0x01, 0x02, 0x04, 0x01);
    world.pair_list = Some(vec![(0, 1)]);

    assert_eq!(world.detect_contacts().len(), 1);
}

#[test]
fn filter_mutation_changes_the_next_pair_set() {
    let mut world = pair_world(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(world.broadphase_pair_count(), 1);

    world.set_geom_filter(0, 0x01, 0x02).unwrap();
    world.set_geom_filter(1, 0x04, 0x01).unwrap();
    assert_eq!(world.broadphase_pair_count(), 0);

    world.set_geom_filter(1, 0x02, 0x01).unwrap();
    assert_eq!(world.broadphase_pair_count(), 1);
}

#[test]
fn user_data_round_trips() {
    let mut world = pair_world(u32::MAX, u32::MAX, u32::MAX, u32::MAX);
    world.set_geom_user_data(0, 42).unwrap();

    assert_eq!(world.geom_user_data(0), Some(42));
}

fn body_snapshot(world: &World) -> Vec<u32> {
    world
        .bodies
        .iter()
        .flat_map(|body| {
            [
                body.position.x,
                body.position.y,
                body.position.z,
                body.linear_velocity.x,
                body.linear_velocity.y,
                body.linear_velocity.z,
                body.orientation.x,
                body.orientation.y,
                body.orientation.z,
                body.orientation.w,
                body.angular_velocity_body.x,
                body.angular_velocity_body.y,
                body.angular_velocity_body.z,
            ]
            .map(f32::to_bits)
        })
        .collect()
}

#[test]
fn filtering_preserves_determinism() {
    let mut filtered = triplet_world();
    let mut manual = triplet_world();
    manual.pair_list = Some(vec![(0, 1), (1, 2)]);

    assert_eq!(filtered.broadphase_pair_count(), 2);
    assert_eq!(manual.pair_list.as_deref(), Some(&[(0, 1), (1, 2)][..]));

    for _ in 0..100 {
        filtered.step();
        manual.step();
    }

    assert_eq!(body_snapshot(&filtered), body_snapshot(&manual));
}

fn triplet_world() -> World {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    for (x, group, mask) in [(0.0, 0x01, 0x02), (0.5, 0x02, 0x05), (0.8, 0x04, 0x02)] {
        let body = world.add_body(Body::solid_sphere(
            1.0,
            0.5,
            Vec3::new(x, 0.0, 0.0),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.5).with_collision_filter(group, mask));
    }
    world
}

#[test]
fn filter_mutation_rechecks_newly_active_unsupported_pairs() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let box_body = world.add_body(Body::solid_sphere(1.0, 0.5, Vec3::ZERO, Quat::IDENTITY));
    let sphere_body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(0.5, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.add_geom(
        Geom::r#box(box_body, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY, 0.5)
            .with_collision_filter(0x01, 0x01),
    );
    world.add_geom(
        Geom::sphere(sphere_body, 0.5, Vec3::ZERO, 0.5).with_collision_filter(0x02, 0x02),
    );

    world.step();
    world.set_geom_filter(1, 0x01, 0x01).unwrap();
    assert!(panic::catch_unwind(AssertUnwindSafe(|| world.step())).is_err());
}

#[test]
fn mjcf_contype_and_conaffinity_filter_pairs() {
    let scene = load_mjcf_str(
        r#"
        <mujoco>
          <worldbody>
            <body name="a" pos="0 0 0">
              <geom name="a_geom" type="sphere" size="0.5" mass="1" contype="1" conaffinity="2"/>
            </body>
            <body name="b" pos="0.75 0 0">
              <geom name="b_geom" type="sphere" size="0.5" mass="1" contype="4" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        "#,
    )
    .unwrap();
    let mut world = scene.world;

    assert_eq!(world.broadphase_pair_count(), 0);
    assert!(world.detect_contacts().is_empty());
}

#[test]
fn loaded_mjcf_filters_apply_after_runtime_mutation() {
    let scene = load_mjcf_str(
        r#"
        <mujoco>
          <worldbody>
            <body name="a">
              <freejoint/>
              <geom name="a_geom" type="sphere" size="0.5" mass="1"/>
            </body>
            <body name="b" pos="0.75 0 0">
              <freejoint/>
              <geom name="b_geom" type="sphere" size="0.5" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        "#,
    )
    .unwrap();
    let mut world = scene.world;

    assert_eq!(world.pair_list, None);
    assert_eq!(world.broadphase_pair_count(), 1);
    world.set_geom_filter(0, 0x01, 0x00).unwrap();
    assert_eq!(world.broadphase_pair_count(), 0);
    world.set_geom_filter(0, 0x01, 0x01).unwrap();
    assert_eq!(world.broadphase_pair_count(), 1);
}

#[test]
fn disabled_tree_self_collision_is_compact_and_runtime_toggleable() {
    let json = r#"{
        "version":"1",
        "trees":[{"name":"tree","self_collide":false,"links":[
            {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
            {"name":"child","parent":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
        ]}],
        "geoms":[
            {"name":"root_geom","shape":{"kind":"sphere","radius":0.5},"attach":{"kind":"link","tree":"tree","link":"root"}},
            {"name":"child_geom","shape":{"kind":"sphere","radius":0.5},"attach":{"kind":"link","tree":"tree","link":"child"}}
        ]
    }"#;
    let mut json_world = load_str(json).unwrap().world;
    assert_eq!(json_world.auto_pair_exclusion_count(), 0);
    assert_eq!(json_world.broadphase_pair_count(), 0);
    json_world.step();
    json_world.set_tree_self_collision(0, true).unwrap();
    json_world.step();
    assert_eq!(json_world.broadphase_pair_count(), 1);
    json_world.set_tree_self_collision(0, false).unwrap();
    json_world.step();
    assert_eq!(json_world.broadphase_pair_count(), 0);
    json_world.pair_list = Some(vec![(0, 1)]);
    assert_eq!(json_world.detect_contacts().len(), 1);

    let mjcf = r#"
        <mujoco>
          <worldbody>
            <body name="root">
              <inertial mass="1" diaginertia="1 1 1"/>
              <geom name="root_geom" type="sphere" size="0.5"/>
              <body name="child">
                <inertial mass="1" diaginertia="1 1 1"/>
                <geom name="child_geom" type="sphere" size="0.5"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
    "#;
    let mut mjcf_world = load_mjcf_str(mjcf).unwrap().world;
    assert_eq!(mjcf_world.auto_pair_exclusion_count(), 0);
    assert_eq!(mjcf_world.broadphase_pair_count(), 0);
    mjcf_world.set_tree_self_collision(0, true).unwrap();
    assert_eq!(mjcf_world.broadphase_pair_count(), 1);
}
