use newt::body::Body;
use newt::broadphase::{Aabb, DynamicAabbTree, Ray};
use newt::geom::Geom;
use newt::math::Vec3;
use newt::world::{BroadPhaseMode, World};

fn bounds(x: f32, y: f32, z: f32) -> Aabb {
    Aabb::from_center_extents(Vec3::new(x, y, z), Vec3::splat(0.5))
}

fn fat_bounds(aabb: Aabb) -> Aabb {
    let extent = aabb.max - aabb.min;
    let margin = Vec3::new(
        (extent.x * 0.1).max(1.0e-4),
        (extent.y * 0.1).max(1.0e-4),
        (extent.z * 0.1).max(1.0e-4),
    );
    Aabb::new(aabb.min - margin, aabb.max + margin)
}

#[test]
fn tree_pair_insert_remove_update_and_queries() {
    let mut tree = DynamicAabbTree::new();
    tree.insert(10, bounds(0.0, 0.0, 0.0));
    tree.insert(20, bounds(0.75, 0.0, 0.0));
    tree.insert(30, bounds(5.0, 0.0, 0.0));
    assert_eq!(tree.compute_pairs(), &[(10, 20)]);

    let mut hits = Vec::new();
    assert!(tree.query_aabb(bounds(0.0, 0.0, 0.0), |proxy| {
        hits.push(proxy);
        true
    }));
    hits.sort_unstable();
    assert_eq!(hits, vec![10, 20]);

    let mut ray_hits = Vec::new();
    assert!(tree.query_ray(
        Ray {
            origin: Vec3::new(-3.0, 0.0, 0.0),
            direction: Vec3::X,
        },
        |proxy| {
            ray_hits.push(proxy);
            true
        },
    ));
    ray_hits.sort_unstable();
    assert_eq!(ray_hits, vec![10, 20, 30]);

    assert!(!tree.update(10, bounds(0.01, 0.0, 0.0)));
    assert!(tree.update(10, bounds(20.0, 0.0, 0.0)));
    assert_eq!(tree.compute_pairs(), &[]);
    assert!(tree.remove(20));
    assert!(!tree.remove(20));
    assert!(tree.query_aabb(bounds(0.0, 0.0, 0.0), |_| true));
}

#[test]
fn tree_query_miss_and_early_exit() {
    let mut tree = DynamicAabbTree::new();
    tree.insert(0, bounds(0.0, 0.0, 0.0));
    let mut hits = 0;
    assert!(tree.query_aabb(bounds(10.0, 0.0, 0.0), |_| {
        hits += 1;
        true
    }));
    assert_eq!(hits, 0);
    assert!(tree.query_ray(
        Ray {
            origin: Vec3::new(-2.0, 2.0, 0.0),
            direction: Vec3::X,
        },
        |_| false,
    ));
}

fn next_random(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state as f32 / u32::MAX as f32) * 40.0 - 20.0
}

#[test]
fn tree_1000_aabbs_match_naive_pairs() {
    let mut tree = DynamicAabbTree::new();
    let mut boxes = Vec::with_capacity(1000);
    let mut state = 0x39_u32;
    for index in 0..1000 {
        let aabb = bounds(
            next_random(&mut state),
            next_random(&mut state),
            next_random(&mut state),
        );
        tree.insert(index, aabb);
        boxes.push(aabb);
    }
    let mut expected = Vec::new();
    for a in 0..boxes.len() {
        for b in (a + 1)..boxes.len() {
            if fat_bounds(boxes[a]).overlaps(fat_bounds(boxes[b])) {
                expected.push((a, b));
            }
        }
    }
    assert_eq!(tree.compute_pairs(), expected.as_slice());
}

#[test]
fn tree_pair_order_is_insertion_order_independent() {
    let boxes: Vec<_> = (0..24)
        .map(|index| bounds(index as f32 * 0.3, 0.0, 0.0))
        .collect();
    let mut forward = DynamicAabbTree::new();
    for (index, aabb) in boxes.iter().copied().enumerate() {
        forward.insert(index, aabb);
    }
    let mut reverse = DynamicAabbTree::new();
    for (index, aabb) in boxes.iter().copied().enumerate().rev() {
        reverse.insert(index, aabb);
    }
    assert_eq!(forward.compute_pairs(), reverse.compute_pairs());
}

fn contact_scene(mode: BroadPhaseMode) -> World {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.broadphase_mode = mode;
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.6));
    for (z, x) in [(0.5, 0.0), (1.7, 0.02), (2.9, 0.0)] {
        let body = Body::solid_box(
            1.0,
            Vec3::splat(0.35),
            Vec3::new(x, 0.0, z),
            newt::math::Quat::IDENTITY,
        );
        let body_index = world.add_body(body);
        world.add_geom(Geom::r#box(
            body_index,
            Vec3::splat(0.35),
            Vec3::ZERO,
            newt::math::Quat::IDENTITY,
            0.6,
        ));
    }
    world
}

fn contact_counts(mut world: World, steps: usize) -> Vec<usize> {
    let mut counts = Vec::with_capacity(steps);
    for _ in 0..steps {
        world.step();
        counts.push(world.detect_contacts().len());
    }
    counts
}

#[test]
fn dynamic_tree_matches_naive_multi_body_contact_baseline() {
    // The naive mode preserves the pre-tree origin/main all-pairs baseline.
    let expected = contact_counts(contact_scene(BroadPhaseMode::Naive), 120);
    let actual = contact_counts(contact_scene(BroadPhaseMode::DynamicAabbTree), 120);
    assert_eq!(actual, expected);
}

#[test]
fn motion_inside_fat_bounds_does_not_reinsert() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::ZERO,
        newt::math::Quat::IDENTITY,
    ));
    world.bodies[body].linear_velocity = Vec3::new(0.01, 0.0, 0.0);
    world.add_geom(Geom::sphere(body, 0.1, Vec3::ZERO, 0.5));
    world.step();
    world.reset_broadphase_reinsert_count();
    world.step();
    assert_eq!(world.broadphase_reinsert_count(), 0);
}

#[test]
fn explicit_pair_list_bypasses_dynamic_tree() {
    let mut world = contact_scene(BroadPhaseMode::DynamicAabbTree);
    world.pair_list = Some(Vec::new());
    world.step();
    assert_eq!(world.broadphase_reinsert_count(), 0);
}
