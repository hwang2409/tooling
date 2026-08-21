use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::world::World;
use std::f32::consts::FRAC_PI_4;

fn box_plane_world() -> World {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 8,
        cone: ConeKind::Pyramidal,
    };
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0));
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::new(0.0, 0.0, 0.49),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
    ));
    world.set_contact_persistence(true);
    world
}

fn box_stack_world(persistence: bool) -> World {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 20,
        cone: ConeKind::Pyramidal,
    };
    world.set_contact_persistence(persistence);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0));
    for index in 0..5 {
        let body = world.add_body(Body::solid_box(
            1.0,
            Vec3::splat(0.5),
            Vec3::new(0.0, 0.0, 0.5 + index as f32),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::r#box(
            body,
            Vec3::splat(0.5),
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
        ));
    }
    world
}

#[test]
fn contact_persistence_disabled_by_default_byte_identical() {
    let mut default_world = World::new();
    let mut control_world = World::new();
    assert!(!default_world.contact_persistence);
    assert_eq!(default_world, control_world);
    for _ in 0..3 {
        default_world.step();
        control_world.step();
    }
    assert_eq!(default_world, control_world);
}

#[test]
fn contact_persistence_reduces_iteration_work() {
    let mut enabled = box_stack_world(true);
    let mut disabled = box_stack_world(false);
    let mut enabled_total = 0u64;
    let mut disabled_total = 0u64;
    for _ in 0..100 {
        enabled.step();
        disabled.step();
        enabled_total += u64::from(enabled.contact_solver_iterations());
        disabled_total += u64::from(disabled.contact_solver_iterations());
    }
    assert!(
        enabled_total < disabled_total,
        "enabled={enabled_total} disabled={disabled_total}"
    );
}

#[test]
fn contact_persistence_survives_small_position_shift() {
    let mut world = box_plane_world();
    let before = world.detect_contacts();
    world.step();
    world.bodies[0].position.x += 0.0001;
    let after = world.detect_contacts();
    world.step();
    assert_eq!(before[0].feature_id, after[0].feature_id);
    assert!(
        world
            .contact_persistence_initial_impulses()
            .iter()
            .any(|impulse| impulse.length_squared() > 0.0)
    );
}

#[test]
fn contact_persistence_manifold_reshuffle_falls_back() {
    let mut world = box_plane_world();
    world.step();
    let old_features: Vec<_> = world
        .detect_contacts()
        .iter()
        .map(|contact| contact.feature_id)
        .collect();
    world.bodies[0].orientation = Quat::from_axis_angle(Vec3::X, FRAC_PI_4 * 2.0);
    let new_features: Vec<_> = world
        .detect_contacts()
        .iter()
        .map(|contact| contact.feature_id)
        .collect();
    world.step();
    let seeds = world.contact_persistence_initial_impulses();
    assert_ne!(old_features, new_features);
    let mut unchanged_seed = false;
    let mut new_seed = false;
    for (feature, seed) in new_features.iter().zip(seeds) {
        if old_features.contains(feature) {
            unchanged_seed |= seed.length_squared() > 0.0;
        } else {
            new_seed |= seed.length_squared() == 0.0;
        }
    }
    assert!(unchanged_seed);
    assert!(new_seed);
}

#[test]
fn contact_persistence_stale_entries_are_evicted() {
    let mut world = box_plane_world();
    world.step();
    assert!(world.contact_persistence_cache_len() > 0);
    world.bodies[0].position.z = 3.0;
    for _ in 0..5 {
        world.step();
    }
    assert_eq!(world.contact_persistence_cache_len(), 0);
    world.solver.mode = SolverMode::Penalty;
    world.bodies[0].position.z = 0.49;
    world.step();
    assert_eq!(world.contact_persistence_cache_len(), 0);
    world.solver.mode = SolverMode::Pgs;
    world.step();
    assert_eq!(world.contact_persistence_hits(), 0);
}
