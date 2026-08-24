use newt::body::Body;
use newt::geom::Geom;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{
    ConeKind, solve_free_bodies_diag_with_tolerance, solve_tree_contacts_with_tolerance,
};
use newt::tree::{Link, Tree};
use newt::world::World;

const MAX_ITERATIONS: u32 = 40;
const LOOSE_TOLERANCE: f32 = 1.0e6;

fn free_body_scene() -> World {
    let mut world = World::new();
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.0));
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(0.0, 0.0, 0.45),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.0));
    world
}

fn tree_scene() -> World {
    let mut world = World::new();
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.0));
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(0.0, 0.0, 0.45), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.1, 0.1, 0.1),
    ));
    let tree_index = world.add_tree(tree);
    world.add_geom(Geom::sphere_on_link(tree_index, 0, 0.5, Vec3::ZERO, 0.0));
    world
}

#[test]
fn both_pgs_paths_exit_on_the_same_iteration() {
    let free = free_body_scene();
    let free_contacts = free.detect_contacts();
    let (_, _, free_iterations) = solve_free_bodies_diag_with_tolerance(
        &free.bodies,
        &free.geoms,
        &free_contacts,
        &free.equalities,
        free.gravity,
        free.dt,
        ConeKind::Pyramidal,
        MAX_ITERATIONS,
        LOOSE_TOLERANCE,
    );

    let tree = tree_scene();
    let tree_contacts = tree.detect_contacts();
    let tree_solution = solve_tree_contacts_with_tolerance(
        &tree.bodies,
        &tree.trees,
        &tree.geoms,
        &tree_contacts,
        tree.gravity,
        tree.dt,
        ConeKind::Pyramidal,
        MAX_ITERATIONS,
        LOOSE_TOLERANCE,
        false,
        None,
    );

    assert_eq!(free_iterations, 1);
    assert_eq!(tree_solution.pgs_iterations, free_iterations);
    assert!(free_iterations < MAX_ITERATIONS);
}

#[test]
fn zero_tolerance_preserves_full_sweep_behavior() {
    let free = free_body_scene();
    let free_contacts = free.detect_contacts();
    let (_, _, free_iterations) = solve_free_bodies_diag_with_tolerance(
        &free.bodies,
        &free.geoms,
        &free_contacts,
        &free.equalities,
        free.gravity,
        free.dt,
        ConeKind::Pyramidal,
        MAX_ITERATIONS,
        0.0,
    );

    let tree = tree_scene();
    let tree_contacts = tree.detect_contacts();
    let tree_solution = solve_tree_contacts_with_tolerance(
        &tree.bodies,
        &tree.trees,
        &tree.geoms,
        &tree_contacts,
        tree.gravity,
        tree.dt,
        ConeKind::Pyramidal,
        MAX_ITERATIONS,
        0.0,
        false,
        None,
    );

    assert_eq!(free_iterations, MAX_ITERATIONS);
    assert_eq!(tree_solution.pgs_iterations, MAX_ITERATIONS);
}
