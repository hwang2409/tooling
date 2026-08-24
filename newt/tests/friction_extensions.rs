use newt::body::Body;
use newt::geom::{AnisotropicFriction, Geom, SolRef};
use newt::joint::JointKind;
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::World;

fn solver_world(gravity: Vec3) -> World {
    solver_world_with_mode(gravity, SolverMode::Pgs)
}

fn solver_world_with_mode(gravity: Vec3, mode: SolverMode) -> World {
    let mut world = World::new();
    world.gravity = gravity;
    world.solver = SolverConfig {
        mode,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    world
}

fn sliding_box(gravity: Vec3, orientation: Quat) -> World {
    let mut world = solver_world(gravity);
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.2),
        Vec3::new(0.0, 0.0, 0.21),
        orientation,
    ));
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0));
    let mut geom = Geom::r#box(body, Vec3::splat(0.2), Vec3::ZERO, Quat::IDENTITY, 1.0);
    geom.friction_anisotropy = Some(AnisotropicFriction {
        axis_local: Vec3::X,
        along_axis_mu: 0.05,
        across_axis_mu: 1.0,
    });
    world.add_geom(geom);
    world
}

fn isotropic_box(gravity: Vec3, orientation: Quat) -> World {
    let mut world = solver_world(gravity);
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.2),
        Vec3::new(0.0, 0.0, 0.21),
        orientation,
    ));
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.2),
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
    ));
    world
}

#[test]
fn friction_isotropic_defaults_unchanged() {
    let mut world = isotropic_box(Vec3::new(2.0, 0.0, -9.81), Quat::IDENTITY);
    for _ in 0..100 {
        world.step();
    }
    assert_eq!(
        [
            world.bodies[0].position.x.to_bits(),
            world.bodies[0].position.y.to_bits(),
            world.bodies[0].position.z.to_bits(),
        ],
        [997023786, 3084178527, 1045213687]
    );
    assert_eq!(
        [
            world.bodies[0].linear_velocity.x.to_bits(),
            world.bodies[0].linear_velocity.y.to_bits(),
            world.bodies[0].linear_velocity.z.to_bits(),
        ],
        [981061749, 3104184134, 891028685]
    );
}

#[test]
fn friction_anisotropic_along_slippier() {
    let along = sliding_box(Vec3::new(2.0, 0.0, -9.81), Quat::IDENTITY);
    let across = sliding_box(Vec3::new(0.0, 2.0, -9.81), Quat::IDENTITY);
    let mut along = along;
    let mut across = across;
    for _ in 0..200 {
        along.step();
        across.step();
    }
    assert!(
        along.bodies[0].position.x > across.bodies[0].position.y + 0.02,
        "along displacement should exceed across displacement: {} vs {}",
        along.bodies[0].position.x,
        across.bodies[0].position.y
    );
}

#[test]
fn friction_anisotropic_world_frame_rotation() {
    let rotation = Quat::from_axis_angle(Vec3::Z, newt::math::PI * 0.5);
    let mut world = sliding_box(Vec3::new(0.0, 2.0, -9.81), rotation);
    for _ in 0..200 {
        world.step();
    }
    assert!(
        world.bodies[0].position.y > 0.02,
        "rotated local axis should slip along world y: {}",
        world.bodies[0].position.y
    );
}

fn spinning_ball(rolling_friction: Option<f32>, height: f32) -> World {
    let mut world = solver_world(Vec3::new(0.0, 0.0, -9.81));
    let radius = 0.2;
    let body = world.add_body(Body::solid_sphere(
        1.0,
        radius,
        Vec3::new(0.0, 0.0, height),
        Quat::IDENTITY,
    ));
    world.bodies[body].rolling_friction = rolling_friction;
    world.bodies[body].angular_velocity_body = Vec3::new(0.0, 0.1, 0.0);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0));
    world.add_geom(Geom::sphere(body, radius, Vec3::ZERO, 1.0));
    world
}

fn newton_spinning_ball(angular_velocity_y: f32) -> World {
    let mut world = solver_world_with_mode(Vec3::new(0.0, 0.0, -9.81), SolverMode::Newton);
    let solref = SolRef::new(-400.0, -1200.0);
    let radius = 0.2;
    let body = world.add_body(Body::solid_sphere(
        1.0,
        radius,
        Vec3::new(0.0, 0.0, 0.1),
        Quat::IDENTITY,
    ));
    world.bodies[body].rolling_friction = Some(10.0);
    world.bodies[body].angular_velocity_body = Vec3::new(0.0, angular_velocity_y, 0.0);
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
    plane.solref = solref;
    world.add_geom(plane);
    let mut sphere = Geom::sphere(body, radius, Vec3::ZERO, 1.0);
    sphere.solref = solref;
    world.add_geom(sphere);
    world
}

fn newton_tree_spinning_ball(angular_velocity_y: f32) -> World {
    let mut world = solver_world_with_mode(Vec3::ZERO, SolverMode::Newton);
    let solref = SolRef::new(-400.0, -1200.0);
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        newt::math::Mat3::diag(0.1, 0.1, 0.1),
    ));
    let tree_index = world.add_tree(tree);
    let mut ground = Geom::box_on_link(
        tree_index,
        0,
        Vec3::new(1.0, 1.0, 0.05),
        Vec3::new(0.0, 0.0, -0.05),
        Quat::IDENTITY,
        1.0,
    );
    ground.solref = solref;
    world.add_geom(ground);
    let half = Vec3::splat(0.2);
    let body = world.add_body(Body::solid_box(
        1.0,
        half,
        Vec3::new(0.0, 0.0, 0.1),
        Quat::IDENTITY,
    ));
    world.bodies[body].rolling_friction = Some(10.0);
    world.bodies[body].angular_velocity_body = Vec3::new(0.0, angular_velocity_y, 0.0);
    let mut geom = Geom::r#box(body, half, Vec3::ZERO, Quat::IDENTITY, 1.0);
    geom.solref = solref;
    world.add_geom(geom);
    world
}

#[test]
fn friction_rolling_slows_spinning_ball() {
    let mut with_rolling = spinning_ball(Some(0.5), 0.2);
    let mut without_rolling = spinning_ball(None, 0.2);
    let mut previous = with_rolling.bodies[0].angular_velocity_world().length();
    for _ in 0..200 {
        with_rolling.step();
        without_rolling.step();
        let current = with_rolling.bodies[0].angular_velocity_world().length();
        assert!(current <= previous + 1.0e-5, "rolling speed increased");
        previous = current;
    }
    let control = without_rolling.bodies[0].angular_velocity_world().length();
    assert!(
        control - previous > 0.01,
        "rolling friction did not add slowdown"
    );
}

#[test]
fn friction_rolling_zero_normal_no_effect() {
    let mut with_normal = spinning_ball(Some(0.5), 0.2);
    let mut without_normal = spinning_ball(Some(0.5), 1.0);
    for world in [&mut with_normal, &mut without_normal] {
        world.geoms[0].friction = 0.0;
        world.geoms[1].friction = 0.0;
    }
    let initial = with_normal.bodies[0].angular_velocity_world().length();
    for _ in 0..20 {
        with_normal.step();
        without_normal.step();
    }
    let with_normal_speed = with_normal.bodies[0].angular_velocity_world().length();
    let without_normal_speed = without_normal.bodies[0].angular_velocity_world().length();
    assert!(with_normal_speed < initial - 0.001);
    assert_eq!(without_normal_speed.to_bits(), initial.to_bits());
}

#[test]
fn friction_rolling_clamp_no_reverse() {
    let mut world = spinning_ball(Some(10.0), 0.1);
    for _ in 0..200 {
        world.step();
    }
    let omega = world.bodies[0].angular_velocity_world();
    assert!(omega.y >= -1.0e-5);
    assert!(
        omega.length() < 1.0e-5,
        "angular velocity was not clamped: {omega:?}"
    );
}

#[test]
fn friction_rolling_free_body_newton_clamp_no_reverse() {
    for initial_spin in [0.1, -0.1] {
        let mut world = newton_spinning_ball(initial_spin);
        world.step();
        let omega = world.bodies[0].angular_velocity_world();
        assert!(omega.y.signum() == initial_spin.signum() || omega.y == 0.0);
        assert!(
            omega.length() < 2.0e-7,
            "initial spin {initial_spin}: angular velocity was not clamped: {omega:?}"
        );
    }
}

#[test]
fn friction_rolling_tree_newton_clamp_no_reverse() {
    for initial_spin in [0.1, -0.1] {
        let mut world = newton_tree_spinning_ball(initial_spin);
        world.step();
        assert!(
            world.detect_contacts().len() >= 2,
            "tree regression fixture must produce multiple contacts"
        );
        let omega = world.bodies[0].angular_velocity_world();
        assert!(omega.y.signum() == initial_spin.signum() || omega.y == 0.0);
        assert!(
            omega.length() < 2.0e-7,
            "initial spin {initial_spin}: angular velocity was not clamped: {omega:?}"
        );
    }
}
