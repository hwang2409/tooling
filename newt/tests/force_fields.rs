use newt::body::Body;
use newt::geom::Geom;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::SolverMode;
use newt::tree::{Link, Tree};
use newt::world::{ForceField, Integrator, Plane, RadialFalloff, World};

fn free_body(position: Vec3) -> Body {
    Body::solid_box(1.0, Vec3::splat(0.5), position, Quat::IDENTITY)
}

fn assert_downward_field_reacts(world: &mut World, body: usize, field: usize, magnitude: f32) {
    let mut control = world.clone();
    assert!(control.remove_force_field(field));
    let ignored_velocity = -magnitude * world.dt;
    world.step();
    control.step();
    let field_velocity = world.bodies[body].linear_velocity.z;
    let control_velocity = control.bodies[body].linear_velocity.z;
    assert!(
        field_velocity - control_velocity > ignored_velocity * 0.5,
        "contact solver ignored the downward field: field={field_velocity}, control={control_velocity}"
    );
}

#[test]
fn force_field_uniform_wind() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.dt = 0.1;
    let body = world.add_body(free_body(Vec3::ZERO));
    world.add_force_field(ForceField::Uniform {
        direction: Vec3::X,
        magnitude: 2.0,
    });

    world.step();

    assert!(world.bodies[body].position.x > 0.0);
    assert!(world.bodies[body].linear_velocity.x > 0.0);
}

#[test]
fn force_field_radial_inverse_square_attracts() {
    fn acceleration_at(distance: f32) -> f32 {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        world.integrator = newt::world::Integrator::Euler;
        world.dt = 0.01;
        let body = world.add_body(free_body(Vec3::new(distance, 0.0, 0.0)));
        world.add_force_field(ForceField::Radial {
            center: Vec3::ZERO,
            magnitude: -10.0,
            falloff: RadialFalloff::InverseSquare,
        });
        world.step();
        world.bodies[body].linear_velocity.x / world.dt
    }

    let near = acceleration_at(2.0);
    let far = acceleration_at(4.0);

    assert!(near < 0.0 && far < 0.0);
    assert!((near / far - 4.0).abs() < 1.0e-5);
}

#[test]
fn force_field_radial_linear_range_clips() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.dt = 0.1;
    let body = world.add_body(free_body(Vec3::new(5.0, 0.0, 0.0)));
    world.add_force_field(ForceField::Radial {
        center: Vec3::ZERO,
        magnitude: -10.0,
        falloff: RadialFalloff::Linear { max_range: 2.0 },
    });
    let initial = world.bodies[body];

    world.step();

    assert_eq!(world.bodies[body], initial);
}

fn buoyant_box(fluid_density: f32) -> World {
    let mut world = World::new();
    world.integrator = newt::world::Integrator::Euler;
    world.dt = 0.01;
    let body = world.add_body(free_body(Vec3::new(0.0, 0.0, -2.0)));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    world.add_force_field(ForceField::Buoyancy {
        plane: Plane::new(Vec3::Z, 0.0),
        fluid_density,
        gravity: world.gravity,
    });
    world
}

fn buoyant_offset_box(offset_z: f32) -> World {
    let mut world = World::new();
    world.integrator = Integrator::Euler;
    world.dt = 0.01;
    let body = world.add_body(free_body(Vec3::ZERO));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.5),
        Vec3::new(0.0, 0.0, offset_z),
        Quat::IDENTITY,
        0.0,
    ));
    world.add_force_field(ForceField::Buoyancy {
        plane: Plane::new(Vec3::Z, 0.0),
        fluid_density: 2.0,
        gravity: world.gravity,
    });
    world
}

#[test]
fn force_field_buoyancy_floats_less_dense() {
    let mut world = buoyant_box(2.0);

    world.step();

    let expected_accel = world.gravity * (1.0 - 2.0 / 1.0);
    let actual_accel = world.bodies[0].linear_velocity.z / world.dt;
    assert!((actual_accel - expected_accel.z).abs() < 1.0e-5);
}

#[test]
fn force_field_buoyancy_sinks_more_dense() {
    let mut world = buoyant_box(0.5);

    world.step();

    let expected_accel = world.gravity * (1.0 - 0.5 / 1.0);
    let actual_accel = world.bodies[0].linear_velocity.z / world.dt;
    assert!((actual_accel - expected_accel.z).abs() < 1.0e-5);
}

#[test]
fn force_field_buoyancy_uses_geom_offset_for_submersion() {
    let mut dry = buoyant_offset_box(2.0);
    dry.step();
    let dry_accel = dry.bodies[0].linear_velocity.z / dry.dt;
    assert!((dry_accel - dry.gravity.z).abs() < 1.0e-5);

    let mut submerged = buoyant_offset_box(-2.0);
    submerged.step();
    let submerged_accel = submerged.bodies[0].linear_velocity.z / submerged.dt;
    let expected_accel = submerged.gravity.z * (1.0 - 2.0 / 1.0);
    assert!((submerged_accel - expected_accel).abs() < 1.0e-5);
}

#[test]
fn force_field_buoyancy_offset_geom_applies_torque_at_center_of_buoyancy() {
    let mut world = World::new();
    world.integrator = Integrator::Euler;
    world.dt = 0.01;
    let body = world.add_body(free_body(Vec3::new(0.0, 0.0, -2.0)));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.5),
        Vec3::new(1.0, 0.0, 0.0),
        Quat::IDENTITY,
        0.0,
    ));
    world.add_force_field(ForceField::Buoyancy {
        plane: Plane::new(Vec3::Z, 0.0),
        fluid_density: 2.0,
        gravity: world.gravity,
    });

    world.step();

    assert!(world.bodies[body].angular_velocity_world().y < -1.0e-3);
}

#[test]
fn force_field_buoyancy_fully_submerged_symmetric_body_has_no_torque() {
    let mut world = buoyant_box(2.0);

    world.step();

    assert!(world.bodies[0].angular_velocity_world().length() < 1.0e-6);
}

fn partial_buoyancy_box(position: Vec3, orientation: Quat) -> World {
    let mut world = World::new();
    world.integrator = Integrator::Euler;
    world.dt = 0.01;
    let body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        position,
        orientation,
    ));
    world.add_geom(Geom::r#box(
        body,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    world.add_force_field(ForceField::Buoyancy {
        plane: Plane::new(Vec3::Z, 0.0),
        fluid_density: 2.0,
        gravity: world.gravity,
    });
    world
}

#[test]
fn force_field_buoyancy_axis_aligned_box_uses_clipped_volume() {
    let mut world = partial_buoyancy_box(Vec3::new(0.0, 0.0, 0.25), Quat::IDENTITY);

    world.step();

    let actual_accel = world.bodies[0].linear_velocity.z / world.dt;
    assert!((actual_accel + 4.905).abs() < 1.0e-5);
}

#[test]
fn force_field_buoyancy_rotated_box_uses_clipped_volume() {
    let diagonal = 1.0 / 3.0_f32.sqrt();
    let local_normal = Vec3::splat(diagonal);
    let orientation = Quat::new(diagonal, -diagonal, 0.0, 1.0 + diagonal).renormalize();
    let tetra_leg = 0.9085603;
    let center_z = (1.5 - tetra_leg) / 3.0_f32.sqrt();
    let mut world = partial_buoyancy_box(Vec3::new(0.0, 0.0, center_z), orientation);
    assert!((orientation.rotate(local_normal) - Vec3::Z).length() < 1.0e-5);

    world.step();

    let actual_accel = world.bodies[0].linear_velocity.z / world.dt;
    assert!((actual_accel + 7.3575).abs() < 2.0e-4);
}

#[test]
fn force_field_buoyancy_sphere_uses_spherical_cap() {
    let mut world = World::new();
    world.integrator = Integrator::Euler;
    world.dt = 0.01;
    let body = world.add_body(Body::solid_sphere(1.0, 0.5, Vec3::ZERO, Quat::IDENTITY));
    world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.0));
    world.add_force_field(ForceField::Buoyancy {
        plane: Plane::new(Vec3::Z, 0.0),
        fluid_density: 2.0,
        gravity: world.gravity,
    });

    world.step();

    let submerged_volume = (2.0 / 3.0) * std::f32::consts::PI * 0.5_f32.powi(3);
    let expected_accel = world.gravity.z * (1.0 - 2.0 * submerged_volume);
    let actual_accel = world.bodies[0].linear_velocity.z / world.dt;
    assert!((actual_accel - expected_accel).abs() < 1.0e-5);
}

#[test]
fn force_field_remove_stops_effect() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.dt = 0.1;
    world.add_body(free_body(Vec3::ZERO));
    let id = world.add_force_field(ForceField::Uniform {
        direction: Vec3::X,
        magnitude: 2.0,
    });

    world.step();
    let velocity_after_field = world.bodies[0].linear_velocity;
    assert!(velocity_after_field.x > 0.0);
    assert!(world.remove_force_field(id));
    assert!(!world.remove_force_field(id));

    world.step();

    assert_eq!(world.bodies[0].linear_velocity, velocity_after_field);
}

#[test]
fn force_field_none_registered_byte_identical() {
    let mut world = World::new();
    world.add_body(free_body(Vec3::new(0.0, 0.0, 5.0)));
    for _ in 0..10 {
        world.step();
    }

    let mut actual = Vec::new();
    for body in &world.bodies {
        for value in [
            body.position.x,
            body.position.y,
            body.position.z,
            body.orientation.x,
            body.orientation.y,
            body.orientation.z,
            body.orientation.w,
            body.linear_velocity.x,
            body.linear_velocity.y,
            body.linear_velocity.z,
            body.angular_velocity_body.x,
            body.angular_velocity_body.y,
            body.angular_velocity_body.z,
        ] {
            actual.extend_from_slice(&value.to_le_bytes());
        }
    }

    const ORIGIN_GOLDEN: &[u8] = &[
        0, 0, 0, 0, 0, 0, 0, 0, 141, 155, 159, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
        63, 0, 0, 0, 0, 0, 0, 0, 0, 211, 34, 251, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(actual, ORIGIN_GOLDEN);
}

fn free_body_contact_fixture(solver_mode: SolverMode) -> (World, usize, usize) {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.integrator = Integrator::Euler;
    world.dt = 0.005;
    world.solver.mode = solver_mode;
    world.solver.iterations = 40;
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.0));
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(0.0, 0.0, 0.49999),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.0));
    let field = world.add_force_field(ForceField::Uniform {
        direction: Vec3::new(0.0, 0.0, -1.0),
        magnitude: 9.81,
    });

    (world, body, field)
}

#[test]
fn force_field_free_body_pgs_contact_react() {
    let (mut world, body, field) = free_body_contact_fixture(SolverMode::Pgs);
    assert_downward_field_reacts(&mut world, body, field, 9.81);
}

#[test]
fn force_field_free_body_newton_contact_react() {
    let (mut world, body, field) = free_body_contact_fixture(SolverMode::Newton);
    assert_downward_field_reacts(&mut world, body, field, 9.81);
}

fn tree_contact_fixture(solver_mode: SolverMode) -> (World, usize, usize) {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.integrator = Integrator::Euler;
    world.dt = 0.005;
    world.solver.mode = solver_mode;
    world.solver.iterations = 40;

    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.1, 0.1, 0.1),
    ));
    let tree = world.add_tree(tree);
    world.add_geom(Geom::sphere_on_link(tree, 0, 0.5, Vec3::ZERO, 0.0));
    let body = world.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(0.0, 0.0, 0.49999),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::sphere(body, 0.5, Vec3::ZERO, 0.0));
    let field = world.add_force_field(ForceField::Uniform {
        direction: Vec3::new(0.0, 0.0, -1.0),
        magnitude: 9.81,
    });

    (world, body, field)
}

#[test]
fn force_field_tree_contact_pgs_react() {
    let (mut world, body, field) = tree_contact_fixture(SolverMode::Pgs);
    assert_downward_field_reacts(&mut world, body, field, 9.81);
}

#[test]
fn force_field_tree_contact_newton_react() {
    let (mut world, body, field) = tree_contact_fixture(SolverMode::Newton);
    assert_downward_field_reacts(&mut world, body, field, 9.81);
}
