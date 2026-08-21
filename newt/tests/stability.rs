use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::solver::SolverMode;
use newt::world::World;

#[test]
fn stability_max_linear_velocity() {
    let mut capped = World::new();
    capped.gravity = Vec3::ZERO;
    let mut body = Body::solid_box(1.0, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY);
    body.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
    body.max_linear_velocity = Some(2.0);
    let capped_body = capped.add_body(body);
    for _ in 0..5 {
        capped.step();
    }
    assert_eq!(capped.bodies[capped_body].linear_velocity.length(), 2.0);

    let mut uncapped = World::new();
    uncapped.gravity = Vec3::ZERO;
    let mut body = Body::solid_box(1.0, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY);
    body.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
    let uncapped_body = uncapped.add_body(body);
    for _ in 0..5 {
        uncapped.step();
    }
    assert_eq!(
        uncapped.bodies[uncapped_body].linear_velocity.length(),
        10.0
    );
}

#[test]
fn stability_max_angular_velocity() {
    let mut capped = World::new();
    capped.gravity = Vec3::ZERO;
    let mut body = Body::solid_box(1.0, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY);
    body.angular_velocity_body = Vec3::new(10.0, 0.0, 0.0);
    body.max_angular_velocity = Some(2.0);
    let capped_body = capped.add_body(body);
    for _ in 0..5 {
        capped.step();
    }
    assert_eq!(
        capped.bodies[capped_body].angular_velocity_body.length(),
        2.0
    );

    let mut uncapped = World::new();
    uncapped.gravity = Vec3::ZERO;
    let mut body = Body::solid_box(1.0, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY);
    body.angular_velocity_body = Vec3::new(10.0, 0.0, 0.0);
    let uncapped_body = uncapped.add_body(body);
    for _ in 0..5 {
        uncapped.step();
    }
    assert_eq!(
        uncapped.bodies[uncapped_body]
            .angular_velocity_body
            .length(),
        10.0
    );
}

#[test]
fn stability_gravity_scale() {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, -10.0, 0.0);
    let mut ignores_gravity = Body::solid_box(1.0, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY);
    ignores_gravity.gravity_scale = 0.0;
    let still_body = world.add_body(ignores_gravity);
    let falling_body = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
    ));

    for _ in 0..10 {
        world.step();
    }

    assert_eq!(world.bodies[still_body].position.y, 0.0);
    assert!(world.bodies[falling_body].position.y < -0.001);
}

#[test]
fn stability_penetration_slop() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.penetration_slop = 0.02;
    world.solver.mode = SolverMode::Pgs;

    let lower = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
    ));
    let upper = world.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::new(0.0, 0.0, 0.99),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        lower,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    world.add_geom(Geom::r#box(
        upper,
        Vec3::splat(0.5),
        Vec3::ZERO,
        Quat::IDENTITY,
        0.0,
    ));
    world.pair_list = Some(vec![(0, 1)]);

    let shallow = world.detect_contacts();
    assert!(!shallow.is_empty());
    assert!(shallow.iter().all(|contact| contact.gap == 0.02));
    world.step();
    assert_eq!(world.bodies[lower].linear_velocity, Vec3::ZERO);
    assert_eq!(world.bodies[upper].linear_velocity, Vec3::ZERO);

    world.bodies[upper].position.z = 0.96;
    world.step();
    assert!(world.bodies[upper].linear_velocity.length() > 0.0);
}
