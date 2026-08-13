//! Resting-contact anchor.
//!
//! A sphere dropped on a static plane must settle to the penetration where the
//! penalty spring force balances gravity, and it must STAY settled — no
//! long-term energy growth from the penalty integration. The closed-form
//! equilibrium penetration `δ_eq = m g / k` is hard-coded below and comes
//! straight from the model description in `docs/contacts.md`.
//!
//! Mutation coverage from this test:
//!  - flipped contact normal: sphere would receive a downward push and fall
//!    through the plane; the settled-z assertion fails immediately.
//!  - friction applied along the normal: at rest, tangential velocity is zero
//!    so this test does not discriminate friction bugs. See
//!    `contacts_friction.rs`.
//!  - missing damping term: after the drop the sphere would overshoot and
//!    oscillate for many steps; the second-half stddev bound fails.

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::world::World;

#[test]
fn sphere_on_plane_settles_to_equilibrium_penetration() {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    let mass = 1.0;
    let radius = 0.5;
    let body_idx = world.add_body(Body::solid_sphere(
        mass,
        radius,
        // Start slightly above the plane so we truly integrate the drop.
        Vec3::new(0.0, 0.0, 0.6),
        Quat::IDENTITY,
    ));

    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
    world.add_geom(Geom::sphere(body_idx, radius, Vec3::ZERO, 0.5));

    // Analytic equilibrium: δ_eq = m g / k. `k` is hand-derived from the
    // documented formula `k = m_eff / timeconst²` with `timeconst = 0.02`
    // (SolRef::DEFAULT) and `m_eff = mass` (body-vs-static). Written as a
    // constant rather than calling `solref_to_kc` on purpose — an
    // impl-derived expectation is self-comparison and would let a
    // wrong-power mutant (`k = m / timeconst³` etc.) survive both sides of
    // the assertion.
    let k_expected: f32 = mass / (0.02 * 0.02);
    let delta_eq: f32 = mass * 9.81 / k_expected;
    let z_eq: f32 = radius - delta_eq;

    // Long enough to bleed off the drop's kinetic energy under critical
    // damping. 5000 steps @ 5 ms = 25 s.
    let n_steps = 5000usize;
    // Buffer z samples in the second half to compute a stability bound.
    let mut second_half = Vec::with_capacity(n_steps / 2);
    for step in 0..n_steps {
        world.step();
        if step >= n_steps / 2 {
            second_half.push(world.bodies[body_idx].position.z);
        }
    }

    let final_z = world.bodies[body_idx].position.z;
    // Loose tolerance because RK4 on a stiff spring adds a small offset.
    assert!(
        (final_z - z_eq).abs() < 5.0e-4,
        "final z {final_z} away from equilibrium {z_eq} (δ_eq = {delta_eq})"
    );

    // Standard deviation over the second half stays tiny: this is the
    // no-bounce-growth check. If damping were missing or the sign were
    // wrong, the sphere would keep bouncing and this ballooned.
    let mean = second_half.iter().sum::<f32>() / (second_half.len() as f32);
    let variance = second_half
        .iter()
        .map(|z| (z - mean) * (z - mean))
        .sum::<f32>()
        / (second_half.len() as f32);
    let stddev = variance.sqrt();
    assert!(
        stddev < 5.0e-5,
        "settling stddev {stddev} — expected no bounce growth"
    );

    // Sanity: penetration is positive (bodies really are in contact, not
    // hovering just above the plane).
    let penetration = radius - final_z;
    assert!(
        penetration > 0.5 * delta_eq && penetration < 1.5 * delta_eq,
        "penetration {penetration} far from analytic δ_eq {delta_eq}"
    );
}
