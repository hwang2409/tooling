//! Bouncing-sphere energy anchor: peaks strictly decrease.
//!
//! With an underdamped normal contact (`dampratio < 1`), each impact
//! dissipates energy. A correct penalty integration produces a monotonically
//! shrinking sequence of peak heights. The classical failure mode of
//! penalty-plus-explicit-integration is `energy GAIN` — a mutant that drops
//! the normal damping term, or that swaps the sign of the closing-velocity
//! damping, tends to pump energy into the sphere each bounce.
//!
//! Mutation coverage: missing normal damping term → peak heights either
//! remain flat or grow; assertion below fails on the first non-monotonic pair.

use newt::body::Body;
use newt::geom::{Geom, SolRef};
use newt::math::{Quat, Vec3};
use newt::world::World;

#[test]
fn bouncing_sphere_peaks_are_monotonically_decreasing() {
    let mut world = World::new();
    world.dt = 0.001;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    let radius = 0.5;
    let mass = 1.0;
    let start_height = 2.0;
    let body_idx = world.add_body(Body::solid_sphere(
        mass,
        radius,
        Vec3::new(0.0, 0.0, start_height),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
    // Underdamped so we get bounces (ζ = 0.05, near-elastic within the
    // penalty model). RK4 on a stiff penalty spring adds numerical damping;
    // ζ = 0.05 keeps enough restitution to see multiple bounces.
    world.add_geom(
        Geom::sphere(body_idx, radius, Vec3::ZERO, 0.5).with_solref(SolRef::new(0.02, 0.05)),
    );

    // Sample the trajectory. Detect peaks where vz crosses positive → negative.
    // At a peak, kinetic energy is at a minimum for that bounce and the height
    // is the total mechanical energy divided by m g. Successive peaks must
    // strictly decrease if energy is dissipated (or stay constant — but we
    // pick ζ > 0 so decrease is required).
    let n_steps = 6000usize;
    let mut peaks: Vec<f32> = Vec::new();
    let mut prev_vz = world.bodies[body_idx].linear_velocity.z;
    let mut prev_z = world.bodies[body_idx].position.z;
    for _ in 0..n_steps {
        world.step();
        let vz = world.bodies[body_idx].linear_velocity.z;
        let z = world.bodies[body_idx].position.z;
        // Peak: sign flip from + to non-positive, and above the settling
        // equilibrium (else post-settle noise fires spurious peaks).
        if prev_vz > 0.0 && vz <= 0.0 && prev_z > radius + 0.003 {
            peaks.push(prev_z);
        }
        prev_vz = vz;
        prev_z = z;
    }
    assert!(
        peaks.len() >= 3,
        "expected at least 3 detectable peaks; got {}: {:?}",
        peaks.len(),
        peaks
    );

    for k in 0..peaks.len() - 1 {
        assert!(
            peaks[k + 1] < peaks[k],
            "peak {k}→{kn} not decreasing: {p0} → {p1} (all peaks: {peaks:?})",
            kn = k + 1,
            p0 = peaks[k],
            p1 = peaks[k + 1],
        );
    }

    // First peak strictly below the start height — the drop deposited energy
    // into the penalty spring's damper. A "no damping" mutant would bounce
    // back to (or above) `start_height`.
    assert!(
        peaks[0] < start_height - 0.05,
        "first peak {} did not lose energy on impact — suspect missing damping",
        peaks[0]
    );
}
