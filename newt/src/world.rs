//! World: free rigid bodies + geoms, penalty contacts, RK4 integration.
//!
//! # Scope (tier 2)
//!
//! Free bodies under uniform gravity plus contact forces from a penalty model.
//! Bodies, geoms, and contact pairs are index-stable across the simulation.
//!
//! # Contact model
//!
//! Full derivation lives in `docs/contacts.md`. In brief:
//!
//! - **Normal**: `f_n = max(0, k pen + c (−v_n))` where `v_n = (v_a − v_b) · n`
//!   is the relative velocity of the contact point along the normal. `k, c`
//!   come from the pair's [`crate::geom::SolRef`] and the reduced mass. `f_n`
//!   is clamped non-negative — the contact cannot pull.
//! - **Friction**: pyramidal, two tangent axes `(t1, t2)` picked
//!   deterministically from `n`. Per-axis force `f_ti = clamp(-c_t v_ti,
//!   −μ|f_n|, +μ|f_n|)` with `c_t = c_normal`. This is a viscous-with-Coulomb-
//!   clamp formulation: at rest, viscous force is small; under slip, it
//!   saturates at the Coulomb cap. The static/kinetic distinction is
//!   *effective*: with the pair's default stiffness, tangential drift at rest
//!   under a subcritical tangent load is small enough for the incline anchor.
//!
//! Contact forces are applied as world-frame wrenches at the contact point
//! and are recomputed at every RK4 sub-stage — that is the correct RK4 form
//! for a forced ODE. The tier-1 empty-geom path is preserved bit-for-bit
//! because every added term is `+ 0` when no contacts exist.
//!
//! # Determinism
//!
//! - Pair enumeration is total: sorted `(min, max)` index pairs, no HashMap
//!   iteration in the hot path.
//! - Narrow-phase output is fixed-order per pair (see [`crate::contact`]).
//! - Only `+ − * / sqrt` and hand-written transcendentals from
//!   [`crate::math`] appear in the compute path.

use crate::body::Body;
use crate::contact::{Contact, narrow_phase};
use crate::geom::{Geom, GeomPose, combine_solref, geom_world_pose, solref_to_kc};
use crate::math::{Quat, Vec3};

/// Simulation world.
#[derive(Clone, Debug, PartialEq)]
pub struct World {
    /// Fixed integration timestep. Default 5 ms (matches biped).
    pub dt: f32,
    /// Uniform gravity vector applied to every body's COM.
    pub gravity: Vec3,
    /// Bodies. Index-stable.
    pub bodies: Vec<Body>,
    /// Geoms. Index-stable. A geom's `body: Some(i)` refers to `bodies[i]`.
    pub geoms: Vec<Geom>,
    /// Optional explicit pair list `(geom_a, geom_b)` with `a < b`. When
    /// `None`, contact detection enumerates every unordered geom pair whose
    /// two geoms don't share a body and aren't both static; the resulting
    /// order is `(min, max)` lexicographic.
    pub pair_list: Option<Vec<(usize, usize)>>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            dt: 0.005,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: Vec::new(),
            geoms: Vec::new(),
            pair_list: None,
        }
    }

    /// Adds a body and returns its stable index.
    pub fn add_body(&mut self, body: Body) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Adds a geom and returns its stable index.
    pub fn add_geom(&mut self, geom: Geom) -> usize {
        let idx = self.geoms.len();
        self.geoms.push(geom);
        idx
    }

    /// Enumerate all valid contact pairs in canonical `(min, max)` order.
    /// Used when `pair_list` is `None`.
    fn auto_pairs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let n = self.geoms.len();
        for a in 0..n {
            for b in (a + 1)..n {
                let ga = &self.geoms[a];
                let gb = &self.geoms[b];
                if ga.body.is_none() && gb.body.is_none() {
                    continue; // static vs static: nothing to accelerate
                }
                if ga.body == gb.body {
                    continue; // same body: no self-contact
                }
                out.push((a, b));
            }
        }
        out
    }

    /// Advance the whole world by one fixed-dt RK4 step.
    ///
    /// Contact forces are recomputed at each RK4 sub-stage from the
    /// interpolated body states. This is the standard RK4 treatment for a
    /// forced ODE (Butcher, *Numerical Methods for ODEs*, §3.1) and is
    /// stable for the stiffness range v0 targets (`SolRef::DEFAULT` gives
    /// a contact period ≈ 125 ms; `dt = 5 ms` is 25 samples per period).
    /// Very stiff underdamped contacts still show some parasitic damping
    /// (the intermediate stages sample deeper penetrations than the true
    /// continuous solution reaches); the bouncing anchor picks parameters
    /// that keep the effective restitution comfortably above zero. With no
    /// geoms, this collapses to tier-1 gravity-only RK4 bit-for-bit, and
    /// the golden `tumbling_3_body.bin` still passes.
    pub fn step(&mut self) {
        let s0 = self.bodies.clone();
        let pairs = match &self.pair_list {
            Some(p) => p.clone(),
            None => self.auto_pairs(),
        };

        let ext1 = self.compute_wrenches(&s0, &pairs);
        let k1 = evaluate_all(&s0, self.gravity, &ext1);

        let s1 = advance_all(&s0, &s0, &k1, self.dt * 0.5);
        let ext2 = self.compute_wrenches(&s1, &pairs);
        let k2 = evaluate_all(&s1, self.gravity, &ext2);

        let s2 = advance_all(&s0, &s0, &k2, self.dt * 0.5);
        let ext3 = self.compute_wrenches(&s2, &pairs);
        let k3 = evaluate_all(&s2, self.gravity, &ext3);

        let s3 = advance_all(&s0, &s0, &k3, self.dt);
        let ext4 = self.compute_wrenches(&s3, &pairs);
        let k4 = evaluate_all(&s3, self.gravity, &ext4);

        for i in 0..self.bodies.len() {
            let state = s0[i];
            let sixth = 1.0 / 6.0;
            let dp =
                (k1[i].dposition + k2[i].dposition * 2.0 + k3[i].dposition * 2.0 + k4[i].dposition)
                    * sixth;
            let dv = (k1[i].dlinear_velocity
                + k2[i].dlinear_velocity * 2.0
                + k3[i].dlinear_velocity * 2.0
                + k4[i].dlinear_velocity)
                * sixth;
            let dq = (k1[i].dorientation
                + k2[i].dorientation * 2.0
                + k3[i].dorientation * 2.0
                + k4[i].dorientation)
                * sixth;
            let dw = (k1[i].dangular_velocity_body
                + k2[i].dangular_velocity_body * 2.0
                + k3[i].dangular_velocity_body * 2.0
                + k4[i].dangular_velocity_body)
                * sixth;

            self.bodies[i] = Body {
                position: state.position + dp * self.dt,
                linear_velocity: state.linear_velocity + dv * self.dt,
                orientation: (state.orientation + dq * self.dt).renormalize(),
                angular_velocity_body: state.angular_velocity_body + dw * self.dt,
                ..state
            };
        }
    }

    /// Public: detect all contacts against the current body state. Useful for
    /// tests that need to inspect contact geometry.
    pub fn detect_contacts(&self) -> Vec<Contact> {
        let pairs = match &self.pair_list {
            Some(p) => p.clone(),
            None => self.auto_pairs(),
        };
        collect_contacts(&self.bodies, &self.geoms, &pairs)
    }

    /// Compute per-body external wrench arrays for a given body-state vector.
    ///
    /// Returns `(force_world, torque_world_about_com)` for each body. Zero for
    /// bodies with no active contacts and (crucially) all-zero when
    /// `self.geoms` is empty, which preserves the tier-1 golden.
    fn compute_wrenches(&self, state: &[Body], pairs: &[(usize, usize)]) -> Vec<(Vec3, Vec3)> {
        let n = state.len();
        let mut out = vec![(Vec3::ZERO, Vec3::ZERO); n];
        if self.geoms.is_empty() {
            return out;
        }
        let contacts = collect_contacts(state, &self.geoms, pairs);
        for c in &contacts {
            apply_contact_wrench(&mut out, state, &self.geoms, c);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// contact assembly and force application
// ---------------------------------------------------------------------------

fn collect_contacts(state: &[Body], geoms: &[Geom], pairs: &[(usize, usize)]) -> Vec<Contact> {
    let mut out = Vec::new();
    // Pre-compute world poses for every geom in stable index order.
    let poses: Vec<GeomPose> = geoms
        .iter()
        .map(|g| match g.body {
            Some(i) => geom_world_pose(g, state[i].position, state[i].orientation),
            None => geom_world_pose(g, Vec3::ZERO, Quat::IDENTITY),
        })
        .collect();

    for &(a, b) in pairs {
        let buf = narrow_phase(a, &geoms[a], &poses[a], b, &geoms[b], &poses[b]);
        for c in buf.as_slice() {
            out.push(*c);
        }
    }
    out
}

/// Apply one contact's wrench to the appropriate body/bodies. Static geoms
/// (no owning body) simply absorb the reaction.
fn apply_contact_wrench(
    ext: &mut [(Vec3, Vec3)],
    state: &[Body],
    geoms: &[Geom],
    contact: &Contact,
) {
    let ga = &geoms[contact.geom_a];
    let gb = &geoms[contact.geom_b];
    let normal = contact.normal_world;

    // Reduced mass. Static geoms behave as infinite mass.
    let m_eff = match (ga.body, gb.body) {
        (Some(ia), Some(ib)) => {
            let ma = state[ia].mass;
            let mb = state[ib].mass;
            ma * mb / (ma + mb)
        }
        (Some(ia), None) => state[ia].mass,
        (None, Some(ib)) => state[ib].mass,
        (None, None) => return, // static-static: ignore (auto_pairs already skips)
    };

    let solref = combine_solref(ga.solref, gb.solref);
    let (k, c) = solref_to_kc(solref, m_eff);
    let c_tangent = c;

    // Point velocities at the contact.
    let (v_a, w_a_world, r_a) = point_velocity(state, ga, contact.position_world);
    let (v_b, w_b_world, r_b) = point_velocity(state, gb, contact.position_world);
    let _ = (w_a_world, w_b_world); // returned for completeness
    let v_rel = v_a - v_b;

    let v_n = v_rel.dot(normal);
    // Normal force magnitude: spring + damping opposing closing motion.
    // Clamped at zero — contacts cannot pull.
    let f_n_raw = k * contact.penetration - c * v_n;
    let f_n = if f_n_raw > 0.0 { f_n_raw } else { 0.0 };
    if f_n <= 0.0 {
        return;
    }

    // Deterministic tangent basis.
    let (t1, t2) = tangent_basis(normal);
    let v_t = v_rel - normal * v_n;
    let v_t1 = v_t.dot(t1);
    let v_t2 = v_t.dot(t2);

    let cap = contact.friction * f_n;
    let f_t1 = clamp_symmetric(-c_tangent * v_t1, cap);
    let f_t2 = clamp_symmetric(-c_tangent * v_t2, cap);

    let force_on_a = normal * f_n + t1 * f_t1 + t2 * f_t2;

    if let Some(ia) = ga.body {
        let (f, tau) = &mut ext[ia];
        *f += force_on_a;
        *tau += r_a.cross(force_on_a);
    }
    if let Some(ib) = gb.body {
        let force_on_b = -force_on_a;
        let (f, tau) = &mut ext[ib];
        *f += force_on_b;
        *tau += r_b.cross(force_on_b);
    }
}

/// World-frame linear velocity of the contact point on a geom's parent body.
/// Returns `(v_point_world, w_world, r_arm_world)` where `r_arm_world` is
/// the vector from the body's COM to the contact point. For static geoms
/// returns zero vectors and a zero arm.
fn point_velocity(state: &[Body], geom: &Geom, contact_pos_world: Vec3) -> (Vec3, Vec3, Vec3) {
    match geom.body {
        Some(i) => {
            let body = &state[i];
            let r = contact_pos_world - body.position;
            let w_world = body.angular_velocity_world();
            (body.linear_velocity + w_world.cross(r), w_world, r)
        }
        None => (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),
    }
}

/// Deterministic orthonormal tangent basis `(t1, t2)` perpendicular to a unit
/// normal `n`. Picks the reference axis (X or Z) whose alignment with `n` is
/// weakest so the cross product does not underflow.
pub fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    // Pick the world reference axis less aligned with `n`.
    let use_z_ref = crate::math::abs(n.z) < 0.9;
    let reference = if use_z_ref { Vec3::Z } else { Vec3::X };
    let t1 = reference.cross(n).normalize();
    let t2 = n.cross(t1);
    (t1, t2)
}

fn clamp_symmetric(x: f32, cap: f32) -> f32 {
    if x > cap {
        cap
    } else if x < -cap {
        -cap
    } else {
        x
    }
}

// ---------------------------------------------------------------------------
// RK4 stage helpers
// ---------------------------------------------------------------------------

/// Body-state derivative under gravity + one external world-frame wrench.
#[derive(Clone, Copy, Debug)]
struct Deriv {
    dposition: Vec3,
    dlinear_velocity: Vec3,
    dorientation: Quat,
    dangular_velocity_body: Vec3,
}

/// Body derivative given gravity and a world-frame external wrench.
///
/// Bit-identical to tier-1 `evaluate` when the wrench is `(ZERO, ZERO)`:
/// `Vec3::ZERO / mass = ZERO`, `gravity + ZERO = gravity`, and
/// `Vec3::ZERO − X = −X` per component in IEEE arithmetic.
fn evaluate(state: Body, gravity: Vec3, ext_force_world: Vec3, ext_torque_world: Vec3) -> Deriv {
    let dlin = gravity + ext_force_world / state.mass;

    let iw = state.inertia_body * state.angular_velocity_body;
    let gyroscopic = -state.angular_velocity_body.cross(iw);
    let tau_body = state.orientation.inverse_rotate(ext_torque_world);
    let dang_body = state.inertia_body_inverse * (tau_body + gyroscopic);

    let dq = state.orientation.derivative(state.angular_velocity_body);
    let dp = state.linear_velocity;

    Deriv {
        dposition: dp,
        dlinear_velocity: dlin,
        dorientation: dq,
        dangular_velocity_body: dang_body,
    }
}

fn evaluate_all(states: &[Body], gravity: Vec3, ext: &[(Vec3, Vec3)]) -> Vec<Deriv> {
    let mut out = Vec::with_capacity(states.len());
    for (i, &s) in states.iter().enumerate() {
        let (f, tau) = ext[i];
        out.push(evaluate(s, gravity, f, tau));
    }
    out
}

/// Advance a whole slice of bodies by (from_state + deriv * dt). `origin` is
/// the RK4 stage anchor — always `s0` in our loop.
fn advance_all(origin: &[Body], _from: &[Body], deriv: &[Deriv], dt: f32) -> Vec<Body> {
    // NOTE on `_from`: kept for signature symmetry; we always advance from
    // the anchor `origin` = s0 (standard RK4). The parameter is unused today
    // but reserved for future integrators that treat the stage state as a
    // proper linearization point.
    let mut out = Vec::with_capacity(origin.len());
    for (i, &state) in origin.iter().enumerate() {
        let d = deriv[i];
        out.push(Body {
            position: state.position + d.dposition * dt,
            linear_velocity: state.linear_velocity + d.dlinear_velocity * dt,
            orientation: state.orientation + d.dorientation * dt,
            angular_velocity_body: state.angular_velocity_body + d.dangular_velocity_body * dt,
            ..state
        });
    }
    out
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_basis_is_orthonormal_for_various_normals() {
        let normals = [
            Vec3::Z,
            -Vec3::Z,
            Vec3::X,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(0.1, 0.7, 0.7).normalize(),
        ];
        for &n in &normals {
            let (t1, t2) = tangent_basis(n);
            assert!((t1.length() - 1.0).abs() < 1e-5, "t1 not unit for {n:?}");
            assert!((t2.length() - 1.0).abs() < 1e-5, "t2 not unit for {n:?}");
            assert!(t1.dot(n).abs() < 1e-5, "t1 not ⟂ n for {n:?}");
            assert!(t2.dot(n).abs() < 1e-5, "t2 not ⟂ n for {n:?}");
            assert!(t1.dot(t2).abs() < 1e-5, "t1 not ⟂ t2 for {n:?}");
        }
    }

    #[test]
    fn empty_world_step_matches_tier1_single_body_free_fall() {
        // Confirm the refactor did not perturb the gravity-only path. Single
        // body under gravity for 100 steps: y' = v0 * t + ½ g t².
        let mut w = World::new();
        w.dt = 0.005;
        w.gravity = Vec3::new(0.0, 0.0, -10.0);
        let b = Body::solid_box(1.0, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY);
        w.add_body(b);
        for _ in 0..100 {
            w.step();
        }
        // t = 0.5 s. Expected z = ½ * (-10) * 0.25 = -1.25.
        let z = w.bodies[0].position.z;
        assert!((z - (-1.25)).abs() < 1e-3, "free-fall z {z}");
    }

    #[test]
    fn auto_pairs_skip_same_body_and_static_static() {
        let mut w = World::new();
        let bi = w.add_body(Body::solid_sphere(1.0, 0.5, Vec3::ZERO, Quat::IDENTITY));
        // Two static planes (both body=None) should not pair with each other.
        let sa = w.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
        let sb = w.add_geom(Geom::static_plane(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, 0.5));
        // Two spheres on the same body should not pair with each other.
        let g1 = w.add_geom(Geom::sphere(bi, 0.1, Vec3::new(-0.2, 0.0, 0.0), 0.5));
        let g2 = w.add_geom(Geom::sphere(bi, 0.1, Vec3::new(0.2, 0.0, 0.0), 0.5));
        let pairs = w.auto_pairs();
        assert!(!pairs.contains(&(sa, sb)));
        assert!(!pairs.contains(&(g1, g2)));
        // sphere-vs-plane pairs must be present.
        assert!(pairs.contains(&(sa, g1)));
    }

    #[test]
    fn evaluate_matches_tier1_when_wrench_zero() {
        // Discriminating case: an asymmetric-inertia body with nontrivial ω;
        // evaluate with zero external wrench must reproduce the tier-1 dω/dt
        // exactly (bit-identical is not required in a mutation test — a tight
        // numerical bound is — but the derivation is that they *are*
        // arithmetically identical up to `+ 0` and `0 − X = −X`).
        let mut b = Body::new(
            1.0,
            crate::math::Mat3::diag(1.0, 2.0, 3.0),
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.3), 0.7),
        );
        b.angular_velocity_body = Vec3::new(0.5, 1.3, -0.7);
        b.linear_velocity = Vec3::new(0.1, 0.2, 0.3);
        let d_new = evaluate(b, Vec3::new(0.0, 0.0, -9.81), Vec3::ZERO, Vec3::ZERO);
        // Recreate tier-1 form directly:
        let iw = b.inertia_body * b.angular_velocity_body;
        let gyro = -b.angular_velocity_body.cross(iw);
        let dang_ref = b.inertia_body_inverse * gyro;
        let diff = d_new.dangular_velocity_body - dang_ref;
        assert!(diff.length() < 1e-6, "wrench-free path diverged");
        assert_eq!(d_new.dlinear_velocity, Vec3::new(0.0, 0.0, -9.81));
    }
}
