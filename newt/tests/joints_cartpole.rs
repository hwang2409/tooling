//! Primary anchor for slide + hinge acting on the same tree: the classic
//! cart-pole system. A cart of mass M slides horizontally along +X; a
//! uniform-rod pole of mass m and length L is hinged to the cart about +Y
//! (the swing plane is x-z). Pole hangs straight down at θ = 0.
//!
//! # Reference EOM (hand-derived Lagrangian, executed in this test)
//!
//! Cart position `x`, pole angle `θ` measured about +Y from the downward
//! vertical (positive θ tilts the pole to −X, matching newt's Rot_Y(θ) on
//! the child body's `(0, 0, −L/2)` COM offset).
//!
//! Pole COM: `(x − (L/2) sin θ, 0, h − (L/2) cos θ)` with `h` the fixed
//! cart height (constant, cancels in dynamics).
//!
//! Kinetic energy: `T = ½(M+m) ẋ² − ½ m L ẋ cos θ · θ̇ + ½ (mL²/3) θ̇²`
//! (rod inertia about pivot = mL²/3 via parallel axis on the COM inertia
//! `mL²/12`).
//!
//! Potential energy (drop constants): `V = −(mgL/2) cos θ`.
//!
//! Euler-Lagrange gives, after canceling the `(m L/2) ẋ sin θ · θ̇` pair
//! in the θ equation:
//!
//! ```text
//! (M+m) ẍ − (m L/2) cos θ · θ̈ + (m L/2) sin θ · θ̇² = 0
//! (m L²/3) θ̈ − (m L/2) cos θ · ẍ + (m g L/2) sin θ = 0
//! ```
//!
//! Solving explicitly:
//!
//! ```text
//! ẍ = [ −(m L/2) sin θ · θ̇²  −  (3 m g / 4) sin θ cos θ ]
//!     / [ (M + m) − (3 m / 4) cos²θ ]
//! θ̈ = (3 / (2 L)) · ẍ · cos θ  −  (3 g / (2 L)) · sin θ
//! ```
//!
//! Test-local RK4 (scalar math only, no engine calls) integrates these
//! coupled ODEs and compares against the engine's `(x, θ)` trajectory over
//! ~2 s with non-degenerate ICs. Any wire-through-zero bug in the slide/
//! hinge combination — swapped axes, wrong Coriolis bias, sign errors in
//! Xup — diverges within a handful of steps.

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, rk4_step};

const CART_M: f32 = 1.2;
const POLE_M: f32 = 0.6;
const POLE_L: f32 = 0.8;
const G: f32 = 9.81;
const H: f32 = 1.0; // fixed cart height (world z of the cart COM)

// ---------------------------------------------------------------------------
// Reference: hand-coded cart-pole EOMs, integrated with test-local RK4.
// ---------------------------------------------------------------------------

fn cartpole_derivs(state: [f32; 4]) -> [f32; 4] {
    // state = [x, xdot, theta, thetadot]
    let x = state[0];
    let xdot = state[1];
    let theta = state[2];
    let thdot = state[3];
    let _ = x; // x itself is not in the EOM; used only for output.
    let s = theta.sin();
    let c = theta.cos();
    let xddot_num = -(POLE_M * POLE_L * 0.5) * s * thdot * thdot - (3.0 * POLE_M * G / 4.0) * s * c;
    let xddot_den = (CART_M + POLE_M) - (3.0 * POLE_M / 4.0) * c * c;
    let xddot = xddot_num / xddot_den;
    let thddot = (3.0 / (2.0 * POLE_L)) * xddot * c - (3.0 * G / (2.0 * POLE_L)) * s;
    [xdot, xddot, thdot, thddot]
}

fn cartpole_rk4_step(state: [f32; 4], dt: f32) -> [f32; 4] {
    let k1 = cartpole_derivs(state);
    let mut s2 = state;
    for i in 0..4 {
        s2[i] += 0.5 * dt * k1[i];
    }
    let k2 = cartpole_derivs(s2);
    let mut s3 = state;
    for i in 0..4 {
        s3[i] += 0.5 * dt * k2[i];
    }
    let k3 = cartpole_derivs(s3);
    let mut s4 = state;
    for i in 0..4 {
        s4[i] += dt * k3[i];
    }
    let k4 = cartpole_derivs(s4);
    let mut out = state;
    for i in 0..4 {
        out[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

// ---------------------------------------------------------------------------
// newt tree construction — cart on slide (X), pole on hinge (Y).
// ---------------------------------------------------------------------------

fn build_cartpole() -> Tree {
    let mut tree = Tree::new();
    // Root: fixed at world (0, 0, H).
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, H), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Cart: slide along +X, mass CART_M. Isotropic small inertia (cart
    // never rotates); COM sits at the slide anchor (child offset = 0).
    tree.push_link(Link::new(
        Some(0),
        JointKind::slide(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        CART_M,
        Mat3::diag(1e-4, 1e-4, 1e-4),
    ));
    // Pole: hinge about +Y, parent = cart. Joint anchor at cart's COM
    // (joint_offset_in_parent = 0), pole's joint anchor at (0, 0, L/2) in
    // child body so the COM sits L/2 below the pivot at θ = 0.
    // Uniform-rod inertia about the COM: (1/12) m L² perpendicular to the
    // rod's axis. Take rod along child +Z (like the pendulum demo), so the
    // large perpendicular components are Ixx and Iyy; Izz is nearly zero
    // (small ε to keep the tensor invertible).
    let i_perp = (1.0 / 12.0) * POLE_M * POLE_L * POLE_L;
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::Y),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, POLE_L * 0.5), Quat::IDENTITY),
        POLE_M,
        Mat3::diag(i_perp, i_perp, 1e-6),
    ));
    tree
}

fn zero_ext(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

// ---------------------------------------------------------------------------
// The anchor.
// ---------------------------------------------------------------------------

#[test]
fn cartpole_matches_hand_lagrangian_over_two_seconds() {
    // Non-degenerate ICs: cart at rest, pole tilted (θ = 0.35 rad ≈ 20°),
    // pole moving (θdot = 0.4 rad/s), cart moving (xdot = -0.15 m/s).
    // Angles < ~30° keep the pole away from the θ = π/2 singular sub-set
    // in the mass matrix (`cos²θ = 0` makes both sides sensitive to
    // numerical precision).
    let x0 = 0.0f32;
    let xdot0 = -0.15f32;
    let theta0 = 0.35f32;
    let thdot0 = 0.4f32;

    let mut tree = build_cartpole();
    tree.set_slide_position(1, x0);
    tree.set_slide_rate(1, xdot0);
    tree.set_hinge_angle(2, theta0);
    tree.set_hinge_rate(2, thdot0);

    let mut ref_state = [x0, xdot0, theta0, thdot0];

    // Match the engine's step size (5 ms) so RK4 error cancels equally.
    let dt = 0.005f32;
    let n_steps = 400usize; // 2.0 s
    let g_vec = Vec3::new(0.0, 0.0, -G);

    // Tolerance derivation: RK4 O(dt⁴) local, O(dt⁴) global for smooth
    // ODEs; with dt = 5 ms and the reference computed by an INDEPENDENT
    // RK4 twin (same scheme, different formulation), the two trajectories
    // agree to about O(dt²) or better on chaotic-adjacent motion. 5e-3
    // rad / 5e-3 m gives a safety factor of ~10 over what mutation probes
    // during development showed drift within 2 s.
    let pos_tol = 5.0e-3f32;
    let ang_tol = 5.0e-3f32;

    for step in 0..n_steps {
        rk4_step(&mut tree, g_vec, dt, zero_ext(3));
        ref_state = cartpole_rk4_step(ref_state, dt);
        let x_got = tree.slide_position(1);
        let th_got = tree.hinge_angle(2);
        let dx = (x_got - ref_state[0]).abs();
        let dth = (th_got - ref_state[2]).abs();
        assert!(
            dx < pos_tol,
            "cart x diverged at step {step} (t={}): newt={x_got} vs ref={} (|Δ|={dx})",
            step as f32 * dt,
            ref_state[0]
        );
        assert!(
            dth < ang_tol,
            "pole θ diverged at step {step} (t={}): newt={th_got} vs ref={} (|Δ|={dth})",
            step as f32 * dt,
            ref_state[2]
        );
    }
}

#[test]
fn cartpole_energy_conservation_no_damping() {
    // Cross-check: with no damping / no actuator, total mechanical energy
    // should drift by less than a few 1e-3 over 2 s. Catches Coriolis /
    // pA-update sign errors that a point-wise trajectory match can miss
    // if the twin has the same latent bug.
    let mut tree = build_cartpole();
    tree.set_slide_rate(1, 0.0);
    tree.set_hinge_angle(2, 0.6); // ~34° tilt, gives real swing
    let dt = 0.001f32;
    let g_vec = Vec3::new(0.0, 0.0, -G);

    let energy = |t: &Tree| -> f32 {
        let x = t.slide_position(1);
        let xdot = t.slide_rate(1);
        let th = t.hinge_angle(2);
        let thdot = t.hinge_rate(2);
        let _ = x;
        // `s` isn't used directly — energy needs only cos θ — but computing
        // it here documents the identity `s² + c² = 1` implicit in the KE
        // reduction and keeps the shape parallel to `cartpole_derivs`.
        let _s = th.sin();
        let c = th.cos();
        let ke = 0.5 * (CART_M + POLE_M) * xdot * xdot - 0.5 * POLE_M * POLE_L * xdot * c * thdot
            + 0.5 * (POLE_M * POLE_L * POLE_L / 3.0) * thdot * thdot;
        let pe = -(POLE_M * G * POLE_L * 0.5) * c;
        ke + pe
    };
    let e0 = energy(&tree);
    let mut e_max: f32 = e0;
    let mut e_min: f32 = e0;
    for _ in 0..2000 {
        rk4_step(&mut tree, g_vec, dt, zero_ext(3));
        let e = energy(&tree);
        if e > e_max {
            e_max = e;
        }
        if e < e_min {
            e_min = e;
        }
    }
    let drift = (e_max - e_min).abs();
    assert!(
        drift < 5e-3,
        "cart-pole energy drifted by {drift} (e0={e0}, min={e_min}, max={e_max})"
    );
}
