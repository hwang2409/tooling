//! General-actuator anchors (v2 tier 2).
//!
//! Expected values are hand-derived from:
//!   * the migration anchor — the Position flavor's PD math is identical
//!     to the v0 PdServo, so a general actuator with position-shaped
//!     parameters must reproduce the shorthand's trajectory to within
//!     f32 associativity residual.
//!   * closed-form second-order responses (equilibrium under gravity for
//!     the velocity-actuator holding case).
//!   * closed-form recurrence for the first-order filter integrator
//!     (forward Euler with `alpha = dt/tau`).
//!   * clamp ordering (ctrl clamp first, force clamp last).

use newt::actuator::{Actuator, ActuatorFlavor, BiasType, DynType, GainType};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, rk4_step};

fn build_hinge(mass: f32, length: f32) -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, length), Quat::IDENTITY),
        mass,
        Mat3::diag(1e-6, 1e-6, 1e-6),
    ));
    tree
}

fn zero_wrenches(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

/// Migration anchor: the arm golden trajectory driven by three general
/// actuators with position-shaped parameters (`gain=Fixed(kp)`,
/// `bias=Affine(0,-kp,-kv)`, `gear=1`, `dyn=None`) must agree with the
/// same trajectory driven by the Position flavor at every step, within
/// a tight f32-associativity bound.
///
/// This is NOT the byte-identical golden — that lives in
/// `tests/actuators_golden.rs` and uses the Position shorthand directly
/// so the v0 goldens stay bit-equal. This test proves the SEMANTICS
/// match end-to-end: an MJCF `<general>` actuator with position-shape
/// parameters simulates equivalently to `<position>` up to f32 rounding.
#[test]
fn general_position_shape_matches_shorthand_arm() {
    fn build_arm() -> Tree {
        let masses = [1.1f32, 0.7, 0.9];
        let lengths = [0.5f32, 0.4, 0.35];
        let axes = [Vec3::X, Vec3::new(1.0, 0.2, 0.0).normalize(), Vec3::X];
        let mut tree = Tree::new();
        tree.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        for i in 0..3 {
            let l = lengths[i];
            let m = masses[i];
            let i_perp = (1.0 / 12.0) * m * l * l;
            let parent_anchor = if i == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(0.0, 0.0, -lengths[i - 1] * 0.5)
            };
            tree.push_link(Link::new(
                Some(i),
                JointKind::hinge(axes[i]),
                (parent_anchor, Quat::IDENTITY),
                (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
                m,
                Mat3::diag(i_perp, i_perp, 1e-6),
            ));
        }
        tree
    }
    let kps = [80.0f32, 60.0, 40.0];
    let zetas = [0.8f32, 0.9, 1.0];
    let irs = [0.3f32, 0.15, 0.10];
    let clamps = [10.0f32, 8.0, 6.0];

    // Baseline: three Position-flavor actuators (byte-identical to v0 PdServo).
    let mut baseline = build_arm();
    for i in 0..3 {
        baseline.add_actuator(Actuator::position_from_dampratio(
            i + 1,
            kps[i],
            zetas[i],
            irs[i],
            clamps[i],
        ));
    }

    // General-flavor twin: same numerical parameters routed through the
    // general gain/bias formula.
    let mut general = build_arm();
    for i in 0..3 {
        // kv = 2·ζ·√(kp·I_ref) — same derivation the position shortcut uses.
        let kv = 2.0 * zetas[i] * (kps[i] * irs[i]).sqrt();
        general.add_actuator(Actuator::general(
            i + 1,
            GainType::Fixed,
            [kps[i], 0.0, 0.0],
            BiasType::Affine,
            [0.0, -kps[i], -kv],
            1.0,
            DynType::None,
            [1.0],
            None,
            // Symmetric force range using the same magnitude as the
            // Position shorthand. `clamp_range` is bit-equal to
            // `clamp_symmetric` for finite inputs (proved in
            // `src/actuator.rs::clamp_range_matches_clamp_symmetric`).
            Some((-clamps[i], clamps[i])),
        ));
    }

    let dt = 0.005f32;
    let g = Vec3::new(0.0, 0.0, -9.81);

    // Sample divergence per-step (not just at the end). Transient blooms
    // — the moment right after a target flip, when the PD error is
    // largest and the clamp engages — must stay bounded too, or a
    // semantic mismatch that self-cancels at steady state would slip
    // through. Two target flips exercise both the initial acceleration
    // burst AND clamp engagement mid-run.
    let mut max_err = 0.0f32;
    let targets: [[f32; 3]; 2] = [[0.3, -0.4, 0.5], [-0.5, 0.6, -0.2]];
    for flip in targets.iter() {
        for (a, &ctrl) in flip.iter().enumerate() {
            baseline.set_actuator_target(a, ctrl);
            general.set_actuator_target(a, ctrl);
        }
        for _ in 0..500 {
            rk4_step(&mut baseline, g, dt, zero_wrenches(4));
            rk4_step(&mut general, g, dt, zero_wrenches(4));
            for (a, b) in baseline.q.iter().zip(general.q.iter()) {
                max_err = max_err.max((a - b).abs());
            }
            for (a, b) in baseline.qdot.iter().zip(general.qdot.iter()) {
                max_err = max_err.max((a - b).abs());
            }
        }
    }
    // f32 associativity residual over 1000 RK4 steps × 4 stages × 3
    // actuators × ~5 float ops per eval sits comfortably under 1e-4 —
    // this bound catches a semantic mismatch (wrong sign, wrong
    // parameter mapping) which would blow to ~kp * ctrl scale (~10).
    assert!(
        max_err < 1.0e-4,
        "general vs position divergence too large across per-step samples: {max_err}"
    );
}

/// Velocity actuator holding against gravity. A P-only velocity servo
/// with `ctrl=0` (target rate = 0) applied to a hanging pendulum with
/// nonzero rest gravity torque should settle into a nonzero equilibrium
/// angle where:
///
/// ```text
/// 0 = -kv * qdot_ss - m·g·L·sin(q_ss)
/// ```
///
/// but `qdot_ss = 0` at rest, so the ONLY torque against gravity is
/// zero — the pendulum swings freely. To probe the actuator, we drive
/// the target rate to a positive value and require the pendulum to
/// track it near the equilibrium `q_ss` where gravity torque balances
/// the velocity error term:
///
/// ```text
/// kv * (ctrl - qdot_ss) = m·g·L·sin(q_ss)
/// ```
///
/// With sustained ctrl > 0 the pendulum spins forever, so we test the
/// steady-state qdot lag: after a long enough transient, the residual
/// (ctrl - qdot) is `m·g·L·sin(q)/kv` ≤ `m·g·L/kv` regardless of q.
/// For kv=50, m·g·L = 9.81 this puts the lag envelope at 0.1962 rad/s.
#[test]
fn velocity_actuator_tracks_target_rate_under_gravity() {
    let mut tree = build_hinge(1.0, 1.0);
    let a = tree.add_actuator(Actuator::velocity(1, /*kv*/ 50.0, /*clamp*/ 0.0));
    tree.set_actuator_target(a, 1.5); // 1.5 rad/s target
    // Add a bit of joint damping so early transients decay (isolates the
    // velocity actuator's tracking behavior from oscillatory startup).
    if let JointKind::Hinge {
        ref mut damping, ..
    } = tree.links[1].joint
    {
        *damping = 0.5;
    }
    let dt = 0.001f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    // Warm up 4 s to shed the initial acceleration transient.
    for _ in 0..4000 {
        rk4_step(&mut tree, g, dt, zero_wrenches(2));
    }
    // Sample qdot over the next 1 s and check the lag envelope holds.
    let mut max_lag = 0.0f32;
    for _ in 0..1000 {
        rk4_step(&mut tree, g, dt, zero_wrenches(2));
        let qdot = tree.hinge_rate(1);
        let lag = (1.5 - qdot).abs();
        if lag > max_lag {
            max_lag = lag;
        }
    }
    // Analytic envelope: |ctrl - qdot| = |m·g·L·sin(q) + joint_damping*qdot| / kv
    //                                  ≤ (m·g·L + damping·|qdot_max|) / kv
    // With kv=50, m·g·L=9.81, damping=0.5, qdot near 1.5:
    //   envelope ≤ (9.81 + 0.5*1.5) / 50 = 10.56/50 = 0.2112
    assert!(
        max_lag < 0.25,
        "velocity actuator lag {max_lag} exceeds analytic envelope 0.2112"
    );
    // And the lag must be nonzero — otherwise gravity is not entering
    // the tracking loop at all (a scary silent bug).
    assert!(
        max_lag > 0.05,
        "velocity actuator lag {max_lag} suspiciously small"
    );
}

/// First-order filter step response. With `ctrl` held at 1.0 from
/// `act=0` and `tau=0.1`, forward Euler at `dt=0.001` gives the closed
/// form `act_n = 1 - (1 - dt/tau)^n = 1 - 0.99^n`. This test drives the
/// filter directly (no tree) so any RK4 interaction is out of the loop.
#[test]
fn filter_step_response_matches_forward_euler_closed_form() {
    let tau = 0.1f32;
    let dt = 0.001f32;
    let mut a = Actuator::general(
        0,
        GainType::Fixed,
        [1.0, 0.0, 0.0],
        BiasType::None,
        [0.0, 0.0, 0.0],
        1.0,
        DynType::Filter,
        [tau],
        None,
        None,
    );
    a.ctrl = 1.0;
    // Sample at n = 50, 100, 250, 500.
    let checkpoints: &[(usize, f32)] = &[
        // 1 - 0.99^50 ≈ 0.39498...
        (50, 0.39498307),
        // 1 - 0.99^100 ≈ 0.63397...
        (100, 0.63396765),
        // 1 - 0.99^250 ≈ 0.918
        (250, 0.918),
        // 1 - 0.99^500 ≈ 0.9934
        (500, 0.9934),
    ];
    let mut cursor = 0;
    for n in 1..=500 {
        a.integrate_activation(dt);
        if cursor < checkpoints.len() && n == checkpoints[cursor].0 {
            let (_, expected) = checkpoints[cursor];
            assert!(
                (a.act - expected).abs() < 5.0e-3,
                "n={n}: act={} expected≈{}",
                a.act,
                expected
            );
            cursor += 1;
        }
    }
    assert_eq!(cursor, checkpoints.len());
}

/// Edge case: `tau < dt`. Forward Euler with alpha > 1 overshoots the
/// target. This is the documented behavior — `dt/tau > 1` produces
/// oscillation, and `Actuator::general` refuses `tau <= 0` at build
/// time to keep the ODE well-defined. Just check that the actuator does
/// not panic and that after one step act reaches `ctrl` (with alpha=1
/// exactly) or overshoots monotonically (alpha in (1, 2)).
#[test]
fn filter_alpha_gt_one_overshoots_but_does_not_diverge() {
    // dt=1, tau=1 → alpha=1: act jumps from 0 to ctrl in one step exactly.
    let mut a = Actuator::general(
        0,
        GainType::Fixed,
        [1.0, 0.0, 0.0],
        BiasType::None,
        [0.0, 0.0, 0.0],
        1.0,
        DynType::Filter,
        [1.0],
        None,
        None,
    );
    a.ctrl = 5.0;
    a.integrate_activation(1.0);
    assert!(
        (a.act - 5.0).abs() < 1e-6,
        "alpha=1 exact tracking failed: {}",
        a.act
    );

    // dt=0.15, tau=0.1 → alpha=1.5: one-step overshoot to 1.5*ctrl.
    let mut a = Actuator::general(
        0,
        GainType::Fixed,
        [1.0, 0.0, 0.0],
        BiasType::None,
        [0.0, 0.0, 0.0],
        1.0,
        DynType::Filter,
        [0.1],
        None,
        None,
    );
    a.ctrl = 2.0;
    a.integrate_activation(0.15);
    // After one step: act = 0 + 1.5 * (2 - 0) = 3.0 (overshoot).
    assert!((a.act - 3.0).abs() < 1e-6, "overshoot value: {}", a.act);
}

/// Ctrl clamp binds BEFORE force clamp. A motor with `gear=10`,
/// `ctrl_range=(-2, 2)`, `force_range=(-25, 25)`, `ctrl=100`:
///
/// - after ctrl clamp: u = 2 → F = 20 (under force cap → no bind)
/// - after force clamp: 20
///
/// If the ctrl clamp were skipped: F = 1000 → force clamp binds to 25.
/// The observed value discriminates.
#[test]
fn ctrl_clamp_binds_before_force_clamp() {
    let mut a = Actuator::motor(0, 10.0, 0.0);
    a.ctrl = 100.0;
    a.ctrl_range = Some((-2.0, 2.0));
    a.force_range = Some((-25.0, 25.0));
    a = a.with_limit_flags(true, true);
    let t = a.torque(0.0, 0.0);
    assert!((t - 20.0).abs() < 1e-6, "expected 20, got {t}");

    // Now widen the ctrl clamp so the force clamp is the binder.
    a.ctrl_range = Some((-100.0, 100.0));
    let t = a.torque(0.0, 0.0);
    assert!(
        (t - 25.0).abs() < 1e-6,
        "expected 25 (force clamp binds), got {t}"
    );
}

/// Transmission-space affine sampling (reviewer's worked example).
///
/// MuJoCo joint transmission with scalar gear: `len = gear·q`,
/// `vel = gear·qdot`. Affine gain and bias MUST evaluate at those
/// transmission-space coordinates, not at the raw joint state — otherwise
/// a `gear!=1` actuator reads its own state wrong.
///
/// Params (hand-tuned so the pre-fix code returns a different, wrong
/// number and the fixed code returns the MuJoCo-correct number):
///
/// ```text
/// gain      = Fixed(2)
/// bias      = Affine(0, -1, 0)   →  b = -1 · len
/// gear      = 2
/// ctrl      = 1
/// q, qdot   = (0.5, 0)
/// ```
///
/// Correct (transmission space): `len = 2·0.5 = 1`, `b = -1`,
/// `F_raw = 2·1 + (-1) = 1`, `τ = F_raw · gear = 2`.
/// Pre-fix (raw space):          `len = 0.5`,    `b = -0.5`,
/// `F_raw = 2·1 + (-0.5) = 1.5`, `τ = 1.5 · 2 = 3`.
///
/// So the assertion is `τ = 2`, and reverting the transmission fix
/// (sampling raw q instead of gear·q) produces τ = 3 — a 50 % force
/// error at the same q. The reviewer verified this by re-mutating the
/// General branch back to raw sampling; this anchor fails immediately
/// under that mutation.
#[test]
fn general_affine_bias_samples_transmission_space() {
    let mut a = Actuator::general(
        0,
        GainType::Fixed,
        [2.0, 0.0, 0.0],
        BiasType::Affine,
        [0.0, -1.0, 0.0],
        2.0, // gear
        DynType::None,
        [1.0],
        None,
        None,
    );
    a.ctrl = 1.0;
    let t = a.torque(0.5, 0.0);
    assert!(
        (t - 2.0).abs() < 1e-6,
        "transmission-space affine: expected τ = 2, got {t} \
         (pre-fix raw-sampling code returns 3.0 — this test is the mutant guard)"
    );

    // Also verify the affine gain path (the other affine slot the fix
    // covers). `gain = Affine(1, 1, 0)` samples len → transmission-space
    // must give len=1 at q=0.5, gear=2.
    let mut a = Actuator::general(
        0,
        GainType::Fixed, // set to Affine below
        [0.0, 0.0, 0.0],
        BiasType::None,
        [0.0, 0.0, 0.0],
        2.0,
        DynType::None,
        [1.0],
        None,
        None,
    );
    if let ActuatorFlavor::General {
        ref mut gain_type,
        ref mut gain_prm,
        ..
    } = a.flavor
    {
        *gain_type = GainType::Affine;
        *gain_prm = [1.0, 1.0, 0.0];
    }
    a.ctrl = 1.0;
    // Correct: len=1, g = 1 + 1 = 2, F = 2·1 = 2, τ = 2·2 = 4.
    // Pre-fix: len=q=0.5, g = 1 + 0.5 = 1.5, F = 1.5, τ = 3.
    let t = a.torque(0.5, 0.0);
    assert!(
        (t - 4.0).abs() < 1e-6,
        "affine gain must sample transmission len: expected 4, got {t} (pre-fix: 3)"
    );

    // vel branch: affine gain sampling vel · gear.
    let mut a = Actuator::general(
        0,
        GainType::Affine,
        [0.0, 0.0, 1.0], // g = vel_tr
        BiasType::None,
        [0.0, 0.0, 0.0],
        3.0,
        DynType::None,
        [1.0],
        None,
        None,
    );
    a.ctrl = 1.0;
    // Correct: vel=3·0.4=1.2, g=1.2, F=1.2, τ=1.2·3=3.6.
    // Pre-fix: vel=0.4, g=0.4, F=0.4, τ=1.2.
    let t = a.torque(0.0, 0.4);
    assert!(
        (t - 3.6).abs() < 1e-5,
        "affine gain vel term must sample transmission vel: expected 3.6, got {t}"
    );
}

/// Affine bias in the general model produces a zero at a hand-derived
/// equilibrium `len`. Setup: `gain=Fixed(0)`, `bias=Affine(k, -k, 0)`,
/// `ctrl=0`, `gear=1` (so transmission-space `len` equals raw `q`).
/// Torque = `k - k*len = k*(1 - len)`. Zero at `len = 1`.
#[test]
fn affine_bias_zero_at_hand_derived_equilibrium() {
    let mut a = Actuator::general(
        0,
        GainType::Fixed,
        [0.0, 0.0, 0.0],
        BiasType::Affine,
        [2.5, -2.5, 0.0],
        1.0,
        DynType::None,
        [1.0],
        None,
        None,
    );
    a.ctrl = 0.0;
    // len = 0.5 → torque = 2.5 - 1.25 = 1.25
    assert!((a.torque(0.5, 0.0) - 1.25).abs() < 1e-6);
    // len = 1.0 → torque = 0
    assert!(a.torque(1.0, 0.0).abs() < 1e-6);
    // len = 2.0 → torque = -2.5
    assert!((a.torque(2.0, 0.0) - (-2.5)).abs() < 1e-6);
}

/// The activation integrator runs ONCE per rk4_step at the boundary
/// (ZOH within the step). Verify by driving a filter actuator on a
/// hinge and comparing `act` after N steps against the forward-Euler
/// closed form `1 - (1 - dt/tau)^n`.
#[test]
fn activation_integrates_once_per_rk4_step() {
    let mut tree = build_hinge(1.0, 1.0);
    let mut act = Actuator::general(
        1,
        GainType::Fixed,
        [1.0, 0.0, 0.0],
        BiasType::None,
        [0.0, 0.0, 0.0],
        1.0,
        DynType::Filter,
        [0.05],
        None,
        None,
    );
    act.ctrl = 1.0;
    let a = tree.add_actuator(act);
    let _ = a;
    let dt = 0.001f32;
    // n=50 steps @ dt/tau = 0.02 → act ≈ 1 - 0.98^50 ≈ 0.63583
    for _ in 0..50 {
        rk4_step(&mut tree, Vec3::ZERO, dt, zero_wrenches(2));
    }
    let observed = tree.actuators[0].act;
    let expected = 0.63583_f32;
    assert!(
        (observed - expected).abs() < 5e-3,
        "act={observed} expected≈{expected}"
    );
}

// ---------------------------------------------------------------------------
// MJCF ctrlrange plumbing (shorthand actuators) — per-shorthand tests
// ---------------------------------------------------------------------------

/// Load a one-hinge MJCF with the given shorthand + attributes and return
/// the loaded scene. Kept local so each per-shorthand test can inline its
/// own MJCF and read out the resulting actuator directly.
fn load_actuator_scene(actuator_xml: &str) -> newt::model::Scene {
    let src = format!(
        r#"<mujoco model="t">
          <compiler angle="radian"/>
          <option timestep="0.005"/>
          <worldbody>
            <body name="anchor">
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <body name="link" pos="0 0 -0.5">
                <joint name="j" type="hinge" axis="1 0 0"/>
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              </body>
            </body>
          </worldbody>
          <actuator>{actuator_xml}</actuator>
        </mujoco>"#
    );
    newt::mjcf::load_mjcf_str(&src).unwrap_or_else(|e| panic!("MJCF load failed: {e}"))
}

/// `<position ctrlrange>` populates `ctrl_range` and the actuator torque
/// evaluates with the clamped ctrl (hand-computed both bind cases).
#[test]
fn position_ctrlrange_wires_into_actuator_and_clamps_torque() {
    let scene = load_actuator_scene(
        r#"<position name="s" joint="j" kp="10" kv="0"
                   ctrlrange="-0.5 0.5" forcerange="-100 100"/>"#,
    );
    let (t, a) = scene.actuators_by_name["s"];
    let mut actuator = scene.world.trees[t].actuators[a];
    assert_eq!(
        actuator.ctrl_range,
        Some((-0.5, 0.5)),
        "ctrl_range not wired from <position ctrlrange>"
    );
    // ctrl = 5 → clamps to 0.5 → τ = 10 · (0.5 - 0) - 0 = 5.
    actuator.ctrl = 5.0;
    assert!((actuator.torque(0.0, 0.0) - 5.0).abs() < 1e-6);
    // ctrl = -5 → clamps to -0.5 → τ = -5.
    actuator.ctrl = -5.0;
    assert!((actuator.torque(0.0, 0.0) - (-5.0)).abs() < 1e-6);
    // ctrl inside range: no clamp, exact PD.
    actuator.ctrl = 0.3;
    assert!((actuator.torque(0.1, 0.0) - 10.0 * 0.2).abs() < 1e-6);
}

/// `<velocity ctrlrange>` same wiring; ctrl clamps before torque eval.
#[test]
fn velocity_ctrlrange_wires_into_actuator_and_clamps_torque() {
    let scene = load_actuator_scene(
        r#"<velocity name="v" joint="j" kv="4" ctrlrange="-2 2" forcerange="-1000 1000"/>"#,
    );
    let (t, a) = scene.actuators_by_name["v"];
    let mut actuator = scene.world.trees[t].actuators[a];
    assert_eq!(actuator.ctrl_range, Some((-2.0, 2.0)));
    // ctrl = 10 → u = 2 → τ = 4·(2 - 0.5) = 6
    actuator.ctrl = 10.0;
    assert!((actuator.torque(0.0, 0.5) - 6.0).abs() < 1e-6);
    // ctrl = -10 → u = -2 → τ = 4·(-2 - (-0.5)) = -6
    actuator.ctrl = -10.0;
    assert!((actuator.torque(0.0, -0.5) - (-6.0)).abs() < 1e-6);
}

/// `<motor ctrlrange>` — plumbing lit up in round 2 (motor was silently
/// dropping ctrlrange in the round-1 impl).
#[test]
fn motor_ctrlrange_wires_into_actuator_and_clamps_torque() {
    let scene = load_actuator_scene(
        r#"<motor name="m" joint="j" gear="5" ctrlrange="-1 1" forcerange="-100 100"/>"#,
    );
    let (t, a) = scene.actuators_by_name["m"];
    let mut actuator = scene.world.trees[t].actuators[a];
    assert_eq!(actuator.ctrl_range, Some((-1.0, 1.0)));
    // ctrl = 3 → u = 1 → τ = 5·1 = 5 (well under forcerange).
    actuator.ctrl = 3.0;
    assert!((actuator.torque(0.0, 0.0) - 5.0).abs() < 1e-6);
    // ctrl = -3 → u = -1 → τ = -5.
    actuator.ctrl = -3.0;
    assert!((actuator.torque(0.0, 0.0) - (-5.0)).abs() < 1e-6);
    // ctrl in range → gear·ctrl direct.
    actuator.ctrl = 0.4;
    assert!((actuator.torque(0.0, 0.0) - 2.0).abs() < 1e-6);
}

/// Round-2 finding 4: `<motor>` must validate `ctrlrange` / `forcerange`
/// like its siblings. Malformed values (wrong count, `lo >= hi`) must
/// error — round-1 silently accepted them.
#[test]
fn motor_malformed_ctrlrange_errors() {
    let cases = [
        (
            r#"<motor name="m" joint="j" gear="1" ctrlrange="1 0"/>"#,
            "ctrlrange low must be < high",
        ),
        (
            r#"<motor name="m" joint="j" gear="1" ctrlrange="1 1"/>"#,
            "ctrlrange low must be < high",
        ),
        (
            r#"<motor name="m" joint="j" gear="1" ctrlrange="1"/>"#,
            "expected 2 numbers",
        ),
    ];
    for (xml, needle) in cases {
        let src = format!(
            r#"<mujoco model="t">
              <compiler angle="radian"/>
              <option timestep="0.005"/>
              <worldbody>
                <body name="anchor">
                  <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                  <body name="link" pos="0 0 -0.5">
                    <joint name="j" type="hinge" axis="1 0 0"/>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                  </body>
                </body>
              </worldbody>
              <actuator>{xml}</actuator>
            </mujoco>"#
        );
        match newt::mjcf::load_mjcf_str(&src) {
            Ok(_) => panic!("expected error containing {needle:?}, got Ok for {xml}"),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains(needle),
                    "expected error containing {needle:?}, got: {msg} (input {xml})"
                );
            }
        }
    }

    // Same shape for forcerange (round-1 motor also let malformed
    // forcerange through — the finding folds them together).
    let src = r#"<mujoco model="t">
      <compiler angle="radian"/>
      <option timestep="0.005"/>
      <worldbody>
        <body name="anchor">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <body name="link" pos="0 0 -0.5">
            <joint name="j" type="hinge" axis="1 0 0"/>
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor name="m" joint="j" gear="1" forcerange="2 1"/></actuator>
    </mujoco>"#;
    match newt::mjcf::load_mjcf_str(src) {
        Ok(_) => panic!("expected error for forcerange low >= high, got Ok"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("forcerange low must be < high"),
                "expected forcerange error, got: {msg}"
            );
        }
    }
}
