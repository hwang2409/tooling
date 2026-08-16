//! Actuators (v2 tier 2).
//!
//! MuJoCo-style general actuator: joint-transmission force
//!
//! ```text
//! signal = if dyn == Filter { act } else { u }         u = clamp(ctrl, ctrl_range)
//! F      = gain(len, vel) * signal + bias(len, vel)
//! τ      = clamp(F * gear, force_range)                // ctrl clamp first, force clamp last
//! ```
//!
//! and enters the tree's generalized-force channel exactly like a raw
//! `qfrc_applied` torque did in v0.
//!
//! # Flavors
//!
//! Real MJCF exposes the general model through four shorthands. This crate
//! materializes each as its own [`ActuatorFlavor`] variant so the evaluation
//! order is fast AND stays byte-identical to the v0 [`PdServo`] path — see
//! the `position` note below for why the shape matters.
//!
//! - `Position { kp, kv }` — MuJoCo `<position>`. Equivalent to
//!   `gain=fixed(kp)`, `bias=affine(0,-kp,-kv)`, `gear=1`, `dyn=none`.
//!   Torque is `kp*(ctrl-len) - kv*vel`, clamped. This is the v0 PD servo
//!   verbatim; the arm golden depends on this exact op order.
//! - `Velocity { kv }` — MuJoCo `<velocity>`. Equivalent to
//!   `gain=fixed(kv)`, `bias=affine(0,0,-kv)`, `gear=1`, `dyn=none`.
//!   Torque is `kv*(ctrl - vel)`.
//! - `Motor { gear }` — MuJoCo `<motor>`. Equivalent to
//!   `gain=fixed(1)`, `bias=none`, `dyn=none`. Torque is `gear * ctrl`.
//! - `General { .. }` — the raw gain/bias/gear composition; use for
//!   parameter shapes the shorthands can't cover.
//!
//! # Activation dynamics (`DynType::Filter`)
//!
//! When `dyn_type = Filter`, the actuator carries a scalar activation state
//! `act` that lags the (clamped) control input:
//!
//! ```text
//! act' = (u - act) / tau                                 u = clamp(ctrl, ctrl_range)
//! ```
//!
//! Integrated with **forward Euler at each step boundary**:
//!
//! ```text
//! act ← act + (dt/tau) * (u - act)
//! ```
//!
//! ZOH within the step: the RK4 sub-stages of `Tree::step` read `act`
//! unchanged; `Tree::integrate_activations(dt)` is called ONCE per step
//! (see [`crate::tree::rk4_step`]). This matches the ZOH convention already
//! used for `ctrl`, `qfrc_applied`, and `applied_wrenches` (documented in
//! `docs/actuators.md`) and mirrors MuJoCo's Euler activation integrator
//! for `integrator=Euler`. For `integrator=RK4` MuJoCo also integrates
//! activation with the same RK4 stages; the difference shows up in the
//! filtered-motor differential scenario as a bounded, documented residual
//! (see `docs/differential.md`).
//!
//! Edge case: `dt >= tau`. Forward Euler overshoots when `dt/tau > 1`
//! and oscillates when `dt/tau > 2`. This is stated behavior — MuJoCo's
//! Euler integrator does the same, and the fix is to use `tau > dt` in
//! the model. `Actuator::position*` constructors reject `tau <= 0`.
//!
//! `actearly` is **not supported** (v2 tier 2 rejects it in MJCF; see
//! `docs/actuators.md`). MuJoCo's actearly evaluates `gain`/`bias` at
//! `(len, vel)` from the NEXT step's forward-kinematics pass — a coupling
//! that would require deferring the activation update past the mechanical
//! step. Deferred to a later tier.
//!
//! # Byte-identity with v0 PdServo (position path)
//!
//! The `Position` flavor stores `(kp, kv)` and evaluates torque as
//! `kp * (ctrl - len) - kv * vel`, then applies `clamp_symmetric` — the
//! **exact** ops the old `PdServo::torque` used. The
//! `actuators_arm_waypoints.bin` golden and every PD test in
//! `tests/actuators_pd.rs` continue to pass without regeneration.
//!
//! If a `<general>` MJCF actuator carries the same numerical parameters
//! (`gainprm=[kp,0,0]`, `biasprm=[0,-kp,-kv]`), its evaluator uses the
//! plain `gain*sig + bias_prm[0] + bias_prm[1]*len + bias_prm[2]*vel`
//! form — algebraically equivalent but NOT bit-equal to the Position
//! flavor. See `tests/actuators_general.rs::position_shape_matches_shortcut`
//! for the residual bound. Callers who care about bit-identity with the
//! v0 PD path should keep using the shorthand.
//!
//! # PD damping model (why `2·ζ·√(kp·I_ref)`)
//!
//! Verbatim from v0: a single hinge with reflected inertia `I_ref` under
//! PD control obeys `I_ref · q̈ = kp·(target − q) − kd·qdot`, i.e. a
//! linear second-order system with natural frequency `ωₙ = √(kp / I_ref)`
//! and damping ratio `ζ = kd / (2·√(kp·I_ref))`. Solving for `kd`:
//!
//! ```text
//! kd = 2 · ζ · √(kp · I_ref)
//! ```
//!
//! `Actuator::position_from_dampratio` computes `kv = 2·ζ·√(kp·I_ref)`.
//! The caller supplies `I_ref` because the actuator does not have access
//! to the tree's articulated-inertia block — same rationale as v0.

/// Which side of the general model the gain samples on. Affine sampling
/// uses TRANSMISSION-space coordinates `(len, vel) = (gear*q, gear*qdot)`
/// (MuJoCo convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainType {
    /// `g = gainprm[0]`.
    Fixed,
    /// `g = gainprm[0] + gainprm[1]*(gear*q) + gainprm[2]*(gear*qdot)`.
    Affine,
}

/// Which side of the general model the bias samples on. Affine sampling
/// uses TRANSMISSION-space coordinates (see [`GainType::Affine`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasType {
    /// `b = 0`.
    None,
    /// `b = biasprm[0] + biasprm[1]*(gear*q) + biasprm[2]*(gear*qdot)`.
    Affine,
}

/// Activation dynamics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynType {
    /// `act` is unused; the signal fed into `gain*signal + bias` is the
    /// (clamped) `ctrl` value directly.
    None,
    /// `act' = (u - act) / tau`, integrated with forward Euler at each
    /// step boundary. `u` is the clamped `ctrl`. `dynprm[0] = tau`.
    Filter,
}

/// Fast-path evaluation variant. Each shorthand carries the parameters
/// needed to compute torque without going through the full general
/// formula — see the module docs for byte-identity implications.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActuatorFlavor {
    /// MuJoCo `<position kp kv>`. Torque = `kp*(ctrl - len) - kv*vel`.
    Position { kp: f32, kv: f32 },
    /// MuJoCo `<velocity kv>`. Torque = `kv*(ctrl - vel)`.
    Velocity { kv: f32 },
    /// MuJoCo `<motor gear>`. Torque = `gear * ctrl`.
    Motor { gear: f32 },
    /// MuJoCo `<general>`. Torque = `(gain*signal + bias) * gear`,
    /// where `signal = act` for `DynType::Filter` and `signal = clamped ctrl`
    /// for `DynType::None`, `gain` and `bias` are evaluated per the field
    /// types below.
    General {
        gain_type: GainType,
        gain_prm: [f32; 3],
        bias_type: BiasType,
        bias_prm: [f32; 3],
        gear: f32,
    },
}

/// One actuator attached to a 1-DOF joint (hinge or slide) OR to a tendon
/// in the containing tree.
///
/// `link_idx` selects the joint slot when `tendon_target` is `None` (the
/// v2-tier-2 default). When `tendon_target` is `Some(i)`, the actuator
/// drives tendon `i` on the tree — see [`crate::tendon`] for the tendon
/// transmission model. In tendon mode, `link_idx` is unused (kept in the
/// struct so the layout stays `Copy`; the JSON loader sets it to 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Actuator {
    /// Link index in the containing tree. Must reference a hinge or
    /// slide WHEN `tendon_target` is `None` (joint-transmission mode).
    /// Ignored in tendon-transmission mode.
    pub link_idx: usize,
    /// Tendon-transmission target: `Some(i)` = drive
    /// `tree.tendons[i]`; `None` = joint transmission on `link_idx`.
    /// Introduced in v2 tier 3 (tendons). All existing constructors
    /// leave this as `None`, keeping the joint-transmission path
    /// bit-for-bit identical.
    pub tendon_target: Option<usize>,
    /// How torque is computed from `(len, vel, ctrl, act)`.
    pub flavor: ActuatorFlavor,
    /// Activation-dynamics shape.
    pub dyn_type: DynType,
    /// Activation-dynamics parameter vector. `dyn_prm[0]` is the filter
    /// time constant `tau` (seconds) for `DynType::Filter`.
    pub dyn_prm: [f32; 1],
    /// Optional `(lo, hi)` clamp applied to `ctrl` before it enters
    /// `gain*signal + bias` (and before activation integration).
    /// `None` = no clamp.
    pub ctrl_range: Option<(f32, f32)>,
    /// Optional `(lo, hi)` clamp applied to the joint torque after gear
    /// multiplication. `None` = no clamp.
    pub force_range: Option<(f32, f32)>,
    /// User command, ZOH per step. Setter: [`crate::tree::Tree::set_actuator_target`]
    /// (kept under the v0 name for API stability; MuJoCo would call this
    /// `data.ctrl[i]`).
    pub ctrl: f32,
    /// Activation state, integrated per step by `integrate_activation`.
    /// Ignored when `dyn_type == None`.
    pub act: f32,
}

impl Actuator {
    // ---- shorthand constructors ---------------------------------------

    /// Build a position actuator: `τ = clamp_sym(kp*(ctrl-len) - kv*vel, force_range)`.
    /// `force_range <= 0.0` disables the clamp (symmetric convention preserved
    /// from v0 `PdServo`).
    pub fn position(link_idx: usize, kp: f32, kv: f32, force_range: f32, target: f32) -> Self {
        assert!(kp >= 0.0, "position kp must be >= 0");
        assert!(kv >= 0.0, "position kv must be >= 0");
        Self {
            link_idx,
            tendon_target: None,
            flavor: ActuatorFlavor::Position { kp, kv },
            dyn_type: DynType::None,
            dyn_prm: [0.0],
            ctrl_range: None,
            force_range: symmetric_range(force_range),
            ctrl: target,
            act: 0.0,
        }
    }

    /// Build a position actuator with `kv = 2·dampratio·√(kp·reflected_inertia)`
    /// (see the module docs' derivation). Target starts at 0.
    pub fn position_from_dampratio(
        link_idx: usize,
        kp: f32,
        dampratio: f32,
        reflected_inertia: f32,
        force_range: f32,
    ) -> Self {
        assert!(dampratio >= 0.0, "dampratio must be >= 0");
        assert!(reflected_inertia > 0.0, "reflected_inertia must be > 0");
        let kv = 2.0 * dampratio * (kp * reflected_inertia).sqrt();
        Self::position(link_idx, kp, kv, force_range, 0.0)
    }

    /// Build a velocity actuator: `τ = clamp_sym(kv*(ctrl - vel), force_range)`.
    pub fn velocity(link_idx: usize, kv: f32, force_range: f32) -> Self {
        assert!(kv >= 0.0, "velocity kv must be >= 0");
        Self {
            link_idx,
            tendon_target: None,
            flavor: ActuatorFlavor::Velocity { kv },
            dyn_type: DynType::None,
            dyn_prm: [0.0],
            ctrl_range: None,
            force_range: symmetric_range(force_range),
            ctrl: 0.0,
            act: 0.0,
        }
    }

    /// Build a motor actuator: `τ = clamp_sym(gear*ctrl, force_range)`.
    pub fn motor(link_idx: usize, gear: f32, force_range: f32) -> Self {
        Self {
            link_idx,
            tendon_target: None,
            flavor: ActuatorFlavor::Motor { gear },
            dyn_type: DynType::None,
            dyn_prm: [0.0],
            ctrl_range: None,
            force_range: symmetric_range(force_range),
            ctrl: 0.0,
            act: 0.0,
        }
    }

    /// Build a raw general actuator. `gear` multiplies the (gain*signal +
    /// bias) result. `dyn_prm[0]` is the filter tau for `DynType::Filter`;
    /// pass `1.0` (or any positive) for `DynType::None`.
    ///
    /// Ranges are `Some((lo, hi))` with `lo < hi`, or `None` for no clamp.
    #[allow(clippy::too_many_arguments)]
    pub fn general(
        link_idx: usize,
        gain_type: GainType,
        gain_prm: [f32; 3],
        bias_type: BiasType,
        bias_prm: [f32; 3],
        gear: f32,
        dyn_type: DynType,
        dyn_prm: [f32; 1],
        ctrl_range: Option<(f32, f32)>,
        force_range: Option<(f32, f32)>,
    ) -> Self {
        if let DynType::Filter = dyn_type {
            assert!(
                dyn_prm[0] > 0.0,
                "DynType::Filter requires dyn_prm[0] (tau) > 0"
            );
        }
        if let Some((lo, hi)) = ctrl_range {
            assert!(lo < hi, "ctrl_range must satisfy lo < hi (got {lo}..{hi})");
        }
        if let Some((lo, hi)) = force_range {
            assert!(lo < hi, "force_range must satisfy lo < hi (got {lo}..{hi})");
        }
        Self {
            link_idx,
            tendon_target: None,
            flavor: ActuatorFlavor::General {
                gain_type,
                gain_prm,
                bias_type,
                bias_prm,
                gear,
            },
            dyn_type,
            dyn_prm,
            ctrl_range,
            force_range,
            ctrl: 0.0,
            act: 0.0,
        }
    }

    // ---- tendon-transmission builder ---------------------------------

    /// Retarget a joint-mode actuator at a tendon. Consumes `self` and
    /// returns the retargeted actuator. `link_idx` becomes unused (set to
    /// 0 in the returned struct) — the actuator's scalar force is
    /// distributed through the tendon's Jacobian by
    /// [`crate::tendon::accumulate_tendon_actuator_qfrc`] instead of
    /// entering the ABA per-link `tau_scalar`.
    ///
    /// The evaluation model is unchanged: `(len, vel)` passed to
    /// [`Actuator::torque`] become the tendon length and velocity (in
    /// transmission space — MuJoCo semantics). For a motor tendon
    /// actuator, `torque = gear * ctrl` and the resulting scalar force
    /// pulls both endpoints in via the length-gradient.
    pub fn on_tendon(mut self, tendon_idx: usize) -> Self {
        self.tendon_target = Some(tendon_idx);
        self.link_idx = 0;
        self
    }

    // ---- evaluation ---------------------------------------------------

    /// (Clamped) `ctrl` — the input to `gain*signal + bias` when
    /// `dyn_type == None`, and the driving input to the filter ODE when
    /// `dyn_type == Filter`.
    #[inline]
    pub fn clamped_ctrl(&self) -> f32 {
        match self.ctrl_range {
            Some((lo, hi)) => clamp_range(self.ctrl, lo, hi),
            None => self.ctrl,
        }
    }

    /// Joint scalar force at state `(len, vel)`. Adds directly into the
    /// tree's `tau_scalar` inside ABA — see
    /// [`crate::tree::aba`].
    #[inline]
    pub fn torque(&self, len: f32, vel: f32) -> f32 {
        let raw = match self.flavor {
            ActuatorFlavor::Position { kp, kv } => {
                // Byte-identical to v0 PdServo::torque_unclamped.
                let u = self.clamped_ctrl();
                kp * (u - len) - kv * vel
            }
            ActuatorFlavor::Velocity { kv } => {
                let u = self.clamped_ctrl();
                kv * (u - vel)
            }
            ActuatorFlavor::Motor { gear } => {
                let u = self.clamped_ctrl();
                gear * u
            }
            ActuatorFlavor::General {
                gain_type,
                gain_prm,
                bias_type,
                bias_prm,
                gear,
            } => {
                let signal = match self.dyn_type {
                    DynType::None => self.clamped_ctrl(),
                    DynType::Filter => self.act,
                };
                // MuJoCo transmission-space sampling: for a joint
                // transmission with scalar gear, `actuator_length = gear*q`
                // and `actuator_velocity = gear*qdot`. Affine gain/bias
                // MUST evaluate at the transmission-space coordinates,
                // not the raw joint values, or a gear!=1 actuator
                // reads its own state wrong. See the reviewer's worked
                // example and `tests/actuators_general.rs::
                // general_affine_bias_samples_transmission_space`.
                //
                // Fixed gain and BiasType::None don't sample len/vel,
                // so this factor is dead for them — but the closures
                // below stay unconditional to keep the branch shape
                // uniform. Cost is two f32 multiplies.
                let len_tr = len * gear;
                let vel_tr = vel * gear;
                let g = match gain_type {
                    GainType::Fixed => gain_prm[0],
                    GainType::Affine => gain_prm[0] + gain_prm[1] * len_tr + gain_prm[2] * vel_tr,
                };
                let b = match bias_type {
                    BiasType::None => 0.0,
                    BiasType::Affine => bias_prm[0] + bias_prm[1] * len_tr + bias_prm[2] * vel_tr,
                };
                (g * signal + b) * gear
            }
        };
        match self.force_range {
            Some((lo, hi)) => clamp_range(raw, lo, hi),
            None => raw,
        }
    }

    /// Return the positive velocity coefficient in the actuator force law.
    ///
    /// This is `-∂τ/∂vel` before the force clamp. It is the derivative that
    /// MuJoCo's `implicitfast` path folds into the velocity solve. A force
    /// clamp is deliberately not differentiated here; the implicitfast
    /// scope in this crate covers the actuator velocity terms themselves.
    #[inline]
    pub fn velocity_damping(&self, _len: f32, _vel: f32) -> f32 {
        match self.flavor {
            ActuatorFlavor::Position { kv, .. } | ActuatorFlavor::Velocity { kv } => kv,
            ActuatorFlavor::Motor { .. } => 0.0,
            ActuatorFlavor::General {
                gain_type,
                gain_prm,
                bias_type,
                bias_prm,
                gear,
            } => {
                let signal = match self.dyn_type {
                    DynType::None => self.clamped_ctrl(),
                    DynType::Filter => self.act,
                };
                let gain_velocity = match gain_type {
                    GainType::Fixed => 0.0,
                    GainType::Affine => gain_prm[2],
                };
                let bias_velocity = match bias_type {
                    BiasType::None => 0.0,
                    BiasType::Affine => bias_prm[2],
                };
                // Keep the same transmission-space derivative as torque:
                // both the sampled velocity and output force carry `gear`.
                -gear * gear * (gain_velocity * signal + bias_velocity)
            }
        }
    }

    /// Advance activation state by one step using forward Euler on
    /// `act' = (u - act) / tau`. No-op for `DynType::None`.
    ///
    /// Called once per `rk4_step` at step end — see the module docs for
    /// the ZOH-within-step convention and its RK4 interaction.
    #[inline]
    pub fn integrate_activation(&mut self, dt: f32) {
        if let DynType::Filter = self.dyn_type {
            let tau = self.dyn_prm[0];
            let u = self.clamped_ctrl();
            // Explicit Euler: act += dt/tau * (u - act).
            let alpha = dt / tau;
            self.act += alpha * (u - self.act);
        }
    }
}

/// Effective moment of inertia of a hinge that swings a point-mass at
/// distance `length` from the pivot, plus rotor `armature`:
///
/// ```text
/// I_ref = mass · length² + armature
/// ```
///
/// For a uniform rod pivoted at one end, `I_pivot = m·L²/3`; call with
/// `mass, length` set so `mass·length² = m·L²/3` (e.g. `length = L/√3`),
/// or pass the pivot inertia directly through the raw [`Actuator::position`].
#[inline]
pub fn reflected_inertia_point_mass(mass: f32, length: f32, armature: f32) -> f32 {
    mass * length * length + armature
}

/// Symmetric clamp with pass-through when `cap <= 0` (v0 convention).
///
/// Kept `pub(crate)` for use inside `tree::set_joint_torque_clamped`.
#[inline]
pub(crate) fn clamp_symmetric(x: f32, cap: f32) -> f32 {
    if cap <= 0.0 {
        x
    } else if x > cap {
        cap
    } else if x < -cap {
        -cap
    } else {
        x
    }
}

/// Asymmetric clamp; equivalent to `x.clamp(lo, hi)` on finite inputs but
/// with the same branchy ordering as [`clamp_symmetric`] so that
/// `clamp_range(x, -cap, cap)` is bit-equal to `clamp_symmetric(x, cap)`
/// for `cap > 0`.
#[inline]
fn clamp_range(x: f32, lo: f32, hi: f32) -> f32 {
    if x > hi {
        hi
    } else if x < lo {
        lo
    } else {
        x
    }
}

/// Turn a scalar force-range cap into an `Option<(lo, hi)>`. `cap <= 0`
/// means "no clamp" per the v0 convention; positive `cap` yields the
/// symmetric interval `(-cap, +cap)`.
#[inline]
fn symmetric_range(cap: f32) -> Option<(f32, f32)> {
    if cap > 0.0 { Some((-cap, cap)) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn position_dampratio_recovers_critical_damping_formula() {
        // ωn = √(kp/I) = 10, kd_critical = 2·1·√(kp·I) = 2·√100 = 20.
        let a = Actuator::position_from_dampratio(0, 100.0, 1.0, 1.0, -1.0);
        match a.flavor {
            ActuatorFlavor::Position { kv, .. } => {
                assert!(approx(kv, 20.0, 1e-5), "kv = {kv}")
            }
            other => panic!("expected Position flavor, got {other:?}"),
        }
    }

    #[test]
    fn position_torque_matches_pd_formula_byte_identical() {
        // Sanity: the Position flavor uses the exact ops the old PdServo
        // used — kp*(ctrl-len) - kv*vel then symmetric-clamp.
        let mut a = Actuator::position(0, 100.0, 0.0, 5.0, 0.0);
        a.ctrl = 10.0; // huge error → raw = 1000
        assert!(approx(a.torque(0.0, 0.0), 5.0, 1e-6));
        a.ctrl = -10.0;
        assert!(approx(a.torque(0.0, 0.0), -5.0, 1e-6));
        // force_range <= 0 disables clamp.
        let mut a = Actuator::position(0, 100.0, 0.0, 0.0, 0.0);
        a.ctrl = 10.0;
        assert!(approx(a.torque(0.0, 0.0), 1000.0, 1e-6));
    }

    #[test]
    fn velocity_torque_matches_hand_computation() {
        // kv=5, ctrl=2, vel=0.5 → 5*(2-0.5) = 7.5
        let mut a = Actuator::velocity(0, 5.0, 0.0);
        a.ctrl = 2.0;
        assert!(approx(a.torque(0.0, 0.5), 7.5, 1e-6));
        // Symmetric clamp @ ±3 binds.
        let mut a = Actuator::velocity(0, 5.0, 3.0);
        a.ctrl = 2.0;
        assert!(approx(a.torque(0.0, 0.5), 3.0, 1e-6));
    }

    #[test]
    fn motor_torque_is_gear_times_ctrl() {
        let mut a = Actuator::motor(0, 4.0, 0.0);
        a.ctrl = 2.5;
        assert!(approx(a.torque(0.0, 0.0), 10.0, 1e-6));
        // Symmetric clamp binds.
        let mut a = Actuator::motor(0, 4.0, 3.0);
        a.ctrl = 2.5;
        assert!(approx(a.torque(0.0, 0.0), 3.0, 1e-6));
    }

    #[test]
    fn general_position_shape_matches_kp_kv_algebra() {
        // A general actuator with position-shaped params should agree with
        // the Position shorthand up to f32 rounding — same algebra, different
        // op order.
        let kp = 100.0f32;
        let kv = 20.0f32;
        let ctrl = 0.3f32;
        let q = 0.05f32;
        let qdot = -0.02f32;
        let mut g = Actuator::general(
            0,
            GainType::Fixed,
            [kp, 0.0, 0.0],
            BiasType::Affine,
            [0.0, -kp, -kv],
            1.0,
            DynType::None,
            [0.0],
            None,
            None,
        );
        g.ctrl = ctrl;
        let mut p = Actuator::position(0, kp, kv, 0.0, ctrl);
        p.ctrl = ctrl;
        let tg = g.torque(q, qdot);
        let tp = p.torque(q, qdot);
        assert!(
            (tg - tp).abs() < 1e-4,
            "general vs position mismatch: {tg} vs {tp}"
        );
    }

    #[test]
    fn filter_ctrl_hold_matches_forward_euler_closed_form() {
        // ctrl held at 1.0 from act = 0, tau = 0.1, dt = 0.001. Forward
        // Euler recurrence: a_{n+1} = a_n + dt/tau * (1 - a_n) with
        // dt/tau = 0.01, closed form a_n = 1 - (1 - 0.01)^n = 1 - 0.99^n.
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
        a.ctrl = 1.0;
        let dt = 0.001f32;
        // 100 steps: a ≈ 1 - 0.99^100 ≈ 1 - 0.36603 = 0.63397.
        for _ in 0..100 {
            a.integrate_activation(dt);
        }
        // Closed form to f32: (1 - 0.99f32.powi(100)) — but this crate is
        // libm-free, so hand-computed reference value here.
        let expected = 1.0 - 0.36603_f32;
        assert!(
            (a.act - expected).abs() < 1e-3,
            "act={} expected≈{}",
            a.act,
            expected
        );
    }

    #[test]
    fn filter_signal_is_act_not_ctrl() {
        // ctrl = 5, act = 2. gain = fixed 1.0, bias = none → torque = act.
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
        a.act = 2.0;
        assert!(approx(a.torque(0.0, 0.0), 2.0, 1e-6));
    }

    #[test]
    fn ctrl_clamp_binds_before_force_clamp() {
        // Motor with gear=10, ctrl=100, ctrl_range=(-2,2), force_range=(-25,25).
        // After ctrl clamp: u=2 → F = 10*2 = 20 (under force cap → no bind).
        // Without ctrl clamp: F = 10*100 = 1000, force clamp gives 25.
        let mut a = Actuator::motor(0, 10.0, 0.0);
        a.ctrl = 100.0;
        a.ctrl_range = Some((-2.0, 2.0));
        a.force_range = Some((-25.0, 25.0));
        assert!(approx(a.torque(0.0, 0.0), 20.0, 1e-6));
        // Bump ctrl clamp so force clamp is the binder.
        a.ctrl_range = Some((-100.0, 100.0));
        assert!(approx(a.torque(0.0, 0.0), 25.0, 1e-6));
    }

    #[test]
    fn clamp_symmetric_passes_negative_cap_through() {
        assert!(approx(clamp_symmetric(42.0, -1.0), 42.0, 1e-6));
        assert!(approx(clamp_symmetric(-42.0, 0.0), -42.0, 1e-6));
    }

    #[test]
    fn clamp_range_matches_clamp_symmetric() {
        // clamp_range(x, -cap, cap) must be bit-equal to clamp_symmetric(x, cap)
        // for cap > 0 — this is what buys position-flavor byte-identity when
        // the storage flows through the shared Option<(lo,hi)> field.
        for cap in [1.0f32, 2.5, 100.0] {
            for x in [-200.0f32, -3.0, -0.5, 0.0, 0.5, 3.0, 200.0] {
                let s = clamp_symmetric(x, cap);
                let r = clamp_range(x, -cap, cap);
                assert_eq!(s.to_bits(), r.to_bits(), "cap={cap} x={x}");
            }
        }
    }

    #[test]
    fn reflected_inertia_point_mass_matches_formula() {
        let i = reflected_inertia_point_mass(2.0, 0.5, 0.05);
        assert!(approx(i, 0.55, 1e-6));
    }
}
