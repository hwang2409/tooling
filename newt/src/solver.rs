//! MuJoCo soft-constraint contact model and PGS solver (v1 tier 4).
//!
//! Full derivation lives in `docs/solver.md`. In brief: contacts and
//! joint-range limits are modeled as *inequality constraints* on a
//! generalized velocity `qdot`. Each constraint contributes
//!
//! - a Jacobian row `J_i` (1 × nv) — the linear map from `qdot` to the
//!   constraint's scalar velocity,
//! - a violation `r_i` (m for a contact along the normal, rad or m for a
//!   joint limit), and
//! - a pair `(solref, solimp)` that parameterizes the reference
//!   acceleration `a_ref(r, r_dot)` and the impedance `d(r) ∈ (0, 1]`.
//!
//! Assembled together the constraint law is
//!
//! ```text
//!     A · f + b = 0        (for the acceptance set defined by projections)
//!     A = J · M⁻¹ · Jᵀ + R
//!     R = diag((1 − d) / d) · diag(A)              # MuJoCo's regularization
//!     b = J · qdot_free + a_ref · dt               # residual velocity + reference term
//! ```
//!
//! where `qdot_free` is the generalized velocity that would arise from all
//! non-constraint forces over one timestep (`qdot_free = qdot + M⁻¹ (τ_ext -
//! h(q, qdot)) · dt`). The projection sets are per-constraint: normal /
//! limit forces are non-negative; friction is bounded by the cone.
//!
//! We solve this with a fixed-iteration Projected Gauss-Seidel sweep. The
//! iteration count is a model parameter — determinism outranks convergence
//! sensitivity in this engine, so we do not early-exit.
//!
//! # Determinism
//!
//! Zero platform libm. The impedance sigmoid restricts `power` to a
//! documented integer ≥ 1 (iterated multiplication). Constraint ordering is
//! total: contacts inherit the narrow-phase ordering (already deterministic
//! per `crate::contact`); joint limits are ordered by tree index then link
//! index then side (low, then high). The PGS sweep visits constraints in
//! this exact order every iteration.
//!
//! # Scope this ticket
//!
//! - condim 1 (frictionless) and condim 3 (normal + 2 tangents), with both
//!   pyramidal and elliptic cone options.
//! - Constraint-based joint limits for hinge and slide (ball still deferred).
//! - condim 4 / 6 (torsional / rolling) land with the equality-constraints
//!   ticket.
//! - RK4 integrator + solve-once-per-step (ZOH): we compute the solver
//!   forces at the START of the step and hold them constant across the four
//!   RK4 stages. MuJoCo does one Euler step per solve; keeping RK4 while
//!   solving once per step is a documented deviation (see `docs/solver.md`,
//!   "Once-per-step under RK4"). Revisiting the integrator is a v3 item.

// ---------------------------------------------------------------------------
// Solver-wide configuration
// ---------------------------------------------------------------------------

/// Which contact/constraint model the world uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverMode {
    /// v0 penalty spring-damper. Existing default; all pre-v1-tier-4 goldens
    /// stay byte-identical under this mode.
    Penalty,
    /// MuJoCo soft-constraint model solved by PGS.
    Pgs,
}

/// Friction cone parameterization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConeKind {
    /// Two independent per-tangent bounds `|f_t_k| ≤ μ · f_n`. Projects to a
    /// square inscribed in the true circular Coulomb cone (conservative:
    /// slightly over-friction on axis-aligned slip, slightly under on 45°
    /// diagonal slip). Cheap: one clamp per tangent.
    Pyramidal,
    /// True circular Coulomb cone `‖f_t‖ ≤ μ · f_n`. Projection is
    /// analytical (`f_t ← f_t · min(1, μ · f_n / ‖f_t‖)`). One projection
    /// per contact instead of per axis.
    Elliptic,
}

/// World-level solver configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig {
    /// Which model to use. Default `Penalty` keeps every existing golden
    /// byte-identical under CI.
    pub mode: SolverMode,
    /// Fixed number of PGS sweeps per step. Higher = tighter convergence,
    /// same runtime cost per iteration. Determinism outranks early-exit
    /// convergence sensitivity here: we always run exactly this many.
    pub iterations: u32,
    /// Cone parameterization (only consulted when `mode == Pgs`).
    pub cone: ConeKind,
}

impl SolverConfig {
    pub const DEFAULT: Self = Self {
        mode: SolverMode::Penalty,
        iterations: 20,
        cone: ConeKind::Pyramidal,
    };
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ---------------------------------------------------------------------------
// SolImp — the 5-parameter impedance sigmoid
// ---------------------------------------------------------------------------

/// MuJoCo-style 5-parameter impedance profile `d(r) ∈ [dmin, dmax]`.
///
/// The impedance `d ∈ (0, 1)` splits the "generalized force to apply this
/// step" into a fraction `d` that goes through the constraint solver as a
/// hard-ish projection (into the cone / into the non-negative half line)
/// and a fraction `1 − d` that appears as the regularization on the
/// diagonal of `A`. Stiffer contacts (large `d`) let very little violation
/// through; soft contacts (small `d`) look like a spring.
///
/// The sigmoid `y(x)` is a piecewise power interpolation on `x = |r|/width`:
///
/// ```text
///     y(x) = 0                                    for x ≤ 0
///     y(x) = midpoint · (x / midpoint)^power      for 0 < x < midpoint
///     y(x) = 1 − (1 − midpoint) · ((1 − x) / (1 − midpoint))^power   for midpoint ≤ x < 1
///     y(x) = 1                                    for x ≥ 1
/// ```
///
/// Both pieces agree at `x = midpoint` (value = midpoint, first derivative =
/// `power`), so the sigmoid is C¹ everywhere in (0, 1). Then
/// `d(r) = dmin + (dmax − dmin) · y(x)`.
///
/// # Power constraint
///
/// `power` is a documented integer ≥ 1 (no libm; iterated multiplication).
/// MuJoCo's default is `power = 2` — quadratic ramp, moderate stiffness
/// increase near the width — and that is our default too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolImp {
    /// Impedance at zero violation (`d(0)`). Must be in `(0, 1)`. Default
    /// `0.9` — most of the constraint acts through projection even at first
    /// contact; the sigmoid climbs to `dmax` as violation deepens.
    pub dmin: f32,
    /// Impedance at (or past) `width` violation (`d(r ≥ width)`). Must be
    /// in `(dmin, 1)`. Default `0.95`.
    pub dmax: f32,
    /// Reference violation width in whatever units `r` uses (m for contact
    /// normals, rad or m for limits). Default `0.001` m — the sigmoid
    /// reaches `dmax` at a 1 mm penetration.
    pub width: f32,
    /// Sigmoid midpoint in `(0, 1)`. The point where the two power branches
    /// meet. Default `0.5`.
    pub midpoint: f32,
    /// Integer power ≥ 1 on the interpolation. Default `2` (matches MuJoCo
    /// defaults).
    pub power: u32,
}

impl SolImp {
    pub const DEFAULT: Self = Self {
        dmin: 0.9,
        dmax: 0.95,
        width: 0.001,
        midpoint: 0.5,
        power: 2,
    };

    pub const fn new(dmin: f32, dmax: f32, width: f32, midpoint: f32, power: u32) -> Self {
        Self {
            dmin,
            dmax,
            width,
            midpoint,
            power,
        }
    }

    /// Validate a SolImp. `Err(msg)` if any field is out of range. Used by
    /// the loader and by [`combine_solimp`] once the combine rule has run.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0 < self.dmin && self.dmin < 1.0) {
            return Err(format!("solimp.dmin must be in (0, 1); got {}", self.dmin));
        }
        if !(self.dmin < self.dmax && self.dmax < 1.0) {
            return Err(format!(
                "solimp.dmax must be in (dmin, 1); got dmin={} dmax={}",
                self.dmin, self.dmax
            ));
        }
        if self.width <= 0.0 {
            return Err(format!("solimp.width must be > 0; got {}", self.width));
        }
        if !(0.0 < self.midpoint && self.midpoint < 1.0) {
            return Err(format!(
                "solimp.midpoint must be in (0, 1); got {}",
                self.midpoint
            ));
        }
        if self.power == 0 {
            return Err("solimp.power must be ≥ 1 (integer)".to_string());
        }
        Ok(())
    }
}

/// Per-parameter minimum for two SolImps at a shared contact. Same
/// rationale as [`crate::geom::combine_solref`] — the "stronger" setting
/// wins: smaller `dmin`/`dmax` → less impedance (softer); smaller `width` →
/// engages sooner. We take the min so a designer who dials in one geom's
/// impedance sees it applied at the contact rather than being averaged
/// away. Determinism-friendly: no float comparisons that tie ambiguously.
///
/// Midpoint and power fall back to the FIRST geom's values (arbitrary but
/// stable), documented so the loader/tests can rely on it.
pub fn combine_solimp(a: SolImp, b: SolImp) -> SolImp {
    SolImp {
        dmin: fmin(a.dmin, b.dmin),
        dmax: fmin(a.dmax, b.dmax),
        width: fmin(a.width, b.width),
        midpoint: a.midpoint,
        power: a.power,
    }
}

#[inline]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

// ---------------------------------------------------------------------------
// Sigmoid, impedance, reference acceleration
// ---------------------------------------------------------------------------

/// Deterministic integer power `x^n` for `n ≥ 1` via iterated multiplication.
/// No libm. Exact under IEEE for `n` up to a few dozen; we use `n ≤ 5` in
/// practice.
pub fn powi(x: f32, n: u32) -> f32 {
    let mut result = 1.0f32;
    for _ in 0..n {
        result *= x;
    }
    result
}

/// The MuJoCo sigmoid `y(x, power, midpoint)`. Returns `0` for `x ≤ 0`, `1`
/// for `x ≥ 1`, otherwise the piecewise power interpolation described on
/// [`SolImp`]. C¹ at `x = midpoint`.
pub fn sigmoid(x: f32, power: u32, midpoint: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x < midpoint {
        // y = midpoint · (x/midpoint)^power
        midpoint * powi(x / midpoint, power)
    } else {
        // y = 1 − (1 − midpoint) · ((1 − x)/(1 − midpoint))^power
        1.0 - (1.0 - midpoint) * powi((1.0 - x) / (1.0 - midpoint), power)
    }
}

/// Impedance `d(r)` for a given violation `r`. Uses `|r|` — negative
/// violations (constraint currently INSIDE the acceptance region) still
/// yield an impedance in case they are treated as pushing back toward the
/// boundary.
pub fn impedance(violation: f32, s: SolImp) -> f32 {
    let r = if violation >= 0.0 {
        violation
    } else {
        -violation
    };
    let x = r / s.width;
    let y = sigmoid(x, s.power, s.midpoint);
    s.dmin + (s.dmax - s.dmin) * y
}

/// Reference acceleration `a_ref(r, r_dot)` for the SolRef `(timeconst,
/// dampratio)` parameterization.
///
/// Derivation: model the constraint's temporal behavior as a critically-ish
/// damped second-order response with natural time constant `timeconst`. The
/// desired constraint-space acceleration that would drive `r` to zero is
///
/// ```text
///     a_ref = -(2 / timeconst) · dampratio · r_dot − (1 / timeconst²) · r
/// ```
///
/// which corresponds to MuJoCo's `solref = (timeconst, dampratio)` positive
/// form (`solref` field 1 positive → time-constant seconds; negative would
/// be direct stiffness — we always take the positive branch here, matching
/// the [`crate::geom::SolRef`] contract). Convention: `r > 0` means
/// "constraint violated by r units"; a positive `a_ref` accelerates the
/// constraint toward violation, negative toward the acceptance set. So the
/// signs above drive `r → 0` for a critical response.
pub fn reference_accel(violation: f32, violation_dot: f32, solref: crate::geom::SolRef) -> f32 {
    let tc = solref.timeconst;
    let z = solref.dampratio;
    // -(2 z / tc) r_dot - (1 / tc²) r
    -(2.0 * z / tc) * violation_dot - (1.0 / (tc * tc)) * violation
}

// ---------------------------------------------------------------------------
// Tests (unit)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn sigmoid_boundaries() {
        approx(sigmoid(-1.0, 2, 0.5), 0.0, 0.0);
        approx(sigmoid(0.0, 2, 0.5), 0.0, 0.0);
        approx(sigmoid(1.0, 2, 0.5), 1.0, 0.0);
        approx(sigmoid(2.0, 2, 0.5), 1.0, 0.0);
    }

    #[test]
    fn sigmoid_at_midpoint_agrees() {
        // Both branches evaluate to midpoint at x == midpoint.
        for &mid in &[0.25f32, 0.5, 0.75] {
            for &power in &[1u32, 2, 3, 4] {
                approx(sigmoid(mid, power, mid), mid, 1e-6);
            }
        }
    }

    #[test]
    fn sigmoid_linear_when_power_1() {
        // power = 1 collapses both branches to the identity.
        for &mid in &[0.3f32, 0.5, 0.7] {
            for &x in &[0.1f32, 0.4, 0.6, 0.9] {
                approx(sigmoid(x, 1, mid), x, 1e-6);
            }
        }
    }

    #[test]
    fn sigmoid_power_2_hand_computed() {
        // mid = 0.5, power = 2, x = 0.25: y = 0.5 · (0.25/0.5)^2 = 0.5 · 0.25 = 0.125
        approx(sigmoid(0.25, 2, 0.5), 0.125, 1e-6);
        // mid = 0.5, power = 2, x = 0.75: y = 1 − 0.5 · ((0.25)/0.5)^2 = 1 − 0.125 = 0.875
        approx(sigmoid(0.75, 2, 0.5), 0.875, 1e-6);
    }

    #[test]
    fn impedance_endpoints() {
        let s = SolImp::DEFAULT;
        approx(impedance(0.0, s), s.dmin, 1e-6);
        approx(impedance(2.0 * s.width, s), s.dmax, 1e-6);
    }

    #[test]
    fn impedance_midpoint_hand_computed() {
        // width = 0.001, midpoint = 0.5, power = 2, dmin = 0.9, dmax = 0.95.
        // At violation = 0.00025 (x = 0.25), y = 0.125, d = 0.9 + 0.05*0.125 = 0.90625.
        let s = SolImp::DEFAULT;
        approx(impedance(0.00025, s), 0.90625, 1e-6);
    }

    #[test]
    fn reference_accel_signs() {
        // Positive violation, zero rate: a_ref should be negative (push back
        // toward acceptance set).
        let solref = crate::geom::SolRef::new(0.02, 1.0);
        let a = reference_accel(0.001, 0.0, solref);
        assert!(a < 0.0);
        // At -1 / tc² · 0.001, exactly.
        approx(a, -0.001 / (0.02 * 0.02), 1e-6);
        // With positive violation_dot (getting worse), damping term adds
        // more negative acceleration.
        let a2 = reference_accel(0.001, 0.1, solref);
        assert!(a2 < a, "damping term must push a_ref more negative");
    }

    #[test]
    fn combine_solimp_takes_component_min() {
        let a = SolImp::new(0.8, 0.9, 0.001, 0.5, 2);
        let b = SolImp::new(0.7, 0.85, 0.002, 0.3, 3);
        let c = combine_solimp(a, b);
        assert_eq!(c.dmin, 0.7);
        assert_eq!(c.dmax, 0.85);
        assert_eq!(c.width, 0.001);
        // Midpoint and power fall back to `a`.
        assert_eq!(c.midpoint, 0.5);
        assert_eq!(c.power, 2);
    }

    #[test]
    fn solimp_validate_catches_bad_ranges() {
        assert!(SolImp::new(0.0, 0.9, 0.001, 0.5, 2).validate().is_err());
        assert!(SolImp::new(0.9, 0.9, 0.001, 0.5, 2).validate().is_err());
        assert!(SolImp::new(0.5, 0.9, 0.0, 0.5, 2).validate().is_err());
        assert!(SolImp::new(0.5, 0.9, 0.001, 1.0, 2).validate().is_err());
        assert!(SolImp::new(0.5, 0.9, 0.001, 0.5, 0).validate().is_err());
        assert!(SolImp::DEFAULT.validate().is_ok());
    }
}
