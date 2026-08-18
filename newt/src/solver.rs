//! MuJoCo soft-constraint contact model with PGS and Newton solvers.
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
//! PGS remains the established dual sweep. Newton uses the same assembled
//! rows and minimizes the dense regularized quadratic in `crate::newton`.
//! Both have fixed iteration caps and deterministic cost convergence.
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
//! - condim 1 (frictionless) and condim 3 (normal + 2 tangents). PGS accepts
//!   both cone options; Newton accepts pyramidal cones in this ticket.
//! - Constraint-based joint limits for hinge and slide (ball still deferred).
//! - condim 4 / 6 (torsional / rolling) use the same row assembly for free
//!   bodies and trees.
//! - Tree-involved contacts use one world-level joint-space system. This
//!   includes tree-vs-static, tree-vs-body, and tree-vs-tree contacts.
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
    /// v0 penalty spring-damper. Existing default; scenes without a changed
    /// contact manifold retain their previous trajectories.
    Penalty,
    /// MuJoCo soft-constraint model solved by PGS.
    Pgs,
    /// MuJoCo soft-constraint model solved by dense Newton iterations.
    ///
    /// Newton is opt-in. The legacy penalty default and the PGS path remain
    /// unchanged for scenes outside a changed contact manifold.
    Newton,
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

const NEWTON_ELLIPTIC_ERROR: &str =
    "solver=newton with cone=elliptic is not supported yet; use cone=pyramidal";

/// World-level solver configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig {
    /// Which model to use. Default `Penalty` keeps the legacy force path.
    pub mode: SolverMode,
    /// Fixed number of PGS sweeps per step. Higher = tighter convergence,
    /// same runtime cost per iteration. Determinism outranks early-exit
    /// convergence sensitivity here: we always run exactly this many.
    pub iterations: u32,
    /// Cone parameterization. Newton currently accepts pyramidal cones only;
    /// elliptic Newton models are rejected during loading.
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

impl SolverConfig {
    /// Validate configuration combinations that loaders can reject before a
    /// simulation starts.
    pub fn validate(&self) -> Result<(), String> {
        if self.mode == SolverMode::Newton && self.cone == ConeKind::Elliptic {
            return Err(NEWTON_ELLIPTIC_ERROR.to_string());
        }
        if self.iterations == 0 {
            return Err("solver iterations must be >= 1".to_string());
        }
        Ok(())
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
    impedance_at_position(-violation, 0.0, s)
}

/// MuJoCo's `getimpedance`: evaluate the sigmoid from signed row position
/// and margin before taking the absolute normalized distance.
fn impedance_at_position(position: f32, margin: f32, s: SolImp) -> f32 {
    let s = effective_solimp(s);
    let mut x = (position - margin) / s.width;
    if x < 0.0 {
        x = -x;
    }
    let y = sigmoid(x, s.power, s.midpoint);
    s.dmin + (s.dmax - s.dmin) * y
}

const MJ_MIN_IMP: f32 = 1.0e-4;
const MJ_MAX_IMP: f32 = 0.9999;

#[inline]
fn clamp_impedance(value: f32) -> f32 {
    value.clamp(MJ_MIN_IMP, MJ_MAX_IMP)
}

#[inline]
fn effective_solimp(s: SolImp) -> SolImp {
    SolImp {
        dmin: clamp_impedance(s.dmin),
        dmax: clamp_impedance(s.dmax),
        width: if s.width > 0.0 { s.width } else { 0.0 },
        midpoint: clamp_impedance(s.midpoint),
        power: s.power.max(1),
    }
}

#[inline]
fn safe_solref(solref: SolRef, dt: f32) -> SolRef {
    if !solref.is_direct() && solref.timeconst < 2.0 * dt {
        SolRef::new(2.0 * dt, solref.dampratio)
    } else {
        solref
    }
}

/// Compute MuJoCo's reference acceleration for one soft-constraint row.
///
/// For positive solref, MuJoCo derives the coefficients from the maximum
/// impedance and the row's current impedance:
///
/// ```text
/// b = 2 / (dmax · timeconst)
/// k = d(r) / (dmax² · timeconst² · dampratio²)
/// a_ref = -b · r_dot - k · r
/// ```
///
/// Negative solref uses direct `(stiffness, damping)` values. The stored
/// values are negative, so `(-timeconst, -dampratio)` are used verbatim.
/// This helper is shared by contact, friction, limit, and equality rows.
pub fn reference_accel(
    position: f32,
    velocity: f32,
    solref: crate::geom::SolRef,
    solimp: SolImp,
) -> f32 {
    let (b, k, _) = reference_coefficients(position, solref, solimp);
    -b * velocity - k * position
}

fn reference_coefficients(
    position: f32,
    solref: crate::geom::SolRef,
    solimp: SolImp,
) -> (f32, f32, f32) {
    let solimp = effective_solimp(solimp);
    let d = impedance_at_position(position, 0.0, solimp);
    let (b, k) = if solref.is_direct() {
        (
            -solref.dampratio / solimp.dmax,
            -solref.timeconst / (solimp.dmax * solimp.dmax),
        )
    } else {
        let tc = solref.timeconst;
        let dmax = solimp.dmax;
        (
            2.0 / (dmax * tc),
            d / (dmax * dmax * tc * tc * solref.dampratio * solref.dampratio),
        )
    };
    (b, k, d)
}

// ---------------------------------------------------------------------------
// Elliptic-cone projection (public helper — the ticket lists it as a test
// anchor; keep it here so both PGS and standalone tests share one impl)
// ---------------------------------------------------------------------------

/// Project a 2-vector `(ft1, ft2)` onto the elliptic Coulomb friction disc
/// of radius `mu · fn`. Analytical: unchanged when inside, radially rescaled
/// to the boundary when outside. Returns the projected `(ft1, ft2)`.
///
/// The `fn` argument is the normal impulse magnitude (already non-negative);
/// a zero or negative `fn` collapses the disc to a point at the origin, so
/// the projection returns `(0, 0)`.
pub fn project_elliptic(ft1: f32, ft2: f32, mu: f32, normal_impulse: f32) -> (f32, f32) {
    if normal_impulse <= 0.0 || mu <= 0.0 {
        return (0.0, 0.0);
    }
    let cap = mu * normal_impulse;
    let mag2 = ft1 * ft1 + ft2 * ft2;
    if mag2 <= cap * cap {
        return (ft1, ft2);
    }
    // Rescale to the boundary. mag2 > 0 here (otherwise the ≤ branch above
    // caught it).
    let scale = cap / mag2.sqrt();
    (ft1 * scale, ft2 * scale)
}

/// Symmetric pyramidal clamp `|f| ≤ cap` on a single tangent axis. Matches
/// the `clamp_symmetric` used by the penalty pathway in `crate::world`.
pub fn project_pyramidal(f: f32, cap: f32) -> f32 {
    if f > cap {
        cap
    } else if f < -cap {
        -cap
    } else {
        f
    }
}

// ---------------------------------------------------------------------------
// Free-body constraint solver
// ---------------------------------------------------------------------------

use crate::body::Body;
use crate::contact::Contact;
use crate::equality::{DISTANCE_DEGENERATE_EPS, Equality};
use crate::geom::{Geom, GeomAttach, SolRef, combine_solref};
use crate::math::{Quat, Vec3};
use crate::world::tangent_basis;

/// Row geometry: linear (force applied at anchor point on each body, with
/// arm cross → torque) vs angular (torque-only about `dir_world`, no linear
/// force). Both share the per-body velocity accumulator; the projection
/// math differs by whether the arm participates.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RowGeom {
    Linear,
    Angular,
}

/// One assembled scalar constraint (a single row of `J`). Kept opaque —
/// callers get their results as per-body wrenches.
#[derive(Clone, Copy, Debug)]
struct ConstraintRow {
    /// Direction in world coordinates (unit-length). For a Linear row this
    /// is the point-velocity axis; for an Angular row it's the world-frame
    /// torque axis.
    dir_world: Vec3,
    /// Body A index (or `None` if A is static / world).
    body_a: Option<u32>,
    /// Body B index (or `None` if B is static / world).
    body_b: Option<u32>,
    /// Arm from body A's COM to the anchor point in world coords. Ignored
    /// for angular rows; zero for static.
    arm_a: Vec3,
    /// Arm from body B's COM to the anchor point in world coords.
    arm_b: Vec3,
    /// Diagonal regularization `R_ii = (1 - d) / d · A_ii`. Populated
    /// during assembly.
    reg: f32,
    /// Diagonal `A_ii + R_ii`. Populated during assembly.
    diag: f32,
    /// Bias `b_i = delta_v_free - a_ref * dt`, using the source reference
    /// acceleration directly.
    bias: f32,
    /// Row geometry.
    geom: RowGeom,
}

/// Per-body scratch: velocity offset accumulated from constraint impulses.
/// `dv_lin` in world coords, `dw_body` in body coords (so ω_body_new =
/// ω_body_start + dw_body + dw_body_from_free_step).
///
/// TODO(perf, biped-scale): every PGS-inner apply/read round-trips the
/// angular delta between body and world coords (see
/// [`apply_impulse_delta`] and [`row_residual`]). Storing dw in world
/// frame and computing torque contributions there would trade the
/// per-update pair of `rotate`/`inverse_rotate` calls for one less
/// rotation per constraint per iteration. Deferred — revisit before
/// wiring the biped (v2 tier 6) so the change lands with the workload
/// that would actually benefit.
#[derive(Clone, Copy, Debug, Default)]
struct BodyDelta {
    dv_lin_world: Vec3,
    dw_ang_body: Vec3,
}

/// PGS solve for the free-body pool.
///
/// Inputs:
/// - `bodies` — start-of-step body states (read-only).
/// - `geoms` — for looking up per-contact SolRef / SolImp / cone override /
///   condim on the geom pair.
/// - `contacts` — narrow-phase contacts touching only free bodies (or
///   free-body-vs-static). Cross-tree contacts must be filtered out by the
///   caller.
/// - `equalities` — free-body equalities (connect / weld / distance) that
///   apply to bodies in this pool. `JointCoupling` equalities are filtered
///   out here (they run in [`solve_tree_limits`] instead).
/// - `gravity`, `dt` — for computing `qdot_free` (Euler forward with
///   gravity + gyroscopic acceleration over one dt).
/// - `cone`, `iterations` — world solver config.
///
/// Returns per-body `(force_world, torque_world_at_com)` to be held constant
/// (ZOH) across the RK4 stages. Bodies not touched by any constraint receive
/// `(ZERO, ZERO)`.
///
/// # Row ordering (deterministic total order)
///
/// 1. Contact rows: narrow-phase order (caller-supplied). Within one
///    contact: `normal`, `t1`, `t2`, `torsion`, `roll1`, `roll2` — the
///    trailing rows appear only when condim ≥ 4 / 6. Torsion is Angular
///    about the normal; rolling rows are Angular about `t1` and `t2`.
/// 2. Equality rows: declaration order of `equalities`. Within one
///    equality:
///    - `Connect` → 3 Linear rows on world axes `(x, y, z)`.
///    - `Weld` → 3 Linear rows then 3 Angular rows, all on world axes.
///    - `Distance` → 1 Linear row on the current separation unit vector.
///    - `JointCoupling` is skipped here.
#[allow(clippy::too_many_arguments)]
pub fn solve_free_bodies(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    equalities: &[Equality],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> Vec<(Vec3, Vec3)> {
    let (wrenches, _) = solve_free_bodies_diag(
        bodies, geoms, contacts, equalities, gravity, dt, cone, iterations,
    );
    wrenches
}

/// Newton counterpart to [`solve_free_bodies`].  It assembles the same rows,
/// bias, and regularized Delassus matrix as PGS, then minimizes the convex
/// quadratic with deterministic pyramidal-cone Newton steps.
#[allow(clippy::too_many_arguments)]
pub fn solve_free_bodies_newton(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    equalities: &[Equality],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> Vec<(Vec3, Vec3)> {
    let (wrenches, _) = solve_free_bodies_newton_diag(
        bodies, geoms, contacts, equalities, gravity, dt, cone, iterations,
    );
    wrenches
}

/// Diagnostic variant of [`solve_free_bodies`]. Returns the same per-body
/// wrenches PLUS a `Vec<f32>` with one entry per input contact: the
/// per-contact normal force (impulse / dt) after the PGS sweep. Used by
/// the touch sensor to read the actual constraint-normal force applied to
/// a specific contact instead of approximating via the penalty formula.
///
/// The order of `contact_normal_forces` matches `contacts` one-to-one.
#[allow(clippy::too_many_arguments)]
pub fn solve_free_bodies_diag(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    equalities: &[Equality],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> (Vec<(Vec3, Vec3)>, Vec<f32>) {
    solve_free_bodies_diag_mode(
        bodies, geoms, contacts, equalities, gravity, dt, cone, iterations, false, None, None,
    )
}

/// Diagnostic Newton variant. The normal-force output has the same shape as
/// the PGS diagnostic API so touch sensors observe the force actually used by
/// the integrator.
#[allow(clippy::too_many_arguments)]
pub fn solve_free_bodies_newton_diag(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    equalities: &[Equality],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> (Vec<(Vec3, Vec3)>, Vec<f32>) {
    solve_free_bodies_diag_mode(
        bodies, geoms, contacts, equalities, gravity, dt, cone, iterations, true, None, None,
    )
}

/// Return the live Newton cost trace for one free-body constraint solve.
///
/// The first value is the zero-impulse cost. Later values are accepted line
/// search costs. An empty trace means that the input has no active rows.
#[allow(clippy::too_many_arguments)]
pub fn solve_free_bodies_newton_trace(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    equalities: &[Equality],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> Vec<f32> {
    let mut trace = Vec::new();
    let _ = solve_free_bodies_diag_mode(
        bodies,
        geoms,
        contacts,
        equalities,
        gravity,
        dt,
        cone,
        iterations,
        true,
        Some(&mut trace),
        None,
    );
    trace
}

/// One assembled soft-constraint row before the impulse solve.
#[derive(Clone, Copy, Debug)]
pub struct ConstraintRowDiagnostic {
    pub position: f32,
    pub velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
    pub impedance: f32,
    pub regularization: f32,
    pub reference_accel: f32,
}

/// Assemble free-body contact rows and return their source factors.
#[allow(clippy::too_many_arguments)]
pub fn diagnose_free_body_contact_rows(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> Vec<ConstraintRowDiagnostic> {
    let mut diagnostics = Vec::new();
    let _ = solve_free_bodies_diag_mode(
        bodies,
        geoms,
        contacts,
        &[],
        gravity,
        dt,
        cone,
        iterations,
        false,
        None,
        Some(&mut diagnostics),
    );
    diagnostics
}

#[allow(clippy::too_many_arguments)]
fn solve_free_bodies_diag_mode(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    equalities: &[Equality],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
    use_newton: bool,
    newton_cost_trace: Option<&mut Vec<f32>>,
    mut row_diagnostics: Option<&mut Vec<ConstraintRowDiagnostic>>,
) -> (Vec<(Vec3, Vec3)>, Vec<f32>) {
    if use_newton && cone == ConeKind::Elliptic {
        panic!("{NEWTON_ELLIPTIC_ERROR}");
    }
    let n_bodies = bodies.len();
    let mut wrenches = vec![(Vec3::ZERO, Vec3::ZERO); n_bodies];
    let mut contact_normal_forces = vec![0.0f32; contacts.len()];
    let has_free_eq = equalities.iter().any(|e| e.is_free_body());
    if (contacts.is_empty() && !has_free_eq) || dt <= 0.0 {
        return (wrenches, contact_normal_forces);
    }

    let mut rows: Vec<ConstraintRow> = Vec::new();
    let mut per_contact: Vec<PerContact> = Vec::with_capacity(contacts.len());
    let mut per_equality: Vec<PerEquality> = Vec::new();

    // Precompute free (Euler) velocity offset from gravity ONLY. Gyroscopic
    // torque affects angular velocity but is small over dt=5ms in our
    // tests and complicates b assembly; we fold it into ZOH via
    // world.step's later stages. For qdot_free within a single step:
    // dv_free_lin = gravity * dt (per body), dw_free_body = 0 (approx).
    // Matches the "solve once per step with ZOH" scope note in the module
    // docs.
    let dv_lin_free_per_body: Vec<Vec3> = (0..n_bodies).map(|_| gravity * dt).collect();
    let dw_body_free_per_body: Vec<Vec3> = vec![Vec3::ZERO; n_bodies];

    // ---- Contact blocks ---------------------------------------------------
    for (contact_index, c) in contacts.iter().enumerate() {
        let ga = &geoms[c.geom_a];
        let gb = &geoms[c.geom_b];
        let body_a = ga.body.map(|i| i as u32);
        let body_b = gb.body.map(|i| i as u32);
        assert!(
            ga.link.is_none() && gb.link.is_none(),
            "solve_free_bodies received a link-attached contact; caller must filter"
        );
        if body_a.is_none() && body_b.is_none() {
            continue;
        }
        let pen_active = c.penetration - c.gap;
        if pen_active <= 0.0 {
            continue;
        }
        let solref = combine_solref(ga.solref, gb.solref);
        let solimp = combine_solimp(ga.solimp, gb.solimp);
        let mu = crate::contact::combine_friction(ga.friction, gb.friction);
        let mu_torsion =
            crate::geom::combine_torsional_friction(ga.torsional_friction, gb.torsional_friction);
        let mu_roll =
            crate::geom::combine_rolling_friction(ga.rolling_friction, gb.rolling_friction);
        let condim = ga.condim.min(gb.condim);

        let n_world = c.normal_world;
        let (t1_world, t2_world) = tangent_basis(n_world);
        let contact_position = c.position_world;
        let arm_a = match body_a {
            Some(i) => contact_position - bodies[i as usize].position,
            None => Vec3::ZERO,
        };
        let arm_b = match body_b {
            Some(i) => contact_position - bodies[i as usize].position,
            None => Vec3::ZERO,
        };

        let start_row = rows.len() as u32;
        let row_directions = contact_row_directions(n_world, t1_world, t2_world, condim, cone, mu);
        for (dir, geom) in row_directions {
            rows.push(ConstraintRow {
                dir_world: dir,
                body_a,
                body_b,
                arm_a,
                arm_b,
                reg: 0.0,
                diag: 0.0,
                bias: 0.0,
                geom: if geom {
                    RowGeom::Angular
                } else {
                    RowGeom::Linear
                },
            });
        }

        if !(cone == ConeKind::Pyramidal && condim == 3) && condim >= 4 {
            // Torsional row about the contact normal. Angular.
            rows.push(ConstraintRow {
                dir_world: n_world,
                body_a,
                body_b,
                arm_a: Vec3::ZERO,
                arm_b: Vec3::ZERO,
                reg: 0.0,
                diag: 0.0,
                bias: 0.0,
                geom: RowGeom::Angular,
            });
        }
        if !(cone == ConeKind::Pyramidal && condim == 3) && condim >= 6 {
            // Rolling rows about the two tangents. Angular.
            for &dir in &[t1_world, t2_world] {
                rows.push(ConstraintRow {
                    dir_world: dir,
                    body_a,
                    body_b,
                    arm_a: Vec3::ZERO,
                    arm_b: Vec3::ZERO,
                    reg: 0.0,
                    diag: 0.0,
                    bias: 0.0,
                    geom: RowGeom::Angular,
                });
            }
        }

        per_contact.push(PerContact {
            contact_index,
            start_row,
            condim,
            solref,
            solimp,
            pen_active,
            mu_slide: mu,
            mu_torsion,
            mu_roll,
            h01_pair01: 0.0,
            h01_pair23: 0.0,
        });
    }

    // ---- Equality blocks --------------------------------------------------
    for eq in equalities.iter().filter(|e| e.is_free_body()) {
        push_equality_rows(eq, bodies, &mut rows, &mut per_equality);
    }

    let n_rows = rows.len();
    if n_rows == 0 {
        return (wrenches, contact_normal_forces);
    }

    // Cache I_world^-1 per body — used by row_body_diagonal /
    // apply_impulse_delta / row_residual on every sweep.
    let inv_i_world: Vec<crate::math::Mat3> = bodies
        .iter()
        .map(|b| {
            let r = b.orientation.to_mat3();
            r * b.inertia_body_inverse * r.transpose()
        })
        .collect();

    // ---- Compute A_ii, R_ii, bias for every row ---------------------------
    for pc in &per_contact {
        for k in 0..contact_block_n_rows(pc.condim, cone) {
            let ri = pc.start_row as usize + k;
            let a_ii = row_body_diagonal(&rows[ri], bodies, &inv_i_world);
            let d = impedance_at_position(-pc.pen_active, 0.0, pc.solimp);
            let reg = contact_regularization(pc, &rows[ri], bodies, &inv_i_world, d, cone);
            rows[ri].reg = reg;
            rows[ri].diag = a_ii + reg;
        }
        for k in 0..contact_block_n_rows(pc.condim, cone) {
            let ri = pc.start_row as usize + k;
            let v_cur = row_current_velocity(&rows[ri], bodies);
            let dv_free = row_free_step_velocity(
                &rows[ri],
                &dv_lin_free_per_body,
                &dw_body_free_per_body,
                bodies,
            );
            let position = if (cone == ConeKind::Pyramidal && pc.condim == 3) || k == 0 {
                -pc.pen_active
            } else {
                0.0
            };
            let solref = safe_solref(pc.solref, dt);
            let a_ref = reference_accel(position, v_cur, solref, pc.solimp);
            // MuJoCo solves the acceleration equation J*qacc - aref = 0.
            // The impulse changes velocity, so the velocity-form residual is
            // the free velocity change minus aref*dt. J*qvel itself is
            // already represented in aref's damping term.
            rows[ri].bias = dv_free - a_ref * dt;
            if let Some(diagnostics) = row_diagnostics.as_deref_mut() {
                let (damping, stiffness, impedance) =
                    reference_coefficients(position, solref, pc.solimp);
                diagnostics.push(ConstraintRowDiagnostic {
                    position,
                    velocity: v_cur,
                    stiffness,
                    damping,
                    impedance,
                    regularization: rows[ri].reg,
                    reference_accel: a_ref,
                });
            }
        }
    }

    for pe in &per_equality {
        for k in 0..pe.n_rows as usize {
            let ri = pe.start_row as usize + k;
            let a_ii = row_body_diagonal(&rows[ri], bodies, &inv_i_world);
            // Row construction has already normalized signs so
            // `pe.violations[k] >= 0` and `dir_world` points such that
            // `+f impulse` reduces the residual (the "escape" convention
            // used by contact normal rows). r_dot = -J·qdot follows the
            // same convention, so the contact bias formula applies
            // unchanged.
            let r = pe.violations[k];
            let d = impedance(r, pe.solimp);
            let reg = if d > 0.0 { (1.0 - d) / d * a_ii } else { 0.0 };
            rows[ri].reg = reg;
            rows[ri].diag = a_ii + reg;
            let v_cur = row_current_velocity(&rows[ri], bodies);
            let dv_free = row_free_step_velocity(
                &rows[ri],
                &dv_lin_free_per_body,
                &dw_body_free_per_body,
                bodies,
            );
            let solref = safe_solref(pe.solref, dt);
            let a_ref = reference_accel(-r, v_cur, solref, pe.solimp);
            rows[ri].bias = dv_free - a_ref * dt;
        }
    }

    // ---- Precompute cross-response terms for pyramidal facet pairs -------
    // `row_cross_response` computes the (i,j) entry of `A = J M⁻¹ Jᵀ` for a
    // pair of rows. It depends only on the rows and the start-of-step body
    // state, not on iteration impulses, so we can precompute the two values
    // each condim=3 pyramidal contact needs and reuse them across every PGS
    // sweep. Previously `pgs_step_pyramidal_pair` called `row_cross_response`
    // on every iteration and every contact, and each call allocated a fresh
    // `Vec<BodyDelta>` scratch.
    if cone == ConeKind::Pyramidal {
        // Scratch buffer reused for both computations to avoid the per-call
        // Vec allocation from the old `row_cross_response` implementation.
        let mut response_scratch = vec![BodyDelta::default(); n_bodies];
        for pc in per_contact.iter_mut() {
            if pc.condim != 3 {
                continue;
            }
            let sr = pc.start_row as usize;
            pc.h01_pair01 = row_cross_response_into(
                &rows[sr],
                &rows[sr + 1],
                bodies,
                &inv_i_world,
                &mut response_scratch,
            );
            pc.h01_pair23 = row_cross_response_into(
                &rows[sr + 2],
                &rows[sr + 3],
                bodies,
                &inv_i_world,
                &mut response_scratch,
            );
        }
    }

    // ---- Solve the shared regularized system -----------------------------
    // Newton and PGS consume the same H = J M⁻¹ Jᵀ + R and bias. This keeps
    // the solver switch a numerical method choice, not a second constraint
    // model.
    let impulses = if use_newton {
        let result = solve_free_body_newton_impulses(
            &rows,
            &per_contact,
            n_bodies,
            bodies,
            &inv_i_world,
            iterations,
            cone,
        );
        if let Some(trace) = newton_cost_trace {
            *trace = result.costs.clone();
        }
        result.solution
    } else {
        let mut impulses = vec![0.0f32; n_rows];
        let mut body_delta = vec![BodyDelta::default(); n_bodies];

        for _iter in 0..iterations {
            // Contact blocks first. Row sweep order within a contact:
            //   normal → t1, t2 (sliding cone cap on impulses[n_row])
            //   → torsion (cone cap: mu_torsion · f_n)
            //   → roll1, roll2 (cone cap: mu_roll · f_n)
            // The just-updated normal impulse drives all cone caps this
            // iteration — same "normal-first" idea as the condim 3 loop.
            for pc in &per_contact {
                let n_row = pc.start_row as usize;
                // NORMAL: half-line projection.
                pgs_step_non_negative(
                    &rows,
                    n_row,
                    &mut impulses,
                    &mut body_delta,
                    bodies,
                    &inv_i_world,
                );
                if cone == ConeKind::Pyramidal && pc.condim == 3 {
                    pgs_step_pyramidal_pair(
                        &rows,
                        pc.start_row as usize,
                        pc.h01_pair01,
                        &mut impulses,
                        &mut body_delta,
                        bodies,
                        &inv_i_world,
                    );
                    pgs_step_pyramidal_pair(
                        &rows,
                        pc.start_row as usize + 2,
                        pc.h01_pair23,
                        &mut impulses,
                        &mut body_delta,
                        bodies,
                        &inv_i_world,
                    );
                } else if pc.condim >= 3 {
                    let cap_normal = impulses[n_row];
                    let t1 = pc.start_row as usize + 1;
                    let t2 = pc.start_row as usize + 2;
                    pgs_step_pair_cone(
                        &rows,
                        t1,
                        t2,
                        cap_normal,
                        pc.mu_slide,
                        cone,
                        &mut impulses,
                        &mut body_delta,
                        bodies,
                        &inv_i_world,
                    );
                    if pc.condim >= 4 {
                        let torsion = pc.start_row as usize + 3;
                        pgs_step_scalar_cap(
                            &rows,
                            torsion,
                            pc.mu_torsion * cap_normal,
                            &mut impulses,
                            &mut body_delta,
                            bodies,
                            &inv_i_world,
                        );
                    }
                    if pc.condim >= 6 {
                        let r1 = pc.start_row as usize + 4;
                        let r2 = pc.start_row as usize + 5;
                        pgs_step_pair_cone(
                            &rows,
                            r1,
                            r2,
                            cap_normal,
                            pc.mu_roll,
                            cone,
                            &mut impulses,
                            &mut body_delta,
                            bodies,
                            &inv_i_world,
                        );
                    }
                }
            }
            // Equality blocks — bilateral update, no clamp.
            for pe in &per_equality {
                for k in 0..pe.n_rows as usize {
                    let ri = pe.start_row as usize + k;
                    pgs_step_bilateral(
                        &rows,
                        ri,
                        &mut impulses,
                        &mut body_delta,
                        bodies,
                        &inv_i_world,
                    );
                }
            }
        }
        impulses
    };

    // ---- Extract per-body wrenches ----------------------------------------
    // Linear rows: force at anchor → force + arm × force per body. Angular
    // rows: pure torque along `dir_world`.
    for (ri, row) in rows.iter().enumerate() {
        let f_impulse = impulses[ri];
        if f_impulse == 0.0 {
            continue;
        }
        match row.geom {
            RowGeom::Linear => {
                let force_on_a_world = row.dir_world * (f_impulse / dt);
                if let Some(i) = row.body_a {
                    let (f, tau) = &mut wrenches[i as usize];
                    *f += force_on_a_world;
                    *tau += row.arm_a.cross(force_on_a_world);
                }
                if let Some(i) = row.body_b {
                    let force_on_b_world = -force_on_a_world;
                    let (f, tau) = &mut wrenches[i as usize];
                    *f += force_on_b_world;
                    *tau += row.arm_b.cross(force_on_b_world);
                }
            }
            RowGeom::Angular => {
                let torque_on_a_world = row.dir_world * (f_impulse / dt);
                if let Some(i) = row.body_a {
                    let (_, tau) = &mut wrenches[i as usize];
                    *tau += torque_on_a_world;
                }
                if let Some(i) = row.body_b {
                    let (_, tau) = &mut wrenches[i as usize];
                    *tau += -torque_on_a_world;
                }
            }
        }
    }

    // Per-contact normal FORCE = normal-row impulse / dt. A contact can be
    // omitted from `per_contact` when its force-free gap is active, so use
    // the original contact index rather than the compact row-block index.
    for pc in &per_contact {
        let rows = contact_block_n_rows(pc.condim, cone);
        let normal_impulse = if cone == ConeKind::Pyramidal && pc.condim == 3 {
            (0..rows)
                .map(|offset| impulses[pc.start_row as usize + offset])
                .sum()
        } else {
            impulses[pc.start_row as usize]
        };
        contact_normal_forces[pc.contact_index] = normal_impulse / dt;
    }

    (wrenches, contact_normal_forces)
}

/// Per-contact solver bookkeeping shared across all rows of one contact.
struct PerContact {
    contact_index: usize,
    start_row: u32,
    condim: u8,
    solref: SolRef,
    solimp: SolImp,
    pen_active: f32,
    mu_slide: f32,
    mu_torsion: f32,
    mu_roll: f32,
    /// Precomputed pyramidal cross-response `A_{i,i+1}` between the first
    /// facet pair (`start_row`, `start_row+1`). Only meaningful for condim=3
    /// under a pyramidal cone; zero otherwise. Cached so
    /// `pgs_step_pyramidal_pair` does not re-invoke `row_cross_response` on
    /// every iteration (it is state-independent — depends only on rows +
    /// bodies + inv_i_world at solve start).
    h01_pair01: f32,
    /// Precomputed pyramidal cross-response between the second facet pair
    /// (`start_row+2`, `start_row+3`).
    h01_pair23: f32,
}

/// Assemble and solve the dense free-body Newton system. The response matrix
/// is built by applying one unit impulse per row, which shares the exact
/// `J M⁻¹ Jᵀ` path used by the PGS residual accumulator.
fn solve_free_body_newton_impulses(
    rows: &[ConstraintRow],
    per_contact: &[PerContact],
    n_bodies: usize,
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
    iterations: u32,
    cone: ConeKind,
) -> crate::newton::NewtonResult {
    let n_rows = rows.len();
    let mut hessian = vec![0.0f32; n_rows * n_rows];
    let mut response = vec![BodyDelta::default(); n_bodies];
    for j in 0..n_rows {
        for slot in response.iter_mut() {
            *slot = BodyDelta::default();
        }
        apply_impulse_delta(&rows[j], 1.0, &mut response, bodies, inv_i_world);
        for i in 0..n_rows {
            hessian[i * n_rows + j] = row_residual(&rows[i], &response, bodies, inv_i_world);
        }
    }
    for i in 0..n_rows {
        hessian[i * n_rows + i] += rows[i].reg;
    }

    let mut projections = Vec::new();
    for contact in per_contact {
        let normal = contact.start_row as usize;
        if cone == ConeKind::Pyramidal && contact.condim == 3 {
            for k in 0..4 {
                projections.push(crate::newton::Projection::NonNegative { index: normal + k });
            }
        } else {
            projections.push(crate::newton::Projection::NonNegative { index: normal });
        }
        if cone != ConeKind::Pyramidal && contact.condim >= 3 {
            let tangent_1 = normal + 1;
            let tangent_2 = normal + 2;
            projections.push(crate::newton::Projection::PyramidalCone {
                normal,
                tangent_1,
                tangent_2,
                mu: contact.mu_slide,
            });
            if contact.condim >= 4 {
                projections.push(crate::newton::Projection::ScalarConeBound {
                    index: normal + 3,
                    normal,
                    mu: contact.mu_torsion,
                });
            }
            if contact.condim >= 6 {
                projections.push(crate::newton::Projection::PyramidalCone {
                    normal,
                    tangent_1: normal + 4,
                    tangent_2: normal + 5,
                    mu: contact.mu_roll,
                });
            }
        }
    }
    // Rows belonging to equalities have no projection and are bilateral.
    let system = crate::newton::NewtonSystem {
        hessian,
        linear: rows.iter().map(|row| row.bias).collect(),
        projections,
        max_iterations: iterations.max(1),
        cost_tolerance: 1e-7,
    };
    system
        .solve()
        .unwrap_or_else(|error| panic!("Newton free-body solve failed: {error}"))
}

/// Row count for a contact block by condim.
///
/// - condim 1: 1 row (normal only).
/// - condim 3: 3 rows (normal + t1 + t2).
/// - condim 4: 4 rows (adds torsion about the normal).
/// - condim 6: 6 rows (adds two rolling rows about the tangents).
///
/// Panics on any other condim — the loader and the Geom programmatic
/// API both restrict `condim` to `{1, 3, 4, 6}`; a stray value here
/// is a caller bug.
fn contact_block_n_rows(condim: u8, cone: ConeKind) -> usize {
    if cone == ConeKind::Pyramidal && condim == 3 {
        return 4;
    }
    match condim {
        1 => 1,
        3 => 3,
        4 => 4,
        6 => 6,
        other => panic!("unsupported condim {other} — expected 1, 3, 4, or 6"),
    }
}

/// Per-equality solver bookkeeping. Every row of the equality shares one
/// `(solref, solimp)`; each row carries its own scalar violation (the
/// signed component of `p_A − p_B` or angular error along that axis).
struct PerEquality {
    start_row: u32,
    n_rows: u8,
    solref: SolRef,
    solimp: SolImp,
    /// Signed violation per row (indexed 0..n_rows). Only entries `< n_rows`
    /// are populated; the fixed-length array keeps the type `Copy`-free
    /// but keeps allocation overhead down (max 6 for weld).
    violations: [f32; 6],
}

/// Half-line PGS update `f ≥ 0` for a normal / limit row.
#[allow(clippy::too_many_arguments)]
fn pgs_step_non_negative(
    rows: &[ConstraintRow],
    ri: usize,
    impulses: &mut [f32],
    body_delta: &mut [BodyDelta],
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) {
    let residual = row_residual(&rows[ri], body_delta, bodies, inv_i_world)
        + rows[ri].reg * impulses[ri]
        + rows[ri].bias;
    let mut delta = -residual / rows[ri].diag;
    let new_f = impulses[ri] + delta;
    let projected = if new_f < 0.0 { 0.0 } else { new_f };
    delta = projected - impulses[ri];
    impulses[ri] = projected;
    apply_impulse_delta(&rows[ri], delta, body_delta, bodies, inv_i_world);
}

/// Update one pair of opposing pyramidal facets as one 2D block.
/// MuJoCo's PGS solver keeps the pair's non-negative cone constraint while
/// minimizing the paired quadratic, rather than clamping each facet alone.
///
/// `h01` is the precomputed `A_{i,j}` cross-response for the facet pair. It
/// is state-independent (see the precompute block in
/// `solve_free_bodies_diag_mode`), so passing it in avoids redoing the
/// scratch-allocating `row_cross_response` call on every iteration.
#[allow(clippy::too_many_arguments)]
fn pgs_step_pyramidal_pair(
    rows: &[ConstraintRow],
    ri: usize,
    h01: f32,
    impulses: &mut [f32],
    body_delta: &mut [BodyDelta],
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) {
    let rj = ri + 1;
    let residual_i = row_residual(&rows[ri], body_delta, bodies, inv_i_world)
        + rows[ri].reg * impulses[ri]
        + rows[ri].bias;
    let residual_j = row_residual(&rows[rj], body_delta, bodies, inv_i_world)
        + rows[rj].reg * impulses[rj]
        + rows[rj].bias;
    let h00 = rows[ri].diag;
    let h11 = rows[rj].diag;
    let det = h00 * h11 - h01 * h01;
    if det <= 0.0 {
        pgs_step_non_negative(rows, ri, impulses, body_delta, bodies, inv_i_world);
        pgs_step_non_negative(rows, rj, impulses, body_delta, bodies, inv_i_world);
        return;
    }
    let delta_i = (-residual_i * h11 + h01 * residual_j) / det;
    let delta_j = (h01 * residual_i - h00 * residual_j) / det;
    let next_i = impulses[ri] + delta_i;
    let next_j = impulses[rj] + delta_j;
    let (new_i, new_j) = if next_i >= 0.0 && next_j >= 0.0 {
        (next_i, next_j)
    } else if next_i < 0.0 && next_j < 0.0 {
        (0.0, 0.0)
    } else if next_i < 0.0 {
        (0.0, (impulses[rj] - residual_j / h11).max(0.0))
    } else {
        ((impulses[ri] - residual_i / h00).max(0.0), 0.0)
    };
    let applied_i = new_i - impulses[ri];
    let applied_j = new_j - impulses[rj];
    impulses[ri] = new_i;
    impulses[rj] = new_j;
    apply_impulse_delta(&rows[ri], applied_i, body_delta, bodies, inv_i_world);
    apply_impulse_delta(&rows[rj], applied_j, body_delta, bodies, inv_i_world);
}

/// Scratch-buffer-reusing cross-response: fills `scratch` with the body-delta
/// resulting from a unit impulse on `applied_row`, then returns
/// `row_residual(row, scratch, ...)`. The caller owns `scratch` and can reuse
/// it across many calls, keeping the total allocation count independent of
/// contact count.
fn row_cross_response_into(
    row: &ConstraintRow,
    applied_row: &ConstraintRow,
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
    scratch: &mut [BodyDelta],
) -> f32 {
    for slot in scratch.iter_mut() {
        *slot = BodyDelta::default();
    }
    apply_impulse_delta(applied_row, 1.0, scratch, bodies, inv_i_world);
    row_residual(row, scratch, bodies, inv_i_world)
}

/// Bilateral PGS update — no clamp. Used by equality rows.
fn pgs_step_bilateral(
    rows: &[ConstraintRow],
    ri: usize,
    impulses: &mut [f32],
    body_delta: &mut [BodyDelta],
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) {
    let residual = row_residual(&rows[ri], body_delta, bodies, inv_i_world)
        + rows[ri].reg * impulses[ri]
        + rows[ri].bias;
    let delta = -residual / rows[ri].diag;
    impulses[ri] += delta;
    apply_impulse_delta(&rows[ri], delta, body_delta, bodies, inv_i_world);
}

/// Scalar-cap PGS update: `|f| ≤ cap`. Used for torsional friction under
/// both pyramidal and elliptic cones (torsion is a scalar row; both cone
/// kinds bound it by the same one-sided cap).
#[allow(clippy::too_many_arguments)]
fn pgs_step_scalar_cap(
    rows: &[ConstraintRow],
    ri: usize,
    cap: f32,
    impulses: &mut [f32],
    body_delta: &mut [BodyDelta],
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) {
    let residual = row_residual(&rows[ri], body_delta, bodies, inv_i_world)
        + rows[ri].reg * impulses[ri]
        + rows[ri].bias;
    let mut delta = -residual / rows[ri].diag;
    let new_f = impulses[ri] + delta;
    let projected = project_pyramidal(new_f, cap);
    delta = projected - impulses[ri];
    impulses[ri] = projected;
    apply_impulse_delta(&rows[ri], delta, body_delta, bodies, inv_i_world);
}

/// Paired-row PGS update for a 2-DOF friction block (sliding tangents or
/// rolling axes). Pyramidal → two independent scalar clamps by `mu · cap`.
/// Elliptic → Jacobi step on both rows, then a single joint projection
/// onto the disc of radius `mu · cap`.
#[allow(clippy::too_many_arguments)]
fn pgs_step_pair_cone(
    rows: &[ConstraintRow],
    ri1: usize,
    ri2: usize,
    cap_normal: f32,
    mu: f32,
    cone: ConeKind,
    impulses: &mut [f32],
    body_delta: &mut [BodyDelta],
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) {
    match cone {
        ConeKind::Pyramidal => {
            let cap = mu * cap_normal;
            pgs_step_scalar_cap(rows, ri1, cap, impulses, body_delta, bodies, inv_i_world);
            pgs_step_scalar_cap(rows, ri2, cap, impulses, body_delta, bodies, inv_i_world);
        }
        ConeKind::Elliptic => {
            let residual1 = row_residual(&rows[ri1], body_delta, bodies, inv_i_world)
                + rows[ri1].reg * impulses[ri1]
                + rows[ri1].bias;
            let residual2 = row_residual(&rows[ri2], body_delta, bodies, inv_i_world)
                + rows[ri2].reg * impulses[ri2]
                + rows[ri2].bias;
            let new_f1 = impulses[ri1] - residual1 / rows[ri1].diag;
            let new_f2 = impulses[ri2] - residual2 / rows[ri2].diag;
            let (proj1, proj2) = project_elliptic(new_f1, new_f2, mu, cap_normal);
            let delta1 = proj1 - impulses[ri1];
            let delta2 = proj2 - impulses[ri2];
            impulses[ri1] = proj1;
            impulses[ri2] = proj2;
            apply_impulse_delta(&rows[ri1], delta1, body_delta, bodies, inv_i_world);
            apply_impulse_delta(&rows[ri2], delta2, body_delta, bodies, inv_i_world);
        }
    }
}

/// Build the row(s) for one free-body equality and record the block in
/// `per_equality`. `JointCoupling` is filtered out earlier — this only
/// handles Connect / Weld / Distance.
fn push_equality_rows(
    eq: &Equality,
    bodies: &[Body],
    rows: &mut Vec<ConstraintRow>,
    per_equality: &mut Vec<PerEquality>,
) {
    let start_row = rows.len() as u32;
    let mut violations = [0.0f32; 6];
    match eq {
        Equality::Connect {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            solref,
            solimp,
        } => {
            let (p_a, arm_a) = anchor_world_and_arm(*body_a, *anchor_a, bodies);
            let (p_b, arm_b) = anchor_world_and_arm(*body_b, *anchor_b, bodies);
            let ba = body_a.map(|i| i as u32);
            let bb = body_b.map(|i| i as u32);
            let delta = p_a - p_b;
            for (k, ax) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
                let raw_r = delta.dot(*ax);
                let (dir, r_pos) = escape_convention(*ax, raw_r);
                rows.push(ConstraintRow {
                    dir_world: dir,
                    body_a: ba,
                    body_b: bb,
                    arm_a,
                    arm_b,
                    reg: 0.0,
                    diag: 0.0,
                    bias: 0.0,
                    geom: RowGeom::Linear,
                });
                violations[k] = r_pos;
            }
            per_equality.push(PerEquality {
                start_row,
                n_rows: 3,
                solref: *solref,
                solimp: *solimp,
                violations,
            });
        }
        Equality::Weld {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            relative_orientation,
            solref,
            solimp,
        } => {
            let (p_a, arm_a) = anchor_world_and_arm(*body_a, *anchor_a, bodies);
            let (p_b, arm_b) = anchor_world_and_arm(*body_b, *anchor_b, bodies);
            let ba = body_a.map(|i| i as u32);
            let bb = body_b.map(|i| i as u32);
            let delta = p_a - p_b;
            // Linear rows: anchor coincidence (same as connect).
            for (k, ax) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
                let raw_r = delta.dot(*ax);
                let (dir, r_pos) = escape_convention(*ax, raw_r);
                rows.push(ConstraintRow {
                    dir_world: dir,
                    body_a: ba,
                    body_b: bb,
                    arm_a,
                    arm_b,
                    reg: 0.0,
                    diag: 0.0,
                    bias: 0.0,
                    geom: RowGeom::Linear,
                });
                violations[k] = r_pos;
            }
            // Angular rows: orientation error = 2 · imag(q_err), where
            //   q_err = q_A · q_target · q_B_conj (world frame).
            // Sign-canonicalize the quaternion (flip when w < 0) so we
            // always take the shortest rotation, then flip per-axis so
            // each row is in escape convention (r >= 0, +dir reduces r).
            let q_a = match body_a {
                Some(i) => bodies[*i].orientation,
                None => Quat::IDENTITY,
            };
            let q_b = match body_b {
                Some(i) => bodies[*i].orientation,
                None => Quat::IDENTITY,
            };
            let q_err = q_a * (*relative_orientation) * q_b.conjugate();
            let sign = if q_err.w >= 0.0 { 1.0 } else { -1.0 };
            let theta = Vec3::new(q_err.x, q_err.y, q_err.z) * (2.0 * sign);
            for (k, ax) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
                let raw_r = theta.dot(*ax);
                let (dir, r_pos) = escape_convention(*ax, raw_r);
                rows.push(ConstraintRow {
                    dir_world: dir,
                    body_a: ba,
                    body_b: bb,
                    arm_a: Vec3::ZERO,
                    arm_b: Vec3::ZERO,
                    reg: 0.0,
                    diag: 0.0,
                    bias: 0.0,
                    geom: RowGeom::Angular,
                });
                violations[3 + k] = r_pos;
            }
            per_equality.push(PerEquality {
                start_row,
                n_rows: 6,
                solref: *solref,
                solimp: *solimp,
                violations,
            });
        }
        Equality::Distance {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            distance,
            solref,
            solimp,
        } => {
            let (p_a, arm_a) = anchor_world_and_arm(*body_a, *anchor_a, bodies);
            let (p_b, arm_b) = anchor_world_and_arm(*body_b, *anchor_b, bodies);
            let ba = body_a.map(|i| i as u32);
            let bb = body_b.map(|i| i as u32);
            let delta = p_a - p_b;
            let d_len = delta.length();
            if d_len < DISTANCE_DEGENERATE_EPS {
                // Elide the row this step — no meaningful direction. The
                // constraint re-engages once the separation grows past the
                // guard.
                return;
            }
            let u = delta / d_len;
            let raw_r = d_len - *distance;
            // Escape convention: +f impulse reduces |raw_r|. When
            // d_len > distance (raw_r > 0), the pair is too far apart,
            // so +dir must pull A toward B — dir = -u (opposite of the
            // separation direction). When raw_r < 0 (too close), +dir
            // must push A away — dir = +u.
            let (dir, r_pos) = escape_convention(u, raw_r);
            rows.push(ConstraintRow {
                dir_world: dir,
                body_a: ba,
                body_b: bb,
                arm_a,
                arm_b,
                reg: 0.0,
                diag: 0.0,
                bias: 0.0,
                geom: RowGeom::Linear,
            });
            violations[0] = r_pos;
            per_equality.push(PerEquality {
                start_row,
                n_rows: 1,
                solref: *solref,
                solimp: *solimp,
                violations,
            });
        }
        Equality::JointCoupling { .. } => {
            // Handled by the tree solver.
        }
    }
}

/// Escape-convention canonicalization for a bilateral-equality row.
/// Given a natural direction `raw_dir` (unit) and a signed violation
/// `raw_r`, returns `(dir, |raw_r|)` where `dir` is `raw_dir` when
/// `raw_r >= 0` and `-raw_dir` when `raw_r < 0`. Wait — the contact
/// convention wants `J·qdot = -r_dot` so that a positive impulse
/// (through dir) drives r DOWN. J·qdot naturally computes
/// `d(raw_r)/dt` in the +raw_dir direction; to have `+dir` reduce r,
/// we need `dir = -raw_dir` when `raw_r > 0` (drive raw_r down) and
/// `dir = +raw_dir` when `raw_r < 0` (drive raw_r up). Then r_row =
/// |raw_r| and `J·qdot_row = -dr_row/dt`, matching the contact-normal
/// escape convention (bias formula reused verbatim).
fn escape_convention(raw_dir: Vec3, raw_r: f32) -> (Vec3, f32) {
    if raw_r >= 0.0 {
        (-raw_dir, raw_r)
    } else {
        (raw_dir, -raw_r)
    }
}

/// Resolve `(anchor world position, arm from body COM to anchor)` for one
/// side of an equality. A `None` body means the anchor is a world-frame
/// point (arm irrelevant — the linear-row math ignores the static-side
/// arm because `body = None` skips the accumulator write).
fn anchor_world_and_arm(body: Option<usize>, anchor: Vec3, bodies: &[Body]) -> (Vec3, Vec3) {
    match body {
        Some(i) => {
            let b = &bodies[i];
            let arm = b.orientation.rotate(anchor);
            (b.position + arm, arm)
        }
        None => (anchor, Vec3::ZERO),
    }
}

/// Compute `A_ii = J_i M^-1 J_i^T` — the rigid-body diagonal contribution
/// for one constraint row, summed over the row's two bodies. Static geoms
/// contribute zero (infinite mass). Branches on [`RowGeom`]: linear rows
/// pick up both `(r × dir) I⁻¹ (r × dir) + 1/m` per body; angular rows
/// drop the linear-mass term AND the arm cross (pure `dir I⁻¹ dir`).
fn row_body_diagonal(
    row: &ConstraintRow,
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) -> f32 {
    match row.geom {
        RowGeom::Linear => {
            let mut a_ii = 0.0f32;
            if let Some(i) = row.body_a {
                let i = i as usize;
                let r_cross_dir = row.arm_a.cross(row.dir_world);
                a_ii += r_cross_dir.dot(inv_i_world[i] * r_cross_dir) + 1.0 / bodies[i].mass;
            }
            if let Some(i) = row.body_b {
                let i = i as usize;
                let r_cross_dir = row.arm_b.cross(row.dir_world);
                a_ii += r_cross_dir.dot(inv_i_world[i] * r_cross_dir) + 1.0 / bodies[i].mass;
            }
            a_ii
        }
        RowGeom::Angular => {
            let mut a_ii = 0.0f32;
            if let Some(i) = row.body_a {
                let i = i as usize;
                a_ii += row.dir_world.dot(inv_i_world[i] * row.dir_world);
            }
            if let Some(i) = row.body_b {
                let i = i as usize;
                a_ii += row.dir_world.dot(inv_i_world[i] * row.dir_world);
            }
            a_ii
        }
    }
}

/// MuJoCo's contact regularizer starts from its diagonal approximation, not
/// the exact row response. Pyramidal condim-3 rows then share the translated
/// cone regularizer `2*mu^2*R_normal`.
fn contact_regularization(
    pc: &PerContact,
    row: &ConstraintRow,
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
    impedance_value: f32,
    cone: ConeKind,
) -> f32 {
    if impedance_value <= 0.0 {
        return 0.0;
    }
    let mut tran = 0.0;
    for body in [row.body_a, row.body_b].into_iter().flatten() {
        tran += 1.0 / bodies[body as usize].mass;
    }
    let base_diag = if cone == ConeKind::Pyramidal && pc.condim == 3 {
        // mj_diagApprox assigns tran + mu^2*tran to each sliding facet.
        tran * (1.0 + pc.mu_slide * pc.mu_slide)
    } else {
        row_body_diagonal(row, bodies, inv_i_world)
    };
    contact_regularization_from_diag(
        base_diag,
        impedance_value,
        pc.mu_slide,
        cone == ConeKind::Pyramidal && pc.condim == 3,
    )
}

/// Apply MuJoCo's `mj_makeImpedance` regularizer to one diagonal
/// approximation. Pyramidal facets use the common `Rpy` value derived from
/// the normal facet, so every solver path shares the same cone rule.
fn contact_regularization_from_diag(
    base_diag: f32,
    impedance_value: f32,
    mu_slide: f32,
    pyramidal_condim3: bool,
) -> f32 {
    let base_reg = (1.0 - impedance_value) / impedance_value * base_diag;
    if pyramidal_condim3 {
        2.0 * mu_slide * mu_slide * base_reg
    } else {
        base_reg
    }
}

/// Compute `J_i · body_delta` for one row — the change in constraint
/// velocity from accumulated impulses so far. For linear rows this is the
/// contact-point relative velocity along `dir_world`; for angular rows it's
/// the relative angular velocity along `dir_world`.
fn row_residual(
    row: &ConstraintRow,
    body_delta: &[BodyDelta],
    bodies: &[Body],
    _inv_i_world: &[crate::math::Mat3],
) -> f32 {
    let mut s = 0.0f32;
    match row.geom {
        RowGeom::Linear => {
            if let Some(i) = row.body_a {
                let i = i as usize;
                let dv_ang_world = bodies[i].orientation.rotate(body_delta[i].dw_ang_body);
                let v_point = body_delta[i].dv_lin_world + dv_ang_world.cross(row.arm_a);
                s += v_point.dot(row.dir_world);
            }
            if let Some(i) = row.body_b {
                let i = i as usize;
                let dv_ang_world = bodies[i].orientation.rotate(body_delta[i].dw_ang_body);
                let v_point = body_delta[i].dv_lin_world + dv_ang_world.cross(row.arm_b);
                s -= v_point.dot(row.dir_world);
            }
        }
        RowGeom::Angular => {
            if let Some(i) = row.body_a {
                let i = i as usize;
                let dw_world = bodies[i].orientation.rotate(body_delta[i].dw_ang_body);
                s += dw_world.dot(row.dir_world);
            }
            if let Some(i) = row.body_b {
                let i = i as usize;
                let dw_world = bodies[i].orientation.rotate(body_delta[i].dw_ang_body);
                s -= dw_world.dot(row.dir_world);
            }
        }
    }
    s
}

/// Apply an impulse delta on row `row` to the two bodies' accumulators.
/// Linear rows: force at anchor point → `(force / m)` linear + `(arm ×
/// force)` torque per body. Angular rows: pure torque `dir * f` on A,
/// `-dir * f` on B — no linear force at all.
fn apply_impulse_delta(
    row: &ConstraintRow,
    delta_f: f32,
    body_delta: &mut [BodyDelta],
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) {
    if delta_f == 0.0 {
        return;
    }
    match row.geom {
        RowGeom::Linear => {
            if let Some(i) = row.body_a {
                let i = i as usize;
                let force_world = row.dir_world * delta_f;
                let torque_world = row.arm_a.cross(force_world);
                body_delta[i].dv_lin_world += force_world / bodies[i].mass;
                let dw_world = inv_i_world[i] * torque_world;
                body_delta[i].dw_ang_body += bodies[i].orientation.inverse_rotate(dw_world);
            }
            if let Some(i) = row.body_b {
                let i = i as usize;
                let force_world = -row.dir_world * delta_f;
                let torque_world = row.arm_b.cross(force_world);
                body_delta[i].dv_lin_world += force_world / bodies[i].mass;
                let dw_world = inv_i_world[i] * torque_world;
                body_delta[i].dw_ang_body += bodies[i].orientation.inverse_rotate(dw_world);
            }
        }
        RowGeom::Angular => {
            if let Some(i) = row.body_a {
                let i = i as usize;
                let torque_world = row.dir_world * delta_f;
                let dw_world = inv_i_world[i] * torque_world;
                body_delta[i].dw_ang_body += bodies[i].orientation.inverse_rotate(dw_world);
            }
            if let Some(i) = row.body_b {
                let i = i as usize;
                let torque_world = -row.dir_world * delta_f;
                let dw_world = inv_i_world[i] * torque_world;
                body_delta[i].dw_ang_body += bodies[i].orientation.inverse_rotate(dw_world);
            }
        }
    }
}

/// Current constraint velocity `J · qdot` at start of step for a row's
/// geometry. Linear rows use point-velocity relative-to-B along `dir`;
/// angular rows use `(ω_A − ω_B) · dir` in world coords.
fn row_current_velocity(row: &ConstraintRow, bodies: &[Body]) -> f32 {
    match row.geom {
        RowGeom::Linear => point_direction_velocity(
            bodies,
            row.body_a,
            row.body_b,
            row.arm_a,
            row.arm_b,
            row.dir_world,
        ),
        RowGeom::Angular => {
            let w_a = match row.body_a {
                Some(i) => bodies[i as usize].angular_velocity_world(),
                None => Vec3::ZERO,
            };
            let w_b = match row.body_b {
                Some(i) => bodies[i as usize].angular_velocity_world(),
                None => Vec3::ZERO,
            };
            (w_a - w_b).dot(row.dir_world)
        }
    }
}

/// Free-step (gravity Euler kick) contribution to a row's velocity bias.
/// Linear rows pick up the free linear velocity delta at each anchor
/// point; angular rows pick up nothing under our current
/// gravity-only-free-step approximation (`dw_body_free ≈ 0`).
fn row_free_step_velocity(
    row: &ConstraintRow,
    dv_lin_free_per_body: &[Vec3],
    dw_body_free_per_body: &[Vec3],
    bodies: &[Body],
) -> f32 {
    match row.geom {
        RowGeom::Linear => free_step_velocity_contribution(
            dv_lin_free_per_body,
            dw_body_free_per_body,
            row.body_a,
            row.body_b,
            row.arm_a,
            row.arm_b,
            row.dir_world,
            bodies,
        ),
        RowGeom::Angular => 0.0,
    }
}

/// Current normal velocity `v_rel · normal` at contact-point (start-of-step
/// state, no free-step delta added).
fn normal_velocity_at_point(
    bodies: &[Body],
    body_a: Option<u32>,
    body_b: Option<u32>,
    arm_a: Vec3,
    arm_b: Vec3,
    normal: Vec3,
) -> f32 {
    let v_a = match body_a {
        Some(i) => {
            let b = &bodies[i as usize];
            b.linear_velocity + b.angular_velocity_world().cross(arm_a)
        }
        None => Vec3::ZERO,
    };
    let v_b = match body_b {
        Some(i) => {
            let b = &bodies[i as usize];
            b.linear_velocity + b.angular_velocity_world().cross(arm_b)
        }
        None => Vec3::ZERO,
    };
    (v_a - v_b).dot(normal)
}

/// Same as [`normal_velocity_at_point`] but for any direction (used for
/// tangent bias assembly).
fn point_direction_velocity(
    bodies: &[Body],
    body_a: Option<u32>,
    body_b: Option<u32>,
    arm_a: Vec3,
    arm_b: Vec3,
    dir: Vec3,
) -> f32 {
    normal_velocity_at_point(bodies, body_a, body_b, arm_a, arm_b, dir)
}

// ---------------------------------------------------------------------------
// Tree joint-limit solver
// ---------------------------------------------------------------------------

use crate::dynamics::{cholesky, cholesky_solve, mass_matrix};
use crate::joint::JointKind;
use crate::tree::{Tree, forward_kinematics};

/// One assembled tree-space row. Joint limits, joint-coupling equalities,
/// AND tendon length limits all reduce to a sparse row on the tree's `nv`
/// velocity vector:
///
/// - Joint limit: 1 non-zero (`±1` at the limited DOF's slot).
/// - Joint coupling: 2 non-zeros (`+1` at `v_slot_a`, `-k` at `v_slot_b`,
///   with `k = c1 + 2·c2·q_b` — the derivative of the polynomial
///   constraint).
/// - Tendon length limit: potentially many non-zeros — the tendon's
///   Jacobian row `dL/dqdot` (or its negation on the high side).
///
/// Storing them uniformly lets one PGS sweep handle all three.
#[derive(Clone, Debug)]
struct TreeRow {
    /// Sparse `(v_slot, coefficient)` entries. Joint limits have length
    /// 1, couplings length 2, tendon limits variable.
    sparse_coeffs: Vec<(u32, f32)>,
    /// Signed violation `r` (rad, m, or polynomial residual). For limits
    /// this is always ≥ 0 (the row is only added when active); for
    /// couplings the value is signed.
    violation: f32,
    solref: SolRef,
    solimp: SolImp,
    projection: TreeRowProjection,
}

/// How the PGS update clamps this row.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TreeRowProjection {
    /// `f ≥ 0` — used for joint-range limits.
    NonNegative,
    /// No clamp — used for equality (joint-coupling) rows.
    Bilateral,
}

impl TreeRow {
    /// `J · qdot` for this row (scalar constraint velocity).
    fn dot_qdot(&self, qdot: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for &(slot, coeff) in &self.sparse_coeffs {
            s += coeff * qdot[slot as usize];
        }
        s
    }

    /// `e_i · d_j` — the dense `A_ij` entry, computed sparsely from this
    /// row's coefficients and the response vector `d_j` for row `j`.
    fn dot_response(&self, d_vec: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for &(slot, coeff) in &self.sparse_coeffs {
            s += coeff * d_vec[slot as usize];
        }
        s
    }
}

/// One joint-coupling row's Jacobian, as the solver assembles it —
/// exposed for tests so hand-derivations can pin the chain-rule
/// coefficient `k = c1 + 2·c2·q_b` directly (dynamic tracking tests
/// alone can mask a wrong Jacobian at steady state, per NEWT-10 R2
/// reviewer note). The two coefficients correspond to `link_a` and
/// `link_b` respectively; the escape-convention sign flip is applied
/// exactly as in the live solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingRowProbe {
    /// `v_offset` slot for the coupling's `link_a`.
    pub slot_a: u32,
    /// `v_offset` slot for the coupling's `link_b`.
    pub slot_b: u32,
    /// Row coefficient on `slot_a` (row's `J` entry there). Post
    /// escape-convention flip.
    pub coeff_a: f32,
    /// Row coefficient on `slot_b`. Equals `-(c1 + 2·c2·q_b)` under
    /// the correct chain rule (post escape flip).
    pub coeff_b: f32,
    /// Signed polynomial residual `q_a − (c0 + c1·q_b + c2·q_b²)` at
    /// the tree's current state.
    pub violation_signed: f32,
}

/// Return one [`CouplingRowProbe`] per `JointCoupling` equality on
/// `tree` (in declaration order). The rows are built by
/// [`build_tree_solver_rows`] — the SAME code path the live solver
/// uses — so a mutation to the row-building propagates here too and
/// the test comparison against a hand-derived `c1 + 2·c2·q_b`
/// discriminates.
pub fn tree_coupling_jacobian_probe(
    tree: &Tree,
    tree_idx: usize,
    equalities: &[Equality],
) -> Vec<CouplingRowProbe> {
    build_tree_solver_rows(tree, tree_idx, equalities)
        .into_iter()
        .filter(|row| row.projection == TreeRowProjection::Bilateral)
        .map(|row| {
            let (s0, c0) = row.sparse_coeffs[0];
            let (s1, c1) = row.sparse_coeffs[1];
            CouplingRowProbe {
                slot_a: s0,
                slot_b: s1,
                coeff_a: c0,
                coeff_b: c1,
                // Violation stored in the row is |r_signed|; reconstruct
                // the sign from the row's flip (coeff_a is +1 pre-flip;
                // negative here means r_signed >= 0).
                violation_signed: if c0 < 0.0 {
                    row.violation
                } else {
                    -row.violation
                },
            }
        })
        .collect()
}

/// A tendon-row probe — one entry per active length-limit row. Mirrors
/// [`CouplingRowProbe`]: exposes the sparse Jacobian assembled by
/// [`build_tree_solver_rows`] so hand-derivation tests can pin the row
/// against a `dL/dqdot` reference and detect a chain-rule mutant.
#[derive(Clone, Debug, PartialEq)]
pub struct TendonRowProbe {
    /// Sparse `(v_slot, coefficient)` entries after the escape-convention
    /// flip (`+f impulse` reduces `violation`). Copy of the live row.
    pub sparse_coeffs: Vec<(u32, f32)>,
    /// Magnitude of the length violation (always ≥ 0).
    pub violation: f32,
    /// SolRef the row uses.
    pub solref: SolRef,
}

/// One [`TendonRowProbe`] per active tendon length-limit row on `tree`.
/// Empty when no tendon range is violated.
pub fn tree_tendon_row_probe(tree: &Tree) -> Vec<TendonRowProbe> {
    build_tree_solver_rows(tree, 0, &[])
        .into_iter()
        .filter_map(|row| {
            // Tendon rows are non-negative (like limits) and can have >2
            // sparse entries — pure joint limits have exactly 1.
            if row.projection != TreeRowProjection::NonNegative || row.sparse_coeffs.len() < 2 {
                None
            } else {
                Some(TendonRowProbe {
                    sparse_coeffs: row.sparse_coeffs.clone(),
                    violation: row.violation,
                    solref: row.solref,
                })
            }
        })
        .collect()
}

/// Assemble the tree-space PGS rows for one tree: active hinge / slide
/// range limits, followed by joint-coupling equalities in declaration
/// order. The row structure (sparse coefficients, signed-residual
/// convention, escape-convention sign flip on coupling rows) lives
/// here so [`solve_tree_limits`] and [`tree_coupling_jacobian_probe`]
/// share it — any mutation to the coefficient formula (e.g., dropping
/// the chain-rule `2·c2·q_b` term) propagates to both the live
/// solver and the probe, and the probe-based test discriminates it
/// against the hand-derived expression.
fn build_tree_solver_rows(tree: &Tree, tree_idx: usize, equalities: &[Equality]) -> Vec<TreeRow> {
    let mut rows: Vec<TreeRow> = Vec::new();
    // Enumerate active hinge/slide limits — same ordering as before
    // (ascending link index, low-side before high-side never triggers on
    // the same link so no tie to break).
    for (li, link) in tree.links.iter().enumerate() {
        let (range, limit_cfg, is_single_dof) = match link.joint {
            JointKind::Hinge { range, limit, .. } | JointKind::Slide { range, limit, .. } => {
                (range, Some(limit), true)
            }
            _ => (None, None, false),
        };
        if !is_single_dof {
            continue;
        }
        let Some((lo, hi)) = range else { continue };
        let limit_cfg = limit_cfg.expect("single-DOF joint carries a JointLimit");
        let per_joint_solref = limit_cfg.solref.unwrap_or(SolRef::DEFAULT);
        let per_joint_solimp = limit_cfg.solimp.unwrap_or(SolImp::DEFAULT);
        let v_slot = tree.v_offset[li] as u32;
        let q = tree.q[tree.q_offset[li]];
        if q < lo {
            rows.push(TreeRow {
                sparse_coeffs: vec![(v_slot, 1.0)],
                violation: lo - q,
                solref: per_joint_solref,
                solimp: per_joint_solimp,
                projection: TreeRowProjection::NonNegative,
            });
        } else if q > hi {
            rows.push(TreeRow {
                sparse_coeffs: vec![(v_slot, -1.0)],
                violation: q - hi,
                solref: per_joint_solref,
                solimp: per_joint_solimp,
                projection: TreeRowProjection::NonNegative,
            });
        }
    }
    // Joint-coupling equalities for this tree (declaration order).
    for eq in equalities {
        let Equality::JointCoupling {
            tree: eq_tree,
            link_a,
            link_b,
            polycoef,
            solref,
            solimp,
        } = eq
        else {
            continue;
        };
        if *eq_tree != tree_idx {
            continue;
        }
        // Both links must be single-DOF (hinge/slide). Validated by the
        // world/loader; assert here so a stray coupling doesn't silently
        // corrupt the row.
        assert!(
            matches!(
                tree.links[*link_a].joint,
                JointKind::Hinge { .. } | JointKind::Slide { .. }
            ),
            "joint-coupling link_a must be hinge/slide"
        );
        assert!(
            matches!(
                tree.links[*link_b].joint,
                JointKind::Hinge { .. } | JointKind::Slide { .. }
            ),
            "joint-coupling link_b must be hinge/slide"
        );
        let slot_a = tree.v_offset[*link_a] as u32;
        let slot_b = tree.v_offset[*link_b] as u32;
        let q_a = tree.q[tree.q_offset[*link_a]];
        let q_b = tree.q[tree.q_offset[*link_b]];
        let [c0, c1, c2] = *polycoef;
        // Residual r = q_a − (c0 + c1·q_b + c2·q_b²). Derivative
        // ∂r/∂qdot = [+1 at slot_a, -(c1 + 2·c2·q_b) at slot_b].
        // Normalize to escape convention (violation ≥ 0, +f impulse
        // reduces r) by flipping the row coefficients when the signed
        // residual is positive — same trick as the free-body bilateral
        // rows.
        let k = c1 + 2.0 * c2 * q_b;
        let r_signed = q_a - (c0 + c1 * q_b + c2 * q_b * q_b);
        let flip = if r_signed >= 0.0 { -1.0 } else { 1.0 };
        rows.push(TreeRow {
            sparse_coeffs: vec![(slot_a, flip), (slot_b, -k * flip)],
            violation: r_signed.abs(),
            solref: *solref,
            solimp: *solimp,
            projection: TreeRowProjection::Bilateral,
        });
    }
    // Tendon length limits (v2 tier 3). Iterate tendons in declaration
    // order — one row per active violation (low OR high side). The row's
    // Jacobian is the tendon's `dL/dqdot` (or its negation on the high
    // side so `+f impulse` reduces the violation in escape convention).
    // Per-tendon solref/solimp are threaded (NEWT-9 lesson: a defaulted
    // solref masks per-tendon tuning).
    if !tree.tendons.is_empty() {
        let poses = crate::tree::forward_kinematics(tree);
        for tendon in &tree.tendons {
            let Some((lo, hi)) = tendon.range else {
                continue;
            };
            let per_solref = tendon.limit_solref.unwrap_or(SolRef::DEFAULT);
            let per_solimp = tendon.limit_solimp.unwrap_or(SolImp::DEFAULT);
            let kin = crate::tendon::tendon_kinematics(tendon, tree, &poses);
            if kin.length < lo {
                // Compress Jacobian into sparse (skip zeros so the PGS
                // Cholesky solve doesn't pay for a dense vector).
                let sparse: Vec<(u32, f32)> = kin
                    .jacobian
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c != 0.0)
                    .map(|(i, &c)| (i as u32, c))
                    .collect();
                if !sparse.is_empty() {
                    rows.push(TreeRow {
                        sparse_coeffs: sparse,
                        violation: lo - kin.length,
                        solref: per_solref,
                        solimp: per_solimp,
                        projection: TreeRowProjection::NonNegative,
                    });
                }
            } else if kin.length > hi {
                // High side: negate the Jacobian entries so `+f` reduces
                // `L` (equivalent to the `-1` sign on a high-side joint
                // limit row).
                let sparse: Vec<(u32, f32)> = kin
                    .jacobian
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c != 0.0)
                    .map(|(i, &c)| (i as u32, -c))
                    .collect();
                if !sparse.is_empty() {
                    rows.push(TreeRow {
                        sparse_coeffs: sparse,
                        violation: kin.length - hi,
                        solref: per_solref,
                        solimp: per_solimp,
                        projection: TreeRowProjection::NonNegative,
                    });
                }
            }
        }
    }
    rows
}

/// Solve per-tree constraints (joint-range limits + joint-coupling
/// equalities) in PGS. Returns a per-DOF generalized force delta
/// (`nv`-length) that the caller adds to `tree.qfrc_applied` for the step
/// (ZOH under RK4).
///
/// We build the tree's mass matrix `M`, factor it once via Cholesky, then
/// solve `M^-1 · e_j` per row to get each row's velocity-response
/// direction. `A` is dense `n_rows × n_rows` with
/// `A_ij = e_i^T · M^-1 · e_j`. Limit rows project to `f ≥ 0`; coupling
/// rows use the bilateral (no-clamp) update.
///
/// `equalities` is the world's equality list; only entries whose
/// `tree_index() == Some(tree_idx)` are consumed here. Passing the whole
/// list keeps the API symmetric with the free-body solver — the tree
/// index tells the function which tree to filter for.
///
/// Zero-length return `vec![0.0; nv]` when neither limits nor couplings
/// are active.
pub fn solve_tree_limits(
    tree: &Tree,
    tree_idx: usize,
    equalities: &[Equality],
    dt: f32,
    iterations: u32,
) -> Vec<f32> {
    let nv = tree.nv();
    let mut qfrc = vec![0.0f32; nv];
    if nv == 0 || dt <= 0.0 {
        return qfrc;
    }

    let rows: Vec<TreeRow> = build_tree_solver_rows(tree, tree_idx, equalities);
    if rows.is_empty() {
        return qfrc;
    }

    // Build M(q), Cholesky. If M is not PD (shouldn't happen for a valid
    // tree), fall back to no impulse — the caller keeps the penalty
    // pathway alive (documented).
    let m = mass_matrix(tree);
    let Some(l) = cholesky(&m, nv) else {
        return qfrc;
    };

    // Precompute d_i = M^-1 · e_i (dense e_i built from the row's sparse
    // coefficients). Coupling rows put two non-zero entries into e_i.
    let n_rows = rows.len();
    let mut d_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
    for row in &rows {
        let mut e = vec![0.0f32; nv];
        for &(slot, coeff) in &row.sparse_coeffs {
            if coeff != 0.0 {
                e[slot as usize] += coeff;
            }
        }
        d_vecs.push(cholesky_solve(&l, nv, &e));
    }

    // Dense A_ij = e_i · d_j — computed sparsely from row_i's two
    // coefficients.
    let mut a = vec![0.0f32; n_rows * n_rows];
    for i in 0..n_rows {
        for j in 0..n_rows {
            a[i * n_rows + j] = rows[i].dot_response(&d_vecs[j]);
        }
    }

    // Bias per row. For a limit the "constraint velocity" is
    // `sign · qdot[v_slot]` (i.e. J·qdot with the appropriate sign
    // baked in). For a coupling it's the polynomial-residual rate
    // `qdot_a − k·qdot_b`. Both are just `row.dot_qdot(&tree.qdot)`.
    let mut bias = vec![0.0f32; n_rows];
    for (i, row) in rows.iter().enumerate() {
        let v_row = row.dot_qdot(&tree.qdot);
        // Both limit and coupling rows are stored in escape convention:
        // `+f impulse` reduces `row.violation`, so r_dot = -v_row.
        // Limit rows: sign chosen so v_row = sign · qdot[v_slot] = escape
        // velocity (positive when escaping). Coupling rows: flipped at
        // construction so `+f` reduces |signed residual|.
        let a_ref = reference_accel(
            -row.violation,
            v_row,
            safe_solref(row.solref, dt),
            row.solimp,
        );
        bias[i] = -a_ref * dt;
    }

    // Regularization R_i = ((1 - d)/d) · A_ii per row.
    let mut diag = vec![0.0f32; n_rows];
    for i in 0..n_rows {
        let a_ii = a[i * n_rows + i];
        let d = impedance(rows[i].violation, rows[i].solimp);
        let r = if d > 0.0 { (1.0 - d) / d * a_ii } else { 0.0 };
        diag[i] = a_ii + r;
    }

    // PGS: fixed iteration count (from `world.solver.iterations`), sweep
    // in row order (limits first per tree link, then couplings in
    // declaration order — determinism-preserving).
    let mut f = vec![0.0f32; n_rows];
    for _iter in 0..iterations {
        for i in 0..n_rows {
            let mut r_i = bias[i];
            for j in 0..n_rows {
                r_i += a[i * n_rows + j] * f[j];
            }
            let extra_reg = (diag[i] - a[i * n_rows + i]) * f[i];
            r_i += extra_reg;
            let delta = -r_i / diag[i];
            let new_f = f[i] + delta;
            let proj = match rows[i].projection {
                TreeRowProjection::NonNegative => {
                    if new_f < 0.0 {
                        0.0
                    } else {
                        new_f
                    }
                }
                TreeRowProjection::Bilateral => new_f,
            };
            f[i] = proj;
        }
    }

    // Accumulate impulses into qfrc slots as generalized force
    // (impulse/dt). Row's J^T distributes over its sparse coefficients.
    for (i, row) in rows.iter().enumerate() {
        let f_dt = f[i] / dt;
        for &(slot, coeff) in &row.sparse_coeffs {
            if coeff != 0.0 {
                qfrc[slot as usize] += coeff * f_dt;
            }
        }
    }

    qfrc
}

/// Newton counterpart to [`solve_tree_limits`]. It uses the same tree rows,
/// mass response, bias, and regularization, then solves the dense convex
/// quadratic with non-negative projections for limits and bilateral rows for
/// couplings.
pub fn solve_tree_limits_newton(
    tree: &Tree,
    tree_idx: usize,
    equalities: &[Equality],
    dt: f32,
    iterations: u32,
) -> Vec<f32> {
    let nv = tree.nv();
    let mut qfrc = vec![0.0f32; nv];
    if nv == 0 || dt <= 0.0 {
        return qfrc;
    }
    let rows = build_tree_solver_rows(tree, tree_idx, equalities);
    if rows.is_empty() {
        return qfrc;
    }
    let m = mass_matrix(tree);
    let Some(l) = cholesky(&m, nv) else {
        panic!("Newton tree solve failed: tree mass matrix is not positive definite");
    };
    let n_rows = rows.len();
    let mut d_vecs = Vec::with_capacity(n_rows);
    for row in &rows {
        let mut e = vec![0.0f32; nv];
        for &(slot, coeff) in &row.sparse_coeffs {
            e[slot as usize] += coeff;
        }
        d_vecs.push(cholesky_solve(&l, nv, &e));
    }
    let mut hessian = vec![0.0f32; n_rows * n_rows];
    for i in 0..n_rows {
        for j in 0..n_rows {
            hessian[i * n_rows + j] = rows[i].dot_response(&d_vecs[j]);
        }
    }
    let mut linear = vec![0.0f32; n_rows];
    for (i, row) in rows.iter().enumerate() {
        let v_row = row.dot_qdot(&tree.qdot);
        let a_ref = reference_accel(
            -row.violation,
            v_row,
            safe_solref(row.solref, dt),
            row.solimp,
        );
        let d = impedance(row.violation, row.solimp);
        linear[i] = -a_ref * dt;
        let reg = if d > 0.0 {
            (1.0 - d) / d * hessian[i * n_rows + i]
        } else {
            0.0
        };
        hessian[i * n_rows + i] += reg;
    }
    let projections = rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| match row.projection {
            TreeRowProjection::NonNegative => {
                Some(crate::newton::Projection::NonNegative { index })
            }
            TreeRowProjection::Bilateral => None,
        })
        .collect();
    let result = crate::newton::NewtonSystem {
        hessian,
        linear,
        projections,
        max_iterations: iterations.max(1),
        cost_tolerance: 1e-7,
    }
    .solve()
    .unwrap_or_else(|error| panic!("Newton tree solve failed: {error}"));
    for (i, row) in rows.iter().enumerate() {
        let f_dt = result.solution[i] / dt;
        for &(slot, coeff) in &row.sparse_coeffs {
            qfrc[slot as usize] += coeff * f_dt;
        }
    }
    qfrc
}

// ---------------------------------------------------------------------------
// Tree-involved contact solver
// ---------------------------------------------------------------------------

/// Constraint forces for contacts with at least one tree link. The contact
/// rows share one generalized system across all free bodies and trees, so a
/// tree-vs-body contact transfers the impulse to both participants.
#[derive(Clone, Debug)]
pub struct TreeContactSolution {
    /// Original tree contact list consumed by this solve, in deterministic
    /// narrow-phase order.
    pub contacts: Vec<Contact>,
    /// Solver-row to original-contact mapping. Contacts in a force-free gap
    /// have no entries here.
    pub row_to_contact: Vec<usize>,
    /// World-frame free-body wrenches, indexed like the input body slice.
    pub body_wrenches: Vec<(Vec3, Vec3)>,
    /// Joint-space generalized forces, indexed by tree and then `nv` slot.
    pub tree_qfrc: Vec<Vec<f32>>,
    /// Per-link world-frame wrenches equivalent to the solved contact
    /// impulses. This feeds post-step acceleration and force sensors.
    pub tree_wrenches: Vec<crate::tree::ExternalWrenches>,
    /// Normal forces, indexed by the original input contact index.
    pub contact_normal_forces: Vec<f32>,
    /// Dense `J M⁻¹ Jᵀ` response for the compact active contact rows.
    /// Rows follow the order assembled from `contacts`.
    pub contact_response: Vec<f32>,
    /// Source factors for the assembled active contact rows.
    pub row_diagnostics: Vec<ConstraintRowDiagnostic>,
}

#[derive(Clone, Debug)]
struct WorldContactRow {
    components: Vec<WorldJacobian>,
    reg: f32,
    diag: f32,
    bias: f32,
}

#[derive(Clone, Debug)]
struct WorldJacobian {
    body: Option<WorldBodyJacobian>,
    tree: Option<WorldTreeJacobian>,
}

#[derive(Clone, Copy, Debug)]
struct WorldBodyJacobian {
    index: usize,
    linear: Vec3,
    angular: Vec3,
}

#[derive(Clone, Debug)]
struct WorldTreeJacobian {
    index: usize,
    link: usize,
    coefficients: Vec<f32>,
    wrench_linear: Vec3,
    wrench_angular: Vec3,
}

struct WorldContactBlock {
    contact_index: usize,
    start_row: usize,
    condim: u8,
    solref: SolRef,
    solimp: SolImp,
    penetration: f32,
    mu_slide: f32,
    mu_torsion: f32,
    mu_roll: f32,
}

/// Solve all active contacts that touch a tree in one joint-space system.
///
/// `tree_implicit` selects the tree free-velocity model used by the
/// integrator: `None` is explicit RK4, `Some(false)` is Euler with implicit
/// joint damping, and `Some(true)` is implicitfast. The returned impulses are
/// converted to body wrenches and tree generalized forces by the caller.
#[allow(clippy::too_many_arguments)]
pub fn solve_tree_contacts(
    bodies: &[Body],
    trees: &[Tree],
    geoms: &[Geom],
    contacts: &[Contact],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
    use_newton: bool,
    tree_implicit: Option<bool>,
) -> TreeContactSolution {
    let mut solution = TreeContactSolution {
        contacts: contacts.to_vec(),
        row_to_contact: Vec::new(),
        body_wrenches: vec![(Vec3::ZERO, Vec3::ZERO); bodies.len()],
        tree_qfrc: trees.iter().map(|tree| vec![0.0; tree.nv()]).collect(),
        tree_wrenches: trees
            .iter()
            .map(|tree| vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()])
            .collect(),
        contact_normal_forces: vec![0.0; contacts.len()],
        contact_response: Vec::new(),
        row_diagnostics: Vec::new(),
    };
    if contacts.is_empty() || dt <= 0.0 || trees.is_empty() {
        return solution;
    }
    if use_newton && cone == ConeKind::Elliptic {
        panic!("{NEWTON_ELLIPTIC_ERROR}");
    }

    let poses: Vec<Vec<(Vec3, Quat)>> = trees.iter().map(forward_kinematics).collect();
    let inv_i_world: Vec<crate::math::Mat3> = bodies
        .iter()
        .map(|body| {
            let rotation = body.orientation.to_mat3();
            rotation * body.inertia_body_inverse * rotation.transpose()
        })
        .collect();
    let tree_factors: Vec<Vec<f32>> = trees
        .iter()
        .map(|tree| {
            let n = tree.nv();
            if n == 0 {
                Vec::new()
            } else {
                let response_matrix = match tree_implicit {
                    None => mass_matrix(tree),
                    Some(implicit_fast) => tree.implicit_mass_matrix(dt, implicit_fast),
                };
                cholesky(&response_matrix, n).unwrap_or_else(|| {
                    panic!("tree contact solve failed: mass matrix is not positive definite")
                })
            }
        })
        .collect();
    // MuJoCo's diagApprox is a model-time body inverse weight. It does not
    // use the exact contact response or the implicit damping matrix.
    let tree_diag_factors: Vec<Vec<f32>> = trees
        .iter()
        .map(|tree| {
            let n = tree.nv();
            if n == 0 {
                Vec::new()
            } else {
                cholesky(&mass_matrix(tree), n).unwrap_or_else(|| {
                    panic!("tree diagApprox failed: mass matrix is not positive definite")
                })
            }
        })
        .collect();
    let tree_free_velocity: Vec<Vec<f32>> = trees
        .iter()
        .map(|tree| tree_free_velocity_delta(tree, gravity, dt, tree_implicit))
        .collect();

    let mut rows = Vec::new();
    let mut blocks = Vec::new();
    for (contact_index, contact) in contacts.iter().enumerate() {
        let ga = &geoms[contact.geom_a];
        let gb = &geoms[contact.geom_b];
        if !matches!(ga.attachment(), GeomAttach::Link(_, _))
            && !matches!(gb.attachment(), GeomAttach::Link(_, _))
        {
            continue;
        }
        let penetration = contact.penetration - contact.gap;
        if penetration <= 0.0 {
            continue;
        }
        let solref = combine_solref(ga.solref, gb.solref);
        let solimp = combine_solimp(ga.solimp, gb.solimp);
        let condim = ga.condim.min(gb.condim);
        let (t1, t2) = tangent_basis(contact.normal_world);
        let mu_slide = crate::contact::combine_friction(ga.friction, gb.friction);
        let row_components: Vec<Vec<WorldJacobian>> =
            contact_row_directions(contact.normal_world, t1, t2, condim, cone, mu_slide)
                .into_iter()
                .map(|(direction, angular)| {
                    [
                        world_jacobian_for_side(
                            ga.attachment(),
                            1.0,
                            contact.position_world,
                            direction,
                            angular,
                            bodies,
                            trees,
                            &poses,
                        ),
                        world_jacobian_for_side(
                            gb.attachment(),
                            -1.0,
                            contact.position_world,
                            direction,
                            angular,
                            bodies,
                            trees,
                            &poses,
                        ),
                    ]
                    .into_iter()
                    .flatten()
                    .collect()
                })
                .collect();
        if row_components.iter().all(|components| {
            components
                .iter()
                .all(|component| !component_is_dynamic(component, trees))
        }) {
            continue;
        }
        let start_row = rows.len();
        for components in row_components {
            rows.push(WorldContactRow {
                components,
                reg: 0.0,
                diag: 0.0,
                bias: 0.0,
            });
        }
        blocks.push(WorldContactBlock {
            contact_index,
            start_row,
            condim,
            solref,
            solimp,
            penetration,
            mu_slide,
            mu_torsion: crate::geom::combine_torsional_friction(
                ga.torsional_friction,
                gb.torsional_friction,
            ),
            mu_roll: crate::geom::combine_rolling_friction(
                ga.rolling_friction,
                gb.rolling_friction,
            ),
        });
    }
    if rows.is_empty() {
        return solution;
    }

    let n_rows = rows.len();
    for block in &blocks {
        let row_count = contact_block_n_rows(block.condim, cone);
        solution
            .row_to_contact
            .extend(std::iter::repeat_n(block.contact_index, row_count));
    }
    let mut response = vec![0.0f32; n_rows * n_rows];
    for column in 0..n_rows {
        let mut body_response = vec![(Vec3::ZERO, Vec3::ZERO); bodies.len()];
        let mut tree_response: Vec<Vec<f32>> =
            trees.iter().map(|tree| vec![0.0; tree.nv()]).collect();
        apply_world_impulse(
            &rows[column],
            1.0,
            &mut body_response,
            &mut tree_response,
            bodies,
            trees,
            &inv_i_world,
            &tree_factors,
        );
        for (row_index, row) in rows.iter().enumerate() {
            response[row_index * n_rows + column] =
                world_velocity_from_delta(row, &body_response, &tree_response, bodies);
        }
    }
    solution.contact_response = response.clone();

    for block in &blocks {
        let row_count = contact_block_n_rows(block.condim, cone);
        for row_offset in 0..row_count {
            let row_index = block.start_row + row_offset;
            let impedance_value = impedance_at_position(-block.penetration, 0.0, block.solimp);
            let diag_approx = tree_contact_diag_approx(
                &rows[row_index],
                bodies,
                trees,
                &inv_i_world,
                &tree_diag_factors,
                block.mu_slide,
                cone == ConeKind::Pyramidal && block.condim == 3,
            );
            rows[row_index].reg = contact_regularization_from_diag(
                diag_approx,
                impedance_value,
                block.mu_slide,
                cone == ConeKind::Pyramidal && block.condim == 3,
            );
            let diagonal = response[row_index * n_rows + row_index];
            rows[row_index].diag = diagonal + rows[row_index].reg;
            let velocity = world_current_velocity(&rows[row_index], bodies, trees);
            let free_velocity =
                world_free_velocity(&rows[row_index], &tree_free_velocity, trees, gravity, dt);
            let position = if (cone == ConeKind::Pyramidal && block.condim == 3) || row_offset == 0
            {
                -block.penetration
            } else {
                0.0
            };
            let reference = reference_accel(
                position,
                velocity,
                safe_solref(block.solref, dt),
                block.solimp,
            );
            rows[row_index].bias = free_velocity - reference * dt;
            let (damping, stiffness, impedance) =
                reference_coefficients(position, safe_solref(block.solref, dt), block.solimp);
            solution.row_diagnostics.push(ConstraintRowDiagnostic {
                position,
                velocity,
                stiffness,
                damping,
                impedance,
                regularization: rows[row_index].reg,
                reference_accel: reference,
            });
        }
    }

    let impulses = if use_newton {
        let mut hessian = response.clone();
        for (index, row) in rows.iter().enumerate() {
            hessian[index * n_rows + index] += row.reg;
        }
        let mut projections = Vec::new();
        for block in &blocks {
            let normal = block.start_row;
            if cone == ConeKind::Pyramidal && block.condim == 3 {
                for offset in 0..4 {
                    projections.push(crate::newton::Projection::NonNegative {
                        index: normal + offset,
                    });
                }
            } else {
                projections.push(crate::newton::Projection::NonNegative { index: normal });
            }
            if cone != ConeKind::Pyramidal && block.condim >= 3 {
                projections.push(crate::newton::Projection::PyramidalCone {
                    normal,
                    tangent_1: normal + 1,
                    tangent_2: normal + 2,
                    mu: block.mu_slide,
                });
                if block.condim >= 4 {
                    projections.push(crate::newton::Projection::ScalarConeBound {
                        index: normal + 3,
                        normal,
                        mu: block.mu_torsion,
                    });
                }
                if block.condim >= 6 {
                    projections.push(crate::newton::Projection::PyramidalCone {
                        normal,
                        tangent_1: normal + 4,
                        tangent_2: normal + 5,
                        mu: block.mu_roll,
                    });
                }
            }
        }
        if crate::dynamics::cholesky(&hessian, n_rows).is_none()
            && blocks.iter().any(|block| {
                cone == ConeKind::Pyramidal && block.condim == 3 && block.mu_slide == 0.0
            })
        {
            // Zero-friction pyramid facets are identical rows. MuJoCo keeps
            // their source Rpy at zero, so the resulting Hessian is positive
            // semidefinite. Add a solver-only pivot for Cholesky without
            // changing the assembled constraint regularizer.
            let scale = hessian
                .iter()
                .copied()
                .fold(0.0f32, |max_value, value| max_value.max(value.abs()));
            let pivot = (scale * 1.0e-6).max(1.0e-7);
            for index in 0..n_rows {
                hessian[index * n_rows + index] += pivot;
            }
        }
        crate::newton::NewtonSystem {
            hessian,
            linear: rows.iter().map(|row| row.bias).collect(),
            projections,
            max_iterations: iterations.max(1),
            cost_tolerance: 1e-7,
        }
        .solve()
        .unwrap_or_else(|error| panic!("Newton tree contact solve failed: {error}"))
        .solution
    } else {
        let mut impulses = vec![0.0f32; n_rows];
        for _ in 0..iterations {
            for block in &blocks {
                let normal = block.start_row;
                world_pgs_non_negative(&rows, &response, normal, &mut impulses);
                if cone == ConeKind::Pyramidal && block.condim == 3 {
                    for offset in 1..4 {
                        world_pgs_non_negative(&rows, &response, normal + offset, &mut impulses);
                    }
                } else if block.condim >= 3 {
                    world_pgs_pair(
                        &rows,
                        &response,
                        normal + 1,
                        normal + 2,
                        block.mu_slide,
                        impulses[normal],
                        cone,
                        &mut impulses,
                    );
                    if block.condim >= 4 {
                        world_pgs_scalar(
                            &rows,
                            &response,
                            normal + 3,
                            block.mu_torsion * impulses[normal],
                            &mut impulses,
                        );
                    }
                    if block.condim >= 6 {
                        world_pgs_pair(
                            &rows,
                            &response,
                            normal + 4,
                            normal + 5,
                            block.mu_roll,
                            impulses[normal],
                            cone,
                            &mut impulses,
                        );
                    }
                }
            }
        }
        impulses
    };

    for block in &blocks {
        let row_count = contact_block_n_rows(block.condim, cone);
        let normal_impulse = if cone == ConeKind::Pyramidal && block.condim == 3 {
            (0..row_count)
                .map(|offset| impulses[block.start_row + offset])
                .sum()
        } else {
            impulses[block.start_row]
        };
        solution.contact_normal_forces[block.contact_index] = normal_impulse / dt;
    }
    for (row_index, row) in rows.iter().enumerate() {
        let force = impulses[row_index] / dt;
        for component in &row.components {
            if let Some(body) = component.body {
                let output = &mut solution.body_wrenches[body.index];
                output.0 += body.linear * force;
                output.1 += body.angular * force;
            }
            if let Some(tree) = &component.tree {
                if !tree_is_mocap(tree.index, trees) {
                    for (slot, coefficient) in tree.coefficients.iter().enumerate() {
                        solution.tree_qfrc[tree.index][slot] += coefficient * force;
                    }
                    let wrench = &mut solution.tree_wrenches[tree.index][tree.link];
                    wrench.0 += tree.wrench_linear * force;
                    wrench.1 += tree.wrench_angular * force;
                }
            }
        }
    }
    solution
}

/// Assemble tree contact rows and return their source factors.
#[allow(clippy::too_many_arguments)]
pub fn diagnose_tree_contact_rows(
    bodies: &[Body],
    trees: &[Tree],
    geoms: &[Geom],
    contacts: &[Contact],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> Vec<ConstraintRowDiagnostic> {
    solve_tree_contacts(
        bodies, trees, geoms, contacts, gravity, dt, cone, iterations, false, None,
    )
    .row_diagnostics
}

fn component_is_dynamic(component: &WorldJacobian, trees: &[Tree]) -> bool {
    component.body.is_some()
        || component
            .tree
            .as_ref()
            .is_some_and(|tree| !tree_is_mocap(tree.index, trees))
}

fn tree_is_mocap(tree_index: usize, trees: &[Tree]) -> bool {
    trees[tree_index]
        .links
        .first()
        .is_some_and(|link| link.mocap)
}

fn contact_row_directions(
    normal: Vec3,
    tangent_1: Vec3,
    tangent_2: Vec3,
    condim: u8,
    cone: ConeKind,
    slide_friction: f32,
) -> Vec<(Vec3, bool)> {
    if cone == ConeKind::Pyramidal && condim == 3 {
        return vec![
            (normal + tangent_1 * slide_friction, false),
            (normal - tangent_1 * slide_friction, false),
            (normal + tangent_2 * slide_friction, false),
            (normal - tangent_2 * slide_friction, false),
        ];
    }
    let mut directions = vec![(normal, false)];
    if condim >= 3 {
        directions.extend([(tangent_1, false), (tangent_2, false)]);
    }
    if condim >= 4 {
        directions.push((normal, true));
    }
    if condim >= 6 {
        directions.extend([(tangent_1, true), (tangent_2, true)]);
    }
    directions
}

#[allow(clippy::too_many_arguments)]
fn world_jacobian_for_side(
    attachment: GeomAttach,
    sign: f32,
    contact_position: Vec3,
    direction: Vec3,
    angular: bool,
    bodies: &[Body],
    trees: &[Tree],
    poses: &[Vec<(Vec3, Quat)>],
) -> Option<WorldJacobian> {
    match attachment {
        GeomAttach::Static => None,
        GeomAttach::Body(index) => {
            let arm = contact_position - bodies[index].position;
            Some(WorldJacobian {
                body: Some(WorldBodyJacobian {
                    index,
                    linear: if angular {
                        Vec3::ZERO
                    } else {
                        direction * sign
                    },
                    angular: if angular {
                        direction * sign
                    } else {
                        arm.cross(direction) * sign
                    },
                }),
                tree: None,
            })
        }
        GeomAttach::Link(tree_index, link_index) => {
            let (com, orientation) = poses[tree_index][link_index];
            let local_point = orientation.inverse_rotate(contact_position - com);
            let jacobian = trees[tree_index].point_jacobian(link_index, local_point);
            let source = if angular {
                jacobian.rotational
            } else {
                jacobian.translational
            };
            Some(WorldJacobian {
                body: None,
                tree: Some(WorldTreeJacobian {
                    index: tree_index,
                    link: link_index,
                    coefficients: source
                        .into_iter()
                        .map(|column| column.dot(direction) * sign)
                        .collect(),
                    wrench_linear: if angular {
                        Vec3::ZERO
                    } else {
                        direction * sign
                    },
                    wrench_angular: if angular {
                        direction * sign
                    } else {
                        (contact_position - com).cross(direction * sign)
                    },
                }),
            })
        }
    }
}

fn tree_free_velocity_delta(
    tree: &Tree,
    gravity: Vec3,
    dt: f32,
    tree_implicit: Option<bool>,
) -> Vec<f32> {
    if tree.nv() == 0 || tree.links.first().is_some_and(|link| link.mocap) {
        return Vec::new();
    }
    let mut free_tree = tree.clone();
    free_tree.disable_penalty_limits = true;
    let poses = forward_kinematics(&free_tree);
    let external = vec![(Vec3::ZERO, Vec3::ZERO); free_tree.links.len()];
    let qddot = match tree_implicit {
        None => crate::tree::aba(&free_tree, &poses, gravity, &external),
        Some(implicit_fast) => {
            crate::tree::aba_implicit(&free_tree, &poses, gravity, &external, dt, implicit_fast)
        }
    };
    qddot.into_iter().map(|accel| accel * dt).collect()
}

fn world_current_velocity(row: &WorldContactRow, bodies: &[Body], trees: &[Tree]) -> f32 {
    row.components
        .iter()
        .map(|component| {
            let body_velocity = component.body.map_or(0.0, |body| {
                body.linear.dot(bodies[body.index].linear_velocity)
                    + body
                        .angular
                        .dot(bodies[body.index].angular_velocity_world())
            });
            let tree_velocity = component.tree.as_ref().map_or(0.0, |tree| {
                let state = &trees[tree.index];
                if state.links.first().is_some_and(|link| link.mocap) {
                    state.mocap_linear_velocity.dot(tree.wrench_linear)
                        + state.mocap_angular_velocity.dot(tree.wrench_angular)
                } else {
                    tree.coefficients
                        .iter()
                        .zip(&state.qdot)
                        .map(|(coefficient, velocity)| coefficient * velocity)
                        .sum()
                }
            });
            body_velocity + tree_velocity
        })
        .sum()
}

fn world_free_velocity(
    row: &WorldContactRow,
    tree_free_velocity: &[Vec<f32>],
    trees: &[Tree],
    gravity: Vec3,
    dt: f32,
) -> f32 {
    row.components
        .iter()
        .map(|component| {
            let body_velocity = component
                .body
                .map_or(0.0, |body| body.linear.dot(gravity * dt));
            let tree_velocity = component.tree.as_ref().map_or(0.0, |tree| {
                if tree_is_mocap(tree.index, trees) {
                    0.0
                } else {
                    tree.coefficients
                        .iter()
                        .zip(&tree_free_velocity[tree.index])
                        .map(|(coefficient, velocity)| coefficient * velocity)
                        .sum()
                }
            });
            body_velocity + tree_velocity
        })
        .sum()
}

fn world_velocity_from_delta(
    row: &WorldContactRow,
    body_delta: &[(Vec3, Vec3)],
    tree_delta: &[Vec<f32>],
    bodies: &[Body],
) -> f32 {
    row.components
        .iter()
        .map(|component| {
            let body_velocity = component.body.map_or(0.0, |body| {
                body.linear.dot(body_delta[body.index].0)
                    + body.angular.dot(
                        bodies[body.index]
                            .orientation
                            .rotate(body_delta[body.index].1),
                    )
            });
            let tree_velocity = component.tree.as_ref().map_or(0.0, |tree| {
                tree.coefficients
                    .iter()
                    .zip(&tree_delta[tree.index])
                    .map(|(coefficient, velocity)| coefficient * velocity)
                    .sum()
            });
            body_velocity + tree_velocity
        })
        .sum()
}

/// Compute MuJoCo's body-level inverse weights for one tree link. MuJoCo
/// averages the diagonal of `J_body M^-1 J_body^T` at model setup. The tree
/// solver uses the same definition at the assembled configuration.
fn tree_body_invweight0(tree: &Tree, link: usize, factor: &[f32]) -> (f32, f32) {
    let jacobian = tree.link_jacobian(link);
    let mut translation = 0.0;
    let mut rotation = 0.0;
    for axis in 0..3 {
        let mut linear = vec![0.0; tree.nv()];
        let mut angular = vec![0.0; tree.nv()];
        for slot in 0..tree.nv() {
            let linear_axis = match axis {
                0 => jacobian.translational[slot].x,
                1 => jacobian.translational[slot].y,
                _ => jacobian.translational[slot].z,
            };
            let angular_axis = match axis {
                0 => jacobian.rotational[slot].x,
                1 => jacobian.rotational[slot].y,
                _ => jacobian.rotational[slot].z,
            };
            linear[slot] = linear_axis;
            angular[slot] = angular_axis;
        }
        let linear_response = cholesky_solve(factor, tree.nv(), &linear);
        let angular_response = cholesky_solve(factor, tree.nv(), &angular);
        translation += linear
            .iter()
            .zip(&linear_response)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        rotation += angular
            .iter()
            .zip(&angular_response)
            .map(|(a, b)| a * b)
            .sum::<f32>();
    }
    (translation / 3.0, rotation / 3.0)
}

fn tree_contact_diag_approx(
    row: &WorldContactRow,
    bodies: &[Body],
    trees: &[Tree],
    inv_i_world: &[crate::math::Mat3],
    tree_diag_factors: &[Vec<f32>],
    mu_slide: f32,
    pyramidal_condim3: bool,
) -> f32 {
    let mut translation = 0.0;
    let mut rotation = 0.0;
    for component in &row.components {
        if let Some(body) = component.body {
            if body.linear != Vec3::ZERO {
                translation += 1.0 / bodies[body.index].mass;
            } else {
                rotation += body.angular.dot(inv_i_world[body.index] * body.angular);
            }
        }
        if let Some(tree) = &component.tree {
            if tree_is_mocap(tree.index, trees) {
                continue;
            }
            let (body_translation, body_rotation) = tree_body_invweight0(
                &trees[tree.index],
                tree.link,
                &tree_diag_factors[tree.index],
            );
            if tree.wrench_linear != Vec3::ZERO {
                translation += body_translation;
            } else {
                rotation += body_rotation;
            }
        }
    }
    if pyramidal_condim3 {
        translation * (1.0 + mu_slide * mu_slide)
    } else {
        translation + rotation
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_world_impulse(
    row: &WorldContactRow,
    impulse: f32,
    body_delta: &mut [(Vec3, Vec3)],
    tree_delta: &mut [Vec<f32>],
    bodies: &[Body],
    trees: &[Tree],
    inv_i_world: &[crate::math::Mat3],
    tree_factors: &[Vec<f32>],
) {
    for component in &row.components {
        if let Some(body) = component.body {
            let body_state = &bodies[body.index];
            let force = body.linear * impulse;
            let torque = body.angular * impulse;
            let angular_velocity = inv_i_world[body.index] * torque;
            body_delta[body.index].0 += force / body_state.mass;
            body_delta[body.index].1 += body_state.orientation.inverse_rotate(angular_velocity);
        }
        if let Some(tree) = &component.tree {
            if tree_is_mocap(tree.index, trees) {
                continue;
            }
            let mut generalized_impulse = vec![0.0; tree.coefficients.len()];
            for (slot, coefficient) in tree.coefficients.iter().enumerate() {
                generalized_impulse[slot] = coefficient * impulse;
            }
            let response = cholesky_solve(
                &tree_factors[tree.index],
                generalized_impulse.len(),
                &generalized_impulse,
            );
            for (slot, velocity) in response.into_iter().enumerate() {
                tree_delta[tree.index][slot] += velocity;
            }
        }
    }
}

fn world_pgs_residual(
    rows: &[WorldContactRow],
    response: &[f32],
    index: usize,
    impulses: &[f32],
) -> f32 {
    let n_rows = rows.len();
    rows[index].bias
        + (0..n_rows)
            .map(|column| response[index * n_rows + column] * impulses[column])
            .sum::<f32>()
}

fn world_pgs_non_negative(
    rows: &[WorldContactRow],
    response: &[f32],
    index: usize,
    impulses: &mut [f32],
) {
    let residual =
        world_pgs_residual(rows, response, index, impulses) + rows[index].reg * impulses[index];
    let projected = (impulses[index] - residual / rows[index].diag).max(0.0);
    impulses[index] = projected;
}

fn world_pgs_scalar(
    rows: &[WorldContactRow],
    response: &[f32],
    index: usize,
    cap: f32,
    impulses: &mut [f32],
) {
    let residual =
        world_pgs_residual(rows, response, index, impulses) + rows[index].reg * impulses[index];
    impulses[index] = project_pyramidal(impulses[index] - residual / rows[index].diag, cap);
}

#[allow(clippy::too_many_arguments)]
fn world_pgs_pair(
    rows: &[WorldContactRow],
    response: &[f32],
    index_1: usize,
    index_2: usize,
    mu: f32,
    normal_impulse: f32,
    cone: ConeKind,
    impulses: &mut [f32],
) {
    match cone {
        ConeKind::Pyramidal => {
            let cap = mu * normal_impulse;
            world_pgs_scalar(rows, response, index_1, cap, impulses);
            world_pgs_scalar(rows, response, index_2, cap, impulses);
        }
        ConeKind::Elliptic => {
            let residual_1 = world_pgs_residual(rows, response, index_1, impulses)
                + rows[index_1].reg * impulses[index_1];
            let residual_2 = world_pgs_residual(rows, response, index_2, impulses)
                + rows[index_2].reg * impulses[index_2];
            let (projected_1, projected_2) = project_elliptic(
                impulses[index_1] - residual_1 / rows[index_1].diag,
                impulses[index_2] - residual_2 / rows[index_2].diag,
                mu,
                normal_impulse,
            );
            impulses[index_1] = projected_1;
            impulses[index_2] = projected_2;
        }
    }
}

/// Contribution of the "free evolution" (gravity + gyroscopic) step to
/// `J_i · qdot_free`. Uses only the free linear velocity delta because our
/// scope drops gyroscopic within one step (see the qdot_free note in
/// [`solve_free_bodies`]).
#[allow(clippy::too_many_arguments)]
fn free_step_velocity_contribution(
    dv_lin_free_per_body: &[Vec3],
    dw_body_free_per_body: &[Vec3],
    body_a: Option<u32>,
    body_b: Option<u32>,
    arm_a: Vec3,
    arm_b: Vec3,
    dir: Vec3,
    bodies: &[Body],
) -> f32 {
    let mut s = 0.0f32;
    if let Some(i) = body_a {
        let i = i as usize;
        let dw_world = bodies[i].orientation.rotate(dw_body_free_per_body[i]);
        let v_point = dv_lin_free_per_body[i] + dw_world.cross(arm_a);
        s += v_point.dot(dir);
    }
    if let Some(i) = body_b {
        let i = i as usize;
        let dw_world = bodies[i].orientation.rotate(dw_body_free_per_body[i]);
        let v_point = dv_lin_free_per_body[i] + dw_world.cross(arm_b);
        s -= v_point.dot(dir);
    }
    s
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
    fn signed_position_keeps_reference_acceleration_direction() {
        let solimp = SolImp::new(0.8, 0.9, 0.001, 0.25, 2);
        let solref = crate::geom::SolRef::new(0.02, 1.0);
        let position = -0.00025;
        let d = impedance(position, solimp);
        let k = d / (solimp.dmax * solimp.dmax * 0.02 * 0.02);
        let expected = k * 0.00025;
        approx(
            reference_accel(position, 0.0, solref, solimp),
            expected,
            1e-5,
        );
        assert!(reference_accel(position, 0.0, solref, solimp) > 0.0);
    }

    #[test]
    fn signed_position_sigmoid_applies_margin_before_absolute_distance() {
        let solimp = SolImp::new(0.8, 0.9, 0.001, 0.5, 2);
        // x = abs((-0.00025 - 0.00025) / 0.001) = 0.5, so y = 0.5.
        approx(impedance_at_position(-0.00025, 0.00025, solimp), 0.85, 1e-6);
    }

    #[test]
    fn effective_solimp_clamps_source_ranges() {
        let unclamped = SolImp::new(0.0, 1.2, 0.001, 0.5, 2);
        let expected = 0.0001 + (0.9999 - 0.0001) * 0.125;
        approx(impedance(0.00025, unclamped), expected, 1e-6);
    }

    #[test]
    fn safe_solref_clamps_timeconst_to_two_timesteps() {
        let dt = 0.01;
        let solref = crate::geom::SolRef::new(0.005, 1.0);
        let effective = safe_solref(solref, dt);
        assert_eq!(effective.timeconst, 2.0 * dt);
        let solimp = SolImp::DEFAULT;
        let position = 0.00025;
        let velocity = 0.3;
        let d = impedance(position, solimp);
        let b = 2.0 / (solimp.dmax * effective.timeconst);
        let k = d
            / (solimp.dmax
                * solimp.dmax
                * effective.timeconst
                * effective.timeconst
                * effective.dampratio
                * effective.dampratio);
        let expected = -b * velocity - k * position;
        approx(
            reference_accel(position, velocity, effective, solimp),
            expected,
            1e-5,
        );
    }

    #[test]
    fn pyramidal_regularizer_uses_shared_rpy_rule() {
        let diagonal_approx = 0.7721514;
        let impedance_value = 0.95;
        let mu = 0.6;
        let expected = 2.0 * mu * mu * (1.0 - impedance_value) / impedance_value * diagonal_approx;
        approx(
            contact_regularization_from_diag(diagonal_approx, impedance_value, mu, true),
            expected,
            1e-7,
        );
    }

    #[test]
    fn reference_accel_signs() {
        let solref = crate::geom::SolRef::new(0.02, 1.0);
        let solimp = SolImp::DEFAULT;
        let a = reference_accel(0.001, 0.0, solref, solimp);
        assert!(a < 0.0);
        let d = impedance(0.001, solimp);
        let k = d / (solimp.dmax * solimp.dmax * 0.02 * 0.02);
        approx(a, -k * 0.001, 1e-5);
        let a2 = reference_accel(0.001, 0.1, solref, solimp);
        assert!(a2 < a, "damping term must push a_ref more negative");
    }

    #[test]
    fn reference_accel_exact_hand_anchors() {
        let solimp = SolImp::new(0.8, 0.9, 0.001, 0.5, 2);
        let cases = [(0.005, 0.5), (0.1, 1.0), (0.2, 2.0)];
        for (tc, zeta) in cases {
            let solref = crate::geom::SolRef::new(tc, zeta);
            let violation = 0.00025;
            let d = impedance(violation, solimp);
            let b = 2.0 / (solimp.dmax * tc);
            let k = d / (solimp.dmax * solimp.dmax * tc * tc * zeta * zeta);
            let expected = -b * 0.3 - k * violation;
            approx(
                reference_accel(violation, 0.3, solref, solimp),
                expected,
                1e-4,
            );
        }
    }

    #[test]
    fn reference_accel_direct_hand_anchor() {
        let solref = crate::geom::SolRef::new(-400.0, -12.0);
        let expected = -(12.0 / 0.95) * 0.25 - (400.0 / (0.95 * 0.95)) * 0.002;
        approx(
            reference_accel(0.002, 0.25, solref, SolImp::DEFAULT),
            expected,
            1e-6,
        );
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
