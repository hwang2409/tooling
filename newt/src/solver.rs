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
use crate::geom::{Geom, SolRef, combine_solref};
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
    /// Bias `b_i = J_i · qdot_free + d · a_ref * dt`.
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
    for c in contacts.iter() {
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
        let arm_a = match body_a {
            Some(i) => c.position_world - bodies[i as usize].position,
            None => Vec3::ZERO,
        };
        let arm_b = match body_b {
            Some(i) => c.position_world - bodies[i as usize].position,
            None => Vec3::ZERO,
        };

        let start_row = rows.len() as u32;
        let v_n_start = normal_velocity_at_point(bodies, body_a, body_b, arm_a, arm_b, n_world);

        // Normal row.
        rows.push(ConstraintRow {
            dir_world: n_world,
            body_a,
            body_b,
            arm_a,
            arm_b,
            reg: 0.0,
            diag: 0.0,
            bias: 0.0,
            geom: RowGeom::Linear,
        });

        let mut n_rows_here: u8 = 1;
        // Sliding tangent rows (condim >= 3).
        if condim >= 3 {
            for &dir in &[t1_world, t2_world] {
                rows.push(ConstraintRow {
                    dir_world: dir,
                    body_a,
                    body_b,
                    arm_a,
                    arm_b,
                    reg: 0.0,
                    diag: 0.0,
                    bias: 0.0,
                    geom: RowGeom::Linear,
                });
            }
            n_rows_here += 2;
        }
        // Torsional row about the contact normal (condim >= 4). Angular.
        if condim >= 4 {
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
            n_rows_here += 1;
        }
        // Rolling rows about the two tangents (condim == 6). Angular.
        if condim >= 6 {
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
            n_rows_here += 2;
        }

        per_contact.push(PerContact {
            start_row,
            condim,
            solref,
            solimp,
            pen_active,
            v_n_current: v_n_start,
            mu_slide: mu,
            mu_torsion,
            mu_roll,
        });
        let _ = n_rows_here; // block boundaries derive from `condim`.
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
        for k in 0..contact_block_n_rows(pc.condim) {
            let ri = pc.start_row as usize + k;
            let a_ii = row_body_diagonal(&rows[ri], bodies, &inv_i_world);
            let d = impedance(pc.pen_active, pc.solimp);
            let reg = if d > 0.0 { (1.0 - d) / d * a_ii } else { 0.0 };
            rows[ri].reg = reg;
            rows[ri].diag = a_ii + reg;
        }
        // Normal row bias includes the impedance-scaled reference accel.
        let r = pc.pen_active;
        let r_dot = -pc.v_n_current;
        let a_ref = reference_accel(r, r_dot, pc.solref);
        let d = impedance(pc.pen_active, pc.solimp);
        let n_row = pc.start_row as usize;
        rows[n_row].bias = pc.v_n_current
            + row_free_step_velocity(
                &rows[n_row],
                &dv_lin_free_per_body,
                &dw_body_free_per_body,
                bodies,
            )
            + d * a_ref * dt;
        // All non-normal rows in a contact block have a "target velocity =
        // 0" reference (stick / no spin / no roll). Bias = current
        // constraint velocity + free-step delta.
        for k in 1..contact_block_n_rows(pc.condim) {
            let ri = pc.start_row as usize + k;
            let v_cur = row_current_velocity(&rows[ri], bodies);
            let dv_free = row_free_step_velocity(
                &rows[ri],
                &dv_lin_free_per_body,
                &dw_body_free_per_body,
                bodies,
            );
            rows[ri].bias = v_cur + dv_free;
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
            let r_dot = -v_cur;
            let a_ref = reference_accel(r, r_dot, pe.solref);
            rows[ri].bias = v_cur + dv_free + d * a_ref * dt;
        }
    }

    // ---- PGS iteration ----------------------------------------------------
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
            if pc.condim >= 3 {
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

    // Per-contact normal FORCE = normal-row impulse / dt. The touch sensor
    // consumes this to report the actual constraint-computed normal force
    // rather than the penalty-formula approximation. `per_contact` was
    // pushed in the same order as `contacts` so the mapping is 1:1.
    for (i, pc) in per_contact.iter().enumerate() {
        contact_normal_forces[i] = impulses[pc.start_row as usize] / dt;
    }

    (wrenches, contact_normal_forces)
}

/// Per-contact solver bookkeeping shared across all rows of one contact.
struct PerContact {
    start_row: u32,
    condim: u8,
    solref: SolRef,
    solimp: SolImp,
    pen_active: f32,
    v_n_current: f32,
    mu_slide: f32,
    mu_torsion: f32,
    mu_roll: f32,
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
fn contact_block_n_rows(condim: u8) -> usize {
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
use crate::tree::Tree;

/// One assembled tree-space row. Both limits and joint-coupling
/// equalities reduce to a sparse row on the tree's `nv` velocity vector:
/// a limit has one non-zero component (`+1` at the limited DOF's slot,
/// negated for the high side); a coupling has two (`+1` at `v_slot_a`,
/// `-k` at `v_slot_b`, where `k = c1 + 2·c2·q_b` — the derivative of the
/// polynomial constraint). Storing them uniformly lets one PGS sweep
/// handle both.
///
/// `sparse_coeffs` is at most two `(slot, coeff)` pairs; the second is
/// zero for a plain limit row.
#[derive(Clone, Copy, Debug)]
struct TreeRow {
    /// Two (slot, coefficient) pairs. For a limit only the first is
    /// populated; the second slot is set equal to the first and its
    /// coefficient is 0.0 (a no-op in dot products).
    sparse_coeffs: [(u32, f32); 2],
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
        let (s0, c0) = self.sparse_coeffs[0];
        let (s1, c1) = self.sparse_coeffs[1];
        c0 * qdot[s0 as usize] + c1 * qdot[s1 as usize]
    }

    /// `e_i · d_j` — the dense `A_ij` entry, computed sparsely from this
    /// row's coefficients and the response vector `d_j` for row `j`.
    fn dot_response(&self, d_vec: &[f32]) -> f32 {
        let (s0, c0) = self.sparse_coeffs[0];
        let (s1, c1) = self.sparse_coeffs[1];
        c0 * d_vec[s0 as usize] + c1 * d_vec[s1 as usize]
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
                sparse_coeffs: [(v_slot, 1.0), (v_slot, 0.0)],
                violation: lo - q,
                solref: per_joint_solref,
                solimp: per_joint_solimp,
                projection: TreeRowProjection::NonNegative,
            });
        } else if q > hi {
            rows.push(TreeRow {
                sparse_coeffs: [(v_slot, -1.0), (v_slot, 0.0)],
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
            sparse_coeffs: [(slot_a, 1.0 * flip), (slot_b, -k * flip)],
            violation: r_signed.abs(),
            solref: *solref,
            solimp: *solimp,
            projection: TreeRowProjection::Bilateral,
        });
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
        for (slot, coeff) in row.sparse_coeffs {
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
        let r_dot = -v_row;
        let a_ref = reference_accel(row.violation, r_dot, row.solref);
        let d = impedance(row.violation, row.solimp);
        bias[i] = v_row + d * a_ref * dt;
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
        for (slot, coeff) in row.sparse_coeffs {
            if coeff != 0.0 {
                qfrc[slot as usize] += coeff * f_dt;
            }
        }
    }

    qfrc
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
