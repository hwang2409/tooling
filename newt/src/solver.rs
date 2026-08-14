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
use crate::geom::{Geom, SolRef, combine_solref};
use crate::math::Vec3;
use crate::world::tangent_basis;

/// One assembled scalar constraint (a single row of `J`). Kept opaque —
/// callers get their results as per-body wrenches.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // contact_idx/kind/cone kept for future diagnostics
struct ConstraintRow {
    /// Which contact this row belongs to (index into the caller's contact
    /// list). Multiple rows (normal, t1, t2) share the same contact_idx.
    contact_idx: u32,
    /// 0 = normal, 1 = tangent 1, 2 = tangent 2.
    kind: u8,
    /// Cone kind (only matters for tangent rows; copied here so the sweep
    /// can look up cheaply).
    cone: ConeKind,
    /// Pair Coulomb coefficient (only matters for tangent rows).
    friction: f32,
    /// Direction in world coordinates (n, t1, or t2). Same-magnitude
    /// projection basis; unit-length.
    dir_world: Vec3,
    /// Body A index (or `None` if A is static).
    body_a: Option<u32>,
    /// Body B index.
    body_b: Option<u32>,
    /// Arm from body A's COM to contact point in world coords (only used
    /// when body_a is Some). Zero for static.
    arm_a: Vec3,
    /// Arm from body B's COM to contact point in world coords.
    arm_b: Vec3,
    /// Diagonal regularization `R_ii = (1 - d) / d · A_ii`. Populated
    /// during assembly.
    reg: f32,
    /// Diagonal `A_ii + R_ii`. Populated during assembly.
    diag: f32,
    /// Bias `b_i = J_i · qdot_free + a_ref * dt` (normal rows only carry
    /// the a_ref term; tangent rows use 0 reference acceleration).
    bias: f32,
    /// Sign convention: `+dir_world` is the "escape" direction (from B into
    /// A for the normal). Tangents have no preferred sign; they map linearly.
    _pad: u8,
}

/// Per-body scratch: velocity offset accumulated from constraint impulses.
/// `dv_lin` in world coords, `dw_body` in body coords (so ω_body_new =
/// ω_body_start + dw_body + dw_body_from_free_step).
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
/// - `gravity`, `dt` — for computing `qdot_free` (Euler forward with
///   gravity + gyroscopic acceleration over one dt).
/// - `cone`, `iterations` — world solver config.
///
/// Returns per-body `(force_world, torque_world_at_com)` to be held constant
/// (ZOH) across the RK4 stages. Bodies not touched by any contact receive
/// `(ZERO, ZERO)`.
pub fn solve_free_bodies(
    bodies: &[Body],
    geoms: &[Geom],
    contacts: &[Contact],
    gravity: Vec3,
    dt: f32,
    cone: ConeKind,
    iterations: u32,
) -> Vec<(Vec3, Vec3)> {
    let n_bodies = bodies.len();
    let mut wrenches = vec![(Vec3::ZERO, Vec3::ZERO); n_bodies];
    if contacts.is_empty() || dt <= 0.0 {
        return wrenches;
    }

    // ---- Build constraint rows ---------------------------------------------
    // Contact ordering is already deterministic (caller passes them in
    // narrow-phase order). Each contact contributes 1 normal row and, when
    // condim >= 3, 2 tangent rows in the same fixed order.
    let mut rows: Vec<ConstraintRow> = Vec::new();
    // Per-contact: (start_row, solref, solimp, mu, gap_effective_pen)
    struct PerContact {
        start_row: u32,
        n_rows: u8,
        solref: SolRef,
        solimp: SolImp,
        pen_active: f32,
        v_n_free_start: f32,
        /// Constraint velocity at the START of the step, BEFORE the free-
        /// step (gravity) kick is applied. Used for the a_ref damping term
        /// so the reference trajectory is computed from the true starting
        /// state (matches MuJoCo — a_ref is defined on x, x_dot at the
        /// solve instant, not after a hypothetical Euler push).
        v_n_current: f32,
    }
    let mut per_contact: Vec<PerContact> = Vec::with_capacity(contacts.len());

    // Precompute free (Euler) velocity offset from gravity ONLY. Gyroscopic
    // torque affects angular velocity but is small over dt=5ms in our tests
    // and complicates b assembly; we fold it into ZOH via world.step's later
    // stages. For qdot_free within a single step: dv_free_lin = gravity * dt
    // (per body), dw_free_body = 0 (approx). This matches the "solve once
    // per step with ZOH" scope note in the module docs.
    let dv_lin_free_per_body: Vec<Vec3> = (0..n_bodies).map(|_| gravity * dt).collect();
    let dw_body_free_per_body: Vec<Vec3> = vec![Vec3::ZERO; n_bodies];

    for (ci, c) in contacts.iter().enumerate() {
        let ga = &geoms[c.geom_a];
        let gb = &geoms[c.geom_b];
        // Body indices; None means static.
        let body_a = ga.body.map(|i| i as u32);
        let body_b = gb.body.map(|i| i as u32);
        // Cross-tree contacts should have been filtered by caller; assert.
        assert!(
            ga.link.is_none() && gb.link.is_none(),
            "solve_free_bodies received a link-attached contact; caller must filter"
        );
        // Skip static-static (auto_pairs already drops them, but be safe).
        if body_a.is_none() && body_b.is_none() {
            continue;
        }
        // Gap: no force applied until pen > gap. If pen ≤ gap, still model
        // the contact so the constraint sees it, but with r=0 which gives
        // a_ref=0 — effectively inactive.
        let pen_active = c.penetration - c.gap;
        if pen_active <= 0.0 {
            continue;
        }
        let solref = combine_solref(ga.solref, gb.solref);
        let solimp = combine_solimp(ga.solimp, gb.solimp);
        let mu = crate::contact::combine_friction(ga.friction, gb.friction);
        let condim = ga.condim.min(gb.condim);

        // Deterministic tangent basis from the normal.
        let n_world = c.normal_world;
        let (t1_world, t2_world) = tangent_basis(n_world);

        // Arms: r_arm = contact_point − body_COM (for each side).
        let arm_a = match body_a {
            Some(i) => c.position_world - bodies[i as usize].position,
            None => Vec3::ZERO,
        };
        let arm_b = match body_b {
            Some(i) => c.position_world - bodies[i as usize].position,
            None => Vec3::ZERO,
        };

        let start_row = rows.len() as u32;
        // Compute the current normal velocity of contact point (v_A - v_B) · n
        // and add the free-step delta (gravity impulse). This is J · qdot_free.
        let v_n_start = normal_velocity_at_point(bodies, body_a, body_b, arm_a, arm_b, n_world);
        let dv_n_free = free_step_velocity_contribution(
            &dv_lin_free_per_body,
            &dw_body_free_per_body,
            body_a,
            body_b,
            arm_a,
            arm_b,
            n_world,
            bodies,
        );
        let v_n_free = v_n_start + dv_n_free;

        // Normal row.
        rows.push(ConstraintRow {
            contact_idx: ci as u32,
            kind: 0,
            cone,
            friction: mu,
            dir_world: n_world,
            body_a,
            body_b,
            arm_a,
            arm_b,
            reg: 0.0,  // filled after A_ii computed
            diag: 0.0, // ditto
            bias: 0.0, // ditto
            _pad: 0,
        });

        let mut n_rows_here: u8 = 1;
        if condim >= 3 {
            for &(kind_id, dir) in &[(1u8, t1_world), (2u8, t2_world)] {
                rows.push(ConstraintRow {
                    contact_idx: ci as u32,
                    kind: kind_id,
                    cone,
                    friction: mu,
                    dir_world: dir,
                    body_a,
                    body_b,
                    arm_a,
                    arm_b,
                    reg: 0.0,
                    diag: 0.0,
                    bias: 0.0,
                    _pad: 0,
                });
            }
            n_rows_here += 2;
        }

        per_contact.push(PerContact {
            start_row,
            n_rows: n_rows_here,
            solref,
            solimp,
            pen_active,
            v_n_free_start: v_n_free,
            v_n_current: v_n_start,
        });
    }

    let n_rows = rows.len();
    if n_rows == 0 {
        return wrenches;
    }

    // ---- Compute A_ii and R_ii per row, then A_ij on demand ---------------
    // A_ii from body contributions. Cache I_world^-1 per body.
    let inv_i_world: Vec<crate::math::Mat3> = bodies
        .iter()
        .map(|b| {
            let r = b.orientation.to_mat3();
            r * b.inertia_body_inverse * r.transpose()
        })
        .collect();

    // Fill A_ii, then use it to compute R_ii from d(r) at first-contact
    // violation. Then diag = A_ii + R_ii. All rows for the SAME contact
    // share the same impedance since they use the same normal violation.
    for pc in &per_contact {
        for k in 0..pc.n_rows as usize {
            let ri = pc.start_row as usize + k;
            let a_ii = row_body_diagonal(&rows[ri], bodies, &inv_i_world);
            let d = impedance(pc.pen_active, pc.solimp);
            let reg = if d > 0.0 { (1.0 - d) / d * a_ii } else { 0.0 };
            rows[ri].reg = reg;
            rows[ri].diag = a_ii + reg;
        }
    }

    // ---- Compute bias b_i --------------------------------------------------
    // Normal rows: b = v_n_free + a_ref * dt where r = pen_active,
    // r_dot = -v_n_free_start (r_dot > 0 = closing = worsening). Applying
    // scale factor d (impedance splits): MuJoCo scales the reference by d;
    // we mirror that so the R * f term completes the split.
    for pc in &per_contact {
        let r = pc.pen_active;
        // r_dot in "violation units per second": positive = getting worse.
        // Use v_n at start of step (BEFORE the free-step gravity kick) —
        // the reference trajectory is defined on the constraint's current
        // state, not on a hypothetical post-Euler state (matches MuJoCo).
        let r_dot = -pc.v_n_current;
        let a_ref = reference_accel(r, r_dot, pc.solref);
        let d = impedance(pc.pen_active, pc.solimp);
        // Normal row.
        let n_row = pc.start_row as usize;
        rows[n_row].bias = pc.v_n_free_start + d * a_ref * dt;
        // Tangent rows: reference velocity is 0 (stick). Bias = tangent
        // component of the free-step contact-point velocity.
        for k in 1..pc.n_rows as usize {
            let ri = pc.start_row as usize + k;
            let v_t_start = point_direction_velocity(
                bodies,
                rows[ri].body_a,
                rows[ri].body_b,
                rows[ri].arm_a,
                rows[ri].arm_b,
                rows[ri].dir_world,
            );
            let dv_t_free = free_step_velocity_contribution(
                &dv_lin_free_per_body,
                &dw_body_free_per_body,
                rows[ri].body_a,
                rows[ri].body_b,
                rows[ri].arm_a,
                rows[ri].arm_b,
                rows[ri].dir_world,
                bodies,
            );
            rows[ri].bias = v_t_start + dv_t_free;
        }
    }

    // ---- PGS iteration -----------------------------------------------------
    // Maintain `impulses` (f_i) and `body_delta` (M^-1 J^T f, per body).
    // The residual for row i is:
    //     residual_i = J_i · body_delta_from_all + R_ii * f_i + bias_i
    //                = point_dir_velocity_from_delta(row_i, body_delta) + R_ii f_i + bias_i
    // Update: delta_f = -residual_i / diag_i; project (per row kind).
    let mut impulses = vec![0.0f32; n_rows];
    let mut body_delta = vec![BodyDelta::default(); n_bodies];

    for _iter in 0..iterations {
        // Sweep per contact; within a contact, order = normal then tangents.
        // This lets the tangent projection see the just-updated normal
        // impulse for its cone cap.
        for pc in &per_contact {
            let n_row = pc.start_row as usize;
            // NORMAL update.
            {
                let residual = row_residual(&rows[n_row], &body_delta, bodies, &inv_i_world)
                    + rows[n_row].reg * impulses[n_row]
                    + rows[n_row].bias;
                let mut delta = -residual / rows[n_row].diag;
                let new_f = impulses[n_row] + delta;
                let projected = if new_f < 0.0 { 0.0 } else { new_f };
                delta = projected - impulses[n_row];
                impulses[n_row] = projected;
                apply_impulse_delta(&rows[n_row], delta, &mut body_delta, bodies, &inv_i_world);
            }
            // TANGENT update(s). Cone cap uses just-updated normal impulse.
            if pc.n_rows >= 3 {
                let cap_normal = impulses[n_row];
                match cone {
                    ConeKind::Pyramidal => {
                        for k in 1..pc.n_rows as usize {
                            let ri = pc.start_row as usize + k;
                            let residual =
                                row_residual(&rows[ri], &body_delta, bodies, &inv_i_world)
                                    + rows[ri].reg * impulses[ri]
                                    + rows[ri].bias;
                            let mut delta = -residual / rows[ri].diag;
                            let new_f = impulses[ri] + delta;
                            let cap = rows[ri].friction * cap_normal;
                            let projected = project_pyramidal(new_f, cap);
                            delta = projected - impulses[ri];
                            impulses[ri] = projected;
                            apply_impulse_delta(
                                &rows[ri],
                                delta,
                                &mut body_delta,
                                bodies,
                                &inv_i_world,
                            );
                        }
                    }
                    ConeKind::Elliptic => {
                        // Do a joint Gauss-Seidel on both tangents, then
                        // project the pair onto the elliptic cone.
                        let ri1 = pc.start_row as usize + 1;
                        let ri2 = pc.start_row as usize + 2;
                        let residual1 = row_residual(&rows[ri1], &body_delta, bodies, &inv_i_world)
                            + rows[ri1].reg * impulses[ri1]
                            + rows[ri1].bias;
                        let residual2 = row_residual(&rows[ri2], &body_delta, bodies, &inv_i_world)
                            + rows[ri2].reg * impulses[ri2]
                            + rows[ri2].bias;
                        let new_f1 = impulses[ri1] - residual1 / rows[ri1].diag;
                        let new_f2 = impulses[ri2] - residual2 / rows[ri2].diag;
                        let (proj1, proj2) =
                            project_elliptic(new_f1, new_f2, rows[ri1].friction, cap_normal);
                        let delta1 = proj1 - impulses[ri1];
                        let delta2 = proj2 - impulses[ri2];
                        impulses[ri1] = proj1;
                        impulses[ri2] = proj2;
                        apply_impulse_delta(
                            &rows[ri1],
                            delta1,
                            &mut body_delta,
                            bodies,
                            &inv_i_world,
                        );
                        apply_impulse_delta(
                            &rows[ri2],
                            delta2,
                            &mut body_delta,
                            bodies,
                            &inv_i_world,
                        );
                    }
                }
            }
        }
    }

    // ---- Extract per-body wrenches -----------------------------------------
    // The impulse J^T f, distributed per body, is a force (through the
    // contact point) and a torque (r × F) at each body's COM. Convert
    // impulse to force by dividing by dt (ZOH: constant over the step).
    for (ri, row) in rows.iter().enumerate() {
        let f_impulse = impulses[ri];
        if f_impulse == 0.0 {
            continue;
        }
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

    wrenches
}

/// Compute `A_ii = J_i M^-1 J_i^T` — the rigid-body diagonal contribution
/// for one constraint row, summed over the row's two bodies. Static geoms
/// contribute zero (infinite mass).
fn row_body_diagonal(
    row: &ConstraintRow,
    bodies: &[Body],
    inv_i_world: &[crate::math::Mat3],
) -> f32 {
    let mut a_ii = 0.0f32;
    if let Some(i) = row.body_a {
        let i = i as usize;
        let r_cross_dir = row.arm_a.cross(row.dir_world);
        // (r × dir)^T I_world^-1 (r × dir) + (1/m)
        a_ii += r_cross_dir.dot(inv_i_world[i] * r_cross_dir) + 1.0 / bodies[i].mass;
    }
    if let Some(i) = row.body_b {
        let i = i as usize;
        let r_cross_dir = row.arm_b.cross(row.dir_world);
        // Same formula — sign of dir flips for B, but (-dir)·M^-1·(-dir) = dir·M^-1·dir.
        a_ii += r_cross_dir.dot(inv_i_world[i] * r_cross_dir) + 1.0 / bodies[i].mass;
    }
    a_ii
}

/// Compute `J_i · body_delta` for one row — the "how much has the contact-
/// point velocity along dir changed due to accumulated impulses on the two
/// bodies?" contribution.
fn row_residual(
    row: &ConstraintRow,
    body_delta: &[BodyDelta],
    bodies: &[Body],
    _inv_i_world: &[crate::math::Mat3],
) -> f32 {
    let mut s = 0.0f32;
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
        // B contribution is subtracted (v_rel = v_A - v_B).
        s -= v_point.dot(row.dir_world);
    }
    s
}

/// Apply an impulse delta on row `row` to the two bodies' accumulators.
/// Uses J^T = [(r × dir)_world (as torque, then rotated to body); dir_world
/// (as linear)] with per-body factors `M^-1`.
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

/// One active joint-limit constraint.
#[derive(Clone, Copy, Debug)]
struct LimitRow {
    /// DOF slot in the tree's `nv` vector.
    v_slot: u32,
    /// +1 for low-side violation (q < lo, escape = +q), -1 for high-side
    /// (q > hi, escape = -q).
    sign: f32,
    /// Violation magnitude `r > 0` (rad or m).
    violation: f32,
    /// SolRef / SolImp (reused for limits; documented default is fine).
    solref: SolRef,
    solimp: SolImp,
}

/// Solve per-tree joint-limit constraints in PGS. Returns a per-DOF
/// generalized force delta (`nv`-length) that the caller should add to
/// `tree.qfrc_applied` for the step (ZOH under RK4).
///
/// Scope this ticket: hinge + slide range limits only (ball deferred as
/// stated in the ticket scope). We build the tree's mass matrix `M`,
/// factor it once via Cholesky, then solve `M^-1 · e_j` per limit to get
/// each limit's velocity-response direction. `A` is dense `n_limits ×
/// n_limits` with `A_ij = e_i^T · M^-1 · e_j` (a submatrix of `M^-1`).
///
/// Zero-length return `vec![0.0; nv]` when there are no active limits.
pub fn solve_tree_limits(tree: &Tree, dt: f32, iterations: u32) -> Vec<f32> {
    let nv = tree.nv();
    let mut qfrc = vec![0.0f32; nv];
    if nv == 0 || dt <= 0.0 {
        return qfrc;
    }

    // Enumerate active hinge/slide limits.
    let mut limits: Vec<LimitRow> = Vec::new();
    for (li, link) in tree.links.iter().enumerate() {
        let (range, is_single_dof) = match link.joint {
            JointKind::Hinge { range, .. } | JointKind::Slide { range, .. } => (range, true),
            _ => (None, false),
        };
        if !is_single_dof {
            continue;
        }
        let Some((lo, hi)) = range else { continue };
        let v_slot = tree.v_offset[li] as u32;
        let q = tree.q[tree.q_offset[li]];
        if q < lo {
            limits.push(LimitRow {
                v_slot,
                sign: 1.0,
                violation: lo - q,
                solref: SolRef::DEFAULT,
                solimp: SolImp::DEFAULT,
            });
        } else if q > hi {
            limits.push(LimitRow {
                v_slot,
                sign: -1.0,
                violation: q - hi,
                solref: SolRef::DEFAULT,
                solimp: SolImp::DEFAULT,
            });
        }
    }
    if limits.is_empty() {
        return qfrc;
    }

    // Build M(q), Cholesky. If M is not PD (shouldn't happen for a valid
    // tree), fall back to no impulse — the caller keeps the penalty
    // pathway alive (documented).
    let m = mass_matrix(tree);
    let Some(l) = cholesky(&m, nv) else {
        return qfrc;
    };

    // Precompute d_i = M^-1 · e_signed_i for each limit — the velocity
    // change per unit generalized force at that limit.
    let n_limits = limits.len();
    let mut d_vecs: Vec<Vec<f32>> = Vec::with_capacity(n_limits);
    for lim in &limits {
        let mut e = vec![0.0f32; nv];
        e[lim.v_slot as usize] = lim.sign;
        d_vecs.push(cholesky_solve(&l, nv, &e));
    }

    // A matrix (dense n_limits × n_limits). A_ij = e_i^T · d_j = sign_i *
    // d_j[v_slot_i].
    let mut a = vec![0.0f32; n_limits * n_limits];
    for i in 0..n_limits {
        for j in 0..n_limits {
            a[i * n_limits + j] = limits[i].sign * d_vecs[j][limits[i].v_slot as usize];
        }
    }

    // Bias b_i. For a limit constraint the "constraint velocity" is
    // sign · qdot[v_slot]. r_dot = -constraint_velocity (violation
    // increasing = closing on the boundary from OUTSIDE = wait,
    // convention flip: r = violation depth from the limit, r > 0. If we
    // define escape as toward the acceptance set, r_dot = -escape_vel.
    // For q < lo: escape = +q direction, escape_vel = +qdot; r = lo - q,
    // r_dot = -qdot = -escape_vel.  For q > hi: escape = -q direction,
    // escape_vel = -qdot; r = q - hi, r_dot = qdot = -escape_vel. Both
    // handled by `sign * qdot` = escape_vel.
    let mut bias = vec![0.0f32; n_limits];
    for (i, lim) in limits.iter().enumerate() {
        let escape_vel = lim.sign * tree.qdot[lim.v_slot as usize];
        // r_dot at start of step (before any impulse) = -escape_vel.
        let r_dot = -escape_vel;
        let a_ref = reference_accel(lim.violation, r_dot, lim.solref);
        let d = impedance(lim.violation, lim.solimp);
        // v_free approximation for the limit: uses just start-of-step
        // escape velocity — non-constraint generalized forces (gravity,
        // etc.) get integrated via ABA over the RK4 step; the solver's
        // reference term dominates for near-boundary cases anyway.
        bias[i] = escape_vel + d * a_ref * dt;
    }

    // Regularization R_i = ((1 - d)/d) · A_ii per limit.
    let mut diag = vec![0.0f32; n_limits];
    for i in 0..n_limits {
        let a_ii = a[i * n_limits + i];
        let d = impedance(limits[i].violation, limits[i].solimp);
        let r = if d > 0.0 { (1.0 - d) / d * a_ii } else { 0.0 };
        diag[i] = a_ii + r;
    }

    // PGS: fixed iteration count (from `world.solver.iterations`), sweep
    // in ascending limit index.
    let mut f = vec![0.0f32; n_limits];
    for _iter in 0..iterations {
        for i in 0..n_limits {
            // residual = Σ_j A_ij f_j + R_ii f_i + bias_i.
            let mut r_i = bias[i];
            for j in 0..n_limits {
                r_i += a[i * n_limits + j] * f[j];
            }
            // A_ii f_i is already included; the "R" contribution equals
            // (diag - A_ii) f_i.
            let extra_reg = (diag[i] - a[i * n_limits + i]) * f[i];
            r_i += extra_reg;
            let delta = -r_i / diag[i];
            let new_f = f[i] + delta;
            // Non-negative projection.
            let proj = if new_f < 0.0 { 0.0 } else { new_f };
            f[i] = proj;
        }
    }

    // Accumulate impulses into qfrc slots (as force = impulse/dt).
    for (i, lim) in limits.iter().enumerate() {
        qfrc[lim.v_slot as usize] += lim.sign * f[i] / dt;
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
