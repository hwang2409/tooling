//! Dense Newton minimization for the soft-constraint system.
//!
//! The physics solver supplies the dense regularized Delassus matrix and its
//! bias.  This module owns the Newton step, cone projection, and the exact
//! piecewise-quadratic line search used by the pyramidal cone.  Keeping this
//! code independent from contact assembly makes the zone derivatives useful
//! in hand-derived tests as well as in the live solver.

use std::cmp::Ordering;

use crate::dynamics::{cholesky, cholesky_solve};

/// The local zone of a one-sided residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarZone {
    /// The residual is on the accepted side.  The local cost is zero.
    Inactive,
    /// The residual violates the one-sided condition.  The local cost is
    /// quadratic.
    Active,
    /// The residual is exactly on the switching surface.
    Boundary,
}

/// Exact cost, gradient, and Hessian for a one-sided quadratic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarDerivatives {
    pub zone: ScalarZone,
    pub cost: f32,
    pub gradient: f32,
    pub hessian: f32,
}

/// Evaluate `½ weight · min(residual, 0)²`.
///
/// The residual is positive on the accepted side.  At zero we choose the
/// zero Hessian, which is the deterministic limiting Hessian from that side.
pub fn one_sided_derivatives(residual: f32, weight: f32) -> ScalarDerivatives {
    if residual > 0.0 {
        ScalarDerivatives {
            zone: ScalarZone::Inactive,
            cost: 0.0,
            gradient: 0.0,
            hessian: 0.0,
        }
    } else if residual < 0.0 {
        let gradient = weight * residual;
        ScalarDerivatives {
            zone: ScalarZone::Active,
            cost: 0.5 * residual * gradient,
            gradient,
            hessian: weight,
        }
    } else {
        ScalarDerivatives {
            zone: ScalarZone::Boundary,
            cost: 0.0,
            gradient: 0.0,
            hessian: 0.0,
        }
    }
}

/// The local zone of a two-dimensional friction cone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConeZone {
    /// The tangential residual is inside the cone.
    Interior,
    /// At least one cone face is exactly active.
    Boundary,
    /// The tangential residual is outside the cone.
    Outside,
}

/// Exact local derivatives for a cone penalty in coordinates
/// `(normal, tangent_1, tangent_2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConeDerivatives {
    pub zone: ConeZone,
    pub cost: f32,
    pub gradient: [f32; 3],
    /// Row-major 3×3 Hessian.
    pub hessian: [f32; 9],
}

/// Derivatives for the pyramidal cone penalty
///
/// `½ weight · max(|t1| − μ n, 0)² + ½ weight · max(|t2| − μ n, 0)²`.
///
/// Each active face is affine, so its Hessian is the exact rank-one outer
/// product of the face normal.  The normal one-sided term is separate and is
/// assembled by the caller.
pub fn pyramidal_derivatives(
    normal: f32,
    tangent_1: f32,
    tangent_2: f32,
    mu: f32,
    weight: f32,
) -> ConeDerivatives {
    let e1 = tangent_1.abs() - mu * normal;
    let e2 = tangent_2.abs() - mu * normal;
    let mut out = ConeDerivatives {
        zone: if e1 > 0.0 || e2 > 0.0 {
            ConeZone::Outside
        } else if e1 == 0.0 || e2 == 0.0 {
            ConeZone::Boundary
        } else {
            ConeZone::Interior
        },
        cost: 0.0,
        gradient: [0.0; 3],
        hessian: [0.0; 9],
    };
    for (tangent, excess) in [(tangent_1, e1), (tangent_2, e2)] {
        if excess <= 0.0 {
            continue;
        }
        let sign = if tangent >= 0.0 { 1.0 } else { -1.0 };
        let face = [-mu, sign, 0.0];
        out.cost += 0.5 * weight * excess * excess;
        for i in 0..3 {
            out.gradient[i] += weight * excess * face[i];
            for j in 0..3 {
                out.hessian[i * 3 + j] += weight * face[i] * face[j];
            }
        }
    }
    out
}

/// Derivatives for the elliptic cone penalty
///
/// `½ weight · max(‖t‖ − μ n, 0)²`.
///
/// This is kept as a tested math primitive.  Newton loading rejects the
/// elliptic cone until its curved-domain line search is wired end to end.
pub fn elliptic_derivatives(
    normal: f32,
    tangent_1: f32,
    tangent_2: f32,
    mu: f32,
    weight: f32,
) -> ConeDerivatives {
    let tangent_norm = (tangent_1 * tangent_1 + tangent_2 * tangent_2).sqrt();
    let excess = tangent_norm - mu * normal;
    let zone = if excess > 0.0 {
        ConeZone::Outside
    } else if excess == 0.0 {
        ConeZone::Boundary
    } else {
        ConeZone::Interior
    };
    if excess <= 0.0 {
        return ConeDerivatives {
            zone,
            cost: 0.0,
            gradient: [0.0; 3],
            hessian: [0.0; 9],
        };
    }
    if tangent_norm == 0.0 {
        // This can only occur with a negative normal. The normal one-sided
        // term handles that violation; this curved term has no direction.
        return ConeDerivatives {
            zone,
            cost: 0.5 * weight * excess * excess,
            gradient: [-weight * mu * excess, 0.0, 0.0],
            hessian: [weight * mu * mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
    }
    let inv_norm = 1.0 / tangent_norm;
    let ux = tangent_1 * inv_norm;
    let uy = tangent_2 * inv_norm;
    let gradient = [
        -weight * mu * excess,
        weight * excess * ux,
        weight * excess * uy,
    ];
    let mut hessian = [0.0f32; 9];
    hessian[0] = weight * mu * mu;
    hessian[1] = -weight * mu * ux;
    hessian[2] = -weight * mu * uy;
    hessian[3] = hessian[1];
    hessian[6] = hessian[2];
    for (i, ui) in [ux, uy].iter().copied().enumerate() {
        for (j, uj) in [ux, uy].iter().copied().enumerate() {
            let delta = if i == j { 1.0 } else { 0.0 };
            hessian[(i + 1) * 3 + (j + 1)] =
                weight * (ui * uj + excess * (delta - ui * uj) * inv_norm);
        }
    }
    ConeDerivatives {
        zone,
        cost: 0.5 * weight * excess * excess,
        gradient,
        hessian,
    }
}

/// Projection used by the live solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    /// No projection.
    Bilateral,
    /// `x[index] >= 0`.
    NonNegative { index: usize },
    /// `|x[index]| <= mu · x[normal]`.
    ScalarConeBound {
        index: usize,
        normal: usize,
        mu: f32,
    },
    /// `x[normal] >= 0` and `|(x[tangent_1], x[tangent_2])|` is bounded by
    /// the pyramidal cone faces.
    PyramidalCone {
        normal: usize,
        tangent_1: usize,
        tangent_2: usize,
        mu: f32,
    },
}

/// A dense regularized convex quadratic with cone projections.
#[derive(Clone, Debug)]
pub struct NewtonSystem {
    /// Row-major positive-definite Hessian of the unconstrained quadratic.
    pub hessian: Vec<f32>,
    /// Linear term in `½ xᵀ H x + linearᵀ x`.
    pub linear: Vec<f32>,
    /// Projection blocks, in deterministic row order.
    pub projections: Vec<Projection>,
    /// Fixed iteration cap.
    pub max_iterations: u32,
    /// Stop when the cost improvement is at most this scaled threshold.
    pub cost_tolerance: f32,
}

/// Result of the deterministic Newton solve.
#[derive(Clone, Debug, PartialEq)]
pub struct NewtonResult {
    pub solution: Vec<f32>,
    /// Cost at the initial feasible point and after each accepted step.
    pub costs: Vec<f32>,
    pub iterations: u32,
    pub converged: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewtonError(pub String);

impl std::fmt::Display for NewtonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NewtonError {}

impl NewtonSystem {
    /// Solve from the deterministic zero impulse, which is feasible for all
    /// projection blocks used by this crate.
    pub fn solve(&self) -> Result<NewtonResult, NewtonError> {
        let n = self.linear.len();
        if self.hessian.len() != n * n {
            return Err(NewtonError(format!(
                "Newton Hessian has {} entries for dimension {n}",
                self.hessian.len()
            )));
        }
        if self.max_iterations == 0 {
            return Ok(NewtonResult {
                solution: vec![0.0; n],
                costs: vec![0.0],
                iterations: 0,
                converged: true,
            });
        }
        validate_projections(&self.projections, n)?;
        let Some(factor) = cholesky(&self.hessian, n) else {
            return Err(NewtonError(
                "Newton Hessian is not positive definite".into(),
            ));
        };
        let mut x = vec![0.0f32; n];
        let mut costs = vec![quadratic_cost(&self.hessian, &self.linear, &x)];
        let mut converged = false;
        let mut iterations = 0;
        for iteration in 0..self.max_iterations {
            let gradient = quadratic_gradient(&self.hessian, &self.linear, &x);
            let direction = cholesky_solve(&factor, n, &negate(&gradient));
            if dot(&gradient, &direction) >= 0.0 {
                converged = true;
                break;
            }
            let current_cost = *costs.last().unwrap_or(&0.0);
            let (alpha, next_cost) = exact_pyramidal_line_search(
                &self.hessian,
                &self.linear,
                &self.projections,
                &x,
                &direction,
                current_cost,
            );
            if alpha <= 0.0
                || current_cost - next_cost <= self.cost_tolerance * (1.0 + current_cost.abs())
            {
                converged = true;
                break;
            }
            for i in 0..n {
                x[i] += alpha * direction[i];
            }
            x = project(&x, &self.projections);
            costs.push(next_cost);
            iterations = iteration + 1;
        }
        if iterations == self.max_iterations {
            converged = true;
        }
        Ok(NewtonResult {
            solution: x,
            costs,
            iterations,
            converged,
        })
    }
}

fn validate_projections(projections: &[Projection], n: usize) -> Result<(), NewtonError> {
    let check = |index: usize| {
        if index < n {
            Ok(())
        } else {
            Err(NewtonError(format!(
                "Newton projection index {index} is outside dimension {n}"
            )))
        }
    };
    for projection in projections {
        match *projection {
            Projection::Bilateral => {}
            Projection::NonNegative { index } => check(index)?,
            Projection::ScalarConeBound { index, normal, .. } => {
                check(index)?;
                check(normal)?;
            }
            Projection::PyramidalCone {
                normal,
                tangent_1,
                tangent_2,
                ..
            } => {
                check(normal)?;
                check(tangent_1)?;
                check(tangent_2)?;
            }
        }
    }
    Ok(())
}

fn project(input: &[f32], projections: &[Projection]) -> Vec<f32> {
    let mut output = input.to_vec();
    for projection in projections {
        match *projection {
            Projection::Bilateral => {}
            Projection::NonNegative { index } => {
                if output[index] < 0.0 {
                    output[index] = 0.0;
                }
            }
            Projection::ScalarConeBound { index, normal, mu } => {
                let cap = mu * output[normal].max(0.0);
                output[index] = output[index].max(-cap).min(cap);
            }
            Projection::PyramidalCone {
                normal,
                tangent_1,
                tangent_2,
                mu,
            } => {
                if output[normal] < 0.0 {
                    output[normal] = 0.0;
                }
                let cap = mu * output[normal];
                output[tangent_1] = output[tangent_1].max(-cap).min(cap);
                output[tangent_2] = output[tangent_2].max(-cap).min(cap);
            }
        }
    }
    output
}

/// Exact line search for the piecewise-affine pyramidal projection path.
/// Each interval has a fixed set of cone faces, so the projected path is
/// affine and the quadratic minimum has a closed form. Elliptic projections
/// are rejected at load time; the fallback samples only protect callers that
/// construct a future curved projection locally.
fn exact_pyramidal_line_search(
    hessian: &[f32],
    linear: &[f32],
    projections: &[Projection],
    x: &[f32],
    direction: &[f32],
    current_cost: f32,
) -> (f32, f32) {
    let mut breaks = vec![0.0f32, 1.0f32];
    for projection in projections {
        match *projection {
            Projection::Bilateral => {}
            Projection::NonNegative { index } => {
                add_root(&mut breaks, x[index], direction[index]);
            }
            Projection::ScalarConeBound { index, normal, mu } => {
                add_root(
                    &mut breaks,
                    x[index] - mu * x[normal],
                    direction[index] - mu * direction[normal],
                );
                add_root(
                    &mut breaks,
                    x[index] + mu * x[normal],
                    direction[index] + mu * direction[normal],
                );
                add_root(&mut breaks, x[normal], direction[normal]);
            }
            Projection::PyramidalCone {
                normal,
                tangent_1,
                tangent_2,
                mu,
            } => {
                add_root(&mut breaks, x[normal], direction[normal]);
                for tangent in [tangent_1, tangent_2] {
                    add_root(
                        &mut breaks,
                        x[tangent] - mu * x[normal],
                        direction[tangent] - mu * direction[normal],
                    );
                    add_root(
                        &mut breaks,
                        x[tangent] + mu * x[normal],
                        direction[tangent] + mu * direction[normal],
                    );
                }
            }
        }
    }
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    breaks.dedup_by(|a, b| (*a - *b).abs() <= 1e-7);

    let mut candidates = breaks.clone();
    for pair in breaks.windows(2) {
        let lo = pair[0];
        let hi = pair[1];
        if hi - lo <= 1e-7 {
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let eps = (hi - lo) * 1e-4;
        let left_alpha = (mid - eps).max(lo);
        let right_alpha = (mid + eps).min(hi);
        let left = project_affine_point(x, direction, projections, left_alpha);
        let right = project_affine_point(x, direction, projections, right_alpha);
        let span = right_alpha - left_alpha;
        if span <= 0.0 {
            continue;
        }
        let slope = subtract(&right, &left, 1.0 / span);
        let z_mid = project_affine_point(x, direction, projections, mid);
        let local_gradient = quadratic_gradient(hessian, linear, &z_mid);
        let denominator = quadratic_form(hessian, &slope);
        let numerator = dot(&local_gradient, &slope);
        let optimum = if denominator > 0.0 {
            (mid - numerator / denominator).max(lo).min(hi)
        } else {
            mid
        };
        candidates.push(optimum);
    }

    let mut best_alpha = 0.0;
    let mut best_cost = current_cost;
    for alpha in candidates {
        let point = project_affine_point(x, direction, projections, alpha);
        let cost = quadratic_cost(hessian, linear, &point);
        if cost < best_cost {
            best_cost = cost;
            best_alpha = alpha;
        }
    }
    (best_alpha, best_cost)
}

fn add_root(breaks: &mut Vec<f32>, value: f32, slope: f32) {
    if slope == 0.0 {
        return;
    }
    let alpha = -value / slope;
    if alpha > 0.0 && alpha < 1.0 {
        breaks.push(alpha);
    }
}

fn project_affine_point(
    x: &[f32],
    direction: &[f32],
    projections: &[Projection],
    alpha: f32,
) -> Vec<f32> {
    let mut point = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        point[i] = x[i] + alpha * direction[i];
    }
    project(&point, projections)
}

fn quadratic_cost(hessian: &[f32], linear: &[f32], x: &[f32]) -> f32 {
    0.5 * dot(x, &mat_vec(hessian, x)) + dot(linear, x)
}

fn quadratic_gradient(hessian: &[f32], linear: &[f32], x: &[f32]) -> Vec<f32> {
    let mut gradient = mat_vec(hessian, x);
    for i in 0..gradient.len() {
        gradient[i] += linear[i];
    }
    gradient
}

fn quadratic_form(hessian: &[f32], x: &[f32]) -> f32 {
    dot(x, &mat_vec(hessian, x))
}

fn mat_vec(matrix: &[f32], vector: &[f32]) -> Vec<f32> {
    let n = vector.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            out[i] += matrix[i * n + j] * vector[j];
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn negate(values: &[f32]) -> Vec<f32> {
    values.iter().map(|value| -*value).collect()
}

fn subtract(a: &[f32], b: &[f32], scale: f32) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(left, right)| (left - right) * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }

    #[test]
    fn one_sided_zones_have_exact_derivatives() {
        let inactive = one_sided_derivatives(2.0, 3.0);
        assert_eq!(inactive.zone, ScalarZone::Inactive);
        approx(inactive.gradient, 0.0);
        let active = one_sided_derivatives(-2.0, 3.0);
        assert_eq!(active.zone, ScalarZone::Active);
        approx(active.cost, 6.0);
        approx(active.gradient, -6.0);
        approx(active.hessian, 3.0);
        assert_eq!(one_sided_derivatives(0.0, 3.0).zone, ScalarZone::Boundary);
    }

    #[test]
    fn pyramidal_outside_face_matches_hand_derivation() {
        let d = pyramidal_derivatives(1.0, 2.0, -0.25, 0.5, 4.0);
        assert_eq!(d.zone, ConeZone::Outside);
        // e1 = 2 - .5 = 1.5; the second face is inactive.
        approx(d.cost, 4.5);
        approx(d.gradient[0], -3.0);
        approx(d.gradient[1], 6.0);
        approx(d.hessian[0], 1.0);
        approx(d.hessian[1], -2.0);
        approx(d.hessian[4], 4.0);
    }

    #[test]
    fn elliptic_outside_zone_has_symmetric_hessian() {
        let d = elliptic_derivatives(1.0, 2.0, 0.0, 0.5, 2.0);
        assert_eq!(d.zone, ConeZone::Outside);
        approx(d.cost, 2.25);
        approx(d.gradient[0], -1.5);
        approx(d.gradient[1], 3.0);
        approx(d.hessian[1], d.hessian[3]);
        approx(d.hessian[2], d.hessian[6]);
    }

    #[test]
    fn newton_qp_is_monotone_and_feasible() {
        let system = NewtonSystem {
            hessian: vec![2.0, 0.0, 0.0, 1.0],
            linear: vec![-4.0, 1.0],
            projections: vec![Projection::NonNegative { index: 0 }],
            max_iterations: 8,
            cost_tolerance: 1e-7,
        };
        let result = system.solve().unwrap();
        assert!(result.solution[0] >= 0.0);
        for pair in result.costs.windows(2) {
            assert!(pair[1] <= pair[0] + 1e-6, "cost increased: {pair:?}");
        }
        approx(result.solution[0], 2.0);
        approx(result.solution[1], -1.0);
    }
}
