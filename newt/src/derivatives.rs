use super::*;

/// `qacc_q` is `nv × nv` in generalized tangent coordinates,
/// `qacc_qvel` is `nv × nv`, and `qacc_ctrl` is
/// `nv × nu`, where `nu` is `tree.actuators.len()`.
///
/// The velocity and control paths differentiate the rigid-body bias and
/// actuator transmission directly. Position derivatives differentiate the
/// smooth ABA recursion. Quaternion columns use right-multiplied body-frame
/// tangent rotations.
/// `external_wrenches` is held fixed during this evaluation. A world-level
/// contact derivative must rebuild its contact wrenches for each perturbation.
#[derive(Clone, Debug, PartialEq)]
pub struct Derivatives {
    /// Forward acceleration at the evaluated state.
    pub qacc: Vec<f32>,
    /// `∂qacc/∂q`, row-major `nv × nv` in tangent coordinates.
    pub qacc_q: Vec<f32>,
    /// `∂qacc/∂qvel`, row-major `nv × nv`.
    pub qacc_qvel: Vec<f32>,
    /// `∂qacc/∂ctrl`, row-major `nv × nu`.
    pub qacc_ctrl: Vec<f32>,
}

impl Derivatives {
    /// Number of generalized acceleration rows.
    pub fn nv(&self) -> usize {
        self.qacc.len()
    }

    /// Number of tangent-position columns.
    pub fn nq(&self) -> usize {
        self.qacc_q.len() / self.nv().max(1)
    }

    /// Number of actuator control columns.
    pub fn nu(&self) -> usize {
        self.qacc_ctrl.len() / self.nv().max(1)
    }
}

/// Compute dense explicit forward-dynamics derivatives for one tree.
pub fn derivatives(
    tree: &Tree,
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
) -> Derivatives {
    assert_eq!(
        external_wrenches.len(),
        tree.links.len(),
        "external_wrenches length must equal number of links"
    );
    let poses = forward_kinematics(tree);
    let qacc = crate::tree::aba(tree, &poses, gravity, external_wrenches);
    let nv = tree.nv();

    let mass = mass_matrix(tree);
    let qacc_qvel_force = qacc_qvel_force_jacobian(tree, &poses);
    let qacc_qvel = solve_mass_columns(&mass, nv, &qacc_qvel_force);
    let qacc_ctrl_force = qacc_ctrl_force_jacobian(tree, &poses);
    let qacc_ctrl = solve_mass_columns(&mass, nv, &qacc_ctrl_force);
    let qacc_q = analytic_qacc_q(tree, gravity, external_wrenches);

    Derivatives {
        qacc,
        qacc_q,
        qacc_qvel,
        qacc_ctrl,
    }
}

/// Differentiate the position path in each generalized tangent coordinate.
/// The forward-mode ABA path also differentiates spatial tendon geometry.
fn analytic_qacc_q(tree: &Tree, gravity: Vec3, external_wrenches: &ExternalWrenches) -> Vec<f32> {
    let nv = tree.nv();
    let mut out = vec![0.0; nv * nv];
    for column in 0..nv {
        let derivative = analytic_qacc_column(tree, gravity, external_wrenches, column);
        for row in 0..nv {
            out[row * nv + column] = derivative[row];
        }
    }
    out
}

/// Central-difference fallback for a caller that owns contact detection and
/// constraint solving. The callback runs for every perturbed tree, so contact
/// forces and friction are not held fixed across samples.
pub fn constrained_derivatives<F>(tree: &Tree, gravity: Vec3, external_wrenches: F) -> Derivatives
where
    F: Fn(&Tree) -> ExternalWrenches,
{
    let base_wrenches = external_wrenches(tree);
    let qacc = crate::tree::aba(tree, &forward_kinematics(tree), gravity, &base_wrenches);
    let nv = tree.nv();
    let nu = tree.actuators.len();
    let step = 1.0e-4;
    let mut qacc_q = vec![0.0; nv * nv];
    let mut qacc_qvel = vec![0.0; nv * nv];
    let mut qacc_ctrl = vec![0.0; nv * nu];
    for column in 0..nv {
        let mut plus = tree.clone();
        let mut minus = tree.clone();
        perturb_position_tangent(&mut plus, column, step);
        perturb_position_tangent(&mut minus, column, -step);
        let plus_acc = constrained_acceleration(&plus, gravity, &external_wrenches);
        let minus_acc = constrained_acceleration(&minus, gravity, &external_wrenches);
        for row in 0..nv {
            qacc_q[row * nv + column] = (plus_acc[row] - minus_acc[row]) / (2.0 * step);
        }

        let mut plus = tree.clone();
        let mut minus = tree.clone();
        plus.qdot[column] += step;
        minus.qdot[column] -= step;
        let plus_acc = constrained_acceleration(&plus, gravity, &external_wrenches);
        let minus_acc = constrained_acceleration(&minus, gravity, &external_wrenches);
        for row in 0..nv {
            qacc_qvel[row * nv + column] = (plus_acc[row] - minus_acc[row]) / (2.0 * step);
        }
    }
    for column in 0..nu {
        let mut plus = tree.clone();
        let mut minus = tree.clone();
        plus.actuators[column].ctrl += step;
        minus.actuators[column].ctrl -= step;
        let plus_acc = constrained_acceleration(&plus, gravity, &external_wrenches);
        let minus_acc = constrained_acceleration(&minus, gravity, &external_wrenches);
        for row in 0..nv {
            qacc_ctrl[row * nu + column] = (plus_acc[row] - minus_acc[row]) / (2.0 * step);
        }
    }
    Derivatives {
        qacc,
        qacc_q,
        qacc_qvel,
        qacc_ctrl,
    }
}

fn constrained_acceleration<F>(tree: &Tree, gravity: Vec3, external_wrenches: &F) -> Vec<f32>
where
    F: Fn(&Tree) -> ExternalWrenches,
{
    let poses = forward_kinematics(tree);
    let wrenches = external_wrenches(tree);
    crate::tree::aba(tree, &poses, gravity, &wrenches)
}

fn analytic_qacc_column(
    tree: &Tree,
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
    column: usize,
) -> Vec<f32> {
    let n = tree.links.len();
    let nv = tree.nv();
    let mut q: Vec<crate::aba::Dual> = tree
        .q
        .iter()
        .copied()
        .map(|value| crate::aba::Dual {
            value,
            derivative: 0.0,
        })
        .collect();
    for (i, link) in tree.links.iter().enumerate() {
        let voff = tree.v_offset[i];
        let qoff = tree.q_offset[i];
        match link.joint {
            JointKind::Free => {
                if (voff..voff + 3).contains(&column) {
                    q[qoff + column - voff].derivative = 1.0;
                } else if (voff + 3..voff + 6).contains(&column) {
                    let axis = basis_axis(column - voff - 3);
                    dualize_quaternion_right_tangent(&mut q[qoff + 3..qoff + 7], axis);
                }
            }
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                if column == voff {
                    q[qoff].derivative = 1.0;
                }
            }
            JointKind::Ball { .. } => {
                if (voff..voff + 3).contains(&column) {
                    dualize_quaternion_right_tangent(
                        &mut q[qoff..qoff + 4],
                        basis_axis(column - voff),
                    );
                }
            }
            JointKind::Fixed => {}
        }
    }
    let qdot: Vec<crate::aba::Dual> = tree
        .qdot
        .iter()
        .copied()
        .map(|value| crate::aba::Dual {
            value,
            derivative: 0.0,
        })
        .collect();
    let orientations = crate::aba::orientations(tree, &q);
    let external_body: Vec<crate::aba::GForce<crate::aba::Dual>> = tree
        .links
        .iter()
        .enumerate()
        .map(|(i, link)| {
            let (force_world, torque_world) = external_wrenches[i];
            let (force_applied, torque_applied) = tree.applied_wrenches[i];
            let world_force =
                crate::aba::GVec3::from_vec3(force_world + force_applied + gravity * link.mass);
            let world_torque = crate::aba::GVec3::from_vec3(torque_world + torque_applied);
            crate::aba::GForce {
                torque: orientations[i].transpose_mul_vec(world_torque),
                linear: orientations[i].transpose_mul_vec(world_force),
            }
        })
        .collect();
    let (tendon_value, tendon_derivative) = tendon_force_derivative(tree, column);
    let mut scalar_value = vec![0.0; n];
    let mut ball_value = vec![Vec3::ZERO; n];
    let free_value = crate::forces::assemble_joint_forces(
        tree,
        &tendon_value,
        &mut scalar_value,
        &mut ball_value,
    );
    let mut scalar = vec![crate::aba::Dual::zero(); n];
    let mut ball = vec![crate::aba::GVec3::zero(); n];
    for (i, link) in tree.links.iter().enumerate() {
        match link.joint {
            JointKind::Hinge { range, limit, .. } | JointKind::Slide { range, limit, .. } => {
                let qoff = tree.q_offset[i];
                let voff = tree.v_offset[i];
                let dq = q[qoff].derivative;
                let actuator_derivative: f32 = tree
                    .actuators
                    .iter()
                    .filter(|act| act.tendon_target.is_none() && act.link_idx == i)
                    .map(|act| act.position_derivative(tree.q[qoff], tree.qdot[voff]))
                    .sum();
                scalar[i] = crate::aba::Dual {
                    value: scalar_value[i],
                    derivative: tendon_derivative[voff]
                        + (qacc_limit_position_derivative(
                            tree.q[qoff],
                            range,
                            limit,
                            tree.disable_penalty_limits,
                        ) + actuator_derivative)
                            * dq,
                };
            }
            JointKind::Ball { .. } => {
                let off = tree.v_offset[i];
                ball[i] = crate::aba::GVec3 {
                    x: crate::aba::Dual {
                        value: ball_value[i].x,
                        derivative: tendon_derivative[off],
                    },
                    y: crate::aba::Dual {
                        value: ball_value[i].y,
                        derivative: tendon_derivative[off + 1],
                    },
                    z: crate::aba::Dual {
                        value: ball_value[i].z,
                        derivative: tendon_derivative[off + 2],
                    },
                };
            }
            JointKind::Free | JointKind::Fixed => {}
        }
    }
    let free = if matches!(tree.links[0].joint, JointKind::Free) {
        crate::aba::GForce {
            torque: crate::aba::GVec3 {
                x: crate::aba::Dual {
                    value: free_value.torque.x,
                    derivative: tendon_derivative[0],
                },
                y: crate::aba::Dual {
                    value: free_value.torque.y,
                    derivative: tendon_derivative[1],
                },
                z: crate::aba::Dual {
                    value: free_value.torque.z,
                    derivative: tendon_derivative[2],
                },
            },
            linear: crate::aba::GVec3 {
                x: crate::aba::Dual {
                    value: free_value.linear.x,
                    derivative: tendon_derivative[3],
                },
                y: crate::aba::Dual {
                    value: free_value.linear.y,
                    derivative: tendon_derivative[4],
                },
                z: crate::aba::Dual {
                    value: free_value.linear.z,
                    derivative: tendon_derivative[5],
                },
            },
        }
    } else {
        crate::aba::GForce::zero()
    };
    let damping_mass = vec![crate::aba::Dual::zero(); n];
    let mut workspace = crate::aba::SharedAbaWorkspace::new(n);
    let mut output = vec![crate::aba::Dual::zero(); nv];
    crate::aba::run(
        tree,
        &q,
        &qdot,
        &external_body,
        crate::aba::JointForces {
            scalar: &scalar,
            ball: &ball,
            free,
        },
        &damping_mass,
        &mut workspace,
        &mut output,
    );
    output.into_iter().map(|value| value.derivative).collect()
}

fn dualize_quaternion_right_tangent(q: &mut [crate::aba::Dual], axis: Vec3) {
    let half = 0.5;
    let dx = q[3].value * axis.x * half + q[1].value * axis.z * half - q[2].value * axis.y * half;
    let dy = q[3].value * axis.y * half - q[0].value * axis.z * half + q[2].value * axis.x * half;
    let dz = q[3].value * axis.z * half + q[0].value * axis.y * half - q[1].value * axis.x * half;
    let dw = -q[0].value * axis.x * half - q[1].value * axis.y * half - q[2].value * axis.z * half;
    q[0].derivative = dx;
    q[1].derivative = dy;
    q[2].derivative = dz;
    q[3].derivative = dw;
}

fn qacc_limit_position_derivative(
    q: f32,
    range: Option<(f32, f32)>,
    limit: crate::joint::JointLimit,
    disabled: bool,
) -> f32 {
    if disabled {
        return 0.0;
    }
    match range {
        Some((lo, _)) if q < lo => -limit.stiffness,
        Some((_, hi)) if q > hi => -limit.stiffness,
        _ => 0.0,
    }
}

fn tendon_force_derivative(tree: &Tree, column: usize) -> (Vec<f32>, Vec<f32>) {
    let nv = tree.nv();
    let poses = forward_kinematics(tree);
    let mut value = vec![0.0; nv];
    let mut derivative = vec![0.0; nv];
    let mut state = crate::tendon::accumulate_tendon_passive(tree, &poses, &mut value);
    crate::tendon::accumulate_tendon_actuator_qfrc(tree, &mut state, &mut value);
    for (tendon_idx, tendon) in tree.tendons.iter().enumerate() {
        let kin = &state.kinematics[tendon_idx];
        let dlength = match &tendon.kind {
            crate::tendon::TendonKind::Fixed { joints } => joints
                .iter()
                .filter(|joint| column == tree.v_offset[joint.link])
                .map(|joint| joint.coef)
                .sum(),
            crate::tendon::TendonKind::Spatial { .. } => kin.jacobian[column],
        };
        let d_jacobian = crate::tendon::tendon_position_derivative(tendon, tree, &poses, column);
        let dvelocity = d_jacobian
            .jacobian
            .iter()
            .zip(&tree.qdot)
            .map(|(jacobian, qdot)| jacobian * qdot)
            .sum::<f32>();
        let mut force_derivative = 0.0;
        if tendon.springlength.is_some() && tendon.stiffness > 0.0 {
            force_derivative -= tendon.stiffness * dlength;
        }
        if tendon.damping > 0.0 {
            force_derivative -= tendon.damping * dvelocity;
        }
        for act in tree
            .actuators
            .iter()
            .filter(|act| act.tendon_target == Some(tendon_idx))
        {
            force_derivative += act.position_derivative(kin.length, kin.velocity) * dlength
                + act.velocity_derivative(kin.length, kin.velocity) * dvelocity;
        }
        for (slot, jacobian) in kin.jacobian.iter().enumerate() {
            derivative[slot] +=
                jacobian * force_derivative + d_jacobian.jacobian[slot] * state.forces[tendon_idx];
        }
    }
    (value, derivative)
}

fn perturb_position_tangent(tree: &mut Tree, column: usize, delta: f32) {
    for (link_idx, link) in tree.links.iter().enumerate() {
        let voff = tree.v_offset[link_idx];
        let qoff = tree.q_offset[link_idx];
        match link.joint {
            JointKind::Free => {
                if column < voff + 3 && column >= voff {
                    tree.q[qoff + column - voff] += delta;
                } else if column < voff + 6 && column >= voff + 3 {
                    let axis = basis_axis(column - voff - 3);
                    let q = Quat::new(
                        tree.q[qoff + 3],
                        tree.q[qoff + 4],
                        tree.q[qoff + 5],
                        tree.q[qoff + 6],
                    );
                    let perturbed = q * Quat::from_axis_angle(axis, delta);
                    tree.q[qoff + 3] = perturbed.x;
                    tree.q[qoff + 4] = perturbed.y;
                    tree.q[qoff + 5] = perturbed.z;
                    tree.q[qoff + 6] = perturbed.w;
                }
            }
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                if column == voff {
                    tree.q[qoff] += delta;
                }
            }
            JointKind::Ball { .. } => {
                if column >= voff && column < voff + 3 {
                    let axis = basis_axis(column - voff);
                    let q = Quat::new(
                        tree.q[qoff],
                        tree.q[qoff + 1],
                        tree.q[qoff + 2],
                        tree.q[qoff + 3],
                    );
                    let perturbed = q * Quat::from_axis_angle(axis, delta);
                    tree.q[qoff] = perturbed.x;
                    tree.q[qoff + 1] = perturbed.y;
                    tree.q[qoff + 2] = perturbed.z;
                    tree.q[qoff + 3] = perturbed.w;
                }
            }
            JointKind::Fixed => {}
        }
    }
}

fn basis_axis(index: usize) -> Vec3 {
    match index {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    }
}

/// Build `dτ/dqdot` for the explicit force balance
/// `M qacc = τ - h`. RNE supplies the analytic `dh/dqdot` term.
fn qacc_qvel_force_jacobian(tree: &Tree, poses: &[(Vec3, Quat)]) -> Vec<f32> {
    let nv = tree.nv();
    let mut force = vec![0.0; nv * nv];
    let dh = bias_velocity_jacobian(tree);
    for row in 0..nv {
        for column in 0..nv {
            force[row * nv + column] = -dh[row * nv + column];
        }
    }

    for (i, link) in tree.links.iter().enumerate() {
        let offset = tree.v_offset[i];
        match link.joint {
            JointKind::Free => {
                for slot in offset..offset + 6 {
                    force[slot * nv + slot] -= link.free_damping;
                }
            }
            JointKind::Fixed => {}
            JointKind::Hinge {
                damping,
                range,
                limit,
                ..
            }
            | JointKind::Slide {
                damping,
                range,
                limit,
                ..
            } => {
                force[offset * nv + offset] -= damping;
                if !tree.disable_penalty_limits {
                    force[offset * nv + offset] += joint_limit_velocity_derivative(
                        tree.q[tree.q_offset[i]],
                        tree.qdot[offset],
                        range,
                        limit,
                    );
                }
                let q = tree.q[tree.q_offset[i]];
                let qdot = tree.qdot[offset];
                for act in &tree.actuators {
                    if act.tendon_target.is_none() && act.link_idx == i {
                        force[offset * nv + offset] += act.velocity_derivative(q, qdot);
                    }
                }
            }
            JointKind::Ball { damping, .. } => {
                for slot in offset..offset + 3 {
                    force[slot * nv + slot] -= damping;
                }
            }
        }
    }

    if !tree.tendons.is_empty() {
        for (tendon_idx, tendon) in tree.tendons.iter().enumerate() {
            let kin = crate::tendon::tendon_kinematics(tendon, tree, poses);
            for row in 0..nv {
                for column in 0..nv {
                    let jj = kin.jacobian[row] * kin.jacobian[column];
                    if tendon.damping > 0.0 {
                        force[row * nv + column] -= tendon.damping * jj;
                    }
                    for act in &tree.actuators {
                        if act.tendon_target == Some(tendon_idx) {
                            force[row * nv + column] +=
                                act.velocity_derivative(kin.length, kin.velocity) * jj;
                        }
                    }
                }
            }
        }
    }
    force
}

/// Build the generalized force columns from actuator controls.
fn qacc_ctrl_force_jacobian(tree: &Tree, poses: &[(Vec3, Quat)]) -> Vec<f32> {
    let nv = tree.nv();
    let nu = tree.actuators.len();
    let mut force = vec![0.0; nv * nu];
    let tendon_kinematics: Vec<_> = tree
        .tendons
        .iter()
        .map(|tendon| crate::tendon::tendon_kinematics(tendon, tree, poses))
        .collect();
    for (control, act) in tree.actuators.iter().enumerate() {
        let (len, vel) = if let Some(tid) = act.tendon_target {
            let kin = &tendon_kinematics[tid];
            (kin.length, kin.velocity)
        } else {
            let q = tree.q[tree.q_offset[act.link_idx]];
            let qdot = tree.qdot[tree.v_offset[act.link_idx]];
            (q, qdot)
        };
        let derivative = act.control_derivative(len, vel);
        if let Some(tid) = act.tendon_target {
            for (row, &jacobian) in tendon_kinematics[tid].jacobian.iter().enumerate() {
                force[row * nu + control] += jacobian * derivative;
            }
        } else {
            force[tree.v_offset[act.link_idx] * nu + control] += derivative;
        }
    }
    force
}

/// Solve `M X = rhs` for a row-major matrix whose columns are derivative
/// force directions. The returned matrix has the same column count.
fn solve_mass_columns(mass: &[f32], nv: usize, rhs: &[f32]) -> Vec<f32> {
    if nv == 0 {
        return Vec::new();
    }
    let columns = rhs.len() / nv;
    let factor = cholesky(mass, nv).expect("tree mass matrix must be positive definite");
    let mut out = vec![0.0; rhs.len()];
    for column in 0..columns {
        let mut b = vec![0.0; nv];
        for row in 0..nv {
            b[row] = rhs[row * columns + column];
        }
        let x = cholesky_solve(&factor, nv, &b);
        for row in 0..nv {
            out[row * columns + column] = x[row];
        }
    }
    out
}

/// Analytic velocity derivative of the RNE bias vector.
fn bias_velocity_jacobian(tree: &Tree) -> Vec<f32> {
    let nv = tree.nv();
    let n = tree.links.len();
    let xup = compute_xup(tree);
    let mut out = vec![0.0; nv * nv];
    for column in 0..nv {
        let mut v = vec![SpatialMotion::ZERO; n];
        let mut a = vec![SpatialMotion::ZERO; n];
        let mut dv = vec![SpatialMotion::ZERO; n];
        let mut da = vec![SpatialMotion::ZERO; n];
        let mut f = vec![SpatialForce::ZERO; n];
        let mut df = vec![SpatialForce::ZERO; n];
        for i in 0..n {
            let link = &tree.links[i];
            match link.joint {
                JointKind::Free => {
                    let offset = tree.v_offset[i];
                    v[i] = SpatialMotion::new(
                        Vec3::new(
                            tree.qdot[offset],
                            tree.qdot[offset + 1],
                            tree.qdot[offset + 2],
                        ),
                        Vec3::new(
                            tree.qdot[offset + 3],
                            tree.qdot[offset + 4],
                            tree.qdot[offset + 5],
                        ),
                    );
                    dv[i] = SpatialMotion::new(
                        Vec3::new(
                            basis(column, offset),
                            basis(column, offset + 1),
                            basis(column, offset + 2),
                        ),
                        Vec3::new(
                            basis(column, offset + 3),
                            basis(column, offset + 4),
                            basis(column, offset + 5),
                        ),
                    );
                }
                JointKind::Fixed => {
                    if let Some(parent) = link.parent {
                        v[i] = xup[i].motion(v[parent]);
                        dv[i] = xup[i].motion(dv[parent]);
                    }
                }
                JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                    let parent = link.parent.expect("single-DOF joint parent");
                    let s = subspace_single(link);
                    let qdot = tree.qdot[tree.v_offset[i]];
                    let sq = s * qdot;
                    v[i] = xup[i].motion(v[parent]) + sq;
                    let dq = if column == tree.v_offset[i] { 1.0 } else { 0.0 };
                    dv[i] = xup[i].motion(dv[parent]) + s * dq;
                    a[i] = xup[i].motion(a[parent]) + v[i].cross_motion(sq);
                    da[i] = xup[i].motion(da[parent])
                        + dv[i].cross_motion(sq)
                        + v[i].cross_motion(s * dq);
                }
                JointKind::Ball { .. } => {
                    let parent = link.parent.expect("ball joint parent");
                    let s3 = subspace_ball(link);
                    let offset = tree.v_offset[i];
                    let omega = Vec3::new(
                        tree.qdot[offset],
                        tree.qdot[offset + 1],
                        tree.qdot[offset + 2],
                    );
                    let sq = s3[0] * omega.x + s3[1] * omega.y + s3[2] * omega.z;
                    let domega = Vec3::new(
                        basis(column, offset),
                        basis(column, offset + 1),
                        basis(column, offset + 2),
                    );
                    let dsq = s3[0] * domega.x + s3[1] * domega.y + s3[2] * domega.z;
                    v[i] = xup[i].motion(v[parent]) + sq;
                    dv[i] = xup[i].motion(dv[parent]) + dsq;
                    a[i] = xup[i].motion(a[parent]) + v[i].cross_motion(sq);
                    da[i] =
                        xup[i].motion(da[parent]) + dv[i].cross_motion(sq) + v[i].cross_motion(dsq);
                }
            }
            let inertia = Mat6::from_spatial_inertia(link.spatial_inertia());
            let iv = inertia.times_motion(v[i]);
            let div = inertia.times_motion(dv[i]);
            f[i] = inertia.times_motion(a[i]) + v[i].cross_force(iv);
            df[i] = inertia.times_motion(da[i]) + dv[i].cross_force(iv) + v[i].cross_force(div);
        }
        let mut column_out = vec![0.0; nv];
        for i in (1..n).rev() {
            let link = &tree.links[i];
            match link.joint {
                JointKind::Fixed => {
                    let parent = link.parent.expect("fixed joint parent");
                    df[parent] = df[parent] + xup[i].transpose_force(df[i]);
                }
                JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                    let offset = tree.v_offset[i];
                    let parent = link.parent.expect("single-DOF joint parent");
                    column_out[offset] = spatial_dot_ms(subspace_single(link), df[i]);
                    df[parent] = df[parent] + xup[i].transpose_force(df[i]);
                }
                JointKind::Ball { .. } => {
                    let offset = tree.v_offset[i];
                    let parent = link.parent.expect("ball joint parent");
                    let s3 = subspace_ball(link);
                    for k in 0..3 {
                        column_out[offset + k] = spatial_dot_ms(s3[k], df[i]);
                    }
                    df[parent] = df[parent] + xup[i].transpose_force(df[i]);
                }
                JointKind::Free => unreachable!("free joint only allowed at root"),
            }
        }
        if matches!(tree.links[0].joint, JointKind::Free) {
            let offset = tree.v_offset[0];
            column_out[offset] = df[0].torque.x;
            column_out[offset + 1] = df[0].torque.y;
            column_out[offset + 2] = df[0].torque.z;
            column_out[offset + 3] = df[0].linear.x;
            column_out[offset + 4] = df[0].linear.y;
            column_out[offset + 5] = df[0].linear.z;
        }
        for row in 0..nv {
            out[row * nv + column] = column_out[row];
        }
    }
    out
}

fn joint_limit_velocity_derivative(
    q: f32,
    qdot: f32,
    range: Option<(f32, f32)>,
    limit: crate::joint::JointLimit,
) -> f32 {
    let Some((lo, hi)) = range else { return 0.0 };
    if (q < lo && qdot < 0.0) || (q > hi && qdot > 0.0) {
        -limit.damping
    } else {
        0.0
    }
}

#[inline]
fn basis(column: usize, slot: usize) -> f32 {
    if column == slot { 1.0 } else { 0.0 }
}
