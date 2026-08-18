use super::*;
use crate::math::Mat3;

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

/// Differentiate the smooth ABA recursion in each generalized tangent
/// coordinate. This is the RNE/CRB chain in forward mode: transforms, bias
/// forces, articulated inertias, and the pass-3 solve all carry one tangent.
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

#[derive(Clone, Copy)]
struct DVec3 {
    value: Vec3,
    derivative: Vec3,
}

#[derive(Clone, Copy)]
struct DMotion {
    value: SpatialMotion,
    derivative: SpatialMotion,
}

#[derive(Clone, Copy)]
struct DForce {
    value: SpatialForce,
    derivative: SpatialForce,
}

#[derive(Clone, Copy)]
struct DXform {
    value: Xform,
    derivative: Xform,
}

#[derive(Clone, Copy)]
struct DMat3 {
    value: Mat3,
    derivative: Mat3,
}

#[derive(Clone, Copy)]
struct DMat6 {
    value: Mat6,
    derivative: Mat6,
}

fn dm_add(a: DMotion, b: DMotion) -> DMotion {
    DMotion {
        value: a.value + b.value,
        derivative: a.derivative + b.derivative,
    }
}

fn dm_cross(a: DMotion, b: DMotion) -> DMotion {
    DMotion {
        value: a.value.cross_motion(b.value),
        derivative: a.derivative.cross_motion(b.value) + a.value.cross_motion(b.derivative),
    }
}

fn df_add(a: DForce, b: DForce) -> DForce {
    DForce {
        value: a.value + b.value,
        derivative: a.derivative + b.derivative,
    }
}

fn df_sub(a: DForce, b: DForce) -> DForce {
    DForce {
        value: a.value - b.value,
        derivative: a.derivative - b.derivative,
    }
}

fn dxf_motion(x: DXform, m: DMotion) -> DMotion {
    DMotion {
        value: x.value.motion(m.value),
        derivative: x.derivative.motion(m.value) + x.value.motion(m.derivative),
    }
}

fn dxf_transpose_force(x: DXform, f: DForce) -> DForce {
    DForce {
        value: x.value.transpose_force(f.value),
        derivative: x.derivative.transpose_force(f.value) + x.value.transpose_force(f.derivative),
    }
}

fn dmat6_motion(a: DMat6, m: DMotion) -> DForce {
    DForce {
        value: a.value.times_motion(m.value),
        derivative: a.derivative.times_motion(m.value) + a.value.times_motion(m.derivative),
    }
}

fn dmat6_add(a: DMat6, b: DMat6) -> DMat6 {
    DMat6 {
        value: a.value.plus(b.value),
        derivative: a.derivative.plus(b.derivative),
    }
}

fn dmat6_sub(a: DMat6, b: DMat6) -> DMat6 {
    DMat6 {
        value: a.value.minus(b.value),
        derivative: a.derivative.minus(b.derivative),
    }
}

fn dmat6_outer(f: DForce, m: DMotion) -> DMat6 {
    DMat6 {
        value: Mat6::outer(f.value, m.value),
        derivative: Mat6::outer(f.derivative, m.value).plus(Mat6::outer(f.value, m.derivative)),
    }
}

fn dmat6_pull_back(a: DMat6, x: DXform) -> DMat6 {
    let basis = [
        SpatialMotion::new(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
        SpatialMotion::new(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO),
        SpatialMotion::new(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO),
        SpatialMotion::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
        SpatialMotion::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SpatialMotion::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)),
    ];
    let mut value = [[0.0; 6]; 6];
    let mut derivative = [[0.0; 6]; 6];
    for (column, basis_motion) in basis.into_iter().enumerate() {
        let transformed = dxf_motion(
            x,
            DMotion {
                value: basis_motion,
                derivative: SpatialMotion::ZERO,
            },
        );
        let force = dmat6_motion(a, transformed);
        let pulled = dxf_transpose_force(x, force);
        let values = [
            pulled.value.torque.x,
            pulled.value.torque.y,
            pulled.value.torque.z,
            pulled.value.linear.x,
            pulled.value.linear.y,
            pulled.value.linear.z,
        ];
        let derivatives = [
            pulled.derivative.torque.x,
            pulled.derivative.torque.y,
            pulled.derivative.torque.z,
            pulled.derivative.linear.x,
            pulled.derivative.linear.y,
            pulled.derivative.linear.z,
        ];
        for row in 0..6 {
            value[row][column] = values[row];
            derivative[row][column] = derivatives[row];
        }
    }
    DMat6 {
        value: Mat6 { rows: value },
        derivative: Mat6 { rows: derivative },
    }
}

fn dmat3_mul(a: DMat3, b: DMat3) -> DMat3 {
    DMat3 {
        value: a.value * b.value,
        derivative: a.derivative * b.value + a.value * b.derivative,
    }
}

fn dmat3_vec(a: DMat3, b: DVec3) -> DVec3 {
    DVec3 {
        value: a.value * b.value,
        derivative: a.derivative * b.value + a.value * b.derivative,
    }
}

fn dmat3_transpose_vec(a: DMat3, b: Vec3) -> DVec3 {
    DVec3 {
        value: a.value.transpose() * b,
        derivative: a.derivative.transpose() * b,
    }
}

fn dmat3_inverse(a: DMat3) -> DMat3 {
    let value = a
        .value
        .inverse()
        .expect("ball articulated-inertia block is singular");
    DMat3 {
        value,
        derivative: -(value * a.derivative * value),
    }
}

fn d_dot(s: SpatialMotion, f: DForce) -> (f32, f32) {
    (spatial_dot_ms(s, f.value), spatial_dot_ms(s, f.derivative))
}

fn d_scalar_div(value: f32, derivative: f32, denominator: f32, denominator_derivative: f32) -> f32 {
    (derivative * denominator - value * denominator_derivative) / (denominator * denominator)
}

fn d_solve_mat6(a: DMat6, b: DForce) -> DMotion {
    let value = a
        .value
        .solve(b.value)
        .expect("root articulated inertia is singular");
    let correction = b.derivative - a.derivative.times_motion(value);
    let derivative = a
        .value
        .solve(correction)
        .expect("root articulated inertia is singular");
    DMotion { value, derivative }
}

fn d_xup(tree: &Tree, link_idx: usize, column: usize) -> DXform {
    let link = &tree.links[link_idx];
    let voff = tree.v_offset[link_idx];
    let qoff = tree.q_offset[link_idx];
    match link.joint {
        JointKind::Free => DXform {
            value: Xform::IDENTITY,
            derivative: Xform::new(Mat3::ZERO, Vec3::ZERO),
        },
        JointKind::Fixed => DXform {
            value: xup_for_link(link, 0.0),
            derivative: Xform::new(Mat3::ZERO, Vec3::ZERO),
        },
        JointKind::Hinge { axis, .. } => {
            let value = xup_for_link_hinge(link, axis, tree.q[qoff]);
            let active = column == voff;
            let derivative_rotation = if active {
                -(Mat3::skew(axis) * value.rot_a_to_b)
            } else {
                Mat3::ZERO
            };
            let derivative_translation = if active {
                -(derivative_rotation * link.joint_offset_in_parent.0)
            } else {
                Vec3::ZERO
            };
            DXform {
                value,
                derivative: Xform::new(derivative_rotation, derivative_translation),
            }
        }
        JointKind::Slide { axis, .. } => {
            let value = xup_for_link_slide(link, axis, tree.q[qoff]);
            let derivative_translation = if column == voff { -axis } else { Vec3::ZERO };
            DXform {
                value,
                derivative: Xform::new(Mat3::ZERO, derivative_translation),
            }
        }
        JointKind::Ball { .. } => {
            let q = Quat::new(
                tree.q[qoff],
                tree.q[qoff + 1],
                tree.q[qoff + 2],
                tree.q[qoff + 3],
            );
            let value = xup_for_link_ball(link, q);
            let axis = if column >= voff && column < voff + 3 {
                basis_axis(column - voff)
            } else {
                Vec3::ZERO
            };
            let derivative_rotation = if axis == Vec3::ZERO {
                Mat3::ZERO
            } else {
                -(Mat3::skew(axis) * value.rot_a_to_b)
            };
            let derivative_translation = -(derivative_rotation * link.joint_offset_in_parent.0);
            DXform {
                value,
                derivative: Xform::new(derivative_rotation, derivative_translation),
            }
        }
    }
}

fn d_world_orientations(tree: &Tree, column: usize) -> Vec<DMat3> {
    let mut out = vec![
        DMat3 {
            value: Mat3::IDENTITY,
            derivative: Mat3::ZERO
        };
        tree.links.len()
    ];
    for i in 0..tree.links.len() {
        let link = &tree.links[i];
        out[i] = match link.joint {
            JointKind::Free => {
                let q = Quat::new(tree.q[3], tree.q[4], tree.q[5], tree.q[6]);
                let value = q.to_mat3();
                let derivative = if (3..6).contains(&column) {
                    value * Mat3::skew(basis_axis(column - 3))
                } else {
                    Mat3::ZERO
                };
                DMat3 { value, derivative }
            }
            JointKind::Fixed => {
                let relative =
                    link.joint_offset_in_parent.1 * link.joint_offset_in_child.1.conjugate();
                let relative_mat = relative.to_mat3();
                let value = if let Some(parent) = link.parent {
                    out[parent].value * relative_mat
                } else {
                    relative_mat
                };
                let derivative = link
                    .parent
                    .map_or(Mat3::ZERO, |parent| out[parent].derivative * relative_mat);
                DMat3 { value, derivative }
            }
            JointKind::Hinge { axis, .. } => {
                let parent = link.parent.expect("hinge parent");
                let rotation = Quat::from_axis_angle(axis, tree.q[tree.q_offset[i]]).to_mat3();
                let derivative_rotation = if column == tree.v_offset[i] {
                    Mat3::skew(axis) * rotation
                } else {
                    Mat3::ZERO
                };
                dmat3_mul(
                    out[parent],
                    DMat3 {
                        value: rotation,
                        derivative: derivative_rotation,
                    },
                )
            }
            JointKind::Slide { .. } => link.parent.map_or(
                DMat3 {
                    value: Mat3::IDENTITY,
                    derivative: Mat3::ZERO,
                },
                |parent| out[parent],
            ),
            JointKind::Ball { .. } => {
                let parent = link.parent.expect("ball parent");
                let off = tree.q_offset[i];
                let q = Quat::new(
                    tree.q[off],
                    tree.q[off + 1],
                    tree.q[off + 2],
                    tree.q[off + 3],
                );
                let rotation = q.to_mat3();
                let derivative_rotation =
                    if column >= tree.v_offset[i] && column < tree.v_offset[i] + 3 {
                        rotation * Mat3::skew(basis_axis(column - tree.v_offset[i]))
                    } else {
                        Mat3::ZERO
                    };
                dmat3_mul(
                    out[parent],
                    DMat3 {
                        value: rotation,
                        derivative: derivative_rotation,
                    },
                )
            }
        };
    }
    out
}

fn analytic_qacc_column(
    tree: &Tree,
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
    column: usize,
) -> Vec<f32> {
    let n = tree.links.len();
    let nv = tree.nv();
    let orientations = d_world_orientations(tree, column);
    let mut xup = vec![
        DXform {
            value: Xform::IDENTITY,
            derivative: Xform::new(Mat3::ZERO, Vec3::ZERO)
        };
        n
    ];
    let mut s = vec![SpatialMotion::ZERO; n];
    let mut s3 = vec![[SpatialMotion::ZERO; 3]; n];
    let mut v = vec![
        DMotion {
            value: SpatialMotion::ZERO,
            derivative: SpatialMotion::ZERO
        };
        n
    ];
    let mut c = v.clone();
    let zero_mat = DMat6 {
        value: Mat6::ZERO,
        derivative: Mat6::ZERO,
    };
    let mut ia = vec![zero_mat; n];
    let mut pa = vec![
        DForce {
            value: SpatialForce::ZERO,
            derivative: SpatialForce::ZERO
        };
        n
    ];
    let mut d_single = vec![(0.0, 0.0); n];
    let mut d_ball = vec![
        DMat3 {
            value: Mat3::ZERO,
            derivative: Mat3::ZERO
        };
        n
    ];
    let mut tau = vec![(0.0, 0.0); n];
    let (tendon_qfrc, tendon_qfrc_derivative) = tendon_force_derivative(tree, column);
    let joint_forces = crate::forces::assemble_joint_forces(tree, &tendon_qfrc);

    for i in 0..n {
        let link = &tree.links[i];
        xup[i] = d_xup(tree, i, column);
        match link.joint {
            JointKind::Free => {
                let off = tree.v_offset[i];
                v[i].value = SpatialMotion::new(
                    Vec3::new(tree.qdot[off], tree.qdot[off + 1], tree.qdot[off + 2]),
                    Vec3::new(tree.qdot[off + 3], tree.qdot[off + 4], tree.qdot[off + 5]),
                );
            }
            JointKind::Fixed => {
                if let Some(parent) = link.parent {
                    v[i] = dxf_motion(xup[i], v[parent]);
                }
            }
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                let parent = link.parent.expect("single-DOF joint parent");
                s[i] = subspace_single(link);
                let sq = s[i] * tree.qdot[tree.v_offset[i]];
                v[i] = dm_add(
                    dxf_motion(xup[i], v[parent]),
                    DMotion {
                        value: sq,
                        derivative: SpatialMotion::ZERO,
                    },
                );
            }
            JointKind::Ball { .. } => {
                let parent = link.parent.expect("ball joint parent");
                s3[i] = subspace_ball(link);
                let off = tree.v_offset[i];
                let omega = Vec3::new(tree.qdot[off], tree.qdot[off + 1], tree.qdot[off + 2]);
                let sq = s3[i][0] * omega.x + s3[i][1] * omega.y + s3[i][2] * omega.z;
                v[i] = dm_add(
                    dxf_motion(xup[i], v[parent]),
                    DMotion {
                        value: sq,
                        derivative: SpatialMotion::ZERO,
                    },
                );
            }
        }
        match link.joint {
            JointKind::Free | JointKind::Fixed => {
                c[i] = DMotion {
                    value: SpatialMotion::ZERO,
                    derivative: SpatialMotion::ZERO,
                }
            }
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                let sq = s[i] * tree.qdot[tree.v_offset[i]];
                c[i] = dm_cross(
                    v[i],
                    DMotion {
                        value: sq,
                        derivative: SpatialMotion::ZERO,
                    },
                );
            }
            JointKind::Ball { .. } => {
                let off = tree.v_offset[i];
                let omega = Vec3::new(tree.qdot[off], tree.qdot[off + 1], tree.qdot[off + 2]);
                let sq = s3[i][0] * omega.x + s3[i][1] * omega.y + s3[i][2] * omega.z;
                c[i] = dm_cross(
                    v[i],
                    DMotion {
                        value: sq,
                        derivative: SpatialMotion::ZERO,
                    },
                );
            }
        }

        ia[i] = DMat6 {
            value: Mat6::from_spatial_inertia(link.spatial_inertia()),
            derivative: Mat6::ZERO,
        };
        let iv = dmat6_motion(ia[i], v[i]);
        let bias = DForce {
            value: v[i].value.cross_force(iv.value),
            derivative: v[i].derivative.cross_force(iv.value)
                + v[i].value.cross_force(iv.derivative),
        };
        let (force_world, torque_world) =
            crate::forces::external_wrench_world(tree, i, gravity, external_wrenches);
        let force_body = dmat3_transpose_vec(orientations[i], force_world);
        let torque_body = dmat3_transpose_vec(orientations[i], torque_world);
        let ext = DForce {
            value: SpatialForce::new(torque_body.value, force_body.value),
            derivative: SpatialForce::new(torque_body.derivative, force_body.derivative),
        };
        pa[i] = df_sub(bias, ext);
    }

    for i in (1..n).rev() {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Fixed => {
                let parent = link.parent.expect("fixed joint parent");
                let child_total = df_add(pa[i], dmat6_motion(ia[i], c[i]));
                let ia_parent = dmat6_pull_back(ia[i], xup[i]);
                ia[parent] = dmat6_add(ia[parent], ia_parent);
                pa[parent] = df_add(pa[parent], dxf_transpose_force(xup[i], child_total));
            }
            JointKind::Hinge {
                armature,
                damping: _,
                range,
                limit,
                ..
            }
            | JointKind::Slide {
                armature,
                damping: _,
                range,
                limit,
                ..
            } => {
                let parent = link.parent.expect("single-DOF joint parent");
                let q = tree.q[tree.q_offset[i]];
                let qdot = tree.qdot[tree.v_offset[i]];
                let ia_s = dmat6_motion(
                    ia[i],
                    DMotion {
                        value: s[i],
                        derivative: SpatialMotion::ZERO,
                    },
                );
                let d_value = spatial_dot_ms(s[i], ia_s.value) + armature;
                let d_derivative = spatial_dot_ms(s[i], ia_s.derivative);
                d_single[i] = (d_value, d_derivative);
                let p_stage = df_add(pa[i], dmat6_motion(ia[i], c[i]));
                let (s_p, ds_p) = d_dot(s[i], p_stage);
                let qacc =
                    qacc_limit_position_derivative(q, range, limit, tree.disable_penalty_limits);
                let actuator_derivative: f32 = tree
                    .actuators
                    .iter()
                    .filter(|act| act.tendon_target.is_none() && act.link_idx == i)
                    .map(|act| act.position_derivative(q, qdot))
                    .sum();
                let dq = if column == tree.v_offset[i] { 1.0 } else { 0.0 };
                let tau_derivative =
                    tendon_qfrc_derivative[tree.v_offset[i]] + (qacc + actuator_derivative) * dq;
                let u_value = joint_forces.scalar[i];
                let p_u_value = u_value - s_p;
                let p_u_derivative = tau_derivative - ds_p;
                let qdd_value = p_u_value / d_value;
                let qdd_derivative = d_scalar_div(p_u_value, p_u_derivative, d_value, d_derivative);
                let pa_full = df_add(p_stage, dforce_scale(ia_s, qdd_value, qdd_derivative));
                let ia_full = dmat6_sub(
                    ia[i],
                    dmat6_scale(
                        dmat6_outer(
                            ia_s,
                            DMotion {
                                value: s[i],
                                derivative: SpatialMotion::ZERO,
                            },
                        ),
                        1.0 / d_value,
                        -d_derivative / (d_value * d_value),
                    ),
                );
                ia[parent] = dmat6_add(ia[parent], dmat6_pull_back(ia_full, xup[i]));
                pa[parent] = df_add(pa[parent], dxf_transpose_force(xup[i], pa_full));
                tau[i] = (u_value, tau_derivative);
            }
            JointKind::Ball { armature, .. } => {
                let parent = link.parent.expect("ball joint parent");
                let ia_s3 = [
                    dmat6_motion(
                        ia[i],
                        DMotion {
                            value: s3[i][0],
                            derivative: SpatialMotion::ZERO,
                        },
                    ),
                    dmat6_motion(
                        ia[i],
                        DMotion {
                            value: s3[i][1],
                            derivative: SpatialMotion::ZERO,
                        },
                    ),
                    dmat6_motion(
                        ia[i],
                        DMotion {
                            value: s3[i][2],
                            derivative: SpatialMotion::ZERO,
                        },
                    ),
                ];
                let dvalue = Mat3::new([
                    spatial_dot_ms(s3[i][0], ia_s3[0].value) + armature,
                    spatial_dot_ms(s3[i][1], ia_s3[0].value),
                    spatial_dot_ms(s3[i][2], ia_s3[0].value),
                    spatial_dot_ms(s3[i][0], ia_s3[1].value),
                    spatial_dot_ms(s3[i][1], ia_s3[1].value) + armature,
                    spatial_dot_ms(s3[i][2], ia_s3[1].value),
                    spatial_dot_ms(s3[i][0], ia_s3[2].value),
                    spatial_dot_ms(s3[i][1], ia_s3[2].value),
                    spatial_dot_ms(s3[i][2], ia_s3[2].value) + armature,
                ]);
                let dderivative = Mat3::new([
                    spatial_dot_ms(s3[i][0], ia_s3[0].derivative),
                    spatial_dot_ms(s3[i][1], ia_s3[0].derivative),
                    spatial_dot_ms(s3[i][2], ia_s3[0].derivative),
                    spatial_dot_ms(s3[i][0], ia_s3[1].derivative),
                    spatial_dot_ms(s3[i][1], ia_s3[1].derivative),
                    spatial_dot_ms(s3[i][2], ia_s3[1].derivative),
                    spatial_dot_ms(s3[i][0], ia_s3[2].derivative),
                    spatial_dot_ms(s3[i][1], ia_s3[2].derivative),
                    spatial_dot_ms(s3[i][2], ia_s3[2].derivative),
                ]);
                let d_inv = dmat3_inverse(DMat3 {
                    value: dvalue,
                    derivative: dderivative,
                });
                d_ball[i] = d_inv;
                let p_stage = df_add(pa[i], dmat6_motion(ia[i], c[i]));
                let sp = Vec3::new(
                    spatial_dot_ms(s3[i][0], p_stage.value),
                    spatial_dot_ms(s3[i][1], p_stage.value),
                    spatial_dot_ms(s3[i][2], p_stage.value),
                );
                let dsp = Vec3::new(
                    spatial_dot_ms(s3[i][0], p_stage.derivative),
                    spatial_dot_ms(s3[i][1], p_stage.derivative),
                    spatial_dot_ms(s3[i][2], p_stage.derivative),
                );
                let off = tree.v_offset[i];
                let tau_value = joint_forces.ball[i];
                let tau_derivative = Vec3::new(
                    tendon_qfrc_derivative[off],
                    tendon_qfrc_derivative[off + 1],
                    tendon_qfrc_derivative[off + 2],
                );
                let u = tau_value - sp;
                let du = tau_derivative - dsp;
                let qdd = dmat3_vec(
                    d_inv,
                    DVec3 {
                        value: u,
                        derivative: du,
                    },
                );
                let ia_full = dmat6_sub(ia[i], dmat6_ball_reduction(&ia_s3, &s3[i], d_inv));
                let pa_full = df_add(
                    p_stage,
                    dforce_scale(ia_s3[0], qdd.value.x, qdd.derivative.x),
                );
                let pa_full = df_add(
                    pa_full,
                    dforce_scale(ia_s3[1], qdd.value.y, qdd.derivative.y),
                );
                let pa_full = df_add(
                    pa_full,
                    dforce_scale(ia_s3[2], qdd.value.z, qdd.derivative.z),
                );
                ia[parent] = dmat6_add(ia[parent], dmat6_pull_back(ia_full, xup[i]));
                pa[parent] = df_add(pa[parent], dxf_transpose_force(xup[i], pa_full));
            }
            JointKind::Free => unreachable!("free joint only allowed at root"),
        }
    }

    let mut acceleration = vec![
        DMotion {
            value: SpatialMotion::ZERO,
            derivative: SpatialMotion::ZERO
        };
        n
    ];
    let mut qacc_derivative = vec![0.0; nv];
    match tree.links[0].joint {
        JointKind::Free => {
            let tau_value = joint_forces.free;
            let rhs = df_sub(
                DForce {
                    value: tau_value,
                    derivative: SpatialForce::new(
                        Vec3::new(
                            tendon_qfrc_derivative[0],
                            tendon_qfrc_derivative[1],
                            tendon_qfrc_derivative[2],
                        ),
                        Vec3::new(
                            tendon_qfrc_derivative[3],
                            tendon_qfrc_derivative[4],
                            tendon_qfrc_derivative[5],
                        ),
                    ),
                },
                pa[0],
            );
            acceleration[0] = d_solve_mat6(ia[0], rhs);
            let root = acceleration[0];
            let root_derivatives = [
                root.derivative.angular.x,
                root.derivative.angular.y,
                root.derivative.angular.z,
                root.derivative.linear.x,
                root.derivative.linear.y,
                root.derivative.linear.z,
            ];
            qacc_derivative[..6].copy_from_slice(&root_derivatives);
        }
        JointKind::Fixed => {}
        _ => unreachable!("only Free/Fixed roots are valid"),
    }
    for i in 1..n {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Fixed => {
                let parent = link.parent.expect("fixed joint parent");
                acceleration[i] = dxf_motion(xup[i], acceleration[parent]);
            }
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                let parent = link.parent.expect("single-DOF joint parent");
                let prime = dm_add(dxf_motion(xup[i], acceleration[parent]), c[i]);
                let inner = df_add(dmat6_motion(ia[i], prime), pa[i]);
                let s_inner = d_dot(s[i], inner);
                let numerator = tau[i].0 - s_inner.0;
                let numerator_derivative = tau[i].1 - s_inner.1;
                let (d_value, d_derivative) = d_single[i];
                let qdd_value = numerator / d_value;
                let qdd_derivative =
                    d_scalar_div(numerator, numerator_derivative, d_value, d_derivative);
                acceleration[i] = dm_add(
                    prime,
                    DMotion {
                        value: s[i] * qdd_value,
                        derivative: s[i] * qdd_derivative,
                    },
                );
                qacc_derivative[tree.v_offset[i]] = qdd_derivative;
            }
            JointKind::Ball { .. } => {
                let parent = link.parent.expect("ball joint parent");
                let prime = dm_add(dxf_motion(xup[i], acceleration[parent]), c[i]);
                let inner = df_add(dmat6_motion(ia[i], prime), pa[i]);
                let off = tree.v_offset[i];
                let tau_value = Vec3::new(
                    tree.qfrc_applied[off] + tendon_qfrc[off] - link_damping(tree, i, 0),
                    tree.qfrc_applied[off + 1] + tendon_qfrc[off + 1] - link_damping(tree, i, 1),
                    tree.qfrc_applied[off + 2] + tendon_qfrc[off + 2] - link_damping(tree, i, 2),
                );
                let tau_derivative = Vec3::new(
                    tendon_qfrc_derivative[off],
                    tendon_qfrc_derivative[off + 1],
                    tendon_qfrc_derivative[off + 2],
                );
                let u = tau_value
                    - Vec3::new(
                        spatial_dot_ms(s3[i][0], inner.value),
                        spatial_dot_ms(s3[i][1], inner.value),
                        spatial_dot_ms(s3[i][2], inner.value),
                    );
                let du = tau_derivative
                    - Vec3::new(
                        spatial_dot_ms(s3[i][0], inner.derivative),
                        spatial_dot_ms(s3[i][1], inner.derivative),
                        spatial_dot_ms(s3[i][2], inner.derivative),
                    );
                let qdd = dmat3_vec(
                    d_ball[i],
                    DVec3 {
                        value: u,
                        derivative: du,
                    },
                );
                acceleration[i] = dm_add(
                    prime,
                    DMotion {
                        value: s3[i][0] * qdd.value.x
                            + s3[i][1] * qdd.value.y
                            + s3[i][2] * qdd.value.z,
                        derivative: s3[i][0] * qdd.derivative.x
                            + s3[i][1] * qdd.derivative.y
                            + s3[i][2] * qdd.derivative.z,
                    },
                );
                qacc_derivative[off] = qdd.derivative.x;
                qacc_derivative[off + 1] = qdd.derivative.y;
                qacc_derivative[off + 2] = qdd.derivative.z;
            }
            JointKind::Free => unreachable!(),
        }
    }
    qacc_derivative
}

fn dforce_scale(force: DForce, value: f32, derivative: f32) -> DForce {
    DForce {
        value: force.value * value,
        derivative: force.derivative * value + force.value * derivative,
    }
}

fn link_damping(tree: &Tree, link_idx: usize, axis: usize) -> f32 {
    match tree.links[link_idx].joint {
        JointKind::Ball { damping, .. } => damping * tree.qdot[tree.v_offset[link_idx] + axis],
        _ => 0.0,
    }
}

fn dmat6_scale(matrix: DMat6, value: f32, derivative: f32) -> DMat6 {
    let scale = |matrix: Mat6, scalar: f32| {
        let mut out = matrix;
        for row in &mut out.rows {
            for slot in row {
                *slot *= scalar;
            }
        }
        out
    };
    DMat6 {
        value: scale(matrix.value, value),
        derivative: scale(matrix.derivative, value).plus(scale(matrix.value, derivative)),
    }
}

fn dmat6_ball_reduction(ia_s3: &[DForce; 3], s3: &[SpatialMotion; 3], inverse: DMat3) -> DMat6 {
    let mut reduction = DMat6 {
        value: Mat6::ZERO,
        derivative: Mat6::ZERO,
    };
    for (column, &s_column) in s3.iter().enumerate() {
        let mut a_col = DForce {
            value: SpatialForce::ZERO,
            derivative: SpatialForce::ZERO,
        };
        for (row, ia_row) in ia_s3.iter().enumerate() {
            a_col = df_add(
                a_col,
                dforce_scale(
                    *ia_row,
                    inverse.value.get(row, column),
                    inverse.derivative.get(row, column),
                ),
            );
        }
        reduction = dmat6_add(
            reduction,
            dmat6_outer(
                a_col,
                DMotion {
                    value: s_column,
                    derivative: SpatialMotion::ZERO,
                },
            ),
        );
    }
    reduction
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
