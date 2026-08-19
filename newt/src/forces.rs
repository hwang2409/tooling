use crate::joint::JointKind;
use crate::math::Vec3;
use crate::spatial::SpatialForce;
use crate::tree::{Tree, joint_limit_scalar_force};

pub(crate) fn assemble_joint_forces(
    tree: &Tree,
    tendon_qfrc: &[f32],
    scalar: &mut [f32],
    ball: &mut [Vec3],
) -> SpatialForce {
    scalar.fill(0.0);
    ball.fill(Vec3::ZERO);
    let mut free = SpatialForce::ZERO;

    for (link_idx, link) in tree.links.iter().enumerate() {
        match link.joint {
            JointKind::Free => {
                if link_idx == 0 {
                    let applied = tree.disable_penalty_limits;
                    free = SpatialForce::new(
                        Vec3::new(
                            tendon_qfrc[0] + if applied { tree.qfrc_applied[0] } else { 0.0 },
                            tendon_qfrc[1] + if applied { tree.qfrc_applied[1] } else { 0.0 },
                            tendon_qfrc[2] + if applied { tree.qfrc_applied[2] } else { 0.0 },
                        ),
                        Vec3::new(
                            tendon_qfrc[3] + if applied { tree.qfrc_applied[3] } else { 0.0 },
                            tendon_qfrc[4] + if applied { tree.qfrc_applied[4] } else { 0.0 },
                            tendon_qfrc[5] + if applied { tree.qfrc_applied[5] } else { 0.0 },
                        ),
                    );
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
                let q_offset = tree.q_offset[link_idx];
                let v_offset = tree.v_offset[link_idx];
                let q = tree.q[q_offset];
                let qdot = tree.qdot[v_offset];
                let tau_limit = if tree.disable_penalty_limits {
                    0.0
                } else {
                    joint_limit_scalar_force(q, qdot, range, limit)
                };
                let tau_actuator = tree
                    .actuators
                    .iter()
                    .filter(|act| act.tendon_target.is_none() && act.link_idx == link_idx)
                    .map(|act| act.torque(q, qdot))
                    .sum::<f32>();
                scalar[link_idx] = tree.qfrc_applied[v_offset] + tendon_qfrc[v_offset]
                    - damping * qdot
                    + tau_limit
                    + tau_actuator;
            }
            JointKind::Ball { damping, .. } => {
                let offset = tree.v_offset[link_idx];
                let omega = Vec3::new(
                    tree.qdot[offset],
                    tree.qdot[offset + 1],
                    tree.qdot[offset + 2],
                );
                ball[link_idx] = Vec3::new(
                    tree.qfrc_applied[offset] + tendon_qfrc[offset] - damping * omega.x,
                    tree.qfrc_applied[offset + 1] + tendon_qfrc[offset + 1] - damping * omega.y,
                    tree.qfrc_applied[offset + 2] + tendon_qfrc[offset + 2] - damping * omega.z,
                );
            }
        }
    }

    free
}
