//! Dense world-frame Jacobians for articulated trees.
//!
//! Columns use the tree's `nv` layout. A free root contributes angular
//! columns `0..3`, then linear columns `3..6`. Hinge, slide, and ball columns
//! use their `v_offset` slots. Fixed joints contribute no columns.

use crate::joint::JointKind;
use crate::math::{Quat, Vec3};
use crate::tree::{Tree, forward_kinematics};

/// World-frame Jacobian of a point and its parent body's orientation.
#[derive(Clone, Debug, PartialEq)]
pub struct Jacobian {
    /// `∂position_world / ∂qdot`, one world vector per `nv` slot.
    pub translational: Vec<Vec3>,
    /// `∂angular_velocity_world / ∂qdot`, one world vector per `nv` slot.
    pub rotational: Vec<Vec3>,
}

impl Jacobian {
    /// Number of dense columns.
    pub fn nv(&self) -> usize {
        self.translational.len()
    }

    /// Return the point velocity from a dense generalized velocity vector.
    pub fn velocity(&self, qdot: &[f32]) -> (Vec3, Vec3) {
        assert_eq!(
            qdot.len(),
            self.nv(),
            "qdot length must match Jacobian columns"
        );
        let mut linear = Vec3::ZERO;
        let mut angular = Vec3::ZERO;
        for (i, &rate) in qdot.iter().enumerate() {
            linear += self.translational[i] * rate;
            angular += self.rotational[i] * rate;
        }
        (linear, angular)
    }
}

/// Jacobian for a point expressed in `link`'s body frame.
pub fn point_jacobian(tree: &Tree, link: usize, point_local: Vec3) -> Jacobian {
    assert!(link < tree.links.len(), "link index out of range");
    let poses = forward_kinematics(tree);
    let (point, _) = {
        let (com, orientation) = poses[link];
        (com + orientation.rotate(point_local), orientation)
    };
    let mut out = Jacobian {
        translational: vec![Vec3::ZERO; tree.nv()],
        rotational: vec![Vec3::ZERO; tree.nv()],
    };
    let mut chain = Vec::new();
    let mut current = Some(link);
    while let Some(i) = current {
        chain.push(i);
        current = tree.links[i].parent;
    }
    chain.reverse();

    for i in chain {
        let joint = &tree.links[i].joint;
        let (com, orientation) = poses[i];
        match *joint {
            JointKind::Free => {
                let r = point - com;
                let voff = tree.v_offset[i];
                for k in 0..3 {
                    let axis_world = orientation.rotate(axis(k));
                    out.rotational[voff + k] = axis_world;
                    out.translational[voff + k] = axis_world.cross(r);
                }
                for k in 0..3 {
                    out.translational[voff + 3 + k] = orientation.rotate(axis(k));
                }
            }
            JointKind::Fixed => {}
            JointKind::Hinge { axis, .. } => {
                let parent = tree.links[i].parent.expect("hinge parent");
                let (parent_com, parent_orientation) = poses[parent];
                let joint_world =
                    parent_com + parent_orientation.rotate(tree.links[i].joint_offset_in_parent.0);
                let axis_world = parent_orientation.rotate(axis);
                out.rotational[tree.v_offset[i]] = axis_world;
                out.translational[tree.v_offset[i]] = axis_world.cross(point - joint_world);
            }
            JointKind::Slide { axis, .. } => {
                let parent = tree.links[i].parent.expect("slide parent");
                let (_, parent_orientation) = poses[parent];
                out.translational[tree.v_offset[i]] = parent_orientation.rotate(axis);
            }
            JointKind::Ball { .. } => {
                let parent = tree.links[i].parent.expect("ball parent");
                let (parent_com, parent_orientation) = poses[parent];
                let (_, child_orientation) = poses[i];
                let joint_world =
                    parent_com + parent_orientation.rotate(tree.links[i].joint_offset_in_parent.0);
                for k in 0..3 {
                    let axis_world = child_orientation.rotate(axis(k));
                    let slot = tree.v_offset[i] + k;
                    out.rotational[slot] = axis_world;
                    out.translational[slot] = axis_world.cross(point - joint_world);
                }
            }
        }
    }
    out
}

/// Jacobian for a link COM.
pub fn link_jacobian(tree: &Tree, link: usize) -> Jacobian {
    point_jacobian(tree, link, Vec3::ZERO)
}

/// Jacobian for a site frame attached to a tree link.
pub fn site_jacobian(tree: &Tree, site_link: usize, site_offset: Vec3) -> Jacobian {
    point_jacobian(tree, site_link, site_offset)
}

fn axis(index: usize) -> Vec3 {
    match index {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    }
}

/// World-frame Jacobian for a free rigid body site.
pub fn free_body_jacobian(position: Vec3, orientation: Quat, point_local: Vec3) -> Jacobian {
    let r = orientation.rotate(point_local);
    let mut out = Jacobian {
        translational: vec![Vec3::ZERO; 6],
        rotational: vec![Vec3::ZERO; 6],
    };
    for k in 0..3 {
        let axis_world = orientation.rotate(axis(k));
        out.rotational[k] = axis_world;
        out.translational[k] = axis_world.cross(r);
        out.translational[3 + k] = axis_world;
    }
    let _ = position;
    out
}
