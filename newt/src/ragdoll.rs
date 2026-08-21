//! Builder for box-link ragdolls backed by a [`crate::tree::Tree`].

use crate::geom::Geom;
use crate::joint::{JointKind, JointLimit};
use crate::math::{Quat, Vec3};
use crate::tree::{Link, Tree};
use crate::world::World;
use std::collections::HashMap;
use std::fmt;

/// Tree-local body handle returned by [`RagdollBuilder::build`].
pub type BodyId = usize;

/// Tree-local joint handle returned by [`RagdollBuilder::build`].
///
/// A joint is stored on the child link, so this is the child link index.
pub type JointId = usize;

/// Joint specification for one ragdoll bone.
#[derive(Clone, Debug, PartialEq)]
pub enum RagdollJointSpec {
    /// Free root link. This variant is valid only for the root bone.
    Free,
    /// Rigidly attach this bone to its parent.
    Fixed,
    /// Revolute joint with an optional radian range.
    Hinge {
        axis: Vec3,
        limit: Option<(f32, f32)>,
    },
    /// Spherical joint with optional orientation limits.
    Ball {
        swing_limit: Option<f32>,
        twist_limit: Option<f32>,
    },
    /// Six-degree-of-freedom joint. Newt has no matching primitive yet.
    SixDof {
        linear_limits: [Option<(f32, f32)>; 3],
        angular_limits: [Option<(f32, f32)>; 3],
    },
}

/// One named bone in a ragdoll hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub struct BoneSpec {
    pub name: String,
    pub parent: Option<String>,
    pub length: f32,
    pub half_extents: Vec3,
    pub mass: f32,
    pub joint: RagdollJointSpec,
}

/// Stable handles for the links and joints created by a ragdoll build.
#[derive(Clone, Debug, PartialEq)]
pub struct RagdollHandles {
    /// Link ids, in the same order as the builder's bone specs.
    pub bodies: Vec<BodyId>,
    /// Child-link ids for non-root joints, in bone-spec order.
    pub joints: Vec<JointId>,
    /// Bone name to link id lookup.
    pub by_name: HashMap<String, BodyId>,
}

/// Failure while validating or constructing a ragdoll.
#[derive(Clone, Debug, PartialEq)]
pub enum RagdollBuildError {
    /// The builder has no root bone.
    RootNotPresent,
    /// More than one bone has no parent.
    MultipleRoots,
    /// A parent name does not identify a submitted bone.
    UnknownParent(String),
    /// A bone name occurs more than once.
    DuplicateBone(String),
    /// The parent graph is not a tree.
    Cycle,
    /// A bone length must be positive and finite.
    ZeroLength(String),
    /// A bone mass must be positive and finite.
    InvalidMass(String),
    /// Every half extent must be positive and finite.
    InvalidHalfExtents(String),
    /// A hinge axis must be non-zero and finite.
    InvalidAxis(String),
    /// A joint limit must be finite and have a lower value below its upper value.
    InvalidJointLimit { bone: String, low: f32, high: f32 },
    /// Root bones support only `Free` and `Fixed` joint specifications.
    UnsupportedRootJoint(String),
    /// A requested joint has no matching Newt primitive.
    UnsupportedJoint(String),
}

impl fmt::Display for RagdollBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootNotPresent => f.write_str("ragdoll has no root bone"),
            Self::MultipleRoots => f.write_str("ragdoll has multiple root bones"),
            Self::UnknownParent(name) => write!(f, "unknown ragdoll parent {name:?}"),
            Self::DuplicateBone(name) => write!(f, "duplicate ragdoll bone {name:?}"),
            Self::Cycle => f.write_str("ragdoll hierarchy contains a cycle"),
            Self::ZeroLength(name) => write!(f, "ragdoll bone {name:?} has zero length"),
            Self::InvalidMass(name) => write!(f, "ragdoll bone {name:?} has invalid mass"),
            Self::InvalidHalfExtents(name) => {
                write!(f, "ragdoll bone {name:?} has invalid half-extents")
            }
            Self::InvalidAxis(name) => {
                write!(f, "ragdoll bone {name:?} has an invalid hinge axis")
            }
            Self::InvalidJointLimit { bone, low, high } => write!(
                f,
                "ragdoll bone {bone:?} has invalid joint limit ({low}, {high})"
            ),
            Self::UnsupportedRootJoint(name) => {
                write!(
                    f,
                    "ragdoll root bone {name:?} requests an unsupported joint"
                )
            }
            Self::UnsupportedJoint(name) => {
                write!(f, "ragdoll bone {name:?} requests an unsupported joint")
            }
        }
    }
}

impl std::error::Error for RagdollBuildError {}

/// Builder for a single articulated ragdoll tree.
#[derive(Clone, Debug, Default)]
pub struct RagdollBuilder {
    bones: Vec<BoneSpec>,
    root_pinned: bool,
}

impl RagdollBuilder {
    /// Create an empty builder with a free root.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a bone. The final tree is topologically ordered even if specs
    /// arrive in a different order.
    pub fn add_bone(&mut self, bone: BoneSpec) -> &mut Self {
        self.bones.push(bone);
        self
    }

    /// Pin the root link to the world at the origin with a fixed joint.
    pub fn pin_root(&mut self) -> &mut Self {
        self.root_pinned = true;
        self
    }

    /// Build the ragdoll and attach its tree and box geoms to `world`.
    pub fn build(&self, world: &mut World) -> Result<RagdollHandles, RagdollBuildError> {
        let parent_indices = self.validate_specs()?;
        let order = topological_order(&parent_indices)?;
        let mut tree = Tree::new();
        let mut link_ids = vec![0; self.bones.len()];

        for &bone_index in &order {
            let bone = &self.bones[bone_index];
            let is_root = parent_indices[bone_index].is_none();
            let joint = if is_root {
                match bone.joint {
                    RagdollJointSpec::Free => {
                        if self.root_pinned {
                            JointKind::Fixed
                        } else {
                            JointKind::Free
                        }
                    }
                    RagdollJointSpec::Fixed => JointKind::Fixed,
                    _ => unreachable!("root joint was validated before construction"),
                }
            } else {
                joint_kind(bone)?
            };
            let (parent_offset, child_offset, parent) = match parent_indices[bone_index] {
                Some(parent_index) => {
                    let parent_id = link_ids[parent_index];
                    (
                        Vec3::Y * (self.bones[parent_index].length * 0.5),
                        Vec3::Y * (-bone.length * 0.5),
                        Some(parent_id),
                    )
                }
                None => (Vec3::ZERO, Vec3::ZERO, None),
            };
            let link = Link::new(
                parent,
                joint,
                (parent_offset, Quat::IDENTITY),
                (child_offset, Quat::IDENTITY),
                bone.mass,
                crate::geom::solid_box_inertia(bone.mass, bone.half_extents),
            );
            let link_id = tree.push_link(link);
            link_ids[bone_index] = link_id;
        }

        let tree_id = world.add_tree(tree);
        let _ = world.set_tree_self_collision(tree_id, false);
        for (bone_index, &link_id) in link_ids.iter().enumerate() {
            let bone = &self.bones[bone_index];
            world.add_geom(Geom::box_on_link(
                tree_id,
                link_id,
                bone.half_extents,
                Vec3::ZERO,
                Quat::IDENTITY,
                0.5,
            ));
        }

        let mut by_name = HashMap::with_capacity(self.bones.len());
        for (bone, &link_id) in self.bones.iter().zip(&link_ids) {
            by_name.insert(bone.name.clone(), link_id);
        }
        let bodies = link_ids;
        let joints = self
            .bones
            .iter()
            .enumerate()
            .filter_map(|(index, _)| parent_indices[index].map(|_| bodies[index]))
            .collect();
        Ok(RagdollHandles {
            bodies,
            joints,
            by_name,
        })
    }

    fn validate_specs(&self) -> Result<Vec<Option<usize>>, RagdollBuildError> {
        if self.bones.is_empty() {
            return Err(RagdollBuildError::RootNotPresent);
        }
        let mut indices = HashMap::with_capacity(self.bones.len());
        for (index, bone) in self.bones.iter().enumerate() {
            if indices.insert(bone.name.clone(), index).is_some() {
                return Err(RagdollBuildError::DuplicateBone(bone.name.clone()));
            }
            if bone.length <= 0.0 || !bone.length.is_finite() {
                return Err(RagdollBuildError::ZeroLength(bone.name.clone()));
            }
            if bone.mass <= 0.0 || !bone.mass.is_finite() {
                return Err(RagdollBuildError::InvalidMass(bone.name.clone()));
            }
            let extents = bone.half_extents;
            if extents.x <= 0.0
                || extents.y <= 0.0
                || extents.z <= 0.0
                || !extents.x.is_finite()
                || !extents.y.is_finite()
                || !extents.z.is_finite()
            {
                return Err(RagdollBuildError::InvalidHalfExtents(bone.name.clone()));
            }
        }

        let mut roots = 0;
        let mut parents = Vec::with_capacity(self.bones.len());
        for bone in &self.bones {
            let parent = match bone.parent.as_deref() {
                None => {
                    roots += 1;
                    None
                }
                Some(name) => Some(
                    indices
                        .get(name)
                        .copied()
                        .ok_or_else(|| RagdollBuildError::UnknownParent(name.to_string()))?,
                ),
            };
            parents.push(parent);
        }
        if roots == 0 {
            return Err(RagdollBuildError::Cycle);
        }
        if roots > 1 {
            return Err(RagdollBuildError::MultipleRoots);
        }
        for (index, bone) in self.bones.iter().enumerate() {
            if parents[index].is_none() {
                validate_root_joint(bone)?;
            } else {
                let _ = joint_kind(bone)?;
            }
        }
        Ok(parents)
    }
}

fn joint_kind(bone: &BoneSpec) -> Result<JointKind, RagdollBuildError> {
    match &bone.joint {
        RagdollJointSpec::Free => Err(RagdollBuildError::UnsupportedJoint(bone.name.clone())),
        RagdollJointSpec::Fixed => Ok(JointKind::Fixed),
        RagdollJointSpec::Hinge { axis, limit } => {
            if axis.length_squared() == 0.0
                || !axis.x.is_finite()
                || !axis.y.is_finite()
                || !axis.z.is_finite()
            {
                return Err(RagdollBuildError::InvalidAxis(bone.name.clone()));
            }
            if let Some((low, high)) = limit {
                validate_range(bone, *low, *high)?;
            }
            Ok(JointKind::Hinge {
                axis: axis.normalize(),
                range: *limit,
                damping: 0.0,
                armature: 0.0,
                limit: JointLimit::DEFAULT,
            })
        }
        RagdollJointSpec::Ball {
            swing_limit,
            twist_limit,
        } => {
            if let Some(limit) = swing_limit {
                validate_positive_limit(bone, *limit)?;
            }
            if let Some(limit) = twist_limit {
                validate_positive_limit(bone, *limit)?;
            }
            if swing_limit.is_some() || twist_limit.is_some() {
                return Err(RagdollBuildError::UnsupportedJoint(bone.name.clone()));
            }
            Ok(JointKind::ball())
        }
        RagdollJointSpec::SixDof {
            linear_limits,
            angular_limits,
        } => {
            for (low, high) in linear_limits.iter().chain(angular_limits).flatten() {
                validate_range(bone, *low, *high)?;
            }
            Err(RagdollBuildError::UnsupportedJoint(bone.name.clone()))
        }
    }
}

fn validate_root_joint(bone: &BoneSpec) -> Result<(), RagdollBuildError> {
    match bone.joint {
        RagdollJointSpec::Free | RagdollJointSpec::Fixed => Ok(()),
        _ => Err(RagdollBuildError::UnsupportedRootJoint(bone.name.clone())),
    }
}

fn validate_range(bone: &BoneSpec, low: f32, high: f32) -> Result<(), RagdollBuildError> {
    if !low.is_finite() || !high.is_finite() || low >= high {
        return Err(RagdollBuildError::InvalidJointLimit {
            bone: bone.name.clone(),
            low,
            high,
        });
    }
    Ok(())
}

fn validate_positive_limit(bone: &BoneSpec, high: f32) -> Result<(), RagdollBuildError> {
    validate_range(bone, 0.0, high)
}

fn topological_order(parents: &[Option<usize>]) -> Result<Vec<usize>, RagdollBuildError> {
    let mut done = vec![false; parents.len()];
    let mut order = Vec::with_capacity(parents.len());
    for _ in 0..parents.len() {
        let next = parents.iter().enumerate().find_map(|(index, parent)| {
            if done[index] || parent.is_some_and(|parent| !done[parent]) {
                None
            } else {
                Some(index)
            }
        });
        let Some(index) = next else {
            return Err(RagdollBuildError::Cycle);
        };
        done[index] = true;
        order.push(index);
    }
    Ok(order)
}
