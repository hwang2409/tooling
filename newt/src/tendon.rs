//! Tendons (v2 tier 3).
//!
//! # Two kinds
//!
//! **Fixed tendons** — length is a linear combination of scalar joint
//! coordinates:
//!
//! ```text
//! L = Σ_i coef_i · q_i     (hinge angle or slide displacement)
//! Ldot = Σ_i coef_i · qdot_i
//! ```
//!
//! The Jacobian row is CONSTANT in `q`: the `coef_i` values, placed at the
//! `v_offset[link_i]` slots.
//!
//! **Spatial tendons** — a chain of sites (fixed to bodies or the world)
//! connected by straight segments, optionally wrapping around a sphere
//! obstacle:
//!
//! ```text
//! L = Σ_k length_k     (segment lengths sum along the chain)
//! ```
//!
//! Each segment's Jacobian falls out of the classical belt-pulley envelope
//! theorem: for a segment with endpoints `(A, B)` in world coordinates,
//! the length gradient at either endpoint equals the negative unit tangent
//! into the segment at that endpoint. So
//!
//! ```text
//! dL_seg / dA_world = −û_A       (unit vector from A toward T_A or B)
//! dL_seg / dB_world = −û_B       (unit vector from B toward T_B or A)
//! ```
//!
//! Chain these with site position Jacobians `∂p_site/∂q` (built from the
//! tree's forward-kinematics recursion) to get the per-DOF row.
//!
//! # Sphere wrap
//!
//! When the segment `AB` would pass through the wrap sphere at center `C`
//! with radius `R`, the tendon deflects around the sphere along a tangent
//! → arc → tangent path. Full derivation lives in `docs/tendons.md`. The
//! two-tangent-arc construction, in the plane containing `A`, `B`, `C`:
//!
//! ```text
//! d_A = |A − C|,  d_B = |B − C|
//! t_A = √(d_A² − R²),  t_B = √(d_B² − R²)     (tangent-segment lengths)
//! φ_A = arccos(R / d_A) = arcsin(t_A / d_A)   (angle at C from CA to CT_A)
//! φ_B = arccos(R / d_B)                        (angle at C from CB to CT_B)
//! γ   = angle ACB                              (interior angle at C)
//! θ   = γ − φ_A − φ_B                          (wrap arc angle at C)
//! L_seg = t_A + R · θ + t_B                    (segment length)
//! ```
//!
//! Wrap engages when the closest-point-on-segment distance from `C` to the
//! line `AB` is less than `R` AND both endpoints lie outside the sphere.
//! On the boundary (grazing), `θ → 0` continuously and the wrap length
//! agrees with the straight-line length. Off the boundary the SUB-Jacobian
//! FORMULA changes (envelope theorem: still evaluates to a plane
//! projection of the endpoint direction), but the numeric value of `dL/dA`
//! is continuous.
//!
//! Sidesite is optional for a sphere (the tangent-arc geometry is uniquely
//! determined when `A`, `B`, `C` are non-colinear). When `A`, `B`, `C` are
//! within `SIDE_HINT_COLINEARITY_EPS` of colinear, the wrap is skipped (or
//! the sidesite hint is used to break the tie — see the `WrapSphere::
//! side_hint_world` docs). Cylinder wrapping uses the same 2D circle
//! construction in the plane normal to the cylinder axis. Pulley branches
//! scale each branch's length and Jacobian by its divisor.
//!
//! # Determinism
//!
//! - Wrap engage/disengage is a discrete switch on `d_perp - R`. `L(q)` is
//!   continuous at the boundary (as noted above); the Jacobian is
//!   continuous too, but the formula path changes.
//! - The switch is a total function of the input state — no time-based
//!   hysteresis — so the same `(q, R, sites)` always produces the same
//!   wrap decision. Trajectories that oscillate exactly on the boundary
//!   are pinned to one branch by the strict-less-than comparison
//!   `d_perp_sq < R²` (grazing = straight).
//! - Fixed tendon ordering: joints in declaration order. Spatial tendon
//!   ordering: sites in declaration order, segments in the same order.

use crate::geom::SolRef;
use crate::joint::JointKind;
use crate::math::{Mat3, Quat, Vec3, asin, atan2};
use crate::solver::SolImp;
use crate::tree::Tree;

// ---------------------------------------------------------------------------
// Public model types
// ---------------------------------------------------------------------------

/// One `(link, coefficient)` entry in a fixed tendon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedTendonJoint {
    /// Link index whose scalar joint coordinate contributes. Must be a
    /// hinge or slide.
    pub link: usize,
    /// Coefficient on that joint's `q` and `qdot`. Units are whatever
    /// keeps `L` consistent: for a hinge, the coefficient has units of
    /// length-per-radian (so the tendon reads like a pulley); for a
    /// slide, it is dimensionless (a linear gear).
    pub coef: f32,
}

/// One waypoint site along a spatial tendon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialTendonSite {
    /// Link this site is anchored to. `None` = world-frame (static
    /// waypoint anchored at `position_local`).
    pub link: Option<usize>,
    /// Site position in the parent's body frame (or world if `link` is
    /// `None`).
    pub position_local: Vec3,
}

/// Wrap sphere sitting between two consecutive spatial-tendon sites.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WrapSphere {
    /// Link the sphere is attached to. `None` = world (static wrap
    /// object).
    pub link: Option<usize>,
    /// Sphere center in the parent's body frame (or world if `link` is
    /// `None`).
    pub center_local: Vec3,
    /// Sphere radius (must be > 0).
    pub radius: f32,
    /// Optional side hint used to disambiguate the wrap plane when the
    /// segment endpoints and the sphere center are colinear. When
    /// `None`, colinear configurations skip the wrap and the tendon
    /// passes through the sphere (documented behavior — see the module
    /// docs; MuJoCo insists on a sidesite in the colinear case).
    pub side_hint_world: Option<Vec3>,
}

/// Infinite cylinder used as a spatial-tendon wrap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WrapCylinder {
    /// Link the cylinder is attached to. `None` means world-fixed.
    pub link: Option<usize>,
    /// Cylinder center in the link frame, or world frame when `link` is None.
    pub center_local: Vec3,
    /// Cylinder axis in the link frame, or world frame when `link` is None.
    pub axis_local: Vec3,
    /// Cylinder radius.
    pub radius: f32,
    /// Optional sidesite. Its world position selects the preferred wrap side.
    pub sidesite: Option<SpatialTendonSite>,
}

/// A spatial tendon wrap object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpatialWrap {
    Sphere(WrapSphere),
    Cylinder(WrapCylinder),
}

/// One segment between adjacent sites in a spatial tendon. Straight or
/// wrapped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialSegment {
    /// Optional wrap. Straight segment when `None`.
    pub wrap: Option<SpatialWrap>,
}

/// One independent branch of a spatial tendon.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialTendonBranch {
    pub sites: Vec<SpatialTendonSite>,
    pub segments: Vec<SpatialSegment>,
    pub divisor: f32,
}

/// Kind-specific tendon data.
#[derive(Clone, Debug, PartialEq)]
pub enum TendonKind {
    /// Linear combination of hinge/slide joint coordinates.
    Fixed { joints: Vec<FixedTendonJoint> },
    /// Independent site branches plus per-segment wrap options.
    Spatial { branches: Vec<SpatialTendonBranch> },
}

/// One tendon. Attached to a `Tree` via `Tree::add_tendon`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tendon {
    pub kind: TendonKind,
    /// Rest length for the passive spring (`None` → no spring, `stiffness`
    /// ignored).
    pub springlength: Option<f32>,
    /// Spring stiffness `k` (N/m). Passive force `−k·(L − L₀)` pulls the
    /// tendon toward `springlength` when `L > L₀` OR `L < L₀` (two-sided
    /// spring — matches MuJoCo's single-scalar springlength behavior; a
    /// range-limited spring lands with a future tier).
    pub stiffness: f32,
    /// Passive damping `c` (N·s/m). Force `−c · Ldot` always opposes
    /// motion.
    pub damping: f32,
    /// Length range `(lo, hi)` for the PGS limit constraint. `None` → no
    /// limit. `lo ≤ hi` enforced at construction time.
    pub range: Option<(f32, f32)>,
    /// Per-tendon SolRef for the limit constraint (`None` → SolRef::DEFAULT).
    pub limit_solref: Option<SolRef>,
    /// Per-tendon SolImp for the limit constraint (`None` → SolImp::DEFAULT).
    pub limit_solimp: Option<SolImp>,
}

impl Tendon {
    /// Construct a fixed tendon with the given joint list and default
    /// passive parameters (no spring, no damping, no limit).
    pub fn fixed(joints: Vec<FixedTendonJoint>) -> Self {
        assert!(!joints.is_empty(), "fixed tendon needs at least one joint");
        Self {
            kind: TendonKind::Fixed { joints },
            springlength: None,
            stiffness: 0.0,
            damping: 0.0,
            range: None,
            limit_solref: None,
            limit_solimp: None,
        }
    }

    /// Construct a spatial tendon over a site chain. `wraps` has one entry
    /// per segment (i.e. `sites.len() - 1`); pass `None` per segment for a
    /// straight leg.
    pub fn spatial(sites: Vec<SpatialTendonSite>, wraps: Vec<Option<WrapSphere>>) -> Self {
        assert!(sites.len() >= 2, "spatial tendon needs ≥ 2 sites");
        assert_eq!(
            wraps.len(),
            sites.len() - 1,
            "spatial tendon needs one wrap-slot per segment"
        );
        for ws in wraps.iter().flatten() {
            assert!(ws.radius > 0.0, "wrap sphere radius must be > 0");
        }
        let segments = wraps
            .into_iter()
            .map(|w| SpatialSegment {
                wrap: w.map(SpatialWrap::Sphere),
            })
            .collect();
        Self {
            kind: TendonKind::Spatial {
                branches: vec![SpatialTendonBranch {
                    sites,
                    segments,
                    divisor: 1.0,
                }],
            },
            springlength: None,
            stiffness: 0.0,
            damping: 0.0,
            range: None,
            limit_solref: None,
            limit_solimp: None,
        }
    }

    /// Construct a spatial tendon from independent pulley branches.
    pub fn spatial_branches(branches: Vec<SpatialTendonBranch>) -> Self {
        assert!(!branches.is_empty(), "spatial tendon needs ≥ 1 branch");
        Self {
            kind: TendonKind::Spatial { branches },
            springlength: None,
            stiffness: 0.0,
            damping: 0.0,
            range: None,
            limit_solref: None,
            limit_solimp: None,
        }
    }

    /// Validate the tendon against the containing tree. Returns an error
    /// message describing the first problem. Runs at load time.
    pub fn validate(&self, tree: &Tree) -> Result<(), String> {
        // Passive params.
        if self.stiffness < 0.0 {
            return Err(format!(
                "tendon stiffness must be ≥ 0 (got {})",
                self.stiffness
            ));
        }
        if self.damping < 0.0 {
            return Err(format!("tendon damping must be ≥ 0 (got {})", self.damping));
        }
        if let Some((lo, hi)) = self.range {
            if lo > hi {
                return Err(format!(
                    "tendon range must satisfy lo ≤ hi (got {lo}..{hi})"
                ));
            }
        }
        if let Some(sl) = self.springlength {
            if !sl.is_finite() {
                return Err("tendon springlength must be finite".to_string());
            }
        }
        if let Some(s) = self.limit_solimp.as_ref() {
            s.validate()?;
        }
        match &self.kind {
            TendonKind::Fixed { joints } => {
                if joints.is_empty() {
                    return Err("fixed tendon must have ≥ 1 joint".to_string());
                }
                for j in joints {
                    if j.link >= tree.links.len() {
                        return Err(format!("fixed tendon joint link {} out of range", j.link));
                    }
                    match tree.links[j.link].joint {
                        JointKind::Hinge { .. } | JointKind::Slide { .. } => {}
                        other => {
                            return Err(format!(
                                "fixed tendon joint {} is not a hinge/slide (got {other:?}) — \
                                 fixed tendons only sum scalar joints",
                                j.link
                            ));
                        }
                    }
                    if !j.coef.is_finite() {
                        return Err(format!(
                            "fixed tendon coefficient on link {} must be finite",
                            j.link
                        ));
                    }
                }
            }
            TendonKind::Spatial { branches } => {
                if branches.is_empty() {
                    return Err("spatial tendon must have ≥ 1 branch".to_string());
                }
                for (branch_idx, branch) in branches.iter().enumerate() {
                    if branch.sites.len() < 2 {
                        return Err(format!(
                            "spatial tendon branch {branch_idx} must have ≥ 2 sites"
                        ));
                    }
                    if branch.segments.len() != branch.sites.len() - 1 {
                        return Err(format!(
                            "spatial tendon branch {branch_idx} segments ({}) must equal sites-1 ({})",
                            branch.segments.len(),
                            branch.sites.len() - 1
                        ));
                    }
                    if !branch.divisor.is_finite() || branch.divisor <= 0.0 {
                        return Err(format!(
                            "spatial tendon branch {branch_idx} divisor must be finite and > 0"
                        ));
                    }
                    for (i, s) in branch.sites.iter().enumerate() {
                        if let Some(l) = s.link {
                            if l >= tree.links.len() {
                                return Err(format!(
                                    "spatial tendon branch {branch_idx} site {i}: link {l} out of range"
                                ));
                            }
                        }
                        for c in [s.position_local.x, s.position_local.y, s.position_local.z] {
                            if !c.is_finite() {
                                return Err(format!(
                                    "spatial tendon branch {branch_idx} site {i}: position must be finite"
                                ));
                            }
                        }
                    }
                    for (i, seg) in branch.segments.iter().enumerate() {
                        if let Some(w) = &seg.wrap {
                            let (link, center, radius) = match w {
                                SpatialWrap::Sphere(w) => (w.link, w.center_local, w.radius),
                                SpatialWrap::Cylinder(w) => (w.link, w.center_local, w.radius),
                            };
                            if let Some(l) = link {
                                if l >= tree.links.len() {
                                    return Err(format!(
                                        "spatial tendon branch {branch_idx} segment {i} wrap: link {l} out of range"
                                    ));
                                }
                            }
                            if radius <= 0.0 {
                                return Err(format!(
                                    "spatial tendon branch {branch_idx} segment {i} wrap: radius must be > 0"
                                ));
                            }
                            for c in [center.x, center.y, center.z] {
                                if !c.is_finite() {
                                    return Err(format!(
                                        "spatial tendon branch {branch_idx} segment {i} wrap: center must be finite"
                                    ));
                                }
                            }
                            if let SpatialWrap::Cylinder(w) = w {
                                if w.axis_local.length_squared()
                                    <= SIDE_HINT_COLINEARITY_EPS * SIDE_HINT_COLINEARITY_EPS
                                {
                                    return Err(format!(
                                        "spatial tendon branch {branch_idx} segment {i} cylinder axis must be nonzero"
                                    ));
                                }
                                if let Some(site) = w.sidesite {
                                    if let Some(l) = site.link {
                                        if l >= tree.links.len() {
                                            return Err(format!(
                                                "spatial tendon branch {branch_idx} segment {i} sidesite link {l} out of range"
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Length + Jacobian assembly
// ---------------------------------------------------------------------------

/// Colinearity threshold for the wrap sidesite check. When `|CA × CB| /
/// (|CA| · |CB|) < this`, the three points are treated as colinear and the
/// wrap is skipped (a straight line in the true colinear limit; documented
/// discrete switch — see module docs).
const SIDE_HINT_COLINEARITY_EPS: f32 = 1.0e-4;

/// Per-tendon length + Jacobian row `(v_slot, coeff)`.
///
/// `jacobian` is a dense `Vec<f32>` of length `tree.nv()` — sparse in
/// entries but stored dense so accumulation is a plain slot add. Callers
/// that want sparse assembly can compress.
#[derive(Clone, Debug)]
pub struct TendonKinematics {
    /// Scalar tendon length.
    pub length: f32,
    /// Scalar tendon rate `dL/dt`.
    pub velocity: f32,
    /// Dense length-Jacobian `dL/dqdot` (length = tree.nv()).
    pub jacobian: Vec<f32>,
}

/// Compute length + velocity + Jacobian for one tendon at the tree's
/// current state.
pub fn tendon_kinematics(tendon: &Tendon, tree: &Tree, poses: &[(Vec3, Quat)]) -> TendonKinematics {
    let nv = tree.nv();
    let mut jacobian = vec![0.0f32; nv];
    let length = match &tendon.kind {
        TendonKind::Fixed { joints } => {
            let mut l = 0.0f32;
            for j in joints {
                let q = tree.q[tree.q_offset[j.link]];
                l += j.coef * q;
                jacobian[tree.v_offset[j.link]] += j.coef;
            }
            l
        }
        TendonKind::Spatial { branches } => {
            let mut total = 0.0;
            for branch in branches {
                total += spatial_length_and_jacobian(
                    tree,
                    poses,
                    &branch.sites,
                    &branch.segments,
                    branch.divisor,
                    &mut jacobian,
                );
            }
            total
        }
    };
    let mut velocity = 0.0f32;
    for (i, &c) in jacobian.iter().enumerate() {
        if c != 0.0 {
            velocity += c * tree.qdot[i];
        }
    }
    TendonKinematics {
        length,
        velocity,
        jacobian,
    }
}

/// Position derivative of the tendon Jacobian for one smooth configuration.
///
/// The spatial path uses the same site and segment geometry as
/// [`tendon_kinematics`]. Wrap branches keep their envelope Jacobian, but do
/// not expose a second derivative in this tier.
pub(crate) struct TendonPositionDerivative {
    pub jacobian: Vec<f32>,
}

pub(crate) fn tendon_position_derivative(
    tendon: &Tendon,
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    column: usize,
) -> TendonPositionDerivative {
    let mut jacobian = vec![0.0; tree.nv()];
    let TendonKind::Spatial { branches } = &tendon.kind else {
        return TendonPositionDerivative { jacobian };
    };
    let dposes = differential_poses(tree, poses, column);
    for branch in branches {
        let sites: Vec<_> = branch
            .sites
            .iter()
            .map(|site| differential_site(tree, &dposes, site))
            .collect();
        for (index, segment) in branch.segments.iter().enumerate() {
            if segment.wrap.is_some() {
                continue;
            }
            straight_segment_jacobian_derivative(
                &sites[index],
                &sites[index + 1],
                branch.divisor,
                &mut jacobian,
            );
        }
    }
    TendonPositionDerivative { jacobian }
}

#[derive(Clone, Copy)]
struct DifferentialPose {
    position: Vec3,
    orientation: Mat3,
    dposition: Vec3,
    dorientation: Mat3,
}

fn differential_poses(tree: &Tree, poses: &[(Vec3, Quat)], column: usize) -> Vec<DifferentialPose> {
    let mut out: Vec<DifferentialPose> = Vec::with_capacity(tree.links.len());
    for (i, link) in tree.links.iter().enumerate() {
        let pose = match link.joint {
            JointKind::Free => {
                let off = tree.v_offset[i];
                let orientation = poses[i].1.to_mat3();
                let dposition = if column >= off && column < off + 3 {
                    basis_vec(column - off)
                } else {
                    Vec3::ZERO
                };
                let dorientation = if column >= off + 3 && column < off + 6 {
                    orientation * Mat3::skew(basis_vec(column - off - 3))
                } else {
                    Mat3::ZERO
                };
                DifferentialPose {
                    position: poses[i].0,
                    orientation,
                    dposition,
                    dorientation,
                }
            }
            JointKind::Fixed => {
                let relative =
                    link.joint_offset_in_parent.1 * link.joint_offset_in_child.1.conjugate();
                let relative_mat = relative.to_mat3();
                let (joint_position, djoint_position, parent_orientation, dparent_orientation) =
                    if let Some(parent) = link.parent {
                        let parent_pose = out[parent];
                        let (offset, _) = link.joint_offset_in_parent;
                        (
                            parent_pose.position + parent_pose.orientation * offset,
                            parent_pose.dposition + parent_pose.dorientation * offset,
                            parent_pose.orientation,
                            parent_pose.dorientation,
                        )
                    } else {
                        (
                            link.joint_offset_in_parent.0,
                            Vec3::ZERO,
                            Mat3::IDENTITY,
                            Mat3::ZERO,
                        )
                    };
                let orientation = parent_orientation * relative_mat;
                let dorientation = dparent_orientation * relative_mat;
                let offset = link.joint_offset_in_child.0;
                DifferentialPose {
                    position: joint_position - orientation * offset,
                    orientation,
                    dposition: djoint_position - dorientation * offset,
                    dorientation,
                }
            }
            JointKind::Hinge { axis, .. } => {
                let parent = out[link.parent.expect("hinge has parent")];
                let q = tree.q[tree.q_offset[i]];
                let rotation = Quat::from_axis_angle(axis, q).to_mat3();
                let drotation = if column == tree.v_offset[i] {
                    Mat3::skew(axis) * rotation
                } else {
                    Mat3::ZERO
                };
                let orientation = parent.orientation * rotation;
                let dorientation = parent.dorientation * rotation + parent.orientation * drotation;
                let offset = link.joint_offset_in_parent.0;
                let joint_position = parent.position + parent.orientation * offset;
                let djoint_position = parent.dposition + parent.dorientation * offset;
                let child_offset = link.joint_offset_in_child.0;
                DifferentialPose {
                    position: joint_position - orientation * child_offset,
                    orientation,
                    dposition: djoint_position - dorientation * child_offset,
                    dorientation,
                }
            }
            JointKind::Slide { axis, .. } => {
                let parent = out[link.parent.expect("slide has parent")];
                let q = tree.q[tree.q_offset[i]];
                let orientation = parent.orientation;
                let dorientation = parent.dorientation;
                let offset = link.joint_offset_in_parent.0;
                let joint_position = parent.position + parent.orientation * offset;
                let djoint_position = parent.dposition + parent.dorientation * offset;
                let slide = parent.orientation * (axis * q);
                let dslide = parent.dorientation * (axis * q)
                    + if column == tree.v_offset[i] {
                        parent.orientation * axis
                    } else {
                        Vec3::ZERO
                    };
                let child_offset = link.joint_offset_in_child.0;
                DifferentialPose {
                    position: joint_position + slide - orientation * child_offset,
                    orientation,
                    dposition: djoint_position + dslide - dorientation * child_offset,
                    dorientation,
                }
            }
            JointKind::Ball { .. } => {
                panic!("spatial tendon position derivatives do not support ball joints")
            }
        };
        out.push(pose);
    }
    out
}

struct DifferentialSite {
    world_pos: Vec3,
    d_world_pos: Vec3,
    columns: Vec<(u32, Vec3)>,
    d_columns: Vec<(u32, Vec3)>,
}

fn differential_site(
    tree: &Tree,
    poses: &[DifferentialPose],
    site: &SpatialTendonSite,
) -> DifferentialSite {
    let Some(link_idx) = site.link else {
        return DifferentialSite {
            world_pos: site.position_local,
            d_world_pos: Vec3::ZERO,
            columns: Vec::new(),
            d_columns: Vec::new(),
        };
    };
    let body = poses[link_idx];
    let world_pos = body.position + body.orientation * site.position_local;
    let d_world_pos = body.dposition + body.dorientation * site.position_local;
    let mut columns = Vec::new();
    let mut d_columns = Vec::new();
    let mut chain = Vec::new();
    let mut current = Some(link_idx);
    while let Some(i) = current {
        chain.push(i);
        current = tree.links[i].parent;
    }
    chain.reverse();
    for i in chain {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Free => {
                let root = poses[i];
                let r = world_pos - root.position;
                let dr = d_world_pos - root.dposition;
                for k in 0..3 {
                    let e = basis_vec(k);
                    let axis = root.orientation * e;
                    let daxis = root.dorientation * e;
                    columns.push(((tree.v_offset[i] + k) as u32, axis.cross(r)));
                    d_columns.push((
                        (tree.v_offset[i] + k) as u32,
                        daxis.cross(r) + axis.cross(dr),
                    ));
                }
                for k in 0..3 {
                    let e = basis_vec(k);
                    columns.push(((tree.v_offset[i] + 3 + k) as u32, root.orientation * e));
                    d_columns.push(((tree.v_offset[i] + 3 + k) as u32, root.dorientation * e));
                }
            }
            JointKind::Fixed => {}
            JointKind::Hinge { axis, .. } => {
                let parent = poses[link.parent.expect("hinge has parent")];
                let offset = link.joint_offset_in_parent.0;
                let joint = parent.position + parent.orientation * offset;
                let djoint = parent.dposition + parent.dorientation * offset;
                let axis_world = parent.orientation * axis;
                let daxis_world = parent.dorientation * axis;
                let r = world_pos - joint;
                let dr = d_world_pos - djoint;
                columns.push((tree.v_offset[i] as u32, axis_world.cross(r)));
                d_columns.push((
                    tree.v_offset[i] as u32,
                    daxis_world.cross(r) + axis_world.cross(dr),
                ));
            }
            JointKind::Slide { axis, .. } => {
                let parent = poses[link.parent.expect("slide has parent")];
                columns.push((tree.v_offset[i] as u32, parent.orientation * axis));
                d_columns.push((tree.v_offset[i] as u32, parent.dorientation * axis));
            }
            JointKind::Ball { .. } => unreachable!(),
        }
    }
    DifferentialSite {
        world_pos,
        d_world_pos,
        columns,
        d_columns,
    }
}

fn straight_segment_jacobian_derivative(
    a: &DifferentialSite,
    b: &DifferentialSite,
    divisor: f32,
    jacobian: &mut [f32],
) {
    let delta = b.world_pos - a.world_pos;
    let length = delta.length();
    if length == 0.0 {
        return;
    }
    let unit = delta / length;
    let ddelta = b.d_world_pos - a.d_world_pos;
    let dunit = (ddelta - unit * unit.dot(ddelta)) / length;
    for (slot, value) in jacobian.iter_mut().enumerate() {
        let a_col = column_value(&a.columns, slot);
        let b_col = column_value(&b.columns, slot);
        let da_col = column_value(&a.d_columns, slot);
        let db_col = column_value(&b.d_columns, slot);
        *value += (dunit.dot(b_col - a_col) + unit.dot(db_col - da_col)) / divisor;
    }
}

fn column_value(columns: &[(u32, Vec3)], slot: usize) -> Vec3 {
    columns
        .iter()
        .find(|(index, _)| *index as usize == slot)
        .map(|(_, value)| *value)
        .unwrap_or(Vec3::ZERO)
}

#[inline]
fn basis_vec(index: usize) -> Vec3 {
    match index {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    }
}

/// Site world position and its Jacobian columns w.r.t. the tree DOFs.
struct SiteJacobian {
    world_pos: Vec3,
    /// One column per contributing v-slot. Stored sparsely as
    /// `(v_slot, dp/dqdot column)`.
    columns: Vec<(u32, Vec3)>,
}

fn site_position_and_jacobian(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    site: &SpatialTendonSite,
) -> SiteJacobian {
    let Some(link_idx) = site.link else {
        // World-static site — no dependence on q.
        return SiteJacobian {
            world_pos: site.position_local,
            columns: Vec::new(),
        };
    };
    let (com, ori) = poses[link_idx];
    let world_pos = com + ori.rotate(site.position_local);
    let mut columns: Vec<(u32, Vec3)> = Vec::new();
    // Walk root → link_idx via the parent chain.
    let mut chain: Vec<usize> = Vec::new();
    let mut cur = Some(link_idx);
    while let Some(i) = cur {
        chain.push(i);
        cur = tree.links[i].parent;
    }
    chain.reverse();
    for &i in &chain {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Free => {
                assert!(link.parent.is_none(), "free joint only supported on root");
                let (root_com, root_ori) = poses[i];
                let r = world_pos - root_com;
                let voff = tree.v_offset[i];
                // ω_body slots (0..3): dp/dω_body_k = (R · e_k) × r
                for k in 0..3 {
                    let e_body = match k {
                        0 => Vec3::X,
                        1 => Vec3::Y,
                        _ => Vec3::Z,
                    };
                    let axis_world = root_ori.rotate(e_body);
                    columns.push(((voff + k) as u32, axis_world.cross(r)));
                }
                // v_body slots (3..6): dp/dv_body_k = R · e_k
                for k in 0..3 {
                    let e_body = match k {
                        0 => Vec3::X,
                        1 => Vec3::Y,
                        _ => Vec3::Z,
                    };
                    let axis_world = root_ori.rotate(e_body);
                    columns.push(((voff + 3 + k) as u32, axis_world));
                }
            }
            JointKind::Fixed => {
                // No DOF contribution.
            }
            JointKind::Hinge { axis, .. } => {
                let parent = link.parent.expect("hinge has parent");
                let (parent_com, parent_ori) = poses[parent];
                // Joint anchor in world coordinates.
                let joint_world = parent_com + parent_ori.rotate(link.joint_offset_in_parent.0);
                let axis_world = parent_ori.rotate(axis);
                let dp = axis_world.cross(world_pos - joint_world);
                columns.push((tree.v_offset[i] as u32, dp));
            }
            JointKind::Slide { axis, .. } => {
                let parent = link.parent.expect("slide has parent");
                let (_parent_com, parent_ori) = poses[parent];
                let axis_world = parent_ori.rotate(axis);
                columns.push((tree.v_offset[i] as u32, axis_world));
            }
            JointKind::Ball { .. } => {
                // Ball joints on a tendon-carrying chain: 3-column contribution
                // is not implemented in v2 tier 3. Reject loudly so the loader
                // catches this at model load (the validator does not know the
                // chain topology, so this runtime assert is the second gate).
                panic!(
                    "spatial tendon: ball joint on link {i} is not supported in v2 tier 3 \
                     (deferred; see docs/tendons.md)"
                );
            }
        }
    }
    SiteJacobian { world_pos, columns }
}

fn wrap_center_world(poses: &[(Vec3, Quat)], link: Option<usize>, center_local: Vec3) -> Vec3 {
    match link {
        Some(l) => {
            let (com, ori) = poses[l];
            com + ori.rotate(center_local)
        }
        None => center_local,
    }
}

fn wrap_point_jacobian(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    link: Option<usize>,
    world_point: Vec3,
) -> SiteJacobian {
    match link {
        None => SiteJacobian {
            world_pos: world_point,
            columns: Vec::new(),
        },
        Some(link_idx) => {
            let (com, ori) = poses[link_idx];
            let local = ori.inverse_rotate(world_point - com);
            site_position_and_jacobian(
                tree,
                poses,
                &SpatialTendonSite {
                    link: Some(link_idx),
                    position_local: local,
                },
            )
        }
    }
}

/// Compute spatial tendon length and accumulate into `jacobian`.
fn spatial_length_and_jacobian(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    sites: &[SpatialTendonSite],
    segments: &[SpatialSegment],
    divisor: f32,
    jacobian: &mut [f32],
) -> f32 {
    // Precompute all site Jacobians once.
    let site_jacs: Vec<SiteJacobian> = sites
        .iter()
        .map(|s| site_position_and_jacobian(tree, poses, s))
        .collect();
    let mut total_length = 0.0f32;
    for (i, seg) in segments.iter().enumerate() {
        let a = &site_jacs[i];
        let b = &site_jacs[i + 1];
        let mut segment_jacobian = vec![0.0f32; jacobian.len()];
        let seg_len = match &seg.wrap {
            None => straight_segment_contribution(a, b, &mut segment_jacobian),
            Some(w) => match w {
                SpatialWrap::Sphere(w) => {
                    let c_world = wrap_center_world(poses, w.link, w.center_local);
                    wrap_segment_contribution(tree, poses, a, b, c_world, w, &mut segment_jacobian)
                }
                SpatialWrap::Cylinder(w) => {
                    cylinder_segment_contribution(tree, poses, a, b, w, &mut segment_jacobian)
                }
            },
        };
        total_length += seg_len / divisor;
        for (dst, src) in jacobian.iter_mut().zip(segment_jacobian) {
            *dst += src / divisor;
        }
    }
    total_length
}

fn straight_segment_contribution(a: &SiteJacobian, b: &SiteJacobian, jacobian: &mut [f32]) -> f32 {
    let d = b.world_pos - a.world_pos;
    let len = d.length();
    if len == 0.0 {
        return 0.0;
    }
    let unit = d / len;
    // dL/dA = -unit,  dL/dB = +unit.
    // Contribution to Jacobian: dL/dq_k = (-unit · dA/dq_k) + (unit · dB/dq_k).
    for &(slot, col) in &a.columns {
        jacobian[slot as usize] -= unit.dot(col);
    }
    for &(slot, col) in &b.columns {
        jacobian[slot as usize] += unit.dot(col);
    }
    len
}

fn is_intersect_2d(a0: [f32; 2], a1: [f32; 2], b0: [f32; 2], b1: [f32; 2]) -> bool {
    let da = [a1[0] - a0[0], a1[1] - a0[1]];
    let db = [b1[0] - b0[0], b1[1] - b0[1]];
    let det = da[0] * db[1] - da[1] * db[0];
    if det == 0.0 {
        return false;
    }
    let d = [b0[0] - a0[0], b0[1] - a0[1]];
    let s = (d[0] * db[1] - d[1] * db[0]) / det;
    let t = (d[0] * da[1] - d[1] * da[0]) / det;
    (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t)
}

fn circle_arc_length(a: [f32; 2], b: [f32; 2], solution: usize, radius: f32) -> f32 {
    let a_len = (a[0] * a[0] + a[1] * a[1]).sqrt();
    let b_len = (b[0] * b[0] + b[1] * b[1]).sqrt();
    let an = [a[0] / a_len, a[1] / a_len];
    let bn = [b[0] / b_len, b[1] / b_len];
    let dot = an[0] * bn[0] + an[1] * bn[1];
    let cross = a[1] * b[0] - a[0] * b[1];
    let mut angle = atan2((1.0 - dot * dot).max(0.0).sqrt(), dot.clamp(-1.0, 1.0));
    if (cross > 0.0 && solution == 1) || (cross < 0.0 && solution == 0) {
        angle = crate::math::TAU - angle;
    }
    radius * angle
}

/// MuJoCo's 2D circle-wrap construction. Returns tangent points and arc
/// length when the straight segment intersects the circle.
fn circle_wrap_points(
    end0: [f32; 2],
    end1: [f32; 2],
    side: Option<[f32; 2]>,
    radius: f32,
) -> Option<([[f32; 2]; 2], f32)> {
    let sqlen0 = end0[0] * end0[0] + end0[1] * end0[1];
    let sqlen1 = end1[0] * end1[0] + end1[1] * end1[1];
    let sqrad = radius * radius;
    if sqlen0 < sqrad || sqlen1 < sqrad || radius <= 0.0 {
        return None;
    }
    let dif = [end1[0] - end0[0], end1[1] - end0[1]];
    let dd = dif[0] * dif[0] + dif[1] * dif[1];
    if dd < 1.0e-12 {
        return None;
    }
    let a = (-(dif[0] * end0[0] + dif[1] * end0[1]) / dd).clamp(0.0, 1.0);
    let closest = [end0[0] + a * dif[0], end0[1] + a * dif[1]];
    let closest_sq = closest[0] * closest[0] + closest[1] * closest[1];
    if closest_sq > sqrad && side.is_none_or(|s| s[0] * closest[0] + s[1] * closest[1] >= 0.0) {
        return None;
    }
    let sqrt0 = (sqlen0 - sqrad).max(0.0).sqrt();
    let sqrt1 = (sqlen1 - sqrad).max(0.0).sqrt();
    let mut best = 0;
    let mut best_good = f32::NEG_INFINITY;
    for (i, sign) in [1.0f32, -1.0].into_iter().enumerate() {
        let sol0 = [
            (end0[0] * sqrad + sign * radius * end0[1] * sqrt0) / sqlen0,
            (end0[1] * sqrad - sign * radius * end0[0] * sqrt0) / sqlen0,
        ];
        let sol1 = [
            (end1[0] * sqrad - sign * radius * end1[1] * sqrt1) / sqlen1,
            (end1[1] * sqrad + sign * radius * end1[0] * sqrt1) / sqlen1,
        ];
        let good = if let Some(side) = side {
            let mid = [sol0[0] + sol1[0], sol0[1] + sol1[1]];
            let mid_len = (mid[0] * mid[0] + mid[1] * mid[1]).sqrt();
            if mid_len == 0.0 {
                f32::NEG_INFINITY
            } else {
                (mid[0] / mid_len) * side[0] + (mid[1] / mid_len) * side[1]
            }
        } else {
            let chord = [sol0[0] - sol1[0], sol0[1] - sol1[1]];
            -(chord[0] * chord[0] + chord[1] * chord[1])
        };
        if is_intersect_2d(end0, sol0, end1, sol1) {
            continue;
        }
        if good > best_good {
            best = i;
            best_good = good;
        }
    }
    if best_good == f32::NEG_INFINITY {
        return None;
    }
    let sign = if best == 0 { 1.0 } else { -1.0 };
    let sol0 = [
        (end0[0] * sqrad + sign * radius * end0[1] * sqrt0) / sqlen0,
        (end0[1] * sqrad - sign * radius * end0[0] * sqrt0) / sqlen0,
    ];
    let sol1 = [
        (end1[0] * sqrad - sign * radius * end1[1] * sqrt1) / sqlen1,
        (end1[1] * sqrad + sign * radius * end1[0] * sqrt1) / sqlen1,
    ];
    Some(([sol0, sol1], circle_arc_length(sol0, sol1, best, radius)))
}

fn cylinder_basis(axis: Vec3) -> (Vec3, Vec3) {
    let reference = if axis.x.abs() < 0.8 { Vec3::X } else { Vec3::Y };
    let first = axis.cross(reference).normalize();
    (first, axis.cross(first).normalize())
}

fn cylinder_segment_contribution(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    a: &SiteJacobian,
    b: &SiteJacobian,
    wrap: &WrapCylinder,
    jacobian: &mut [f32],
) -> f32 {
    let center = wrap_center_world(poses, wrap.link, wrap.center_local);
    let axis = match wrap.link {
        Some(link) => poses[link].1.rotate(wrap.axis_local).normalize(),
        None => wrap.axis_local.normalize(),
    };
    let (basis0, basis1) = cylinder_basis(axis);
    let pa = a.world_pos - center;
    let pb = b.world_pos - center;
    let end0 = [pa.dot(basis0), pa.dot(basis1)];
    let end1 = [pb.dot(basis0), pb.dot(basis1)];
    let side = wrap.sidesite.map(|s| {
        let p = site_position_and_jacobian(tree, poses, &s).world_pos - center;
        let radial = [p.dot(basis0), p.dot(basis1)];
        let n = (radial[0] * radial[0] + radial[1] * radial[1]).sqrt();
        if n == 0.0 {
            [0.0, 0.0]
        } else {
            [wrap.radius * radial[0] / n, wrap.radius * radial[1] / n]
        }
    });
    let Some(([ta2, tb2], arc)) = circle_wrap_points(end0, end1, side, wrap.radius) else {
        return straight_segment_contribution(a, b, jacobian);
    };
    let radial_a =
        ((end0[0] - ta2[0]) * (end0[0] - ta2[0]) + (end0[1] - ta2[1]) * (end0[1] - ta2[1])).sqrt();
    let radial_b =
        ((end1[0] - tb2[0]) * (end1[0] - tb2[0]) + (end1[1] - tb2[1]) * (end1[1] - tb2[1])).sqrt();
    let total_2d = radial_a + arc + radial_b;
    if total_2d == 0.0 {
        return straight_segment_contribution(a, b, jacobian);
    }
    let za = pa.dot(axis);
    let zb = pb.dot(axis);
    let zta = za + (zb - za) * radial_a / total_2d;
    let ztb = za + (zb - za) * (radial_a + arc) / total_2d;
    let ta = center + basis0 * ta2[0] + basis1 * ta2[1] + axis * zta;
    let tb = center + basis0 * tb2[0] + basis1 * tb2[1] + axis * ztb;
    let u_a = (ta - a.world_pos).normalize();
    let u_b = (tb - b.world_pos).normalize();
    for &(slot, col) in &a.columns {
        jacobian[slot as usize] -= u_a.dot(col);
    }
    for &(slot, col) in &b.columns {
        jacobian[slot as usize] -= u_b.dot(col);
    }
    add_wrap_body_jacobian(
        tree,
        poses,
        WrapForceData {
            link: wrap.link,
            center,
            point_a: ta,
            point_b: tb,
            force_a: u_a,
            force_b: u_b,
        },
        jacobian,
    );
    let central = (arc * arc + (ztb - zta) * (ztb - zta)).sqrt();
    (ta - a.world_pos).length() + central + (b.world_pos - tb).length()
}

/// Wrap segment length + Jacobian. Returns straight length (with straight
/// Jacobian contribution) when the segment doesn't actually intersect the
/// sphere.
fn wrap_segment_contribution(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    a: &SiteJacobian,
    b: &SiteJacobian,
    c: Vec3,
    wrap: &WrapSphere,
    jacobian: &mut [f32],
) -> f32 {
    let r = wrap.radius;
    let r2 = r * r;
    let d_ac = a.world_pos - c;
    let d_bc = b.world_pos - c;
    let d_a2 = d_ac.length_squared();
    let d_b2 = d_bc.length_squared();
    // If either endpoint is inside the sphere, wrap geometry is degenerate.
    // Fall back to straight — the model is ill-posed and the user gets a
    // physically-sensible tendon anyway (documented behavior in
    // docs/tendons.md). No panic.
    if d_a2 <= r2 || d_b2 <= r2 {
        return straight_segment_contribution(a, b, jacobian);
    }
    // Closest point on line AB to C. Parameterize line as A + t·(B - A).
    let ab = b.world_pos - a.world_pos;
    let ab_len2 = ab.length_squared();
    if ab_len2 == 0.0 {
        return 0.0;
    }
    let t_closest = (-(a.world_pos - c)).dot(ab) / ab_len2;
    // Point on the segment (or its extension) nearest to C.
    let p_closest = a.world_pos + ab * t_closest;
    let perp_sq = (p_closest - c).length_squared();
    let inside_segment = (0.0..=1.0).contains(&t_closest);
    // Wrap engages only when the closest point falls WITHIN the segment
    // and inside the sphere. Line intersections outside the segment mean
    // the endpoints "see" past the sphere without touching it.
    if !inside_segment || perp_sq >= r2 {
        return straight_segment_contribution(a, b, jacobian);
    }
    // Plane of wrap: spanned by CA and CB. Normal:
    let n_plane_raw = d_ac.cross(d_bc);
    let n_plane_len = n_plane_raw.length();
    if n_plane_len < SIDE_HINT_COLINEARITY_EPS {
        // A, B, C are (nearly) colinear. Sidesite gives the plane normal.
        let hint = wrap.side_hint_world.unwrap_or(Vec3::ZERO);
        if hint.length_squared() < SIDE_HINT_COLINEARITY_EPS * SIDE_HINT_COLINEARITY_EPS {
            // No usable hint; skip wrap (documented behavior).
            return straight_segment_contribution(a, b, jacobian);
        }
        // Project hint into plane perpendicular to AB, then build a wrap
        // plane containing AB and pointing toward the hint.
        let ab_hat = ab / ab.length();
        let hint_perp = hint - ab_hat * hint.dot(ab_hat);
        if hint_perp.length_squared() < SIDE_HINT_COLINEARITY_EPS * SIDE_HINT_COLINEARITY_EPS {
            return straight_segment_contribution(a, b, jacobian);
        }
        // Wrap geometry with the hint as the in-plane "up" direction.
        return wrap_with_perp_hint(a, b, c, r, hint_perp.normalize(), jacobian);
    }
    // Non-colinear: use the natural wrap plane.
    let d_a = d_a2.sqrt();
    let d_b = d_b2.sqrt();
    let t_a = (d_a2 - r2).sqrt();
    let t_b = (d_b2 - r2).sqrt();
    // Angles at C. phi_A = arccos(R / d_A); phi_B likewise. Compute via asin
    // for better accuracy near R ≈ d.
    let phi_a = asin(t_a / d_a); // asin(√(1 - (R/d_A)²)) = arccos(R/d_A)
    let phi_b = asin(t_b / d_b);
    // Interior angle ACB.
    // cos γ = (CA · CB) / (d_A · d_B)
    let cos_gamma_raw = d_ac.dot(d_bc) / (d_a * d_b);
    let cos_gamma = cos_gamma_raw.clamp(-1.0, 1.0);
    let sin_gamma = (1.0 - cos_gamma * cos_gamma).sqrt();
    let gamma = atan2(sin_gamma, cos_gamma);
    let arc_angle = (gamma - phi_a - phi_b).max(0.0);
    let seg_len = t_a + r * arc_angle + t_b;

    // Tangent point from A: T_A = C + (R/d_A)² · (A-C) + (R·t_A/d_A²) · perp_in_plane_toward_B
    // where perp_in_plane_toward_B is the in-plane unit vector perpendicular
    // to (A-C) and on the side of B.
    let x_a_hat = d_ac / d_a; // unit from C to A, in plane
    // In-plane perpendicular to x_a_hat, in the plane (CA, CB), pointing
    // toward B. Take out the component of CB along CA:
    let cb_perp = d_bc - x_a_hat * (d_bc.dot(x_a_hat));
    let cb_perp_len = cb_perp.length();
    if cb_perp_len == 0.0 {
        return straight_segment_contribution(a, b, jacobian);
    }
    let y_a_hat = cb_perp / cb_perp_len; // unit toward B, in plane
    let t_a_point = c + x_a_hat * (r2 / d_a) + y_a_hat * (r * t_a / d_a);
    // Symmetric on the B side: x_b_hat points from C toward B, and the
    // in-plane perpendicular points back toward A.
    let x_b_hat = d_bc / d_b;
    let ca_perp = d_ac - x_b_hat * (d_ac.dot(x_b_hat));
    let ca_perp_len = ca_perp.length();
    if ca_perp_len == 0.0 {
        return straight_segment_contribution(a, b, jacobian);
    }
    let y_b_hat = ca_perp / ca_perp_len;
    let t_b_point = c + x_b_hat * (r2 / d_b) + y_b_hat * (r * t_b / d_b);
    // Envelope theorem: dL/dA_world = −(unit from A to T_A), dL/dB_world =
    // −(unit from B to T_B). The wrap arc + sphere-center contributions
    // vanish under the envelope theorem (T_A, T_B are constrained to the
    // sphere, and C's motion is a stationary point of L w.r.t. wrap
    // geometry). The remaining explicit dependence on C would kick in
    // only if the sphere is body-attached AND the body moves — for the
    // scope of this tier the wrap sphere is either world-fixed or
    // rigidly attached to a link that a spatial tendon doesn't move via
    // its OWN forces (documented; verified in the wrap-arc test that
    // uses a static sphere). Sphere motion contribution deferred to a
    // later tier.
    let u_a = (t_a_point - a.world_pos).normalize();
    let u_b = (t_b_point - b.world_pos).normalize();
    for &(slot, col) in &a.columns {
        jacobian[slot as usize] -= u_a.dot(col);
    }
    for &(slot, col) in &b.columns {
        jacobian[slot as usize] -= u_b.dot(col);
    }
    add_wrap_body_jacobian(
        tree,
        poses,
        WrapForceData {
            link: wrap.link,
            center: c,
            point_a: t_a_point,
            point_b: t_b_point,
            force_a: u_a,
            force_b: u_b,
        },
        jacobian,
    );
    seg_len
}

struct WrapForceData {
    link: Option<usize>,
    center: Vec3,
    point_a: Vec3,
    point_b: Vec3,
    force_a: Vec3,
    force_b: Vec3,
}

fn add_wrap_body_jacobian(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    data: WrapForceData,
    jacobian: &mut [f32],
) {
    let Some(link) = data.link else { return };
    let center_jac = wrap_point_jacobian(tree, poses, Some(link), data.center);
    let total_force = data.force_a + data.force_b;
    for &(slot, col) in &center_jac.columns {
        jacobian[slot as usize] += total_force.dot(col);
    }
    let point_a_jac = wrap_point_jacobian(tree, poses, Some(link), data.point_a);
    for &(slot, col) in &point_a_jac.columns {
        let center_col = center_jac
            .columns
            .iter()
            .find(|(center_slot, _)| *center_slot == slot)
            .map(|(_, center_col)| *center_col)
            .unwrap_or(Vec3::ZERO);
        jacobian[slot as usize] += data.force_a.dot(col - center_col);
    }
    let point_b_jac = wrap_point_jacobian(tree, poses, Some(link), data.point_b);
    for &(slot, col) in &point_b_jac.columns {
        let center_col = center_jac
            .columns
            .iter()
            .find(|(center_slot, _)| *center_slot == slot)
            .map(|(_, center_col)| *center_col)
            .unwrap_or(Vec3::ZERO);
        jacobian[slot as usize] += data.force_b.dot(col - center_col);
    }
}

/// Colinear-fallback branch of wrap_segment_contribution when the sphere
/// center is on the AB line and a side hint is supplied. Builds an
/// artificial plane so the tangent geometry is well-defined.
fn wrap_with_perp_hint(
    a: &SiteJacobian,
    b: &SiteJacobian,
    c: Vec3,
    r: f32,
    y_hint: Vec3,
    jacobian: &mut [f32],
) -> f32 {
    // Perp-line construction: pretend A' = A + ε·y_hint slightly off axis,
    // then recompute. Since we can't perturb positions, we compute the
    // wrap in the plane containing AB and y_hint directly.
    let d_ac = a.world_pos - c;
    let d_bc = b.world_pos - c;
    // Distances collapse to the axial component when A, B, C are colinear.
    let d_a = d_ac.length();
    let d_b = d_bc.length();
    let t_a = (d_a * d_a - r * r).sqrt();
    let t_b = (d_b * d_b - r * r).sqrt();
    // Symmetric wrap over the sphere with the hint direction as "up":
    // arc angle from CT_A to CT_B, both in the plane spanned by CA and
    // y_hint. Since A and B lie on opposite sides of C on the axis (in
    // the colinear case), the wrap is a half-circle less the two
    // tangent-half-angles from A and B.
    let phi_a = asin(t_a / d_a);
    let phi_b = asin(t_b / d_b);
    let arc_angle = (crate::math::PI - phi_a - phi_b).max(0.0);
    let seg_len = t_a + r * arc_angle + t_b;
    // Tangent points along the hint direction.
    let x_a_hat = d_ac / d_a;
    let t_a_point = c + x_a_hat * (r * r / d_a) + y_hint * (r * t_a / d_a);
    let x_b_hat = d_bc / d_b;
    let t_b_point = c + x_b_hat * (r * r / d_b) + y_hint * (r * t_b / d_b);
    let u_a = (t_a_point - a.world_pos).normalize();
    let u_b = (t_b_point - b.world_pos).normalize();
    for &(slot, col) in &a.columns {
        jacobian[slot as usize] -= u_a.dot(col);
    }
    for &(slot, col) in &b.columns {
        jacobian[slot as usize] -= u_b.dot(col);
    }
    seg_len
}

// ---------------------------------------------------------------------------
// Passive + actuator qfrc contribution (consumed inside ABA)
// ---------------------------------------------------------------------------

/// Per-tendon cached kinematics: length, velocity, Jacobian.
/// Threaded between the "compute" step (once per aba call) and the
/// consumer (pass-2 tau assembly + actuator dispatch).
pub struct TendonState {
    pub kinematics: Vec<TendonKinematics>,
    pub forces: Vec<f32>,
}

/// Compute every tendon's kinematics AND accumulate passive spring/damper
/// generalized forces into `qfrc`. Actuator-on-tendon contributions are
/// handled separately by [`accumulate_tendon_actuator_qfrc`] because the
/// actuator loop already lives inside `aba`'s pass-2 branch. Splitting
/// keeps that branch's per-link scan intact while feeding the necessary
/// tendon state through one shared cache.
pub fn accumulate_tendon_passive(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    qfrc: &mut [f32],
) -> TendonState {
    let n = tree.tendons.len();
    let mut kinematics = Vec::with_capacity(n);
    let mut forces = Vec::with_capacity(n);
    for tendon in &tree.tendons {
        let kin = tendon_kinematics(tendon, tree, poses);
        // Passive scalar force at the tendon:
        //   F = -k · (L - L0) - c · Ldot   (when springlength given)
        // Jᵀ · F distributes to qfrc slots.
        let mut f_pass = 0.0f32;
        if let (Some(l0), true) = (tendon.springlength, tendon.stiffness > 0.0) {
            f_pass += -tendon.stiffness * (kin.length - l0);
        }
        if tendon.damping > 0.0 {
            f_pass += -tendon.damping * kin.velocity;
        }
        if f_pass != 0.0 {
            for (i, &c) in kin.jacobian.iter().enumerate() {
                if c != 0.0 {
                    qfrc[i] += c * f_pass;
                }
            }
        }
        forces.push(f_pass);
        kinematics.push(kin);
    }
    TendonState { kinematics, forces }
}

/// Accumulate per-tendon actuator force into qfrc. Called by ABA after
/// [`accumulate_tendon_passive`] has computed the cached kinematics.
///
/// For a tendon actuator, the transmission-space length/velocity are
/// `(gear · L, gear · Ldot)` — same convention as joint actuators. The
/// resulting scalar force is distributed via `Jᵀ · F`.
pub fn accumulate_tendon_actuator_qfrc(tree: &Tree, state: &mut TendonState, qfrc: &mut [f32]) {
    for act in &tree.actuators {
        if let Some(tid) = act.tendon_target {
            if tid >= state.kinematics.len() {
                continue;
            }
            let kin = &state.kinematics[tid];
            // Actuator sees (len, vel) at the tendon.
            let torque = act.torque(kin.length, kin.velocity);
            state.forces[tid] += torque;
            if torque != 0.0 {
                for (i, &c) in kin.jacobian.iter().enumerate() {
                    if c != 0.0 {
                        qfrc[i] += c * torque;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::JointKind;
    use crate::math::FRAC_PI_2;
    use crate::tree::{Link, forward_kinematics};

    fn approx(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol {tol})");
    }

    // Helper: single-branch tree with N hinges chained in +z direction.
    // Root Fixed, then N hinges about the given axis, each with unit
    // inertia and unit length to the next joint.
    fn make_hinge_chain(n: usize, axis: Vec3, arm: f32) -> Tree {
        use crate::math::Mat3;
        let mut t = Tree::new();
        t.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        for _ in 0..n {
            let parent = t.links.len() - 1;
            t.push_link(Link::new(
                Some(parent),
                JointKind::hinge(axis),
                (Vec3::ZERO, Quat::IDENTITY),
                (Vec3::new(0.0, 0.0, arm), Quat::IDENTITY),
                1.0,
                Mat3::diag(1e-3, 1e-3, 1e-3),
            ));
        }
        t
    }

    #[test]
    fn fixed_tendon_length_and_jacobian_matches_hand_computation() {
        // Chain: fixed root, hinge, hinge. Tendon = coef1·q1 + coef2·q2.
        let mut t = make_hinge_chain(2, Vec3::X, 1.0);
        t.set_hinge_angle(1, 0.3);
        t.set_hinge_angle(2, -0.4);
        t.set_hinge_rate(1, 0.7);
        t.set_hinge_rate(2, -0.5);
        let tendon = Tendon::fixed(vec![
            FixedTendonJoint { link: 1, coef: 2.0 },
            FixedTendonJoint {
                link: 2,
                coef: -1.5,
            },
        ]);
        let poses = forward_kinematics(&t);
        let kin = tendon_kinematics(&tendon, &t, &poses);
        // L = 2*0.3 + (-1.5)*(-0.4) = 0.6 + 0.6 = 1.2
        approx(kin.length, 1.2, 1e-6);
        // Ldot = 2*0.7 + (-1.5)*(-0.5) = 1.4 + 0.75 = 2.15
        approx(kin.velocity, 2.15, 1e-6);
        // Jacobian entries at v_offset[1], v_offset[2] equal coefs.
        approx(kin.jacobian[t.v_offset[1]], 2.0, 1e-6);
        approx(kin.jacobian[t.v_offset[2]], -1.5, 1e-6);
    }

    #[test]
    fn spatial_tendon_straight_length_matches_euclidean_distance() {
        // Root fixed at origin, one slide link along +z. Site 1 at world
        // origin (on the fixed root), site 2 at (0, 0, +1) on the slide.
        // At slide=0, both sites coincide (dist=1 due to link offset).
        // Compute for hand-picked value.
        use crate::math::Mat3;
        let mut t = Tree::new();
        t.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        t.push_link(Link::new(
            Some(0),
            JointKind::slide(Vec3::Z),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        // Site 1 is on the root at (2, 0, 0), site 2 is on the slide link
        // at its origin. When slide=0, distance = 2. When slide=3, the
        // slide link COM sits at (0, 0, 3) so distance = √(2² + 3²) = √13.
        let tendon = Tendon::spatial(
            vec![
                SpatialTendonSite {
                    link: Some(0),
                    position_local: Vec3::new(2.0, 0.0, 0.0),
                },
                SpatialTendonSite {
                    link: Some(1),
                    position_local: Vec3::ZERO,
                },
            ],
            vec![None],
        );
        t.set_slide_position(1, 3.0);
        let poses = forward_kinematics(&t);
        let kin = tendon_kinematics(&tendon, &t, &poses);
        approx(kin.length, (2.0f32 * 2.0 + 3.0 * 3.0).sqrt(), 1e-5);
        // Jacobian at slide slot: dL/dq_slide = (r_z / L) where
        // r_z = z-distance from site1 to site2 = 3.
        // Verify: dL/dq = r · (dr/dq) / |r| where r = p_site2 - p_site1.
        // dp_site2/dq_slide = (0, 0, 1); dp_site1/dq_slide = 0.
        // So dL/dq = r_z / L = 3 / √13.
        let expected = 3.0f32 / (13.0f32).sqrt();
        approx(kin.jacobian[t.v_offset[1]], expected, 1e-5);
    }

    #[test]
    fn sphere_wrap_grazing_matches_straight_length() {
        // Sphere positioned so line AB grazes it. Length should equal |AB|.
        // A = (-2, 0, 0), B = (2, 0, 0), C = (0, R, 0), R = 0.5.
        // perp_dist from C to line AB = R (grazing). Choose R exactly
        // equal to distance so wrap doesn't engage (strict < check).
        use crate::math::Mat3;
        let mut t = Tree::new();
        t.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        let sites = vec![
            SpatialTendonSite {
                link: Some(0),
                position_local: Vec3::new(-2.0, 0.0, 0.0),
            },
            SpatialTendonSite {
                link: Some(0),
                position_local: Vec3::new(2.0, 0.0, 0.0),
            },
        ];
        let wrap = WrapSphere {
            link: Some(0),
            center_local: Vec3::new(0.0, 0.5, 0.0),
            radius: 0.5,
            side_hint_world: None,
        };
        let tendon = Tendon::spatial(sites, vec![Some(wrap)]);
        let poses = forward_kinematics(&t);
        let kin = tendon_kinematics(&tendon, &t, &poses);
        approx(kin.length, 4.0, 1e-4);
    }

    #[test]
    fn sphere_wrap_hand_derived_length() {
        // A = (-2, 0, 0), B = (2, 0, 0), sphere at C = (0, 0.3, 0), R = 0.4.
        // perp dist = 0.3 < R = 0.4 → wrap engages.
        // d_A = d_B = √(4 + 0.09) = √4.09 ≈ 2.0224
        // t_A = t_B = √(4.09 - 0.16) = √3.93 ≈ 1.9824
        // phi_A = asin(t_A / d_A) = asin(1.9824 / 2.0224) = asin(0.9802) ≈ 1.3707
        // cos gamma = (CA·CB) / (d_A·d_B) = ((-2)(2) + (-0.3)(-0.3)) / (2.0224²)
        //           = (-4 + 0.09) / 4.09 = -3.91 / 4.09 = -0.9560
        // gamma = arccos(-0.9560) ≈ 2.844
        // arc_angle = gamma - 2*phi_A = 2.844 - 2*1.3707 = 0.1026
        // L = 2*t_A + R*arc_angle = 2*1.9824 + 0.4*0.1026 = 3.9648 + 0.0410 = 4.0058
        use crate::math::Mat3;
        let mut t = Tree::new();
        t.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        let sites = vec![
            SpatialTendonSite {
                link: Some(0),
                position_local: Vec3::new(-2.0, 0.0, 0.0),
            },
            SpatialTendonSite {
                link: Some(0),
                position_local: Vec3::new(2.0, 0.0, 0.0),
            },
        ];
        let wrap = WrapSphere {
            link: Some(0),
            center_local: Vec3::new(0.0, 0.3, 0.0),
            radius: 0.4,
            side_hint_world: None,
        };
        let tendon = Tendon::spatial(sites, vec![Some(wrap)]);
        let poses = forward_kinematics(&t);
        let kin = tendon_kinematics(&tendon, &t, &poses);
        let expected = 4.0058_f32;
        approx(kin.length, expected, 5e-3);
    }

    #[test]
    fn wrap_engage_transition_length_continuous() {
        // Sweep sphere y-offset across the engage boundary; length should
        // vary continuously (no discontinuous jump at engagement).
        use crate::math::Mat3;
        let mut t = Tree::new();
        t.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        let sites = |y: f32| {
            (
                vec![
                    SpatialTendonSite {
                        link: Some(0),
                        position_local: Vec3::new(-2.0, 0.0, 0.0),
                    },
                    SpatialTendonSite {
                        link: Some(0),
                        position_local: Vec3::new(2.0, 0.0, 0.0),
                    },
                ],
                Some(WrapSphere {
                    link: Some(0),
                    center_local: Vec3::new(0.0, y, 0.0),
                    radius: 0.4,
                    side_hint_world: None,
                }),
            )
        };
        // Just outside engagement (y = 0.400001 → perp = 0.400001 > 0.4 →
        // straight).
        let (s1, w1) = sites(0.4001);
        let tendon1 = Tendon::spatial(s1, vec![w1]);
        let poses = forward_kinematics(&t);
        let len_straight = tendon_kinematics(&tendon1, &t, &poses).length;
        // Just inside engagement.
        let (s2, w2) = sites(0.3999);
        let tendon2 = Tendon::spatial(s2, vec![w2]);
        let len_wrap = tendon_kinematics(&tendon2, &t, &poses).length;
        // Both should be very close to 4.0 (straight line) — wrap-engaged
        // length exceeds straight by a tiny amount and continuity means the
        // gap should be O((y - R)^{3/2}) not O(1).
        assert!(
            (len_wrap - len_straight).abs() < 1e-3,
            "wrap engage jumped: {len_straight} → {len_wrap}"
        );
        assert!(
            len_wrap >= 4.0 - 1e-5 && len_straight >= 4.0 - 1e-5,
            "engaged and straight paths must respect the endpoint lower bound: straight={len_straight}, wrap={len_wrap}"
        );
    }

    #[test]
    fn validate_rejects_bad_configs() {
        // Fixed tendon on ball joint.
        use crate::math::Mat3;
        let mut t = Tree::new();
        t.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        t.push_link(Link::new(
            Some(0),
            JointKind::ball(),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
            1.0,
            Mat3::diag(1e-3, 1e-3, 1e-3),
        ));
        let bad = Tendon::fixed(vec![FixedTendonJoint { link: 1, coef: 1.0 }]);
        assert!(bad.validate(&t).is_err());
        // Fixed tendon on out-of-range link.
        let bad2 = Tendon::fixed(vec![FixedTendonJoint {
            link: 42,
            coef: 1.0,
        }]);
        assert!(bad2.validate(&t).is_err());
        // Negative stiffness.
        let mut bad3 = Tendon::fixed(vec![FixedTendonJoint { link: 1, coef: 1.0 }]);
        bad3.stiffness = -1.0;
        // Actually link 1 is ball → error before stiffness check; use fresh good tendon.
        // (Test negative stiffness on a spatial tendon with sensible sites.)
        let mut bad_stiff = Tendon::spatial(
            vec![
                SpatialTendonSite {
                    link: Some(0),
                    position_local: Vec3::ZERO,
                },
                SpatialTendonSite {
                    link: Some(0),
                    position_local: Vec3::X,
                },
            ],
            vec![None],
        );
        bad_stiff.stiffness = -0.5;
        assert!(bad_stiff.validate(&t).is_err());
    }

    #[test]
    fn hinge_length_matches_endpoint_arc() {
        // Single hinge about x with the joint anchor at (0, 0, 1) in the
        // child body. At q=0, the child COM sits at (0, 0, -1) world; the
        // tip site at local (0, 0, -1) is at world (0, 0, -2). Fixed site
        // on the root at (0, 0, 1) world.
        //
        // At q = π/2, the child body rotates about x. Rot_x(π/2) maps
        //   (0, 0, 1)  → (0, -1, 0)   (joint anchor)
        //   (0, 0, -1) → (0, 1, 0)    (tip site's rotated local offset)
        // Child COM = joint_world - Rot_x(π/2)·(0,0,1) = (0,0,0) - (0,-1,0)
        //           = (0, 1, 0).
        // Tip site world = COM + Rot_x(π/2)·(0,0,-1) = (0,1,0) + (0,1,0)
        //           = (0, 2, 0).
        // Distance from (0, 0, 1) to (0, 2, 0) = √(4 + 1) = √5.
        use crate::math::Mat3;
        let mut t = Tree::new();
        t.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        t.push_link(Link::new(
            Some(0),
            JointKind::hinge(Vec3::X),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
            1.0,
            Mat3::diag(1e-3, 1e-3, 1e-3),
        ));
        let tendon = Tendon::spatial(
            vec![
                SpatialTendonSite {
                    link: Some(0),
                    position_local: Vec3::new(0.0, 0.0, 1.0),
                },
                SpatialTendonSite {
                    link: Some(1),
                    position_local: Vec3::new(0.0, 0.0, -1.0),
                },
            ],
            vec![None],
        );
        t.set_hinge_angle(1, FRAC_PI_2);
        let poses = forward_kinematics(&t);
        let kin = tendon_kinematics(&tendon, &t, &poses);
        approx(kin.length, (5.0f32).sqrt(), 1e-4);
    }
}
