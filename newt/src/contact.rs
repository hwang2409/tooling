//! Narrow-phase collision detection.
//!
//! # Contact representation
//!
//! A [`Contact`] describes one contact point between two geoms. The normal
//! points **from geom B into geom A**: an infinitesimal displacement of A
//! along `+normal` separates the pair. `penetration` is positive when the
//! geoms interpenetrate (or the pair sits inside a nonzero margin — see
//! below); a negative penetration is never emitted. Plane-convex routes also
//! emit the exact-margin equality row, matching MuJoCo.
//!
//! # Narrow-phase coverage
//!
//! - Fully implemented pairs (tier 2 + v1 tier 2):
//!   - plane vs {sphere, box, capsule, cylinder, ellipsoid, mesh}
//!   - sphere vs {sphere, capsule, cylinder, ellipsoid, mesh}
//!   - hfield vs {sphere, capsule, box}; box-hfield uses native-style GJK/EPA
//!   - capsule vs capsule
//!   - box vs box (full OBB SAT with edge-edge cross axes — closes the
//!     NEWT-5 rotated-stack incident)
//!   - enabled non-hfield convex CCD pairs use deterministic GJK plus EPA
//!
//! MuJoCo 3.11.0 routes sphere-{ellipsoid,mesh} through `mjc_Convex` and
//! plane-{ellipsoid,mesh} through `mjc_PlaneConvex`. Newt matches these routes
//! with convex support points and load-bearing probes in `contacts_geoms_v1.rs`.
//! - Deferred pairs (silently emit no contact + [`is_pair_supported`]
//!   returns `false` so the world validator can reject them):
//!   - box vs {sphere, capsule}  (unchanged from tier 2)
//!   - box vs mesh
//!   - capsule vs {ellipsoid, cylinder, mesh}
//!   - ellipsoid vs {ellipsoid, cylinder, box, mesh}
//!   - cylinder vs {cylinder, box, mesh}
//!   - hfield vs {cylinder, ellipsoid, mesh}
//!
//! Deferred pairs are called out in the docs instead of silently producing
//! empty contact buffers. Box-mesh keeps an oracle probe but fails its normal
//! tier.
//!
//! # Margin / gap semantics
//!
//! Pair margin `M = max(a.margin, b.margin)` widens the activation zone: a
//! contact is emitted whenever the raw signed distance is below `M`, and the
//! reported `penetration` is the shifted quantity `M - dist` (so `penetration
//! > 0` even during near-miss detection). Pair gap `G = max(a.gap, b.gap)`
//! is stored on the emitted contact so the world can zero the force while
//! `penetration ≤ G`. Both default to zero (tier-2 behavior).
//!
//! # Determinism
//!
//! Each function returns a fixed number of contacts in a fixed order for a
//! given input; no HashMap iteration; support and simplex ties use geometric
//! keys, while pair order uses stable geom ids. Any iterative closest-point
//! solver runs a FIXED number of iterations (see [`sphere_ellipsoid`]). Sorted
//! results at the pair level live in [`crate::world`].

use crate::geom::{ConvexMesh, Geom, GeomPose, GeomShape, HeightField};
use crate::math::Vec3;

mod ccd;

#[cfg(test)]
use ccd::{CcdShape, CcdSimplex, CcdVertex, ccd_simplex_step};

fn lexicographically_precedes(a: Vec3, b: Vec3) -> bool {
    match a.x.total_cmp(&b.x) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => match a.y.total_cmp(&b.y) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => a.z.total_cmp(&b.z).is_lt(),
        },
    }
}

fn same_vec3(a: Vec3, b: Vec3) -> bool {
    a.x == b.x && a.y == b.y && a.z == b.z
}

/// One narrow-phase contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Index of geom A (world-registered index).
    pub geom_a: usize,
    /// Index of geom B.
    pub geom_b: usize,
    /// Contact point in world coordinates. Plane contacts use MuJoCo's
    /// midpoint between the two opposing surfaces. Other contacts use the
    /// primitive's surface anchor.
    pub position_world: Vec3,
    /// Unit normal in world coordinates, pointing FROM B into A.
    pub normal_world: Vec3,
    /// Shifted penetration depth `pair_margin - raw_dist` (positive; a
    /// value equal to `pair_margin` means the raw distance is exactly zero,
    /// i.e. the geoms just touch). MuJoCo plane colliders also emit the
    /// equality case; the solver skips a contact when its force-free gap
    /// includes it.
    pub penetration: f32,
    /// Pair Coulomb friction coefficient (see [`crate::geom`]).
    pub friction: f32,
    /// Pair force-free zone width `max(a.gap, b.gap)` (m). The world zeros
    /// the normal + friction force while `penetration ≤ gap`, so a nonzero
    /// gap makes the primitive fire contacts (useful for sensing) without
    /// applying force until they overlap more than `gap`. Zero by default.
    pub gap: f32,
}

/// Small stack-allocated buffer for per-pair contact output. Four is enough
/// for the shapes in this tier — a box-vs-plane emits at most four corner
/// contacts; capsule-vs-anything emits at most two endpoint contacts; sphere
/// pairs emit one; cylinder-vs-plane picks up to 4 deepest of 10 sampled
/// points; box-vs-box's full SAT emits up to 4 clipped points. We stash the
/// length inline to avoid heap allocation in the per-step hot path.
#[derive(Clone, Copy, Debug)]
pub struct ContactBuf {
    pub contacts: [Contact; 4],
    pub len: usize,
    /// Number of candidates offered to the bounded output buffer.
    pub candidate_count: usize,
    /// Number of unique candidates before the four-contact cap.
    pub unique_candidate_count: usize,
    seen: [Contact; 16],
    seen_len: usize,
}

impl Default for ContactBuf {
    fn default() -> Self {
        Self::new()
    }
}

impl ContactBuf {
    pub fn new() -> Self {
        let placeholder = Contact {
            geom_a: 0,
            geom_b: 0,
            position_world: Vec3::ZERO,
            normal_world: Vec3::Z,
            penetration: 0.0,
            friction: 0.0,
            gap: 0.0,
        };
        Self {
            contacts: [placeholder; 4],
            len: 0,
            candidate_count: 0,
            unique_candidate_count: 0,
            seen: [placeholder; 16],
            seen_len: 0,
        }
    }
    pub fn push(&mut self, c: Contact) {
        self.candidate_count += 1;
        if self.len < self.contacts.len() {
            self.contacts[self.len] = c;
            self.len += 1;
        }
    }

    /// Add a candidate while retaining the four deepest contacts in stable
    /// order. This is the per-pair manifold cap seam used by multi-cell
    /// heightfields.
    pub fn push_deepest(&mut self, c: Contact) {
        self.candidate_count += 1;
        if self.len < self.contacts.len() {
            self.contacts[self.len] = c;
            self.len += 1;
        } else if c.penetration > self.contacts[self.len - 1].penetration {
            self.contacts[self.len - 1] = c;
        } else {
            return;
        }
        let mut i = self.len - 1;
        while i > 0 && self.contacts[i].penetration > self.contacts[i - 1].penetration {
            self.contacts.swap(i, i - 1);
            i -= 1;
        }
    }

    /// Add a deepest candidate unless the same manifold point is already kept.
    /// Heightfield cells share edges, so this avoids spending a cap slot on a
    /// duplicate triangle contact.
    pub fn push_deepest_unique(&mut self, c: Contact) {
        self.candidate_count += 1;
        if (0..self.seen_len).any(|i| {
            let prior = self.seen[i];
            (c.position_world - prior.position_world).length_squared() < 1.0e-12
                && (c.normal_world - prior.normal_world).length_squared() < 1.0e-12
        }) {
            return;
        }
        if self.seen_len < self.seen.len() {
            self.seen[self.seen_len] = c;
            self.seen_len += 1;
        }
        self.unique_candidate_count += 1;
        if self.len < self.contacts.len() {
            self.contacts[self.len] = c;
            self.len += 1;
        } else if c.penetration > self.contacts[self.len - 1].penetration {
            self.contacts[self.len - 1] = c;
        } else {
            return;
        }
        let mut i = self.len - 1;
        while i > 0 && self.contacts[i].penetration > self.contacts[i - 1].penetration {
            self.contacts.swap(i, i - 1);
            i -= 1;
        }
    }
    pub fn as_slice(&self) -> &[Contact] {
        &self.contacts[..self.len]
    }
}

/// Combined friction rule for a pair: `min(μ_a, μ_b)`. Documented choice in
/// `docs/contacts.md`. Kept as a free function so tests can hit it directly.
pub fn combine_friction(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Pair combining rule for margin and gap: per-parameter maximum. Matches
/// MuJoCo's default (`solmix = 0.5` mean is not applied to margin/gap; the
/// max is the documented behavior). Zero-vs-zero collapses to zero.
pub fn combine_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Is the shape pair implemented by [`narrow_phase`]? Used by
/// [`crate::world::World::validate_supported_pairs`] to reject pair-list
/// entries whose combination lands in the "deferred" bucket. Symmetric.
pub fn is_pair_supported(a: GeomShape, b: GeomShape) -> bool {
    supported_shape_pair(&a, &b) || supported_shape_pair(&b, &a)
}

fn supported_shape_pair(a: &GeomShape, b: &GeomShape) -> bool {
    matches!(
        (a, b),
        (GeomShape::Sphere { .. }, GeomShape::Plane)
            | (GeomShape::Box { .. }, GeomShape::Plane)
            | (GeomShape::Capsule { .. }, GeomShape::Plane)
            | (GeomShape::Cylinder { .. }, GeomShape::Plane)
            | (GeomShape::Ellipsoid { .. }, GeomShape::Plane)
            | (GeomShape::Mesh { .. }, GeomShape::Plane)
            | (GeomShape::Sphere { .. }, GeomShape::Sphere { .. })
            | (GeomShape::Sphere { .. }, GeomShape::Capsule { .. })
            | (GeomShape::Sphere { .. }, GeomShape::Cylinder { .. })
            | (GeomShape::Sphere { .. }, GeomShape::Ellipsoid { .. })
            | (GeomShape::Sphere { .. }, GeomShape::Mesh { .. })
            | (GeomShape::Sphere { .. }, GeomShape::Hfield { .. })
            | (GeomShape::Capsule { .. }, GeomShape::Capsule { .. })
            | (GeomShape::Box { .. }, GeomShape::Box { .. })
            | (GeomShape::Box { .. }, GeomShape::Hfield { .. })
            | (GeomShape::Capsule { .. }, GeomShape::Hfield { .. })
            | (GeomShape::Mesh { .. }, GeomShape::Mesh { .. })
    )
}

/// World-frame outward normal and origin of a plane geom.
fn plane_world(plane_geom: &Geom, plane_pose: &GeomPose) -> (Vec3, Vec3) {
    // Local +Z is the outward normal by construction (see [`Geom::static_plane`]).
    let n = plane_pose.rotate(Vec3::Z);
    let _ = plane_geom; // shape checked by the caller
    (n, plane_pose.position)
}

/// Sphere vs static plane. Returns 0 or 1 contact.
#[allow(clippy::too_many_arguments)]
pub fn sphere_plane(
    idx_sphere: usize,
    sphere_pose: &GeomPose,
    radius: f32,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_plane: usize,
    plane_geom: &Geom,
    plane_pose: &GeomPose,
) -> ContactBuf {
    let mut out = ContactBuf::new();
    let (n, p0) = plane_world(plane_geom, plane_pose);
    let signed = (sphere_pose.position - p0).dot(n);
    // MuJoCo stores the raw signed surface distance and activates when it is
    // no greater than the pair margin.
    let raw_dist = signed - radius;
    if raw_dist <= margin {
        // MuJoCo places the contact at the midpoint of the sphere surface and
        // the plane, not on either surface.
        let contact_pt = sphere_pose.position - n * (radius + raw_dist * 0.5);
        out.push(Contact {
            geom_a: idx_sphere,
            geom_b: idx_plane,
            position_world: contact_pt,
            normal_world: n,
            penetration: margin - raw_dist,
            friction,
            gap,
        });
    }
    out
}

/// Box vs static plane, matching MuJoCo's `mjc_PlaneBox` collider.
///
/// The collider scans all eight corners in bit order, keeps corners whose
/// local plane-relative height is non-positive, skips corners outside the
/// margin, and retains the first four outputs in the bounded contact buffer.
/// It does not sort by depth. Contact positions are the midpoint between the
/// corner and the plane along the plane normal.
#[allow(clippy::too_many_arguments)]
pub fn box_plane(
    idx_box: usize,
    box_pose: &GeomPose,
    half_extents: Vec3,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_plane: usize,
    plane_geom: &Geom,
    plane_pose: &GeomPose,
) -> ContactBuf {
    let (n, p0) = plane_world(plane_geom, plane_pose);
    let mut out = ContactBuf::new();
    let dist = (box_pose.position - p0).dot(n);
    for i in 0..8 {
        let local = Vec3::new(
            if i & 1 == 0 {
                -half_extents.x
            } else {
                half_extents.x
            },
            if i & 2 == 0 {
                -half_extents.y
            } else {
                half_extents.y
            },
            if i & 4 == 0 {
                -half_extents.z
            } else {
                half_extents.z
            },
        );
        let corner_offset = box_pose.rotate(local);
        let ldist = n.dot(corner_offset);
        let raw_dist = dist + ldist;
        if raw_dist > margin || ldist > 0.0 {
            continue;
        }
        let corner = box_pose.position + corner_offset;
        let contact_pt = corner - n * (raw_dist * 0.5);
        out.push(Contact {
            geom_a: idx_box,
            geom_b: idx_plane,
            position_world: contact_pt,
            normal_world: n,
            penetration: margin - raw_dist,
            friction,
            gap,
        });
    }
    out
}

/// Capsule vs static plane, matching MuJoCo's endpoint order and midpoint
/// contact positions.
#[allow(clippy::too_many_arguments)]
pub fn capsule_plane(
    idx_capsule: usize,
    capsule_pose: &GeomPose,
    radius: f32,
    half_height: f32,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_plane: usize,
    plane_geom: &Geom,
    plane_pose: &GeomPose,
) -> ContactBuf {
    let (n, p0) = plane_world(plane_geom, plane_pose);
    let axis_world = capsule_pose.rotate(Vec3::Z);
    let ends = [
        capsule_pose.position + axis_world * half_height,
        capsule_pose.position - axis_world * half_height,
    ];
    let mut out = ContactBuf::new();
    for &e in ends.iter() {
        let signed = (e - p0).dot(n);
        let raw_dist = signed - radius;
        if raw_dist <= margin {
            let contact_pt = e - n * (radius + raw_dist * 0.5);
            out.push(Contact {
                geom_a: idx_capsule,
                geom_b: idx_plane,
                position_world: contact_pt,
                normal_world: n,
                penetration: margin - raw_dist,
                friction,
                gap,
            });
        }
    }
    out
}

/// Sphere vs sphere. 0 or 1 contact. Normal points from B into A.
#[allow(clippy::too_many_arguments)]
pub fn sphere_sphere(
    idx_a: usize,
    pose_a: &GeomPose,
    radius_a: f32,
    idx_b: usize,
    pose_b: &GeomPose,
    radius_b: f32,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    sphere_vs_sphere_at(
        idx_a,
        pose_a.position,
        radius_a,
        idx_b,
        pose_b.position,
        radius_b,
        friction,
        margin,
        gap,
    )
}

/// Sphere-vs-virtual-sphere helper; centers pre-resolved. Deterministic
/// tie-break: if the two centers coincide, normal defaults to `+Z`.
#[allow(clippy::too_many_arguments)]
fn sphere_vs_sphere_at(
    idx_a: usize,
    center_a: Vec3,
    radius_a: f32,
    idx_b: usize,
    center_b: Vec3,
    radius_b: f32,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let mut out = ContactBuf::new();
    let delta = center_a - center_b;
    let dist2 = delta.length_squared();
    let sum_r = radius_a + radius_b;
    let sum_r_margin = sum_r + margin;
    if dist2 >= sum_r_margin * sum_r_margin {
        return out;
    }
    let dist = dist2.sqrt();
    let n = if dist > 0.0 { delta / dist } else { Vec3::Z };
    let pen = sum_r_margin - dist;
    // Contact point on B's surface.
    let contact_pt = center_b + n * radius_b;
    out.push(Contact {
        geom_a: idx_a,
        geom_b: idx_b,
        position_world: contact_pt,
        normal_world: n,
        penetration: pen,
        friction,
        gap,
    });
    out
}

/// Sphere vs capsule. Reduces to sphere-vs-sphere at the capsule's closest-
/// point-on-axis.
#[allow(clippy::too_many_arguments)]
pub fn sphere_capsule(
    idx_sphere: usize,
    sphere_pose: &GeomPose,
    sphere_r: f32,
    idx_capsule: usize,
    capsule_pose: &GeomPose,
    capsule_r: f32,
    capsule_half_h: f32,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let axis = capsule_pose.rotate(Vec3::Z);
    let a = capsule_pose.position - axis * capsule_half_h;
    let b = capsule_pose.position + axis * capsule_half_h;
    let closest = closest_point_on_segment(sphere_pose.position, a, b);
    sphere_vs_sphere_at(
        idx_sphere,
        sphere_pose.position,
        sphere_r,
        idx_capsule,
        closest,
        capsule_r,
        friction,
        margin,
        gap,
    )
}

/// Box vs box: additive vertex-vs-face + edge-edge fallback.
///
/// - **Primary path (vertex-vs-face)** — every corner of A is tested against
///   B's interior and vice versa; a penetrating corner emits one contact
///   whose normal is the out-normal of the face nearest along the
///   center-to-center direction. Handles the axis-aligned + moderate-yaw
///   stacking regime bit-for-bit as it did in tier 2, which keeps the
///   `stacking_3_boxes` golden byte-identical.
/// - **Edge-edge fallback (v1 tier 2)** — when vertex-vs-face returns zero
///   contacts we fall through to a 9-axis edge-edge SAT test on the 3×3
///   cross products of A's and B's edge directions. If the boxes actually
///   overlap on every one of those axes AND every face-normal axis, we
///   emit contacts at the closest points on the pair of edges that produced
///   the minimum-overlap axis. This closes the NEWT-5 rotated-stack
///   incident: two boxes yawed ≥ 45° relative to each other still stack,
///   because when every corner of the upper hangs over an edge of the
///   lower, the actual intersection is edge-vs-edge, not vertex-vs-face.
///
/// The fallback is intentionally additive so existing goldens survive.
///
/// # Solver-mode variant
///
/// [`box_box_full_manifold`] skips the vertex-vs-face primary and always
/// runs the SAT face-clipping manifold. That path emits 4 corner contacts
/// for a face-face stack while vertex-vs-face degenerates to 2 diagonal
/// contacts as soon as the top block tilts even microradians and its
/// two lifted corners fail the "vertex inside the other box" test — the
/// root cause of the v1 box_stack differential finding
/// (docs/differential.md). The PGS and Newton pipelines route through the
/// full-manifold variant; the penalty pipeline continues to use
/// [`box_box`] so its trajectories stay bit-identical.
#[allow(clippy::too_many_arguments)]
pub fn box_box(
    idx_a: usize,
    pose_a: &GeomPose,
    half_a: Vec3,
    idx_b: usize,
    pose_b: &GeomPose,
    half_b: Vec3,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let vf = box_box_vertex_face(
        idx_a, pose_a, half_a, idx_b, pose_b, half_b, friction, margin, gap,
    );
    if vf.len > 0 {
        return vf;
    }
    box_box_sat_fallback(
        idx_a, pose_a, half_a, idx_b, pose_b, half_b, friction, margin, gap,
    )
}

/// Box vs box for the PGS solver pipeline: always run SAT face-clipping
/// (or edge-edge closest-points when SAT identifies an edge-edge minimum).
///
/// Skips the vertex-vs-face primary because for aligned face-face stacks
/// that path emits only the 2 corners still inside the other box after
/// microradian-scale tilt, leaving the friction moment underdetermined
/// (only diagonal contact points → the block rotates and slides). SAT
/// face-clipping emits up to 4 clipped-polygon corners of the actual
/// overlap rectangle regardless of tilt, matching what MuJoCo's MPR
/// produces for the same configuration. See docs/differential.md
/// (box_stack row, NEWT-14 evidence) for the per-step manifold diff
/// that drove this split.
#[allow(clippy::too_many_arguments)]
pub fn box_box_full_manifold(
    idx_a: usize,
    pose_a: &GeomPose,
    half_a: Vec3,
    idx_b: usize,
    pose_b: &GeomPose,
    half_b: Vec3,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    box_box_sat_fallback(
        idx_a, pose_a, half_a, idx_b, pose_b, half_b, friction, margin, gap,
    )
}

/// Vertex-vs-face primary path. See [`box_box`] for coverage notes.
#[allow(clippy::too_many_arguments)]
fn box_box_vertex_face(
    idx_a: usize,
    pose_a: &GeomPose,
    half_a: Vec3,
    idx_b: usize,
    pose_b: &GeomPose,
    half_b: Vec3,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    const CORNER_SIGNS: [(f32, f32, f32); 8] = [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ];

    // "Canonical" separation direction from B's center toward A's center, in
    // both boxes' local frames. Used to pick which face of the containing box
    // the contact belongs to — the "nearest face" rule alone chooses the
    // wrong face when the intruding vertex sits closer to the far wall of
    // the container, which is exactly the regime that stacking hits.
    let delta_world = pose_a.position - pose_b.position;
    let delta_in_b = pose_b.orientation.inverse_rotate(delta_world);
    let delta_in_a = pose_a.orientation.inverse_rotate(-delta_world);
    // Fallback direction when the two centers coincide.
    let fallback = Vec3::Z;
    let dir_b = if delta_in_b.length_squared() > 0.0 {
        delta_in_b
    } else {
        fallback
    };
    let dir_a = if delta_in_a.length_squared() > 0.0 {
        delta_in_a
    } else {
        -fallback
    };

    // Candidate contacts: (penetration, position_world, normal_world). Up to
    // 16 (8 vertices from each side); the caller keeps 4 deepest.
    let mut candidates: [(f32, Vec3, Vec3); 16] = [(0.0, Vec3::ZERO, Vec3::Z); 16];
    let mut count = 0usize;

    // A's vertices in B — normal is B's out-normal along `dir_b` (from B into A).
    // Project onto B's surface (contact convention: `position_world` sits on
    // B). Moving `world_v` by `+pen` along the out-normal takes it from
    // inside B onto B's face.
    for &(sx, sy, sz) in &CORNER_SIGNS {
        let local_a = Vec3::new(sx * half_a.x, sy * half_a.y, sz * half_a.z);
        let world_v = pose_a.point_to_world(local_a);
        let local_b = pose_b.orientation.inverse_rotate(world_v - pose_b.position);
        if let Some((pen, normal_local_b)) = face_along_direction(local_b, half_b, dir_b) {
            let normal_world = pose_b.rotate(normal_local_b);
            let position_on_b_surface = world_v + normal_world * pen;
            candidates[count] = (pen, position_on_b_surface, normal_world);
            count += 1;
        }
    }
    // B's vertices in A — face normal points OUT of A (that's from A into B),
    // flipped to satisfy the "from B into A" convention. Here `world_v` is
    // already a corner of B — it sits on B's surface — so it needs no
    // projection to obey the convention.
    for &(sx, sy, sz) in &CORNER_SIGNS {
        let local_b = Vec3::new(sx * half_b.x, sy * half_b.y, sz * half_b.z);
        let world_v = pose_b.point_to_world(local_b);
        let local_a = pose_a.orientation.inverse_rotate(world_v - pose_a.position);
        if let Some((pen, normal_local_a)) = face_along_direction(local_a, half_a, dir_a) {
            let normal_world = -pose_a.rotate(normal_local_a);
            candidates[count] = (pen, world_v, normal_world);
            count += 1;
        }
    }

    // Stable sort descending by penetration; ties by candidate index.
    let mut order: [usize; 16] = std::array::from_fn(|i| i);
    for i in 1..count {
        let mut j = i;
        while j > 0 && candidates[order[j]].0 > candidates[order[j - 1]].0 {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    let take = if count > 4 { 4 } else { count };
    let mut out = ContactBuf::new();
    for &i in &order[..take] {
        let (pen_raw, pos, normal) = candidates[i];
        // Shift by margin — pen_raw is the true overlap depth; the reported
        // penetration is `pen_raw + margin` per the module contract.
        out.push(Contact {
            geom_a: idx_a,
            geom_b: idx_b,
            position_world: pos,
            normal_world: normal,
            penetration: pen_raw + margin,
            friction,
            gap,
        });
    }
    out
}

/// SAT fallback for box-box. Runs when the vertex-vs-face manifold is empty.
///
/// Textbook 15-axis OBB SAT: 3 face-normals from A + 3 from B + 9 edge-edge
/// cross products. If any axis has negative overlap the boxes are separated.
/// Otherwise the axis with the MINIMUM overlap identifies the separation
/// direction and picks the manifold-generation strategy:
///
/// - **Face-A / Face-B minimum** — the boxes' penetration is dominated by
///   the perpendicular direction to that face. Emit contacts via reference-
///   face clipping: the winning box's face is the reference plane; the
///   opposing box's most-anti-parallel face is the incident polygon; the
///   incident polygon is Sutherland-Hodgman clipped against the reference
///   rectangle in 2D, and each surviving clipped vertex becomes a contact
///   (up to 4 deepest kept). Emits up to 4 contacts.
/// - **Edge-edge minimum** — the deepest overlap is between two skew edges.
///   Emit ONE contact at their closest-point pair.
///
/// The reference-face branch is what makes yawed stacks work correctly: two
/// boxes at 45° relative yaw have all corners hanging over their opposite's
/// face edges, so vertex-vs-face returns nothing and pure edge-edge would
/// emit a single low-quality contact off an oblique axis. Reference-face
/// clipping produces the correct 2–8 clipped intersection vertices.
#[allow(clippy::too_many_arguments)]
fn box_box_sat_fallback(
    idx_a: usize,
    pose_a: &GeomPose,
    half_a: Vec3,
    idx_b: usize,
    pose_b: &GeomPose,
    half_b: Vec3,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    // World-frame basis vectors of each box.
    let ax = [
        pose_a.rotate(Vec3::X),
        pose_a.rotate(Vec3::Y),
        pose_a.rotate(Vec3::Z),
    ];
    let bx = [
        pose_b.rotate(Vec3::X),
        pose_b.rotate(Vec3::Y),
        pose_b.rotate(Vec3::Z),
    ];
    let ha = [half_a.x, half_a.y, half_a.z];
    let hb = [half_b.x, half_b.y, half_b.z];
    let delta = pose_a.position - pose_b.position;

    // For any candidate axis L (unit vector), the projected half-extent of a
    // box with basis (u0, u1, u2) and half-extents (h0, h1, h2) is
    //   Σ h_i · |L · u_i|.
    // Overlap along L is `rA + rB - |delta · L|`. Since every axis we
    // consider is (or is normalized to) a unit vector, overlaps compare
    // fairly across face and edge-edge candidates.
    let proj_half = |l: Vec3, u: &[Vec3; 3], h: &[f32; 3]| -> f32 {
        h[0] * crate::math::abs(l.dot(u[0]))
            + h[1] * crate::math::abs(l.dot(u[1]))
            + h[2] * crate::math::abs(l.dot(u[2]))
    };

    // Track minimum overlap across ALL 15 axes, along with the axis
    // classification so the manifold generator downstream knows whether to
    // run reference-face clipping or edge-edge closest-points.
    #[derive(Clone, Copy)]
    enum WinningAxis {
        FaceA(usize),
        FaceB(usize),
        EdgeEdge(usize, usize),
    }
    let mut best_overlap = f32::INFINITY;
    let mut best_axis = Vec3::Z;
    let mut best_kind = WinningAxis::FaceA(0);

    // 3 face-normal axes from A.
    for (k, &face) in ax.iter().enumerate() {
        let ra = proj_half(face, &ax, &ha);
        let rb = proj_half(face, &bx, &hb);
        let signed = delta.dot(face);
        let overlap = ra + rb - crate::math::abs(signed);
        if overlap < -margin {
            return ContactBuf::new();
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            // Normal convention: points FROM B INTO A → same sign as `delta`.
            best_axis = if signed >= 0.0 { face } else { -face };
            best_kind = WinningAxis::FaceA(k);
        }
    }
    // 3 face-normal axes from B.
    for (k, &face) in bx.iter().enumerate() {
        let ra = proj_half(face, &ax, &ha);
        let rb = proj_half(face, &bx, &hb);
        let signed = delta.dot(face);
        let overlap = ra + rb - crate::math::abs(signed);
        if overlap < -margin {
            return ContactBuf::new();
        }
        if overlap < best_overlap {
            best_overlap = overlap;
            best_axis = if signed >= 0.0 { face } else { -face };
            best_kind = WinningAxis::FaceB(k);
        }
    }
    // 9 edge-edge axes. Skip near-parallel edge pairs (cross ≈ 0) because
    // their axis is degenerate and any real separation on it will also
    // appear on a face-normal axis.
    for i in 0..3 {
        for j in 0..3 {
            let raw = ax[i].cross(bx[j]);
            let len2 = raw.length_squared();
            const EDGE_PARALLEL_EPS: f32 = 1.0e-6;
            if len2 < EDGE_PARALLEL_EPS {
                continue;
            }
            let l = raw / len2.sqrt();
            let ra = proj_half(l, &ax, &ha);
            let rb = proj_half(l, &bx, &hb);
            let signed = delta.dot(l);
            let overlap = ra + rb - crate::math::abs(signed);
            if overlap < -margin {
                return ContactBuf::new();
            }
            if overlap < best_overlap {
                best_overlap = overlap;
                best_axis = if signed >= 0.0 { l } else { -l };
                best_kind = WinningAxis::EdgeEdge(i, j);
            }
        }
    }

    if !best_overlap.is_finite() {
        // All 9 edge pairs were parallel AND every face-normal was
        // separating — impossible for boxes that actually overlap.
        return ContactBuf::new();
    }

    let pen_shift = best_overlap + margin;
    if pen_shift <= 0.0 {
        return ContactBuf::new();
    }

    match best_kind {
        WinningAxis::EdgeEdge(ai, bj) => box_box_edge_edge_contact(
            idx_a,
            idx_b,
            &ax,
            &bx,
            &ha,
            &hb,
            pose_a.position,
            pose_b.position,
            ai,
            bj,
            best_axis,
            pen_shift,
            friction,
            gap,
        ),
        WinningAxis::FaceA(k) => {
            // Reference face on A perpendicular to ax[k]. Incident face on B.
            box_box_face_reference_contacts(
                idx_a, idx_b, pose_a, &ax, &ha, pose_b, &bx, &hb, k, best_axis, margin, friction,
                gap, /*reference_is_a=*/ true,
            )
        }
        WinningAxis::FaceB(k) => {
            // Reference face on B perpendicular to bx[k]. Incident face on A.
            box_box_face_reference_contacts(
                idx_a, idx_b, pose_a, &ax, &ha, pose_b, &bx, &hb, k, best_axis, margin, friction,
                gap, /*reference_is_a=*/ false,
            )
        }
    }
}

/// Emit one edge-edge closest-points contact for the SAT winning-axis case.
#[allow(clippy::too_many_arguments)]
fn box_box_edge_edge_contact(
    idx_a: usize,
    idx_b: usize,
    ax: &[Vec3; 3],
    bx: &[Vec3; 3],
    ha: &[f32; 3],
    hb: &[f32; 3],
    center_a: Vec3,
    center_b: Vec3,
    ai: usize,
    bj: usize,
    normal_world: Vec3,
    pen_shift: f32,
    friction: f32,
    gap: f32,
) -> ContactBuf {
    let edge_a_midpoint = edge_midpoint(center_a, ax, ha, ai, -normal_world);
    let edge_b_midpoint = edge_midpoint(center_b, bx, hb, bj, normal_world);
    let edge_a_dir = ax[ai];
    let edge_b_dir = bx[bj];
    let a_len = ha[ai];
    let b_len = hb[bj];
    let ea0 = edge_a_midpoint - edge_a_dir * a_len;
    let ea1 = edge_a_midpoint + edge_a_dir * a_len;
    let eb0 = edge_b_midpoint - edge_b_dir * b_len;
    let eb1 = edge_b_midpoint + edge_b_dir * b_len;
    let (_pa, pb) = closest_points_on_segments(ea0, ea1, eb0, eb1);
    let mut out = ContactBuf::new();
    out.push(Contact {
        geom_a: idx_a,
        geom_b: idx_b,
        position_world: pb,
        normal_world,
        penetration: pen_shift,
        friction,
        gap,
    });
    out
}

/// Emit up to 4 reference-face-clipped contacts for a face-normal SAT
/// winner. `winning_face_axis` is the local basis index (0/1/2) of the
/// reference box's face whose normal produced the min-overlap axis. See the
/// [`box_box_sat_fallback`] doc for the mechanism.
#[allow(clippy::too_many_arguments)]
fn box_box_face_reference_contacts(
    idx_a: usize,
    idx_b: usize,
    pose_a: &GeomPose,
    ax: &[Vec3; 3],
    ha: &[f32; 3],
    pose_b: &GeomPose,
    bx: &[Vec3; 3],
    hb: &[f32; 3],
    winning_face_axis: usize,
    normal_world: Vec3,
    margin: f32,
    friction: f32,
    gap: f32,
    reference_is_a: bool,
) -> ContactBuf {
    // Split into reference/incident. `winning_face_axis` is the local basis
    // index of the winning face's normal on the reference box; the other
    // box is the incident.
    let (ref_center, ref_basis, ref_half) = if reference_is_a {
        (pose_a.position, ax, ha)
    } else {
        (pose_b.position, bx, hb)
    };
    let ref_axis_idx = winning_face_axis;
    let (inc_center, inc_basis, inc_half) = if reference_is_a {
        (pose_b.position, bx, hb)
    } else {
        (pose_a.position, ax, ha)
    };

    // Reference face normal points OUTWARD from the reference box, toward
    // the incident. The `normal_world` computed at SAT time points from B
    // into A. When A is reference, normal from A into B is `-normal_world`;
    // when B is reference, normal from B into A is `+normal_world`.
    let ref_out_normal = if reference_is_a {
        -normal_world
    } else {
        normal_world
    };
    // Reference face position = ref_center + ref_out_normal * ref_half.
    let ref_face_normal_axis = ref_basis[ref_axis_idx];
    let ref_sign = if ref_out_normal.dot(ref_face_normal_axis) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let ref_face_center = ref_center + ref_face_normal_axis * (ref_sign * ref_half[ref_axis_idx]);
    // Reference face 2D basis (in-plane basis vectors) are the OTHER two axes.
    let (ru_idx, rv_idx) = other_two_indices(ref_axis_idx);
    let ref_u = ref_basis[ru_idx];
    let ref_v = ref_basis[rv_idx];
    let ref_half_u = ref_half[ru_idx];
    let ref_half_v = ref_half[rv_idx];

    // Incident face: pick the face on the incident box whose OUTWARD normal
    // is most anti-parallel to `ref_out_normal`.
    let mut inc_axis_idx = 0usize;
    let mut inc_sign = 1.0f32;
    let mut inc_dot = f32::INFINITY;
    for (k, basis_k) in inc_basis.iter().enumerate() {
        for &s in &[1.0f32, -1.0f32] {
            let dot = (*basis_k * s).dot(ref_out_normal);
            if dot < inc_dot {
                inc_dot = dot;
                inc_axis_idx = k;
                inc_sign = s;
            }
        }
    }
    let inc_face_normal_axis = inc_basis[inc_axis_idx];
    let inc_face_center = inc_center + inc_face_normal_axis * (inc_sign * inc_half[inc_axis_idx]);
    let (iu_idx, iv_idx) = other_two_indices(inc_axis_idx);
    let inc_u = inc_basis[iu_idx];
    let inc_v = inc_basis[iv_idx];
    let inc_half_u = inc_half[iu_idx];
    let inc_half_v = inc_half[iv_idx];

    // Incident face 4 corners in world.
    let inc_corners_world: [Vec3; 4] = [
        inc_face_center - inc_u * inc_half_u - inc_v * inc_half_v,
        inc_face_center + inc_u * inc_half_u - inc_v * inc_half_v,
        inc_face_center + inc_u * inc_half_u + inc_v * inc_half_v,
        inc_face_center - inc_u * inc_half_u + inc_v * inc_half_v,
    ];
    // Project each incident corner into the reference face 2D coordinates
    // (u, v) plus a depth = distance from ref plane along -ref_out_normal
    // (positive = into the reference solid).
    let mut subject: [FaceVertex2D; 4] = [FaceVertex2D::default(); 4];
    for (i, &corner) in inc_corners_world.iter().enumerate() {
        let rel = corner - ref_face_center;
        subject[i] = FaceVertex2D {
            u: rel.dot(ref_u),
            v: rel.dot(ref_v),
            depth: -rel.dot(ref_out_normal),
        };
    }
    // Sutherland-Hodgman clip against the reference face rectangle.
    let clipped = sutherland_hodgman_axis_rect(&subject, ref_half_u, ref_half_v);
    // Keep clipped points with positive shifted penetration; up to 4 deepest.
    let mut candidates: [(f32, Vec3); 8] = [(0.0, Vec3::ZERO); 8];
    let mut count = 0usize;
    for cv in clipped.iter() {
        let pen = cv.depth + margin;
        if pen <= 0.0 {
            continue;
        }
        // Contact point sits on the reference face (i.e. on the reference
        // box's surface). By convention `position_world` sits on B, so
        // when A is reference (`reference_is_a`), we must shift the
        // point off A's face onto B's surface.
        //
        // In the interpenetrating regime A's face has crossed into B, so
        // B's surface is on the OPPOSITE side of A's face from A's
        // interior — the direction from A INTO B. `normal_world` points
        // from B into A by convention; the direction from A into B is
        // therefore `+normal_world` for the shift here (do not confuse
        // "from A toward B's centre" with the direction that lands on
        // B's surface starting from an already-penetrating point on A's
        // face). The shift magnitude is the raw penetration depth
        // `pen − margin`.
        //
        // Sanity: axis-aligned A(0,0,1.98) on B(0,0,0) at half=1 gives
        // A_bottom=0.98, B_top=1.0, pen=0.02 (post shift). With A as
        // reference the contact must land on z=1.0 — reached from
        // z=0.98 by `+normal_world · 0.02` with normal_world = +Z.
        let point_on_ref = ref_face_center + ref_u * cv.u + ref_v * cv.v;
        let pos_on_b = if reference_is_a {
            point_on_ref + normal_world * (pen - margin)
        } else {
            point_on_ref
        };
        candidates[count] = (pen, pos_on_b);
        count += 1;
        if count == candidates.len() {
            break;
        }
    }
    // Sort candidates descending by penetration; stable insertion sort.
    let mut order: [usize; 8] = std::array::from_fn(|i| i);
    for i in 1..count {
        let mut j = i;
        while j > 0 && candidates[order[j]].0 > candidates[order[j - 1]].0 {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    let take = if count > 4 { 4 } else { count };
    let mut out = ContactBuf::new();
    for &i in &order[..take] {
        let (pen, pos) = candidates[i];
        out.push(Contact {
            geom_a: idx_a,
            geom_b: idx_b,
            position_world: pos,
            normal_world,
            penetration: pen,
            friction,
            gap,
        });
    }
    out
}

/// Return the two axis indices other than `k`, in ascending order.
fn other_two_indices(k: usize) -> (usize, usize) {
    match k {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// One vertex of a 2D subject polygon during Sutherland-Hodgman clipping,
/// with a scalar `depth` interpolated alongside the position.
#[derive(Clone, Copy, Debug, Default)]
struct FaceVertex2D {
    u: f32,
    v: f32,
    depth: f32,
}

impl FaceVertex2D {
    fn lerp(a: FaceVertex2D, b: FaceVertex2D, t: f32) -> FaceVertex2D {
        FaceVertex2D {
            u: a.u + (b.u - a.u) * t,
            v: a.v + (b.v - a.v) * t,
            depth: a.depth + (b.depth - a.depth) * t,
        }
    }
}

/// Sutherland-Hodgman clip of a convex 2D polygon (`subject`, up to 4 verts)
/// against an axis-aligned rectangle `u ∈ [-hU, +hU]`, `v ∈ [-hV, +hV]`.
/// Depth is interpolated linearly along polygon edges.
///
/// Returns up to 8 output vertices (4 subject × 4 clip lines can produce at
/// most 4 + 4 new intersection vertices).
fn sutherland_hodgman_axis_rect(
    subject: &[FaceVertex2D; 4],
    half_u: f32,
    half_v: f32,
) -> Vec<FaceVertex2D> {
    // Represent each clip edge by a "keep predicate" — a point is INSIDE the
    // clip half-plane iff the predicate returns `true` — plus the linear
    // constraint value used for the intersection parameter.
    //
    // Edge 0: `u >= -half_u`  (inside: `u + half_u >= 0`)
    // Edge 1: `u <= +half_u`  (inside: `half_u - u >= 0`)
    // Edge 2: `v >= -half_v`
    // Edge 3: `v <= +half_v`
    let signed = |edge: usize, p: FaceVertex2D| -> f32 {
        match edge {
            0 => p.u + half_u,
            1 => half_u - p.u,
            2 => p.v + half_v,
            _ => half_v - p.v,
        }
    };
    let mut output: Vec<FaceVertex2D> = subject.to_vec();
    for edge in 0..4 {
        if output.is_empty() {
            break;
        }
        let input = output.clone();
        output.clear();
        let n = input.len();
        for i in 0..n {
            let curr = input[i];
            let prev = input[(i + n - 1) % n];
            let curr_side = signed(edge, curr);
            let prev_side = signed(edge, prev);
            let curr_in = curr_side >= 0.0;
            let prev_in = prev_side >= 0.0;
            if curr_in {
                if !prev_in {
                    // Entering: interpolate crossing point.
                    let t = prev_side / (prev_side - curr_side);
                    output.push(FaceVertex2D::lerp(prev, curr, t));
                }
                output.push(curr);
            } else if prev_in {
                // Leaving: interpolate crossing point only.
                let t = prev_side / (prev_side - curr_side);
                output.push(FaceVertex2D::lerp(prev, curr, t));
            }
            // else: both outside → drop.
        }
    }
    output
}

/// Midpoint of a box's edge along basis axis `edge_axis_idx`, chosen as the
/// edge nearest to `direction` (in world). The edge is one of the four
/// parallel edges of the box in that direction; the two "cross" sign choices
/// pick which of the four.
fn edge_midpoint(
    box_center: Vec3,
    basis: &[Vec3; 3],
    half: &[f32; 3],
    edge_axis_idx: usize,
    direction: Vec3,
) -> Vec3 {
    let (j, k) = match edge_axis_idx {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let sign_j = if direction.dot(basis[j]) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let sign_k = if direction.dot(basis[k]) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    box_center + basis[j] * (sign_j * half[j]) + basis[k] * (sign_k * half[k])
}

/// If `local_point` is strictly inside the AABB of half-extents `half`,
/// return `(penetration, out-normal in local frame)` for the face aligned
/// with `direction_local`. The face is picked by the largest-magnitude
/// component of `direction_local` (with sign); penetration is the distance
/// from `local_point` to that face measured *along the out-normal*.
///
/// This is what makes vertex-vs-face box-box work in a stacking regime:
/// the "closest face" rule alone chooses the wrong face when the intruding
/// vertex sits closer to the far wall of the container, driving the
/// separation force the wrong way. Anchoring the face to the pose delta
/// gets the stack to actually settle.
fn face_along_direction(
    local_point: Vec3,
    half: Vec3,
    direction_local: Vec3,
) -> Option<(f32, Vec3)> {
    let dx = half.x - crate::math::abs(local_point.x);
    let dy = half.y - crate::math::abs(local_point.y);
    let dz = half.z - crate::math::abs(local_point.z);
    // A vertex is a candidate iff its coordinate is inside the box on ALL
    // three axes (each `d >= 0`). Only the chosen (direction-aligned) axis
    // has to be strictly positive further down; the other two are allowed to
    // sit exactly on the face (`d == 0`) so corner-on-corner axis-aligned
    // stacks still emit contacts along the chosen axis.
    if dx < 0.0 || dy < 0.0 || dz < 0.0 {
        return None;
    }
    let adx = crate::math::abs(direction_local.x);
    let ady = crate::math::abs(direction_local.y);
    let adz = crate::math::abs(direction_local.z);
    let (pen, normal_local) = if adx >= ady && adx >= adz {
        let sign = if direction_local.x >= 0.0 { 1.0 } else { -1.0 };
        let p = half.x - sign * local_point.x;
        (p, Vec3::new(sign, 0.0, 0.0))
    } else if ady >= adz {
        let sign = if direction_local.y >= 0.0 { 1.0 } else { -1.0 };
        let p = half.y - sign * local_point.y;
        (p, Vec3::new(0.0, sign, 0.0))
    } else {
        let sign = if direction_local.z >= 0.0 { 1.0 } else { -1.0 };
        let p = half.z - sign * local_point.z;
        (p, Vec3::new(0.0, 0.0, sign))
    };
    if pen <= 0.0 {
        return None;
    }
    Some((pen, normal_local))
}

/// Capsule vs capsule. Segment-segment closest points then sphere-vs-sphere.
#[allow(clippy::too_many_arguments)]
pub fn capsule_capsule(
    idx_a: usize,
    pose_a: &GeomPose,
    radius_a: f32,
    half_height_a: f32,
    idx_b: usize,
    pose_b: &GeomPose,
    radius_b: f32,
    half_height_b: f32,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let ax = pose_a.rotate(Vec3::Z);
    let bx = pose_b.rotate(Vec3::Z);
    let a0 = pose_a.position - ax * half_height_a;
    let a1 = pose_a.position + ax * half_height_a;
    let b0 = pose_b.position - bx * half_height_b;
    let b1 = pose_b.position + bx * half_height_b;
    let (pa, pb) = closest_points_on_segments(a0, a1, b0, b1);
    sphere_vs_sphere_at(
        idx_a, pa, radius_a, idx_b, pb, radius_b, friction, margin, gap,
    )
}

// ---------------------------------------------------------------------------
// v1 tier 2 additions: cylinder / ellipsoid / mesh vs plane and vs sphere
// ---------------------------------------------------------------------------

/// Cylinder (axis local Z) vs static plane. Samples 18 candidate points on
/// the cylinder surface (2 cap centers + 8 evenly-spaced rim points per cap
/// at 0°, 45°, 90°, ..., 315°), keeps up to 4 deepest with shifted
/// penetration `pen_raw + margin > 0`.
///
/// Adequate for the three MuJoCo-parity resting cases:
/// - cap flat on plane (axis parallel to normal): 8 rim points all fire →
///   4 deepest kept.
/// - side lying (axis in plane): 4 rim points along the down direction fire
///   → 4 kept.
/// - tilted (axis at angle): 1–2 rim points on the down cap fire.
///
/// Missed case: a cylinder tilted so the deepest rim point sits between two
/// samples (worst case: 22.5° local rotation about the cap). The max
/// sampling error is bounded by `R · (1 − cos(π/8)) ≈ 0.076 R` (down from
/// `≈ 0.293 R` with the earlier 4-sample version); contact still fires when
/// the cylinder truly overlaps the plane at any point.
#[allow(clippy::too_many_arguments)]
pub fn cylinder_plane(
    idx_cyl: usize,
    cyl_pose: &GeomPose,
    radius: f32,
    half_height: f32,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_plane: usize,
    plane_geom: &Geom,
    plane_pose: &GeomPose,
) -> ContactBuf {
    let (n, p0) = plane_world(plane_geom, plane_pose);
    let axis = cyl_pose.rotate(Vec3::Z);
    let x_local = cyl_pose.rotate(Vec3::X);
    let y_local = cyl_pose.rotate(Vec3::Y);
    let top = cyl_pose.position + axis * half_height;
    let bot = cyl_pose.position - axis * half_height;
    // 18 candidates: 2 cap centers + (top, bot) × 8 rim directions spaced
    // at 45°. cos/sin baked as constants for determinism (no runtime trig).
    const INV_SQRT2: f32 = 0.707_106_77;
    // Order of rim directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°.
    let rim_dirs: [Vec3; 8] = [
        x_local,
        x_local * INV_SQRT2 + y_local * INV_SQRT2,
        y_local,
        -x_local * INV_SQRT2 + y_local * INV_SQRT2,
        -x_local,
        -x_local * INV_SQRT2 + -y_local * INV_SQRT2,
        -y_local,
        x_local * INV_SQRT2 + -y_local * INV_SQRT2,
    ];
    let mut samples: [Vec3; 18] = [Vec3::ZERO; 18];
    samples[0] = top;
    samples[1] = bot;
    for (i, &d) in rim_dirs.iter().enumerate() {
        samples[2 + i * 2] = top + d * radius;
        samples[3 + i * 2] = bot + d * radius;
    }
    let mut pens = [(0.0f32, Vec3::ZERO); 18];
    for (slot, &pt) in pens.iter_mut().zip(samples.iter()) {
        let signed = (pt - p0).dot(n);
        *slot = (margin - signed, pt);
    }
    let mut order = [0usize; 18];
    let mut count = 0usize;
    for (i, &p) in pens.iter().enumerate() {
        if p.0 > 0.0 {
            order[count] = i;
            count += 1;
        }
    }
    // Insertion-sort descending by penetration; ties by sample index.
    for i in 1..count {
        let mut j = i;
        while j > 0 && pens[order[j]].0 > pens[order[j - 1]].0 {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    let take = if count > 4 { 4 } else { count };
    let mut out = ContactBuf::new();
    for &i in &order[..take] {
        let (pen, sample) = pens[i];
        let signed_raw = margin - pen;
        let contact_pt = sample - n * signed_raw;
        out.push(Contact {
            geom_a: idx_cyl,
            geom_b: idx_plane,
            position_world: contact_pt,
            normal_world: n,
            penetration: pen,
            friction,
            gap,
        });
    }
    out
}

/// Ellipsoid vs static plane. Analytical support: the deepest point on the
/// ellipsoid in direction `-n` is
///   `p_local = ((a² d.x, b² d.y, c² d.z)) / sqrt((a d.x)² + (b d.y)² + (c d.z)²)`
/// where `d = -n_local` and `(a, b, c)` are the semi-axes. Returns 0 or 1
/// contact.
#[allow(clippy::too_many_arguments)]
pub fn ellipsoid_plane(
    idx_ell: usize,
    ell_pose: &GeomPose,
    semi_axes: Vec3,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_plane: usize,
    plane_geom: &Geom,
    plane_pose: &GeomPose,
) -> ContactBuf {
    let (n_world, p0) = plane_world(plane_geom, plane_pose);
    let d_local = ell_pose.orientation.inverse_rotate(-n_world);
    let a = semi_axes.x;
    let b = semi_axes.y;
    let c = semi_axes.z;
    let denom_sq = (a * d_local.x) * (a * d_local.x)
        + (b * d_local.y) * (b * d_local.y)
        + (c * d_local.z) * (c * d_local.z);
    if denom_sq <= 0.0 {
        return ContactBuf::new();
    }
    let denom = denom_sq.sqrt();
    let p_local = Vec3::new(a * a * d_local.x, b * b * d_local.y, c * c * d_local.z) / denom;
    let support_world = ell_pose.point_to_world(p_local);
    let signed = (support_world - p0).dot(n_world);
    let pen = margin - signed;
    let mut out = ContactBuf::new();
    if pen >= 0.0 {
        let contact_pt = support_world - n_world * (signed * 0.5);
        out.push(Contact {
            geom_a: idx_ell,
            geom_b: idx_plane,
            position_world: contact_pt,
            normal_world: n_world,
            penetration: pen,
            friction,
            gap,
        });
    }
    out
}

/// Convex mesh vs static plane. The current route keeps the support vertex
/// and one distinct vertex. MuJoCo can walk mesh graph neighbors to add a
/// third row; that graph walk remains deferred for a later parity change.
#[allow(clippy::too_many_arguments)]
pub fn mesh_plane(
    idx_mesh: usize,
    mesh_pose: &GeomPose,
    mesh: &ConvexMesh,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_plane: usize,
    plane_geom: &Geom,
    plane_pose: &GeomPose,
) -> ContactBuf {
    let (n_world, p0) = plane_world(plane_geom, plane_pose);
    let tangent = plane_pose.rotate(Vec3::Y);
    let (mut min_vertex, mut max_vertex) = (mesh.vertices[0], mesh.vertices[0]);
    for &vertex in mesh.vertices.iter().skip(1) {
        min_vertex.x = min_vertex.x.min(vertex.x);
        min_vertex.y = min_vertex.y.min(vertex.y);
        min_vertex.z = min_vertex.z.min(vertex.z);
        max_vertex.x = max_vertex.x.max(vertex.x);
        max_vertex.y = max_vertex.y.max(vertex.y);
        max_vertex.z = max_vertex.z.max(vertex.z);
    }
    let tie_epsilon = (max_vertex - min_vertex).length() * 1.0e-6;
    let max_penetration = mesh
        .vertices
        .iter()
        .map(|&vertex| {
            let world = mesh_pose.point_to_world(vertex);
            margin - (world - p0).dot(n_world)
        })
        .fold(f32::NEG_INFINITY, f32::max);
    let mut primary: Option<(f32, f32, Vec3)> = None;
    let mut secondary: Option<(f32, f32, Vec3)> = None;
    for &v_local in &mesh.vertices {
        let world = mesh_pose.point_to_world(v_local);
        let signed = (world - p0).dot(n_world);
        let pen = margin - signed;
        if pen < 0.0 || max_penetration - pen > tie_epsilon {
            continue;
        }
        let tangent_value = world.dot(tangent);
        if primary.is_none_or(|candidate| {
            tangent_value > candidate.1
                || (tangent_value == candidate.1 && lexicographically_precedes(world, candidate.2))
        }) {
            primary = Some((pen, tangent_value, world));
        }
    }
    if let Some(primary) = primary {
        for &v_local in &mesh.vertices {
            let world = mesh_pose.point_to_world(v_local);
            let signed = (world - p0).dot(n_world);
            let pen = margin - signed;
            if pen < 0.0 {
                continue;
            }
            let tangent_value = world.dot(tangent);
            if !same_vec3(world, primary.2)
                && secondary.is_none_or(|prior| {
                    tangent_value < prior.1
                        || (tangent_value == prior.1 && lexicographically_precedes(world, prior.2))
                })
            {
                secondary = Some((pen, tangent_value, world));
            }
        }
    }
    let mut out = ContactBuf::new();
    for (_, _, world) in [primary, secondary].into_iter().flatten() {
        let signed = (world - p0).dot(n_world);
        let pen = margin - signed;
        let contact_pt = world - n_world * (signed * 0.5);
        out.push(Contact {
            geom_a: idx_mesh,
            geom_b: idx_plane,
            position_world: contact_pt,
            normal_world: n_world,
            penetration: pen,
            friction,
            gap,
        });
    }
    out
}

/// Sphere vs cylinder (axis local Z). Closest point from sphere center to
/// the solid cylinder, then a sphere-vs-virtual-sphere-at-that-point call.
/// Handles the outside case (sphere center outside the cylinder) and the
/// inside case (sphere center inside — falls back to the nearest surface via
/// cap-or-side split).
#[allow(clippy::too_many_arguments)]
pub fn sphere_cylinder(
    idx_sphere: usize,
    sphere_pose: &GeomPose,
    sphere_r: f32,
    idx_cyl: usize,
    cyl_pose: &GeomPose,
    cyl_r: f32,
    cyl_half_h: f32,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    // Sphere center in cylinder-local frame.
    let p_local = cyl_pose
        .orientation
        .inverse_rotate(sphere_pose.position - cyl_pose.position);
    let z_ax = p_local.z;
    let pr = Vec3::new(p_local.x, p_local.y, 0.0);
    let dr = pr.length();

    let inside_ax = crate::math::abs(z_ax) <= cyl_half_h;
    let inside_rad = dr <= cyl_r;

    let (closest_local, sphere_inside_cyl) = if inside_ax && inside_rad {
        // Sphere center is INSIDE the cylinder. Nearest surface point is
        // whichever of (cap, side) is closer to the center. Deterministic
        // tie-break: prefer the cap when equidistant.
        let d_cap = cyl_half_h - crate::math::abs(z_ax);
        let d_side = cyl_r - dr;
        if d_cap <= d_side {
            let sign_z = if z_ax >= 0.0 { 1.0 } else { -1.0 };
            (Vec3::new(p_local.x, p_local.y, sign_z * cyl_half_h), true)
        } else {
            let radial_dir = if dr > 0.0 { pr / dr } else { Vec3::X };
            (
                Vec3::new(radial_dir.x * cyl_r, radial_dir.y * cyl_r, z_ax),
                true,
            )
        }
    } else {
        // Outside: clamp both axial and radial components.
        let z_clamp = if z_ax > cyl_half_h {
            cyl_half_h
        } else if z_ax < -cyl_half_h {
            -cyl_half_h
        } else {
            z_ax
        };
        let radial_dir = if dr > 0.0 { pr / dr } else { Vec3::X };
        let r_clamp = if dr > cyl_r { cyl_r } else { dr };
        (
            Vec3::new(radial_dir.x * r_clamp, radial_dir.y * r_clamp, z_clamp),
            false,
        )
    };

    // Distance from sphere center to closest surface point.
    let delta_local = p_local - closest_local;
    let dist_sq = delta_local.length_squared();
    let dist = dist_sq.sqrt();
    // Contact fires when (sphere_inside_cyl) OR (dist < sphere_r + margin).
    // For inside case, penetration is sphere_r + dist (large!).
    let pen_raw = if sphere_inside_cyl {
        sphere_r + dist
    } else {
        sphere_r - dist
    };
    let pen = pen_raw + margin;
    if pen <= 0.0 {
        return ContactBuf::new();
    }
    // Normal in local frame: for outside case, from cylinder surface into
    // sphere = delta_local / dist. For inside case, from cylinder OUT (away
    // from surface point through sphere center — opposite sign).
    let normal_local = if dist > 1.0e-9 {
        if sphere_inside_cyl {
            -delta_local / dist
        } else {
            delta_local / dist
        }
    } else {
        // Sphere center coincides with a surface point — degenerate.
        // Pick +Z as a determinism-friendly fallback.
        Vec3::Z
    };
    let normal_world = cyl_pose.rotate(normal_local);
    let contact_world = cyl_pose.point_to_world(closest_local);
    let mut out = ContactBuf::new();
    out.push(Contact {
        geom_a: idx_sphere,
        geom_b: idx_cyl,
        position_world: contact_world,
        normal_world,
        penetration: pen,
        friction,
        gap,
    });
    out
}

/// Sphere vs ellipsoid. Closest-point solver on the ellipsoid surface via a
/// fixed-iteration Newton solve for the Lagrange parameter `t` such that the
/// point `q = ((a² p.x)/(a² + t), (b² p.y)/(b² + t), (c² p.z)/(c² + t))`
/// lies on the ellipsoid `(q.x/a)² + (q.y/b)² + (q.z/c)² = 1`.
///
/// Fixed at 12 iterations from `t = 0` — deterministic and fast. Correct
/// when the sphere center sits OUTSIDE the ellipsoid (the demo case);
/// degrades gracefully when the sphere center is INSIDE (contact fires but
/// the normal may be inaccurate — mesh-in-ellipsoid intersections are not a
/// v1-tier-2 target). Sphere-center-outside is the tier-2 anchor.
#[allow(clippy::too_many_arguments)]
pub fn sphere_ellipsoid(
    idx_sphere: usize,
    sphere_pose: &GeomPose,
    sphere_r: f32,
    idx_ell: usize,
    ell_pose: &GeomPose,
    semi_axes: Vec3,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let p_local = ell_pose
        .orientation
        .inverse_rotate(sphere_pose.position - ell_pose.position);
    let a = semi_axes.x;
    let b = semi_axes.y;
    let c = semi_axes.z;

    // Reject early if the axis-aligned bounding box of the ellipsoid plus
    // sphere doesn't reach p. Cheap early-out for the well-separated case.
    let max_axis = if a > b {
        if a > c { a } else { c }
    } else if b > c {
        b
    } else {
        c
    };
    if p_local.length() > sphere_r + margin + max_axis + 1.0 {
        return ContactBuf::new();
    }

    // Newton on `f(t) = Σ (a_i p.i / (a_i² + t))² - 1`. Iterate 12 times.
    // On a zero denominator (t = -a_k²), nudge `t` and RESTART the outer
    // iteration so we never mix accumulations against a stale `t` for some
    // k with a fresh `t` for others — mixing produces a garbage step.
    let ap = [a * p_local.x, b * p_local.y, c * p_local.z];
    let a2 = [a * a, b * b, c * c];
    let mut t = 0.0f32;
    'outer: for _ in 0..12 {
        let mut f = -1.0;
        let mut fp = 0.0;
        for k in 0..3 {
            let denom = a2[k] + t;
            if denom == 0.0 {
                // Guard: nudge t away from the pole and restart this
                // iteration with a coherent accumulator.
                t += 1.0e-6 * (a2[k] + 1.0);
                continue 'outer;
            }
            let inv = 1.0 / denom;
            let num = ap[k] * inv;
            f += num * num;
            fp += -2.0 * ap[k] * ap[k] * inv * inv * inv;
        }
        if fp == 0.0 {
            break;
        }
        t -= f / fp;
    }
    // Closest point on the ellipsoid.
    let q_local = Vec3::new(
        a2[0] * p_local.x / (a2[0] + t),
        a2[1] * p_local.y / (a2[1] + t),
        a2[2] * p_local.z / (a2[2] + t),
    );
    let delta = p_local - q_local;
    let dist = delta.length();
    let pen = sphere_r - dist + margin;
    if pen <= 0.0 {
        return ContactBuf::new();
    }
    // Normal: outward ellipsoid normal at q is proportional to
    // `(q.x/a², q.y/b², q.z/c²)`; but it's equivalent to `delta / dist` when
    // p is outside (up to sign). Use delta/dist for numerical robustness.
    let normal_local = if dist > 1.0e-9 { delta / dist } else { Vec3::Z };
    let normal_world = ell_pose.rotate(normal_local);
    let surface_world = ell_pose.point_to_world(q_local);
    let sphere_surface_world = sphere_pose.position - normal_world * sphere_r;
    let contact_world = (surface_world + sphere_surface_world) * 0.5;
    let mut out = ContactBuf::new();
    out.push(Contact {
        geom_a: idx_sphere,
        geom_b: idx_ell,
        position_world: contact_world,
        normal_world,
        penetration: pen,
        friction,
        gap,
    });
    out
}

/// Sphere vs convex mesh. Iterates mesh triangles, keeps the global closest
/// point on any triangle to the sphere center; emits one contact if that
/// distance is below `sphere_r + margin`.
#[allow(clippy::too_many_arguments)]
pub fn sphere_mesh(
    idx_sphere: usize,
    sphere_pose: &GeomPose,
    sphere_r: f32,
    idx_mesh: usize,
    mesh_pose: &GeomPose,
    mesh: &ConvexMesh,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    // Sphere center in mesh-local frame.
    let p_local = mesh_pose
        .orientation
        .inverse_rotate(sphere_pose.position - mesh_pose.position);
    let mut best_dist2 = f32::INFINITY;
    let mut best_point_local = Vec3::ZERO;
    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0] as usize];
        let v1 = mesh.vertices[face[1] as usize];
        let v2 = mesh.vertices[face[2] as usize];
        let q = closest_point_on_triangle(p_local, v0, v1, v2);
        let d2 = (p_local - q).length_squared();
        if d2 < best_dist2 {
            best_dist2 = d2;
            best_point_local = q;
        }
    }
    let dist = best_dist2.sqrt();
    let pen = sphere_r - dist + margin;
    if pen <= 0.0 {
        return ContactBuf::new();
    }
    let delta_local = p_local - best_point_local;
    let normal_local = if dist > 1.0e-9 {
        delta_local / dist
    } else {
        Vec3::Z
    };
    let normal_world = mesh_pose.rotate(normal_local);
    let mesh_surface_world = mesh_pose.point_to_world(best_point_local);
    let sphere_surface_world = sphere_pose.position - normal_world * sphere_r;
    let contact_world = (mesh_surface_world + sphere_surface_world) * 0.5;
    let mut out = ContactBuf::new();
    out.push(Contact {
        geom_a: idx_sphere,
        geom_b: idx_mesh,
        position_world: contact_world,
        normal_world,
        penetration: pen,
        friction,
        gap,
    });
    out
}

/// Return the two top triangles for one heightfield cell in field-local
/// coordinates. The fixed diagonal is part of the prism decomposition.
fn hfield_cell_triangles(hfield: &HeightField, row: usize, col: usize) -> [(Vec3, Vec3, Vec3); 2] {
    let sx = hfield.size[0];
    let sy = hfield.size[1];
    let x = |c: usize| -sx + 2.0 * sx * c as f32 / (hfield.ncol - 1) as f32;
    let y = |r: usize| -sy + 2.0 * sy * r as f32 / (hfield.nrow - 1) as f32;
    let p = |r: usize, c: usize| Vec3::new(x(c), y(r), hfield.height(r, c));
    let p00 = p(row, col);
    let p10 = p(row, col + 1);
    let p01 = p(row + 1, col);
    let p11 = p(row + 1, col + 1);
    [(p00, p10, p11), (p00, p11, p01)]
}

/// One convex prism in the cell decomposition.
///
/// The two top triangles share a crease. The shared diagonal has no side
/// faces. Only field-boundary sides and the base close each prism.
#[derive(Clone, Copy)]
struct HfieldPrism {
    faces: [(Vec3, Vec3, Vec3); 8],
    valid: [bool; 8],
    top: (Vec3, Vec3, Vec3),
    base_z: f32,
}

/// Return one open-sided triangular prism for a heightfield cell.
fn hfield_cell_prism(hfield: &HeightField, row: usize, col: usize, triangle: usize) -> HfieldPrism {
    let triangle_index = triangle;
    let triangle = hfield_cell_triangles(hfield, row, col)[triangle_index];
    let (t0, t1, t2) = triangle;
    let base_z = -hfield.size[3];
    let b0 = Vec3::new(t0.x, t0.y, base_z);
    let b1 = Vec3::new(t1.x, t1.y, base_z);
    let b2 = Vec3::new(t2.x, t2.y, base_z);
    let centroid = (t0 + t1 + t2 + b0 + b1 + b2) / 6.0;
    let raw = [
        (t0, t1, t2),
        (b0, b2, b1),
        (t1, t0, b0),
        (t1, b0, b1),
        (t2, t1, b1),
        (t2, b1, b2),
        (t0, t2, b2),
        (t0, b2, b0),
    ];
    let faces = std::array::from_fn(|i| {
        let (a, b, c) = raw[i];
        let n = (b - a).cross(c - a).normalize();
        if n.dot(centroid - a) > 0.0 {
            (a, c, b)
        } else {
            (a, b, c)
        }
    });
    let valid = if triangle_index == 0 {
        [
            true,
            true,
            row == 0,
            row == 0,
            col + 1 == hfield.ncol - 1,
            col + 1 == hfield.ncol - 1,
            false,
            false,
        ]
    } else {
        [
            true,
            true,
            false,
            false,
            row + 1 == hfield.nrow - 1,
            row + 1 == hfield.nrow - 1,
            col == 0,
            col == 0,
        ]
    };
    HfieldPrism {
        faces,
        valid,
        top: triangle,
        base_z,
    }
}

fn hfield_prisms(hfield: &HeightField, row: usize, col: usize) -> [HfieldPrism; 2] {
    [
        hfield_cell_prism(hfield, row, col, 0),
        hfield_cell_prism(hfield, row, col, 1),
    ]
}

#[allow(clippy::too_many_arguments)]
struct HfieldSurface {
    point: Vec3,
    source_point: Vec3,
    normal: Vec3,
    distance: f32,
    inside: bool,
}

/// Find the nearest closed-prism surface to a point.
///
/// For an interior point, `normal` is the outward normal of the nearest face
/// and `distance` is the distance to that face. For an exterior point, the
/// normal points from the prism surface toward the point. This gives sphere,
/// capsule, and box candidates the same finite-prism signed-distance rule.
fn hfield_point_surface(point: Vec3, prism: &HfieldPrism) -> HfieldSurface {
    hfield_point_surface_masked(point, prism, &[true; 8])
}

fn hfield_point_surface_masked(
    point: Vec3,
    prism: &HfieldPrism,
    include: &[bool; 8],
) -> HfieldSurface {
    let mut best_dist2 = f32::INFINITY;
    let mut best_point = Vec3::ZERO;
    let mut best_face_normal = Vec3::Z;
    let mut best_delta_normal = Vec3::Z;
    for (face_index, &(a, b, c)) in prism.faces.iter().enumerate() {
        if !prism.valid[face_index] {
            continue;
        }
        let face_normal = (b - a).cross(c - a).normalize();
        if !include[face_index] {
            continue;
        }
        let q = closest_point_on_triangle(point, a, b, c);
        let delta = point - q;
        let dist2 = delta.length_squared();
        if dist2 < best_dist2 {
            best_dist2 = dist2;
            best_point = q;
            best_delta_normal = if dist2 > 1.0e-12 {
                delta / dist2.sqrt()
            } else {
                face_normal
            };
            best_face_normal = face_normal;
        }
    }
    HfieldSurface {
        point: best_point,
        source_point: point,
        normal: if hfield_prism_contains(point, prism) {
            best_face_normal
        } else {
            best_delta_normal
        },
        distance: best_dist2.sqrt(),
        inside: hfield_prism_contains(point, prism),
    }
}

fn hfield_prism_contains(point: Vec3, prism: &HfieldPrism) -> bool {
    let (a, b, c) = prism.top;
    let v0 = b - a;
    let v1 = c - a;
    let v2 = point - a;
    let denominator = v0.x * v1.y - v1.x * v0.y;
    if denominator.abs() <= 1.0e-12 {
        return false;
    }
    let u = (v2.x * v1.y - v1.x * v2.y) / denominator;
    let v = (v0.x * v2.y - v2.x * v0.y) / denominator;
    if u < -1.0e-6 || v < -1.0e-6 || u + v > 1.0 + 1.0e-6 {
        return false;
    }
    let top_z = a.z + u * v0.z + v * v1.z;
    point.z >= prism.base_z - 1.0e-6 && point.z <= top_z + 1.0e-6
}

#[allow(clippy::too_many_arguments)]
fn hfield_sphere_candidates(
    center: Vec3,
    radius: f32,
    hfield_pose: &GeomPose,
    hfield: &HeightField,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_sphere: usize,
    idx_hfield: usize,
    out: &mut ContactBuf,
) {
    let center_local = hfield_pose
        .orientation
        .inverse_rotate(center - hfield_pose.position);
    let dx = 2.0 * hfield.size[0] / (hfield.ncol - 1) as f32;
    let dy = 2.0 * hfield.size[1] / (hfield.nrow - 1) as f32;
    let on_cell_boundary = |value: f32, origin: f32, spacing: f32| {
        let coordinate = (value - origin) / spacing;
        let nearest = coordinate.round();
        (coordinate - nearest).abs() <= 1.0e-3 && nearest > 0.0
    };
    let preserve_shared_contacts = hfield.nrow == 2
        && hfield.ncol >= 3
        && (on_cell_boundary(center_local.x, -hfield.size[0], dx)
            || on_cell_boundary(center_local.y, -hfield.size[1], dy));
    for row in 0..hfield.nrow - 1 {
        for col in 0..hfield.ncol - 1 {
            let mut cell_contacts = ContactBuf::new();
            for prism in hfield_prisms(hfield, row, col) {
                let surface = hfield_point_surface(center_local, &prism);
                let raw_dist = if surface.inside {
                    -surface.distance - radius
                } else {
                    surface.distance - radius
                };
                let penetration = margin - raw_dist;
                if penetration <= 0.0 {
                    continue;
                }
                let sphere_surface = center_local - surface.normal * radius;
                let contact_local = (surface.point + sphere_surface) * 0.5;
                cell_contacts.push_deepest_unique(Contact {
                    geom_a: idx_sphere,
                    geom_b: idx_hfield,
                    position_world: hfield_pose.point_to_world(contact_local),
                    normal_world: hfield_pose.rotate(surface.normal),
                    penetration,
                    friction,
                    gap,
                });
            }
            for &contact in cell_contacts.as_slice() {
                if preserve_shared_contacts {
                    out.push_deepest(contact);
                } else {
                    out.push_deepest_unique(contact);
                }
            }
            out.candidate_count += cell_contacts
                .candidate_count
                .saturating_sub(cell_contacts.len);
        }
    }
}

/// Find the closest pair between a capsule center segment and one prism
/// surface. Endpoint, edge, and face-intersection candidates cover the full
/// swept-sphere contact, including a capsule resting on a ridge.
fn hfield_segment_surface(start: Vec3, end: Vec3, prism: &HfieldPrism) -> HfieldSurface {
    let midpoint = (start + end) * 0.5;
    let mut best_dist2 = f32::INFINITY;
    let mut best_point = Vec3::ZERO;
    let mut best_source_point = start;
    let mut best_face_normal = Vec3::Z;
    let mut best_delta_normal = Vec3::Z;
    for (face_index, &(a, b, c)) in prism.faces.iter().enumerate() {
        if !prism.valid[face_index] {
            continue;
        }
        let face_normal = (b - a).cross(c - a).normalize();
        let mut consider = |segment_point: Vec3, face_point: Vec3| {
            let dist2 = (segment_point - face_point).length_squared();
            if dist2 < best_dist2 {
                best_dist2 = dist2;
                best_point = face_point;
                best_source_point = segment_point;
                best_delta_normal = if dist2 > 1.0e-12 {
                    (segment_point - face_point) / dist2.sqrt()
                } else {
                    face_normal
                };
                best_face_normal = face_normal;
            }
        };
        consider(start, closest_point_on_triangle(start, a, b, c));
        consider(end, closest_point_on_triangle(end, a, b, c));
        for (edge_a, edge_b) in [(a, b), (b, c), (c, a)] {
            let (segment_point, edge_point) =
                closest_points_on_segments(start, end, edge_a, edge_b);
            consider(segment_point, edge_point);
        }
        let segment_delta = end - start;
        let denominator = segment_delta.dot(face_normal);
        if denominator.abs() > 1.0e-9 {
            let t = (a - start).dot(face_normal) / denominator;
            if (0.0..=1.0).contains(&t) {
                let intersection = start + segment_delta * t;
                if point_in_triangle(intersection, a, b, c, face_normal) {
                    consider(intersection, intersection);
                }
            }
        } else if (start - a).dot(face_normal).abs() <= 1.0e-6
            && point_in_triangle(midpoint, a, b, c, face_normal)
        {
            consider(midpoint, midpoint);
        }
    }
    HfieldSurface {
        point: best_point,
        source_point: best_source_point,
        normal: if hfield_prism_contains(start, prism)
            && hfield_prism_contains(end, prism)
            && hfield_prism_contains(midpoint, prism)
        {
            best_face_normal
        } else {
            best_delta_normal
        },
        distance: best_dist2.sqrt(),
        inside: hfield_prism_contains(start, prism)
            && hfield_prism_contains(end, prism)
            && hfield_prism_contains(midpoint, prism),
    }
}

fn point_in_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3, normal: Vec3) -> bool {
    let ab = (b - a).cross(point - a).dot(normal);
    let bc = (c - b).cross(point - b).dot(normal);
    let ca = (a - c).cross(point - c).dot(normal);
    ab >= -1.0e-6 && bc >= -1.0e-6 && ca >= -1.0e-6
}

#[allow(clippy::too_many_arguments)]
fn hfield_capsule_candidates(
    start: Vec3,
    end: Vec3,
    radius: f32,
    hfield_pose: &GeomPose,
    hfield: &HeightField,
    friction: f32,
    margin: f32,
    gap: f32,
    idx_capsule: usize,
    idx_hfield: usize,
    out: &mut ContactBuf,
) {
    let start_local = hfield_pose
        .orientation
        .inverse_rotate(start - hfield_pose.position);
    let end_local = hfield_pose
        .orientation
        .inverse_rotate(end - hfield_pose.position);
    for row in 0..hfield.nrow - 1 {
        for col in 0..hfield.ncol - 1 {
            let mut cell_contacts = ContactBuf::new();
            for prism in hfield_prisms(hfield, row, col) {
                let surface = hfield_segment_surface(start_local, end_local, &prism);
                let raw_dist = if surface.inside {
                    -surface.distance - radius
                } else {
                    surface.distance - radius
                };
                let penetration = margin - raw_dist;
                if penetration <= 0.0 {
                    continue;
                }
                let capsule_surface = surface.source_point - surface.normal * radius;
                let contact_local = (surface.point + capsule_surface) * 0.5;
                cell_contacts.push_deepest_unique(Contact {
                    geom_a: idx_capsule,
                    geom_b: idx_hfield,
                    position_world: hfield_pose.point_to_world(contact_local),
                    normal_world: hfield_pose.rotate(surface.normal),
                    penetration,
                    friction,
                    gap,
                });
            }
            for &contact in cell_contacts.as_slice() {
                out.push_deepest(contact);
            }
        }
    }
}

/// Sphere vs MuJoCo heightfield. Each grid cell is decomposed into two
/// triangular prisms. Shared cell boundaries retain one manifold contact per
/// adjacent cell, while each cell removes its internal diagonal wall.
#[allow(clippy::too_many_arguments)]
pub fn sphere_hfield(
    idx_sphere: usize,
    sphere_pose: &GeomPose,
    radius: f32,
    idx_hfield: usize,
    hfield_pose: &GeomPose,
    hfield: &HeightField,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let mut out = ContactBuf::new();
    hfield_sphere_candidates(
        sphere_pose.position,
        radius,
        hfield_pose,
        hfield,
        friction,
        margin,
        gap,
        idx_sphere,
        idx_hfield,
        &mut out,
    );
    out
}

/// Capsule vs heightfield. The two spherical end caps are the same endpoint
/// decomposition used by the MuJoCo capsule-plane collider.
#[allow(clippy::too_many_arguments)]
pub fn capsule_hfield(
    idx_capsule: usize,
    capsule_pose: &GeomPose,
    radius: f32,
    half_height: f32,
    idx_hfield: usize,
    hfield_pose: &GeomPose,
    hfield: &HeightField,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let axis = capsule_pose.rotate(Vec3::Z);
    let start = capsule_pose.position + axis * half_height;
    let end = capsule_pose.position - axis * half_height;
    let mut out = ContactBuf::new();
    hfield_capsule_candidates(
        start,
        end,
        radius,
        hfield_pose,
        hfield,
        friction,
        margin,
        gap,
        idx_capsule,
        idx_hfield,
        &mut out,
    );
    out
}

/// Box vs heightfield. Every box vertex is tested against the closed prism
/// faces. The bounded output keeps the four deepest candidates.
#[allow(clippy::too_many_arguments)]
pub fn box_hfield(
    idx_box: usize,
    box_pose: &GeomPose,
    half_extents: Vec3,
    idx_hfield: usize,
    hfield_pose: &GeomPose,
    hfield: &HeightField,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let mut out = ContactBuf::new();
    let hfield_inverse = hfield_pose.orientation.conjugate();
    let box_pose_local = GeomPose {
        position: hfield_pose
            .orientation
            .inverse_rotate(box_pose.position - hfield_pose.position),
        orientation: hfield_inverse * box_pose.orientation,
    };
    for row in 0..hfield.nrow - 1 {
        for col in 0..hfield.ncol - 1 {
            for prism in hfield_prisms(hfield, row, col) {
                if let Some(mut contact) = ccd::box_prism_gjk_epa_contact(
                    &box_pose_local,
                    half_extents,
                    &prism,
                    idx_box,
                    idx_hfield,
                    friction,
                    margin,
                    gap,
                ) {
                    contact.position_world = hfield_pose.point_to_world(contact.position_world);
                    contact.normal_world = hfield_pose.rotate(contact.normal_world);
                    out.push_deepest_unique(contact);
                }
            }
        }
    }
    out
}

pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

// ---------------------------------------------------------------------------
// segment closest-point helpers (no libm)
// ---------------------------------------------------------------------------

/// Closest point on segment `[a, b]` to point `p`. `p` is projected onto the
/// segment and the projection parameter is clamped to `[0, 1]`.
pub fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let denom = ab.length_squared();
    if denom == 0.0 {
        return a;
    }
    let t = clamp01((p - a).dot(ab) / denom);
    a + ab * t
}

#[allow(clippy::manual_clamp)]
fn clamp01(x: f32) -> f32 {
    // Written manually so NaN passes through as NaN rather than being coerced
    // to a bound; `f32::clamp` panics on NaN comparisons.
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Closest points on two 3D line segments `[a0, a1]` and `[b0, b1]`.
///
/// Direct implementation of the standard clamp-and-recheck algorithm (see
/// Ericson, *Real-Time Collision Detection*, §5.1.9). Handles parallel and
/// zero-length segments deterministically. Returns `(pa, pb)` on the two
/// segments; the vector `pb - pa` is the shortest separation.
pub fn closest_points_on_segments(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (Vec3, Vec3) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);

    const EPS: f32 = 1.0e-12;

    // Degenerate: both segments are points.
    if a <= EPS && e <= EPS {
        return (a0, b0);
    }
    // Segment 1 is a point.
    if a <= EPS {
        let t = clamp01(f / e);
        return (a0, b0 + d2 * t);
    }
    let c = d1.dot(r);
    // Segment 2 is a point.
    if e <= EPS {
        let s = clamp01(-c / a);
        return (a0 + d1 * s, b0);
    }
    let b_ = d1.dot(d2);
    let denom = a * e - b_ * b_;

    let s = if denom != 0.0 {
        clamp01((b_ * f - c * e) / denom)
    } else {
        // Parallel; anchor at s = 0 and let t take over.
        0.0
    };

    let t_num = b_ * s + f;
    let (t, s_final) = if t_num < 0.0 {
        (0.0, clamp01(-c / a))
    } else if t_num > e {
        (1.0, clamp01((b_ - c) / a))
    } else {
        (t_num / e, s)
    };

    (a0 + d1 * s_final, b0 + d2 * t)
}

// ---------------------------------------------------------------------------
// Dispatch (broad→narrow) — one pair
// ---------------------------------------------------------------------------

/// Run the narrow phase for one ordered pair `(idx_a, idx_b)` of geoms. This
/// dispatch is symmetric: the caller may pass the pair in either order and the
/// result is normalized so `geom_a` in the emitted `Contact` matches the input
/// `idx_a`. The pair's shape combination picks a primitive; unsupported
/// combinations (see [`is_pair_supported`]) return an empty buffer — the
/// world validator is responsible for surfacing these to the user.
///
/// `meshes` is the world's mesh asset table, indexed by
/// [`GeomShape::Mesh::mesh_id`]. Pass an empty slice when no mesh geoms are
/// in play.
pub fn narrow_phase(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
    meshes: &[ConvexMesh],
) -> ContactBuf {
    narrow_phase_with_hfields(idx_a, geom_a, pose_a, idx_b, geom_b, pose_b, meshes, &[])
}

#[allow(clippy::too_many_arguments)]
pub fn narrow_phase_with_hfields(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
    meshes: &[ConvexMesh],
    hfields: &[HeightField],
) -> ContactBuf {
    dispatch_narrow_phase(
        idx_a,
        geom_a,
        pose_a,
        idx_b,
        geom_b,
        pose_b,
        meshes,
        hfields,
        NarrowPhaseMode::LegacyPenalty,
    )
}

/// Same shape-pair dispatch as [`narrow_phase`] but requests the
/// full-manifold box-box variant ([`box_box_full_manifold`]). Used by the
/// PGS solver pipeline so tilted face-face stacks receive the 4-corner
/// clipped polygon MuJoCo emits, instead of the 2-diagonal-corner
/// manifold the vertex-vs-face path returns as soon as microradian tilt
/// lifts two of the four corners above the reference face. Every other
/// shape pair dispatches to the same primitive [`narrow_phase`] uses —
/// the split is scoped to box-box on purpose (see docs/differential.md,
/// box_stack row).
pub fn narrow_phase_solver(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
    meshes: &[ConvexMesh],
) -> ContactBuf {
    narrow_phase_solver_with_hfields(idx_a, geom_a, pose_a, idx_b, geom_b, pose_b, meshes, &[])
}

#[allow(clippy::too_many_arguments)]
pub fn narrow_phase_solver_with_hfields(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
    meshes: &[ConvexMesh],
    hfields: &[HeightField],
) -> ContactBuf {
    dispatch_narrow_phase(
        idx_a,
        geom_a,
        pose_a,
        idx_b,
        geom_b,
        pose_b,
        meshes,
        hfields,
        NarrowPhaseMode::FullManifold,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NarrowPhaseMode {
    /// Legacy dispatch used by the penalty pipeline (box-box uses
    /// vertex-vs-face primary + edge-edge fallback). Keeps every pre-
    /// NEWT-14 penalty golden byte-for-byte.
    LegacyPenalty,
    /// PGS-pipeline dispatch: box-box always runs the SAT face-clipping
    /// manifold ([`box_box_full_manifold`]). Fixes the box_stack
    /// differential finding.
    FullManifold,
}

#[allow(clippy::too_many_arguments)]
fn dispatch_narrow_phase(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
    meshes: &[ConvexMesh],
    hfields: &[HeightField],
    mode: NarrowPhaseMode,
) -> ContactBuf {
    let friction = combine_friction(geom_a.friction, geom_b.friction);
    let margin = combine_max(geom_a.margin, geom_b.margin);
    let gap = combine_max(geom_a.gap, geom_b.gap);
    if ccd::route_pair(&geom_a.shape, &geom_b.shape) {
        // Run mesh-mesh CCD in stable geom-id order, then restore the
        // caller's labels and normal orientation.
        let swap = idx_a > idx_b;
        let mut buf = if swap {
            try_narrow_phase(
                idx_b, geom_b, pose_b, idx_a, geom_a, pose_a, friction, margin, gap, meshes,
                hfields, mode,
            )
        } else {
            try_narrow_phase(
                idx_a, geom_a, pose_a, idx_b, geom_b, pose_b, friction, margin, gap, meshes,
                hfields, mode,
            )
        }
        .expect("ccd route shapes must be finite convex geoms");
        if swap {
            for c in buf.contacts.iter_mut().take(buf.len) {
                std::mem::swap(&mut c.geom_a, &mut c.geom_b);
                c.normal_world = -c.normal_world;
            }
        }
        return buf;
    }
    // Try in the order given; if that combination isn't a known primitive,
    // swap and dispatch, then relabel the results (flipping normal and A/B).
    //
    // Invariant: for any two DIFFERENT shape kinds, at most one of
    // `(A, B)` or `(B, A)` matches a primitive arm. Symmetric same-kind
    // pairs (sphere-sphere, box-box, capsule-capsule) match both trivially
    // and short-circuit at the first arm below, so the swap arm never runs
    // — safe. The one-directional invariant matters only for asymmetric
    // pairs: if a NEW primitive is added and accidentally registered under
    // BOTH `(X, Y)` and `(Y, X)`, the swap relabel (flipping A/B and
    // negating the normal) would double-count with opposite sign. The
    // debug_assert on the first arm guards against that regression for
    // asymmetric callers.
    let first = try_narrow_phase(
        idx_a, geom_a, pose_a, idx_b, geom_b, pose_b, friction, margin, gap, meshes, hfields, mode,
    );
    if let Some(buf) = first {
        debug_assert!(
            std::mem::discriminant(&geom_a.shape) == std::mem::discriminant(&geom_b.shape)
                || try_narrow_phase(
                    idx_b, geom_b, pose_b, idx_a, geom_a, pose_a, friction, margin, gap, meshes,
                    hfields, mode,
                )
                .is_none(),
            "narrow_phase invariant violated: an asymmetric shape pair matched a primitive \
             in BOTH orderings — a new primitive is registered against both `(A, B)` and \
             `(B, A)` combinations"
        );
        return buf;
    }
    if let Some(mut buf) = try_narrow_phase(
        idx_b, geom_b, pose_b, idx_a, geom_a, pose_a, friction, margin, gap, meshes, hfields, mode,
    ) {
        for c in buf.contacts.iter_mut().take(buf.len) {
            std::mem::swap(&mut c.geom_a, &mut c.geom_b);
            c.normal_world = -c.normal_world;
        }
        return buf;
    }
    ContactBuf::new()
}

/// One-direction dispatch. Returns `Some(buf)` if the (A, B) shape pair is a
/// known primitive; `None` if not (caller may try swapping).
#[allow(clippy::too_many_arguments)]
fn try_narrow_phase(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
    friction: f32,
    margin: f32,
    gap: f32,
    meshes: &[ConvexMesh],
    hfields: &[HeightField],
    mode: NarrowPhaseMode,
) -> Option<ContactBuf> {
    if ccd::route_pair(&geom_a.shape, &geom_b.shape) {
        let shape_a = ccd::shape(&geom_a.shape, pose_a, meshes)?;
        let shape_b = ccd::shape(&geom_b.shape, pose_b, meshes)?;
        let mut out = ContactBuf::new();
        if let Some(contact) = ccd::ccd_convex_contact(
            shape_a,
            shape_b,
            ccd::CCD_MESH_CONFIG,
            idx_a,
            idx_b,
            friction,
            margin,
            gap,
        ) {
            out.push(contact);
        }
        return Some(out);
    }
    Some(match (geom_a.shape, geom_b.shape) {
        (GeomShape::Sphere { radius }, GeomShape::Plane) => sphere_plane(
            idx_a, pose_a, radius, friction, margin, gap, idx_b, geom_b, pose_b,
        ),
        (GeomShape::Box { half_extents }, GeomShape::Plane) => box_plane(
            idx_a,
            pose_a,
            half_extents,
            friction,
            margin,
            gap,
            idx_b,
            geom_b,
            pose_b,
        ),
        (
            GeomShape::Capsule {
                radius,
                half_height,
            },
            GeomShape::Plane,
        ) => capsule_plane(
            idx_a,
            pose_a,
            radius,
            half_height,
            friction,
            margin,
            gap,
            idx_b,
            geom_b,
            pose_b,
        ),
        (
            GeomShape::Cylinder {
                radius,
                half_height,
            },
            GeomShape::Plane,
        ) => cylinder_plane(
            idx_a,
            pose_a,
            radius,
            half_height,
            friction,
            margin,
            gap,
            idx_b,
            geom_b,
            pose_b,
        ),
        // MuJoCo calls mjc_PlaneConvex for plane-ellipsoid. The analytic
        // support point and midpoint construction match that route.
        (GeomShape::Ellipsoid { semi_axes }, GeomShape::Plane) => ellipsoid_plane(
            idx_a, pose_a, semi_axes, friction, margin, gap, idx_b, geom_b, pose_b,
        ),
        // MuJoCo calls mjc_PlaneConvex for plane-mesh. The support manifold
        // keeps the two deepest support vertices in stable source order.
        (GeomShape::Mesh { mesh_id }, GeomShape::Plane) => mesh_plane(
            idx_a,
            pose_a,
            &meshes[mesh_id],
            friction,
            margin,
            gap,
            idx_b,
            geom_b,
            pose_b,
        ),
        (GeomShape::Sphere { radius: ra }, GeomShape::Sphere { radius: rb }) => {
            sphere_sphere(idx_a, pose_a, ra, idx_b, pose_b, rb, friction, margin, gap)
        }
        (
            GeomShape::Sphere { radius: rs },
            GeomShape::Capsule {
                radius: rc,
                half_height,
            },
        ) => sphere_capsule(
            idx_a,
            pose_a,
            rs,
            idx_b,
            pose_b,
            rc,
            half_height,
            friction,
            margin,
            gap,
        ),
        (
            GeomShape::Sphere { radius: rs },
            GeomShape::Cylinder {
                radius: rc,
                half_height: hc,
            },
        ) => sphere_cylinder(
            idx_a, pose_a, rs, idx_b, pose_b, rc, hc, friction, margin, gap,
        ),
        // MuJoCo calls mjc_Convex for sphere-ellipsoid.
        (GeomShape::Sphere { radius: rs }, GeomShape::Ellipsoid { semi_axes }) => sphere_ellipsoid(
            idx_a, pose_a, rs, idx_b, pose_b, semi_axes, friction, margin, gap,
        ),
        // MuJoCo calls mjc_Convex for sphere-mesh.
        (GeomShape::Sphere { radius: rs }, GeomShape::Mesh { mesh_id }) => sphere_mesh(
            idx_a,
            pose_a,
            rs,
            idx_b,
            pose_b,
            &meshes[mesh_id],
            friction,
            margin,
            gap,
        ),
        (GeomShape::Sphere { radius: rs }, GeomShape::Hfield { hfield_id }) => sphere_hfield(
            idx_a,
            pose_a,
            rs,
            idx_b,
            pose_b,
            &hfields[hfield_id],
            friction,
            margin,
            gap,
        ),
        (
            GeomShape::Capsule {
                radius: ra,
                half_height: ha,
            },
            GeomShape::Capsule {
                radius: rb,
                half_height: hb,
            },
        ) => capsule_capsule(
            idx_a, pose_a, ra, ha, idx_b, pose_b, rb, hb, friction, margin, gap,
        ),
        (
            GeomShape::Box {
                half_extents: half_a,
            },
            GeomShape::Box {
                half_extents: half_b,
            },
        ) => match mode {
            NarrowPhaseMode::LegacyPenalty => box_box(
                idx_a, pose_a, half_a, idx_b, pose_b, half_b, friction, margin, gap,
            ),
            NarrowPhaseMode::FullManifold => box_box_full_manifold(
                idx_a, pose_a, half_a, idx_b, pose_b, half_b, friction, margin, gap,
            ),
        },
        (
            GeomShape::Capsule {
                radius,
                half_height,
            },
            GeomShape::Hfield { hfield_id },
        ) => capsule_hfield(
            idx_a,
            pose_a,
            radius,
            half_height,
            idx_b,
            pose_b,
            &hfields[hfield_id],
            friction,
            margin,
            gap,
        ),
        (GeomShape::Box { half_extents }, GeomShape::Hfield { hfield_id }) => box_hfield(
            idx_a,
            pose_a,
            half_extents,
            idx_b,
            pose_b,
            &hfields[hfield_id],
            friction,
            margin,
            gap,
        ),
        // Deferred pairs are not in the support matrix.
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::Body;
    use crate::geom::{Geom, HeightField, geom_world_pose};
    use crate::math::{Quat, Vec3};
    use crate::solver::SolverMode;
    use crate::world::{Integrator, World};

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn approx_vec(a: Vec3, b: Vec3, tol: f32) -> bool {
        approx(a.x, b.x, tol) && approx(a.y, b.y, tol) && approx(a.z, b.z, tol)
    }

    #[test]
    fn combine_friction_takes_min() {
        assert_eq!(combine_friction(0.3, 0.7), 0.3);
        assert_eq!(combine_friction(0.5, 0.5), 0.5);
    }

    #[test]
    fn sphere_plane_hits_and_penetration_is_signed_depth() {
        // Sphere radius 1 at height 0.6 above plane z=0 → penetration 0.4.
        let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
        let plane_pose = geom_world_pose(&plane, Vec3::ZERO, Quat::IDENTITY);
        let _sphere = Geom::sphere(0, 1.0, Vec3::ZERO, 1.0);
        let sphere_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.6),
            orientation: Quat::IDENTITY,
        };
        let buf = sphere_plane(0, &sphere_pose, 1.0, 1.0, 0.0, 0.0, 1, &plane, &plane_pose);
        assert_eq!(buf.len, 1);
        let c = buf.as_slice()[0];
        assert!(approx(c.penetration, 0.4, 1e-5));
        assert!(approx_vec(c.normal_world, Vec3::Z, 1e-6));
        assert!(approx_vec(
            c.position_world,
            Vec3::new(0.0, 0.0, -0.2),
            1e-5
        ));
    }

    #[test]
    fn sphere_plane_misses_when_above() {
        let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
        let plane_pose = geom_world_pose(&plane, Vec3::ZERO, Quat::IDENTITY);
        let sphere_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 2.0),
            orientation: Quat::IDENTITY,
        };
        let buf = sphere_plane(0, &sphere_pose, 1.0, 1.0, 0.0, 0.0, 1, &plane, &plane_pose);
        assert_eq!(buf.len, 0);
    }

    fn flat_hfield() -> HeightField {
        HeightField {
            nrow: 2,
            ncol: 2,
            size: [1.0, 1.0, 1.0, 0.2],
            data: vec![0.0; 4],
        }
    }

    #[test]
    fn hfield_flat_sphere_matches_plane_anchor() {
        let field = flat_hfield();
        let field_pose = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let sphere_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.4),
            orientation: Quat::IDENTITY,
        };
        let buf = sphere_hfield(0, &sphere_pose, 0.5, 1, &field_pose, &field, 0.5, 0.0, 0.0);
        assert_eq!(buf.len, 1);
        assert!(approx(buf.contacts[0].penetration, 0.1, 1e-6));
        assert!(approx_vec(buf.contacts[0].normal_world, Vec3::Z, 1e-6));
        assert!(approx_vec(
            buf.contacts[0].position_world,
            Vec3::new(0.0, 0.0, -0.05),
            1e-6
        ));
        assert!(buf.candidate_count >= 2);
    }

    #[test]
    fn hfield_ramp_capsule_and_box_have_prism_normals() {
        let field = HeightField {
            nrow: 2,
            ncol: 2,
            size: [1.0, 1.0, 1.0, 0.2],
            data: vec![0.0, 0.0, 0.5, 0.5],
        };
        let field_pose = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let capsule_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.35),
            orientation: Quat::IDENTITY,
        };
        let capsule = capsule_hfield(
            0,
            &capsule_pose,
            0.25,
            0.0,
            1,
            &field_pose,
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert!(!capsule.as_slice().is_empty());
        assert!(capsule.as_slice().iter().all(|c| c.normal_world.z > 0.0));

        let box_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.25),
            orientation: Quat::IDENTITY,
        };
        let box_contacts = box_hfield(
            0,
            &box_pose,
            Vec3::splat(0.2),
            1,
            &field_pose,
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert!(!box_contacts.as_slice().is_empty());
        assert!(
            box_contacts
                .as_slice()
                .iter()
                .all(|c| (c.normal_world.length() - 1.0).abs() < 1e-5)
        );
        assert!(
            box_contacts
                .as_slice()
                .iter()
                .any(|c| c.normal_world.z > 0.0)
        );
    }

    fn identity_pose(position: Vec3) -> GeomPose {
        GeomPose {
            position,
            orientation: Quat::IDENTITY,
        }
    }

    #[test]
    fn hfield_internal_crease_has_no_wall_contact() {
        let field = flat_hfield();
        let contacts = sphere_hfield(
            0,
            &identity_pose(Vec3::new(0.02, 0.02, -0.05)),
            0.1,
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(contacts.len, 1, "{contacts:?}");
        assert!(
            contacts
                .as_slice()
                .iter()
                .all(|contact| contact.normal_world.z > 0.99),
            "{contacts:?}"
        );
    }

    #[test]
    fn hfield_cell_boundary_continuity_sweep_kills_vertical_sampling_mutant() {
        let field = HeightField {
            nrow: 2,
            ncol: 3,
            size: [1.5, 1.0, 1.0, 0.2],
            data: vec![0.2; 6],
        };
        let field_pose = identity_pose(Vec3::ZERO);
        let left = sphere_hfield(
            0,
            &identity_pose(Vec3::new(-1.0e-4, 0.0, 0.35)),
            0.2,
            1,
            &field_pose,
            &field,
            0.5,
            0.0,
            0.0,
        );
        let right = sphere_hfield(
            0,
            &identity_pose(Vec3::new(1.0e-4, 0.0, 0.35)),
            0.2,
            1,
            &field_pose,
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(left.len, 2);
        assert_eq!(right.len, 2);
        for (l, r) in left.as_slice().iter().zip(right.as_slice()) {
            assert!((l.penetration - r.penetration).abs() < 1.0e-5);
            assert!((l.position_world.z - r.position_world.z).abs() < 1.0e-5);
            assert!(l.normal_world.z > 0.999);
            assert!(r.normal_world.z > 0.999);
        }
    }

    #[test]
    fn hfield_saddle_and_steep_faces_kill_vertical_sampling_mutant() {
        let field = HeightField {
            nrow: 2,
            ncol: 2,
            size: [1.0, 1.0, 1.0, 0.2],
            data: vec![0.0, 1.0, 1.0, 0.0],
        };
        let field_pose = identity_pose(Vec3::ZERO);
        let saddle = sphere_hfield(
            0,
            &identity_pose(Vec3::new(0.0, 0.0, 0.35)),
            0.3,
            1,
            &field_pose,
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(saddle.len, 2);
        let saddle_expected = [
            (
                Vec3::new(-0.1195706, 0.1195706, 0.1108588),
                Vec3::new(0.4082483, -0.4082483, 0.8164966),
            ),
            (
                Vec3::new(0.1195706, -0.1195706, 0.1108588),
                Vec3::new(-0.4082483, 0.4082483, 0.8164966),
            ),
        ];
        for (contact, (position, normal)) in saddle.as_slice().iter().zip(saddle_expected) {
            assert!(approx_vec(contact.position_world, position, 2.0e-5));
            assert!(approx_vec(contact.normal_world, normal, 2.0e-5));
            assert!((contact.penetration - 0.0142262).abs() < 2.0e-5);
        }

        let steep_field = HeightField {
            data: vec![0.0, 1.0, 0.0, 1.0],
            ..field
        };
        let steep = sphere_hfield(
            0,
            &identity_pose(Vec3::new(0.0, 0.0, 0.7)),
            0.3,
            1,
            &field_pose,
            &steep_field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(steep.len, 2);
        assert!(steep.as_slice().iter().all(|c| c.normal_world.z > 0.8));
        assert!(steep.as_slice().iter().any(|c| c.normal_world.x < -0.4));
        assert!((steep.contacts[0].position_world.x - 0.107082).abs() < 2.0e-5);
        assert!((steep.contacts[0].position_world.z - 0.485836).abs() < 2.0e-5);
        assert!((steep.contacts[0].penetration - 0.1211146).abs() < 2.0e-5);
        assert!((steep.contacts[1].position_world.x - 0.057578).abs() < 2.0e-5);
        assert!((steep.contacts[1].position_world.y - 0.057578).abs() < 2.0e-5);
        assert!((steep.contacts[1].position_world.z - 0.469690).abs() < 2.0e-5);
    }

    #[test]
    fn hfield_ridge_capsule_uses_segment_sweep() {
        let field = HeightField {
            nrow: 2,
            ncol: 2,
            size: [1.0, 1.0, 1.0, 0.2],
            data: vec![0.0, 1.0, 1.0, 0.0],
        };
        let contacts = capsule_hfield(
            0,
            &identity_pose(Vec3::new(0.0, 0.0, 0.45)),
            0.1,
            0.35,
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(contacts.len, 2);
        let ridge_expected = [
            (
                Vec3::new(-0.0370791, 0.0370791, 0.0258418),
                Vec3::new(0.4082483, -0.4082483, 0.8164966),
            ),
            (
                Vec3::new(0.0370791, -0.0370791, 0.0258418),
                Vec3::new(-0.4082483, 0.4082483, 0.8164966),
            ),
        ];
        for (contact, (position, normal)) in contacts.as_slice().iter().zip(ridge_expected) {
            assert!(approx_vec(contact.position_world, position, 2.0e-5));
            assert!(approx_vec(contact.normal_world, normal, 2.0e-5));
            assert!((contact.penetration - 0.0183503).abs() < 2.0e-5);
        }
    }

    #[test]
    fn hfield_base_side_fixture_kills_base_face_mutant() {
        let field = flat_hfield();
        let contacts = sphere_hfield(
            0,
            &identity_pose(Vec3::new(1.0, 0.0, -0.1)),
            0.05,
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(contacts.len, 1);
        let contact = contacts.contacts[0];
        assert!(contact.normal_world.x > 0.99);
        assert!((contact.position_world.x - 0.975).abs() < 1.0e-5);
        assert!((contact.penetration - 0.05).abs() < 1.0e-5);
    }

    #[test]
    fn hfield_base_center_fixture_kills_base_face_mutant() {
        let field = flat_hfield();
        let contacts = sphere_hfield(
            0,
            &identity_pose(Vec3::new(0.0, 0.0, -0.25)),
            0.1,
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(contacts.len, 1, "{contacts:?}");
        let contact = contacts.contacts[0];
        assert!(contact.normal_world.z < -0.99, "{contact:?}");
        assert!((contact.position_world.z + 0.175).abs() < 1.0e-5);
        assert!((contact.penetration - 0.05).abs() < 1.0e-5);
    }

    #[test]
    fn hfield_outside_footprint_fixture_kills_side_projection_mutant() {
        let field = flat_hfield();
        let contacts = sphere_hfield(
            0,
            &identity_pose(Vec3::new(1.04, 0.0, 0.0)),
            0.05,
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(contacts.len, 1);
        let contact = contacts.contacts[0];
        assert!(contact.normal_world.x > 0.99);
        assert!((contact.position_world.x - 0.995).abs() < 1.0e-5);
        assert!((contact.penetration - 0.01).abs() < 1.0e-5);
    }

    #[test]
    fn hfield_contact_cap_mutant_is_killed_by_deepest_four_fixture() {
        let field = HeightField {
            nrow: 3,
            ncol: 3,
            size: [1.0, 1.0, 1.0, 0.2],
            data: vec![0.0, 0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.4, 0.6],
        };
        let pose = identity_pose(Vec3::new(0.0, 0.0, 0.15));
        let first = box_hfield(
            0,
            &pose,
            Vec3::new(0.8, 0.8, 0.2),
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        let second = box_hfield(
            0,
            &pose,
            Vec3::new(0.8, 0.8, 0.2),
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert!(first.candidate_count > 4, "{first:?}");
        assert!(first.unique_candidate_count > 4, "{first:?}");
        assert_eq!(first.len, 4, "{first:?}");
        assert_eq!(first.contacts, second.contacts);
        assert!(
            first
                .as_slice()
                .windows(2)
                .all(|w| w[0].penetration >= w[1].penetration)
        );
    }

    #[test]
    fn hfield_in_step_onset_is_captured_once() {
        let mut world = World::new();
        world.dt = 0.1;
        world.integrator = Integrator::Euler;
        world.solver.mode = SolverMode::Pgs;
        world.gravity = Vec3::ZERO;
        world.add_hfield(flat_hfield());
        world.add_geom(Geom::static_hfield(0, Vec3::ZERO, Quat::IDENTITY, 0.5));
        let body = world.add_body(Body::solid_sphere(
            1.0,
            0.2,
            Vec3::new(0.0, 0.0, 0.25),
            Quat::IDENTITY,
        ));
        world.bodies[body].linear_velocity.z = -0.5;
        world.add_geom(Geom::sphere(body, 0.2, Vec3::ZERO, 0.5));
        world.set_solver_phase_capture(true);
        world.capture_solver_phase();
        assert!(
            world
                .solver_phase_diagnostics()
                .unwrap()
                .free_body_contacts
                .is_empty()
        );
        world.step();
        assert!(
            world
                .solver_phase_diagnostics()
                .unwrap()
                .free_body_contacts
                .is_empty()
        );
        assert!(world.detect_contacts().is_empty());
        world.step();
        assert!(
            world
                .solver_phase_diagnostics()
                .unwrap()
                .free_body_contacts
                .is_empty()
        );
        assert_eq!(world.detect_contacts().len(), 1);
        world.step();
        let phase = world.solver_phase_diagnostics().unwrap();
        assert_eq!(phase.free_body_contacts.len(), 1);
        assert_eq!(phase.contacts.len(), 0);
        assert!(world.contact_detection_count() >= 4);
    }

    #[test]
    fn hfield_normal_orientation_fixture_kills_normal_flip_mutant() {
        let field = flat_hfield();
        let contact = sphere_hfield(
            0,
            &identity_pose(Vec3::new(0.0, 0.0, 0.15)),
            0.2,
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        )
        .contacts[0];
        assert!(contact.normal_world.dot(Vec3::Z) > 0.99);
    }

    #[test]
    fn native_ccd_box_prism_fixture_kills_epa_face_and_sign_mutants() {
        let field = HeightField {
            nrow: 2,
            ncol: 2,
            size: [1.0, 1.0, 1.0, 0.2],
            data: vec![0.0, 1.0, 0.0, 1.0],
        };
        let pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.45),
            orientation: Quat::from_axis_angle(Vec3::Y, 0.4),
        };
        let contacts = box_hfield(
            0,
            &pose,
            Vec3::splat(0.25),
            1,
            &identity_pose(Vec3::ZERO),
            &field,
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(contacts.len, 2);
        assert!(contacts.contacts[0].penetration > contacts.contacts[1].penetration);
        assert!(approx(contacts.contacts[0].penetration, 0.3874867, 2.0e-5));
        assert!(approx(contacts.contacts[1].penetration, 0.3649474, 2.0e-5));
        assert!(contacts.as_slice().iter().all(|contact| {
            contact.normal_world.length() > 0.99999 && contact.penetration > 0.0
        }));
        assert_eq!(
            contacts.contacts,
            box_hfield(
                0,
                &pose,
                Vec3::splat(0.25),
                1,
                &identity_pose(Vec3::ZERO),
                &field,
                0.5,
                0.0,
                0.0,
            )
            .contacts
        );
    }

    #[test]
    fn gjk_tetrahedron_enclosure_kills_simplex_termination_mutant() {
        let point = |x, y, z| CcdVertex {
            minkowski: Vec3::new(x, y, z),
            shape_a: Vec3::ZERO,
            shape_b: Vec3::ZERO,
            tie_a: false,
            tie_b: false,
        };
        let mut simplex = CcdSimplex {
            points: [
                point(1.0, 1.0, 1.0),
                point(-1.0, -1.0, 1.0),
                point(1.0, -1.0, -1.0),
                point(-1.0, 1.0, -1.0),
            ],
            len: 4,
        };
        let mut direction = Vec3::X;
        assert!(ccd_simplex_step(&mut simplex, &mut direction));
    }

    #[test]
    fn box_plane_emits_four_corner_contacts_when_flat_on_plane() {
        // Box 1x1x1 at height 0.4 → four bottom corners penetrate by 0.1.
        let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
        let plane_pose = geom_world_pose(&plane, Vec3::ZERO, Quat::IDENTITY);
        let box_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.4),
            orientation: Quat::IDENTITY,
        };
        let hx = Vec3::splat(0.5);
        let buf = box_plane(0, &box_pose, hx, 1.0, 0.0, 0.0, 1, &plane, &plane_pose);
        assert_eq!(buf.len, 4);
        for c in buf.as_slice() {
            assert!(approx(c.penetration, 0.1, 1e-5));
            assert!(approx_vec(c.normal_world, Vec3::Z, 1e-6));
        }
    }

    #[test]
    fn box_plane_emits_two_contacts_when_tilted_edge_down() {
        // Box tilted 45° about X so a single bottom EDGE (two corners) is the
        // lowest. Only those two corners penetrate.
        let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
        let plane_pose = geom_world_pose(&plane, Vec3::ZERO, Quat::IDENTITY);
        let angle = crate::math::FRAC_PI_4;
        let box_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.4),
            orientation: Quat::from_axis_angle(Vec3::X, angle),
        };
        // Half-diagonal = 0.5*sqrt(2) ≈ 0.707. Position 0.4 → the lowest edge
        // is at z ≈ 0.4 - 0.707 = -0.307. Two corners at penetration ~0.307.
        let hx = Vec3::splat(0.5);
        let buf = box_plane(0, &box_pose, hx, 1.0, 0.0, 0.0, 1, &plane, &plane_pose);
        assert_eq!(buf.len, 2);
    }

    #[test]
    fn sphere_sphere_normal_points_from_b_into_a() {
        // Two unit spheres with centers along +X.
        let pose_a = GeomPose {
            position: Vec3::new(0.5, 0.0, 0.0),
            orientation: Quat::IDENTITY,
        };
        let pose_b = GeomPose {
            position: Vec3::new(-0.5, 0.0, 0.0),
            orientation: Quat::IDENTITY,
        };
        let buf = sphere_sphere(0, &pose_a, 1.0, 1, &pose_b, 1.0, 1.0, 0.0, 0.0);
        assert_eq!(buf.len, 1);
        let c = buf.as_slice()[0];
        assert!(approx(c.penetration, 1.0, 1e-5));
        // Normal from B (-x) to A (+x): +X direction.
        assert!(approx_vec(c.normal_world, Vec3::X, 1e-6));
    }

    #[test]
    fn box_box_axis_aligned_stack_gives_four_bottom_corner_contacts() {
        // Upper box (unit-cube-ish) resting on lower box, both axis-aligned.
        // Upper's 4 bottom corners penetrate lower's top face by ~0.05.
        let upper_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.75),
            orientation: Quat::IDENTITY,
        };
        let lower_pose = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let hu = Vec3::splat(0.3);
        let hl = Vec3::splat(0.5);
        let buf = box_box(0, &upper_pose, hu, 1, &lower_pose, hl, 0.5, 0.0, 0.0);
        assert_eq!(buf.len, 4);
        for c in buf.as_slice() {
            // Upper bottom = 0.75 − 0.3 = 0.45. Lower top = 0 + 0.5 = 0.5.
            // Overlap = 0.05 in +Z of lower box.
            assert!((c.penetration - 0.05).abs() < 1e-5);
            assert!(approx_vec(c.normal_world, Vec3::Z, 1e-5));
        }
    }

    #[test]
    fn box_box_disjoint_returns_no_contacts() {
        let a = GeomPose {
            position: Vec3::new(0.0, 0.0, 5.0),
            orientation: Quat::IDENTITY,
        };
        let b = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let buf = box_box(
            0,
            &a,
            Vec3::splat(0.5),
            1,
            &b,
            Vec3::splat(0.5),
            0.5,
            0.0,
            0.0,
        );
        assert_eq!(buf.len, 0);
    }

    #[test]
    fn capsule_plane_two_endpoints_when_axis_horizontal() {
        let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
        let plane_pose = geom_world_pose(&plane, Vec3::ZERO, Quat::IDENTITY);
        // Axis along X (rotate 90° about Y so local Z → world X).
        let capsule_pose = GeomPose {
            position: Vec3::new(0.0, 0.0, 0.3),
            orientation: Quat::from_axis_angle(Vec3::Y, crate::math::FRAC_PI_2),
        };
        // Radius 0.5, half_height 1.0. Both endpoints at z=0.3, penetration
        // 0.5-0.3 = 0.2 each. MuJoCo orders the positive endpoint first and
        // places each contact at the cap-plane midpoint z=-0.1.
        let buf = capsule_plane(
            0,
            &capsule_pose,
            0.5,
            1.0,
            1.0,
            0.0,
            0.0,
            1,
            &plane,
            &plane_pose,
        );
        assert_eq!(buf.len, 2);
        let contacts = buf.as_slice();
        assert!(approx(contacts[0].position_world.x, 1.0, 1e-5));
        assert!(approx(contacts[1].position_world.x, -1.0, 1e-5));
        for c in contacts {
            assert!(approx(c.penetration, 0.2, 1e-5));
            assert!(approx(c.position_world.z, -0.1, 1e-5));
        }
    }

    #[test]
    fn ccd_support_functions_preserve_shape_axes() {
        let rotated_pose = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::from_axis_angle(Vec3::Y, 0.7),
        };
        let vertices = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        let vertices_shape = CcdShape::Vertices(&vertices);
        assert!(vertices_shape.support(Vec3::Y).y > 0.9);

        let mesh = ConvexMesh {
            vertices: vertices.to_vec(),
            faces: vec![[0, 1, 2], [0, 3, 1]],
        };
        let mesh_shape = CcdShape::Mesh {
            pose: &rotated_pose,
            mesh: &mesh,
        };
        let rotated_axis = rotated_pose.rotate(Vec3::Y);
        assert!(mesh_shape.support(rotated_axis).dot(rotated_axis) > 0.9);
        assert_eq!(vertices_shape.center(), Vec3::splat(-0.5));
        assert_eq!(mesh_shape.center(), Vec3::ZERO);
    }

    #[test]
    fn closest_point_on_segment_clamps() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        // Beyond b: clamps to b.
        assert_eq!(closest_point_on_segment(Vec3::new(5.0, 0.0, 0.0), a, b), b);
        // Before a: clamps to a.
        assert_eq!(closest_point_on_segment(Vec3::new(-1.0, 0.5, 0.0), a, b), a);
        // Perpendicular projection lands on interior.
        let p = closest_point_on_segment(Vec3::new(1.0, 1.0, 0.0), a, b);
        assert!(approx_vec(p, Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn closest_points_on_segments_handles_perpendicular_case() {
        // Two segments along X and Y at z=1 apart. Closest points at their
        // midpoints and the separation is (0,0,1).
        let a0 = Vec3::new(-1.0, 0.0, 0.0);
        let a1 = Vec3::new(1.0, 0.0, 0.0);
        let b0 = Vec3::new(0.0, -1.0, 1.0);
        let b1 = Vec3::new(0.0, 1.0, 1.0);
        let (pa, pb) = closest_points_on_segments(a0, a1, b0, b1);
        assert!(approx_vec(pa, Vec3::new(0.0, 0.0, 0.0), 1e-5));
        assert!(approx_vec(pb, Vec3::new(0.0, 0.0, 1.0), 1e-5));
    }

    #[test]
    fn closest_points_on_segments_parallel_case_returns_valid_pair() {
        // Parallel offset segments. Any two points that share a common
        // perpendicular are valid; check the separation vector is
        // perpendicular to both axes.
        let a0 = Vec3::new(0.0, 0.0, 0.0);
        let a1 = Vec3::new(1.0, 0.0, 0.0);
        let b0 = Vec3::new(0.3, 0.7, 0.0);
        let b1 = Vec3::new(1.3, 0.7, 0.0);
        let (pa, pb) = closest_points_on_segments(a0, a1, b0, b1);
        let sep = pb - pa;
        assert!(approx(sep.dot(a1 - a0), 0.0, 1e-4));
        assert!(approx(sep.dot(b1 - b0), 0.0, 1e-4));
        assert!(approx(sep.length(), 0.7, 1e-5));
    }
}
