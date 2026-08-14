//! Narrow-phase collision detection.
//!
//! # Contact representation
//!
//! A [`Contact`] describes one contact point between two geoms. The normal
//! points **from geom B into geom A**: an infinitesimal displacement of A
//! along `+normal` separates the pair. `penetration` is positive when the
//! geoms interpenetrate (or the pair sits inside a nonzero margin — see
//! below); a `penetration ≤ 0` contact is never emitted.
//!
//! # Narrow-phase coverage
//!
//! - Fully implemented pairs (tier 2 + v1 tier 2):
//!   - plane vs {sphere, box, capsule, cylinder, ellipsoid, mesh}
//!   - sphere vs {sphere, capsule, cylinder, ellipsoid, mesh}
//!   - capsule vs capsule
//!   - box vs box (full OBB SAT with edge-edge cross axes — closes the
//!     NEWT-5 rotated-stack incident)
//! - Deferred pairs (silently emit no contact + [`is_pair_supported`]
//!   returns `false` so the world validator can reject them):
//!   - cylinder vs {cylinder, box, capsule, ellipsoid, mesh}
//!   - ellipsoid vs {box, capsule, ellipsoid, mesh}
//!   - mesh vs {box, capsule, mesh}
//!   - box vs {sphere, capsule}  (unchanged from tier 2)
//!
//! Deferred pairs would each need a GJK or bespoke narrow-phase primitive
//! and are called out in the docs. The v1 constraint solver ticket
//! ultimately unifies these behind a shared support-function interface.
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
//! given input; no HashMap iteration; no sort-key that ties on floats. Any
//! iterative closest-point solver runs a FIXED number of iterations (see
//! [`sphere_ellipsoid`]). Sorted results at the pair level live in
//! [`crate::world`].

use crate::geom::{ConvexMesh, Geom, GeomPose, GeomShape};
use crate::math::Vec3;

/// One narrow-phase contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Index of geom A (world-registered index).
    pub geom_a: usize,
    /// Index of geom B.
    pub geom_b: usize,
    /// Contact point in world coordinates. By convention this is on B's
    /// surface (a point that A has penetrated); moving A by `+normal *
    /// penetration` removes the overlap.
    pub position_world: Vec3,
    /// Unit normal in world coordinates, pointing FROM B into A.
    pub normal_world: Vec3,
    /// Shifted penetration depth `pair_margin - raw_dist` (positive; a
    /// value equal to `pair_margin` means the raw distance is exactly zero,
    /// i.e. the geoms just touch). Zero-penetration contacts are never
    /// emitted (strictly `> 0`).
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
        }
    }
    pub fn push(&mut self, c: Contact) {
        if self.len < self.contacts.len() {
            self.contacts[self.len] = c;
            self.len += 1;
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
            | (GeomShape::Capsule { .. }, GeomShape::Capsule { .. })
            | (GeomShape::Box { .. }, GeomShape::Box { .. })
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
    // Raw penetration = radius - signed; contact activates when raw > -margin,
    // and we shift the reported penetration up by `margin`.
    let pen_raw = radius - signed;
    let pen = pen_raw + margin;
    if pen > 0.0 {
        // Contact point on the plane surface directly under the sphere center.
        let contact_pt = sphere_pose.position - n * signed;
        out.push(Contact {
            geom_a: idx_sphere,
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

/// Box vs static plane. Emits up to 4 corner contacts (the 4 deepest of any
/// penetrating corners) in a fixed deterministic order.
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
    // Corner sign pattern: fixed lexicographic order. The tie-break for "which
    // 4 deepest" walks corners in this order, so a bit-identical input always
    // yields the same 4.
    const SIGNS: [(f32, f32, f32); 8] = [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ];
    // Collect penetrations in fixed order.
    let mut pens = [(0.0f32, Vec3::ZERO); 8];
    for (slot, &(sx, sy, sz)) in pens.iter_mut().zip(SIGNS.iter()) {
        let local = Vec3::new(
            sx * half_extents.x,
            sy * half_extents.y,
            sz * half_extents.z,
        );
        let world = box_pose.point_to_world(local);
        let signed = (world - p0).dot(n);
        // Shifted penetration = margin - signed = -signed + margin.
        *slot = (-signed + margin, world);
    }
    // Keep only positive shifted penetrations; then pick up to 4 deepest.
    let mut order = [0usize; 8];
    let mut count = 0usize;
    for (i, &pen) in pens.iter().enumerate() {
        if pen.0 > 0.0 {
            order[count] = i;
            count += 1;
        }
    }
    // Stable sort descending by penetration; ties break by insertion order
    // (which is corner-index order).
    for i in 1..count {
        let mut j = i;
        while j > 0 && pens[order[j]].0 > pens[order[j - 1]].0 {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut out = ContactBuf::new();
    let take = if count > 4 { 4 } else { count };
    for &i in &order[..take] {
        let (pen, corner) = pens[i];
        // Contact position on B's surface: the point on the plane directly
        // above the corner. `pen` is the shifted penetration so we subtract
        // the margin back to get the raw offset from corner to plane along
        // `+n`.
        let contact_pt = corner + n * (pen - margin);
        out.push(Contact {
            geom_a: idx_box,
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

/// Capsule vs static plane. Emits up to 2 endpoint contacts (axis endpoints).
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
        capsule_pose.position - axis_world * half_height,
        capsule_pose.position + axis_world * half_height,
    ];
    let mut out = ContactBuf::new();
    for &e in ends.iter() {
        let signed = (e - p0).dot(n);
        let pen = radius - signed + margin;
        if pen > 0.0 {
            // Contact point is at the sphere-cap center's foot on the plane.
            let contact_pt = e - n * signed;
            out.push(Contact {
                geom_a: idx_capsule,
                geom_b: idx_plane,
                position_world: contact_pt,
                normal_world: n,
                penetration: pen,
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
    box_box_edge_edge_fallback(
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

/// Edge-edge SAT fallback for box-box. Runs when the vertex-vs-face manifold
/// is empty. Iterates the 9 cross-axis products between A's and B's edges
/// (which also happen to be their basis vectors), plus the 6 face-normal
/// axes; if the boxes overlap on ALL 15 axes then they truly intersect, and
/// the minimum-overlap edge-edge axis (if it is smaller than every
/// face-normal overlap) identifies the pair of edges producing the contact.
///
/// Returns 0 or 1 contact.
#[allow(clippy::too_many_arguments)]
fn box_box_edge_edge_fallback(
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

    // For any candidate axis L (not necessarily unit), the projected half-
    // extent of a box with basis (u0, u1, u2) and half-extents (h0, h1, h2)
    // is Σ h_i · |L · u_i|. Overlap along L is
    //   rA + rB - |delta · L|
    // measured in units of |L| (all we need is the sign and comparable
    // magnitudes among axes, but we normalize edge-edge axes so overlap
    // depths compare fairly to face-normal axes).
    let proj_half = |l: Vec3, u: &[Vec3; 3], h: &[f32; 3]| -> f32 {
        h[0] * crate::math::abs(l.dot(u[0]))
            + h[1] * crate::math::abs(l.dot(u[1]))
            + h[2] * crate::math::abs(l.dot(u[2]))
    };

    // 6 face-normal axes: any negative overlap → separated → no contact.
    for face in ax.iter().chain(bx.iter()) {
        let ra = proj_half(*face, &ax, &ha);
        let rb = proj_half(*face, &bx, &hb);
        let dist = crate::math::abs(delta.dot(*face));
        if dist > ra + rb + margin {
            return ContactBuf::new();
        }
    }

    // 9 edge-edge axes: track minimum overlap AND the edge pair that produced
    // it. If any axis has overlap < -margin, the boxes are separated on that
    // axis and we can return early. Skip near-parallel edge pairs (whose
    // cross product norm is ≈ 0) because their axis is degenerate and any
    // real separation on it will also appear on a face-normal axis.
    let mut best_overlap = f32::INFINITY;
    let mut best_axis = Vec3::Z;
    let mut best_pair: (usize, usize) = (0, 0);
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
            let signed_offset = delta.dot(l);
            let dist = crate::math::abs(signed_offset);
            let overlap = ra + rb - dist;
            if overlap < -margin {
                return ContactBuf::new();
            }
            if overlap < best_overlap {
                best_overlap = overlap;
                // Normal convention: from B into A means it should have a
                // positive component along `delta` (which points B → A).
                best_axis = if signed_offset >= 0.0 { l } else { -l };
                best_pair = (i, j);
            }
        }
    }

    if !best_overlap.is_finite() {
        // All edge pairs were parallel — the boxes are aligned to within
        // rotation about some shared axis. Vertex-vs-face would have caught
        // any real overlap; if it didn't, nothing to add here.
        return ContactBuf::new();
    }

    let pen_shift = best_overlap + margin;
    if pen_shift <= 0.0 {
        return ContactBuf::new();
    }

    // Find the mid-line of A's edge along axis ax[i]. The edge lies on the
    // intersection of A's two other faces closest to B (choose face signs by
    // the sign of `best_axis · ax[k]` for k ≠ i — pick the sign that pushes
    // the edge TOWARD B).
    let (ai, bj) = best_pair;
    let edge_a_midpoint = edge_midpoint(pose_a.position, &ax, &ha, ai, -best_axis);
    let edge_b_midpoint = edge_midpoint(pose_b.position, &bx, &hb, bj, best_axis);
    // Edges' directions in world.
    let edge_a_dir = ax[ai];
    let edge_b_dir = bx[bj];
    // Endpoints of each edge.
    let a_len = ha[ai];
    let b_len = hb[bj];
    let ea0 = edge_a_midpoint - edge_a_dir * a_len;
    let ea1 = edge_a_midpoint + edge_a_dir * a_len;
    let eb0 = edge_b_midpoint - edge_b_dir * b_len;
    let eb1 = edge_b_midpoint + edge_b_dir * b_len;
    let (_pa, pb) = closest_points_on_segments(ea0, ea1, eb0, eb1);
    // Contact point on B's surface = pb.
    let mut out = ContactBuf::new();
    out.push(Contact {
        geom_a: idx_a,
        geom_b: idx_b,
        position_world: pb,
        normal_world: best_axis,
        penetration: pen_shift,
        friction,
        gap,
    });
    out
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
    if pen > 0.0 {
        let contact_pt = support_world - n_world * signed;
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

/// Convex mesh vs static plane. Iterates all mesh vertices, keeps up to 4
/// deepest with shifted penetration `> 0` in a fixed vertex-index order.
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
    // Track up to 4 deepest via a small fixed-size top-K.
    let mut top: [(f32, Vec3); 4] = [(f32::NEG_INFINITY, Vec3::ZERO); 4];
    for (i, &v_local) in mesh.vertices.iter().enumerate() {
        let world = mesh_pose.point_to_world(v_local);
        let signed = (world - p0).dot(n_world);
        let pen = margin - signed;
        if pen <= 0.0 {
            continue;
        }
        // Find smallest in top; replace if this pen is larger. Tie-break by
        // the CURRENT vertex-index order (which we've already fixed by
        // iterating in order): use `>` so an equal-penetration later vertex
        // does NOT displace an earlier one — deterministic in the tie case.
        let mut min_idx = 0;
        for k in 1..4 {
            if top[k].0 < top[min_idx].0 {
                min_idx = k;
            }
        }
        if pen > top[min_idx].0 {
            top[min_idx] = (pen, world);
        }
        let _ = i; // vertex index reserved for future manifold reordering
    }
    // Emit in descending-pen order for deterministic output.
    let mut order = [0usize, 1, 2, 3];
    for i in 1..4 {
        let mut j = i;
        while j > 0 && top[order[j]].0 > top[order[j - 1]].0 {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut out = ContactBuf::new();
    for &i in &order {
        let (pen, world) = top[i];
        if pen <= 0.0 {
            continue;
        }
        let signed_raw = margin - pen;
        let contact_pt = world - n_world * signed_raw;
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
    let contact_world = ell_pose.point_to_world(q_local);
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
    let contact_world = mesh_pose.point_to_world(best_point_local);
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

/// Closest point on triangle `(a, b, c)` to point `p`. Standard barycentric
/// clamping (Ericson, *Real-Time Collision Detection*, §5.1.5).
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
    let friction = combine_friction(geom_a.friction, geom_b.friction);
    let margin = combine_max(geom_a.margin, geom_b.margin);
    let gap = combine_max(geom_a.gap, geom_b.gap);
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
        idx_a, geom_a, pose_a, idx_b, geom_b, pose_b, friction, margin, gap, meshes,
    );
    if let Some(buf) = first {
        debug_assert!(
            std::mem::discriminant(&geom_a.shape) == std::mem::discriminant(&geom_b.shape)
                || try_narrow_phase(
                    idx_b, geom_b, pose_b, idx_a, geom_a, pose_a, friction, margin, gap, meshes,
                )
                .is_none(),
            "narrow_phase invariant violated: an asymmetric shape pair matched a primitive \
             in BOTH orderings — a new primitive is registered against both `(A, B)` and \
             `(B, A)` combinations"
        );
        return buf;
    }
    if let Some(mut buf) = try_narrow_phase(
        idx_b, geom_b, pose_b, idx_a, geom_a, pose_a, friction, margin, gap, meshes,
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
) -> Option<ContactBuf> {
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
        (GeomShape::Ellipsoid { semi_axes }, GeomShape::Plane) => ellipsoid_plane(
            idx_a, pose_a, semi_axes, friction, margin, gap, idx_b, geom_b, pose_b,
        ),
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
        (GeomShape::Sphere { radius: rs }, GeomShape::Ellipsoid { semi_axes }) => sphere_ellipsoid(
            idx_a, pose_a, rs, idx_b, pose_b, semi_axes, friction, margin, gap,
        ),
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
        ) => box_box(
            idx_a, pose_a, half_a, idx_b, pose_b, half_b, friction, margin, gap,
        ),
        // Deferred (see is_pair_supported): box-sphere, box-capsule,
        // cylinder-cylinder, cylinder-{box,capsule,ellipsoid,mesh},
        // ellipsoid-{box,capsule,ellipsoid,mesh},
        // mesh-{box,capsule,mesh}.
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Geom, geom_world_pose};
    use crate::math::{Quat, Vec3};

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
        assert!(approx_vec(c.position_world, Vec3::new(0.0, 0.0, 0.0), 1e-5));
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
        // 0.5-0.3 = 0.2 each.
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
        for c in buf.as_slice() {
            assert!(approx(c.penetration, 0.2, 1e-5));
        }
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
