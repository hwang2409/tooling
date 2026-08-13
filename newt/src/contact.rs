//! Narrow-phase collision detection.
//!
//! # Contact representation
//!
//! A [`Contact`] describes one contact point between two geoms. The normal
//! points **from geom B into geom A**: an infinitesimal displacement of A
//! along `+normal` separates the pair. `penetration` is positive when the
//! geoms interpenetrate; a `penetration ≤ 0` contact is never emitted.
//!
//! # Narrow-phase coverage (tier 2)
//!
//! [`sphere_plane`], [`box_plane`], [`capsule_plane`], [`sphere_sphere`],
//! [`sphere_capsule`], [`capsule_capsule`], [`box_box`] (vertex-vs-face).
//! Box-sphere and box-capsule are deferred to a later tier; the deferral
//! note is in the PR body.
//!
//! # Determinism
//!
//! Each function returns a fixed number of contacts in a fixed order for a
//! given input; no HashMap iteration; no sort-key that ties on floats. Sorted
//! results at the pair level live in [`crate::world`].

use crate::geom::{Geom, GeomPose, GeomShape};
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
    /// Interpenetration depth (positive). Zero indicates a just-touching
    /// contact, which is not emitted (strictly `> 0`).
    pub penetration: f32,
    /// Pair Coulomb friction coefficient (see [`crate::geom`]).
    pub friction: f32,
}

/// Small stack-allocated buffer for per-pair contact output. Four is enough
/// for the shapes in this tier — a box-vs-plane emits at most four corner
/// contacts; capsule-vs-anything emits at most two endpoint contacts; sphere
/// pairs emit one. We stash the length inline to avoid heap allocation in the
/// per-step hot path.
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

/// World-frame outward normal and origin of a plane geom.
fn plane_world(plane_geom: &Geom, plane_pose: &GeomPose) -> (Vec3, Vec3) {
    // Local +Z is the outward normal by construction (see [`Geom::static_plane`]).
    let n = plane_pose.rotate(Vec3::Z);
    let _ = plane_geom; // shape checked by the caller
    (n, plane_pose.position)
}

/// Sphere vs static plane. Returns 0 or 1 contact.
pub fn sphere_plane(
    idx_sphere: usize,
    sphere_pose: &GeomPose,
    radius: f32,
    friction: f32,
    idx_plane: usize,
    plane_geom: &Geom,
    plane_pose: &GeomPose,
) -> ContactBuf {
    let mut out = ContactBuf::new();
    let (n, p0) = plane_world(plane_geom, plane_pose);
    let signed = (sphere_pose.position - p0).dot(n);
    let pen = radius - signed;
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
        });
    }
    out
}

/// Box vs static plane. Emits up to 4 corner contacts (the 4 deepest of any
/// penetrating corners) in a fixed deterministic order.
pub fn box_plane(
    idx_box: usize,
    box_pose: &GeomPose,
    half_extents: Vec3,
    friction: f32,
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
        *slot = (-signed, world); // penetration = -(signed distance)
    }
    // Keep only positive penetrations; then pick up to 4 deepest with a stable
    // insertion sort by (-penetration, index).
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
        // Contact position is the corner itself; that's the surface point of
        // the "B" side (plane) closest to the corner is `corner + pen * n`
        // — but our convention says the contact point sits on B's surface.
        let contact_pt = corner + n * pen;
        out.push(Contact {
            geom_a: idx_box,
            geom_b: idx_plane,
            position_world: contact_pt,
            normal_world: n,
            penetration: pen,
            friction,
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
        let pen = radius - signed;
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
            });
        }
    }
    out
}

/// Sphere vs sphere. 0 or 1 contact. Normal points from B into A.
pub fn sphere_sphere(
    idx_a: usize,
    pose_a: &GeomPose,
    radius_a: f32,
    idx_b: usize,
    pose_b: &GeomPose,
    radius_b: f32,
    friction: f32,
) -> ContactBuf {
    sphere_vs_sphere_at(
        idx_a,
        pose_a.position,
        radius_a,
        idx_b,
        pose_b.position,
        radius_b,
        friction,
    )
}

/// Sphere-vs-virtual-sphere helper; centers pre-resolved. Deterministic
/// tie-break: if the two centers coincide, normal defaults to `+Z`.
fn sphere_vs_sphere_at(
    idx_a: usize,
    center_a: Vec3,
    radius_a: f32,
    idx_b: usize,
    center_b: Vec3,
    radius_b: f32,
    friction: f32,
) -> ContactBuf {
    let mut out = ContactBuf::new();
    let delta = center_a - center_b;
    let dist2 = delta.length_squared();
    let sum_r = radius_a + radius_b;
    if dist2 >= sum_r * sum_r {
        return out;
    }
    let dist = dist2.sqrt();
    let n = if dist > 0.0 { delta / dist } else { Vec3::Z };
    let pen = sum_r - dist;
    // Contact point on B's surface.
    let contact_pt = center_b + n * radius_b;
    out.push(Contact {
        geom_a: idx_a,
        geom_b: idx_b,
        position_world: contact_pt,
        normal_world: n,
        penetration: pen,
        friction,
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
    )
}

/// Box vs box: vertex-vs-face (Sutherland-style pruning is deferred to a
/// later tier).
///
/// For each pair (A's 8 vertices vs B's interior, and B's 8 vertices vs A's
/// interior), a penetrating vertex generates one contact whose normal is the
/// out-normal of the nearest face of the containing box. Contacts are pooled,
/// sorted by penetration depth, and the four deepest are kept.
///
/// **Coverage limits.** This misses pure edge-vs-edge intersections (two
/// obliquely oriented boxes clashing on edges with all vertices outside the
/// other). The stacking regime this tier targets is dominated by
/// vertex-vs-face — 4 corners of the upper box resting on the top face of
/// the lower — so the shortcut buys the coverage we need. A full SAT + face
/// clipping arrives with tier v1 alongside cylinder/mesh geoms.
#[allow(clippy::too_many_arguments)]
pub fn box_box(
    idx_a: usize,
    pose_a: &GeomPose,
    half_a: Vec3,
    idx_b: usize,
    pose_b: &GeomPose,
    half_b: Vec3,
    friction: f32,
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
    for &(sx, sy, sz) in &CORNER_SIGNS {
        let local_a = Vec3::new(sx * half_a.x, sy * half_a.y, sz * half_a.z);
        let world_v = pose_a.point_to_world(local_a);
        let local_b = pose_b.orientation.inverse_rotate(world_v - pose_b.position);
        if let Some((pen, normal_local_b)) = face_along_direction(local_b, half_b, dir_b) {
            let normal_world = pose_b.rotate(normal_local_b);
            candidates[count] = (pen, world_v, normal_world);
            count += 1;
        }
    }
    // B's vertices in A — face normal points OUT of A (that's from A into B).
    // Flip to get "from B into A".
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
        let (pen, pos, normal) = candidates[i];
        out.push(Contact {
            geom_a: idx_a,
            geom_b: idx_b,
            position_world: pos,
            normal_world: normal,
            penetration: pen,
            friction,
        });
    }
    out
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
    // Outside the box on any axis → no penetration. Note the strict `< 0.0`:
    // a vertex sitting ON a face (`d = 0`) still counts along orthogonal
    // axes, which is what makes corner-on-corner axis-aligned stacks emit
    // contacts. Only the chosen axis has to be strictly positive.
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
) -> ContactBuf {
    let ax = pose_a.rotate(Vec3::Z);
    let bx = pose_b.rotate(Vec3::Z);
    let a0 = pose_a.position - ax * half_height_a;
    let a1 = pose_a.position + ax * half_height_a;
    let b0 = pose_b.position - bx * half_height_b;
    let b1 = pose_b.position + bx * half_height_b;
    let (pa, pb) = closest_points_on_segments(a0, a1, b0, b1);
    sphere_vs_sphere_at(idx_a, pa, radius_a, idx_b, pb, radius_b, friction)
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
/// combinations (deferred to a later tier) return an empty buffer.
pub fn narrow_phase(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
) -> ContactBuf {
    let friction = combine_friction(geom_a.friction, geom_b.friction);
    // Try in the order given; if that combination isn't a known primitive,
    // swap and dispatch, then relabel the results (flipping normal and A/B).
    if let Some(buf) = try_narrow_phase(idx_a, geom_a, pose_a, idx_b, geom_b, pose_b, friction) {
        return buf;
    }
    if let Some(mut buf) = try_narrow_phase(idx_b, geom_b, pose_b, idx_a, geom_a, pose_a, friction)
    {
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
fn try_narrow_phase(
    idx_a: usize,
    geom_a: &Geom,
    pose_a: &GeomPose,
    idx_b: usize,
    geom_b: &Geom,
    pose_b: &GeomPose,
    friction: f32,
) -> Option<ContactBuf> {
    Some(match (geom_a.shape, geom_b.shape) {
        (GeomShape::Sphere { radius }, GeomShape::Plane) => {
            sphere_plane(idx_a, pose_a, radius, friction, idx_b, geom_b, pose_b)
        }
        (GeomShape::Box { half_extents }, GeomShape::Plane) => {
            box_plane(idx_a, pose_a, half_extents, friction, idx_b, geom_b, pose_b)
        }
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
            idx_b,
            geom_b,
            pose_b,
        ),
        (GeomShape::Sphere { radius: ra }, GeomShape::Sphere { radius: rb }) => {
            sphere_sphere(idx_a, pose_a, ra, idx_b, pose_b, rb, friction)
        }
        (
            GeomShape::Sphere { radius: rs },
            GeomShape::Capsule {
                radius: rc,
                half_height,
            },
        ) => sphere_capsule(idx_a, pose_a, rs, idx_b, pose_b, rc, half_height, friction),
        (
            GeomShape::Capsule {
                radius: ra,
                half_height: ha,
            },
            GeomShape::Capsule {
                radius: rb,
                half_height: hb,
            },
        ) => capsule_capsule(idx_a, pose_a, ra, ha, idx_b, pose_b, rb, hb, friction),
        (
            GeomShape::Box {
                half_extents: half_a,
            },
            GeomShape::Box {
                half_extents: half_b,
            },
        ) => box_box(idx_a, pose_a, half_a, idx_b, pose_b, half_b, friction),
        // Still deferred: box-sphere, box-capsule.
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
        let buf = sphere_plane(0, &sphere_pose, 1.0, 1.0, 1, &plane, &plane_pose);
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
        let buf = sphere_plane(0, &sphere_pose, 1.0, 1.0, 1, &plane, &plane_pose);
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
        let buf = box_plane(0, &box_pose, hx, 1.0, 1, &plane, &plane_pose);
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
        let buf = box_plane(0, &box_pose, hx, 1.0, 1, &plane, &plane_pose);
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
        let buf = sphere_sphere(0, &pose_a, 1.0, 1, &pose_b, 1.0, 1.0);
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
        let buf = box_box(0, &upper_pose, hu, 1, &lower_pose, hl, 0.5);
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
        let buf = box_box(0, &a, Vec3::splat(0.5), 1, &b, Vec3::splat(0.5), 0.5);
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
        let buf = capsule_plane(0, &capsule_pose, 0.5, 1.0, 1.0, 1, &plane, &plane_pose);
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
