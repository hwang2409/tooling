//! Collision geometry: plane, sphere, box, capsule.
//!
//! # Attachment
//!
//! A geom belongs either to a body (`body: Some(index)`) or to the static
//! world (`body: None`). Only the plane geom is allowed to be static in tier 2
//! (an infinite half-space is naturally static; dynamic planes are not
//! meaningful). All other shapes MUST attach to a body — the panic in
//! [`Geom::sphere`] et al. is a construction-time check.
//!
//! # Local pose
//!
//! - `local_offset` — position of the geom origin relative to the parent body
//!   COM, expressed in body-frame coordinates. For static geoms this is the
//!   world-frame position and `local_orientation` is the world orientation.
//! - `local_orientation` — orientation of the geom frame relative to the parent
//!   body frame (body → geom). For static geoms, world → geom.
//!
//! # Shape frames
//!
//! - [`GeomShape::Sphere`] — origin at center.
//! - [`GeomShape::Box`] — origin at center, half-extents along local axes.
//! - [`GeomShape::Capsule`] — origin at center, axis along local Z. The
//!   cylindrical portion has half-length `half_height`; total tip-to-tip
//!   length is `2*(half_height + radius)`.
//! - [`GeomShape::Plane`] — infinite half-space. Its local +Z is the outward
//!   normal; the plane passes through the geom origin.
//!
//! # Friction
//!
//! [`Geom::friction`] is a per-geom Coulomb coefficient. Pair friction for a
//! contact between geoms `a` and `b` is `min(a.friction, b.friction)`. This is
//! the choice documented in `docs/contacts.md`; a product form would also be
//! defensible but MIN matches the Bullet/ODE defaults and is monotone in either
//! coefficient.
//!
//! # Contact stiffness parameters
//!
//! [`Geom::solref`] mirrors MuJoCo's `solref` parameter shape: a
//! `(timeconst, dampratio)` pair. For a contact, the effective normal spring
//! constant is `k = m_eff / timeconst²` and damping is `c = 2 * dampratio *
//! m_eff / timeconst`, where `m_eff` is the reduced mass of the pair
//! (`m_eff = m_a` for a body-vs-static contact, `m_a m_b / (m_a + m_b)` for
//! two dynamic bodies). See [`solref_to_kc`]. When two geoms disagree, the
//! stiffer setting (smaller `timeconst`) wins — matches MuJoCo's `solmix` in
//! the equal-weight case.

use crate::math::{Mat3, Quat, Vec3};

/// Contact stiffness/damping parameterization, mirroring MuJoCo's `solref`.
///
/// - `timeconst` — time constant of the contact spring, in seconds. Smaller is
///   stiffer (less penetration under load).
/// - `dampratio` — damping ratio of the underlying spring-damper.
///   `1.0` = critical, `< 1.0` = underdamped (bounce), `> 1.0` = overdamped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolRef {
    pub timeconst: f32,
    pub dampratio: f32,
}

impl SolRef {
    /// MuJoCo-style default: timeconst 0.02 s, critical damping. Chosen so a
    /// unit mass under 1g settles in a few timesteps of dt=5 ms without
    /// bouncing.
    pub const DEFAULT: Self = Self {
        timeconst: 0.02,
        dampratio: 1.0,
    };

    pub const fn new(timeconst: f32, dampratio: f32) -> Self {
        Self {
            timeconst,
            dampratio,
        }
    }
}

/// Convert `(SolRef, m_eff)` into `(k, c)` spring/damper constants.
///
/// The single-DOF spring-damper `m_eff x'' = -k x - c x'` with the returned
/// coefficients has angular frequency `ω = 1/timeconst` and damping ratio
/// `dampratio`. Derivation: `k = m_eff ω² = m_eff / timeconst²`;
/// `c = 2 dampratio m_eff ω = 2 dampratio m_eff / timeconst`.
pub fn solref_to_kc(solref: SolRef, m_eff: f32) -> (f32, f32) {
    let tc = solref.timeconst;
    let k = m_eff / (tc * tc);
    let c = 2.0 * solref.dampratio * m_eff / tc;
    (k, c)
}

/// Combine two geoms' SolRefs for a shared contact. Rule: per-parameter
/// minimum — the stiffer time constant wins AND the less-damped ratio wins.
///
/// Rationale: users typically over-set the property they care about on one
/// geom and leave the other on defaults. A "smaller wins" rule means the
/// non-default setting shows through — a designer who dials in a bouncy
/// (`dampratio = 0.1`) ball against a default plane gets a bouncy contact,
/// not a critically-damped one. Both parameters mix independently so this
/// stays deterministic and monotone.
pub fn combine_solref(a: SolRef, b: SolRef) -> SolRef {
    let tc = if a.timeconst <= b.timeconst {
        a.timeconst
    } else {
        b.timeconst
    };
    let zeta = if a.dampratio <= b.dampratio {
        a.dampratio
    } else {
        b.dampratio
    };
    SolRef::new(tc, zeta)
}

/// Convex collision shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeomShape {
    /// Infinite half-space. The plane's outward normal is local +Z; a point
    /// with a positive local-Z coordinate is above the plane (no contact).
    Plane,
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    /// Capsule aligned with local Z. `half_height` is the half-length of the
    /// CYLINDRICAL portion; total tip-to-tip length is `2*(half_height + radius)`.
    Capsule {
        radius: f32,
        half_height: f32,
    },
}

/// A geom attached to a body, a tree link, or the static world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geom {
    pub shape: GeomShape,
    /// Owning free-body index, or `None` when this geom is static or
    /// attached to a tree link. See [`Geom::attachment`].
    pub body: Option<usize>,
    /// Owning tree + link index (tier 3). `Some((t, l))` means the geom
    /// attaches to `world.trees[t].links[l]`; takes precedence over `body`.
    /// `None` when the geom is attached to a free body or is static.
    pub link: Option<(usize, usize)>,
    /// Local offset from the parent's body-frame origin to the geom origin
    /// (in the parent's body frame). For a static geom this is the world-
    /// frame position of the geom.
    pub local_offset: Vec3,
    /// Local orientation from parent body frame to geom frame. For a static
    /// geom, world → geom.
    pub local_orientation: Quat,
    /// Coulomb friction coefficient.
    pub friction: f32,
    /// Contact stiffness parameters. See [`SolRef`] and [`solref_to_kc`].
    pub solref: SolRef,
}

/// Where a geom is attached. Convenience view over the `body`/`link` fields
/// so callers do not have to open-code the priority rule (link over body
/// over static).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeomAttach {
    /// Static geom in the world (planes, etc.).
    Static,
    /// Attached to `world.bodies[i]`.
    Body(usize),
    /// Attached to `world.trees[t].links[l]`.
    Link(usize, usize),
}

impl Geom {
    /// Return the attachment kind. `link` takes precedence over `body`.
    pub fn attachment(&self) -> GeomAttach {
        if let Some((t, l)) = self.link {
            GeomAttach::Link(t, l)
        } else if let Some(b) = self.body {
            GeomAttach::Body(b)
        } else {
            GeomAttach::Static
        }
    }
}

impl Geom {
    /// Static infinite plane. `normal` is the outward normal in world coords;
    /// the plane passes through `point`. The stored local frame is chosen so
    /// local +Z = `normal`.
    pub fn static_plane(point: Vec3, normal: Vec3, friction: f32) -> Self {
        let n = normal.normalize();
        // Choose any orthonormal frame whose +Z is `n`; determinism-friendly.
        let orientation = quat_align_z_to(n);
        Self {
            shape: GeomShape::Plane,
            body: None,
            link: None,
            local_offset: point,
            local_orientation: orientation,
            friction,
            solref: SolRef::DEFAULT,
        }
    }

    /// Sphere attached to a body. `local_offset` is body-frame COM → sphere
    /// center.
    pub fn sphere(body: usize, radius: f32, local_offset: Vec3, friction: f32) -> Self {
        Self {
            shape: GeomShape::Sphere { radius },
            body: Some(body),
            link: None,
            local_offset,
            local_orientation: Quat::IDENTITY,
            friction,
            solref: SolRef::DEFAULT,
        }
    }

    /// Sphere attached to a tree link (tier 3). `local_offset` is
    /// link-body-frame COM → sphere center.
    pub fn sphere_on_link(
        tree: usize,
        link: usize,
        radius: f32,
        local_offset: Vec3,
        friction: f32,
    ) -> Self {
        Self {
            shape: GeomShape::Sphere { radius },
            body: None,
            link: Some((tree, link)),
            local_offset,
            local_orientation: Quat::IDENTITY,
            friction,
            solref: SolRef::DEFAULT,
        }
    }

    /// Box attached to a body. `local_offset` and `local_orientation` place the
    /// box center and orientation relative to the body COM/frame.
    pub fn r#box(
        body: usize,
        half_extents: Vec3,
        local_offset: Vec3,
        local_orientation: Quat,
        friction: f32,
    ) -> Self {
        Self {
            shape: GeomShape::Box { half_extents },
            body: Some(body),
            link: None,
            local_offset,
            local_orientation,
            friction,
            solref: SolRef::DEFAULT,
        }
    }

    /// Box attached to a tree link (tier 3).
    pub fn box_on_link(
        tree: usize,
        link: usize,
        half_extents: Vec3,
        local_offset: Vec3,
        local_orientation: Quat,
        friction: f32,
    ) -> Self {
        Self {
            shape: GeomShape::Box { half_extents },
            body: None,
            link: Some((tree, link)),
            local_offset,
            local_orientation,
            friction,
            solref: SolRef::DEFAULT,
        }
    }

    /// Capsule attached to a body. Axis along local Z.
    pub fn capsule(
        body: usize,
        radius: f32,
        half_height: f32,
        local_offset: Vec3,
        local_orientation: Quat,
        friction: f32,
    ) -> Self {
        Self {
            shape: GeomShape::Capsule {
                radius,
                half_height,
            },
            body: Some(body),
            link: None,
            local_offset,
            local_orientation,
            friction,
            solref: SolRef::DEFAULT,
        }
    }

    /// Capsule attached to a tree link (tier 3). Axis along link body-frame
    /// local Z after the `local_orientation`.
    pub fn capsule_on_link(
        tree: usize,
        link: usize,
        radius: f32,
        half_height: f32,
        local_offset: Vec3,
        local_orientation: Quat,
        friction: f32,
    ) -> Self {
        Self {
            shape: GeomShape::Capsule {
                radius,
                half_height,
            },
            body: None,
            link: Some((tree, link)),
            local_offset,
            local_orientation,
            friction,
            solref: SolRef::DEFAULT,
        }
    }

    /// Override the contact stiffness parameters (builder-style).
    pub fn with_solref(mut self, solref: SolRef) -> Self {
        self.solref = solref;
        self
    }
}

/// World-space geom pose derived from a body pose (or identity for static).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeomPose {
    /// World-frame position of the geom origin.
    pub position: Vec3,
    /// Body → geom in world coordinates: rotates a vector expressed in the
    /// geom frame into world coordinates.
    pub orientation: Quat,
}

impl GeomPose {
    /// Rotate a geom-local vector into world coordinates.
    pub fn rotate(&self, v: Vec3) -> Vec3 {
        self.orientation.rotate(v)
    }

    /// Transform a geom-local point into world coordinates.
    pub fn point_to_world(&self, p_local: Vec3) -> Vec3 {
        self.position + self.rotate(p_local)
    }
}

/// Build the world-space pose of a geom given its parent's pose (whether
/// that parent is a free body or a tree link — the caller resolves which).
/// `parent_position`/`parent_orientation` are unused when the geom is
/// static (see [`GeomAttach::Static`]).
pub fn geom_world_pose(geom: &Geom, parent_position: Vec3, parent_orientation: Quat) -> GeomPose {
    match geom.attachment() {
        GeomAttach::Static => GeomPose {
            position: geom.local_offset,
            orientation: geom.local_orientation,
        },
        GeomAttach::Body(_) | GeomAttach::Link(_, _) => GeomPose {
            position: parent_position + parent_orientation.rotate(geom.local_offset),
            orientation: parent_orientation * geom.local_orientation,
        },
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Build a unit quaternion whose rotated +Z equals the target unit vector.
///
/// Uses the axis = `+Z × target`, angle = `acos(target.z)` construction, but
/// avoids acos (no libm) by using the half-angle identity
/// `sin(θ/2) = sqrt((1 - cosθ)/2)`, `cos(θ/2) = sqrt((1 + cosθ)/2)` on
/// `cosθ = target.z`. Correct for target.z in [-1, 1].
fn quat_align_z_to(target: Vec3) -> Quat {
    let t = target.normalize();
    let cos_theta = t.z;
    if cos_theta > 1.0 - 1.0e-6 {
        return Quat::IDENTITY;
    }
    if cos_theta < -1.0 + 1.0e-6 {
        // Antipode: 180° flip about any horizontal axis; pick +X.
        return Quat::new(1.0, 0.0, 0.0, 0.0);
    }
    // axis = normalize(+Z × target) = normalize((-t.y, t.x, 0)); horizontal.
    let ax = Vec3::new(-t.y, t.x, 0.0).normalize();
    // half-angle from cos(θ):
    let half_cos = ((1.0 + cos_theta) * 0.5).sqrt(); // cos(θ/2)
    let half_sin = ((1.0 - cos_theta) * 0.5).sqrt(); // sin(θ/2)
    Quat::new(ax.x * half_sin, ax.y * half_sin, ax.z * half_sin, half_cos)
}

// ---------------------------------------------------------------------------
// inertia helpers for solid shapes (bodies may still override)
// ---------------------------------------------------------------------------

/// Uniform-density solid sphere principal moments: `I = (2/5) m r²` about any
/// axis through the center.
pub fn solid_sphere_inertia(mass: f32, radius: f32) -> Mat3 {
    let i = (2.0 / 5.0) * mass * radius * radius;
    Mat3::diag(i, i, i)
}

/// Uniform-density solid box principal moments about the COM in the box's
/// body-frame axes: `Ixx = (m/12)((2hy)² + (2hz)²)` and cyclic — same shape
/// [`crate::body::Body::solid_box`] uses.
pub fn solid_box_inertia(mass: f32, half_extents: Vec3) -> Mat3 {
    let hx2 = half_extents.x * half_extents.x;
    let hy2 = half_extents.y * half_extents.y;
    let hz2 = half_extents.z * half_extents.z;
    let ixx = (mass / 3.0) * (hy2 + hz2);
    let iyy = (mass / 3.0) * (hx2 + hz2);
    let izz = (mass / 3.0) * (hx2 + hy2);
    Mat3::diag(ixx, iyy, izz)
}

/// Uniform-density solid capsule (cylinder body + two hemispherical caps),
/// axis along local Z. Derivation in `docs/contacts.md`.
///
/// - Volume: `V = π r² (2h) + (4/3) π r³` where `h = half_height`.
/// - Mass split by volume: `m_c` cylinder, `m_s = m − m_c` combined caps.
/// - `I_zz = ½ m_c r² + (2/5) m_s r²` (both caps' centers lie on the axis).
/// - `I_xx = I_yy = (1/12) m_c (3r² + 4h²) + m_s (83/320) r² + m_s (h + 3r/8)²`.
///
/// The `83/320 r²` term is a solid hemisphere's inertia about a diameter
/// *through its COM*: I_diameter_at_base = (2/5) m r², minus the parallel-axis
/// shift to the COM (`(3r/8)²`) gives `(2/5 − 9/64) = 83/320`. The final
/// `m_s (h + 3r/8)²` shifts both hemispheres from their own COMs to the capsule
/// center. At `h = 0` this collapses to the (2/5) m r² of a sphere — the test
/// [`capsule_inertia_reduces_to_sphere_when_height_zero`] pins that limit.
pub fn solid_capsule_inertia(mass: f32, radius: f32, half_height: f32) -> Mat3 {
    let r = radius;
    let h = half_height;
    // Volume-weighted mass split, avoiding shared π factors that cancel.
    let vol_cyl = r * r * (2.0 * h); // /π
    let vol_caps = (4.0 / 3.0) * r * r * r; // /π
    let total = vol_cyl + vol_caps;
    let m_c = mass * (vol_cyl / total);
    let m_s = mass - m_c;

    let i_zz = 0.5 * m_c * r * r + (2.0 / 5.0) * m_s * r * r;
    let offset = h + 3.0 / 8.0 * r;
    let hemi_own_i = (83.0 / 320.0) * m_s * r * r;
    let i_xx =
        (1.0 / 12.0) * m_c * (3.0 * r * r + 4.0 * h * h) + hemi_own_i + m_s * offset * offset;
    Mat3::diag(i_xx, i_xx, i_zz)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::FRAC_PI_2;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn solref_conversion_matches_hand_derived_form() {
        // Unit mass, timeconst 0.1 s, critical damping.
        // ω = 10 rad/s → k = 100, c = 20.
        let (k, c) = solref_to_kc(SolRef::new(0.1, 1.0), 1.0);
        assert!(approx(k, 100.0, 1e-4));
        assert!(approx(c, 20.0, 1e-4));
    }

    #[test]
    fn combine_solref_picks_per_parameter_minimum() {
        let a = SolRef::new(0.05, 0.9);
        let b = SolRef::new(0.02, 0.3);
        // Per-parameter min: tc=0.02, ζ=0.3.
        assert_eq!(combine_solref(a, b), SolRef::new(0.02, 0.3));
        assert_eq!(combine_solref(b, a), SolRef::new(0.02, 0.3));

        // Cross case: `a` wins one axis, `b` wins the other.
        let c = SolRef::new(0.01, 0.8);
        let d = SolRef::new(0.05, 0.2);
        assert_eq!(combine_solref(c, d), SolRef::new(0.01, 0.2));
    }

    #[test]
    fn plane_local_frame_aligns_z_with_normal() {
        // Diagonal normal (1, 1, 0)/√2 — rotating (0,0,1) by the constructed
        // quaternion must land on that normal.
        let normal = Vec3::new(1.0, 1.0, 0.0).normalize();
        let plane = Geom::static_plane(Vec3::ZERO, normal, 0.5);
        let rotated_z = plane.local_orientation.rotate(Vec3::Z);
        assert!(approx(rotated_z.x, normal.x, 1e-5));
        assert!(approx(rotated_z.y, normal.y, 1e-5));
        assert!(approx(rotated_z.z, normal.z, 1e-5));
    }

    #[test]
    fn plane_local_frame_handles_pure_z_normal() {
        // Identity path.
        let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5);
        let rotated_z = plane.local_orientation.rotate(Vec3::Z);
        assert!(approx(rotated_z.z, 1.0, 1e-6));
    }

    #[test]
    fn plane_local_frame_handles_antipodal_normal() {
        // -Z. Constructed quaternion is a 180° flip about X.
        let plane = Geom::static_plane(Vec3::ZERO, -Vec3::Z, 0.5);
        let rotated_z = plane.local_orientation.rotate(Vec3::Z);
        assert!(approx(rotated_z.z, -1.0, 1e-5));
    }

    #[test]
    fn geom_world_pose_composes_body_offset() {
        // Body at (10, 0, 0), rotated 90° about Z. Geom local_offset (1, 0, 0)
        // should land at world (10, 1, 0).
        let body_pos = Vec3::new(10.0, 0.0, 0.0);
        let body_ori = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let g = Geom::sphere(0, 0.1, Vec3::new(1.0, 0.0, 0.0), 0.5);
        let pose = geom_world_pose(&g, body_pos, body_ori);
        assert!(approx(pose.position.x, 10.0, 1e-5));
        assert!(approx(pose.position.y, 1.0, 1e-5));
    }

    #[test]
    fn sphere_inertia_matches_analytic() {
        let i = solid_sphere_inertia(2.0, 3.0);
        // (2/5) * 2 * 9 = 7.2
        assert!(approx(i.get(0, 0), 7.2, 1e-5));
        assert!(approx(i.get(1, 1), 7.2, 1e-5));
        assert!(approx(i.get(2, 2), 7.2, 1e-5));
    }

    #[test]
    fn capsule_inertia_reduces_to_sphere_when_height_zero() {
        // With half_height = 0, capsule is a sphere; I should be (2/5) m r²
        // isotropic. This is the sphere limit and pins down the offset math.
        let i = solid_capsule_inertia(1.0, 0.5, 0.0);
        let sphere_i = (2.0 / 5.0) * 1.0 * 0.25;
        assert!(approx(i.get(0, 0), sphere_i, 1e-5));
        assert!(approx(i.get(2, 2), sphere_i, 1e-5));
    }
}
