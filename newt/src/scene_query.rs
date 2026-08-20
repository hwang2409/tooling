//! Read-only scene queries built on the dynamic AABB tree.

use crate::broadphase::{Aabb, Ray, geom_aabb};
use crate::contact;
use crate::geom::{ConvexMesh, Geom, GeomPose, GeomShape};
use crate::math::{Quat, Vec3};

/// A finite convex shape used as a scene-query sweep volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShapeDesc {
    /// A sphere with the given radius.
    Sphere { radius: f32 },
    /// A box with local-frame half extents.
    Box { half_extents: Vec3 },
    /// A capsule aligned with local +Z.
    Capsule { radius: f32, half_height: f32 },
    /// A convex mesh asset from [`crate::world::World::meshes`].
    ConvexMesh { mesh_id: usize },
    /// Compatibility spelling for a convex mesh asset.
    Mesh { mesh_id: usize },
}

impl ShapeDesc {
    pub(crate) fn geom_shape(self) -> GeomShape {
        match self {
            Self::Sphere { radius } => GeomShape::Sphere { radius },
            Self::Box { half_extents } => GeomShape::Box { half_extents },
            Self::Capsule {
                radius,
                half_height,
            } => GeomShape::Capsule {
                radius,
                half_height,
            },
            Self::ConvexMesh { mesh_id } | Self::Mesh { mesh_id } => GeomShape::Mesh { mesh_id },
        }
    }

    pub(crate) fn aabb(self, pose: &GeomPose, meshes: &[ConvexMesh]) -> Aabb {
        let geom = Geom {
            shape: self.geom_shape(),
            body: None,
            link: None,
            local_offset: Vec3::ZERO,
            local_orientation: Quat::IDENTITY,
            friction: 0.0,
            solref: crate::geom::SolRef::DEFAULT,
            margin: 0.0,
            gap: 0.0,
            condim: 3,
            torsional_friction: 0.0,
            rolling_friction: 0.0,
            solimp: crate::solver::SolImp::DEFAULT,
            collision_group: u32::MAX,
            collision_mask: u32::MAX,
            user_data: 0,
        };
        geom_aabb(&geom, pose, meshes, &[])
    }
}

/// One ray-cast result. All fields own their values and remain valid after
/// the world changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Stable world geom index.
    pub geom_id: usize,
    /// Free-body owner, or `None` for static and tree-link geoms.
    pub body_id: Option<usize>,
    /// Distance along the ray direction.
    pub t: f32,
    /// Hit position in world coordinates.
    pub point_world: Vec3,
    /// Outward surface normal in world coordinates.
    pub normal_world: Vec3,
}

/// One shape-cast result. All fields own their values and remain valid after
/// the world changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeHit {
    /// Stable world geom index.
    pub geom_id: usize,
    /// Free-body owner, or `None` for static and tree-link geoms.
    pub body_id: Option<usize>,
    /// Fraction along the linear sweep, in `[0, 1]`.
    pub t: f32,
    /// Contact position in world coordinates.
    pub point_world: Vec3,
    /// Normal pointing from the target geom toward the swept shape.
    pub normal_world: Vec3,
}

pub(crate) fn ray_hit(
    geom: &Geom,
    pose: &GeomPose,
    ray: Ray,
    max_dist: f32,
    meshes: &[ConvexMesh],
) -> Option<(f32, Vec3, Vec3)> {
    if max_dist < 0.0 || !max_dist.is_finite() {
        return None;
    }
    match geom.shape {
        GeomShape::Sphere { radius } => ray_sphere(pose.position, radius, ray, max_dist),
        GeomShape::Box { half_extents } => ray_box(pose, half_extents, ray, max_dist),
        GeomShape::Capsule {
            radius,
            half_height,
        } => ray_capsule(pose, radius, half_height, ray, max_dist),
        GeomShape::Plane => ray_plane(pose, ray, max_dist),
        GeomShape::Mesh { mesh_id } => ray_mesh(pose, meshes, mesh_id, ray, max_dist),
        GeomShape::Cylinder { .. } | GeomShape::Ellipsoid { .. } | GeomShape::Hfield { .. } => None,
    }
}

pub(crate) fn ray_mesh(
    pose: &GeomPose,
    meshes: &[ConvexMesh],
    mesh_id: usize,
    ray: Ray,
    max_dist: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let mesh = meshes.get(mesh_id)?;
    let local_origin = pose.orientation.inverse_rotate(ray.origin - pose.position);
    let local_direction = pose.orientation.inverse_rotate(ray.direction);
    let mut best = None;
    for &face in &mesh.faces {
        let a = mesh.vertices[face[0] as usize];
        let b = mesh.vertices[face[1] as usize];
        let c = mesh.vertices[face[2] as usize];
        let Some((t, normal)) = ray_triangle(local_origin, local_direction, a, b, c, max_dist)
        else {
            continue;
        };
        if best.is_none_or(|(current, _, _)| t < current) {
            best = Some((t, ray.origin + ray.direction * t, pose.rotate(normal)));
        }
    }
    best.map(|(t, point, normal)| (t, point, normal.normalize()))
}

pub(crate) fn shape_cast(
    shape: ShapeDesc,
    from_pose: GeomPose,
    to_pose: GeomPose,
    geom: &Geom,
    target_pose: &GeomPose,
    meshes: &[ConvexMesh],
) -> Option<(f32, Vec3, Vec3)> {
    if matches!(geom.shape, GeomShape::Plane | GeomShape::Hfield { .. }) {
        return None;
    }
    let (t, contact) = contact::sweep_convex(
        &shape.geom_shape(),
        &from_pose,
        &to_pose,
        &geom.shape,
        target_pose,
        meshes,
    )?;
    Some((t, contact.position_world, contact.normal_world))
}

fn ray_sphere(center: Vec3, radius: f32, ray: Ray, max_dist: f32) -> Option<(f32, Vec3, Vec3)> {
    let offset = ray.origin - center;
    let a = ray.direction.dot(ray.direction);
    if a <= 0.0 {
        return None;
    }
    let half_b = offset.dot(ray.direction);
    let c = offset.dot(offset) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let near = (-half_b - root) / a;
    let far = (-half_b + root) / a;
    let t = if near >= 0.0 { near } else { far };
    if !(0.0..=max_dist).contains(&t) {
        return None;
    }
    let point = ray.origin + ray.direction * t;
    Some((t, point, (point - center).normalize()))
}

fn ray_box(
    pose: &GeomPose,
    half_extents: Vec3,
    ray: Ray,
    max_dist: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let origin = pose.orientation.inverse_rotate(ray.origin - pose.position);
    let direction = pose.orientation.inverse_rotate(ray.direction);
    let mut near = 0.0;
    let mut far = max_dist;
    let mut near_normal = Vec3::ZERO;
    let mut far_normal = Vec3::ZERO;
    let axes = [
        (origin.x, direction.x, half_extents.x, Vec3::X),
        (origin.y, direction.y, half_extents.y, Vec3::Y),
        (origin.z, direction.z, half_extents.z, Vec3::Z),
    ];
    for (origin_axis, direction_axis, extent, axis) in axes {
        if direction_axis == 0.0 {
            if origin_axis.abs() > extent {
                return None;
            }
            continue;
        }
        let mut axis_near = (-extent - origin_axis) / direction_axis;
        let mut axis_far = (extent - origin_axis) / direction_axis;
        let mut normal = -axis;
        if axis_near > axis_far {
            core::mem::swap(&mut axis_near, &mut axis_far);
            normal = axis;
        }
        if axis_near > near {
            near = axis_near;
            near_normal = normal;
        }
        if axis_far < far {
            far = axis_far;
            far_normal = normal;
        }
        if near > far {
            return None;
        }
    }
    let t = if near >= 0.0 { near } else { far };
    if !(0.0..=max_dist).contains(&t) {
        return None;
    }
    let point = ray.origin + ray.direction * t;
    let normal = if near >= 0.0 { near_normal } else { far_normal };
    Some((t, point, pose.rotate(normal)))
}

fn ray_capsule(
    pose: &GeomPose,
    radius: f32,
    half_height: f32,
    ray: Ray,
    max_dist: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let axis = pose.rotate(Vec3::Z);
    let a = pose.position - axis * half_height;
    let b = pose.position + axis * half_height;
    let mut best = ray_sphere(a, radius, ray, max_dist);
    if let Some(candidate) = ray_sphere(b, radius, ray, max_dist)
        && best.is_none_or(|current| candidate.0 < current.0)
    {
        best = Some(candidate);
    }
    best
}

fn ray_plane(pose: &GeomPose, ray: Ray, max_dist: f32) -> Option<(f32, Vec3, Vec3)> {
    let normal = pose.rotate(Vec3::Z);
    let denominator = ray.direction.dot(normal);
    if denominator == 0.0 {
        return None;
    }
    let t = (pose.position - ray.origin).dot(normal) / denominator;
    if !(0.0..=max_dist).contains(&t) {
        return None;
    }
    Some((t, ray.origin + ray.direction * t, normal))
}

fn ray_triangle(
    origin: Vec3,
    direction: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    max_dist: f32,
) -> Option<(f32, Vec3)> {
    let edge1 = b - a;
    let edge2 = c - a;
    let cross = direction.cross(edge2);
    let determinant = edge1.dot(cross);
    if determinant.abs() <= 1.0e-8 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let distance = origin - a;
    let u = distance.dot(cross) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = distance.cross(edge1);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(q) * inverse;
    if !(0.0..=max_dist).contains(&t) {
        return None;
    }
    Some((t, edge1.cross(edge2).normalize()))
}
