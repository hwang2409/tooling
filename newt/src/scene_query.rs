//! Read-only scene queries built on the dynamic AABB tree.
//!
//! # Query consistency
//!
//! These queries are consistent with public method-based mutations of world
//! state, such as [`crate::world::World::set_body_pose`],
//! [`crate::world::World::apply_mujoco_qpos`],
//! [`crate::world::World::apply_mujoco_qvel`],
//! [`crate::world::World::reset_to_keyframe`],
//! [`crate::world::World::set_mocap_pose`], and
//! [`crate::world::World::add_geom`], plus the guarded [`crate::tree::Tree`]
//! pose and velocity setters. Direct writes to the public
//! [`crate::world::World::bodies`], [`crate::world::World::trees`],
//! [`crate::world::World::geoms`], [`crate::world::World::meshes`], or
//! [`crate::world::World::hfields`] fields bypass query-proxy refresh, so a
//! query after such a write may miss the mutated geometry. Prefer the
//! method-based writers. Tracked follow-up: NEWT-60 (make pose- and
//! geometry-bearing state private).

use crate::broadphase::{Aabb, Ray, geom_aabb};
use crate::contact;
use crate::geom::{ConvexMesh, Geom, GeomPose, GeomShape, HeightField};
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
            friction_anisotropy: None,
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

    pub(crate) fn sweep_aabb(
        self,
        from_pose: &GeomPose,
        to_pose: &GeomPose,
        meshes: &[ConvexMesh],
    ) -> Aabb {
        let from = self.aabb(from_pose, meshes);
        let to = self.aabb(to_pose, meshes);
        let radius = self.pivot_radius(meshes);
        from.union(to)
            .expanded(4.0 * quat_distance(from_pose.orientation, to_pose.orientation) * radius)
    }

    fn pivot_radius(self, meshes: &[ConvexMesh]) -> f32 {
        match self {
            Self::Sphere { radius } => radius.abs(),
            Self::Box { half_extents } => half_extents.length(),
            Self::Capsule {
                radius,
                half_height,
            } => radius.abs() + half_height.abs(),
            Self::ConvexMesh { mesh_id } | Self::Mesh { mesh_id } => meshes
                .get(mesh_id)
                .map(|mesh| {
                    mesh.vertices
                        .iter()
                        .map(|vertex| vertex.length())
                        .fold(0.0, f32::max)
                })
                .unwrap_or(0.0),
        }
    }
}

fn quat_distance(a: Quat, b: Quat) -> f32 {
    let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    let sign = if dot < 0.0 { -1.0 } else { 1.0 };
    let dx = a.x - sign * b.x;
    let dy = a.y - sign * b.y;
    let dz = a.z - sign * b.z;
    let dw = a.w - sign * b.w;
    (dx * dx + dy * dy + dz * dz + dw * dw).sqrt()
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
    hfields: &[HeightField],
) -> Option<(f32, Vec3, Vec3)> {
    if max_dist < 0.0 || !max_dist.is_finite() {
        return None;
    }
    let local_ray = Ray {
        origin: pose.orientation.inverse_rotate(ray.origin - pose.position),
        direction: pose.orientation.inverse_rotate(ray.direction),
    };
    let (t, _, local_normal) = match geom.shape {
        GeomShape::Hfield { hfield_id } => {
            ray_hfield_local(hfields.get(hfield_id)?, local_ray, max_dist)?
        }
        shape => ray_vs_geom(local_ray, shape, meshes, max_dist)?,
    };
    let point = ray.origin + ray.direction * t;
    Some((t, point, pose.rotate(local_normal).normalize()))
}

/// Ray intersection in a geom's local frame. This is shared by scene queries
/// and the rangefinder sensor so every supported surface uses one kernel.
pub(crate) fn ray_vs_geom(
    ray: Ray,
    shape: GeomShape,
    meshes: &[ConvexMesh],
    max_dist: f32,
) -> Option<(f32, Vec3, Vec3)> {
    if ray.direction.length_squared() <= 0.0 {
        return None;
    }
    match shape {
        GeomShape::Sphere { radius } => ray_sphere(Vec3::ZERO, radius, ray, max_dist),
        GeomShape::Box { half_extents } => ray_box_local(half_extents, ray, max_dist),
        GeomShape::Capsule {
            radius,
            half_height,
        } => ray_capsule_local(radius, half_height, ray, max_dist),
        GeomShape::Plane => ray_plane_local(ray, max_dist),
        GeomShape::Cylinder {
            radius,
            half_height,
        } => ray_cylinder_local(radius, half_height, ray, max_dist),
        GeomShape::Ellipsoid { semi_axes } => ray_ellipsoid_local(semi_axes, ray, max_dist),
        GeomShape::Mesh { mesh_id } => ray_mesh_local(meshes.get(mesh_id)?, ray, max_dist),
        GeomShape::Hfield { .. } => None,
    }
}

fn ray_mesh_local(mesh: &ConvexMesh, ray: Ray, max_dist: f32) -> Option<(f32, Vec3, Vec3)> {
    let mut best = None;
    for &face in &mesh.faces {
        let a = mesh.vertices[face[0] as usize];
        let b = mesh.vertices[face[1] as usize];
        let c = mesh.vertices[face[2] as usize];
        let Some((t, normal)) = ray_triangle(ray.origin, ray.direction, a, b, c, max_dist) else {
            continue;
        };
        if best.is_none_or(|(current, _, _)| t < current) {
            best = Some((t, ray.origin + ray.direction * t, normal));
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

fn ray_box_local(half_extents: Vec3, ray: Ray, max_dist: f32) -> Option<(f32, Vec3, Vec3)> {
    let mut near = f32::NEG_INFINITY;
    let mut far = f32::INFINITY;
    let mut near_normal = Vec3::ZERO;
    let mut far_normal = Vec3::ZERO;
    let axes = [
        (ray.origin.x, ray.direction.x, half_extents.x, Vec3::X),
        (ray.origin.y, ray.direction.y, half_extents.y, Vec3::Y),
        (ray.origin.z, ray.direction.z, half_extents.z, Vec3::Z),
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
        let far_normal_for_axis = -normal;
        if axis_near > near {
            near = axis_near;
            near_normal = normal;
        }
        if axis_far < far {
            far = axis_far;
            far_normal = far_normal_for_axis;
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
    Some((t, point, normal.normalize()))
}

fn ray_capsule_local(
    radius: f32,
    half_height: f32,
    ray: Ray,
    max_dist: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let mut best = ray_cylinder_side_local(radius, half_height, ray, max_dist);
    if let Some(candidate) = ray_sphere(Vec3::new(0.0, 0.0, -half_height), radius, ray, max_dist)
        && best.is_none_or(|current| candidate.0 < current.0)
    {
        best = Some(candidate);
    }
    if let Some(candidate) = ray_sphere(Vec3::new(0.0, 0.0, half_height), radius, ray, max_dist)
        && best.is_none_or(|current| candidate.0 < current.0)
    {
        best = Some(candidate);
    }
    best
}

fn ray_plane_local(ray: Ray, max_dist: f32) -> Option<(f32, Vec3, Vec3)> {
    let denominator = ray.direction.z;
    if denominator == 0.0 {
        return None;
    }
    let t = -ray.origin.z / denominator;
    if !(0.0..=max_dist).contains(&t) {
        return None;
    }
    Some((t, ray.origin + ray.direction * t, Vec3::Z))
}

fn ray_cylinder_local(
    radius: f32,
    half_height: f32,
    ray: Ray,
    max_dist: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let mut best = ray_cylinder_side_local(radius, half_height, ray, max_dist);
    if ray.direction.z != 0.0 {
        for (z, normal) in [(-half_height, -Vec3::Z), (half_height, Vec3::Z)] {
            let t = (z - ray.origin.z) / ray.direction.z;
            if (0.0..=max_dist).contains(&t) {
                let point = ray.origin + ray.direction * t;
                if point.x * point.x + point.y * point.y <= radius * radius
                    && best.is_none_or(|current| t < current.0)
                {
                    best = Some((t, point, normal));
                }
            }
        }
    }
    best
}

fn ray_cylinder_side_local(
    radius: f32,
    half_height: f32,
    ray: Ray,
    max_dist: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y;
    if a <= 1.0e-8 {
        return None;
    }
    let half_b = ray.origin.x * ray.direction.x + ray.origin.y * ray.direction.y;
    let c = ray.origin.x * ray.origin.x + ray.origin.y * ray.origin.y - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let mut best = None;
    for t in [(-half_b - root) / a, (-half_b + root) / a] {
        if (0.0..=max_dist).contains(&t) {
            let point = ray.origin + ray.direction * t;
            if (-half_height..=half_height).contains(&point.z)
                && best.is_none_or(|current: (f32, Vec3, Vec3)| t < current.0)
            {
                best = Some((t, point, Vec3::new(point.x, point.y, 0.0).normalize()));
            }
        }
    }
    best
}

fn ray_ellipsoid_local(semi_axes: Vec3, ray: Ray, max_dist: f32) -> Option<(f32, Vec3, Vec3)> {
    let scaled_origin = Vec3::new(
        ray.origin.x / semi_axes.x,
        ray.origin.y / semi_axes.y,
        ray.origin.z / semi_axes.z,
    );
    let scaled_direction = Vec3::new(
        ray.direction.x / semi_axes.x,
        ray.direction.y / semi_axes.y,
        ray.direction.z / semi_axes.z,
    );
    let a = scaled_direction.dot(scaled_direction);
    if a <= 0.0 {
        return None;
    }
    let half_b = scaled_origin.dot(scaled_direction);
    let c = scaled_origin.dot(scaled_origin) - 1.0;
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
    let normal = Vec3::new(
        point.x / (semi_axes.x * semi_axes.x),
        point.y / (semi_axes.y * semi_axes.y),
        point.z / (semi_axes.z * semi_axes.z),
    )
    .normalize();
    Some((t, point, normal))
}

fn ray_hfield_local(hfield: &HeightField, ray: Ray, max_dist: f32) -> Option<(f32, Vec3, Vec3)> {
    let sx = hfield.size[0];
    let sy = hfield.size[1];
    let x = |column: usize| -sx + 2.0 * sx * column as f32 / (hfield.ncol - 1) as f32;
    let y = |row: usize| -sy + 2.0 * sy * row as f32 / (hfield.nrow - 1) as f32;
    let point =
        |row: usize, column: usize| Vec3::new(x(column), y(row), hfield.height(row, column));
    let mut best = None;
    for row in 0..hfield.nrow - 1 {
        for column in 0..hfield.ncol - 1 {
            let p00 = point(row, column);
            let p01 = point(row, column + 1);
            let p10 = point(row + 1, column);
            let p11 = point(row + 1, column + 1);
            for triangle in [(p00, p01, p11), (p00, p11, p10)] {
                let Some((t, mut normal)) = ray_triangle(
                    ray.origin,
                    ray.direction,
                    triangle.0,
                    triangle.1,
                    triangle.2,
                    max_dist,
                ) else {
                    continue;
                };
                if normal.z < 0.0 {
                    normal = -normal;
                }
                let point = ray.origin + ray.direction * t;
                if best.is_none_or(|current: (f32, Vec3, Vec3)| t < current.0) {
                    best = Some((t, point, normal));
                }
            }
        }
    }
    best
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotating_mesh_sweep_bound_uses_the_geom_pivot() {
        let meshes = [ConvexMesh {
            vertices: vec![
                Vec3::new(4.9, -0.1, -0.1),
                Vec3::new(5.1, -0.1, -0.1),
                Vec3::new(5.0, 0.1, -0.1),
                Vec3::new(5.0, 0.0, 0.1),
            ],
            faces: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        }];
        let from_pose = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let to_pose = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::from_axis_angle(Vec3::Z, core::f32::consts::PI),
        };
        let bounds = ShapeDesc::ConvexMesh { mesh_id: 0 }.sweep_aabb(&from_pose, &to_pose, &meshes);
        assert!(bounds.max.y >= 5.0);
    }
}
