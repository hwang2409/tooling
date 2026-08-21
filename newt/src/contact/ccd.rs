use super::*;
use crate::math::Quat;

const MULTI_FEATURE_CAP: usize = 16;

pub(super) fn route_pair(a: &GeomShape, b: &GeomShape) -> bool {
    matches!((a, b), (GeomShape::Mesh { .. }, GeomShape::Mesh { .. }))
}

pub(super) fn shape<'a>(
    shape: &GeomShape,
    pose: &'a GeomPose,
    meshes: &'a [ConvexMesh],
) -> Option<CcdShape<'a>> {
    match *shape {
        GeomShape::Sphere { radius } => Some(CcdShape::Sphere { pose, radius }),
        GeomShape::Box { half_extents } => Some(CcdShape::Box { pose, half_extents }),
        GeomShape::Capsule {
            radius,
            half_height,
        } => Some(CcdShape::Capsule {
            pose,
            radius,
            half_height,
        }),
        GeomShape::Mesh { mesh_id } => Some(CcdShape::Mesh {
            pose,
            mesh: &meshes[mesh_id],
        }),
        GeomShape::Plane
        | GeomShape::Hfield { .. }
        | GeomShape::Cylinder { .. }
        | GeomShape::Ellipsoid { .. } => None,
    }
}

#[derive(Clone, Copy)]
pub(super) struct CcdVertex {
    pub(super) minkowski: Vec3,
    pub(super) shape_a: Vec3,
    pub(super) shape_b: Vec3,
    pub(super) tie_a: bool,
    pub(super) tie_b: bool,
}

#[derive(Clone, Copy)]
pub(super) enum CcdShape<'a> {
    Vertices(&'a [Vec3]),
    Sphere {
        pose: &'a GeomPose,
        radius: f32,
    },
    Box {
        pose: &'a GeomPose,
        half_extents: Vec3,
    },
    Capsule {
        pose: &'a GeomPose,
        radius: f32,
        half_height: f32,
    },
    Mesh {
        pose: &'a GeomPose,
        mesh: &'a ConvexMesh,
    },
}

impl CcdShape<'_> {
    pub(super) fn center(self) -> Vec3 {
        match self {
            Self::Vertices(vertices) => {
                vertices
                    .iter()
                    .copied()
                    .fold(Vec3::ZERO, |sum, point| sum + point)
                    / vertices.len() as f32
            }
            Self::Sphere { pose, .. } | Self::Box { pose, .. } | Self::Capsule { pose, .. } => {
                pose.position
            }
            Self::Mesh { pose, .. } => pose.position,
        }
    }

    fn extent(self) -> f32 {
        let vertices = match self {
            Self::Vertices(vertices) => return vertices_extent(vertices),
            Self::Sphere { radius, .. } => return radius * 2.0,
            Self::Box { half_extents, .. } => return (half_extents * 2.0).length(),
            Self::Capsule {
                radius,
                half_height,
                ..
            } => return 2.0 * (half_height + radius),
            Self::Mesh { mesh, .. } => &mesh.vertices,
        };
        vertices_extent(vertices)
    }

    fn pivot_radius(self) -> f32 {
        match self {
            Self::Vertices(vertices) => vertices
                .iter()
                .map(|vertex| vertex.length())
                .fold(0.0, f32::max),
            Self::Sphere { radius, .. } => radius.abs(),
            Self::Box { half_extents, .. } => half_extents.length(),
            Self::Capsule {
                radius,
                half_height,
                ..
            } => radius.abs() + half_height.abs(),
            Self::Mesh { mesh, .. } => mesh
                .vertices
                .iter()
                .map(|vertex| vertex.length())
                .fold(0.0, f32::max),
        }
    }

    pub(super) fn support(self, direction: Vec3) -> Vec3 {
        self.support_with_tie(direction).0
    }

    fn support_with_tie(self, direction: Vec3) -> (Vec3, bool) {
        let direction = if direction.length_squared() > 1.0e-20 {
            direction
        } else {
            Vec3::X
        };
        match self {
            Self::Vertices(vertices) => (support_vertices_legacy(vertices, direction), false),
            Self::Sphere { pose, radius } => {
                (pose.position + direction.normalize() * radius, false)
            }
            Self::Box { pose, half_extents } => {
                let local = pose.orientation.inverse_rotate(direction);
                let point = Vec3::new(
                    if local.x >= 0.0 {
                        half_extents.x
                    } else {
                        -half_extents.x
                    },
                    if local.y >= 0.0 {
                        half_extents.y
                    } else {
                        -half_extents.y
                    },
                    if local.z >= 0.0 {
                        half_extents.z
                    } else {
                        -half_extents.z
                    },
                );
                (pose.point_to_world(point), false)
            }
            Self::Capsule {
                pose,
                radius,
                half_height,
            } => {
                let axis = pose.rotate(Vec3::Z);
                let endpoint = if direction.dot(axis) >= 0.0 {
                    pose.position + axis * half_height
                } else {
                    pose.position - axis * half_height
                };
                (endpoint + direction.normalize() * radius, false)
            }
            Self::Mesh { pose, mesh } => {
                support_vertices_transformed(&mesh.vertices, pose, direction)
            }
        }
    }

    fn centered_support(self, direction: Vec3) -> Option<Vec3> {
        match self {
            Self::Vertices(_) => None,
            Self::Sphere { .. } | Self::Box { .. } | Self::Capsule { .. } => None,
            Self::Mesh { pose, mesh } => {
                let local_direction = pose.orientation.inverse_rotate(direction);
                support_feature_centroid(&mesh.vertices, local_direction)
                    .map(|point| pose.point_to_world(point))
            }
        }
    }
}

fn vertices_extent(vertices: &[Vec3]) -> f32 {
    let (mut min, mut max) = (vertices[0], vertices[0]);
    for &vertex in vertices.iter().skip(1) {
        min.x = min.x.min(vertex.x);
        min.y = min.y.min(vertex.y);
        min.z = min.z.min(vertex.z);
        max.x = max.x.max(vertex.x);
        max.y = max.y.max(vertex.y);
        max.z = max.z.max(vertex.z);
    }
    (max - min).length()
}

fn support_vertices(vertices: &[Vec3], direction: Vec3) -> (Vec3, bool) {
    const SUPPORT_TIE_FRACTION: f32 = 1.0e-7;
    let direction_length = direction.length();
    if direction_length <= 1.0e-20 {
        return (vertices[0], false);
    }
    let direction = direction / direction_length;
    let (mut min, mut max) = (vertices[0], vertices[0]);
    for &vertex in vertices.iter().skip(1) {
        min.x = min.x.min(vertex.x);
        min.y = min.y.min(vertex.y);
        min.z = min.z.min(vertex.z);
        max.x = max.x.max(vertex.x);
        max.y = max.y.max(vertex.y);
        max.z = max.z.max(vertex.z);
    }
    let tie_epsilon = (max - min).length() * SUPPORT_TIE_FRACTION;
    let best = vertices
        .iter()
        .map(|point| point.dot(direction))
        .fold(f32::NEG_INFINITY, f32::max);
    let mut selected = Vec3::ZERO;
    let mut tied_count = 0;
    for &candidate in vertices {
        if (candidate.dot(direction) - best).abs() <= tie_epsilon {
            if tied_count == 0 || lexicographically_precedes(candidate, selected) {
                selected = candidate;
            }
            tied_count += 1;
        }
    }
    (selected, tied_count >= 2)
}

fn support_feature_centroid(vertices: &[Vec3], direction: Vec3) -> Option<Vec3> {
    const SUPPORT_TIE_FRACTION: f32 = 1.0e-7;
    let direction_length = direction.length();
    if direction_length <= 1.0e-20 {
        return None;
    }
    let direction = direction / direction_length;
    let (mut min, mut max) = (vertices[0], vertices[0]);
    for &vertex in vertices.iter().skip(1) {
        min.x = min.x.min(vertex.x);
        min.y = min.y.min(vertex.y);
        min.z = min.z.min(vertex.z);
        max.x = max.x.max(vertex.x);
        max.y = max.y.max(vertex.y);
        max.z = max.z.max(vertex.z);
    }
    let tie_epsilon = (max - min).length() * SUPPORT_TIE_FRACTION;
    let best = vertices
        .iter()
        .map(|point| point.dot(direction))
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_z = 0.0_f64;
    let mut tied_count = 0;
    for &candidate in vertices {
        if (candidate.dot(direction) - best).abs() <= tie_epsilon {
            tied_count += 1;
        }
    }
    let mut previous = None;
    for _ in 0..tied_count {
        let next = vertices
            .iter()
            .copied()
            .filter(|&candidate| {
                (candidate.dot(direction) - best).abs() <= tie_epsilon
                    && previous.is_none_or(|prior| lexicographically_precedes(prior, candidate))
            })
            .min_by(|&a, &b| {
                if lexicographically_precedes(a, b) {
                    std::cmp::Ordering::Less
                } else if lexicographically_precedes(b, a) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });
        let Some(next) = next else {
            break;
        };
        let count = vertices
            .iter()
            .filter(|&&candidate| same_vec3(candidate, next))
            .count();
        sum_x += f64::from(next.x) * count as f64;
        sum_y += f64::from(next.y) * count as f64;
        sum_z += f64::from(next.z) * count as f64;
        previous = Some(next);
    }
    if tied_count >= 2 {
        let count = tied_count as f64;
        Some(Vec3::new(
            (sum_x / count) as f32,
            (sum_y / count) as f32,
            (sum_z / count) as f32,
        ))
    } else {
        None
    }
}

// Hfield prism contacts keep their existing source-order tie behavior.
fn support_vertices_legacy(vertices: &[Vec3], direction: Vec3) -> Vec3 {
    let mut point = vertices[0];
    let mut best = point.dot(direction);
    for &candidate in vertices.iter().skip(1) {
        let dot = candidate.dot(direction);
        if dot > best {
            point = candidate;
            best = dot;
        }
    }
    point
}

fn ccd_vertex_precedes(a: CcdVertex, b: CcdVertex) -> bool {
    for (left, right) in [
        (a.minkowski, b.minkowski),
        (a.shape_a, b.shape_a),
        (a.shape_b, b.shape_b),
    ] {
        if same_vec3(left, right) {
            continue;
        }
        return lexicographically_precedes(left, right);
    }
    false
}

fn sort_ccd_vertices(vertices: &mut [CcdVertex]) {
    for index in 1..vertices.len() {
        let mut position = index;
        while position > 0 && ccd_vertex_precedes(vertices[position], vertices[position - 1]) {
            vertices.swap(position, position - 1);
            position -= 1;
        }
    }
}

fn support_vertices_transformed(
    vertices: &[Vec3],
    pose: &GeomPose,
    direction: Vec3,
) -> (Vec3, bool) {
    let local_direction = pose.orientation.inverse_rotate(direction);
    let (point, tied) = support_vertices(vertices, local_direction);
    (pose.point_to_world(point), tied)
}

pub(super) fn ccd_support(a: CcdShape<'_>, b: CcdShape<'_>, direction: Vec3) -> CcdVertex {
    let direction = if direction.length_squared() > CCD_DEGENERATE_SQUARED {
        direction
    } else {
        Vec3::X
    };
    let (shape_a, tie_a) = a.support_with_tie(direction);
    let (shape_b, tie_b) = b.support_with_tie(-direction);
    CcdVertex {
        minkowski: shape_a - shape_b,
        shape_a,
        shape_b,
        tie_a,
        tie_b,
    }
}

fn ccd_support_normalized(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    direction: Vec3,
    pair_extent: f32,
) -> CcdVertex {
    let mut support = ccd_support(shape_a, shape_b, direction);
    support.minkowski = support.minkowski / pair_extent;
    support
}

#[allow(clippy::too_many_arguments)]
fn triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    b * a.dot(c) - a * b.dot(c)
}

#[derive(Clone, Copy)]
pub(super) struct CcdSimplex {
    pub(super) points: [CcdVertex; 4],
    pub(super) len: usize,
}

#[derive(Clone, Copy)]
pub(super) struct CcdSolverConfig {
    gjk_support_epsilon: f32,
    distance_tolerance: f32,
    epa_support_epsilon: f32,
    max_epa_iterations: usize,
}

pub(super) const CCD_MESH_CONFIG: CcdSolverConfig = CcdSolverConfig {
    gjk_support_epsilon: 0.0,
    distance_tolerance: 1.0e-10,
    epa_support_epsilon: 1.0e-7,
    max_epa_iterations: 128,
};

const CCD_DEGENERATE_EPSILON: f32 = 1.0e-12;
const CCD_DEGENERATE_SQUARED: f32 = CCD_DEGENERATE_EPSILON * CCD_DEGENERATE_EPSILON;
const CCD_DISTANCE_FLOOR: f32 = CCD_DEGENERATE_SQUARED;

const CCD_HFIELD_CONFIG: CcdSolverConfig = CcdSolverConfig {
    gjk_support_epsilon: 1.0e-7,
    distance_tolerance: 1.0e-10,
    epa_support_epsilon: 1.0e-5,
    max_epa_iterations: 64,
};

impl CcdSimplex {
    fn new(point: CcdVertex) -> Self {
        Self {
            points: [point; 4],
            len: 1,
        }
    }

    fn push(&mut self, point: CcdVertex) {
        self.points[self.len] = point;
        self.len += 1;
    }

    fn set(&mut self, points: &[CcdVertex]) {
        self.len = points.len();
        self.points[..self.len].copy_from_slice(points);
    }
}

/// Update a GJK simplex. The newest point is at the last occupied index.
/// Returns true when the simplex encloses the origin.
pub(super) fn ccd_simplex_step(simplex: &mut CcdSimplex, direction: &mut Vec3) -> bool {
    let a = simplex.points[simplex.len - 1];
    let ao = -a.minkowski;
    match simplex.len {
        2 => {
            let b = simplex.points[0];
            let ab = b.minkowski - a.minkowski;
            if ab.dot(ao) > 0.0 {
                *direction = triple_product(ab, ao, ab);
                if direction.length_squared() <= CCD_DEGENERATE_SQUARED {
                    *direction = Vec3::Z;
                }
            } else {
                simplex.set(&[a]);
                *direction = ao;
            }
        }
        3 => {
            let b = simplex.points[1];
            let c = simplex.points[0];
            let ab = b.minkowski - a.minkowski;
            let ac = c.minkowski - a.minkowski;
            let abc = ab.cross(ac);
            if abc.length_squared() <= CCD_DEGENERATE_SQUARED {
                simplex.set(&[a, b]);
                let edge_length_squared = ab.length_squared();
                let t = if edge_length_squared > CCD_DEGENERATE_SQUARED {
                    clamp01(-a.minkowski.dot(ab) / edge_length_squared)
                } else {
                    0.0
                };
                let closest = a.minkowski + ab * t;
                if closest.length_squared() > CCD_DEGENERATE_SQUARED {
                    *direction = -closest;
                    return false;
                }
                let axis = if ab.x.abs() <= ab.y.abs() && ab.x.abs() <= ab.z.abs() {
                    Vec3::X
                } else if ab.y.abs() <= ab.z.abs() {
                    Vec3::Y
                } else {
                    Vec3::Z
                };
                *direction = ab.cross(axis);
                if direction.length_squared() <= CCD_DEGENERATE_SQUARED {
                    *direction = ao;
                }
                return false;
            }
            if abc.cross(ac).dot(ao) > 0.0 {
                if ac.dot(ao) > 0.0 {
                    simplex.set(&[a, c]);
                    *direction = triple_product(ac, ao, ac);
                } else if ab.dot(ao) > 0.0 {
                    simplex.set(&[a, b]);
                    *direction = triple_product(ab, ao, ab);
                } else {
                    simplex.set(&[a]);
                    *direction = ao;
                }
            } else if ab.cross(abc).dot(ao) > 0.0 {
                if ab.dot(ao) > 0.0 {
                    simplex.set(&[a, b]);
                    *direction = triple_product(ab, ao, ab);
                } else {
                    simplex.set(&[a]);
                    *direction = ao;
                }
            } else if abc.dot(ao) > 0.0 {
                *direction = abc;
            } else {
                simplex.set(&[a, c, b]);
                *direction = -abc;
            }
        }
        4 => {
            let b = simplex.points[2];
            let c = simplex.points[1];
            let d = simplex.points[0];
            let ab = b.minkowski - a.minkowski;
            let ac = c.minkowski - a.minkowski;
            let ad = d.minkowski - a.minkowski;
            let abc = ab.cross(ac);
            let acd = ac.cross(ad);
            let adb = ad.cross(ab);
            if abc.dot(ao) > 0.0 {
                simplex.set(&[a, c, b]);
                *direction = abc;
            } else if acd.dot(ao) > 0.0 {
                simplex.set(&[a, d, c]);
                *direction = acd;
            } else if adb.dot(ao) > 0.0 {
                simplex.set(&[a, b, d]);
                *direction = adb;
            } else {
                return true;
            }
        }
        _ => unreachable!(),
    }
    if direction.length_squared() <= CCD_DEGENERATE_SQUARED {
        *direction = Vec3::X;
    }
    false
}

fn ccd_weighted_witness(points: &[CcdVertex], weights: &[f32]) -> (Vec3, Vec3) {
    let mut point_a = Vec3::ZERO;
    let mut point_b = Vec3::ZERO;
    for (point, &weight) in points.iter().zip(weights) {
        point_a += point.shape_a * weight;
        point_b += point.shape_b * weight;
    }
    (point_a, point_b)
}

fn ccd_centered_witness(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    point_a: Vec3,
    point_b: Vec3,
    direction: Vec3,
) -> (Vec3, Vec3) {
    let direction = if direction.length_squared() > CCD_DEGENERATE_SQUARED {
        direction
    } else {
        Vec3::X
    };
    (
        shape_a.centered_support(direction).unwrap_or(point_a),
        shape_b.centered_support(-direction).unwrap_or(point_b),
    )
}

fn ccd_polytope_witness(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    feature: &[CcdVertex; 3],
    weights: [f32; 3],
    normal: Vec3,
) -> (Vec3, Vec3) {
    let (mut point_a, mut point_b) = ccd_weighted_witness(feature, &weights);
    // EPA can derive a final feature from a direction not used for support.
    // High-valence meshes need the geometric tie check in that case.
    let center_untagged_a = matches!(
        shape_a,
        CcdShape::Mesh { mesh, .. } if mesh.vertices.len() > 4
    );
    let center_untagged_b = matches!(
        shape_b,
        CcdShape::Mesh { mesh, .. } if mesh.vertices.len() > 4
    );
    let (matches_a, centered_a) = if center_untagged_a || feature.iter().any(|point| point.tie_a) {
        ccd_feature_support(shape_a, feature, normal, true)
    } else {
        (true, None)
    };
    let (matches_b, centered_b) = if center_untagged_b || feature.iter().any(|point| point.tie_b) {
        ccd_feature_support(shape_b, feature, -normal, false)
    } else {
        (true, None)
    };
    if matches_a {
        if let Some(centered_a) = centered_a {
            point_a = centered_a;
        }
    }
    if matches_b {
        if let Some(centered_b) = centered_b {
            point_b = centered_b;
        }
    }
    (point_a, point_b)
}

fn ccd_feature_support(
    shape: CcdShape<'_>,
    points: &[CcdVertex; 3],
    direction: Vec3,
    shape_a: bool,
) -> (bool, Option<Vec3>) {
    let point = |vertex: &CcdVertex| {
        if shape_a {
            vertex.shape_a
        } else {
            vertex.shape_b
        }
    };
    let direction = if direction.length_squared() > CCD_DEGENERATE_SQUARED {
        direction.normalize()
    } else {
        return (false, None);
    };
    let support = shape.support(direction);
    let feature_best = points
        .iter()
        .map(|vertex| point(vertex).dot(direction))
        .fold(f32::NEG_INFINITY, f32::max);
    let support_best = support.dot(direction);
    let tolerance = shape.extent() * 1.0e-7;
    if (feature_best - support_best).abs() > tolerance {
        (false, None)
    } else {
        (true, shape.centered_support(direction))
    }
}

fn ccd_triangle_witness(a: CcdVertex, b: CcdVertex, c: CcdVertex) -> (Vec3, Vec3) {
    let nearest = closest_point_on_triangle(Vec3::ZERO, a.minkowski, b.minkowski, c.minkowski);
    let bary = barycentric_triangle_origin(a.minkowski, b.minkowski, c.minkowski, nearest);
    ccd_weighted_witness(&[a, b, c], &[bary.0, bary.1, bary.2])
}

/// Return the closest pair represented by a non-enclosing GJK simplex.
/// Keeping the support witnesses avoids the center-axis approximation in the
/// margin-only path, which is not a distance witness for arbitrary meshes.
fn ccd_closest_witness(simplex: &CcdSimplex) -> Option<(Vec3, Vec3)> {
    match simplex.len {
        1 => Some((simplex.points[0].shape_a, simplex.points[0].shape_b)),
        2 => {
            let a = simplex.points[0];
            let b = simplex.points[1];
            let edge = b.minkowski - a.minkowski;
            let denominator = edge.length_squared();
            let t = if denominator > CCD_DEGENERATE_SQUARED {
                clamp01(-a.minkowski.dot(edge) / denominator)
            } else {
                0.0
            };
            Some(ccd_weighted_witness(&[a, b], &[1.0 - t, t]))
        }
        3 => Some(ccd_triangle_witness(
            simplex.points[0],
            simplex.points[1],
            simplex.points[2],
        )),
        4 => {
            let faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
            let mut best: Option<(Vec3, Vec3, f32)> = None;
            for [a, b, c] in faces {
                let (point_a, point_b) =
                    ccd_triangle_witness(simplex.points[a], simplex.points[b], simplex.points[c]);
                let distance_squared = (point_a - point_b).length_squared();
                if best.is_none_or(|candidate| distance_squared < candidate.2) {
                    best = Some((point_a, point_b, distance_squared));
                }
            }
            best.map(|(point_a, point_b, _)| (point_a, point_b))
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct CcdDistanceCandidate {
    point: Vec3,
    weights: [f32; 4],
    feature: [CcdVertex; 3],
    feature_len: usize,
}

/// Return whether the origin is inside a non-degenerate tetrahedron.
fn ccd_origin_inside_tetrahedron(simplex: &CcdSimplex) -> bool {
    let a = simplex.points[0].minkowski;
    let b = simplex.points[1].minkowski;
    let c = simplex.points[2].minkowski;
    let d = simplex.points[3].minkowski;
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let denominator = ab.cross(ac).dot(ad);
    if denominator.abs() <= CCD_DEGENERATE_SQUARED {
        return false;
    }
    let ao = -a;
    let weight_b = ao.cross(ac).dot(ad) / denominator;
    let weight_c = ab.cross(ao).dot(ad) / denominator;
    let weight_d = ab.cross(ac).dot(ao) / denominator;
    let weight_a = 1.0 - weight_b - weight_c - weight_d;
    [weight_a, weight_b, weight_c, weight_d]
        .into_iter()
        .all(|weight| (-1.0e-6..=1.0 + 1.0e-6).contains(&weight))
}

/// Find the closest point on one vertex, edge, or face region of a simplex.
fn ccd_distance_candidate(simplex: &CcdSimplex, mask: u8) -> Option<CcdDistanceCandidate> {
    let mut indices = [0usize; 3];
    let mut index_len = 0;
    for index in 0..simplex.len {
        if mask & (1 << index) != 0 {
            if index_len == indices.len() {
                return None;
            }
            indices[index_len] = index;
            index_len += 1;
        }
    }
    for index in 1..index_len {
        let mut position = index;
        while position > 0
            && ccd_vertex_precedes(
                simplex.points[indices[position]],
                simplex.points[indices[position - 1]],
            )
        {
            indices.swap(position, position - 1);
            position -= 1;
        }
    }
    let (point, subset_weights) = match index_len {
        1 => (simplex.points[indices[0]].minkowski, [1.0, 0.0, 0.0]),
        2 => {
            let a = simplex.points[indices[0]].minkowski;
            let b = simplex.points[indices[1]].minkowski;
            let edge = b - a;
            let denominator = edge.length_squared();
            let t = if denominator > CCD_DEGENERATE_SQUARED {
                clamp01(-a.dot(edge) / denominator)
            } else {
                0.0
            };
            (a + edge * t, [1.0 - t, t, 0.0])
        }
        3 => {
            let a = simplex.points[indices[0]].minkowski;
            let b = simplex.points[indices[1]].minkowski;
            let c = simplex.points[indices[2]].minkowski;
            if (b - a).cross(c - a).length_squared() <= CCD_DEGENERATE_SQUARED {
                return None;
            }
            let point = closest_point_on_triangle(Vec3::ZERO, a, b, c);
            let bary = barycentric_triangle_origin(a, b, c, point);
            (point, [bary.0, bary.1, bary.2])
        }
        _ => return None,
    };
    let mut weights = [0.0; 4];
    for (index, &weight) in indices[..index_len].iter().zip(subset_weights.iter()) {
        weights[*index] = weight;
    }
    let mut feature = [simplex.points[indices[0]]; 3];
    for (position, &index) in indices[..index_len].iter().enumerate() {
        feature[position] = simplex.points[index];
    }
    Some(CcdDistanceCandidate {
        point,
        weights,
        feature,
        feature_len: index_len,
    })
}

fn ccd_distance_candidate_precedes(
    candidate: CcdDistanceCandidate,
    current: CcdDistanceCandidate,
) -> bool {
    let candidate_distance = candidate.point.length_squared();
    let current_distance = current.point.length_squared();
    if candidate_distance != current_distance {
        return candidate_distance < current_distance;
    }
    if candidate.feature_len != current.feature_len {
        return candidate.feature_len < current.feature_len;
    }
    for index in 0..candidate.feature_len {
        if ccd_vertex_precedes(candidate.feature[index], current.feature[index]) {
            return true;
        }
        if ccd_vertex_precedes(current.feature[index], candidate.feature[index]) {
            return false;
        }
    }
    lexicographically_precedes(candidate.point, current.point)
}

/// Reduce a distance simplex by walking its vertex, edge, and face regions.
///
/// This is separate from `ccd_simplex_step`, which tests whether a simplex
/// encloses the origin for the intersection phase. Distance GJK needs the
/// closest Voronoi feature instead.
fn ccd_distance_simplex_step(simplex: &mut CcdSimplex) -> Option<Vec3> {
    if simplex.len == 4 && ccd_origin_inside_tetrahedron(simplex) {
        return None;
    }
    let mut best: Option<CcdDistanceCandidate> = None;
    for mask in 1..(1 << simplex.len) {
        let Some(candidate) = ccd_distance_candidate(simplex, mask as u8) else {
            continue;
        };
        if best.is_none_or(|current| ccd_distance_candidate_precedes(candidate, current)) {
            best = Some(candidate);
        }
    }
    let best = best?;
    let mut selected = [CcdVertex {
        minkowski: Vec3::ZERO,
        shape_a: Vec3::ZERO,
        shape_b: Vec3::ZERO,
        tie_a: false,
        tie_b: false,
    }; 4];
    let mut selected_len = 0;
    for index in 0..simplex.len {
        if best.weights[index] > 1.0e-6 {
            selected[selected_len] = simplex.points[index];
            selected_len += 1;
        }
    }
    if selected_len == 0 {
        let mut largest_index = 0;
        for index in 1..simplex.len {
            let weight_difference = best.weights[index] - best.weights[largest_index];
            if weight_difference > 1.0e-6
                || (weight_difference.abs() <= 1.0e-6
                    && ccd_vertex_precedes(simplex.points[index], simplex.points[largest_index]))
            {
                largest_index = index;
            }
        }
        selected[0] = simplex.points[largest_index];
        selected_len = 1;
    }
    sort_ccd_vertices(&mut selected[..selected_len]);
    simplex.set(&selected[..selected_len]);
    Some(best.point)
}

#[allow(clippy::too_many_arguments)]
fn ccd_distance_contact(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    simplex: &CcdSimplex,
    idx_a: usize,
    idx_b: usize,
    friction: f32,
    margin: f32,
    gap: f32,
) -> Option<Contact> {
    let (raw_point_a, raw_point_b) = ccd_closest_witness(simplex)?;
    let separation = raw_point_a - raw_point_b;
    let raw_distance = separation.length();
    let penetration = margin - raw_distance;
    if penetration <= 0.0 {
        return None;
    }
    let (point_a, point_b) = ccd_centered_witness(
        shape_a,
        shape_b,
        raw_point_a,
        raw_point_b,
        raw_point_b - raw_point_a,
    );
    let normal_world = if raw_distance > 0.0 {
        separation / raw_distance
    } else {
        Vec3::X
    };
    Some(Contact {
        geom_a: idx_a,
        geom_b: idx_b,
        position_world: (point_a + point_b) * 0.5,
        normal_world,
        penetration,
        friction,
        gap,
    })
}

fn ccd_distance_gjk(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    mut simplex: CcdSimplex,
    config: CcdSolverConfig,
    pair_extent: f32,
) -> CcdSimplex {
    let mut best_simplex = simplex;
    let mut best_distance_squared = f32::INFINITY;
    let mut previous_distance_squared = f32::INFINITY;
    for _ in 0..64 {
        let Some(closest) = ccd_distance_simplex_step(&mut simplex) else {
            return best_simplex;
        };
        let distance_squared = closest.length_squared();
        debug_assert!(
            distance_squared
                <= previous_distance_squared
                    + config.distance_tolerance * previous_distance_squared.max(CCD_DISTANCE_FLOOR),
            "distance simplex reduction increased the distance"
        );
        previous_distance_squared = distance_squared;
        if distance_squared < best_distance_squared {
            best_distance_squared = distance_squared;
            best_simplex = simplex;
        }
        if distance_squared <= CCD_DISTANCE_FLOOR {
            return best_simplex;
        }
        let direction = -closest;
        let point = ccd_support_normalized(shape_a, shape_b, direction, pair_extent);
        let support_dot = point.minkowski.dot(direction);
        let support_progress = support_dot - closest.dot(direction);
        if support_progress <= config.gjk_support_epsilon {
            return best_simplex;
        }
        if simplex.points[..simplex.len].iter().any(|candidate| {
            (candidate.minkowski - point.minkowski).length_squared() <= CCD_DISTANCE_FLOOR
        }) {
            return best_simplex;
        }
        simplex.push(point);
        let Some(next_closest) = ccd_distance_simplex_step(&mut simplex) else {
            return best_simplex;
        };
        let next_distance_squared = next_closest.length_squared();
        debug_assert!(
            next_distance_squared
                <= distance_squared
                    + config.distance_tolerance * distance_squared.max(CCD_DISTANCE_FLOOR),
            "distance GJK increased the distance"
        );
        if next_distance_squared < best_distance_squared {
            best_distance_squared = next_distance_squared;
            best_simplex = simplex;
        }
        let improvement = distance_squared - next_distance_squared;
        if improvement <= config.distance_tolerance * distance_squared.max(CCD_DISTANCE_FLOOR) {
            return best_simplex;
        }
    }
    best_simplex
}

/// Box versus one triangular prism using MuJoCo's native convex path shape:
/// GJK finds an enclosing simplex and EPA expands it to the closest face.
#[allow(clippy::too_many_arguments)]
pub(super) fn box_prism_gjk_epa_contact(
    box_pose: &GeomPose,
    half_extents: Vec3,
    prism: &HfieldPrism,
    idx_box: usize,
    idx_hfield: usize,
    friction: f32,
    margin: f32,
    gap: f32,
) -> Option<Contact> {
    let top = prism.top;
    let prism_vertices = [
        top.0,
        top.1,
        top.2,
        Vec3::new(top.0.x, top.0.y, prism.base_z),
        Vec3::new(top.1.x, top.1.y, prism.base_z),
        Vec3::new(top.2.x, top.2.y, prism.base_z),
    ];
    let mut box_vertices = [Vec3::ZERO; 8];
    for (index, vertex) in box_vertices.iter_mut().enumerate() {
        let local = Vec3::new(
            if index & 1 == 0 {
                -half_extents.x
            } else {
                half_extents.x
            },
            if index & 2 == 0 {
                -half_extents.y
            } else {
                half_extents.y
            },
            if index & 4 == 0 {
                -half_extents.z
            } else {
                half_extents.z
            },
        );
        *vertex = box_pose.point_to_world(local);
    }
    ccd_convex_contact(
        CcdShape::Vertices(&box_vertices),
        CcdShape::Vertices(&prism_vertices),
        CCD_HFIELD_CONFIG,
        idx_box,
        idx_hfield,
        friction,
        margin,
        gap,
    )
}

#[derive(Clone, Copy)]
struct CcdFace {
    indices: [usize; 3],
    normal: Vec3,
    distance: f32,
}

fn ccd_face(vertices: &[CcdVertex; 128], indices: [usize; 3]) -> Option<CcdFace> {
    let a = vertices[indices[0]].minkowski;
    let b = vertices[indices[1]].minkowski;
    let c = vertices[indices[2]].minkowski;
    let raw = (b - a).cross(c - a);
    if raw.length_squared() <= CCD_DEGENERATE_SQUARED {
        return None;
    }
    let mut normal = raw.normalize();
    let mut distance = normal.dot(a);
    let mut oriented = indices;
    if distance < 0.0 {
        oriented = [indices[0], indices[2], indices[1]];
        normal = -normal;
        distance = -distance;
    }
    Some(CcdFace {
        indices: oriented,
        normal,
        distance,
    })
}

fn ccd_face_precedes(
    vertices: &[CcdVertex; 128],
    candidate: CcdFace,
    current: CcdFace,
    tie_epsilon: f32,
) -> bool {
    if candidate.distance + tie_epsilon < current.distance {
        return true;
    }
    if current.distance + tie_epsilon < candidate.distance {
        return false;
    }
    let mut candidate_key = [
        vertices[candidate.indices[0]],
        vertices[candidate.indices[1]],
        vertices[candidate.indices[2]],
    ];
    let mut current_key = [
        vertices[current.indices[0]],
        vertices[current.indices[1]],
        vertices[current.indices[2]],
    ];
    sort_ccd_vertices(&mut candidate_key);
    sort_ccd_vertices(&mut current_key);
    for (candidate_vertex, current_vertex) in candidate_key.into_iter().zip(current_key) {
        if ccd_vertex_precedes(candidate_vertex, current_vertex) {
            return true;
        }
        if ccd_vertex_precedes(current_vertex, candidate_vertex) {
            return false;
        }
    }
    false
}

fn ccd_mesh_face_normal(pose: &GeomPose, mesh: &ConvexMesh, face: [u32; 3]) -> Option<Vec3> {
    let a = pose.point_to_world(mesh.vertices[face[0] as usize]);
    let b = pose.point_to_world(mesh.vertices[face[1] as usize]);
    let c = pose.point_to_world(mesh.vertices[face[2] as usize]);
    let normal = (b - a).cross(c - a);
    (normal.length_squared() > 0.0).then(|| normal.normalize())
}

/// Choose the shallowest deterministic support axis when EPA cannot converge.
/// The retained polytope, source mesh faces, center axis, and world axes cover
/// both generated hulls without emitting an unverified EPA face.
#[allow(clippy::too_many_arguments)]
fn ccd_epa_fallback_contact(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    faces: &[CcdFace],
    pair_extent: f32,
    idx_a: usize,
    idx_b: usize,
    friction: f32,
    margin: f32,
    gap: f32,
) -> Option<Contact> {
    let mut best: Option<(Vec3, f32)> = None;
    let mut consider = |axis: Vec3| {
        if axis.length_squared() <= CCD_DEGENERATE_SQUARED {
            return;
        }
        let axis = axis.normalize();
        for axis in [axis, -axis] {
            let support = ccd_support_normalized(shape_a, shape_b, axis, pair_extent);
            let distance = support.minkowski.dot(axis);
            if distance <= 0.0
                || best.is_some_and(|(current_axis, current_distance)| {
                    distance > current_distance
                        || (distance == current_distance
                            && !lexicographically_precedes(axis, current_axis))
                })
            {
                continue;
            }
            best = Some((axis, distance));
        }
    };
    for face in faces {
        consider(face.normal);
    }
    match shape_a {
        CcdShape::Mesh { pose, mesh } => {
            for &face in &mesh.faces {
                if let Some(normal) = ccd_mesh_face_normal(pose, mesh, face) {
                    consider(normal);
                }
            }
        }
        CcdShape::Vertices(_)
        | CcdShape::Sphere { .. }
        | CcdShape::Box { .. }
        | CcdShape::Capsule { .. } => {}
    }
    match shape_b {
        CcdShape::Mesh { pose, mesh } => {
            for &face in &mesh.faces {
                if let Some(normal) = ccd_mesh_face_normal(pose, mesh, face) {
                    consider(normal);
                }
            }
        }
        CcdShape::Vertices(_)
        | CcdShape::Sphere { .. }
        | CcdShape::Box { .. }
        | CcdShape::Capsule { .. } => {}
    }
    consider(shape_b.center() - shape_a.center());
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        consider(axis);
    }
    let (axis, distance) = best?;
    let support = ccd_support(shape_a, shape_b, axis);
    let penetration = distance * pair_extent + margin;
    (penetration > 0.0).then(|| Contact {
        geom_a: idx_a,
        geom_b: idx_b,
        position_world: support.shape_b + axis * (distance * pair_extent * 0.5),
        normal_world: -axis,
        penetration,
        friction,
        gap,
    })
}

const MULTI_CLIP_CAP: usize = MULTI_FEATURE_CAP * 2;
const MULTI_FACE_TOL: f32 = 0.996;
const MULTI_EDGE_TOL: f32 = 0.0888;

fn mesh_support_ids(
    shape: CcdShape<'_>,
    direction: Vec3,
    ids: &mut [usize; MULTI_FEATURE_CAP],
) -> Option<usize> {
    let CcdShape::Mesh { pose, mesh } = shape else {
        return Some(0);
    };
    let local_direction = pose.orientation.inverse_rotate(direction);
    let direction_length = local_direction.length();
    if direction_length <= CCD_DEGENERATE_EPSILON {
        return Some(0);
    }
    let direction = local_direction / direction_length;
    let best = mesh
        .vertices
        .iter()
        .map(|point| point.dot(direction))
        .fold(f32::NEG_INFINITY, f32::max);
    let extent = shape.extent();
    let tolerance = extent * 1.0e-7;
    let mut len = 0;
    for (index, point) in mesh.vertices.iter().enumerate() {
        if (point.dot(direction) - best).abs() <= tolerance {
            if len == ids.len() {
                return None;
            }
            ids[len] = index;
            len += 1;
        }
    }
    Some(len)
}

fn mesh_face_contains(face: [u32; 3], ids: &[usize; MULTI_FEATURE_CAP], id_len: usize) -> bool {
    face.iter()
        .all(|id| ids[..id_len].contains(&(*id as usize)))
}

fn mesh_feature_faces(
    mesh: &ConvexMesh,
    ids: &[usize; MULTI_FEATURE_CAP],
    id_len: usize,
    faces: &mut [usize; MULTI_FEATURE_CAP],
) -> Option<usize> {
    let mut len = 0;
    for (index, &face) in mesh.faces.iter().enumerate() {
        let matches = if id_len >= 3 {
            mesh_face_contains(face, ids, id_len)
        } else {
            face.iter()
                .filter(|id| ids[..id_len].contains(&(**id as usize)))
                .count()
                == id_len
        };
        if matches {
            if len == faces.len() {
                return None;
            }
            faces[len] = index;
            len += 1;
        }
    }
    Some(len)
}

fn mesh_face_world(
    pose: &GeomPose,
    mesh: &ConvexMesh,
    face_index: usize,
) -> Option<([Vec3; 3], Vec3)> {
    let face = mesh.faces[face_index];
    let points = [
        pose.point_to_world(mesh.vertices[face[0] as usize]),
        pose.point_to_world(mesh.vertices[face[1] as usize]),
        pose.point_to_world(mesh.vertices[face[2] as usize]),
    ];
    let local_a = mesh.vertices[face[0] as usize];
    let local_b = mesh.vertices[face[1] as usize];
    let local_c = mesh.vertices[face[2] as usize];
    let local_normal = (local_b - local_a).cross(local_c - local_a);
    if local_normal.length_squared() <= CCD_DEGENERATE_SQUARED {
        return None;
    }
    Some((points, pose.rotate(local_normal.normalize()).normalize()))
}

fn mesh_feature_normal(
    pose: &GeomPose,
    mesh: &ConvexMesh,
    ids: &[usize; MULTI_FEATURE_CAP],
    id_len: usize,
    expected: Vec3,
) -> Option<Vec3> {
    let mut faces = [0usize; MULTI_FEATURE_CAP];
    let face_len = mesh_feature_faces(mesh, ids, id_len, &mut faces)?;
    let mut best = None;
    for &face_index in &faces[..face_len] {
        let Some((_, normal)) = mesh_face_world(pose, mesh, face_index) else {
            continue;
        };
        if best.is_none_or(|current: Vec3| normal.dot(expected) > current.dot(expected)) {
            best = Some(normal);
        }
    }
    best
}

fn mesh_feature_polygon(
    pose: &GeomPose,
    mesh: &ConvexMesh,
    ids: &[usize; MULTI_FEATURE_CAP],
    id_len: usize,
    normal: Vec3,
) -> Option<([Vec3; MULTI_CLIP_CAP], usize)> {
    let mut polygon = [Vec3::ZERO; MULTI_CLIP_CAP];
    let mut len = 0;
    for &id in &ids[..id_len] {
        let point = pose.point_to_world(mesh.vertices[id]);
        if !polygon[..len]
            .iter()
            .any(|prior| (*prior - point).length_squared() <= CCD_DEGENERATE_SQUARED)
        {
            if len == polygon.len() {
                return None;
            }
            polygon[len] = point;
            len += 1;
        }
    }
    if len < 3 {
        return Some((polygon, len));
    }
    let center = polygon[..len]
        .iter()
        .copied()
        .fold(Vec3::ZERO, |sum, point| sum + point)
        / len as f32;
    let mut axis = (polygon[0] - center).normalize();
    if axis.length_squared() <= CCD_DEGENERATE_SQUARED {
        axis = Vec3::X;
    }
    let tangent = normal.cross(axis).normalize();
    for index in 1..len {
        let mut position = index;
        while position > 0
            && polygon_angle_precedes(
                polygon[position],
                polygon[position - 1],
                center,
                axis,
                tangent,
            )
        {
            polygon.swap(position, position - 1);
            position -= 1;
        }
    }
    Some((polygon, len))
}

fn polygon_start_precedes(a: Vec3, b: Vec3) -> bool {
    match a.x.total_cmp(&b.x) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => match b.y.total_cmp(&a.y) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => a.z.total_cmp(&b.z).is_lt(),
        },
    }
}

fn polygon_angle_precedes(a: Vec3, b: Vec3, center: Vec3, axis: Vec3, tangent: Vec3) -> bool {
    let a = a - center;
    let b = b - center;
    let ax = a.dot(axis);
    let ay = a.dot(tangent);
    let bx = b.dot(axis);
    let by = b.dot(tangent);
    let a_upper = ay > 0.0 || (ay == 0.0 && ax >= 0.0);
    let b_upper = by > 0.0 || (by == 0.0 && bx >= 0.0);
    if a_upper != b_upper {
        return a_upper;
    }
    let cross = ax * by - ay * bx;
    if cross != 0.0 {
        return cross > 0.0;
    }
    lexicographically_precedes(a, b)
}

fn mesh_feature_edges(
    pose: &GeomPose,
    mesh: &ConvexMesh,
    ids: &[usize; MULTI_FEATURE_CAP],
    id_len: usize,
    edges: &mut [[Vec3; 2]; MULTI_FEATURE_CAP],
) -> usize {
    if id_len >= 2 {
        edges[0] = [
            pose.point_to_world(mesh.vertices[ids[0]]),
            pose.point_to_world(mesh.vertices[ids[1]]),
        ];
        return 1;
    }
    if id_len == 0 {
        return 0;
    }
    let mut face_indices = [0usize; MULTI_FEATURE_CAP];
    let Some(face_len) = mesh_feature_faces(mesh, ids, id_len, &mut face_indices) else {
        return 0;
    };
    let mut len = 0;
    for &face_index in &face_indices[..face_len] {
        let face = mesh.faces[face_index];
        for &other in &face {
            let other = other as usize;
            if other == ids[0] {
                continue;
            }
            let edge = [ids[0], other];
            let points = [
                pose.point_to_world(mesh.vertices[edge[0]]),
                pose.point_to_world(mesh.vertices[edge[1]]),
            ];
            if !edges[..len].iter().any(|prior| {
                (prior[0] - points[0]).length_squared() <= CCD_DEGENERATE_SQUARED
                    && (prior[1] - points[1]).length_squared() <= CCD_DEGENERATE_SQUARED
                    || (prior[0] - points[1]).length_squared() <= CCD_DEGENERATE_SQUARED
                        && (prior[1] - points[0]).length_squared() <= CCD_DEGENERATE_SQUARED
            }) {
                if len == edges.len() {
                    return 0;
                }
                edges[len] = points;
                len += 1;
            }
        }
    }
    len
}

fn polygon_clip(
    subject: &[Vec3; MULTI_CLIP_CAP],
    subject_len: usize,
    reference: &[Vec3; MULTI_CLIP_CAP],
    reference_len: usize,
    normal: Vec3,
) -> Option<([Vec3; MULTI_CLIP_CAP], usize)> {
    // Overflow returns None so the caller keeps the single EPA contact.
    let mut polygon = *subject;
    let mut polygon_len = subject_len;
    let mut clipped = [Vec3::ZERO; MULTI_CLIP_CAP];
    for edge_index in 0..reference_len {
        let p = reference[edge_index];
        let q = reference[(edge_index + 1) % reference_len];
        let edge = q - p;
        let mut clipped_len = 0;
        for index in 0..polygon_len {
            let start = polygon[index];
            let end = polygon[(index + 1) % polygon_len];
            let start_distance = edge.cross(start - p).dot(normal);
            let end_distance = edge.cross(end - p).dot(normal);
            let start_inside = start_distance >= -1.0e-6;
            let end_inside = end_distance >= -1.0e-6;
            if start_inside != end_inside {
                let denominator = start_distance - end_distance;
                if denominator.abs() > CCD_DEGENERATE_EPSILON {
                    if clipped_len == clipped.len() {
                        return None;
                    }
                    let t = start_distance / denominator;
                    clipped[clipped_len] = start + (end - start) * t;
                    clipped_len += 1;
                }
            }
            if end_inside {
                if clipped_len == clipped.len() {
                    return None;
                }
                clipped[clipped_len] = end;
                clipped_len += 1;
            }
        }
        polygon = clipped;
        polygon_len = clipped_len;
        clipped = [Vec3::ZERO; MULTI_CLIP_CAP];
        if polygon_len == 0 {
            break;
        }
    }
    Some((polygon, polygon_len))
}

fn quad_area(points: &[Vec3; MULTI_CLIP_CAP], indices: [usize; 4]) -> f32 {
    let a = points[indices[0]];
    let b = points[indices[1]];
    let c = points[indices[2]];
    let d = points[indices[3]];
    ((b - a).cross(c - a).length() + (c - a).cross(d - a).length()) * 0.5
}

fn quad_indices(points: &[Vec3; MULTI_CLIP_CAP], len: usize) -> [usize; 4] {
    let mut best = [0, 1, 2, 3];
    let mut best_area = quad_area(points, best);
    for a in 0..len.saturating_sub(3) {
        for b in (a + 1)..len.saturating_sub(2) {
            for c in (b + 1)..len.saturating_sub(1) {
                for d in (c + 1)..len {
                    let candidate = [a, b, c, d];
                    let area = quad_area(points, candidate);
                    if area > best_area {
                        best_area = area;
                        best = candidate;
                    }
                }
            }
        }
    }
    best
}

fn ordered_indices(points: &[Vec3; MULTI_CLIP_CAP], indices: [usize; 4], len: usize) -> [usize; 4] {
    let mut first = 0;
    for index in 1..len {
        let precedes = if len == 2 {
            lexicographically_precedes(points[indices[index]], points[indices[first]])
        } else {
            polygon_start_precedes(points[indices[index]], points[indices[first]])
        };
        if precedes {
            first = index;
        }
    }
    let mut ordered = [0; 4];
    for index in 0..len {
        ordered[index] = indices[(first + len - index) % len];
    }
    ordered
}

#[allow(clippy::too_many_arguments)]
fn append_multi_contacts(
    out: &mut ContactBuf,
    subject: &[Vec3; MULTI_CLIP_CAP],
    subject_len: usize,
    reference: &[Vec3; MULTI_CLIP_CAP],
    reference_len: usize,
    reference_normal: Vec3,
    projection_normal: Vec3,
    subject_is_a: bool,
    base: Contact,
    margin: f32,
) -> Option<()> {
    let (polygon, polygon_len) = polygon_clip(
        subject,
        subject_len,
        reference,
        reference_len,
        reference_normal,
    )?;
    if polygon_len == 0 {
        return Some(());
    }
    let selected = if polygon_len > 4 {
        let indices = quad_indices(&polygon, polygon_len);
        [indices[0], indices[1], indices[2], indices[3]]
    } else {
        [0, 1, 2, 3]
    };
    let selected = ordered_indices(&polygon, selected, polygon_len.min(4));
    let selected_len = polygon_len.min(4);
    let reference_point = reference[0];
    for &index in &selected[..selected_len] {
        let subject_point = polygon[index];
        let signed = (subject_point - reference_point).dot(projection_normal);
        let (point_a, point_b) = if subject_is_a {
            (subject_point, subject_point - projection_normal * signed)
        } else {
            (subject_point - projection_normal * signed, subject_point)
        };
        let separation = point_a - point_b;
        let penetration = margin - separation.dot(base.normal_world);
        if penetration <= 0.0 {
            continue;
        }
        let contact = Contact {
            position_world: (point_a + point_b) * 0.5,
            penetration,
            normal_world: base.normal_world,
            ..base
        };
        if contact.penetration > 0.0
            && !out.as_slice().iter().any(|prior| {
                (prior.position_world - contact.position_world).length_squared()
                    <= CCD_DEGENERATE_SQUARED
            })
        {
            out.push(contact);
        }
    }
    Some(())
}

fn mesh_multicontact(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    base: Contact,
    margin: f32,
) -> Option<ContactBuf> {
    if margin > 0.0 {
        return None;
    }
    let CcdShape::Mesh {
        pose: pose_a,
        mesh: mesh_a,
    } = shape_a
    else {
        return None;
    };
    let CcdShape::Mesh {
        pose: pose_b,
        mesh: mesh_b,
    } = shape_b
    else {
        return None;
    };
    let mut ids_a = [0usize; MULTI_FEATURE_CAP];
    let mut ids_b = [0usize; MULTI_FEATURE_CAP];
    let len_a = mesh_support_ids(shape_a, -base.normal_world, &mut ids_a)?;
    let len_b = mesh_support_ids(shape_b, base.normal_world, &mut ids_b)?;
    if len_a == 0 || len_b == 0 {
        return None;
    }
    let normal_a = mesh_feature_normal(pose_a, mesh_a, &ids_a, len_a, -base.normal_world)?;
    let normal_b = mesh_feature_normal(pose_b, mesh_b, &ids_b, len_b, base.normal_world)?;
    let mut out = ContactBuf::new();
    if len_a >= 3 && len_b >= 3 && normal_a.dot(normal_b) < -MULTI_FACE_TOL {
        let (polygon_a, polygon_a_len) =
            mesh_feature_polygon(pose_a, mesh_a, &ids_a, len_a, normal_a)?;
        let (polygon_b, polygon_b_len) =
            mesh_feature_polygon(pose_b, mesh_b, &ids_b, len_b, normal_b)?;
        append_multi_contacts(
            &mut out,
            &polygon_b,
            polygon_b_len,
            &polygon_a,
            polygon_a_len,
            normal_a,
            normal_a,
            false,
            base,
            margin,
        )?;
    } else if len_a < 3 && len_b >= 3 {
        let mut edges = [[Vec3::ZERO; 2]; MULTI_FEATURE_CAP];
        let edge_len = mesh_feature_edges(pose_a, mesh_a, &ids_a, len_a, &mut edges);
        let (polygon_b, polygon_b_len) =
            mesh_feature_polygon(pose_b, mesh_b, &ids_b, len_b, normal_b)?;
        for edge in &edges[..edge_len] {
            let direction = (edge[1] - edge[0]).normalize();
            if direction.dot(normal_b).abs() < MULTI_EDGE_TOL {
                let mut subject = [Vec3::ZERO; MULTI_CLIP_CAP];
                subject[0] = edge[0];
                subject[1] = edge[1];
                append_multi_contacts(
                    &mut out,
                    &subject,
                    2,
                    &polygon_b,
                    polygon_b_len,
                    normal_b,
                    base.normal_world,
                    true,
                    base,
                    margin,
                )?;
                break;
            }
        }
    } else if len_b < 3 && len_a >= 3 {
        let mut edges = [[Vec3::ZERO; 2]; MULTI_FEATURE_CAP];
        let edge_len = mesh_feature_edges(pose_b, mesh_b, &ids_b, len_b, &mut edges);
        let (polygon_a, polygon_a_len) =
            mesh_feature_polygon(pose_a, mesh_a, &ids_a, len_a, normal_a)?;
        for edge in &edges[..edge_len] {
            let direction = (edge[1] - edge[0]).normalize();
            if direction.dot(normal_a).abs() < MULTI_EDGE_TOL {
                let mut subject = [Vec3::ZERO; MULTI_CLIP_CAP];
                subject[0] = edge[0];
                subject[1] = edge[1];
                append_multi_contacts(
                    &mut out,
                    &subject,
                    2,
                    &polygon_a,
                    polygon_a_len,
                    normal_a,
                    base.normal_world,
                    false,
                    base,
                    margin,
                )?;
                break;
            }
        }
    }
    (out.len > 1).then_some(out)
}

/// GJK plus EPA for two finite convex shapes. The single-contact wrapper is
/// retained for hfield prism callers.
#[allow(clippy::too_many_arguments)]
pub(super) fn ccd_convex_contact(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    config: CcdSolverConfig,
    idx_a: usize,
    idx_b: usize,
    friction: f32,
    margin: f32,
    gap: f32,
) -> Option<Contact> {
    let pair_extent = shape_a.extent().max(shape_b.extent());
    if pair_extent <= 0.0 {
        return None;
    }
    let mut direction = shape_b.center() - shape_a.center();
    if direction.length_squared() == 0.0 {
        direction = Vec3::X;
    }
    let mut simplex = CcdSimplex::new(ccd_support_normalized(
        shape_a,
        shape_b,
        direction,
        pair_extent,
    ));
    direction = -simplex.points[0].minkowski;
    let mut enclosed = false;
    let mut separated = false;
    for _ in 0..32 {
        let point = ccd_support_normalized(shape_a, shape_b, direction, pair_extent);
        if point.minkowski.dot(direction) <= config.gjk_support_epsilon {
            separated = true;
            break;
        }
        simplex.push(point);
        if ccd_simplex_step(&mut simplex, &mut direction) {
            enclosed = true;
            break;
        }
    }
    if !enclosed && separated {
        simplex = ccd_distance_gjk(shape_a, shape_b, simplex, config, pair_extent);
    }
    if !enclosed || simplex.len != 4 {
        return ccd_distance_contact(
            shape_a, shape_b, &simplex, idx_a, idx_b, friction, margin, gap,
        );
    }

    let mut vertices = [CcdVertex {
        minkowski: Vec3::ZERO,
        shape_a: Vec3::ZERO,
        shape_b: Vec3::ZERO,
        tie_a: false,
        tie_b: false,
    }; 128];
    vertices[..4].copy_from_slice(&simplex.points);
    let mut vertex_len = 4;
    let mut faces = [CcdFace {
        indices: [0; 3],
        normal: Vec3::Z,
        distance: f32::INFINITY,
    }; 512];
    let epa_support_epsilon = config.epa_support_epsilon;
    let mut face_len = 0;
    for indices in [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]] {
        if let Some(face) = ccd_face(&vertices, indices) {
            faces[face_len] = face;
            face_len += 1;
        }
    }
    if face_len == 0 {
        return ccd_distance_contact(
            shape_a, shape_b, &simplex, idx_a, idx_b, friction, margin, gap,
        );
    }
    let mut best_face = faces[0];
    let face_distance_epsilon = 1.0e-4;
    let mut converged = false;
    for _ in 0..config.max_epa_iterations {
        let mut best_index = 0;
        for index in 0..face_len {
            if ccd_face_precedes(
                &vertices,
                faces[index],
                faces[best_index],
                face_distance_epsilon,
            ) {
                best_index = index;
            }
        }
        best_face = faces[best_index];
        let support = ccd_support_normalized(shape_a, shape_b, best_face.normal, pair_extent);
        let support_distance = support.minkowski.dot(best_face.normal);
        if support_distance - best_face.distance <= epa_support_epsilon {
            converged = true;
            break;
        }
        if vertex_len == vertices.len() {
            break;
        }
        if vertices[..vertex_len].iter().any(|vertex| {
            (vertex.minkowski - support.minkowski).length_squared() <= CCD_DEGENERATE_SQUARED
        }) {
            break;
        }
        vertices[vertex_len] = support;
        let new_vertex = vertex_len;
        vertex_len += 1;
        let mut next_faces = [CcdFace {
            indices: [0; 3],
            normal: Vec3::Z,
            distance: f32::INFINITY,
        }; 512];
        let mut next_len = 0;
        let mut edges = [[0usize; 2]; 1024];
        let mut edge_len = 0;
        for face in faces[..face_len].iter().copied() {
            if face.normal.dot(support.minkowski) > face.distance + face_distance_epsilon {
                for edge in [
                    [face.indices[0], face.indices[1]],
                    [face.indices[1], face.indices[2]],
                    [face.indices[2], face.indices[0]],
                ] {
                    let reverse = [edge[1], edge[0]];
                    if let Some(index) = edges[..edge_len]
                        .iter()
                        .position(|candidate| *candidate == reverse)
                    {
                        edges[index] = edges[edge_len - 1];
                        edge_len -= 1;
                    } else {
                        edges[edge_len] = edge;
                        edge_len += 1;
                    }
                }
            } else {
                next_faces[next_len] = face;
                next_len += 1;
            }
        }
        for edge in edges[..edge_len].iter().copied() {
            if next_len == next_faces.len() {
                break;
            }
            if let Some(face) = ccd_face(&vertices, [edge[0], edge[1], new_vertex]) {
                next_faces[next_len] = face;
                next_len += 1;
            }
        }
        faces = next_faces;
        face_len = next_len;
        if face_len == 0 {
            return None;
        }
    }

    if !converged {
        return ccd_epa_fallback_contact(
            shape_a,
            shape_b,
            &faces[..face_len],
            pair_extent,
            idx_a,
            idx_b,
            friction,
            margin,
            gap,
        );
    }

    let nearest = best_face.normal * best_face.distance;
    let a = vertices[best_face.indices[0]].minkowski;
    let b = vertices[best_face.indices[1]].minkowski;
    let c = vertices[best_face.indices[2]].minkowski;
    let mut bary = barycentric_triangle_origin(a, b, c, nearest);
    let sum = bary.0 + bary.1 + bary.2;
    if sum <= 1.0e-8 {
        return None;
    }
    bary.0 /= sum;
    bary.1 /= sum;
    bary.2 /= sum;
    let feature = [
        vertices[best_face.indices[0]],
        vertices[best_face.indices[1]],
        vertices[best_face.indices[2]],
    ];
    let (point_a, point_b) = ccd_polytope_witness(
        shape_a,
        shape_b,
        &feature,
        [bary.0, bary.1, bary.2],
        best_face.normal,
    );
    let penetration = best_face.distance * pair_extent + margin;
    if penetration <= 0.0 {
        return None;
    }
    let normal_world = -best_face.normal;
    let position_world = (point_a + point_b) * 0.5;
    Some(Contact {
        geom_a: idx_a,
        geom_b: idx_b,
        position_world,
        normal_world,
        penetration,
        friction,
        gap,
    })
}

/// Sweep two finite convex shapes with conservative advancement.
///
/// GJK supplies a separating distance at each pose. The advancement bound is
/// deliberately conservative for both translation and the shortest quaternion
/// interpolation, so a narrow overlap window cannot be skipped.
#[allow(clippy::too_many_arguments)]
pub(super) fn ccd_sweep_convex(
    shape_a_desc: &GeomShape,
    from_pose: &GeomPose,
    to_pose: &GeomPose,
    shape_b: &GeomShape,
    pose_b: &GeomPose,
    meshes: &[ConvexMesh],
) -> Option<(f32, Contact)> {
    let shape_b = shape(shape_b, pose_b, meshes)?;
    let shape_a_from = shape(shape_a_desc, from_pose, meshes)?;
    let translation_speed = (to_pose.position - from_pose.position).length();
    let rotation_speed = 4.0
        * quat_distance(from_pose.orientation, to_pose.orientation)
        * shape_a_from.pivot_radius();
    let max_speed = translation_speed + rotation_speed;
    const DISTANCE_TOLERANCE: f32 = 1.0e-5;
    const MIN_ADVANCE: f32 = 1.0e-6;
    const MAX_STEPS: usize = 256;

    let mut t = 0.0;
    let mut closest_sample: Option<(f32, f32, Vec3, Vec3)> = None;
    for _ in 0..MAX_STEPS {
        let pose = interpolated_pose(from_pose, to_pose, t);
        let shape_a = shape(shape_a_desc, &pose, meshes)?;
        let pair_extent = shape_a.extent().max(shape_b.extent());
        if pair_extent <= 0.0 {
            return None;
        }
        let mut direction = shape_b.center() - shape_a.center();
        if direction.length_squared() == 0.0 {
            direction = Vec3::X;
        }
        let simplex = CcdSimplex::new(ccd_support_normalized(
            shape_a,
            shape_b,
            direction,
            pair_extent,
        ));
        let simplex = ccd_distance_gjk(shape_a, shape_b, simplex, CCD_MESH_CONFIG, pair_extent);
        let (point_a, point_b) = ccd_closest_witness(&simplex)?;
        let distance = (point_a - point_b).length();
        if closest_sample.is_none_or(|sample| distance < sample.1) {
            closest_sample = Some((t, distance, point_a, point_b));
        }
        if distance <= DISTANCE_TOLERANCE || max_speed <= MIN_ADVANCE {
            let mut contact = ccd_distance_contact(
                shape_a,
                shape_b,
                &simplex,
                0,
                0,
                0.0,
                DISTANCE_TOLERANCE,
                0.0,
            )?;
            contact.normal_world = sweep_normal(shape_a, shape_b, point_a, point_b, distance);
            return Some((t, contact));
        }
        let advance = distance / max_speed;
        if advance < MIN_ADVANCE {
            let mut contact = ccd_distance_contact(
                shape_a,
                shape_b,
                &simplex,
                0,
                0,
                0.0,
                DISTANCE_TOLERANCE,
                0.0,
            )
            .unwrap_or_else(|| ccd_near_miss_contact(point_a, point_b, distance));
            contact.normal_world = sweep_normal(shape_a, shape_b, point_a, point_b, distance);
            return Some((t, contact));
        }
        let next_t = t + advance;
        if next_t >= 1.0 {
            let end_pose = interpolated_pose(from_pose, to_pose, 1.0);
            let shape_a = shape(shape_a_desc, &end_pose, meshes)?;
            return ccd_convex_contact(shape_a, shape_b, CCD_MESH_CONFIG, 0, 0, 0.0, 0.0, 0.0)
                .map(|contact| (1.0, contact));
        }
        t = next_t;
    }
    let (t, distance, point_a, point_b) = closest_sample?;
    // A capped advancement is a conservative near-miss at the closest sample.
    // Returning it keeps a valid crossing observable instead of silently losing
    // the query when a narrow rotational window needs more than MAX_STEPS.
    Some((t, ccd_near_miss_contact(point_a, point_b, distance)))
}

fn ccd_near_miss_contact(point_a: Vec3, point_b: Vec3, distance: f32) -> Contact {
    Contact {
        geom_a: 0,
        geom_b: 0,
        position_world: (point_a + point_b) * 0.5,
        normal_world: if distance > 1.0e-5 {
            (point_a - point_b) / distance
        } else {
            Vec3::X
        },
        penetration: 0.0,
        friction: 0.0,
        gap: 0.0,
    }
}

fn sweep_normal(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    point_a: Vec3,
    point_b: Vec3,
    distance: f32,
) -> Vec3 {
    if distance > 1.0e-5 {
        return (point_a - point_b) / distance;
    }
    let center_delta = shape_a.center() - shape_b.center();
    if center_delta.length_squared() > 0.0 {
        center_delta.normalize()
    } else {
        Vec3::X
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

fn interpolated_pose(from: &GeomPose, to: &GeomPose, t: f32) -> GeomPose {
    let mut to_orientation = to.orientation;
    if from.orientation.x * to_orientation.x
        + from.orientation.y * to_orientation.y
        + from.orientation.z * to_orientation.z
        + from.orientation.w * to_orientation.w
        < 0.0
    {
        to_orientation = to_orientation * -1.0;
    }
    GeomPose {
        position: from.position + (to.position - from.position) * t,
        orientation: (from.orientation * (1.0 - t) + to_orientation * t).renormalize(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ccd_convex_contacts(
    shape_a: CcdShape<'_>,
    shape_b: CcdShape<'_>,
    config: CcdSolverConfig,
    idx_a: usize,
    idx_b: usize,
    friction: f32,
    margin: f32,
    gap: f32,
) -> ContactBuf {
    let mut out = ContactBuf::new();
    let Some(base) = ccd_convex_contact(
        shape_a, shape_b, config, idx_a, idx_b, friction, margin, gap,
    ) else {
        return out;
    };
    if let Some(multi) = mesh_multicontact(shape_a, shape_b, base, margin) {
        return multi;
    }
    out.push(base);
    out
}

fn barycentric_triangle_origin(a: Vec3, b: Vec3, c: Vec3, point: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = point - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1.0e-12 {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w, v, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn off_pivot_cube_mesh() -> ConvexMesh {
        ConvexMesh {
            vertices: vec![
                Vec3::new(-0.5, 9.5, -0.5),
                Vec3::new(0.5, 9.5, -0.5),
                Vec3::new(-0.5, 10.5, -0.5),
                Vec3::new(0.5, 10.5, -0.5),
                Vec3::new(-0.5, 9.5, 0.5),
                Vec3::new(0.5, 9.5, 0.5),
                Vec3::new(-0.5, 10.5, 0.5),
                Vec3::new(0.5, 10.5, 0.5),
            ],
            faces: vec![],
        }
    }

    #[test]
    fn quad_reducer_selects_maximum_area_subset() {
        let mut points = [Vec3::ZERO; MULTI_CLIP_CAP];
        points[0] = Vec3::new(-1.0, -1.0, 0.0);
        points[1] = Vec3::new(0.0, 0.0, 0.0);
        points[2] = Vec3::new(1.0, -1.0, 0.0);
        points[3] = Vec3::new(1.0, 1.0, 0.0);
        points[4] = Vec3::new(-1.0, 1.0, 0.0);

        let selected = quad_indices(&points, 5);

        assert_eq!(selected, [0, 2, 3, 4]);
        assert!(quad_area(&points, selected) > 0.0);
    }

    #[test]
    fn off_pivot_rotational_crossing_uses_pivot_speed() {
        let meshes = [off_pivot_cube_mesh()];
        let from = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let to = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
        };
        let target = GeomPose {
            position: Vec3::new(-7.071_068, 7.071_068, 0.0),
            orientation: Quat::IDENTITY,
        };
        let result = ccd_sweep_convex(
            &GeomShape::Mesh { mesh_id: 0 },
            &from,
            &to,
            &GeomShape::Sphere { radius: 0.001 },
            &target,
            &meshes,
        );
        let (toi, _) = result.expect("off-pivot rotation should cross the target");
        assert!((0.4..0.55).contains(&toi));
    }

    #[test]
    fn capped_off_pivot_rotational_crossing_returns_contact() {
        let meshes = [off_pivot_cube_mesh()];
        let from = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let to = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2),
        };
        let target = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let result = ccd_sweep_convex(
            &GeomShape::Mesh { mesh_id: 0 },
            &from,
            &to,
            &GeomShape::Sphere { radius: 84.9 },
            &target,
            &meshes,
        );
        let (_, contact) = result.expect("capped rotational crossing should remain observable");
        assert_eq!(contact.penetration, 0.0);
    }
}
