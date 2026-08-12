//! Deterministic quadric-error mesh levels and screen-space selection.

use crate::culling::Aabb;
use crate::math::{Mat4, Vec3};
use crate::mesh::{Mesh, MeshVertex};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// A cached level-of-detail mesh chain.
///
/// Level zero is the source mesh. Further levels use deterministic
/// Garland-Heckbert quadric error metric edge collapses. Collapsed vertices
/// use edge midpoints, midpoint UVs, and normals rebuilt from the new faces.
/// See Garland and Heckbert, "Surface Simplification Using Quadric Error
/// Metrics", SIGGRAPH 1997.
#[derive(Clone, Debug, PartialEq)]
pub struct LodMesh {
    levels: Vec<Mesh>,
    thresholds: Vec<f32>,
}

/// The selected level and its screen-space extent.
///
/// Keeping this value lets camera and shadow submissions share one decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodSelection {
    level: usize,
    screen_extent: f32,
}

/// Optional reprojection for callers that know a mesh is a sphere.
///
/// General QEM simplification does not infer or require spherical geometry.
/// Reprojection is explicit and uses a relative tolerance for the center
/// distance check.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereProjection {
    pub center: Vec3,
    pub radius: f32,
    pub relative_tolerance: f32,
}

impl SphereProjection {
    pub const fn new(center: Vec3, radius: f32, relative_tolerance: f32) -> Self {
        Self {
            center,
            radius,
            relative_tolerance,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimplifyOptions {
    pub sphere_projection: Option<SphereProjection>,
}

impl LodSelection {
    pub(crate) const fn at_level(level: usize) -> Self {
        Self {
            level,
            screen_extent: 0.0,
        }
    }

    pub const fn level(self) -> usize {
        self.level
    }

    pub const fn screen_extent(self) -> f32 {
        self.screen_extent
    }
}

impl LodMesh {
    pub const DEFAULT_RATIOS: [f32; 3] = [0.5, 0.25, 0.125];
    pub const DEFAULT_THRESHOLDS: [f32; 3] = [256.0, 128.0, 64.0];

    /// Builds the default three-level chain at approximately 50%, 25%, and
    /// 12.5% of the source triangle count.
    pub fn new(mesh: Mesh) -> Self {
        Self::with_ratios(mesh, &Self::DEFAULT_RATIOS)
    }

    pub fn with_ratios(mesh: Mesh, ratios: &[f32]) -> Self {
        Self::with_ratios_and_options(mesh, ratios, SimplifyOptions::default())
    }

    pub fn with_ratios_and_options(mesh: Mesh, ratios: &[f32], options: SimplifyOptions) -> Self {
        let source = mesh.clone();
        let mut levels = vec![source.clone()];
        let source_triangles = source.indices().len();
        for &ratio in ratios {
            let ratio = if ratio.is_finite() {
                ratio.clamp(0.0, 1.0)
            } else {
                0.0
            };
            let target = ((source_triangles as f32 * ratio).round() as usize).max(1);
            let previous = levels.last().expect("lod chain has level zero");
            let target = target.min(previous.indices().len().saturating_sub(1).max(1));
            levels.push(simplify_qem_with_options(previous, target, options));
        }
        let thresholds = default_thresholds(levels.len().saturating_sub(1));
        Self { levels, thresholds }
    }

    pub fn from_levels(levels: Vec<Mesh>, thresholds: Vec<f32>) -> Option<Self> {
        if levels.is_empty() {
            return None;
        }
        let mut lod = Self {
            levels,
            thresholds: Vec::new(),
        };
        lod.set_thresholds(thresholds);
        Some(lod)
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn level(&self, index: usize) -> Option<&Mesh> {
        self.levels.get(index)
    }

    pub fn original(&self) -> &Mesh {
        &self.levels[0]
    }

    pub fn levels(&self) -> &[Mesh] {
        &self.levels
    }

    pub fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Sets thresholds from largest to smallest projected extent.
    /// Invalid values are removed, and the result is clamped to the number
    /// of simplified levels. Selection reads this cache directly.
    pub fn set_thresholds(&mut self, thresholds: Vec<f32>) {
        let count = self.levels.len().saturating_sub(1);
        let mut sanitized = thresholds
            .into_iter()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .collect::<Vec<_>>();
        sanitized.truncate(count);
        for index in 1..sanitized.len() {
            sanitized[index] = sanitized[index].min(sanitized[index - 1]);
        }
        while sanitized.len() < count {
            let next = sanitized.last().copied().unwrap_or(0.0) * 0.5;
            sanitized.push(next);
        }
        self.thresholds = sanitized;
    }

    pub fn set_selection_thresholds(&mut self, thresholds: Vec<f32>) {
        self.set_thresholds(thresholds);
    }

    pub fn select_level_for_extent(&self, screen_extent: f32) -> usize {
        let extent = if screen_extent.is_finite() {
            screen_extent.max(0.0)
        } else {
            f32::INFINITY
        };
        self.thresholds
            .iter()
            .take_while(|threshold| extent < **threshold)
            .count()
            .min(self.levels.len().saturating_sub(1))
    }

    pub fn select(
        &self,
        view: Mat4,
        projection: Mat4,
        model: Mat4,
        width: usize,
        height: usize,
    ) -> LodSelection {
        let extent = projected_screen_extent(
            self.original().bounds(),
            view,
            projection,
            model,
            width,
            height,
        );
        LodSelection {
            level: self.select_level_for_extent(extent),
            screen_extent: extent,
        }
    }

    pub fn select_level(
        &self,
        view: Mat4,
        projection: Mat4,
        model: Mat4,
        width: usize,
        height: usize,
    ) -> usize {
        self.select(view, projection, model, width, height).level()
    }

    pub fn mesh_for(&self, selection: LodSelection) -> &Mesh {
        self.levels
            .get(selection.level)
            .unwrap_or_else(|| &self.levels[0])
    }

    pub fn mesh_at_level(&self, level: usize) -> &Mesh {
        self.levels.get(level).unwrap_or(&self.levels[0])
    }
}

fn default_thresholds(count: usize) -> Vec<f32> {
    let mut thresholds = Vec::with_capacity(count);
    let mut value = LodMesh::DEFAULT_THRESHOLDS[0];
    for index in 0..count {
        if let Some(default) = LodMesh::DEFAULT_THRESHOLDS.get(index) {
            value = *default;
        } else {
            value *= 0.5;
        }
        thresholds.push(value);
    }
    thresholds
}

/// Returns the world AABB's conservative projected diameter in pixels.
pub fn projected_screen_extent(
    bounds: Aabb,
    view: Mat4,
    projection: Mat4,
    model: Mat4,
    width: usize,
    height: usize,
) -> f32 {
    use crate::math::Vec4;
    let center = (bounds.min() + bounds.max()) * 0.5;
    let half = (bounds.max() - bounds.min()) * 0.5;
    let world_center = model * Vec4::new(center.x, center.y, center.z, 1.0);
    let world_radius = (model.upper_left3() * Vec3::new(half.x, half.y, half.z)).length();
    let view_center = view * world_center;
    let depth = -view_center.z;
    let row_x = matrix_row_norm(projection, 0);
    let row_y = matrix_row_norm(projection, 1);
    if !depth.is_finite()
        || depth <= f32::EPSILON
        || !world_radius.is_finite()
        || !row_x.is_finite()
        || !row_y.is_finite()
    {
        return f32::INFINITY;
    }
    let viewport_x = width as f32;
    let viewport_y = height as f32;
    let radius = world_radius / depth;
    (radius * row_x * viewport_x).max(radius * row_y * viewport_y)
}

pub fn projection_row_norms(projection: Mat4) -> (f32, f32) {
    (
        matrix_row_norm(projection, 0),
        matrix_row_norm(projection, 1),
    )
}

fn matrix_row_norm(matrix: Mat4, row: usize) -> f32 {
    let x = matrix.get(row, 0);
    let y = matrix.get(row, 1);
    let z = matrix.get(row, 2);
    (x * x + y * y + z * z).sqrt()
}

#[derive(Clone, Copy, Debug)]
struct Quadric {
    values: [f32; 10],
}

impl Quadric {
    const ZERO: Self = Self { values: [0.0; 10] };

    fn from_plane(normal: Vec3, distance: f32) -> Self {
        let p = [normal.x, normal.y, normal.z, distance];
        Self {
            values: [
                p[0] * p[0],
                p[0] * p[1],
                p[0] * p[2],
                p[0] * p[3],
                p[1] * p[1],
                p[1] * p[2],
                p[1] * p[3],
                p[2] * p[2],
                p[2] * p[3],
                p[3] * p[3],
            ],
        }
    }

    fn add_assign(&mut self, other: Self) {
        for (left, right) in self.values.iter_mut().zip(other.values) {
            *left += right;
        }
    }

    fn evaluate(self, position: Vec3) -> f32 {
        let [xx, xy, xz, xw, yy, yz, yw, zz, zw, ww] = self.values;
        let x = position.x;
        let y = position.y;
        let z = position.z;
        xx * x * x
            + 2.0 * xy * x * y
            + 2.0 * xz * x * z
            + 2.0 * xw * x
            + yy * y * y
            + 2.0 * yz * y * z
            + 2.0 * yw * y
            + zz * z * z
            + 2.0 * zw * z
            + ww
    }
}

impl std::ops::Add for Quadric {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = self;
        result.add_assign(rhs);
        result
    }
}

#[derive(Clone, Copy, Debug)]
struct CollapseCandidate {
    error: f32,
    edge_index: usize,
    a: usize,
    b: usize,
    position: Vec3,
}

impl Eq for CollapseCandidate {}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.error.to_bits() == other.error.to_bits() && self.edge_index == other.edge_index
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .error
            .total_cmp(&self.error)
            .then_with(|| other.edge_index.cmp(&self.edge_index))
    }
}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Simplifies one mesh to at most `target_triangles` with deterministic QEM.
pub fn simplify_qem(mesh: &Mesh, target_triangles: usize) -> Mesh {
    simplify_qem_with_options(mesh, target_triangles, SimplifyOptions::default())
}

/// Simplifies one mesh with explicit optional sphere reprojection.
pub fn simplify_qem_with_options(
    mesh: &Mesh,
    target_triangles: usize,
    options: SimplifyOptions,
) -> Mesh {
    if mesh.indices().len() <= target_triangles.max(1) {
        return mesh.clone();
    }
    let mut vertices = mesh.vertices().to_vec();
    let mut triangles = mesh.indices().to_vec();
    let preserve_closed_manifold = is_closed_manifold(&triangles);
    let orientation_center = consistent_orientation_center(mesh);
    while triangles.len() > target_triangles.max(1) {
        let quadrics = vertex_quadrics(&vertices, &triangles);
        let (edges, edge_indices) = unique_edges(&triangles);
        let mut queue = std::collections::BinaryHeap::with_capacity(edges.len());
        for (edge_index, &(a, b)) in edges.iter().enumerate() {
            let midpoint = (vertices[a].position() + vertices[b].position()) * 0.5;
            let quadric = quadrics[a] + quadrics[b];
            let project = |position: Vec3| {
                options
                    .sphere_projection
                    .filter(|projection| {
                        projection.radius.is_finite()
                            && projection.radius > 0.0
                            && projection.relative_tolerance.is_finite()
                            && projection.relative_tolerance >= 0.0
                    })
                    .and_then(|projection| {
                        let direction = position - projection.center;
                        let distance = direction.length();
                        (distance > projection.radius * projection.relative_tolerance)
                            .then(|| projection.center + direction.normalize() * projection.radius)
                    })
                    .unwrap_or(position)
            };
            let position = project(midpoint);
            let error = quadric.evaluate(position);
            queue.push(CollapseCandidate {
                error: if error.is_finite() {
                    error.max(0.0)
                } else {
                    f32::MAX
                },
                edge_index: edge_indices[edge_index],
                a,
                b,
                position,
            });
        }
        let candidate = loop {
            let Some(candidate) = queue.pop() else {
                break None;
            };
            if collapse_is_valid(
                &vertices,
                &triangles,
                candidate.a,
                candidate.b,
                candidate.position,
                orientation_center,
                options.sphere_projection.is_some(),
                preserve_closed_manifold,
            ) {
                break Some(candidate);
            }
        };
        let Some(candidate) = candidate else {
            break;
        };
        let position = candidate.position;
        let texcoord = match (
            vertices[candidate.a].texcoord(),
            vertices[candidate.b].texcoord(),
        ) {
            (Some(left), Some(right)) => Some((left + right) * 0.5),
            _ => None,
        };
        vertices[candidate.a] = MeshVertex::new(position, texcoord, None);
        triangles = triangles
            .into_iter()
            .filter_map(|mut triangle| {
                for index in &mut triangle {
                    if *index == candidate.b {
                        *index = candidate.a;
                    }
                }
                (triangle[0] != triangle[1]
                    && triangle[1] != triangle[2]
                    && triangle[2] != triangle[0])
                    .then_some(triangle)
            })
            .collect();
        vertices[candidate.b] = MeshVertex::new(position, texcoord, None);
    }
    compact_simplified_mesh(vertices, triangles)
}

fn vertex_quadrics(vertices: &[MeshVertex], triangles: &[[usize; 3]]) -> Vec<Quadric> {
    let mut quadrics = vec![Quadric::ZERO; vertices.len()];
    for &[a, b, c] in triangles {
        let Some(vertex_a) = vertices.get(a) else {
            continue;
        };
        let Some(vertex_b) = vertices.get(b) else {
            continue;
        };
        let Some(vertex_c) = vertices.get(c) else {
            continue;
        };
        let normal = (vertex_b.position() - vertex_a.position())
            .cross(vertex_c.position() - vertex_a.position());
        let length = normal.length();
        if !length.is_finite() || length <= f32::EPSILON {
            continue;
        }
        let normal = normal / length;
        let plane = Quadric::from_plane(normal, -normal.dot(vertex_a.position()));
        for index in [a, b, c] {
            quadrics[index].add_assign(plane);
        }
    }
    quadrics
}

fn unique_edges(triangles: &[[usize; 3]]) -> (Vec<(usize, usize)>, Vec<usize>) {
    let mut edge_map = BTreeMap::new();
    let mut edges = Vec::new();
    let mut indices = Vec::new();
    for &[a, b, c] in triangles {
        for (left, right) in [(a, b), (b, c), (c, a)] {
            let edge = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            if !edge_map.contains_key(&edge) {
                let index = edge_map.len();
                edge_map.insert(edge, index);
                edges.push(edge);
                indices.push(index);
            }
        }
    }
    (edges, indices)
}

#[allow(clippy::too_many_arguments)]
fn collapse_is_valid(
    vertices: &[MeshVertex],
    triangles: &[[usize; 3]],
    a: usize,
    b: usize,
    position: Vec3,
    orientation_center: Option<(Vec3, bool)>,
    preserve_closed_manifold: bool,
    preserve_convex: bool,
) -> bool {
    if preserve_closed_manifold && !collapse_keeps_closed_manifold(triangles, a, b) {
        return false;
    }
    if preserve_closed_manifold && !collapse_has_valid_link(triangles, a, b) {
        return false;
    }
    for &[x, y, z] in triangles {
        if x != a && x != b && y != a && y != b && z != a && z != b {
            continue;
        }
        let old_a = vertices[x].position();
        let old_b = vertices[y].position();
        let old_c = vertices[z].position();
        let old_normal = (old_b - old_a).cross(old_c - old_a);
        let mut replacement = [old_a, old_b, old_c];
        for (index, vertex) in [x, y, z].into_iter().enumerate() {
            if vertex == a || vertex == b {
                replacement[index] = position;
            }
        }
        if replacement[0] == replacement[1]
            || replacement[1] == replacement[2]
            || replacement[2] == replacement[0]
        {
            continue;
        }
        let new_normal = (replacement[1] - replacement[0]).cross(replacement[2] - replacement[0]);
        if old_normal.dot(new_normal) <= 0.0 || new_normal.length() <= old_normal.length() * 1.0e-3
        {
            return false;
        }
        if let Some((center, outward_sign)) = orientation_center {
            for point in replacement {
                if (new_normal.dot(point - center).is_sign_positive()) != outward_sign {
                    return false;
                }
            }
            if preserve_convex {
                let centroid = (replacement[0] + replacement[1] + replacement[2]) / 3.0;
                let components = [
                    new_normal.x * (centroid.x - center.x),
                    new_normal.y * (centroid.y - center.y),
                    new_normal.z * (centroid.z - center.z),
                ];
                if components.iter().any(|&component| component < 0.0) {
                    return false;
                }
                let scale = vertices
                    .iter()
                    .map(|vertex| vertex.position().length())
                    .fold(0.0, f32::max);
                let tolerance = new_normal.length() * scale * 1.0e-5;
                for vertex in vertices {
                    if new_normal.dot(vertex.position() - replacement[0]) > tolerance {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn is_closed_manifold(triangles: &[[usize; 3]]) -> bool {
    let mut edge_counts = BTreeMap::new();
    for &[a, b, c] in triangles {
        for (left, right) in [(a, b), (b, c), (c, a)] {
            let edge = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            *edge_counts.entry(edge).or_insert(0usize) += 1;
        }
    }
    !edge_counts.is_empty() && edge_counts.values().all(|&count| count == 2)
}

fn collapse_keeps_closed_manifold(triangles: &[[usize; 3]], a: usize, b: usize) -> bool {
    let mut edge_counts = BTreeMap::new();
    for &[mut x, mut y, mut z] in triangles {
        for index in [&mut x, &mut y, &mut z] {
            if *index == b {
                *index = a;
            }
        }
        if x == y || y == z || z == x {
            continue;
        }
        for (left, right) in [(x, y), (y, z), (z, x)] {
            let edge = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            *edge_counts.entry(edge).or_insert(0usize) += 1;
        }
    }
    !edge_counts.is_empty() && edge_counts.values().all(|&count| count == 2)
}

fn collapse_has_valid_link(triangles: &[[usize; 3]], a: usize, b: usize) -> bool {
    let mut neighbors_a = std::collections::BTreeSet::new();
    let mut neighbors_b = std::collections::BTreeSet::new();
    let mut opposite = std::collections::BTreeSet::new();
    let mut incident = 0;
    for &[x, y, z] in triangles {
        let triangle = [x, y, z];
        if triangle.contains(&a) {
            for &vertex in &triangle {
                if vertex != a {
                    neighbors_a.insert(vertex);
                }
            }
        }
        if triangle.contains(&b) {
            for &vertex in &triangle {
                if vertex != b {
                    neighbors_b.insert(vertex);
                }
            }
        }
        if triangle.contains(&a) && triangle.contains(&b) {
            incident += 1;
            for &vertex in &triangle {
                if vertex != a && vertex != b {
                    opposite.insert(vertex);
                }
            }
        }
    }
    incident == 2
        && opposite.len() == 2
        && neighbors_a
            .intersection(&neighbors_b)
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            == opposite
}

fn consistent_orientation_center(mesh: &Mesh) -> Option<(Vec3, bool)> {
    let center = (mesh.bounds().min() + mesh.bounds().max()) * 0.5;
    let scale = (mesh.bounds().max() - mesh.bounds().min()).length();
    let tolerance = (scale * 1.0e-6).max(f32::MIN_POSITIVE);
    if mesh.indices().is_empty() {
        return None;
    }
    let mut sign = None;
    for &[a, b, c] in mesh.indices() {
        let pa = mesh.vertex(a)?.position();
        let pb = mesh.vertex(b)?.position();
        let pc = mesh.vertex(c)?.position();
        let normal = (pb - pa).cross(pc - pa);
        let side = normal.dot((pa + pb + pc) / 3.0 - center);
        if side.abs() <= tolerance {
            return None;
        }
        let current = side.is_sign_positive();
        if let Some(expected) = sign {
            if expected != current {
                return None;
            }
        } else {
            sign = Some(current);
        }
    }
    Some((center, sign?))
}

fn compact_simplified_mesh(vertices: Vec<MeshVertex>, triangles: Vec<[usize; 3]>) -> Mesh {
    let mut remap = vec![usize::MAX; vertices.len()];
    let mut compact_vertices = Vec::new();
    let mut compact_triangles = Vec::with_capacity(triangles.len());
    for [a, b, c] in triangles {
        let mut compact = [0; 3];
        for (slot, index) in [a, b, c].into_iter().enumerate() {
            if remap[index] == usize::MAX {
                remap[index] = compact_vertices.len();
                compact_vertices.push(vertices[index]);
            }
            compact[slot] = remap[index];
        }
        compact_triangles.push(compact);
    }
    Mesh::new(compact_vertices, compact_triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subdivided_octahedron(levels: usize) -> Mesh {
        let mut vertices = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];
        let mut triangles = vec![
            [0, 2, 1],
            [0, 3, 2],
            [0, 4, 3],
            [0, 1, 4],
            [5, 1, 2],
            [5, 2, 3],
            [5, 3, 4],
            [5, 4, 1],
        ];
        for _ in 0..levels {
            let mut midpoint_cache = BTreeMap::new();
            let mut next = Vec::with_capacity(triangles.len() * 4);
            for [a, b, c] in triangles {
                let midpoint =
                    |left: usize,
                     right: usize,
                     vertices: &mut Vec<Vec3>,
                     cache: &mut BTreeMap<(usize, usize), usize>| {
                        let edge = if left < right {
                            (left, right)
                        } else {
                            (right, left)
                        };
                        if let Some(&index) = cache.get(&edge) {
                            return index;
                        }
                        let index = vertices.len();
                        vertices.push((vertices[left] + vertices[right]).normalize());
                        cache.insert(edge, index);
                        index
                    };
                let ab = midpoint(a, b, &mut vertices, &mut midpoint_cache);
                let bc = midpoint(b, c, &mut vertices, &mut midpoint_cache);
                let ca = midpoint(c, a, &mut vertices, &mut midpoint_cache);
                next.extend_from_slice(&[[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
            }
            triangles = next;
        }
        Mesh::new(
            vertices
                .into_iter()
                .map(|position| MeshVertex::new(position, None, None))
                .collect(),
            triangles,
        )
    }

    fn cube_mesh() -> Mesh {
        let positions = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        .map(|(x, y, z)| MeshVertex::new(Vec3::new(x, y, z), None, None))
        .to_vec();
        Mesh::new(
            positions,
            vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
                [0, 1, 5],
                [0, 5, 4],
                [3, 7, 6],
                [3, 6, 2],
            ],
        )
    }

    fn cylinder_mesh(sides: usize) -> Mesh {
        let mut vertices = Vec::with_capacity(sides * 2 + 2);
        for z in [-1.0, 1.0] {
            for side in 0..sides {
                let angle = side as f32 * std::f32::consts::TAU / sides as f32;
                vertices.push(MeshVertex::new(
                    Vec3::new(angle.cos(), angle.sin(), z),
                    None,
                    None,
                ));
            }
        }
        let bottom_center = vertices.len();
        vertices.push(MeshVertex::new(Vec3::new(0.0, 0.0, -1.0), None, None));
        let top_center = vertices.len();
        vertices.push(MeshVertex::new(Vec3::new(0.0, 0.0, 1.0), None, None));
        let mut triangles = Vec::with_capacity(sides * 4);
        for side in 0..sides {
            let next = (side + 1) % sides;
            let bottom = side;
            let bottom_next = next;
            let top = sides + side;
            let top_next = sides + next;
            triangles.extend_from_slice(&[
                [bottom, bottom_next, top_next],
                [bottom, top_next, top],
                [bottom_center, bottom_next, bottom],
                [top_center, top, top_next],
            ]);
        }
        Mesh::new(vertices, triangles)
    }

    fn assert_simplified_shape(source: Mesh, target: usize, closed: bool) {
        let center = consistent_orientation_center(&source);
        let simplified = simplify_qem(&source, target);
        assert_eq!(simplified.indices().len(), target);
        for &[a, b, c] in simplified.indices() {
            let pa = simplified.vertex(a).unwrap().position();
            let pb = simplified.vertex(b).unwrap().position();
            let pc = simplified.vertex(c).unwrap().position();
            let normal = (pb - pa).cross(pc - pa);
            assert!(normal.length() > 0.0);
            if closed {
                let (center, outward_sign) = center.unwrap();
                assert_eq!(
                    normal.dot((pa + pb + pc) / 3.0 - center).is_sign_positive(),
                    outward_sign
                );
            }
        }
    }

    #[test]
    fn qem_evaluation_matches_plane_equations() {
        let axis_aligned = Quadric::from_plane(Vec3::new(1.0, 0.0, 0.0), -1.0);
        assert_eq!(axis_aligned.evaluate(Vec3::new(1.0, 7.0, -3.0)), 0.0);
        assert_eq!(axis_aligned.evaluate(Vec3::new(0.0, 7.0, -3.0)), 1.0);

        let oblique = Quadric::from_plane(Vec3::new(1.0, 2.0, 3.0), -4.0);
        assert_eq!(oblique.evaluate(Vec3::new(4.0, 0.0, 0.0)), 0.0);
        assert_eq!(oblique.evaluate(Vec3::ZERO), 16.0);
        let combined = axis_aligned + oblique;
        assert_eq!(combined.evaluate(Vec3::new(1.0, 1.0, 1.0)), 4.0);
    }

    #[test]
    fn projected_extent_uses_render_projection_and_pixel_radius() {
        let bounds = Aabb::new(Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0));
        let projection = Mat4::perspective_from_focal_length(2.0, 1.0, 0.1, 100.0);
        let extent = projected_screen_extent(
            bounds,
            Mat4::IDENTITY,
            projection,
            Mat4::translate(Vec3::new(0.0, 0.0, -10.0)),
            100,
            100,
        );
        assert_eq!(extent, 10.0);
    }

    #[test]
    fn qem_simplifies_sphere_cube_cylinder_and_skinned_inputs() {
        assert_simplified_shape(subdivided_octahedron(2), 32, true);
        assert_simplified_shape(cube_mesh(), 6, true);
        assert_simplified_shape(cylinder_mesh(12), 24, true);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/arm.gltf");
        let skinned = crate::gltf::GltfAsset::load(path)
            .unwrap()
            .pose_mesh(0, 0, 0, Some(0), 1.0)
            .unwrap();
        assert_simplified_shape(skinned, 2, false);
    }

    #[test]
    fn qem_relative_scale_does_not_protect_tiny_spheres() {
        let source = subdivided_octahedron(2);
        let tiny = Mesh::new(
            source
                .vertices()
                .iter()
                .map(|vertex| MeshVertex::new(vertex.position() * 1.0e-5, None, None))
                .collect(),
            source.indices().to_vec(),
        );
        assert_eq!(simplify_qem(&tiny, 32).indices().len(), 32);
    }

    #[test]
    fn qem_builds_identical_levels_and_hits_targets() {
        let source = subdivided_octahedron(3);
        assert_eq!(
            consistent_orientation_center(&source).map(|(_, sign)| sign),
            Some(true)
        );
        let first = LodMesh::new(source.clone());
        let second = LodMesh::new(source);
        assert_eq!(first.levels(), second.levels());
        let counts = first
            .levels()
            .iter()
            .map(|mesh| mesh.indices().len())
            .collect::<Vec<_>>();
        assert_eq!(counts[0], 512);
        assert!(counts.windows(2).all(|window| window[0] > window[1]));
        assert!((counts[1] as f32 / counts[0] as f32 - 0.5).abs() <= 0.1);
        assert!((counts[2] as f32 / counts[0] as f32 - 0.25).abs() <= 0.1);
        assert!((counts[3] as f32 / counts[0] as f32 - 0.125).abs() <= 0.1);
        for (level, mesh) in first.levels().iter().enumerate() {
            for &[a, b, c] in mesh.indices() {
                let pa = mesh.vertex(a).unwrap().position();
                let pb = mesh.vertex(b).unwrap().position();
                let pc = mesh.vertex(c).unwrap().position();
                let normal = (pb - pa).cross(pc - pa);
                assert!(
                    normal.length() > 0.0,
                    "degenerate triangle at level {level}"
                );
                assert!(
                    normal.dot((pa + pb + pc) / 3.0) > 0.0,
                    "inward triangle at level {level}: {pa:?} {pb:?} {pc:?} normal {normal:?}"
                );
            }
        }
    }

    #[test]
    fn qem_preserves_sphere_orientation_and_attributes() {
        let source = subdivided_octahedron(2);
        let simplified = simplify_qem(&source, 32);
        for vertex in simplified.vertices() {
            assert!(
                (vertex.position().length() - 1.0).abs() < 0.2,
                "position {:?} radius {}",
                vertex.position(),
                vertex.position().length()
            );
            assert!(vertex.normal().is_some());
        }
        for (triangle_index, &[a, b, c]) in simplified.indices().iter().enumerate() {
            let pa = simplified.vertex(a).unwrap().position();
            let pb = simplified.vertex(b).unwrap().position();
            let pc = simplified.vertex(c).unwrap().position();
            let normal = (pb - pa).cross(pc - pa);
            assert!(normal.length() > 0.0);
            assert!(
                normal.dot((pa + pb + pc) / 3.0) > 0.0,
                "triangle {triangle_index}: {pa:?} {pb:?} {pc:?} normal {normal:?}"
            );
        }
    }

    #[test]
    fn lod_threshold_setter_sanitizes_and_updates_selection() {
        let mut lod = LodMesh::new(subdivided_octahedron(1));
        lod.set_thresholds(vec![f32::NAN, 300.0, -1.0, 100.0]);
        assert_eq!(lod.thresholds(), &[300.0, 100.0, 50.0]);
        assert_eq!(lod.select_level_for_extent(400.0), 0);
        assert_eq!(lod.select_level_for_extent(20.0), 3);
    }

    #[test]
    fn qem_priority_breaks_equal_errors_by_stable_edge_index() {
        let lower = CollapseCandidate {
            error: 1.0,
            edge_index: 3,
            a: 0,
            b: 1,
            position: Vec3::ZERO,
        };
        let higher = CollapseCandidate {
            error: 1.0,
            edge_index: 7,
            a: 0,
            b: 1,
            position: Vec3::ZERO,
        };
        let mut queue = std::collections::BinaryHeap::from([higher, lower]);
        assert_eq!(queue.pop().unwrap().edge_index, 3);
    }

    #[test]
    fn collapse_rejects_a_face_orientation_flip_without_surface_center() {
        let vertices = vec![
            MeshVertex::new(Vec3::ZERO, None, None),
            MeshVertex::new(Vec3::new(1.0, 0.0, 0.0), None, None),
            MeshVertex::new(Vec3::new(0.0, 1.0, 0.0), None, None),
            MeshVertex::new(Vec3::new(0.0, 0.0, 1.0), None, None),
        ];
        let triangles = vec![[0, 2, 3]];
        assert!(!collapse_is_valid(
            &vertices,
            &triangles,
            0,
            1,
            Vec3::new(0.0, 2.0, 0.0),
            None,
            false,
            false,
        ));
    }
}
