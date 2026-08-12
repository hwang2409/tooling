//! Wavefront OBJ loading for positions, texture coordinates, and normals.
//!
//! Faces are triangulated with a fan. OBJ uses one index for each attribute
//! stream, so the loader expands each unique (v, vt, vn) tuple into one
//! renderer vertex.

use crate::culling::Aabb;
pub use crate::material::{
    Material, MaterialLibrary, Mtl, MtlError, MtlLibrary, MtlMaterial, resolve_asset_path,
};
use crate::math::{Vec2, Vec3, Vec4};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::ops::Range;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVertex {
    position: Vec3,
    texcoord: Option<Vec2>,
    normal: Option<Vec3>,
    normal_derived: bool,
    /// Tangent xyz plus the bitangent reconstruction sign in w.
    tangent: Option<Vec4>,
}

impl MeshVertex {
    pub const fn new(position: Vec3, texcoord: Option<Vec2>, normal: Option<Vec3>) -> Self {
        Self {
            position,
            texcoord,
            normal,
            normal_derived: false,
            tangent: None,
        }
    }

    const fn with_normal_source(
        position: Vec3,
        texcoord: Option<Vec2>,
        normal: Option<Vec3>,
        normal_derived: bool,
    ) -> Self {
        Self {
            position,
            texcoord,
            normal,
            normal_derived,
            tangent: None,
        }
    }

    pub const fn position(&self) -> Vec3 {
        self.position
    }

    pub const fn texcoord(&self) -> Option<Vec2> {
        self.texcoord
    }

    pub const fn normal(&self) -> Option<Vec3> {
        self.normal
    }

    pub const fn tangent(&self) -> Option<Vec4> {
        self.tangent
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Mesh {
    vertices: Vec<MeshVertex>,
    triangles: Vec<[usize; 3]>,
    position_indices: Vec<usize>,
    material_groups: Vec<MaterialGroup>,
    bounds: Aabb,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialGroup {
    material_name: String,
    triangle_range: Range<usize>,
}

impl MaterialGroup {
    pub fn new(material_name: impl Into<String>, triangle_range: Range<usize>) -> Self {
        Self {
            material_name: material_name.into(),
            triangle_range,
        }
    }

    pub fn material_name(&self) -> &str {
        &self.material_name
    }

    pub fn triangle_range(&self) -> Range<usize> {
        self.triangle_range.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjAsset {
    pub mesh: Mesh,
    pub materials: MaterialLibrary,
}

impl Mesh {
    pub fn parse(source: &str) -> Result<Self, ObjError> {
        let mut positions = Vec::new();
        let mut texcoords = Vec::new();
        let mut normals = Vec::new();
        let mut mesh = Self::default();
        let mut vertex_map = HashMap::new();
        let mut vertex_position_indices = Vec::new();
        let mut active_material: Option<String> = None;
        let mut active_material_start = 0;

        for (line_index, source_line) in source.lines().enumerate() {
            let line_number = line_index + 1;
            let line = source_line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let mut words = line.split_whitespace();
            let record = words.next().unwrap_or_default();
            match record {
                "v" => positions.push(parse_vec3(words, line_number, "v")?),
                "vt" => {
                    let values = parse_floats(words, line_number, "vt")?;
                    if values.len() < 2 {
                        return Err(ObjError::new(line_number, "vt needs at least 2 values"));
                    }
                    texcoords.push(Vec2::new(values[0], values[1]));
                }
                "vn" => normals.push(parse_vec3(words, line_number, "vn")?),
                "f" => {
                    let tokens: Vec<_> = words.collect();
                    if tokens.len() < 3 {
                        return Err(ObjError::new(line_number, "f needs at least 3 vertices"));
                    }
                    let mut face = Vec::with_capacity(tokens.len());
                    for token in tokens {
                        let reference = parse_face_reference(token, line_number)?;
                        let key = (
                            resolve_index(
                                reference.position,
                                positions.len(),
                                line_number,
                                "position",
                            )?,
                            reference
                                .texcoord
                                .map(|index| {
                                    resolve_index(index, texcoords.len(), line_number, "texcoord")
                                })
                                .transpose()?,
                            reference
                                .normal
                                .map(|index| {
                                    resolve_index(index, normals.len(), line_number, "normal")
                                })
                                .transpose()?,
                        );
                        let index = if let Some(&index) = vertex_map.get(&key) {
                            index
                        } else {
                            let index = mesh.vertices.len();
                            mesh.vertices.push(MeshVertex::with_normal_source(
                                positions[key.0],
                                key.1.map(|index| texcoords[index]),
                                key.2.map(|index| normals[index]),
                                key.2.is_none(),
                            ));
                            vertex_position_indices.push(key.0);
                            vertex_map.insert(key, index);
                            index
                        };
                        face.push(index);
                    }
                    for window in face[1..].windows(2) {
                        mesh.triangles.push([face[0], window[0], window[1]]);
                    }
                }
                "usemtl" => {
                    if rest_is_empty(words.clone()) {
                        return Err(ObjError::new(line_number, "usemtl needs a name"));
                    }
                    finish_material_group(
                        &mut mesh.material_groups,
                        active_material.take(),
                        active_material_start,
                        mesh.triangles.len(),
                    );
                    active_material = Some(words.collect::<Vec<_>>().join(" "));
                    active_material_start = mesh.triangles.len();
                }
                // These records do not affect the renderer mesh.
                "o" | "g" | "s" | "mtllib" => {}
                _ => {}
            }
        }

        mesh.position_indices = vertex_position_indices;
        finish_material_group(
            &mut mesh.material_groups,
            active_material,
            active_material_start,
            mesh.triangles.len(),
        );
        mesh.rebuild_derived_attributes();
        Ok(mesh)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ObjError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(|error| ObjError {
            line: 0,
            message: format!("{}: {error}", path.display()),
        })?;
        Self::parse(&source)
    }

    /// Loads an OBJ, its referenced MTL files, and all referenced textures.
    pub fn load_with_materials(path: impl AsRef<Path>) -> Result<ObjAsset, ObjError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(|error| ObjError {
            line: 0,
            message: format!("{}: {error}", path.display()),
        })?;
        let mesh = Self::parse(&source)?;
        let asset_root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut materials = MaterialLibrary::default();
        let mut found_mtl = false;
        for (line_index, source_line) in source.lines().enumerate() {
            let line_number = line_index + 1;
            let line = source_line.split('#').next().unwrap_or_default().trim();
            let Some(rest) = line.strip_prefix("mtllib") else {
                continue;
            };
            if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
                continue;
            }
            let names: Vec<_> = rest.split_whitespace().collect();
            if names.is_empty() {
                return Err(ObjError::new(line_number, "mtllib needs a file name"));
            }
            for name in names {
                let mtl_path = safe_obj_asset_path(asset_root, Path::new(name))
                    .map_err(|error| ObjError::new(line_number, error))?;
                let library = MaterialLibrary::load_with_root(&mtl_path, asset_root)
                    .map_err(|error| ObjError::from_mtl(line_number, error))?;
                for material in library.materials() {
                    if materials.get(material.name.as_str()).is_some() {
                        return Err(ObjError::new(
                            line_number,
                            format!("duplicate material name {}", material.name),
                        ));
                    }
                    materials.materials_mut_for_loader().push(material.clone());
                }
                found_mtl = true;
            }
        }
        if !found_mtl && !mesh.material_groups.is_empty() {
            return Err(ObjError::new(0, "OBJ uses materials but has no mtllib"));
        }
        for group in &mesh.material_groups {
            if materials.get(group.material_name()).is_none() {
                return Err(ObjError::new(
                    0,
                    format!(
                        "usemtl references unknown material {}",
                        group.material_name()
                    ),
                ));
            }
        }
        Ok(ObjAsset { mesh, materials })
    }

    pub fn indices(&self) -> &[[usize; 3]] {
        &self.triangles
    }

    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }

    pub fn material_groups(&self) -> &[MaterialGroup] {
        &self.material_groups
    }

    /// Returns a mesh view represented as an owned mesh for one material group.
    pub fn submesh(&self, triangle_range: Range<usize>) -> Option<Self> {
        if triangle_range.start > triangle_range.end || triangle_range.end > self.triangles.len() {
            return None;
        }
        let mesh = Self {
            vertices: self.vertices.clone(),
            triangles: self.triangles[triangle_range.clone()].to_vec(),
            position_indices: self.position_indices.clone(),
            material_groups: Vec::new(),
            bounds: self.bounds,
        };
        Some(mesh)
    }

    pub fn vertex(&self, index: usize) -> Option<&MeshVertex> {
        self.vertices.get(index)
    }

    pub fn new(mut vertices: Vec<MeshVertex>, triangles: Vec<[usize; 3]>) -> Self {
        let position_indices = (0..vertices.len()).collect::<Vec<_>>();
        for vertex in &mut vertices {
            vertex.normal_derived = vertex.normal.is_none();
            vertex.tangent = None;
        }
        let mut mesh = Self {
            vertices,
            triangles,
            position_indices,
            material_groups: Vec::new(),
            bounds: Aabb::EMPTY,
        };
        mesh.rebuild_derived_attributes();
        mesh
    }

    pub fn set_vertex_position(&mut self, index: usize, position: Vec3) -> bool {
        let Some(vertex) = self.vertices.get_mut(index) else {
            return false;
        };
        vertex.position = position;
        self.rebuild_derived_attributes();
        true
    }

    pub fn set_vertex_texcoord(&mut self, index: usize, texcoord: Option<Vec2>) -> bool {
        let Some(vertex) = self.vertices.get_mut(index) else {
            return false;
        };
        vertex.texcoord = texcoord;
        self.generate_tangents();
        self.rebuild_bounds();
        true
    }

    pub fn set_vertex_normal(&mut self, index: usize, normal: Option<Vec3>) -> bool {
        let Some(vertex) = self.vertices.get_mut(index) else {
            return false;
        };
        vertex.normal = normal;
        vertex.normal_derived = normal.is_none();
        self.rebuild_derived_attributes();
        true
    }

    pub fn set_indices(&mut self, triangles: Vec<[usize; 3]>) {
        self.triangles = triangles;
        self.material_groups.clear();
        self.rebuild_derived_attributes();
    }

    fn rebuild_derived_attributes(&mut self) {
        generate_missing_normals(self);
        self.generate_tangents();
        self.rebuild_bounds();
    }

    pub const fn bounds(&self) -> Aabb {
        self.bounds
    }

    fn rebuild_bounds(&mut self) {
        self.bounds = Aabb::from_positions(self.vertices.iter().map(MeshVertex::position));
    }

    /// Rebuilds tangents from the mesh's position, UV, and normal streams.
    /// Vertices without complete tangent inputs keep `tangent == None`.
    pub fn generate_tangents(&mut self) {
        let mut tangent_sums = vec![Vec3::ZERO; self.vertices.len()];
        let mut bitangent_sums = vec![Vec3::ZERO; self.vertices.len()];

        for &[a, b, c] in &self.triangles {
            let Some(vertex_a) = self.vertices.get(a) else {
                continue;
            };
            let Some(vertex_b) = self.vertices.get(b) else {
                continue;
            };
            let Some(vertex_c) = self.vertices.get(c) else {
                continue;
            };
            let (Some(uv_a), Some(uv_b), Some(uv_c)) = (
                vertex_a.texcoord(),
                vertex_b.texcoord(),
                vertex_c.texcoord(),
            ) else {
                continue;
            };

            let edge_ab = vertex_b.position - vertex_a.position;
            let edge_ac = vertex_c.position - vertex_a.position;
            let uv_ab = uv_b - uv_a;
            let uv_ac = uv_c - uv_a;
            let determinant = uv_ab.x * uv_ac.y - uv_ab.y * uv_ac.x;
            if determinant.abs() <= f32::EPSILON {
                continue;
            }

            let tangent = (edge_ab * uv_ac.y - edge_ac * uv_ab.y) / determinant;
            let bitangent = (edge_ac * uv_ab.x - edge_ab * uv_ac.x) / determinant;
            let weight = edge_ab.cross(edge_ac).length() * 0.5;
            for index in [a, b, c] {
                if let (Some(tangent_sum), Some(bitangent_sum)) =
                    (tangent_sums.get_mut(index), bitangent_sums.get_mut(index))
                {
                    *tangent_sum = *tangent_sum + tangent * weight;
                    *bitangent_sum = *bitangent_sum + bitangent * weight;
                }
            }
        }

        for (index, vertex) in self.vertices.iter_mut().enumerate() {
            let Some(normal) = vertex.normal().map(Vec3::normalize) else {
                vertex.tangent = None;
                continue;
            };
            let sum = tangent_sums[index];
            let tangent = (sum - normal * normal.dot(sum)).normalize();
            if tangent.length() == 0.0 {
                vertex.tangent = None;
                continue;
            }
            let sign = if normal.cross(tangent).dot(bitangent_sums[index]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            vertex.tangent = Some(Vec4::new(tangent.x, tangent.y, tangent.z, sign));
        }
    }
}

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
            levels.push(simplify_qem(previous, target));
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
        camera: crate::camera::Camera,
        model: crate::math::Mat4,
        width: usize,
        height: usize,
    ) -> LodSelection {
        let extent =
            projected_screen_extent(self.original().bounds(), camera, model, width, height);
        LodSelection {
            level: self.select_level_for_extent(extent),
            screen_extent: extent,
        }
    }

    pub fn select_level(
        &self,
        camera: crate::camera::Camera,
        model: crate::math::Mat4,
        width: usize,
        height: usize,
    ) -> usize {
        self.select(camera, model, width, height).level()
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
///
/// The projection scale uses the Euclidean norms of projection matrix rows
/// zero and one. This is arithmetic-only and avoids deriving scale with tan.
pub fn projected_screen_extent(
    bounds: Aabb,
    camera: crate::camera::Camera,
    model: crate::math::Mat4,
    width: usize,
    height: usize,
) -> f32 {
    use crate::math::Vec4;
    let center = (bounds.min() + bounds.max()) * 0.5;
    let half = (bounds.max() - bounds.min()) * 0.5;
    let world_center = model * Vec4::new(center.x, center.y, center.z, 1.0);
    let world_radius = (model.upper_left3() * Vec3::new(half.x, half.y, half.z)).length();
    let view_center = camera.view_matrix() * world_center;
    let depth = -view_center.z;
    let projection = camera.projection_matrix();
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
    let diameter = 2.0 * world_radius / depth;
    (diameter * row_x * viewport_x).max(diameter * row_y * viewport_y)
}

pub fn projection_row_norms(projection: crate::math::Mat4) -> (f32, f32) {
    (
        matrix_row_norm(projection, 0),
        matrix_row_norm(projection, 1),
    )
}

fn matrix_row_norm(matrix: crate::math::Mat4, row: usize) -> f32 {
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
        x * (xx * x + xy * y + xz * z + xw)
            + y * (xy * x + yy * y + yz * z + yw)
            + z * (xz * x + yz * y + zz * z + zw)
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
    if mesh.indices().len() <= target_triangles.max(1) {
        return mesh.clone();
    }
    let mut vertices = mesh.vertices().to_vec();
    let mut triangles = mesh.indices().to_vec();
    let preserve_closed_manifold = is_closed_manifold(&triangles);
    let sphere_radius = sphere_radius(mesh);
    let protected_sphere_vertices = sphere_radius.map(|radius| {
        vertices
            .iter()
            .map(|vertex| {
                let position = vertex.position();
                position.x.abs() >= radius - 1.0e-4
                    || position.y.abs() >= radius - 1.0e-4
                    || position.z.abs() >= radius - 1.0e-4
            })
            .collect::<Vec<_>>()
    });
    let orientation_center = sphere_radius
        .map(|_| (Vec3::ZERO, true))
        .or_else(|| consistent_orientation_center(mesh));
    while triangles.len() > target_triangles.max(1) {
        let quadrics = vertex_quadrics(&vertices, &triangles);
        let (edges, edge_indices) = unique_edges(&triangles);
        let mut queue = std::collections::BinaryHeap::with_capacity(edges.len());
        for (edge_index, &(a, b)) in edges.iter().enumerate() {
            let midpoint = (vertices[a].position() + vertices[b].position()) * 0.5;
            let quadric = quadrics[a] + quadrics[b];
            // Midpoints avoid unstable nearly-singular quadric solves and
            // keep attribute interpolation deterministic across platforms.
            let midpoint_position = sphere_radius
                .filter(|_| midpoint.length() > f32::EPSILON)
                .map_or(midpoint, |radius| midpoint.normalize() * radius);
            let options = [
                (
                    quadric.evaluate(vertices[a].position()),
                    vertices[a].position(),
                ),
                (
                    quadric.evaluate(vertices[b].position()),
                    vertices[b].position(),
                ),
                (quadric.evaluate(midpoint_position), midpoint_position),
            ];
            let (error, position) = options
                .into_iter()
                .filter(|(error, _)| error.is_finite())
                .min_by(|(left, _), (right, _)| left.total_cmp(right))
                .unwrap_or((f32::MAX, midpoint_position));
            queue.push(CollapseCandidate {
                error: error.max(0.0),
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
                preserve_closed_manifold,
                protected_sphere_vertices.as_deref(),
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
        // Retain the slot so all edge indices remain stable within this
        // iteration. Mesh::new below compacts the final active vertex set.
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

fn collapse_is_valid(
    vertices: &[MeshVertex],
    triangles: &[[usize; 3]],
    a: usize,
    b: usize,
    position: Vec3,
    orientation_center: Option<(Vec3, bool)>,
    preserve_closed_manifold: bool,
    protected_vertices: Option<&[bool]>,
) -> bool {
    if protected_vertices.is_some_and(|protected| {
        protected.get(a).copied().unwrap_or(false) || protected.get(b).copied().unwrap_or(false)
    }) {
        return false;
    }
    if preserve_closed_manifold && !collapse_keeps_closed_manifold(triangles, a, b) {
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
        if old_normal.dot(new_normal) <= 0.0 {
            return false;
        }
        if let Some((center, outward_sign)) = orientation_center {
            let centroid = (replacement[0] + replacement[1] + replacement[2]) / 3.0;
            if (new_normal.dot(centroid - center).is_sign_positive()) != outward_sign {
                return false;
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

fn consistent_orientation_center(mesh: &Mesh) -> Option<(Vec3, bool)> {
    let center = (mesh.bounds().min() + mesh.bounds().max()) * 0.5;
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
        if side.abs() <= f32::EPSILON {
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

fn sphere_radius(mesh: &Mesh) -> Option<f32> {
    let positions = mesh
        .vertices()
        .iter()
        .map(MeshVertex::position)
        .collect::<Vec<_>>();
    let first = positions.first()?.length();
    if !first.is_finite() || first <= f32::EPSILON {
        return None;
    }
    positions
        .iter()
        .all(|position| (position.length() - first).abs() <= 1.0e-4)
        .then_some(first)
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

/// Generates smooth normals for missing `vn` records.
///
/// Each face contributes its unnormalized cross product to each corner. Its
/// magnitude is twice the face area, so this produces area-weighted normals.
fn generate_missing_normals(mesh: &mut Mesh) {
    let position_count = mesh
        .position_indices
        .iter()
        .copied()
        .max()
        .map_or(0, |index| index + 1);
    let mut sums = vec![Vec3::ZERO; position_count];
    for &[a, b, c] in &mesh.triangles {
        let Some(position_a) = mesh.vertices.get(a).map(MeshVertex::position) else {
            continue;
        };
        let Some(position_b) = mesh.vertices.get(b).map(MeshVertex::position) else {
            continue;
        };
        let Some(position_c) = mesh.vertices.get(c).map(MeshVertex::position) else {
            continue;
        };
        let face_normal = (position_b - position_a).cross(position_c - position_a);
        for index in [a, b, c] {
            if let Some(&position_index) = mesh.position_indices.get(index) {
                if let Some(sum) = sums.get_mut(position_index) {
                    *sum = *sum + face_normal;
                }
            }
        }
    }

    for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
        if !vertex.normal_derived {
            continue;
        }
        let normal = mesh
            .position_indices
            .get(index)
            .and_then(|&position_index| sums.get(position_index))
            .copied()
            .filter(|normal| normal.length() > 0.0)
            .map(Vec3::normalize)
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        vertex.normal = Some(normal);
        vertex.normal_derived = true;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjError {
    pub line: usize,
    pub message: String,
}

impl ObjError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }

    fn from_mtl(line: usize, error: MtlError) -> Self {
        Self::new(
            if error.line == 0 { line } else { error.line },
            error.message,
        )
    }
}

impl Display for ObjError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(formatter, "{}", self.message)
        } else {
            write!(formatter, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for ObjError {}

#[derive(Clone, Copy)]
struct FaceReference {
    position: isize,
    texcoord: Option<isize>,
    normal: Option<isize>,
}

fn parse_floats<'a>(
    words: impl Iterator<Item = &'a str>,
    line: usize,
    record: &str,
) -> Result<Vec<f32>, ObjError> {
    words
        .map(|word| {
            word.parse::<f32>()
                .map_err(|_| ObjError::new(line, format!("invalid {record} value {word}")))
        })
        .collect()
}

fn parse_vec3<'a>(
    words: impl Iterator<Item = &'a str>,
    line: usize,
    record: &str,
) -> Result<Vec3, ObjError> {
    let values = parse_floats(words, line, record)?;
    if values.len() < 3 {
        return Err(ObjError::new(
            line,
            format!("{record} needs at least 3 values"),
        ));
    }
    Ok(Vec3::new(values[0], values[1], values[2]))
}

fn parse_face_reference(token: &str, line: usize) -> Result<FaceReference, ObjError> {
    let fields: Vec<_> = token.split('/').collect();
    if fields.len() > 3 || fields.first().is_none_or(|field| field.is_empty()) {
        return Err(ObjError::new(line, format!("invalid face vertex {token}")));
    }
    let parse_index = |field: &str, name: &str| {
        let index = field
            .parse::<isize>()
            .map_err(|_| ObjError::new(line, format!("invalid {name} index {field} in {token}")))?;
        if index == 0 {
            return Err(ObjError::new(line, "OBJ indices cannot be zero"));
        }
        Ok(index)
    };
    let position = parse_index(fields[0], "position")?;
    if fields.len() == 2 && fields[1].is_empty() {
        return Err(ObjError::new(line, format!("invalid face vertex {token}")));
    }
    if fields.len() == 3 && fields[2].is_empty() {
        return Err(ObjError::new(line, format!("invalid face vertex {token}")));
    }
    let texcoord = match fields.get(1).copied() {
        Some("") => None,
        Some(field) => Some(parse_index(field, "texcoord")?),
        None => None,
    };
    let normal = match fields.get(2).copied() {
        Some("") => None,
        Some(field) => Some(parse_index(field, "normal")?),
        None => None,
    };
    Ok(FaceReference {
        position,
        texcoord,
        normal,
    })
}

fn resolve_index(index: isize, length: usize, line: usize, kind: &str) -> Result<usize, ObjError> {
    let resolved = if index > 0 {
        usize::try_from(index - 1).ok()
    } else {
        usize::try_from(length as isize + index).ok()
    };
    resolved
        .filter(|&index| index < length)
        .ok_or_else(|| ObjError::new(line, format!("{kind} index {index} is out of range")))
}

fn rest_is_empty<'a>(mut words: impl Iterator<Item = &'a str>) -> bool {
    words.next().is_none()
}

fn finish_material_group(
    groups: &mut Vec<MaterialGroup>,
    material_name: Option<String>,
    start: usize,
    end: usize,
) {
    if let Some(material_name) = material_name
        && start < end
    {
        groups.push(MaterialGroup::new(material_name, start..end));
    }
}

fn safe_obj_asset_path(
    asset_root: &Path,
    relative_path: &Path,
) -> Result<std::path::PathBuf, String> {
    if relative_path.is_absolute() {
        return Err("absolute mtllib paths are not allowed".to_string());
    }
    let root = fs::canonicalize(asset_root)
        .map_err(|error| format!("asset root {}: {error}", asset_root.display()))?;
    let candidate = asset_root.join(relative_path);
    let canonical = fs::canonicalize(&candidate)
        .map_err(|error| format!("mtllib {}: {error}", candidate.display()))?;
    if !canonical.starts_with(&root) {
        return Err(format!(
            "mtllib path escapes asset root: {}",
            relative_path.display()
        ));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_face_forms_and_triangulates() {
        let mesh = Mesh::parse(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1 4/4/1\nf 1//1 2//1 3//1\nf 1/1 2/2 3/3\nf 1 2 3\n",
        )
        .unwrap();
        assert_eq!(mesh.indices().len(), 5);
        assert_eq!(mesh.vertex(0).unwrap().position(), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(
            mesh.vertex(0).unwrap().texcoord(),
            Some(Vec2::new(0.0, 0.0))
        );
        assert_eq!(
            mesh.vertex(0).unwrap().normal(),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );
    }

    #[test]
    fn supports_negative_indices() {
        let mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n").unwrap();
        assert_eq!(mesh.indices(), &[[0, 1, 2]]);
    }

    #[test]
    fn resolves_indices_at_the_face_line() {
        let mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 2 -1\nv 0 0 1\n").unwrap();
        assert_eq!(
            mesh.indices(),
            vec![[0, 1, 2]],
            "the face uses the first three positions"
        );
        assert_eq!(
            mesh.vertices()
                .iter()
                .map(MeshVertex::position)
                .collect::<Vec<_>>(),
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ]
        );
    }

    #[test]
    fn reports_malformed_input_with_line_context() {
        let error = Mesh::parse("v 0 0 0\nf 1 2 nope\n").unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.to_string().contains("line 2"));
        assert!(error.to_string().contains("position index"));
    }

    #[test]
    fn rejects_corrupt_index() {
        let error = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n").unwrap_err();
        assert!(error.to_string().contains("position index 4"));
    }

    #[test]
    fn rejects_zero_and_incomplete_attribute_indices() {
        assert!(Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/0 2/1 3/1\n").is_err());
        assert!(Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/ 2/1 3/1\n").is_err());
        assert!(Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1// 2//1 3//1\n").is_err());
    }

    #[test]
    fn records_contiguous_usemtl_triangle_ranges() {
        let mesh = Mesh::parse(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n\
             v 0 0 1\nv 1 0 1\nv 0 1 1\n\
             usemtl first\nf 1 2 3\n\
             usemtl second\nf 4 5 6\n",
        )
        .unwrap();
        assert_eq!(mesh.material_groups().len(), 2);
        assert_eq!(mesh.material_groups()[0].material_name(), "first");
        assert_eq!(mesh.material_groups()[0].triangle_range(), 0..1);
        assert_eq!(mesh.material_groups()[1].material_name(), "second");
        assert_eq!(mesh.material_groups()[1].triangle_range(), 1..2);
    }

    #[test]
    fn index_mutation_clears_material_ranges() {
        let mut mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl first\nf 1 2 3\n").unwrap();
        assert_eq!(mesh.material_groups().len(), 1);
        mesh.set_indices(Vec::new());
        assert!(mesh.material_groups().is_empty());
    }

    #[test]
    fn submesh_preserves_parent_global_normals_and_tangents() {
        let mesh = Mesh::parse(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n\
             vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\n\
             f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n",
        )
        .unwrap();
        let parent_normal = mesh.vertex(0).unwrap().normal().unwrap();
        assert!((parent_normal.x - 0.70710677).abs() < 1e-5);
        assert!((parent_normal.z - 0.70710677).abs() < 1e-5);
        let group = mesh.submesh(0..1).unwrap();
        assert_eq!(group.vertex(0).unwrap().normal(), Some(parent_normal));
        assert_eq!(
            group.vertex(0).unwrap().tangent(),
            mesh.vertex(0).unwrap().tangent()
        );
    }

    #[test]
    fn loads_fixture_mtl_and_maps_with_obj() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/multi_material.obj");
        let asset = Mesh::load_with_materials(path).unwrap();
        assert_eq!(asset.mesh.material_groups().len(), 2);
        assert_eq!(
            asset
                .materials
                .get("checker")
                .unwrap()
                .albedo_texture()
                .unwrap()
                .color_space(),
            crate::image::ColorSpace::Srgb
        );
    }

    #[test]
    fn generates_area_weighted_normals_when_obj_has_none() {
        let mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        assert!(
            mesh.vertices()
                .iter()
                .all(|vertex| vertex.normal().is_some())
        );
        assert!(
            mesh.vertices()
                .iter()
                .all(|vertex| vertex.normal() == Some(Vec3::new(0.0, 0.0, 1.0)))
        );
    }

    #[test]
    fn area_weighted_normals_keep_unequal_face_areas() {
        let mesh = Mesh::parse("v 0 0 0\nv 4 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\n").unwrap();
        let normal = mesh.vertex(0).unwrap().normal().expect("generated normal");
        // The unnormalized face crosses are (0, 0, 4) and (1, 0, 0).
        // Their sum is (1, 0, 4), which normalizes by sqrt(17).
        let length = 17.0_f32.sqrt();
        assert!((normal.x - 1.0 / length).abs() < 1e-5);
        assert!(normal.y.abs() < 1e-5);
        assert!((normal.z - 4.0 / length).abs() < 1e-5);
    }

    #[test]
    fn generates_known_quad_tangents_with_handedness() {
        let mesh = Mesh::parse(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
             vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
             f 1/1 2/2 3/3 4/4\n",
        )
        .unwrap();
        for vertex in mesh.vertices() {
            assert_eq!(
                vertex.tangent(),
                Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
                "tangent should follow +u and reconstruct +bitangent"
            );
        }
    }

    #[test]
    fn mesh_without_texcoords_has_no_tangents_for_normal_map_fallback() {
        let mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        assert!(
            mesh.vertices()
                .iter()
                .all(|vertex| vertex.tangent().is_none())
        );
    }

    #[test]
    fn tangent_mutators_rebuild_the_cached_frame_immediately() {
        let mut mesh = Mesh::parse(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
             vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
             f 1/1 2/2 3/3 4/4\n",
        )
        .unwrap();
        assert_eq!(
            mesh.vertex(0).unwrap().tangent(),
            Some(Vec4::new(1.0, 0.0, 0.0, 1.0))
        );
        assert!(mesh.set_vertex_texcoord(1, Some(Vec2::new(0.0, 1.0))));
        assert_eq!(
            mesh.vertex(0).unwrap().tangent(),
            Some(Vec4::new(
                1.0 / 2.0_f32.sqrt(),
                1.0 / 2.0_f32.sqrt(),
                0.0,
                1.0,
            ))
        );
        assert!(mesh.set_vertex_texcoord(3, Some(Vec2::new(1.0, 0.0))));
        assert_eq!(
            mesh.vertex(0).unwrap().tangent(),
            Some(Vec4::new(0.0, 1.0, 0.0, -1.0))
        );

        let mut derived = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        assert!(derived.set_vertex_position(2, Vec3::new(0.0, 1.0, 1.0)));
        let expected_normal = Vec3::new(0.0, -1.0, 1.0).normalize();
        let normal = derived.vertex(0).unwrap().normal().unwrap();
        assert!((normal.y - expected_normal.y).abs() < 1e-6);
        assert!((normal.z - expected_normal.z).abs() < 1e-6);

        let mut normal_mesh = Mesh::parse(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
             vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
             f 1/1 2/2 3/3 4/4\n",
        )
        .unwrap();
        let changed_normal = Vec3::new(1.0, 0.0, 1.0).normalize();
        assert!(normal_mesh.set_vertex_normal(0, Some(changed_normal)));
        let changed_tangent = normal_mesh.vertex(0).unwrap().tangent().unwrap();
        let tangent = Vec3::new(changed_tangent.x, changed_tangent.y, changed_tangent.z);
        assert!(tangent.dot(changed_normal).abs() < 1e-6);
    }

    #[test]
    fn position_and_index_mutators_rebuild_aabb_immediately() {
        let mut mesh = Mesh::new(
            vec![
                MeshVertex::new(Vec3::new(-1.0, 0.0, 0.0), None, None),
                MeshVertex::new(Vec3::new(1.0, 2.0, 3.0), None, None),
                MeshVertex::new(Vec3::new(0.0, -2.0, -3.0), None, None),
            ],
            vec![[0, 1, 2]],
        );
        assert_eq!(mesh.bounds().min(), Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(mesh.bounds().max(), Vec3::new(1.0, 2.0, 3.0));
        assert!(mesh.set_vertex_position(0, Vec3::new(4.0, 5.0, 6.0)));
        assert_eq!(mesh.bounds().min(), Vec3::new(0.0, -2.0, -3.0));
        assert_eq!(mesh.bounds().max(), Vec3::new(4.0, 5.0, 6.0));
        mesh.set_indices(vec![[0, 1, 1]]);
        assert_eq!(mesh.bounds().min(), Vec3::new(0.0, -2.0, -3.0));
        assert_eq!(mesh.bounds().max(), Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn set_indices_rebuilds_derived_frame_immediately() {
        let vertices = vec![
            MeshVertex::new(Vec3::new(0.0, 0.0, 0.0), Some(Vec2::new(0.0, 0.0)), None),
            MeshVertex::new(Vec3::new(1.0, 0.0, 0.0), Some(Vec2::new(1.0, 0.0)), None),
            MeshVertex::new(Vec3::new(0.0, 1.0, 0.0), Some(Vec2::new(0.0, 1.0)), None),
            MeshVertex::new(Vec3::new(0.0, 0.0, 1.0), Some(Vec2::new(1.0, 0.0)), None),
        ];
        let mut mesh = Mesh::new(vertices, vec![[0, 1, 2]]);
        assert_eq!(
            mesh.vertex(0).unwrap().normal(),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );
        assert_eq!(
            mesh.vertex(0).unwrap().tangent(),
            Some(Vec4::new(1.0, 0.0, 0.0, 1.0))
        );

        mesh.set_indices(vec![[0, 2, 3]]);

        assert_eq!(mesh.indices(), &[[0, 2, 3]]);
        assert_eq!(
            mesh.vertex(0).unwrap().normal(),
            Some(Vec3::new(1.0, 0.0, 0.0))
        );
        assert_eq!(
            mesh.vertex(0).unwrap().tangent(),
            Some(Vec4::new(0.0, 0.0, 1.0, -1.0))
        );
    }

    #[test]
    fn equal_area_faces_use_area_only_tangent_weights() {
        let vertices = vec![
            MeshVertex::new(
                Vec3::new(0.0, 0.0, 0.0),
                Some(Vec2::new(0.0, 0.0)),
                Some(Vec3::new(0.0, 0.0, 1.0)),
            ),
            MeshVertex::new(
                Vec3::new(1.0, 0.0, 0.0),
                Some(Vec2::new(1.0, 0.0)),
                Some(Vec3::new(0.0, 0.0, 1.0)),
            ),
            MeshVertex::new(
                Vec3::new(3.0_f32.sqrt(), 1.0, 0.0),
                Some(Vec2::new(0.0, 1.0)),
                Some(Vec3::new(0.0, 0.0, 1.0)),
            ),
            MeshVertex::new(
                Vec3::new(0.0, 1.0, 0.0),
                Some(Vec2::new(1.0, 0.0)),
                Some(Vec3::new(0.0, 0.0, 1.0)),
            ),
            MeshVertex::new(
                Vec3::new(-1.0, 0.0, 0.0),
                Some(Vec2::new(0.0, 1.0)),
                Some(Vec3::new(0.0, 0.0, 1.0)),
            ),
        ];
        let mesh = Mesh::new(vertices, vec![[0, 1, 2], [0, 3, 4]]);
        let tangent = mesh.vertex(0).unwrap().tangent().unwrap();
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!((tangent.x - expected).abs() < 1e-5);
        assert!((tangent.y - expected).abs() < 1e-5);
        assert_eq!(tangent.w, 1.0);
    }

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
                (vertex.position().length() - 1.0).abs() < 0.08,
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
            None,
        ));
    }
}
