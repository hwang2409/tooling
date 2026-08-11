//! Wavefront OBJ loading for positions, texture coordinates, and normals.
//!
//! Faces are triangulated with a fan. OBJ uses one index for each attribute
//! stream, so the loader expands each unique (v, vt, vn) tuple into one
//! renderer vertex.

use crate::math::{Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::fs;
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
}

impl Mesh {
    pub fn parse(source: &str) -> Result<Self, ObjError> {
        let mut positions = Vec::new();
        let mut texcoords = Vec::new();
        let mut normals = Vec::new();
        let mut mesh = Self::default();
        let mut vertex_map = HashMap::new();
        let mut vertex_position_indices = Vec::new();

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
                // These records do not affect the renderer mesh.
                "o" | "g" | "s" | "usemtl" | "mtllib" => {}
                _ => {}
            }
        }

        mesh.position_indices = vertex_position_indices;
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

    pub fn indices(&self) -> &[[usize; 3]] {
        &self.triangles
    }

    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
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
        self.rebuild_derived_attributes();
    }

    fn rebuild_derived_attributes(&mut self) {
        generate_missing_normals(self);
        self.generate_tangents();
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
        let rotated_uvs = [
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
        ];
        for (index, uv) in rotated_uvs.into_iter().enumerate() {
            assert!(mesh.set_vertex_texcoord(index, Some(uv)));
        }
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
}
