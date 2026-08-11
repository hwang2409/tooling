//! Wavefront OBJ loading for positions, texture coordinates, and normals.
//!
//! Faces are triangulated with a fan. OBJ uses one index for each attribute
//! stream, so the loader expands each unique (v, vt, vn) tuple into one
//! renderer vertex.

use crate::math::{Vec2, Vec3};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVertex {
    pub position: Vec3,
    pub texcoord: Option<Vec2>,
    pub normal: Option<Vec3>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub triangles: Vec<[usize; 3]>,
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
                            mesh.vertices.push(MeshVertex {
                                position: positions[key.0],
                                texcoord: key.1.map(|index| texcoords[index]),
                                normal: key.2.map(|index| normals[index]),
                            });
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

        generate_missing_normals(&mut mesh, &vertex_position_indices, positions.len());
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
}

/// Generates smooth normals for missing `vn` records.
///
/// Each face contributes its unnormalized cross product to each corner. Its
/// magnitude is twice the face area, so this produces area-weighted normals.
fn generate_missing_normals(
    mesh: &mut Mesh,
    vertex_position_indices: &[usize],
    position_count: usize,
) {
    let mut sums = vec![Vec3::ZERO; position_count];
    for &[a, b, c] in &mesh.triangles {
        let Some(position_a) = mesh.vertices.get(a).map(|vertex| vertex.position) else {
            continue;
        };
        let Some(position_b) = mesh.vertices.get(b).map(|vertex| vertex.position) else {
            continue;
        };
        let Some(position_c) = mesh.vertices.get(c).map(|vertex| vertex.position) else {
            continue;
        };
        let face_normal = (position_b - position_a).cross(position_c - position_a);
        for index in [a, b, c] {
            if let Some(&position_index) = vertex_position_indices.get(index) {
                if let Some(sum) = sums.get_mut(position_index) {
                    *sum = *sum + face_normal;
                }
            }
        }
    }

    for (index, vertex) in mesh.vertices.iter_mut().enumerate() {
        if vertex.normal.is_some() {
            continue;
        }
        let normal = vertex_position_indices
            .get(index)
            .and_then(|&position_index| sums.get(position_index))
            .copied()
            .filter(|normal| normal.length() > 0.0)
            .map(Vec3::normalize)
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        vertex.normal = Some(normal);
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
        assert_eq!(mesh.triangles.len(), 5);
        assert_eq!(mesh.vertices[0].position, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(mesh.vertices[0].texcoord, Some(Vec2::new(0.0, 0.0)));
        assert_eq!(mesh.vertices[0].normal, Some(Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn supports_negative_indices() {
        let mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n").unwrap();
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
    }

    #[test]
    fn resolves_indices_at_the_face_line() {
        let mesh = Mesh::parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 2 -1\nv 0 0 1\n").unwrap();
        assert_eq!(
            mesh.triangles,
            vec![[0, 1, 2]],
            "the face uses the first three positions"
        );
        assert_eq!(
            mesh.vertices
                .iter()
                .map(|vertex| vertex.position)
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
        assert!(mesh.vertices.iter().all(|vertex| vertex.normal.is_some()));
        assert!(
            mesh.vertices
                .iter()
                .all(|vertex| vertex.normal == Some(Vec3::new(0.0, 0.0, 1.0)))
        );
    }

    #[test]
    fn area_weighted_normals_keep_unequal_face_areas() {
        let mesh = Mesh::parse("v 0 0 0\nv 4 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 3 4\n").unwrap();
        let normal = mesh.vertices[0].normal.expect("generated normal");
        // The unnormalized face crosses are (0, 0, 4) and (1, 0, 0).
        // Their sum is (1, 0, 4), which normalizes by sqrt(17).
        let length = 17.0_f32.sqrt();
        assert!((normal.x - 1.0 / length).abs() < 1e-5);
        assert!(normal.y.abs() < 1e-5);
        assert!((normal.z - 4.0 / length).abs() < 1e-5);
    }
}
