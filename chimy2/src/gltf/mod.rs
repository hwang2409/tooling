//! glTF 2.0 loading, animation sampling, and deterministic CPU skinning.
//!
//! Supported features are documented by SUPPORTED_SUBSET. Unsupported
//! features return an error. The loader uses no serialization crate because
//! glTF is parsed by the local JSON parser.

use crate::image::{ColorSpace, Texture};
use crate::json::{self, Value};
use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const SUPPORTED_SUBSET: &str = "glTF JSON (.gltf), external .bin buffers and base64 binary data URIs; TRIANGLES primitives with POSITION, optional NORMAL, TEXCOORD_0, JOINTS_0, WEIGHTS_0, and indices; SCALAR/VEC2/VEC3/VEC4/MAT4 accessors with component types 5120-5126, normalized values, and byteStride; node TRS or matrix hierarchy; PBR baseColorFactor/baseColorTexture and normalTexture; skins with inverseBindMatrices; translation, rotation, and scale animations with LINEAR and STEP. Unsupported primitive modes, sparse accessors, morph targets, cubic animation, GLB, and unsupported image formats return errors.";

const WEIGHT_TOLERANCE: f32 = 1.0e-5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GltfError {
    pub offset: usize,
    pub message: String,
}

impl GltfError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            offset: 0,
            message: message.into(),
        }
    }

    fn at(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl Display for GltfError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if self.offset == 0 {
            write!(formatter, "{}", self.message)
        } else {
            write!(formatter, "byte {}: {}", self.offset, self.message)
        }
    }
}

impl std::error::Error for GltfError {}

impl From<json::Error> for GltfError {
    fn from(error: json::Error) -> Self {
        Self::at(error.offset, error.message)
    }
}

#[derive(Clone, Debug)]
pub struct GltfAsset {
    pub meshes: Vec<GltfMesh>,
    pub nodes: Vec<GltfNode>,
    pub scenes: Vec<GltfScene>,
    pub skins: Vec<GltfSkin>,
    pub animations: Vec<GltfAnimation>,
    pub materials: Vec<GltfMaterial>,
    pub default_scene: usize,
}

#[derive(Clone, Debug)]
pub struct GltfMesh {
    pub name: String,
    pub primitives: Vec<GltfPrimitive>,
}

#[derive(Clone, Debug)]
pub struct GltfPrimitive {
    pub mesh: Mesh,
    pub material: Option<usize>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
}

#[derive(Clone, Debug)]
pub struct GltfNode {
    pub name: String,
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub matrix: Option<Mat4>,
    pub translation: Vec3,
    pub rotation: [f32; 4],
    pub scale: Vec3,
}

#[derive(Clone, Debug)]
pub struct GltfScene {
    pub name: String,
    pub nodes: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct GltfSkin {
    pub name: String,
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
    pub skeleton: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct GltfAnimation {
    pub name: String,
    pub samplers: Vec<GltfAnimationSampler>,
    pub channels: Vec<GltfAnimationChannel>,
    pub duration: f32,
}

#[derive(Clone, Debug)]
pub struct GltfAnimationSampler {
    pub input: Vec<f32>,
    pub output: Vec<[f32; 4]>,
    pub output_components: usize,
    pub interpolation: Interpolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationPath {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Debug)]
pub struct GltfAnimationChannel {
    pub sampler: usize,
    pub node: usize,
    pub path: AnimationPath,
}

#[derive(Clone, Debug)]
pub struct GltfMaterial {
    pub name: String,
    pub base_color_factor: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub albedo_texture: Option<Texture>,
    pub normal_map_texture: Option<Texture>,
}

impl GltfMaterial {
    /// Maps the parked metallic and roughness values to the current shader.
    /// Diffuse uses baseColorFactor.rgb. Specular is 4% for dielectrics and
    /// approaches white for metallic materials. Roughness maps to a
    /// Blinn-Phong exponent in the range 1..129.
    pub fn blinn_phong_parameters(&self) -> (Vec3, Vec3, f32, f32) {
        let metallic = self.metallic_factor.clamp(0.0, 1.0);
        let roughness = self.roughness_factor.clamp(0.0, 1.0);
        let diffuse = Vec3::new(
            self.base_color_factor.x,
            self.base_color_factor.y,
            self.base_color_factor.z,
        ) * (1.0 - metallic);
        let specular = Vec3::new(
            0.04 + 0.96 * metallic,
            0.04 + 0.96 * metallic,
            0.04 + 0.96 * metallic,
        );
        let shininess = 1.0 + (1.0 - roughness) * 128.0;
        (
            diffuse,
            specular,
            shininess,
            self.base_color_factor.w.clamp(0.0, 1.0),
        )
    }
}

#[derive(Clone, Debug)]
pub struct GltfDraw {
    pub mesh: Mesh,
    pub model: Mat4,
    pub material: Option<usize>,
}

impl GltfAsset {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GltfError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .map_err(|error| GltfError::new(format!("{}: {error}", path.display())))?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        Self::from_str(&source, root)
    }

    pub fn from_str(source: &str, asset_root: impl AsRef<Path>) -> Result<Self, GltfError> {
        let root = asset_root.as_ref().to_path_buf();
        let document = json::parse(source)?;
        let object = as_object(&document, "root")?;
        let asset = get_object(object, "asset")?;
        let version = get_string(asset, "version")?;
        if version != "2.0" {
            return Err(GltfError::new(format!(
                "unsupported glTF version {version}"
            )));
        }

        let buffers = load_buffers(get_array(object, "buffers")?, &root)?;
        let views = parse_views(get_array(object, "bufferViews")?)?;
        let accessors = parse_accessors(get_array(object, "accessors")?)?;
        let images = load_images(object, &root)?;
        let textures = parse_textures(object)?;
        let materials = parse_materials(object, &images, &textures)?;
        let meshes = parse_meshes(object, &buffers, &views, &accessors)?;
        let nodes = parse_nodes(object)?;
        validate_nodes(&nodes)?;
        let skins = parse_skins(object, &buffers, &views, &accessors, nodes.len())?;
        let animations = parse_animations(object, &buffers, &views, &accessors, nodes.len())?;
        let scenes = parse_scenes(object, nodes.len())?;
        let default_scene = get_optional_usize(object, "scene")?.unwrap_or(0);
        if !scenes.is_empty() && default_scene >= scenes.len() {
            return Err(GltfError::new("default scene index is out of range"));
        }
        Ok(Self {
            meshes,
            nodes,
            scenes,
            skins,
            animations,
            materials,
            default_scene,
        })
    }

    pub fn node_world_transforms(
        &self,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Vec<Mat4>, GltfError> {
        let locals = self.local_matrices(animation, time)?;
        let mut worlds = vec![Mat4::IDENTITY; self.nodes.len()];
        let mut state = vec![0u8; self.nodes.len()];
        for index in 0..self.nodes.len() {
            if state[index] == 0 {
                self.world(index, &locals, &mut worlds, &mut state, 0)?;
            }
        }
        Ok(worlds)
    }

    pub fn sample_animation(
        &self,
        animation: usize,
        time: f32,
    ) -> Result<Vec<NodeTransform>, GltfError> {
        self.sample_transforms(Some(animation), time)
    }

    pub fn pose_mesh(
        &self,
        mesh_index: usize,
        primitive_index: usize,
        node_index: usize,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Mesh, GltfError> {
        let primitive = self
            .meshes
            .get(mesh_index)
            .and_then(|mesh| mesh.primitives.get(primitive_index))
            .ok_or_else(|| GltfError::new("mesh primitive index is out of range"))?;
        let node = self
            .nodes
            .get(node_index)
            .ok_or_else(|| GltfError::new("node index is out of range"))?;
        let Some(skin_index) = node.skin else {
            return Ok(primitive.mesh.clone());
        };
        let worlds = self.node_world_transforms(animation, time)?;
        let skin = self
            .skins
            .get(skin_index)
            .ok_or_else(|| GltfError::new("skin index is out of range"))?;
        if primitive.joints.len() != primitive.mesh.vertices().len()
            || primitive.weights.len() != primitive.mesh.vertices().len()
        {
            return Err(GltfError::new("skin attributes do not match mesh vertices"));
        }
        let mut joint_matrices = Vec::with_capacity(skin.joints.len());
        let mut normal_matrices = Vec::with_capacity(skin.joints.len());
        for (joint_index, &joint) in skin.joints.iter().enumerate() {
            let inverse_bind = skin
                .inverse_bind_matrices
                .get(joint_index)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let matrix = worlds
                .get(joint)
                .copied()
                .ok_or_else(|| GltfError::new("skin joint index is out of range"))?
                * inverse_bind;
            let normal = matrix
                .normal_matrix()
                .ok_or_else(|| GltfError::new("singular joint matrix cannot transform normals"))?;
            joint_matrices.push(matrix);
            normal_matrices.push(normal);
        }
        let mut vertices = Vec::with_capacity(primitive.mesh.vertices().len());
        for (index, source) in primitive.mesh.vertices().iter().enumerate() {
            let weights = normalize_weights(primitive.weights[index]);
            let mut position = Vec3::ZERO;
            let mut normal = Vec3::ZERO;
            let local_position = source.position();
            let local_normal = source.normal().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            for (influence, &weight) in weights.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let joint = usize::from(primitive.joints[index][influence]);
                let matrix = joint_matrices
                    .get(joint)
                    .ok_or_else(|| GltfError::new("vertex joint index is out of range"))?;
                let transformed =
                    *matrix * Vec4::new(local_position.x, local_position.y, local_position.z, 1.0);
                position =
                    position + Vec3::new(transformed.x, transformed.y, transformed.z) * weight;
                normal = normal + normal_matrices[joint] * local_normal * weight;
            }
            if weights.iter().copied().sum::<f32>() == 0.0 {
                position = local_position;
                normal = local_normal;
            }
            vertices.push(MeshVertex::new(
                position,
                source.texcoord(),
                Some(normal.normalize()),
            ));
        }
        Ok(Mesh::new(vertices, primitive.mesh.indices().to_vec()))
    }

    pub fn scene_draws(
        &self,
        scene_index: usize,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Vec<GltfDraw>, GltfError> {
        let scene = self
            .scenes
            .get(scene_index)
            .ok_or_else(|| GltfError::new("scene index is out of range"))?;
        let worlds = self.node_world_transforms(animation, time)?;
        let mut draws = Vec::new();
        for &root in &scene.nodes {
            self.collect_draws(root, &worlds, animation, time, &mut draws)?;
        }
        Ok(draws)
    }

    fn collect_draws(
        &self,
        node_index: usize,
        worlds: &[Mat4],
        animation: Option<usize>,
        time: f32,
        draws: &mut Vec<GltfDraw>,
    ) -> Result<(), GltfError> {
        let node = self
            .nodes
            .get(node_index)
            .ok_or_else(|| GltfError::new("node index is out of range"))?;
        if let Some(mesh_index) = node.mesh {
            let mesh = self
                .meshes
                .get(mesh_index)
                .ok_or_else(|| GltfError::new("node mesh index is out of range"))?;
            for primitive_index in 0..mesh.primitives.len() {
                let posed = if node.skin.is_some() {
                    self.pose_mesh(mesh_index, primitive_index, node_index, animation, time)?
                } else {
                    mesh.primitives[primitive_index].mesh.clone()
                };
                draws.push(GltfDraw {
                    mesh: posed,
                    model: if node.skin.is_some() {
                        Mat4::IDENTITY
                    } else {
                        worlds[node_index]
                    },
                    material: mesh.primitives[primitive_index].material,
                });
            }
        }
        for &child in &node.children {
            self.collect_draws(child, worlds, animation, time, draws)?;
        }
        Ok(())
    }

    fn sample_transforms(
        &self,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Vec<NodeTransform>, GltfError> {
        let mut transforms: Vec<_> = self.nodes.iter().map(NodeTransform::from_node).collect();
        if let Some(animation_index) = animation {
            let animation = self
                .animations
                .get(animation_index)
                .ok_or_else(|| GltfError::new("animation index is out of range"))?;
            for channel in &animation.channels {
                let node = transforms
                    .get_mut(channel.node)
                    .ok_or_else(|| GltfError::new("animation node index is out of range"))?;
                let sampler = animation
                    .samplers
                    .get(channel.sampler)
                    .ok_or_else(|| GltfError::new("animation sampler index is out of range"))?;
                let value = sampler.sample(time)?;
                match channel.path {
                    AnimationPath::Translation => {
                        node.translation = Vec3::new(value[0], value[1], value[2])
                    }
                    AnimationPath::Rotation => node.rotation = value,
                    AnimationPath::Scale => node.scale = Vec3::new(value[0], value[1], value[2]),
                }
            }
        }
        Ok(transforms)
    }

    fn local_matrices(&self, animation: Option<usize>, time: f32) -> Result<Vec<Mat4>, GltfError> {
        let transforms = self.sample_transforms(animation, time)?;
        if let Some(animation_index) = animation {
            let animation = self
                .animations
                .get(animation_index)
                .ok_or_else(|| GltfError::new("animation index is out of range"))?;
            if animation
                .channels
                .iter()
                .any(|channel| self.nodes[channel.node].matrix.is_some())
            {
                return Err(GltfError::new(
                    "animation channels targeting matrix nodes are unsupported",
                ));
            }
        }
        Ok(self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| node.matrix.unwrap_or_else(|| transforms[index].matrix()))
            .collect())
    }

    fn world(
        &self,
        index: usize,
        locals: &[Mat4],
        worlds: &mut [Mat4],
        state: &mut [u8],
        depth: usize,
    ) -> Result<(), GltfError> {
        if depth > self.nodes.len() {
            return Err(GltfError::new("node hierarchy contains a cycle"));
        }
        if state[index] == 1 {
            return Err(GltfError::new("node hierarchy contains a cycle"));
        }
        if state[index] == 2 {
            return Ok(());
        }
        state[index] = 1;
        let children = self.nodes[index].children.clone();
        worlds[index] = locals[index];
        for child in children {
            self.world(child, locals, worlds, state, depth + 1)?;
            worlds[child] = worlds[index] * worlds[child];
        }
        state[index] = 2;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeTransform {
    pub translation: Vec3,
    pub rotation: [f32; 4],
    pub scale: Vec3,
}

impl NodeTransform {
    fn from_node(node: &GltfNode) -> Self {
        Self {
            translation: node.translation,
            rotation: node.rotation,
            scale: node.scale,
        }
    }
    fn matrix(self) -> Mat4 {
        if self.rotation == [0.0, 0.0, 0.0, 1.0] {
            Mat4::translate(self.translation) * Mat4::scale(self.scale)
        } else {
            Mat4::translate(self.translation) * quat_matrix(self.rotation) * Mat4::scale(self.scale)
        }
    }
}

impl GltfAnimationSampler {
    pub fn sample(&self, time: f32) -> Result<[f32; 4], GltfError> {
        if self.input.is_empty() || self.output.is_empty() {
            return Err(GltfError::new("animation sampler has no keyframes"));
        }
        let last = self.input.len() - 1;
        if time <= self.input[0] {
            return Ok(self.output[0]);
        }
        if time >= self.input[last] {
            return Ok(self.output[last]);
        }
        let index = self
            .input
            .windows(2)
            .position(|window| time < window[1])
            .unwrap_or(last - 1);
        let factor = ((time - self.input[index]) / (self.input[index + 1] - self.input[index]))
            .clamp(0.0, 1.0);
        if self.interpolation == Interpolation::Step {
            return Ok(self.output[index]);
        }
        if self.output_components == 4 {
            Ok(slerp(self.output[index], self.output[index + 1], factor))
        } else {
            Ok(lerp4(self.output[index], self.output[index + 1], factor))
        }
    }
}

pub fn normalize_weights(weights: [f32; 4]) -> [f32; 4] {
    let sum = weights.into_iter().map(|value| value.max(0.0)).sum::<f32>();
    if sum <= WEIGHT_TOLERANCE {
        [0.0; 4]
    } else {
        weights.map(|value| value.max(0.0) / sum)
    }
}

pub fn slerp(a: [f32; 4], b: [f32; 4], factor: f32) -> [f32; 4] {
    let mut b = b;
    let mut dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    if dot < 0.0 {
        b = b.map(|value| -value);
        dot = -dot;
    }
    if dot > 0.9995 {
        return normalize_quat(lerp4(a, b, factor));
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let first = ((1.0 - factor) * theta).sin() / sin_theta;
    let second = (factor * theta).sin() / sin_theta;
    normalize_quat([
        a[0] * first + b[0] * second,
        a[1] * first + b[1] * second,
        a[2] * first + b[2] * second,
        a[3] * first + b[3] * second,
    ])
}

fn lerp4(a: [f32; 4], b: [f32; 4], factor: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * factor,
        a[1] + (b[1] - a[1]) * factor,
        a[2] + (b[2] - a[2]) * factor,
        a[3] + (b[3] - a[3]) * factor,
    ]
}
fn normalize_quat(value: [f32; 4]) -> [f32; 4] {
    let length = value.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length == 0.0 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        value.map(|v| v / length)
    }
}
fn quat_matrix(value: [f32; 4]) -> Mat4 {
    let q = normalize_quat(value);
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    Mat4::new([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        0.0,
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        0.0,
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}

#[derive(Clone, Debug)]
struct BufferView {
    buffer: usize,
    offset: usize,
    length: usize,
    stride: Option<usize>,
}
#[derive(Clone, Debug)]
struct Accessor {
    view: Option<usize>,
    offset: usize,
    count: usize,
    component_type: u32,
    kind: String,
    normalized: bool,
}
#[derive(Clone, Copy, Debug)]
struct TextureRef {
    source: usize,
}

fn load_buffers(values: &[Value], root: &Path) -> Result<Vec<Vec<u8>>, GltfError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let object = as_object(value, "buffer")?;
            let expected = get_usize(object, "byteLength")?;
            let data = if let Some(uri) = get_optional_string(object, "uri")? {
                if let Some(encoded) = uri.strip_prefix("data:") {
                    let (_, payload) = encoded.split_once(',').ok_or_else(|| {
                        GltfError::new(format!("buffer {index} has an invalid data URI"))
                    })?;
                    decode_base64(payload)?
                } else {
                    let path = safe_asset_path(root, Path::new(uri))?;
                    fs::read(&path).map_err(|error| {
                        GltfError::new(format!("buffer {}: {error}", path.display()))
                    })?
                }
            } else {
                return Err(GltfError::new(
                    "GLB binary buffers are unsupported; buffer needs uri",
                ));
            };
            if data.len() < expected {
                return Err(GltfError::new(format!(
                    "buffer {index} is shorter than byteLength"
                )));
            }
            Ok(data)
        })
        .collect()
}

fn decode_base64(source: &str) -> Result<Vec<u8>, GltfError> {
    let bytes = source.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(GltfError::new("base64 length is not a multiple of four"));
    }
    let mut output = Vec::with_capacity(bytes.len() / 4 * 3);
    for (chunk_index, chunk) in bytes.chunks_exact(4).enumerate() {
        let a = base64_value(chunk[0]).ok_or_else(|| GltfError::new("invalid base64 character"))?;
        let b = base64_value(chunk[1]).ok_or_else(|| GltfError::new("invalid base64 character"))?;
        let c = if chunk[2] == b'=' {
            0
        } else {
            base64_value(chunk[2]).ok_or_else(|| GltfError::new("invalid base64 character"))?
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            base64_value(chunk[3]).ok_or_else(|| GltfError::new("invalid base64 character"))?
        };
        if chunk[2] == b'=' && chunk[3] != b'=' {
            return Err(GltfError::new("invalid base64 padding"));
        }
        let final_chunk = chunk_index + 1 == bytes.len() / 4;
        if !final_chunk && (chunk[2] == b'=' || chunk[3] == b'=') {
            return Err(GltfError::new("base64 padding must be at the end"));
        }
        if chunk[2] == b'=' && (b & 0x0f) != 0 {
            return Err(GltfError::new("non-zero base64 padding bits"));
        }
        if chunk[3] == b'=' && chunk[2] != b'=' && (c & 0x03) != 0 {
            return Err(GltfError::new("non-zero base64 padding bits"));
        }
        output.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            output.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            output.push((c << 6) | d);
        }
    }
    Ok(output)
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn parse_views(values: &[Value]) -> Result<Vec<BufferView>, GltfError> {
    values
        .iter()
        .map(|value| {
            let o = as_object(value, "bufferView")?;
            Ok(BufferView {
                buffer: get_usize(o, "buffer")?,
                offset: get_optional_usize(o, "byteOffset")?.unwrap_or(0),
                length: get_usize(o, "byteLength")?,
                stride: get_optional_usize(o, "byteStride")?,
            })
        })
        .collect()
}
fn parse_accessors(values: &[Value]) -> Result<Vec<Accessor>, GltfError> {
    values
        .iter()
        .map(|value| {
            let o = as_object(value, "accessor")?;
            if o.iter().any(|(key, _)| key == "sparse") {
                return Err(GltfError::new("sparse accessors are unsupported"));
            }
            let component_type = get_u32(o, "componentType")?;
            if !(5120..=5126).contains(&component_type) {
                return Err(GltfError::new("unsupported accessor component type"));
            }
            let kind = get_string(o, "type")?.to_string();
            if !matches!(kind.as_str(), "SCALAR" | "VEC2" | "VEC3" | "VEC4" | "MAT4") {
                return Err(GltfError::new("unsupported accessor type"));
            }
            Ok(Accessor {
                view: get_optional_usize(o, "bufferView")?,
                offset: get_optional_usize(o, "byteOffset")?.unwrap_or(0),
                count: get_usize(o, "count")?,
                component_type,
                kind,
                normalized: get_optional_bool(o, "normalized")?.unwrap_or(false),
            })
        })
        .collect()
}

fn parse_meshes(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
) -> Result<Vec<GltfMesh>, GltfError> {
    let Some(values) = get_optional_array(object, "meshes")? else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|value| {
            let mesh_object = as_object(value, "mesh")?;
            let primitives = get_array(mesh_object, "primitives")?
                .iter()
                .map(|value| {
                    parse_primitive(as_object(value, "primitive")?, buffers, views, accessors)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GltfMesh {
                name: get_optional_string(mesh_object, "name")?
                    .unwrap_or_default()
                    .to_string(),
                primitives,
            })
        })
        .collect()
}

fn parse_primitive(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
) -> Result<GltfPrimitive, GltfError> {
    if get_optional_u32(object, "mode")?.unwrap_or(4) != 4 {
        return Err(GltfError::new("only TRIANGLES primitives are supported"));
    }
    let attributes = get_object(object, "attributes")?;
    let position_accessor = get_u32(attributes, "POSITION")? as usize;
    let positions: Vec<[f32; 4]> =
        read_accessor::<4>(buffers, views, accessors, position_accessor, 3)?;
    let normals: Option<Vec<[f32; 4]>> = get_optional_u32(attributes, "NORMAL")?
        .map(|index| read_accessor::<4>(buffers, views, accessors, index as usize, 3))
        .transpose()?;
    let texcoords: Option<Vec<[f32; 4]>> = get_optional_u32(attributes, "TEXCOORD_0")?
        .map(|index| read_accessor::<4>(buffers, views, accessors, index as usize, 2))
        .transpose()?;
    let joints = get_optional_u32(attributes, "JOINTS_0")?
        .map(|index| read_joints(buffers, views, accessors, index as usize))
        .transpose()?
        .unwrap_or_else(|| vec![[0; 4]; positions.len()]);
    let weights = get_optional_u32(attributes, "WEIGHTS_0")?
        .map(|index| read_accessor::<4>(buffers, views, accessors, index as usize, 4))
        .transpose()?
        .map(|values| {
            values
                .into_iter()
                .map(|value| [value[0], value[1], value[2], value[3]])
                .collect()
        })
        .unwrap_or_else(|| vec![[0.0; 4]; positions.len()]);
    if normals
        .as_ref()
        .is_some_and(|values| values.len() != positions.len())
        || texcoords
            .as_ref()
            .is_some_and(|values| values.len() != positions.len())
        || joints.len() != positions.len()
        || weights.len() != positions.len()
    {
        return Err(GltfError::new(
            "mesh attribute counts do not match POSITION",
        ));
    }
    let indices = if let Some(index) = get_optional_u32(object, "indices")? {
        read_indices(buffers, views, accessors, index as usize)?
    } else {
        (0..positions.len()).collect()
    };
    if indices.len() % 3 != 0 || indices.iter().any(|&index| index >= positions.len()) {
        return Err(GltfError::new("primitive indices are invalid"));
    }
    let vertices = positions
        .iter()
        .enumerate()
        .map(|(index, position)| {
            MeshVertex::new(
                Vec3::new(position[0], position[1], position[2]),
                texcoords
                    .as_ref()
                    .map(|values| Vec2::new(values[index][0], values[index][1])),
                normals
                    .as_ref()
                    .map(|values| Vec3::new(values[index][0], values[index][1], values[index][2])),
            )
        })
        .collect();
    let triangles = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    Ok(GltfPrimitive {
        mesh: Mesh::new(vertices, triangles),
        material: get_optional_u32(object, "material")?.map(|index| index as usize),
        joints,
        weights,
    })
}

fn read_accessor<const N: usize>(
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    index: usize,
    expected_components: usize,
) -> Result<Vec<[f32; N]>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new("accessor index is out of range"))?;
    let components = component_count(&accessor.kind);
    if components != expected_components {
        return Err(GltfError::new("accessor has the wrong component count"));
    }
    let bytes_per_component = component_size(accessor.component_type)?;
    let element_size = bytes_per_component
        .checked_mul(components)
        .ok_or_else(|| GltfError::new("accessor element size overflow"))?;
    let (data, stride, view_offset) = accessor_data(buffers, views, accessor, element_size)?;
    let mut output = Vec::with_capacity(accessor.count);
    for element in 0..accessor.count {
        let start = element
            .checked_mul(stride)
            .and_then(|value| value.checked_add(view_offset))
            .and_then(|value| value.checked_add(accessor.offset))
            .ok_or_else(|| GltfError::new("accessor offset overflow"))?;
        let mut result = [0.0; N];
        for component in 0..components {
            result[component] = read_component(
                &data[start + component * bytes_per_component..],
                accessor.component_type,
                accessor.normalized,
            )?;
        }
        output.push(result);
    }
    Ok(output)
}

fn read_joints(
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    index: usize,
) -> Result<Vec<[u16; 4]>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new("joint accessor index is out of range"))?;
    if component_count(&accessor.kind) != 4 || !matches!(accessor.component_type, 5121 | 5123) {
        return Err(GltfError::new(
            "JOINTS_0 needs UNSIGNED_BYTE or UNSIGNED_SHORT VEC4",
        ));
    }
    let values = read_accessor::<4>(buffers, views, accessors, index, 4)?;
    Ok(values
        .into_iter()
        .map(|value| {
            [
                value[0] as u16,
                value[1] as u16,
                value[2] as u16,
                value[3] as u16,
            ]
        })
        .collect())
}

fn read_indices(
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    index: usize,
) -> Result<Vec<usize>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new("index accessor is out of range"))?;
    if accessor.kind != "SCALAR" || !matches!(accessor.component_type, 5121 | 5123 | 5125) {
        return Err(GltfError::new("indices need unsigned scalar components"));
    }
    Ok(read_accessor::<4>(buffers, views, accessors, index, 1)?
        .into_iter()
        .map(|value| value[0] as usize)
        .collect())
}

fn accessor_data<'a>(
    buffers: &'a [Vec<u8>],
    views: &[BufferView],
    accessor: &Accessor,
    element_size: usize,
) -> Result<(&'a [u8], usize, usize), GltfError> {
    let view_index = accessor
        .view
        .ok_or_else(|| GltfError::new("accessor without bufferView is unsupported"))?;
    let view = views
        .get(view_index)
        .ok_or_else(|| GltfError::new("bufferView index is out of range"))?;
    let data = buffers
        .get(view.buffer)
        .ok_or_else(|| GltfError::new("bufferView buffer index is out of range"))?;
    let end = view
        .offset
        .checked_add(view.length)
        .ok_or_else(|| GltfError::new("bufferView range overflow"))?;
    if end > data.len() {
        return Err(GltfError::new("bufferView exceeds its buffer"));
    }
    let stride = view.stride.unwrap_or(element_size);
    if stride < element_size {
        return Err(GltfError::new(
            "byteStride is smaller than the accessor element",
        ));
    }
    let last = accessor
        .count
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|last| last.checked_add(accessor.offset))
        .and_then(|last| last.checked_add(element_size))
        .ok_or_else(|| GltfError::new("accessor range overflow"))?;
    if last > view.length {
        return Err(GltfError::new(format!(
            "accessor overruns its bufferView: end {last}, length {}",
            view.length
        )));
    }
    Ok((&data[..end], stride, view.offset))
}

fn read_component(data: &[u8], component_type: u32, normalized: bool) -> Result<f32, GltfError> {
    let raw = match component_type {
        5120 => i8::from_le_bytes([data[0]]) as f32,
        5121 => data[0] as f32,
        5122 => i16::from_le_bytes([data[0], data[1]]) as f32,
        5123 => u16::from_le_bytes([data[0], data[1]]) as f32,
        5124 => i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f32,
        5125 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f32,
        5126 => f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
        _ => return Err(GltfError::new("unsupported component type")),
    };
    if !normalized {
        return Ok(raw);
    }
    Ok(match component_type {
        5120 => (raw / 127.0).max(-1.0),
        5121 => raw / 255.0,
        5122 => (raw / 32767.0).max(-1.0),
        5123 => raw / 65535.0,
        5124 => (raw / 2147483647.0).max(-1.0),
        5125 => raw / 4294967295.0,
        5126 => raw,
        _ => raw,
    })
}

fn component_size(component_type: u32) -> Result<usize, GltfError> {
    Ok(match component_type {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5124..=5126 => 4,
        _ => return Err(GltfError::new("unsupported component type")),
    })
}
fn component_count(kind: &str) -> usize {
    match kind {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT4" => 16,
        _ => 0,
    }
}

fn parse_nodes(object: &[(String, Value)]) -> Result<Vec<GltfNode>, GltfError> {
    let values = get_optional_array(object, "nodes")?.unwrap_or(&[]);
    values
        .iter()
        .map(|value| {
            let o = as_object(value, "node")?;
            let matrix = get_optional_f32_array(o, "matrix")?
                .map(|values| {
                    if values.len() != 16 {
                        return Err(GltfError::new("node matrix needs 16 values"));
                    }
                    let mut data = [0.0; 16];
                    data.copy_from_slice(&values);
                    Ok(Mat4::new(data))
                })
                .transpose()?;
            if matrix.is_some()
                && ["translation", "rotation", "scale"]
                    .iter()
                    .any(|key| get(o, key).is_some())
            {
                return Err(GltfError::new("node cannot contain both matrix and TRS"));
            }
            let translation = vec3(
                get_optional_f32_array(o, "translation")?.unwrap_or_else(|| vec![0.0, 0.0, 0.0]),
                "translation",
            )?;
            let rotation = vec4(
                get_optional_f32_array(o, "rotation")?.unwrap_or_else(|| vec![0.0, 0.0, 0.0, 1.0]),
                "rotation",
            )?;
            let scale = vec3(
                get_optional_f32_array(o, "scale")?.unwrap_or_else(|| vec![1.0, 1.0, 1.0]),
                "scale",
            )?;
            Ok(GltfNode {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                children: get_optional_usize_array(o, "children")?.unwrap_or_default(),
                mesh: get_optional_usize(o, "mesh")?,
                skin: get_optional_usize(o, "skin")?,
                matrix,
                translation,
                rotation,
                scale,
            })
        })
        .collect()
}

fn validate_nodes(nodes: &[GltfNode]) -> Result<(), GltfError> {
    for node in nodes {
        for &child in &node.children {
            if child >= nodes.len() {
                return Err(GltfError::new("node child index is out of range"));
            }
        }
    }
    Ok(())
}

fn parse_skins(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    node_count: usize,
) -> Result<Vec<GltfSkin>, GltfError> {
    get_optional_array(object, "skins")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "skin")?;
            let joints = get_usize_array(o, "joints")?;
            if joints.iter().any(|&joint| joint >= node_count) {
                return Err(GltfError::new("skin joint index is out of range"));
            }
            let inverse = if let Some(index) = get_optional_usize(o, "inverseBindMatrices")? {
                let values = read_accessor::<16>(buffers, views, accessors, index, 16)?;
                values.into_iter().map(Mat4::new).collect()
            } else {
                vec![Mat4::IDENTITY; joints.len()]
            };
            if inverse.len() != joints.len() {
                return Err(GltfError::new(
                    "inverse bind matrix count does not match joints",
                ));
            }
            Ok(GltfSkin {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                joints,
                inverse_bind_matrices: inverse,
                skeleton: get_optional_usize(o, "skeleton")?,
            })
        })
        .collect()
}

fn parse_animations(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    node_count: usize,
) -> Result<Vec<GltfAnimation>, GltfError> {
    get_optional_array(object, "animations")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "animation")?;
            let samplers = get_array(o, "samplers")?
                .iter()
                .map(|value| {
                    let s = as_object(value, "animation sampler")?;
                    let input =
                        read_accessor::<4>(buffers, views, accessors, get_usize(s, "input")?, 1)?
                            .into_iter()
                            .map(|v| v[0])
                            .collect::<Vec<_>>();
                    let output_accessor = get_usize(s, "output")?;
                    let accessor = accessors.get(output_accessor).ok_or_else(|| {
                        GltfError::new("animation output accessor is out of range")
                    })?;
                    let components = component_count(&accessor.kind);
                    if !matches!(components, 3 | 4) {
                        return Err(GltfError::new("animation output must be VEC3 or VEC4"));
                    }
                    let output =
                        read_accessor::<4>(buffers, views, accessors, output_accessor, components)?;
                    if input.len() != output.len() {
                        return Err(GltfError::new("animation input and output counts differ"));
                    }
                    let interpolation =
                        match get_optional_string(s, "interpolation")?.unwrap_or("LINEAR") {
                            "LINEAR" => Interpolation::Linear,
                            "STEP" => Interpolation::Step,
                            "CUBICSPLINE" => {
                                return Err(GltfError::new("CUBICSPLINE animation is unsupported"));
                            }
                            other => {
                                return Err(GltfError::new(format!(
                                    "unsupported animation interpolation {other}"
                                )));
                            }
                        };
                    Ok(GltfAnimationSampler {
                        input,
                        output,
                        output_components: components,
                        interpolation,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let channels = get_array(o, "channels")?
                .iter()
                .map(|value| {
                    let c = as_object(value, "animation channel")?;
                    let target = get_object(c, "target")?;
                    let node = get_usize(target, "node")?;
                    if node >= node_count {
                        return Err(GltfError::new("animation target node is out of range"));
                    }
                    let path = match get_string(target, "path")? {
                        "translation" => AnimationPath::Translation,
                        "rotation" => AnimationPath::Rotation,
                        "scale" => AnimationPath::Scale,
                        other => {
                            return Err(GltfError::new(format!(
                                "unsupported animation target {other}"
                            )));
                        }
                    };
                    Ok(GltfAnimationChannel {
                        sampler: get_usize(c, "sampler")?,
                        node,
                        path,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            for channel in &channels {
                let sampler = samplers
                    .get(channel.sampler)
                    .ok_or_else(|| GltfError::new("animation channel sampler is out of range"))?;
                let expected = if channel.path == AnimationPath::Rotation {
                    4
                } else {
                    3
                };
                if sampler.output_components != expected {
                    return Err(GltfError::new(
                        "animation output type does not match its target path",
                    ));
                }
            }
            let duration = samplers
                .iter()
                .flat_map(|sampler| sampler.input.iter().copied())
                .fold(0.0, f32::max);
            Ok(GltfAnimation {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                samplers,
                channels,
                duration,
            })
        })
        .collect()
}

fn parse_scenes(
    object: &[(String, Value)],
    node_count: usize,
) -> Result<Vec<GltfScene>, GltfError> {
    get_optional_array(object, "scenes")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "scene")?;
            let nodes = get_usize_array(o, "nodes")?;
            if nodes.iter().any(|&node| node >= node_count) {
                return Err(GltfError::new("scene node index is out of range"));
            }
            Ok(GltfScene {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                nodes,
            })
        })
        .collect()
}

fn load_images(object: &[(String, Value)], root: &Path) -> Result<Vec<Option<Texture>>, GltfError> {
    get_optional_array(object, "images")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "image")?;
            if get_optional_usize(o, "bufferView")?.is_some() {
                return Err(GltfError::new("bufferView images are unsupported"));
            }
            let Some(uri) = get_optional_string(o, "uri")? else {
                return Err(GltfError::new("image needs uri"));
            };
            let (mime, bytes) = if let Some(encoded) = uri.strip_prefix("data:") {
                let (header, payload) = encoded
                    .split_once(',')
                    .ok_or_else(|| GltfError::new("invalid image data URI"))?;
                (
                    header.split(';').next().unwrap_or(""),
                    decode_base64(payload)?,
                )
            } else {
                let path = safe_asset_path(root, Path::new(uri))?;
                let mime = match path.extension().and_then(|extension| extension.to_str()) {
                    Some("qoi") => "image/qoi",
                    Some("ppm") => "image/x-portable-pixmap",
                    _ => return Err(GltfError::new("unsupported glTF image format")),
                };
                (
                    mime,
                    fs::read(path).map_err(|error| GltfError::new(error.to_string()))?,
                )
            };
            let texture = match mime {
                "image/qoi" | "image/x-qoi" => {
                    Texture::from_qoi_with_color_space(&bytes, ColorSpace::Srgb)
                }
                "image/x-portable-pixmap" | "image/ppm" => {
                    Texture::from_ppm_with_color_space(&bytes, ColorSpace::Srgb)
                }
                _ => {
                    return Err(GltfError::new(format!(
                        "unsupported glTF image MIME type {mime}"
                    )));
                }
            }
            .map_err(|error| GltfError::new(error.to_string()))?;
            Ok(Some(texture))
        })
        .collect()
}

fn parse_textures(object: &[(String, Value)]) -> Result<Vec<TextureRef>, GltfError> {
    get_optional_array(object, "textures")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "texture")?;
            Ok(TextureRef {
                source: get_usize(o, "source")?,
            })
        })
        .collect()
}

fn parse_materials(
    object: &[(String, Value)],
    images: &[Option<Texture>],
    textures: &[TextureRef],
) -> Result<Vec<GltfMaterial>, GltfError> {
    get_optional_array(object, "materials")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "material")?;
            let pbr = get_optional_object(o, "pbrMetallicRoughness")?;
            let factor = vec4(
                pbr.and_then(|p| get_optional_f32_array(p, "baseColorFactor").transpose())
                    .transpose()?
                    .unwrap_or_else(|| vec![1.0, 1.0, 1.0, 1.0]),
                "baseColorFactor",
            )?;
            let base_color_factor = Vec4::new(factor[0], factor[1], factor[2], factor[3]);
            let metallic_factor = pbr
                .map(|p| get_optional_f32(p, "metallicFactor"))
                .transpose()?
                .flatten()
                .unwrap_or(1.0);
            let roughness_factor = pbr
                .map(|p| get_optional_f32(p, "roughnessFactor"))
                .transpose()?
                .flatten()
                .unwrap_or(1.0);
            let albedo_texture = if let Some(pbr) = pbr {
                get_optional_object(pbr, "baseColorTexture")?
                    .map(|texture| texture_from_ref(texture, textures, images, ColorSpace::Srgb))
                    .transpose()?
            } else {
                None
            };
            let normal_map_texture = get_optional_object(o, "normalTexture")?
                .map(|texture| texture_from_ref(texture, textures, images, ColorSpace::Linear))
                .transpose()?;
            Ok(GltfMaterial {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                base_color_factor,
                metallic_factor,
                roughness_factor,
                albedo_texture,
                normal_map_texture,
            })
        })
        .collect()
}

fn texture_from_ref(
    object: &[(String, Value)],
    textures: &[TextureRef],
    images: &[Option<Texture>],
    color_space: ColorSpace,
) -> Result<Texture, GltfError> {
    let index = get_usize(object, "index")?;
    let texture = textures
        .get(index)
        .ok_or_else(|| GltfError::new("texture index is out of range"))?;
    let image = images
        .get(texture.source)
        .ok_or_else(|| GltfError::new("image index is out of range"))?
        .clone()
        .ok_or_else(|| GltfError::new("image is missing"))?;
    if color_space == ColorSpace::Linear {
        let pixels = image.pixels().to_vec();
        return Texture::new_with_color_space(
            image.width(),
            image.height(),
            pixels,
            ColorSpace::Linear,
        )
        .map_err(|error| GltfError::new(error.to_string()));
    }
    Ok(image)
}

fn safe_asset_path(root: &Path, relative: &Path) -> Result<PathBuf, GltfError> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(GltfError::new(format!(
            "asset path is absolute or traverses outside root: {}",
            relative.display()
        )));
    }
    let root =
        fs::canonicalize(root).map_err(|error| GltfError::new(format!("asset root: {error}")))?;
    let candidate = root.join(relative);
    let canonical = fs::canonicalize(&candidate)
        .map_err(|error| GltfError::new(format!("asset {}: {error}", candidate.display())))?;
    if !canonical.starts_with(&root) {
        return Err(GltfError::new("asset path escapes root"));
    }
    Ok(canonical)
}

fn as_object<'a>(value: &'a Value, what: &str) -> Result<&'a [(String, Value)], GltfError> {
    match value {
        Value::Object(value) => Ok(value),
        _ => Err(GltfError::new(format!("{what} must be an object"))),
    }
}
fn get_object<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<&'a [(String, Value)], GltfError> {
    get(object, key)
        .map(|value| as_object(value, key))
        .transpose()?
        .ok_or_else(|| GltfError::new(format!("missing object {key}")))
}
fn get_optional_object<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<Option<&'a [(String, Value)]>, GltfError> {
    get(object, key)
        .map(|value| as_object(value, key))
        .transpose()
}
fn get_array<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a [Value], GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Array(value) => Ok(value.as_slice()),
            _ => Err(GltfError::new(format!("{key} must be an array"))),
        })
        .transpose()?
        .ok_or_else(|| GltfError::new(format!("missing array {key}")))
}
fn get_optional_array<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<Option<&'a [Value]>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Array(value) => Ok(value.as_slice()),
            _ => Err(GltfError::new(format!("{key} must be an array"))),
        })
        .transpose()
}
fn get<'a>(object: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    object
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}
fn get_string<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a str, GltfError> {
    match get(object, key) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(GltfError::new(format!("{key} must be a string"))),
        None => Err(GltfError::new(format!("missing string {key}"))),
    }
}
fn get_optional_string<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<Option<&'a str>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::String(value) => Ok(value.as_str()),
            _ => Err(GltfError::new(format!("{key} must be a string"))),
        })
        .transpose()
}
fn get_f64(object: &[(String, Value)], key: &str) -> Result<f64, GltfError> {
    match get(object, key) {
        Some(Value::Number(value)) => Ok(*value),
        Some(_) => Err(GltfError::new(format!("{key} must be a number"))),
        None => Err(GltfError::new(format!("missing number {key}"))),
    }
}
fn get_optional_f32(object: &[(String, Value)], key: &str) -> Result<Option<f32>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Number(value)
                if value.is_finite()
                    && *value >= f64::from(f32::MIN)
                    && *value <= f64::from(f32::MAX) =>
            {
                Ok(*value as f32)
            }
            Value::Number(_) => Err(GltfError::new(format!("{key} is not a finite f32"))),
            _ => Err(GltfError::new(format!("{key} must be a number"))),
        })
        .transpose()
}
fn get_u32(object: &[(String, Value)], key: &str) -> Result<u32, GltfError> {
    let value = get_f64(object, key)?;
    if value < 0.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
        return Err(GltfError::new(format!("{key} must be a u32")));
    }
    Ok(value as u32)
}
fn get_optional_u32(object: &[(String, Value)], key: &str) -> Result<Option<u32>, GltfError> {
    get(object, key).map(|_| get_u32(object, key)).transpose()
}
fn get_usize(object: &[(String, Value)], key: &str) -> Result<usize, GltfError> {
    let value = get_f64(object, key)?;
    if value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(GltfError::new(format!("{key} must be a usize")));
    }
    Ok(value as usize)
}
fn get_optional_usize(object: &[(String, Value)], key: &str) -> Result<Option<usize>, GltfError> {
    get(object, key).map(|_| get_usize(object, key)).transpose()
}
fn get_optional_bool(object: &[(String, Value)], key: &str) -> Result<Option<bool>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Bool(value) => Ok(*value),
            _ => Err(GltfError::new(format!("{key} must be a bool"))),
        })
        .transpose()
}
fn get_usize_array(object: &[(String, Value)], key: &str) -> Result<Vec<usize>, GltfError> {
    get_array(object, key)?
        .iter()
        .map(|value| match value {
            Value::Number(value) if *value >= 0.0 && value.fract() == 0.0 => Ok(*value as usize),
            _ => Err(GltfError::new(format!("{key} needs integer values"))),
        })
        .collect()
}
fn get_optional_usize_array(
    object: &[(String, Value)],
    key: &str,
) -> Result<Option<Vec<usize>>, GltfError> {
    get_optional_array(object, key)?
        .map(|values| {
            values
                .iter()
                .map(|value| match value {
                    Value::Number(value) if *value >= 0.0 && value.fract() == 0.0 => {
                        Ok(*value as usize)
                    }
                    _ => Err(GltfError::new(format!("{key} needs integer values"))),
                })
                .collect()
        })
        .transpose()
}
fn get_optional_f32_array(
    object: &[(String, Value)],
    key: &str,
) -> Result<Option<Vec<f32>>, GltfError> {
    get_optional_array(object, key)?
        .map(|values| {
            values
                .iter()
                .map(|value| match value {
                    Value::Number(value)
                        if value.is_finite()
                            && *value >= f64::from(f32::MIN)
                            && *value <= f64::from(f32::MAX) =>
                    {
                        Ok(*value as f32)
                    }
                    _ => Err(GltfError::new(format!("{key} needs finite number values"))),
                })
                .collect()
        })
        .transpose()
}
fn vec3(values: Vec<f32>, key: &str) -> Result<Vec3, GltfError> {
    if values.len() != 3 {
        return Err(GltfError::new(format!("{key} needs 3 values")));
    }
    Ok(Vec3::new(values[0], values[1], values[2]))
}
fn vec4(values: Vec<f32>, key: &str) -> Result<[f32; 4], GltfError> {
    if values.len() != 4 {
        return Err(GltfError::new(format!("{key} needs 4 values")));
    }
    Ok([values[0], values[1], values[2], values[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn base64_vectors_decode_without_a_crate() {
        assert_eq!(decode_base64(""), Ok(Vec::new()));
        assert_eq!(decode_base64("SGVsbG8="), Ok(b"Hello".to_vec()));
        assert_eq!(decode_base64("AAEC"), Ok(vec![0, 1, 2]));
        assert!(decode_base64("A===").is_err());
    }

    #[test]
    fn slerp_uses_the_shortest_path() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0, (FRAC_PI_2 / 2.0).sin(), (FRAC_PI_2 / 2.0).cos()];
        let halfway = slerp(a, b, 0.5);
        let expected = (FRAC_PI_2 / 4.0).sin();
        assert!((halfway[2] - expected).abs() < 1.0e-5);
        assert!((halfway[3] - (FRAC_PI_2 / 4.0).cos()).abs() < 1.0e-5);
    }

    #[test]
    fn trs_is_translation_rotation_scale() {
        let transform = NodeTransform {
            translation: Vec3::new(2.0, 0.0, 0.0),
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: Vec3::new(3.0, 3.0, 3.0),
        }
        .matrix();
        let result = transform * Vec4::new(1.0, 0.0, 0.0, 1.0);
        assert_eq!(result, Vec4::new(5.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn rejects_accessor_overrun() {
        let source = r#"{"asset":{"version":"2.0"},"buffers":[{"uri":"data:application/octet-stream;base64,AA==","byteLength":1}],"bufferViews":[{"buffer":0,"byteLength":1}],"accessors":[{"bufferView":0,"componentType":5126,"count":1,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]}"#;
        assert!(GltfAsset::from_str(source, ".").is_err());
    }

    #[test]
    fn skins_the_hand_authored_arm_at_bind_and_rotated_pose() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/arm.gltf");
        let asset = GltfAsset::load(path).unwrap();
        let bind = asset.pose_mesh(0, 0, 0, Some(0), 0.0).unwrap();
        assert_eq!(bind.vertex(2).unwrap().position(), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(
            bind.vertex(2).unwrap().normal(),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );
        let posed = asset.pose_mesh(0, 0, 0, Some(0), 1.0).unwrap();
        let position = posed.vertex(2).unwrap().position();
        assert!((position.x - 0.0).abs() < 1.0e-5);
        assert!((position.y - 0.0).abs() < 1.0e-5);
        assert_eq!(
            posed.vertex(2).unwrap().normal(),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );
    }

    #[test]
    fn skin_weights_are_normalized_with_zero_weight_fallback() {
        assert_eq!(
            normalize_weights([2.0, 1.0, 0.0, 0.0]),
            [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]
        );
        assert_eq!(normalize_weights([0.0; 4]), [0.0; 4]);
    }

    #[test]
    fn animation_sampling_covers_exact_between_and_step_times() {
        let linear = GltfAnimationSampler {
            input: vec![0.0, 1.0],
            output: vec![[0.0, 0.0, 0.0, 0.0], [2.0, 4.0, 6.0, 0.0]],
            output_components: 3,
            interpolation: Interpolation::Linear,
        };
        assert_eq!(linear.sample(0.0).unwrap(), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(linear.sample(1.0).unwrap(), [2.0, 4.0, 6.0, 0.0]);
        assert_eq!(linear.sample(0.5).unwrap(), [1.0, 2.0, 3.0, 0.0]);
        let step = GltfAnimationSampler {
            interpolation: Interpolation::Step,
            ..linear
        };
        assert_eq!(step.sample(0.5).unwrap(), [0.0, 0.0, 0.0, 0.0]);
    }
}
