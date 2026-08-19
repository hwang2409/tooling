//! Shared rendered showcase adapter for the newt examples.
//!
//! The adapter owns only demo-side geometry and presentation code. It reads
//! newt state and submits shaded meshes to chimy2's public pipeline.

#![allow(dead_code)]

use chimy2::camera::Camera;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Quat as CQuat, Vec3};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{CookTorranceShader, CookTorranceUniforms, DirectionalLight, PointLight};

use newt::geom::{GeomShape, geom_world_pose};
use newt::math::{Quat, Vec3 as NVec3};
use newt::tree::forward_kinematics;
use newt::world::World;

use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::process::Command;

pub const VIDEO_FPS: u32 = 60;
pub const SIM_STEPS_PER_VIDEO_FRAME: usize = 10;

pub fn video_speed_factor(sim_dt: f32) -> f32 {
    VIDEO_FPS as f32 * SIM_STEPS_PER_VIDEO_FRAME as f32 * sim_dt
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub albedo: Vec3,
    pub metallic: f32,
    pub roughness: f32,
}

impl Material {
    pub const fn new(albedo: Vec3, metallic: f32, roughness: f32) -> Self {
        Self {
            albedo,
            metallic,
            roughness,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Composition {
    pub target: Vec3,
    pub camera_position: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub accent: Vec3,
}

impl Composition {
    pub const fn new(target: Vec3, camera_position: Vec3, accent: Vec3) -> Self {
        Self {
            target,
            camera_position,
            fov: PI / 4.0,
            near: 0.03,
            far: 80.0,
            accent,
        }
    }

    pub fn camera(self, width: usize, height: usize) -> Camera {
        Camera::new(
            self.camera_position,
            CQuat::IDENTITY,
            self.fov,
            width as f32 / height.max(1) as f32,
            self.near,
            self.far,
        )
    }

    pub fn view_matrix(self) -> Mat4 {
        Mat4::look_at(self.camera_position, self.target, Vec3::new(0.0, 0.0, 1.0))
    }
}

/// Fixed camera presets used by the demos. Keep this table compact so framing
/// stays easy to adjust without changing simulation code.
pub fn composition(name: &str) -> Composition {
    match name {
        "tendon" => Composition::new(
            Vec3::new(0.5, 0.0, -0.55),
            Vec3::new(3.0, -3.5, 1.8),
            Vec3::new(0.05, 0.75, 0.9),
        ),
        "stack" => Composition::new(
            Vec3::new(0.0, 0.0, 1.8),
            Vec3::new(4.8, -5.8, 3.8),
            Vec3::new(1.0, 0.35, 0.1),
        ),
        "pile" => Composition::new(
            Vec3::new(0.0, 0.0, 0.65),
            Vec3::new(3.4, -4.1, 2.6),
            Vec3::new(0.95, 0.25, 0.2),
        ),
        "hfield" => Composition::new(
            Vec3::new(0.0, 0.0, 0.65),
            Vec3::new(4.2, -5.2, 3.0),
            Vec3::new(0.35, 0.8, 0.25),
        ),
        "cartpole" => Composition::new(
            Vec3::new(0.0, 0.0, 1.1),
            Vec3::new(0.0, -4.5, 1.6),
            Vec3::new(0.95, 0.42, 0.08),
        ),
        "arm" => Composition::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(3.0, -0.2, 1.4),
            Vec3::new(0.8, 0.3, 0.75),
        ),
        "features" => Composition::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(4.8, -7.5, 3.5),
            Vec3::new(0.1, 0.75, 0.9),
        ),
        "compare" => Composition::new(
            Vec3::new(0.0, 0.0, 1.6),
            Vec3::new(7.8, -10.0, 5.4),
            Vec3::new(0.95, 0.45, 0.1),
        ),
        "tumble" => Composition::new(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(8.0, -10.0, 7.0),
            Vec3::new(0.95, 0.5, 0.1),
        ),
        "linkage" => Composition::new(
            Vec3::new(0.0, 0.0, 0.7),
            Vec3::new(2.8, -4.0, 2.4),
            Vec3::new(0.95, 0.45, 0.1),
        ),
        "pendulum" => Composition::new(
            Vec3::new(0.0, 0.0, 0.6),
            Vec3::new(4.5, 0.0, 1.0),
            Vec3::new(0.1, 0.65, 0.9),
        ),
        _ => Composition::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(4.0, -5.0, 2.8),
            Vec3::new(0.1, 0.65, 0.95),
        ),
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub mesh: Mesh,
    pub model: Mat4,
    pub material: Material,
}

pub fn item(mesh: Mesh, model: Mat4, material: Material) -> Item {
    Item {
        mesh,
        model,
        material,
    }
}

pub fn render_items(
    items: &[Item],
    composition: Composition,
    width: usize,
    height: usize,
    hud: &str,
) -> Framebuffer {
    let camera = composition.camera(width, height);
    let mut framebuffer = Framebuffer::new(width, height);
    framebuffer.clear(argb8888(255, 8, 10, 16));

    let light_direction = Vec3::new(-0.45, -0.65, 0.9).normalize();
    let uniforms = items
        .iter()
        .map(|value| {
            let lighting = chimy2::shaders::BlinnPhongUniforms::new_with_linear_colors(
                value.model,
                composition.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.025, 0.03, 0.04),
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(0.18, 0.18, 0.18),
                64.0,
                camera.position,
                DirectionalLight::new(light_direction, Vec3::new(2.2, 2.0, 1.8)),
                PointLight::new(
                    composition.camera_position + Vec3::new(0.0, 0.0, 4.0),
                    Vec3::new(0.35, 0.2, 0.12),
                    1.0,
                    0.08,
                    0.03,
                ),
            );
            CookTorranceUniforms::new_with_linear_base_color(
                lighting,
                value.material.albedo,
                value.material.metallic,
                value.material.roughness,
            )
        })
        .collect::<Vec<_>>();
    let mut pipeline = Pipeline::new(CookTorranceShader, CookTorranceShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        for (value, uniforms) in items.iter().zip(&uniforms) {
            frame.draw_mesh(target, &value.mesh, uniforms);
        }
    });
    framebuffer.draw_text(18, 16, hud, 1, argb8888(255, 220, 230, 245));
    framebuffer
}

pub fn write_frame(
    items: &[Item],
    composition: Composition,
    width: usize,
    height: usize,
    hud: &str,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let framebuffer = render_items(items, composition, width, height, hud);
    chimy2::demo::write_ppm(path, &framebuffer)?;
    Ok(())
}

pub struct VideoWriter {
    output: std::path::PathBuf,
    frame_dir: std::path::PathBuf,
    next_frame: usize,
}

impl VideoWriter {
    pub fn new(output: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let output = output.as_ref().to_path_buf();
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let frame_dir = std::env::temp_dir().join(format!(
            "newt-showcase-{}-{}",
            std::process::id(),
            output
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("demo")
        ));
        if frame_dir.exists() {
            fs::remove_dir_all(&frame_dir)?;
        }
        fs::create_dir_all(&frame_dir)?;
        Ok(Self {
            output,
            frame_dir,
            next_frame: 0,
        })
    }

    pub fn push(&mut self, framebuffer: &Framebuffer) -> Result<(), Box<dyn std::error::Error>> {
        let path = self
            .frame_dir
            .join(format!("frame-{:06}.ppm", self.next_frame));
        chimy2::demo::write_ppm(path, framebuffer)?;
        self.next_frame += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<(), Box<dyn std::error::Error>> {
        let input = self.frame_dir.join("frame-%06d.ppm");
        let result = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                &VIDEO_FPS.to_string(),
                "-i",
                input.to_str().unwrap_or_default(),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                self.output.to_str().unwrap_or_default(),
            ])
            .output();
        let cleanup = fs::remove_dir_all(&self.frame_dir);
        match (result, cleanup) {
            (Err(error), _) if error.kind() == std::io::ErrorKind::NotFound => {
                Err("ffmpeg is required for video output; install it (for example, /opt/homebrew/bin/ffmpeg) and retry".into())
            }
            (Err(error), _) => Err(error.into()),
            (Ok(output), Err(error)) if !output.status.success() => Err(format!(
                "ffmpeg failed: {} (cleanup also failed: {error})",
                String::from_utf8_lossy(&output.stderr)
            )
            .into()),
            (Ok(output), _) if !output.status.success() => {
                Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr)).into())
            }
            (Ok(_), Err(error)) => Err(error.into()),
            (Ok(_), Ok(())) => Ok(()),
        }
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.frame_dir);
    }
}

pub fn write_video<F>(
    output: impl AsRef<Path>,
    total_steps: usize,
    mut render: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(usize) -> Framebuffer,
{
    let mut writer = VideoWriter::new(output)?;
    let mut simulated = 0;
    while simulated < total_steps {
        let next = simulated + SIM_STEPS_PER_VIDEO_FRAME.min(total_steps - simulated);
        writer.push(&render(next))?;
        simulated = next;
    }
    writer.finish()
}

pub fn world_items(world: &World) -> Vec<Item> {
    let mut items = Vec::new();
    for (index, geom) in world.geoms.iter().enumerate() {
        let (parent_position, parent_orientation) = match geom.attachment() {
            newt::geom::GeomAttach::Static => (NVec3::ZERO, Quat::IDENTITY),
            newt::geom::GeomAttach::Body(body) => {
                let value = &world.bodies[body];
                (value.position, value.orientation)
            }
            newt::geom::GeomAttach::Link(tree, link) => {
                forward_kinematics(&world.trees[tree])[link]
            }
        };
        let pose = geom_world_pose(geom, parent_position, parent_orientation);
        let material = palette(index);
        match geom.shape {
            GeomShape::Plane => {
                items.push(item(
                    cuboid_mesh(Vec3::new(7.0, 7.0, 0.035)),
                    Mat4::translate(to_cvec(pose.position - NVec3::new(0.0, 0.0, 0.035))),
                    Material::new(Vec3::new(0.045, 0.055, 0.07), 0.0, 0.88),
                ));
            }
            GeomShape::Sphere { radius } => items.push(item(
                sphere_mesh(16, 10),
                transform(
                    pose.position,
                    pose.orientation,
                    Vec3::new(radius, radius, radius),
                ),
                material,
            )),
            GeomShape::Box { half_extents } => items.push(item(
                cuboid_mesh(to_cvec(half_extents)),
                transform(pose.position, pose.orientation, Vec3::new(1.0, 1.0, 1.0)),
                material,
            )),
            GeomShape::Capsule {
                radius,
                half_height,
            } => items.push(item(
                capsule_mesh(16, 12, radius, half_height),
                transform(pose.position, pose.orientation, Vec3::new(1.0, 1.0, 1.0)),
                material,
            )),
            GeomShape::Cylinder {
                radius,
                half_height,
            } => items.push(item(
                cylinder_mesh(20, radius, half_height),
                transform(pose.position, pose.orientation, Vec3::new(1.0, 1.0, 1.0)),
                material,
            )),
            GeomShape::Ellipsoid { semi_axes } => items.push(item(
                sphere_mesh(16, 10),
                transform(pose.position, pose.orientation, to_cvec(semi_axes)),
                material,
            )),
            GeomShape::Mesh { mesh_id } => {
                let mesh = &world.meshes[mesh_id];
                let vertices = mesh
                    .vertices
                    .iter()
                    .map(|&position| MeshVertex::new(to_cvec(position), None, None))
                    .collect();
                let triangles = mesh
                    .faces
                    .iter()
                    .map(|face| [face[0] as usize, face[1] as usize, face[2] as usize])
                    .collect();
                items.push(item(
                    Mesh::new(vertices, triangles),
                    transform(pose.position, pose.orientation, Vec3::new(1.0, 1.0, 1.0)),
                    material,
                ));
            }
            GeomShape::Hfield { hfield_id } => {
                items.push(item(
                    hfield_mesh(&world.hfields[hfield_id]),
                    transform(pose.position, pose.orientation, Vec3::new(1.0, 1.0, 1.0)),
                    Material::new(Vec3::new(0.22, 0.34, 0.18), 0.0, 0.92),
                ));
            }
        }
    }
    items
}

fn hfield_mesh(hfield: &newt::geom::HeightField) -> Mesh {
    let mut vertices = Vec::with_capacity(hfield.nrow * hfield.ncol);
    for row in 0..hfield.nrow {
        let y = -hfield.size[1] + 2.0 * hfield.size[1] * row as f32 / (hfield.nrow - 1) as f32;
        for col in 0..hfield.ncol {
            let x = -hfield.size[0] + 2.0 * hfield.size[0] * col as f32 / (hfield.ncol - 1) as f32;
            vertices.push(MeshVertex::new(
                Vec3::new(x, y, hfield.height(row, col)),
                None,
                None,
            ));
        }
    }
    let mut triangles = Vec::with_capacity((hfield.nrow - 1) * (hfield.ncol - 1) * 2);
    for row in 0..hfield.nrow - 1 {
        for col in 0..hfield.ncol - 1 {
            let i = row * hfield.ncol + col;
            triangles.push([i, i + 1, i + hfield.ncol + 1]);
            triangles.push([i, i + hfield.ncol + 1, i + hfield.ncol]);
        }
    }
    Mesh::new(vertices, triangles)
}

pub fn add_capsule(items: &mut Vec<Item>, a: NVec3, b: NVec3, radius: f32, material: Material) {
    let delta = b - a;
    let length = delta.length();
    let midpoint = (a + b) * 0.5;
    let orientation = align_z(delta);
    items.push(item(
        capsule_mesh(10, 8, radius, length * 0.5),
        transform(midpoint, orientation, Vec3::new(1.0, 1.0, 1.0)),
        material,
    ));
}

pub fn add_marker(items: &mut Vec<Item>, position: NVec3, radius: f32, material: Material) {
    items.push(item(
        sphere_mesh(10, 6),
        Mat4::translate(to_cvec(position)) * Mat4::scale(Vec3::new(radius, radius, radius)),
        material,
    ));
}

pub fn to_cvec(value: NVec3) -> Vec3 {
    Vec3::new(value.x, value.y, value.z)
}

pub fn transform(position: NVec3, orientation: Quat, scale: Vec3) -> Mat4 {
    Mat4::translate(to_cvec(position))
        * CQuat::new(orientation.x, orientation.y, orientation.z, orientation.w).to_mat4()
        * Mat4::scale(scale)
}

fn palette(index: usize) -> Material {
    let colors = [
        Vec3::new(0.12, 0.42, 0.78),
        Vec3::new(0.88, 0.26, 0.12),
        Vec3::new(0.15, 0.68, 0.52),
        Vec3::new(0.72, 0.28, 0.74),
        Vec3::new(0.82, 0.55, 0.12),
        Vec3::new(0.35, 0.42, 0.54),
    ];
    Material::new(colors[index % colors.len()], 0.18, 0.38)
}

fn align_z(vector: NVec3) -> Quat {
    let direction = vector.normalize();
    if direction.length() == 0.0 {
        return Quat::IDENTITY;
    }
    let dot = NVec3::Z.dot(direction).clamp(-1.0, 1.0);
    if dot > 0.9999 {
        return Quat::IDENTITY;
    }
    if dot < -0.9999 {
        return Quat::from_axis_angle(NVec3::X, PI);
    }
    Quat::from_axis_angle(NVec3::Z.cross(direction).normalize(), dot.acos())
}

fn vertex(position: Vec3) -> MeshVertex {
    MeshVertex::new(position, None, None)
}

pub fn cuboid_mesh(half: Vec3) -> Mesh {
    let vertices = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    .map(|(x, y, z)| vertex(Vec3::new(x * half.x, y * half.y, z * half.z)));
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    Mesh::new(vertices.to_vec(), triangles)
}

pub fn sphere_mesh(segments: usize, rings: usize) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = PI * v;
        let z = phi.cos();
        let radius = phi.sin();
        for segment in 0..segments {
            let theta = 2.0 * PI * segment as f32 / segments as f32;
            vertices.push(vertex(Vec3::new(
                radius * theta.cos(),
                radius * theta.sin(),
                z,
            )));
        }
    }
    for ring in 0..rings {
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            let a = ring * segments + segment;
            let b = ring * segments + next;
            let c = (ring + 1) * segments + next;
            let d = (ring + 1) * segments + segment;
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
    }
    Mesh::new(vertices, triangles)
}

fn cylinder_mesh(segments: usize, radius: f32, half_height: f32) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for &z in &[-half_height, half_height] {
        for segment in 0..segments {
            let theta = 2.0 * PI * segment as f32 / segments as f32;
            vertices.push(vertex(Vec3::new(
                radius * theta.cos(),
                radius * theta.sin(),
                z,
            )));
        }
    }
    for segment in 0..segments {
        let next = (segment + 1) % segments;
        triangles.push([segment, next, segments + next]);
        triangles.push([segment, segments + next, segments + segment]);
    }
    let bottom = vertices.len();
    vertices.push(vertex(Vec3::new(0.0, 0.0, -half_height)));
    let top = vertices.len();
    vertices.push(vertex(Vec3::new(0.0, 0.0, half_height)));
    for segment in 0..segments {
        let next = (segment + 1) % segments;
        triangles.push([bottom, next, segment]);
        triangles.push([top, segments + segment, segments + next]);
    }
    Mesh::new(vertices, triangles)
}

fn capsule_mesh(segments: usize, rings: usize, radius: f32, half_height: f32) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut push_ring = |ring_radius: f32, z: f32| {
        for segment in 0..segments {
            let theta = 2.0 * PI * segment as f32 / segments as f32;
            vertices.push(vertex(Vec3::new(
                ring_radius * theta.cos(),
                ring_radius * theta.sin(),
                z,
            )));
        }
    };

    // Lower hemisphere, from the south pole to the cylinder.
    for ring in 0..=rings {
        let t = ring as f32 / rings as f32;
        let angle = -PI * 0.5 + t * PI * 0.5;
        push_ring(radius * angle.cos(), -half_height + radius * angle.sin());
    }
    // Include interior cylinder rings so the cylindrical section is explicit.
    for ring in 1..=rings + 1 {
        let t = ring as f32 / (rings + 1) as f32;
        push_ring(radius, -half_height + 2.0 * half_height * t);
    }
    // Upper hemisphere, from the cylinder to the north pole.
    for ring in 1..=rings {
        let t = ring as f32 / rings as f32;
        let angle = t * PI * 0.5;
        push_ring(radius * angle.cos(), half_height + radius * angle.sin());
    }

    let ring_count = vertices.len() / segments;
    for ring in 0..ring_count - 1 {
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            let a = ring * segments + segment;
            let b = ring * segments + next;
            let c = (ring + 1) * segments + next;
            let d = (ring + 1) * segments + segment;
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
    }
    Mesh::new(vertices, triangles)
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capsule_rings_have_monotone_z_and_a_cylinder_section() {
        let segments = 8;
        let rings = 4;
        let half_height = 2.0;
        let mesh = capsule_mesh(segments, rings, 0.5, half_height);
        let ring_z = mesh
            .vertices()
            .chunks(segments)
            .map(|ring| ring[0].position().z)
            .collect::<Vec<_>>();

        assert!(ring_z.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            ring_z
                .iter()
                .filter(|&&z| -half_height < z && z < half_height)
                .count()
                >= rings
        );
    }

    fn push_f32(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn push_vec3(bytes: &mut Vec<u8>, value: NVec3) {
        push_f32(bytes, value.x);
        push_f32(bytes, value.y);
        push_f32(bytes, value.z);
    }

    fn push_quat(bytes: &mut Vec<u8>, value: Quat) {
        push_f32(bytes, value.x);
        push_f32(bytes, value.y);
        push_f32(bytes, value.z);
        push_f32(bytes, value.w);
    }

    fn push_mat3(bytes: &mut Vec<u8>, value: newt::math::Mat3) {
        for component in value.data {
            push_f32(bytes, component);
        }
    }

    fn push_geom_shape(bytes: &mut Vec<u8>, shape: &GeomShape) {
        match shape {
            GeomShape::Plane => bytes.push(0),
            GeomShape::Sphere { radius } => {
                bytes.push(1);
                push_f32(bytes, *radius);
            }
            GeomShape::Box { half_extents } => {
                bytes.push(2);
                push_vec3(bytes, *half_extents);
            }
            GeomShape::Capsule {
                radius,
                half_height,
            } => {
                bytes.push(3);
                push_f32(bytes, *radius);
                push_f32(bytes, *half_height);
            }
            GeomShape::Cylinder {
                radius,
                half_height,
            } => {
                bytes.push(4);
                push_f32(bytes, *radius);
                push_f32(bytes, *half_height);
            }
            GeomShape::Ellipsoid { semi_axes } => {
                bytes.push(5);
                push_vec3(bytes, *semi_axes);
            }
            GeomShape::Mesh { mesh_id } => {
                bytes.push(6);
                bytes.extend_from_slice(&(*mesh_id as u64).to_le_bytes());
            }
            GeomShape::Hfield { hfield_id } => {
                bytes.push(7);
                bytes.extend_from_slice(&(*hfield_id as u64).to_le_bytes());
            }
        }
    }

    fn world_float_bits(world: &World) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_f32(&mut bytes, world.dt);
        push_vec3(&mut bytes, world.gravity);
        push_vec3(&mut bytes, world.magnetic_field);
        for body in &world.bodies {
            push_f32(&mut bytes, body.mass);
            push_mat3(&mut bytes, body.inertia_body);
            push_mat3(&mut bytes, body.inertia_body_inverse);
            push_vec3(&mut bytes, body.position);
            push_vec3(&mut bytes, body.linear_velocity);
            push_quat(&mut bytes, body.orientation);
            push_vec3(&mut bytes, body.angular_velocity_body);
        }
        for geom in &world.geoms {
            push_geom_shape(&mut bytes, &geom.shape);
            push_vec3(&mut bytes, geom.local_offset);
            push_quat(&mut bytes, geom.local_orientation);
            push_f32(&mut bytes, geom.friction);
            push_f32(&mut bytes, geom.solref.timeconst);
            push_f32(&mut bytes, geom.solref.dampratio);
            push_f32(&mut bytes, geom.margin);
            push_f32(&mut bytes, geom.gap);
            push_f32(&mut bytes, geom.torsional_friction);
            push_f32(&mut bytes, geom.rolling_friction);
            push_f32(&mut bytes, geom.solimp.dmin);
            push_f32(&mut bytes, geom.solimp.dmax);
            push_f32(&mut bytes, geom.solimp.width);
            push_f32(&mut bytes, geom.solimp.midpoint);
        }
        for mesh in &world.meshes {
            for vertex in &mesh.vertices {
                push_vec3(&mut bytes, *vertex);
            }
        }
        bytes
    }

    fn run_with_render_divisor(divisor: Option<usize>) -> World {
        let mut world = newt::model::load_from_path("models/pile.json")
            .expect("pile model")
            .world;
        let composition = Composition::new(
            Vec3::new(0.0, 0.0, 0.6),
            Vec3::new(3.4, -4.1, 2.6),
            Vec3::new(1.0, 0.3, 0.2),
        );
        for step in 1..=60 {
            world.step();
            if divisor.is_some_and(|value| step % value == 0 || step == 60) {
                let items = world_items(&world);
                let _ = render_items(&items, composition, 32, 32, "test");
            }
        }
        world
    }

    #[test]
    fn rendering_does_not_change_deterministic_simulation_state() {
        let headless = run_with_render_divisor(None);
        let headless_bits = world_float_bits(&headless);
        for divisor in [1, 10, 60] {
            let rendered = run_with_render_divisor(Some(divisor));
            assert_eq!(
                headless, rendered,
                "state differs at render divisor {divisor}"
            );
            assert_eq!(
                headless_bits,
                world_float_bits(&rendered),
                "float to_bits state differs at render divisor {divisor}"
            );
        }
    }
}
