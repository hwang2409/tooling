use chimy2::camera::OrbitController;
use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::pipeline::Pipeline;
use chimy2::present::{
    InputState, PresentKeyCode as KeyCode, PresentMouseButton as MouseButton, run_with_input,
};
use chimy2::shaders::{FlatColorShader, FlatColorUniforms};
use newt::geom::{Geom, GeomAttach, GeomShape, geom_world_pose};
use newt::math::{Quat as NewtQuat, Vec3 as NewtVec3};
use newt::mjcf::load_mjcf_path;
use newt::model::{Scene as NewtScene, load_from_path};
use newt::world::World;
use std::f32::consts::{FRAC_PI_2, PI};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 640;
const SCENE_COUNT: usize = 5;

const PALETTE: [u32; 8] = [
    0xffe76f51, 0xff2a9d8f, 0xffe9c46a, 0xff457b9d, 0xfff4a261, 0xff8ab17d, 0xffb56576, 0xff6d597a,
];

#[derive(Clone, Debug)]
struct Shape {
    vertices: Vec<Vec4>,
    triangles: Vec<[usize; 3]>,
    geom_index: usize,
    color: u32,
}

struct SceneData {
    world: World,
    initial_world: World,
    shapes: Vec<Shape>,
    target: Vec3,
    distance: f32,
}

struct Viewer {
    scene_index: usize,
    scene: SceneData,
    pipeline: Pipeline<FlatColorShader, FlatColorShader>,
    uniforms: Vec<FlatColorUniforms>,
    orbit: OrbitController,
    paused: bool,
    speed: f32,
    accumulator: f32,
    last_elapsed: f32,
    previous_input: InputState,
    steps: u64,
}

fn map_vector(value: NewtVec3) -> Vec3 {
    // newt is Z-up. chimy2's orbit controller is Y-up.
    Vec3::new(value.x, value.z, -value.y)
}

fn newt_quaternion_matrix(value: NewtQuat) -> Mat4 {
    let q = value.renormalize();
    // B * R(newt) maps the newt-local vector into the renderer basis.
    Mat4::rotate(Vec3::new(1.0, 0.0, 0.0), -FRAC_PI_2) * chimy_quaternion_matrix(q)
}

fn chimy_quaternion_matrix(value: NewtQuat) -> Mat4 {
    let (x, y, z, w) = (value.x, value.y, value.z, value.w);
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

fn geom_pose(world: &World, geom: &Geom) -> newt::geom::GeomPose {
    match geom.attachment() {
        GeomAttach::Static => geom_world_pose(geom, NewtVec3::ZERO, NewtQuat::IDENTITY),
        GeomAttach::Body(index) => {
            let body = world.bodies[index];
            geom_world_pose(geom, body.position, body.orientation)
        }
        GeomAttach::Link(tree, link) => {
            let (position, orientation) = world.tree_link_pose(tree, link);
            geom_world_pose(geom, position, orientation)
        }
    }
}

fn mesh(vertices: Vec<NewtVec3>, triangles: Vec<[usize; 3]>) -> (Vec<Vec4>, Vec<[usize; 3]>) {
    (
        vertices
            .into_iter()
            .map(|value| Vec4::new(value.x, value.y, value.z, 1.0))
            .collect(),
        triangles,
    )
}

fn box_mesh(half: NewtVec3) -> (Vec<Vec4>, Vec<[usize; 3]>) {
    let vertices = vec![
        NewtVec3::new(-half.x, -half.y, -half.z),
        NewtVec3::new(half.x, -half.y, -half.z),
        NewtVec3::new(half.x, half.y, -half.z),
        NewtVec3::new(-half.x, half.y, -half.z),
        NewtVec3::new(-half.x, -half.y, half.z),
        NewtVec3::new(half.x, -half.y, half.z),
        NewtVec3::new(half.x, half.y, half.z),
        NewtVec3::new(-half.x, half.y, half.z),
    ];
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
    mesh(vertices, triangles)
}

fn sphere_mesh(radii: NewtVec3, rings: usize, segments: usize) -> (Vec<Vec4>, Vec<[usize; 3]>) {
    let mut vertices = Vec::with_capacity((rings + 1) * segments);
    for ring in 0..=rings {
        let theta = PI * ring as f32 / rings as f32;
        let z = theta.cos();
        let radial = theta.sin();
        for segment in 0..segments {
            let phi = 2.0 * PI * segment as f32 / segments as f32;
            vertices.push(NewtVec3::new(
                radii.x * radial * phi.cos(),
                radii.y * radial * phi.sin(),
                radii.z * z,
            ));
        }
    }
    let mut triangles = Vec::with_capacity(rings * segments * 2);
    for ring in 0..rings {
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            let a = ring * segments + segment;
            let b = ring * segments + next;
            let c = (ring + 1) * segments + next;
            let d = (ring + 1) * segments + segment;
            triangles.push([a, c, b]);
            triangles.push([a, d, c]);
        }
    }
    mesh(vertices, triangles)
}

fn capsule_mesh(radius: f32, half_height: f32) -> (Vec<Vec4>, Vec<[usize; 3]>) {
    let rings = 16;
    let segments = 12;
    let total = half_height + radius;
    let mut vertices = Vec::with_capacity((rings + 1) * segments);
    for ring in 0..=rings {
        let z = -total + 2.0 * total * ring as f32 / rings as f32;
        let center = if z < -half_height {
            -half_height
        } else if z > half_height {
            half_height
        } else {
            z
        };
        let radial = (radius * radius - (z - center) * (z - center))
            .max(0.0)
            .sqrt();
        for segment in 0..segments {
            let phi = 2.0 * PI * segment as f32 / segments as f32;
            vertices.push(NewtVec3::new(radial * phi.cos(), radial * phi.sin(), z));
        }
    }
    let mut triangles = Vec::with_capacity(rings * segments * 2);
    for ring in 0..rings {
        for segment in 0..segments {
            let next = (segment + 1) % segments;
            let a = ring * segments + segment;
            let b = ring * segments + next;
            let c = (ring + 1) * segments + next;
            let d = (ring + 1) * segments + segment;
            triangles.push([a, c, b]);
            triangles.push([a, d, c]);
        }
    }
    mesh(vertices, triangles)
}

fn cylinder_mesh(radius: f32, half_height: f32) -> (Vec<Vec4>, Vec<[usize; 3]>) {
    let segments = 16;
    let mut vertices = Vec::with_capacity(segments * 2 + 2);
    vertices.push(NewtVec3::new(0.0, 0.0, -half_height));
    vertices.push(NewtVec3::new(0.0, 0.0, half_height));
    for z in [-half_height, half_height] {
        for segment in 0..segments {
            let phi = 2.0 * PI * segment as f32 / segments as f32;
            vertices.push(NewtVec3::new(radius * phi.cos(), radius * phi.sin(), z));
        }
    }
    let mut triangles = Vec::with_capacity(segments * 4);
    for segment in 0..segments {
        let next = (segment + 1) % segments;
        let lower = 2 + segment;
        let lower_next = 2 + next;
        let upper = 2 + segments + segment;
        let upper_next = 2 + segments + next;
        triangles.push([0, lower_next, lower]);
        triangles.push([1, upper, upper_next]);
        triangles.push([lower, lower_next, upper_next]);
        triangles.push([lower, upper_next, upper]);
    }
    mesh(vertices, triangles)
}

fn plane_mesh() -> (Vec<Vec4>, Vec<[usize; 3]>) {
    box_mesh(NewtVec3::new(10.0, 10.0, 0.01))
}

fn hfield_mesh(hfield: &newt::geom::HeightField) -> (Vec<Vec4>, Vec<[usize; 3]>) {
    let mut vertices = Vec::with_capacity(hfield.nrow * hfield.ncol);
    for row in 0..hfield.nrow {
        let y = -hfield.size[1] + 2.0 * hfield.size[1] * row as f32 / (hfield.nrow - 1) as f32;
        for col in 0..hfield.ncol {
            let x = -hfield.size[0] + 2.0 * hfield.size[0] * col as f32 / (hfield.ncol - 1) as f32;
            vertices.push(NewtVec3::new(x, y, hfield.height(row, col)));
        }
    }
    let mut triangles = Vec::with_capacity((hfield.nrow - 1) * (hfield.ncol - 1) * 2);
    for row in 0..hfield.nrow - 1 {
        for col in 0..hfield.ncol - 1 {
            let a = row * hfield.ncol + col;
            let b = a + 1;
            let c = a + hfield.ncol + 1;
            let d = a + hfield.ncol;
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
    }
    mesh(vertices, triangles)
}

fn shape_mesh(world: &World, shape: GeomShape) -> (Vec<Vec4>, Vec<[usize; 3]>) {
    match shape {
        GeomShape::Plane => plane_mesh(),
        GeomShape::Sphere { radius } => sphere_mesh(NewtVec3::new(radius, radius, radius), 10, 16),
        GeomShape::Box { half_extents } => box_mesh(half_extents),
        GeomShape::Capsule {
            radius,
            half_height,
        } => capsule_mesh(radius, half_height),
        GeomShape::Cylinder {
            radius,
            half_height,
        } => cylinder_mesh(radius, half_height),
        GeomShape::Ellipsoid { semi_axes } => sphere_mesh(semi_axes, 10, 16),
        GeomShape::Mesh { mesh_id } => {
            let asset = &world.meshes[mesh_id];
            mesh(
                asset.vertices.clone(),
                asset
                    .faces
                    .iter()
                    .map(|face| [face[0] as usize, face[1] as usize, face[2] as usize])
                    .collect(),
            )
        }
        GeomShape::Hfield { hfield_id } => hfield_mesh(&world.hfields[hfield_id]),
    }
}

fn model_matrix(pose: newt::geom::GeomPose) -> Mat4 {
    Mat4::translate(map_vector(pose.position)) * newt_quaternion_matrix(pose.orientation)
}

fn scene_bounds(world: &World, shapes: &[Shape]) -> (Vec3, f32) {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for shape in shapes {
        if world.geoms[shape.geom_index].shape == GeomShape::Plane {
            continue;
        }
        let transform = model_matrix(geom_pose(world, &world.geoms[shape.geom_index]));
        for vertex in &shape.vertices {
            let value = transform * *vertex;
            min.x = min.x.min(value.x);
            min.y = min.y.min(value.y);
            min.z = min.z.min(value.z);
            max.x = max.x.max(value.x);
            max.y = max.y.max(value.y);
            max.z = max.z.max(value.z);
        }
    }
    if !min.x.is_finite() {
        return (Vec3::ZERO, 5.0);
    }
    let target = (min + max) * 0.5;
    let extent = (max - min).length().max(1.0);
    (target, extent * 1.35)
}

fn load_scene(index: usize) -> Result<SceneData, Box<dyn std::error::Error>> {
    let scene: NewtScene = match index % SCENE_COUNT {
        0 => load_mjcf_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../newt/models/biped-simple.xml"
        ))?,
        1 => load_mjcf_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../newt/models/pendulum.xml"
        ))?,
        2 => load_from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../newt/models/arm.json"
        ))?,
        3 => load_mjcf_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../newt/models/stack.xml"
        ))?,
        4 => load_from_path(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../newt/models/pile.json"
        ))?,
        _ => unreachable!(),
    };
    let world = scene.world;
    let mut shapes = Vec::with_capacity(world.geoms.len());
    for (geom_index, geom) in world.geoms.iter().enumerate() {
        let (vertices, triangles) = shape_mesh(&world, geom.shape);
        shapes.push(Shape {
            vertices,
            triangles,
            geom_index,
            color: if geom.shape == GeomShape::Plane {
                0xff58616b
            } else {
                PALETTE[geom_index % PALETTE.len()]
            },
        });
    }
    let (target, distance) = scene_bounds(&world, &shapes);
    Ok(SceneData {
        initial_world: world.clone(),
        world,
        shapes,
        target,
        distance,
    })
}

fn edge(current: &InputState, previous: &InputState, key: KeyCode) -> bool {
    current.is_down(key) && !previous.is_down(key)
}

impl Viewer {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let scene = load_scene(0)?;
        let uniforms = scene
            .shapes
            .iter()
            .map(|_| FlatColorUniforms::new(Mat4::IDENTITY, 0))
            .collect();
        let orbit = OrbitController::new(scene.target, scene.distance, 0.0, 0.22);
        Ok(Self {
            scene_index: 0,
            scene,
            pipeline: Pipeline::new(FlatColorShader, FlatColorShader),
            uniforms,
            orbit,
            paused: false,
            speed: 1.0,
            accumulator: 0.0,
            last_elapsed: 0.0,
            previous_input: InputState::default(),
            steps: 0,
        })
    }

    fn select_scene(&mut self, index: usize) {
        match load_scene(index) {
            Ok(scene) => {
                self.scene_index = index % SCENE_COUNT;
                self.orbit = OrbitController::new(scene.target, scene.distance, 0.0, 0.22);
                self.uniforms = scene
                    .shapes
                    .iter()
                    .map(|_| FlatColorUniforms::new(Mat4::IDENTITY, 0))
                    .collect();
                self.scene = scene;
                self.accumulator = 0.0;
                self.steps = 0;
            }
            Err(error) => eprintln!("cannot load scene {index}: {error}"),
        }
    }

    fn update(&mut self, elapsed: f32, input: &InputState) {
        if edge(input, &self.previous_input, KeyCode::Space) {
            self.paused = !self.paused;
        }
        if edge(input, &self.previous_input, KeyCode::KeyR) {
            self.scene.world = self.scene.initial_world.clone();
            self.accumulator = 0.0;
            self.steps = 0;
        }
        if edge(input, &self.previous_input, KeyCode::BracketLeft)
            || edge(input, &self.previous_input, KeyCode::Minus)
        {
            self.speed = (self.speed * 0.5).max(0.25);
        }
        if edge(input, &self.previous_input, KeyCode::BracketRight)
            || edge(input, &self.previous_input, KeyCode::Equal)
        {
            self.speed = (self.speed * 2.0).min(4.0);
        }
        if edge(input, &self.previous_input, KeyCode::KeyN) {
            self.select_scene(self.scene_index + 1);
        }
        if edge(input, &self.previous_input, KeyCode::KeyP) {
            self.select_scene((self.scene_index + SCENE_COUNT - 1) % SCENE_COUNT);
        }
        for (key, index) in [
            (KeyCode::Digit1, 0),
            (KeyCode::Digit2, 1),
            (KeyCode::Digit3, 2),
            (KeyCode::Digit4, 3),
            (KeyCode::Digit5, 4),
        ] {
            if edge(input, &self.previous_input, key) {
                self.select_scene(index);
            }
        }

        if input.is_mouse_down(MouseButton::Left) {
            let (dx, dy) = input.mouse_delta();
            self.orbit.step(-dx * 0.008, -dy * 0.008, 0.0);
        }
        self.orbit.step(0.0, 0.0, -input.scroll_delta() * 0.01);

        let frame_delta = (elapsed - self.last_elapsed).clamp(0.0, 0.1);
        self.last_elapsed = elapsed;
        if !self.paused {
            self.accumulator += frame_delta * self.speed;
            while self.accumulator >= self.scene.world.dt {
                self.scene.world.step();
                self.accumulator -= self.scene.world.dt;
                self.steps += 1;
            }
        }
        self.previous_input = input.clone();
    }

    fn draw(&mut self, framebuffer: &mut Framebuffer, elapsed: f32, input: &InputState) {
        self.update(elapsed, input);
        framebuffer.clear(0xff101820);
        let camera = self.orbit.camera(
            0.85,
            framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32,
            0.03,
            1000.0,
        );
        let transform = camera.view_projection();
        for (uniform, shape) in self.uniforms.iter_mut().zip(&self.scene.shapes) {
            let pose = geom_pose(&self.scene.world, &self.scene.world.geoms[shape.geom_index]);
            uniform.transform = transform * model_matrix(pose);
            uniform.color = shape.color;
        }
        let shapes = &self.scene.shapes;
        let uniforms = &self.uniforms;
        self.pipeline.render(framebuffer, |frame, target| {
            for (shape, uniform) in shapes.iter().zip(uniforms) {
                frame.draw(target, &shape.vertices, &shape.triangles, uniform);
            }
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut viewer = Viewer::new()?;
    run_with_input(
        "newt sandbox viewer",
        WIDTH,
        HEIGHT,
        None,
        move |framebuffer, elapsed, input| viewer.draw(framebuffer, elapsed, input),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_built_in_scene_builds_render_geometry() {
        for index in 0..SCENE_COUNT {
            let scene = load_scene(index).expect("built-in scene should load");
            assert_eq!(scene.shapes.len(), scene.world.geoms.len());
            assert!(scene.shapes.iter().all(|shape| !shape.vertices.is_empty()));
            assert!(scene.distance.is_finite() && scene.distance > 0.0);
        }
    }
}
