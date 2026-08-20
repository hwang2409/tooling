use chimy2::camera::OrbitController;
use chimy2::demo::write_ppm;
use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::pipeline::Pipeline;
use chimy2::present::{
    FrameTiming, InputState, PresentKeyCode as KeyCode, PresentMouseButton as MouseButton,
    run_with_input, run_with_input_timed,
};
use chimy2::shaders::{FlatColorShader, FlatColorUniforms};
use newt::geom::{Geom, GeomAttach, GeomShape, geom_world_pose};
use newt::math::{Quat as NewtQuat, Vec3 as NewtVec3};
use newt::mjcf::load_mjcf_path;
use newt::model::{Scene as NewtScene, load_from_path};
use newt::world::World;
use std::cell::RefCell;
use std::f32::consts::{FRAC_PI_2, PI};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use std::time::Instant;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 640;
const RECORD_WIDTH: u32 = 640;
const RECORD_HEIGHT: u32 = 400;
const VIDEO_FPS: f32 = 60.0;
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
    previous_input: [bool; EDGE_KEYS.len()],
    metrics: Option<Rc<RefCell<Metrics>>>,
}

#[derive(Default)]
struct Metrics {
    sim_ms: Vec<f64>,
    render_ms: Vec<f64>,
    present_ms: Vec<f64>,
    frame_ms: Vec<f64>,
    last_present: Option<Instant>,
}

impl Metrics {
    fn record_draw(&mut self, sim: std::time::Duration, render: std::time::Duration) {
        self.sim_ms.push(sim.as_secs_f64() * 1000.0);
        self.render_ms.push(render.as_secs_f64() * 1000.0);
    }

    fn record_present(&mut self, timing: FrameTiming) {
        let now = Instant::now();
        self.present_ms.push(timing.present.as_secs_f64() * 1000.0);
        if let Some(previous) = self.last_present.replace(now) {
            self.frame_ms
                .push(now.duration_since(previous).as_secs_f64() * 1000.0);
        }
    }

    fn report(&self, seconds: f32) {
        println!(
            "measurement: {} frames over {:.1}s",
            self.frame_ms.len(),
            seconds
        );
        report_metric("frame", &self.frame_ms);
        report_metric("sim", &self.sim_ms);
        report_metric("render", &self.render_ms);
        report_metric("present", &self.present_ms);
        if let Some(avg) = average(&self.frame_ms) {
            println!("effective fps: {:.2}", 1000.0 / avg);
        }
    }
}

fn average(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn percentile(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index])
}

fn report_metric(name: &str, values: &[f64]) {
    let Some(avg) = average(values) else {
        println!("{name} ms: no samples");
        return;
    };
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let p99 = percentile(values, 0.99).unwrap_or(avg);
    println!("{name} ms: min={min:.3} avg={avg:.3} p99={p99:.3}");
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
    let mut vertices = Vec::with_capacity(hfield.nrow * hfield.ncol + 4);
    for row in 0..hfield.nrow {
        let y = -hfield.size[1] + 2.0 * hfield.size[1] * row as f32 / (hfield.nrow - 1) as f32;
        for col in 0..hfield.ncol {
            let x = -hfield.size[0] + 2.0 * hfield.size[0] * col as f32 / (hfield.ncol - 1) as f32;
            vertices.push(NewtVec3::new(x, y, hfield.height(row, col)));
        }
    }
    let mut triangles = Vec::with_capacity(
        (hfield.nrow - 1) * (hfield.ncol - 1) * 2
            + 2 * (2 * (hfield.nrow - 1) + 2 * (hfield.ncol - 1))
            + 2,
    );
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

    let base = vertices.len();
    vertices.extend([
        NewtVec3::new(-hfield.size[0], -hfield.size[1], -hfield.size[3]),
        NewtVec3::new(hfield.size[0], -hfield.size[1], -hfield.size[3]),
        NewtVec3::new(hfield.size[0], hfield.size[1], -hfield.size[3]),
        NewtVec3::new(-hfield.size[0], hfield.size[1], -hfield.size[3]),
    ]);
    triangles.push([base, base + 2, base + 1]);
    triangles.push([base, base + 3, base + 2]);

    for col in 0..(hfield.ncol - 1) {
        let a = col;
        let b = col + 1;
        triangles.push([a, base, base + 1]);
        triangles.push([a, base + 1, b]);
    }
    for row in 0..(hfield.nrow - 1) {
        let a = row * hfield.ncol + hfield.ncol - 1;
        let b = (row + 1) * hfield.ncol + hfield.ncol - 1;
        triangles.push([a, base + 2, base + 1]);
        triangles.push([a, b, base + 2]);
    }
    let top_row = (hfield.nrow - 1) * hfield.ncol;
    for col in 0..(hfield.ncol - 1) {
        let a = top_row + col;
        let b = a + 1;
        triangles.push([a, b, base + 2]);
        triangles.push([a, base + 2, base + 3]);
    }
    for row in 0..(hfield.nrow - 1) {
        let a = row * hfield.ncol;
        let b = (row + 1) * hfield.ncol;
        triangles.push([a, b, base + 3]);
        triangles.push([a, base + 3, base]);
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

const EDGE_KEYS: [KeyCode; 13] = [
    KeyCode::Space,
    KeyCode::KeyR,
    KeyCode::BracketLeft,
    KeyCode::Minus,
    KeyCode::BracketRight,
    KeyCode::Equal,
    KeyCode::KeyN,
    KeyCode::KeyP,
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
];

fn edge(current: &InputState, previous: &[bool; EDGE_KEYS.len()], key: KeyCode) -> bool {
    let index = EDGE_KEYS
        .iter()
        .position(|candidate| *candidate == key)
        .expect("edge key must be listed in EDGE_KEYS");
    current.is_down(key) && !previous[index]
}

impl Viewer {
    fn new(metrics: Option<Rc<RefCell<Metrics>>>) -> Result<Self, Box<dyn std::error::Error>> {
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
            previous_input: [false; EDGE_KEYS.len()],
            metrics,
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
        self.advance_simulation(frame_delta);
        self.previous_input = std::array::from_fn(|index| input.is_down(EDGE_KEYS[index]));
    }

    fn advance_simulation(&mut self, frame_delta: f32) {
        if !self.paused {
            self.accumulator += frame_delta * self.speed;
            while self.accumulator >= self.scene.world.dt {
                self.scene.world.step();
                self.accumulator -= self.scene.world.dt;
            }
        }
    }

    fn render_frame(&mut self, framebuffer: &mut Framebuffer) {
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

    fn draw(&mut self, framebuffer: &mut Framebuffer, elapsed: f32, input: &InputState) {
        let sim_started = self.metrics.as_ref().map(|_| Instant::now());
        self.update(elapsed, input);
        let simulation = sim_started.map(|started| started.elapsed());
        let render_started = self.metrics.as_ref().map(|_| Instant::now());
        self.render_frame(framebuffer);
        if let (Some(simulation), Some(render_started), Some(metrics)) =
            (simulation, render_started, &self.metrics)
        {
            metrics
                .borrow_mut()
                .record_draw(simulation, render_started.elapsed());
        }
    }

    fn record_frame(&mut self, frame: usize, framebuffer: &mut Framebuffer) {
        match frame {
            180 => self.paused = true,
            300 => self.paused = false,
            360 => self.speed = 0.5,
            540 => self.speed = 2.0,
            600 => self.select_scene(1),
            720 => self.paused = true,
            780 => self.paused = false,
            900 => self.select_scene(2),
            1050 => self.select_scene(3),
            1140 => self.select_scene(4),
            _ => {}
        }
        let phase = frame as f32 / VIDEO_FPS;
        self.orbit.step(0.012, (phase * 0.7).sin() * 0.004, 0.0);
        self.advance_simulation(1.0 / VIDEO_FPS);
        self.render_frame(framebuffer);
    }
}

struct VideoWriter {
    output: PathBuf,
    frame_dir: PathBuf,
    next_frame: usize,
}

impl VideoWriter {
    fn new(root: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        fs::create_dir_all(root)?;
        let frame_dir = root.join(format!("frames-{}", std::process::id()));
        fs::create_dir_all(&frame_dir)?;
        Ok(Self {
            output: root.join("newt-sandbox-demo.mp4"),
            frame_dir,
            next_frame: 0,
        })
    }

    fn push(&mut self, framebuffer: &Framebuffer) -> Result<(), Box<dyn std::error::Error>> {
        let path = self
            .frame_dir
            .join(format!("frame_{:05}.ppm", self.next_frame));
        write_ppm(path, framebuffer)?;
        self.next_frame += 1;
        Ok(())
    }

    fn finish(self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let input = self.frame_dir.join("frame_%05d.ppm");
        let result = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                "60",
                "-i",
                input.to_str().unwrap_or_default(),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                self.output.to_str().unwrap_or_default(),
            ])
            .output();
        let cleanup = fs::remove_dir_all(&self.frame_dir);
        match (result, cleanup) {
            (Err(error), _) if error.kind() == std::io::ErrorKind::NotFound => {
                Err("ffmpeg is required for video output".into())
            }
            (Err(error), _) => Err(error.into()),
            (Ok(output), _) if !output.status.success() => {
                Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr)).into())
            }
            (Ok(_), Err(error)) => Err(error.into()),
            (Ok(_), Ok(())) => Ok(self.output.clone()),
        }
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.frame_dir);
    }
}

fn record_demo(root: &Path, seconds: f32) -> Result<(), Box<dyn std::error::Error>> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("recording duration must be finite and positive".into());
    }
    let frames = (seconds * VIDEO_FPS).ceil() as usize;
    let mut viewer = Viewer::new(None)?;
    let mut writer = VideoWriter::new(root)?;
    let mut framebuffer = Framebuffer::new(RECORD_WIDTH as usize, RECORD_HEIGHT as usize);
    for frame in 0..frames {
        viewer.record_frame(frame, &mut framebuffer);
        writer.push(&framebuffer)?;
    }
    let output = writer.finish()?;
    println!(
        "wrote {} ({} frames at {} fps)",
        output.display(),
        frames,
        VIDEO_FPS
    );
    Ok(())
}

fn measure_viewer(seconds: f32) -> Result<(), Box<dyn std::error::Error>> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("measurement duration must be finite and positive".into());
    }
    let metrics = Rc::new(RefCell::new(Metrics::default()));
    let viewer_metrics = metrics.clone();
    let mut viewer = Viewer::new(Some(viewer_metrics))?;
    let observer_metrics = metrics.clone();
    run_with_input_timed(
        "newt sandbox viewer measurement",
        WIDTH,
        HEIGHT,
        seconds,
        move |framebuffer, elapsed, input| viewer.draw(framebuffer, elapsed, input),
        move |timing| observer_metrics.borrow_mut().record_present(timing),
    )?;
    metrics.borrow().report(seconds);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, root, seconds] if flag == "--record" => {
            return record_demo(Path::new(root), seconds.parse()?);
        }
        [flag, seconds] if flag == "--measure" => {
            return measure_viewer(seconds.parse()?);
        }
        [] => {}
        _ => return Err("usage: sandbox [--record DIR SECONDS | --measure SECONDS]".into()),
    }
    let mut viewer = Viewer::new(None)?;
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

    #[test]
    fn hfield_render_geometry_includes_collision_base_and_sides() {
        let hfield = newt::geom::HeightField {
            nrow: 2,
            ncol: 2,
            size: [2.0, 3.0, 4.0, 5.0],
            data: vec![0.0; 4],
        };

        let (vertices, triangles) = hfield_mesh(&hfield);

        assert_eq!(vertices.len(), 8);
        assert_eq!(triangles.len(), 12);
        assert!(vertices[4..].iter().all(|vertex| vertex.z == -5.0));
    }
}
