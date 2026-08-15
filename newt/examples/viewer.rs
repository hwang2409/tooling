//! Live newt showcase viewer.
//!
//! The simulation uses a fixed engine timestep. Rendering runs as fast as the
//! native window can present. Keyboard orbit is available through chimy2's
//! public input surface; the renderer keeps the camera state independent from
//! simulation stepping.

mod showcase_support;

use chimy2::fb::Framebuffer;
use chimy2::math::Vec3;
use chimy2::present::{InputState, run_with_input};
use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3 as NVec3};
use newt::mjcf::load_mjcf_path;
use newt::model::{Scene, load_from_path};
use newt::solver::{SolverConfig, SolverMode};
use newt::world::World;
use std::path::PathBuf;
use winit::keyboard::KeyCode;

struct Args {
    scenario: String,
    model: Option<PathBuf>,
    size: (u32, u32),
    frames: Option<usize>,
}

fn parse_args() -> Args {
    let mut args = Args {
        scenario: "walk".to_string(),
        model: None,
        size: (960, 640),
        frames: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(value) = it.next() {
        match value.as_str() {
            "--scenario" => args.scenario = it.next().expect("--scenario value"),
            "--model" => args.model = Some(PathBuf::from(it.next().expect("--model value"))),
            "--size" => {
                let value = it.next().expect("--size WxH");
                let (width, height) = value.split_once('x').expect("--size WxH");
                args.size = (width.parse().unwrap(), height.parse().unwrap());
            }
            "--frames" => args.frames = Some(it.next().unwrap().parse().unwrap()),
            _ => panic!("unknown argument: {value}"),
        }
    }
    args
}

struct Viewer {
    world: World,
    scenario: String,
    paused: bool,
    speed: f32,
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
    previous: InputState,
    step_accumulator: f32,
    steps: usize,
}

impl Viewer {
    fn edge(&mut self, input: &InputState, key: KeyCode) -> bool {
        let current = input.is_down(key);
        let previous = self.previous.is_down(key);
        current && !previous
    }

    fn update(&mut self, input: &InputState) {
        if self.edge(input, KeyCode::Space) {
            self.paused = !self.paused;
        }
        if self.paused && self.edge(input, KeyCode::Period) {
            self.world.step();
            self.steps += 1;
        }
        if self.edge(input, KeyCode::BracketLeft) {
            self.speed = (self.speed * 0.5).max(0.25);
        }
        if self.edge(input, KeyCode::BracketRight) {
            self.speed = (self.speed * 2.0).min(4.0);
        }
        if self.edge(input, KeyCode::Digit1) {
            self.yaw = 0.0;
            self.pitch = 0.18;
        }
        if self.edge(input, KeyCode::Digit2) {
            self.yaw = 1.2;
            self.pitch = 0.3;
        }
        if self.edge(input, KeyCode::Digit3) {
            self.yaw = 2.5;
            self.pitch = 0.12;
        }
        let orbit_step = 0.035;
        if input.is_down(KeyCode::KeyA) {
            self.yaw -= orbit_step;
        }
        if input.is_down(KeyCode::KeyD) {
            self.yaw += orbit_step;
        }
        if input.is_down(KeyCode::KeyW) {
            self.pitch = (self.pitch + orbit_step).min(1.2);
        }
        if input.is_down(KeyCode::KeyS) {
            self.pitch = (self.pitch - orbit_step).max(-0.4);
        }
        if !self.paused {
            self.step_accumulator += self.speed;
            let steps = self.step_accumulator.floor() as usize;
            self.step_accumulator -= steps as f32;
            for _ in 0..steps {
                self.world.step();
                self.steps += 1;
            }
        }
        self.previous = input.clone();
    }

    fn draw(&mut self, framebuffer: &mut Framebuffer, input: &InputState) {
        self.update(input);
        let items = showcase_support::world_items(&self.world);
        let position = self.target
            + Vec3::new(
                self.yaw.sin() * self.pitch.cos() * self.distance,
                self.yaw.cos() * self.pitch.cos() * self.distance,
                self.pitch.sin() * self.distance,
            );
        let composition =
            showcase_support::Composition::new(self.target, position, Vec3::new(0.1, 0.65, 0.95));
        let rendered = showcase_support::render_items(
            &items,
            composition,
            framebuffer.width,
            framebuffer.height,
            &format!(
                "{}  |  t={:.3}  step={}  speed={:.2}x  {}",
                self.scenario,
                self.steps as f32 * self.world.dt,
                self.steps,
                self.speed,
                if self.paused { "paused" } else { "running" },
            ),
        );
        *framebuffer = rendered;
    }
}

fn load_world(args: &Args) -> Result<(World, String), Box<dyn std::error::Error>> {
    if let Some(path) = &args.model {
        let scene = load_scene(path)?;
        return Ok((scene.world, path.display().to_string()));
    }
    let scenario = args.scenario.as_str();
    let scene = match scenario {
        "walk" => load_mjcf_path("models/biped-walk.xml")?,
        "stack" => load_from_path("models/stack.json")?,
        "pile" => load_from_path("models/pile.json")?,
        "arm" => load_from_path("models/arm.json")?,
        "cartpole" | "tendon" => fallback_scene(),
        other => return Err(format!("unknown scenario: {other}").into()),
    };
    Ok((scene.world, scenario.to_string()))
}

fn load_scene(path: &PathBuf) -> Result<Scene, Box<dyn std::error::Error>> {
    if path.extension().and_then(|value| value.to_str()) == Some("xml") {
        Ok(load_mjcf_path(path)?)
    } else {
        Ok(load_from_path(path)?)
    }
}

fn fallback_scene() -> Scene {
    let mut world = World::new();
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        ..SolverConfig::DEFAULT
    };
    world.add_geom(Geom::static_plane(NVec3::ZERO, NVec3::Z, 0.7));
    let body = world.add_body(Body::solid_box(
        1.0,
        NVec3::new(0.35, 0.35, 0.35),
        NVec3::new(0.0, 0.0, 1.0),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::r#box(
        body,
        NVec3::new(0.35, 0.35, 0.35),
        NVec3::ZERO,
        Quat::IDENTITY,
        0.7,
    ));
    let scene = newt::model::load_str(
        r#"{"version":"1","gravity":[0,0,-9.81],"timestep":0.005,"bodies":[],"geoms":[]}"#,
    )
    .expect("fallback scene shell");
    Scene { world, ..scene }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let (world, scenario) = load_world(&args)?;
    let mut viewer = Viewer {
        world,
        scenario,
        paused: false,
        speed: 1.0,
        yaw: 0.0,
        pitch: 0.24,
        distance: 5.0,
        target: Vec3::new(0.0, 0.0, 1.0),
        previous: InputState::default(),
        step_accumulator: 0.0,
        steps: 0,
    };
    run_with_input(
        "newt showcase viewer",
        args.size.0,
        args.size.1,
        args.frames,
        move |framebuffer, _, input| viewer.draw(framebuffer, input),
    )?;
    Ok(())
}
