use std::env;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use newt::mjcf::load_mjcf_path;
use newt::model::load_from_path;
use newt::solver::{ConeKind, SolverMode};
use newt::world::Integrator;

#[path = "../examples/biped_walk_support.rs"]
mod biped_walk_support;

const DEFAULT_ITERATIONS: usize = 15;
const DEFAULT_WARMUP: usize = 3;
const DEFAULT_STEPS: usize = 1_000;

#[derive(Clone, Copy)]
struct Config {
    name: &'static str,
    integrator: Integrator,
    solver: SolverMode,
}

const CONFIGS: [Config; 3] = [
    Config {
        name: "penalty-rk4",
        integrator: Integrator::Rk4,
        solver: SolverMode::Penalty,
    },
    Config {
        name: "pgs-euler",
        integrator: Integrator::Euler,
        solver: SolverMode::Pgs,
    },
    Config {
        name: "newton-euler",
        integrator: Integrator::Euler,
        solver: SolverMode::Newton,
    },
];

#[derive(Clone, Copy)]
struct SceneSpec {
    name: &'static str,
    path: &'static str,
    format: SceneFormat,
}

#[derive(Clone, Copy)]
enum SceneFormat {
    Mjcf,
    Json,
}

const SCENES: [SceneSpec; 7] = [
    SceneSpec {
        name: "sphere_drop",
        path: "tests/references/sphere_drop.xml",
        format: SceneFormat::Mjcf,
    },
    SceneSpec {
        name: "box_stack",
        path: "tests/references/box_stack.xml",
        format: SceneFormat::Mjcf,
    },
    SceneSpec {
        name: "biped_assisted_walk",
        path: "models/biped-walk.xml",
        format: SceneFormat::Mjcf,
    },
    SceneSpec {
        name: "tendon_arm",
        path: "tests/references/tendon_mixed_wrap.xml",
        format: SceneFormat::Mjcf,
    },
    SceneSpec {
        name: "hfield_terrain_roll",
        path: "tests/references/hfield_sphere_ramp.xml",
        format: SceneFormat::Mjcf,
    },
    SceneSpec {
        name: "pile",
        path: "models/pile.json",
        format: SceneFormat::Json,
    },
    SceneSpec {
        name: "muscle_pendulum",
        path: "tests/references/muscle_pendulum.xml",
        format: SceneFormat::Mjcf,
    },
];

struct Options {
    iterations: usize,
    warmup: usize,
    steps: usize,
    scene: Option<String>,
    profile: bool,
}

#[derive(Default)]
struct ProfileTotals {
    total_ns: u128,
    collision_ns: u128,
    solver_ns: u128,
    integration_ns: u128,
    sensors_ns: u128,
}

impl ProfileTotals {
    #[cfg(feature = "instrumentation")]
    fn add(&mut self, timings: newt::world::StepTimings) {
        self.total_ns += timings.total_ns;
        self.collision_ns += timings.collision_ns;
        self.solver_ns += timings.solver_ns;
        self.integration_ns += timings.integration_ns;
        self.sensors_ns += timings.sensors_ns;
    }
}

struct Stats {
    samples_ns: Vec<u128>,
}

impl Stats {
    fn new(mut samples_ns: Vec<u128>) -> Self {
        samples_ns.sort_unstable();
        Self { samples_ns }
    }

    fn percentile_ns(&self, fraction: f64) -> u128 {
        let index = (fraction * (self.samples_ns.len() - 1) as f64).floor() as usize;
        self.samples_ns[index]
    }

    fn median_ns(&self) -> u128 {
        self.percentile_ns(0.5)
    }

    fn steps_per_second(&self, steps: usize) -> f64 {
        steps as f64 * 1_000_000_000.0 / self.median_ns() as f64
    }
}

fn main() {
    let options = parse_options();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut results = Vec::new();

    for scene in SCENES {
        if options
            .scene
            .as_deref()
            .is_some_and(|wanted| wanted != scene.name)
        {
            continue;
        }
        for config in CONFIGS {
            let result = benchmark(
                &root,
                scene,
                config,
                options.iterations,
                options.warmup,
                options.steps,
            );
            println!(
                "{{\"scene\":\"{}\",\"config\":\"{}\",\"steps\":{},\"warmup\":{},\"iterations\":{},\"median_ns\":{},\"p10_ns\":{},\"p90_ns\":{},\"ns_per_step\":{},\"steps_per_sec\":{:.3},\"checksum\":{:.9}}}",
                scene.name,
                config.name,
                options.steps,
                options.warmup,
                options.iterations,
                result.stats.median_ns(),
                result.stats.percentile_ns(0.1),
                result.stats.percentile_ns(0.9),
                result.stats.median_ns() / options.steps as u128,
                result.stats.steps_per_second(options.steps),
                result.checksum,
            );
            results.push((scene.name, config.name, result));
        }
    }

    eprintln!(
        "scene                 config          median/step   p10/step   p90/step   steps/sec"
    );
    eprintln!(
        "--------------------  --------------  ------------  ---------  ---------  ---------"
    );
    for (scene, config, result) in results {
        let stats = &result.stats;
        eprintln!(
            "{scene:<20}  {config:<14}  {:>10.1} ns  {:>7.1} ns  {:>7.1} ns  {:>9.1}",
            stats.percentile_ns(0.5) as f64 / options.steps as f64,
            stats.percentile_ns(0.1) as f64 / options.steps as f64,
            stats.percentile_ns(0.9) as f64 / options.steps as f64,
            stats.steps_per_second(options.steps),
        );
        if options.profile {
            print_profile(scene, config, &result.profile);
        }
    }
}

struct ResultRow {
    stats: Stats,
    checksum: f32,
    profile: ProfileTotals,
}

fn benchmark(
    root: &Path,
    scene: SceneSpec,
    config: Config,
    iterations: usize,
    warmup: usize,
    steps: usize,
) -> ResultRow {
    assert!(iterations > 0, "--iterations must be positive");
    let path = root.join(scene.path);
    if scene.name == "biped_assisted_walk" {
        let run = || {
            biped_walk_support::run_walk_with_solver(
                biped_walk_support::GaitConfig::stable_joint_walk(steps),
                config.integrator,
                config.solver,
            )
        };
        for _ in 0..warmup {
            black_box(run());
        }
        let mut samples_ns = Vec::with_capacity(iterations);
        let mut checksum = 0.0;
        for _ in 0..iterations {
            let start = Instant::now();
            let result = black_box(run());
            samples_ns.push(start.elapsed().as_nanos());
            checksum += result.final_root_height + result.final_forward_speed;
        }
        return ResultRow {
            stats: Stats::new(samples_ns),
            checksum,
            profile: ProfileTotals::default(),
        };
    }

    let initial = load_scene(scene.format, &path);
    let mut warmup_state = configure(initial.clone(), config);
    let mut warmup_profile = ProfileTotals::default();
    for _ in 0..warmup {
        run_world(&mut warmup_state, steps, &mut warmup_profile);
    }

    let mut samples_ns = Vec::with_capacity(iterations);
    let mut checksum = 0.0;
    let mut profile = ProfileTotals::default();
    for _ in 0..iterations {
        let mut state = configure(initial.clone(), config);
        let start = Instant::now();
        checksum += run_world(&mut state, steps, &mut profile);
        samples_ns.push(start.elapsed().as_nanos());
    }
    ResultRow {
        stats: Stats::new(samples_ns),
        checksum,
        profile,
    }
}

fn load_scene(format: SceneFormat, path: &Path) -> newt::model::Scene {
    match format {
        SceneFormat::Mjcf => {
            load_mjcf_path(path).unwrap_or_else(|error| panic!("{path:?}: {error}"))
        }
        SceneFormat::Json => {
            load_from_path(path).unwrap_or_else(|error| panic!("{path:?}: {error}"))
        }
    }
}

fn configure(mut scene: newt::model::Scene, config: Config) -> newt::world::World {
    scene.world.integrator = config.integrator;
    scene.world.solver.mode = config.solver;
    if config.solver == SolverMode::Newton {
        scene.world.solver.cone = ConeKind::Pyramidal;
    }
    scene.world
}

fn run_world(world: &mut newt::world::World, steps: usize, profile: &mut ProfileTotals) -> f32 {
    #[cfg(not(feature = "instrumentation"))]
    let _ = profile;
    for _ in 0..steps {
        world.step();
        #[cfg(feature = "instrumentation")]
        profile.add(world.step_timings());
    }
    let body_sum: f32 = world
        .bodies
        .iter()
        .map(|body| body.position.x + body.position.y + body.position.z)
        .sum();
    let tree_sum: f32 = world
        .trees
        .iter()
        .flat_map(|tree| tree.q.iter().chain(tree.qdot.iter()))
        .copied()
        .sum();
    black_box(body_sum + tree_sum)
}

fn parse_options() -> Options {
    let mut options = Options {
        iterations: DEFAULT_ITERATIONS,
        warmup: DEFAULT_WARMUP,
        steps: DEFAULT_STEPS,
        scene: None,
        profile: false,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {
                let _ = args.next();
            }
            "--iterations" => options.iterations = next_usize(&mut args, "--iterations"),
            "--warmup" => options.warmup = next_usize(&mut args, "--warmup"),
            "--steps" => options.steps = next_usize(&mut args, "--steps"),
            "--scene" => options.scene = Some(args.next().expect("--scene needs a name")),
            "--profile" => options.profile = true,
            "--help" => {
                println!(
                    "usage: newt_perf [--scene NAME] [--steps N] [--warmup N] [--iterations N] [--profile]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    options
}

fn print_profile(scene: &str, config: &str, profile: &ProfileTotals) {
    if profile.total_ns == 0 {
        eprintln!("profile unavailable: rebuild with --features instrumentation");
        return;
    }
    let pct = |value: u128| value as f64 * 100.0 / profile.total_ns as f64;
    eprintln!(
        "profile {scene} {config}: collision={:.1}% solver={:.1}% integration={:.1}% sensors={:.1}%",
        pct(profile.collision_ns),
        pct(profile.solver_ns),
        pct(profile.integration_ns),
        pct(profile.sensors_ns),
    );
}

fn next_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> usize {
    args.next()
        .unwrap_or_else(|| panic!("{flag} needs a value"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} needs a positive integer"))
}
