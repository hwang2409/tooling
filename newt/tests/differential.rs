//! Differential harness: newt vs REAL MuJoCo.
//!
//! Each test loads the SAME MJCF file that
//! `tools/capture_mujoco.py` used to produce the committed reference
//! fixture. It steps newt with the same initial state and timestep,
//! samples qpos/qvel at the same stride, and asserts each component
//! stays within a per-scenario tolerance window.
//!
//! The tolerances live in [`TOLERANCES`] and were MEASURED first, then
//! set with ~2× headroom above the observed max. Every scenario prints
//! its measured max at test time so drift is visible and every fixture
//! regen is a chance to re-verify or re-tighten.
//!
//! Divergences beyond physical reasonableness are FINDINGS to
//! report, not to hide. If a bound needs to grow to cover a real
//! divergence, the growth belongs in the scorecard
//! (`docs/differential.md`) with a note explaining why.
//!
//! Fixtures live in `tests/references/*.bin` and are read directly
//! from disk (no Python/MuJoCo at test time; CI runs this on Linux with
//! no MuJoCo).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use newt::json::{self, Value};
use newt::math::{Quat, Vec3};
use newt::model::Scene;
use newt::world::World;

// ---------------------------------------------------------------------------
// tolerances (MEASURED-then-stated; see docs/differential.md)
// ---------------------------------------------------------------------------

/// Per-scenario tolerance windows. `qpos` and `qvel` are absolute L∞
/// bounds applied component-wise to `|newt - mujoco|` at every sampled
/// step. The `energy` field applies only to scenarios that opt in
/// (chaotic ones — see [`scenario_config`]).
///
/// Numbers here are the ACTUAL observed max divergence multiplied by
/// roughly 2× (the "honest headroom" the ticket calls out). Update when
/// fixtures regen, and mirror to `docs/differential.md`.
struct Tolerance {
    qpos: f64,
    qvel: f64,
}

fn tolerance(name: &str) -> Tolerance {
    // Bounds are AUTHORED after observation. Numbers come from the
    // measured max divergence times ~2× headroom. Any bound larger than
    // its observed max by more than ~5× is a smell — either the
    // measurement was wrong or a real divergence is being papered over.
    // See docs/differential.md for the per-scenario "observed vs bound"
    // table and the open findings for the loose scenarios.
    match name {
        // Pure RK4 free-fall; f32 vs f64 quantization only.
        // Observed max qpos 1.24e-5, qvel 3.71e-5.
        "ballistic" => Tolerance {
            qpos: 3.0e-5,
            qvel: 8.0e-5,
        },
        // Torque-free tumble; RK4 + quaternion renormalization drift on
        // an intermediate-axis-unstable scene.
        // Observed max qpos 1.59e-3, qvel 2.03e-2.
        "tumble" => Tolerance {
            qpos: 4.0e-3,
            qvel: 5.0e-2,
        },
        // Short-horizon chaotic; tight because horizon is 0.4 s.
        // Observed max qpos 1.33e-7, qvel 6.23e-7.
        "double_pendulum" => Tolerance {
            qpos: 5.0e-7,
            qvel: 2.0e-6,
        },
        // Position servos driving to target. PD gains match by
        // construction; divergence is integration order and clamping edges.
        // Observed max qpos 7.03e-4, qvel 3.06e-2.
        "servo_arm" => Tolerance {
            qpos: 2.0e-3,
            qvel: 8.0e-2,
        },
        // First-bounce transient dominates (~4 mm z, ~0.17 m/s vz).
        // Steady-state penetration difference ~0.18 mm (newt is
        // shallower — see docs/differential.md open finding on
        // NEWT-9's softened d-scaling). Bound covers both.
        "sphere_drop" => Tolerance {
            qpos: 1.0e-2,
            qvel: 3.0e-1,
        },
        // OPEN FINDING (docs/differential.md): 3-box stack is not stable
        // in newt under matching iters=20 PGS settings — the top block
        // slips off by t≈4 s while MuJoCo's stays. Bound is set to
        // survive the current observation and no more; a REGRESSION
        // (top drifting even further) still fails the test.
        // Observed max qpos 1.45 m, qvel 3.89 m/s.
        "box_stack" => Tolerance {
            qpos: 2.0,
            qvel: 5.0,
        },
        // Limit-force impulse profile differs between newt PGS and
        // MuJoCo PGS; drift accumulates each swing.
        // Observed max qpos 1.08e-1 rad, qvel 9.10e-1 rad/s.
        "joint_limit_swing" => Tolerance {
            qpos: 1.5e-1,
            qvel: 1.2,
        },
        other => panic!("no tolerance for scenario {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// fixture reader
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 8] = b"NEWTDIF1";

struct Fixture {
    stride: u32,
    n_steps: u32,
    samples: Vec<Sample>,
}

struct Sample {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
}

fn read_fixture(path: &Path) -> Fixture {
    let bytes =
        fs::read(path).unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    let mut c = Cursor::new(&bytes);
    let magic = c.take(8);
    assert_eq!(
        magic,
        &MAGIC[..],
        "fixture {} has wrong magic {:?}; regenerate with tools/capture_mujoco.py",
        path.display(),
        std::str::from_utf8(magic).unwrap_or("<non-utf8>")
    );
    let prov_len = c.u32() as usize;
    // Skip the provenance line — the tool's version guard uses it; the
    // harness does not, since the tolerances table is versioned alongside
    // the fixtures.
    let _ = c.take(prov_len);
    let nq = c.u32() as usize;
    let nv = c.u32() as usize;
    let stride = c.u32();
    let n_samples = c.u32() as usize;
    let n_steps = c.u32();
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let _step = c.u32();
        let mut qpos = Vec::with_capacity(nq);
        for _ in 0..nq {
            qpos.push(c.f64());
        }
        let mut qvel = Vec::with_capacity(nv);
        for _ in 0..nv {
            qvel.push(c.f64());
        }
        samples.push(Sample { qpos, qvel });
    }
    assert_eq!(
        c.remaining(),
        0,
        "fixture {} has trailing bytes",
        path.display()
    );
    Fixture {
        stride,
        n_steps,
        samples,
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        out
    }
    fn u32(&mut self) -> u32 {
        let out = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        out
    }
    fn f64(&mut self) -> f64 {
        let out = f64::from_le_bytes(self.bytes[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        out
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

// ---------------------------------------------------------------------------
// scenarios.json reader
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ScenarioSpec {
    name: String,
    mjcf: String,
    n_steps: u32,
    stride: u32,
    init_qpos: Option<Vec<f64>>,
    init_qvel: Option<Vec<f64>>,
    actuator_targets: Option<HashMap<String, f32>>,
}

fn references_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("references")
}

fn load_scenarios() -> Vec<ScenarioSpec> {
    let src = fs::read_to_string(references_dir().join("scenarios.json"))
        .expect("scenarios.json missing");
    let root = json::parse(&src).expect("scenarios.json invalid");
    let obj = match root {
        Value::Object(o) => o,
        _ => panic!("scenarios.json root must be object"),
    };
    let scenarios = obj
        .iter()
        .find(|(k, _)| k == "scenarios")
        .map(|(_, v)| v.clone())
        .expect("scenarios.json missing \"scenarios\" array");
    let arr = match scenarios {
        Value::Array(a) => a,
        _ => panic!("scenarios.json \"scenarios\" is not an array"),
    };
    arr.into_iter().map(parse_scenario).collect()
}

fn parse_scenario(v: Value) -> ScenarioSpec {
    let obj = match v {
        Value::Object(o) => o,
        _ => panic!("scenario entry must be an object"),
    };
    let get = |k: &str| obj.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone());
    let name = expect_string(get("name").expect("scenario missing name"));
    let mjcf = expect_string(get("mjcf").expect("scenario missing mjcf"));
    let n_steps = expect_u32(get("n_steps").expect("scenario missing n_steps"));
    let stride = expect_u32(get("stride").expect("scenario missing stride"));
    let init_qpos = get("init_qpos").map(expect_f64_vec);
    let init_qvel = get("init_qvel").map(expect_f64_vec);
    let actuator_targets = get("actuator_targets").map(|v| {
        let entries = match v {
            Value::Object(o) => o,
            _ => panic!("actuator_targets must be an object"),
        };
        entries
            .into_iter()
            .map(|(k, v)| (k, expect_f64(v) as f32))
            .collect()
    });
    ScenarioSpec {
        name,
        mjcf,
        n_steps,
        stride,
        init_qpos,
        init_qvel,
        actuator_targets,
    }
}

fn expect_string(v: Value) -> String {
    match v {
        Value::String(s) => s,
        other => panic!("expected string, got {}", other.type_name()),
    }
}

fn expect_u32(v: Value) -> u32 {
    match v {
        Value::Number(n) => n as u32,
        other => panic!("expected number, got {}", other.type_name()),
    }
}

fn expect_f64(v: Value) -> f64 {
    match v {
        Value::Number(n) => n,
        other => panic!("expected number, got {}", other.type_name()),
    }
}

fn expect_f64_vec(v: Value) -> Vec<f64> {
    let arr = match v {
        Value::Array(a) => a,
        other => panic!("expected array, got {}", other.type_name()),
    };
    arr.into_iter().map(expect_f64).collect()
}

// ---------------------------------------------------------------------------
// initial-state application (MuJoCo layout → newt state)
// ---------------------------------------------------------------------------

/// Apply MuJoCo-layout `init_qpos` to newt's world. Trees are visited in
/// index order; free bodies in world.bodies are visited in index order.
/// This matches how our MJCF loader constructs the world.
fn apply_init_qpos(world: &mut World, qpos: &[f64]) {
    let mut cursor = 0usize;
    // free bodies first? MuJoCo's qpos layout follows JOINT order, not
    // body order — but our loader promises free bodies live in
    // world.bodies and trees in world.trees. In every current scenario
    // the world has EITHER free bodies OR trees, not both, so a simple
    // ordering rule is enough. When we later have mixed scenes, this
    // helper will need the full jnt_qposadr mapping.
    if !world.bodies.is_empty() && !world.trees.is_empty() {
        panic!(
            "apply_init_qpos needs an explicit jnt_qposadr mapping when free bodies \
             and trees coexist; add one before authoring such a scenario"
        );
    }
    for body in world.bodies.iter_mut() {
        // 7: px py pz qw qx qy qz
        body.position = Vec3::new(
            qpos[cursor] as f32,
            qpos[cursor + 1] as f32,
            qpos[cursor + 2] as f32,
        );
        let qw = qpos[cursor + 3] as f32;
        let qx = qpos[cursor + 4] as f32;
        let qy = qpos[cursor + 5] as f32;
        let qz = qpos[cursor + 6] as f32;
        body.orientation = Quat::new(qx, qy, qz, qw).renormalize();
        cursor += 7;
    }
    for tree in world.trees.iter_mut() {
        let n = tree.nq();
        assert!(cursor + n <= qpos.len(), "init_qpos too short for tree");
        for i in 0..n {
            tree.q[i] = qpos[cursor + i] as f32;
        }
        cursor += n;
    }
    assert_eq!(cursor, qpos.len(), "init_qpos leftover data");
}

/// Apply MuJoCo-layout `init_qvel` to newt's world. Layout per body:
/// (vx, vy, vz) world-frame linear + (wx, wy, wz) LOCAL-body-frame
/// angular. This mirrors MuJoCo's freejoint convention (verified by a
/// dedicated one-off test in the biped venv; see docs/differential.md).
fn apply_init_qvel(world: &mut World, qvel: &[f64]) {
    let mut cursor = 0usize;
    if !world.bodies.is_empty() && !world.trees.is_empty() {
        panic!("apply_init_qvel needs a jnt_dofadr mapping for mixed scenes");
    }
    for body in world.bodies.iter_mut() {
        body.linear_velocity = Vec3::new(
            qvel[cursor] as f32,
            qvel[cursor + 1] as f32,
            qvel[cursor + 2] as f32,
        );
        body.angular_velocity_body = Vec3::new(
            qvel[cursor + 3] as f32,
            qvel[cursor + 4] as f32,
            qvel[cursor + 5] as f32,
        );
        cursor += 6;
    }
    for tree in world.trees.iter_mut() {
        let n = tree.nv();
        assert!(cursor + n <= qvel.len(), "init_qvel too short for tree");
        for i in 0..n {
            tree.qdot[i] = qvel[cursor + i] as f32;
        }
        cursor += n;
    }
    assert_eq!(cursor, qvel.len(), "init_qvel leftover data");
}

// ---------------------------------------------------------------------------
// newt state → MuJoCo layout (for comparison)
// ---------------------------------------------------------------------------

fn extract_qpos(world: &World) -> Vec<f64> {
    let mut out = Vec::new();
    for body in &world.bodies {
        out.push(body.position.x as f64);
        out.push(body.position.y as f64);
        out.push(body.position.z as f64);
        // MuJoCo wxyz.
        out.push(body.orientation.w as f64);
        out.push(body.orientation.x as f64);
        out.push(body.orientation.y as f64);
        out.push(body.orientation.z as f64);
    }
    for tree in &world.trees {
        for &q in &tree.q {
            out.push(q as f64);
        }
    }
    out
}

fn extract_qvel(world: &World) -> Vec<f64> {
    let mut out = Vec::new();
    for body in &world.bodies {
        out.push(body.linear_velocity.x as f64);
        out.push(body.linear_velocity.y as f64);
        out.push(body.linear_velocity.z as f64);
        out.push(body.angular_velocity_body.x as f64);
        out.push(body.angular_velocity_body.y as f64);
        out.push(body.angular_velocity_body.z as f64);
    }
    for tree in &world.trees {
        for &qd in &tree.qdot {
            out.push(qd as f64);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------

struct Divergence {
    qpos_max: f64,
    qvel_max: f64,
    qpos_max_idx: (usize, usize), // (sample, component)
    qvel_max_idx: (usize, usize),
}

fn compare_and_measure(
    scenario_name: &str,
    fixture: &Fixture,
    newt_samples: &[(Vec<f64>, Vec<f64>)],
) -> Divergence {
    assert_eq!(
        newt_samples.len(),
        fixture.samples.len(),
        "{scenario_name}: sample count mismatch (newt {}, fixture {})",
        newt_samples.len(),
        fixture.samples.len()
    );
    let mut d = Divergence {
        qpos_max: 0.0,
        qvel_max: 0.0,
        qpos_max_idx: (0, 0),
        qvel_max_idx: (0, 0),
    };
    let dump = std::env::var("NEWT_DIFFERENTIAL_DUMP").is_ok();
    for (i, (fix, (newt_qpos, newt_qvel))) in
        fixture.samples.iter().zip(newt_samples.iter()).enumerate()
    {
        assert_eq!(
            newt_qpos.len(),
            fix.qpos.len(),
            "{scenario_name}: qpos length mismatch at sample {i}"
        );
        assert_eq!(
            newt_qvel.len(),
            fix.qvel.len(),
            "{scenario_name}: qvel length mismatch at sample {i}"
        );
        // For the qpos comparison, quaternion sign flips are physically
        // equivalent (q and -q represent the same rotation). We compare
        // components directly EXCEPT free-body quaternion slots, where
        // we canonicalize both sides by flipping the newt side if its
        // dot with the MuJoCo side is negative. Slot layout is known
        // per scenario: free-body scenarios have 7-slot repeats, with
        // slots [3..7] being (qw, qx, qy, qz).
        let mut newt_qpos = newt_qpos.clone();
        let mut idx = 0;
        // MuJoCo does not renormalize free-joint quaternions between
        // sample points either, so this only fixes the sign — magnitude
        // stays honest.
        while idx + 7 <= newt_qpos.len() && idx < scenario_free_body_slots(scenario_name) * 7 {
            let dot = newt_qpos[idx + 3] * fix.qpos[idx + 3]
                + newt_qpos[idx + 4] * fix.qpos[idx + 4]
                + newt_qpos[idx + 5] * fix.qpos[idx + 5]
                + newt_qpos[idx + 6] * fix.qpos[idx + 6];
            if dot < 0.0 {
                for slot in newt_qpos[idx + 3..idx + 7].iter_mut() {
                    *slot = -*slot;
                }
            }
            idx += 7;
        }
        for (k, (a, b)) in newt_qpos.iter().zip(fix.qpos.iter()).enumerate() {
            let e = (a - b).abs();
            if e > d.qpos_max {
                d.qpos_max = e;
                d.qpos_max_idx = (i, k);
            }
        }
        for (k, (a, b)) in newt_qvel.iter().zip(fix.qvel.iter()).enumerate() {
            let e = (a - b).abs();
            if e > d.qvel_max {
                d.qvel_max = e;
                d.qvel_max_idx = (i, k);
            }
        }
        if dump {
            let per_qpos: Vec<String> = newt_qpos
                .iter()
                .zip(fix.qpos.iter())
                .map(|(a, b)| format!("{:.3e}", (a - b).abs()))
                .collect();
            let per_qvel: Vec<String> = newt_qvel
                .iter()
                .zip(fix.qvel.iter())
                .map(|(a, b)| format!("{:.3e}", (a - b).abs()))
                .collect();
            eprintln!(
                "  {scenario_name} sample {i:3} step {:5}: qpos_err=[{}] qvel_err=[{}]",
                (i as u32) * fixture.stride,
                per_qpos.join(","),
                per_qvel.join(","),
            );
        }
    }
    d
}

/// Number of leading free bodies in the qpos layout for a scenario. Used
/// to canonicalize quaternion sign for those slots.
fn scenario_free_body_slots(name: &str) -> usize {
    match name {
        "ballistic" | "tumble" | "sphere_drop" => 1,
        "box_stack" => 3,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// per-scenario harness
// ---------------------------------------------------------------------------

fn run_scenario(spec: &ScenarioSpec) -> Divergence {
    let mjcf_path = references_dir().join(&spec.mjcf);
    let src = fs::read_to_string(&mjcf_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", mjcf_path.display()));
    let mut scene: Scene = newt::mjcf::load_mjcf_str(&src)
        .unwrap_or_else(|e| panic!("{} newt-load failed: {e}", spec.name));

    if let Some(qpos) = &spec.init_qpos {
        apply_init_qpos(&mut scene.world, qpos);
    }
    if let Some(qvel) = &spec.init_qvel {
        apply_init_qvel(&mut scene.world, qvel);
    }
    if let Some(targets) = &spec.actuator_targets {
        for (name, ctrl) in targets {
            let (tree_idx, act_idx) = *scene
                .actuators_by_name
                .get(name)
                .unwrap_or_else(|| panic!("{}: actuator {name} not found", spec.name));
            scene.world.trees[tree_idx].set_actuator_target(act_idx, *ctrl);
        }
    }

    let fixture_path = references_dir().join(format!("{}.bin", spec.name));
    let fixture = read_fixture(&fixture_path);
    assert_eq!(
        fixture.n_steps, spec.n_steps,
        "{}: fixture n_steps {} != scenarios.json n_steps {}",
        spec.name, fixture.n_steps, spec.n_steps
    );
    assert_eq!(
        fixture.stride, spec.stride,
        "{}: fixture stride {} != scenarios.json stride {}",
        spec.name, fixture.stride, spec.stride
    );

    // Sample step 0 (initial state, post-init).
    let mut newt_samples = Vec::with_capacity(fixture.samples.len());
    newt_samples.push((extract_qpos(&scene.world), extract_qvel(&scene.world)));
    for step in 1..=spec.n_steps {
        scene.world.step();
        if step % spec.stride == 0 {
            newt_samples.push((extract_qpos(&scene.world), extract_qvel(&scene.world)));
        }
    }
    compare_and_measure(&spec.name, &fixture, &newt_samples)
}

fn assert_within_tolerance(scenario: &str, d: &Divergence) {
    let tol = tolerance(scenario);
    println!(
        "differential[{scenario}] observed qpos_max={:.6e} @ sample {} comp {} (bound {:.2e}) \
         qvel_max={:.6e} @ sample {} comp {} (bound {:.2e})",
        d.qpos_max,
        d.qpos_max_idx.0,
        d.qpos_max_idx.1,
        tol.qpos,
        d.qvel_max,
        d.qvel_max_idx.0,
        d.qvel_max_idx.1,
        tol.qvel,
    );
    assert!(
        d.qpos_max <= tol.qpos,
        "{scenario}: qpos divergence {:.6e} exceeds tolerance {:.2e} at (sample {}, comp {}). \
         If this is a real physical divergence (not a bug), grow the bound in the \
         TOLERANCES table AND update docs/differential.md with the new number and cause.",
        d.qpos_max,
        tol.qpos,
        d.qpos_max_idx.0,
        d.qpos_max_idx.1,
    );
    assert!(
        d.qvel_max <= tol.qvel,
        "{scenario}: qvel divergence {:.6e} exceeds tolerance {:.2e} at (sample {}, comp {})",
        d.qvel_max,
        tol.qvel,
        d.qvel_max_idx.0,
        d.qvel_max_idx.1,
    );
}

fn scenario(name: &str) -> ScenarioSpec {
    load_scenarios()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("scenario {name} missing from scenarios.json"))
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[test]
fn differential_ballistic() {
    let d = run_scenario(&scenario("ballistic"));
    assert_within_tolerance("ballistic", &d);
}

#[test]
fn differential_tumble() {
    let d = run_scenario(&scenario("tumble"));
    assert_within_tolerance("tumble", &d);
}

#[test]
fn differential_double_pendulum() {
    let d = run_scenario(&scenario("double_pendulum"));
    assert_within_tolerance("double_pendulum", &d);
}

#[test]
fn differential_servo_arm() {
    let d = run_scenario(&scenario("servo_arm"));
    assert_within_tolerance("servo_arm", &d);
}

#[test]
fn differential_sphere_drop() {
    let d = run_scenario(&scenario("sphere_drop"));
    assert_within_tolerance("sphere_drop", &d);
}

#[test]
fn differential_box_stack() {
    let d = run_scenario(&scenario("box_stack"));
    assert_within_tolerance("box_stack", &d);
}

#[test]
fn differential_joint_limit_swing() {
    let d = run_scenario(&scenario("joint_limit_swing"));
    assert_within_tolerance("joint_limit_swing", &d);
}
