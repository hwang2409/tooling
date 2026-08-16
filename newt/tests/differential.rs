//! Differential harness: newt vs REAL MuJoCo.
//!
//! Each test loads the SAME MJCF file that
//! `tools/capture_mujoco.py` used to produce the committed reference
//! fixture. It steps newt with the same initial state and timestep,
//! samples qpos/qvel at the same stride, and asserts each component
//! stays within a per-scenario tolerance window.
//!
//! The tolerances in [`tolerance`] and [`energy_tolerance`] were
//! MEASURED first, then set with per-scenario headroom above the
//! observed max — mostly ~2× for the clean scenarios, up to ~5× for
//! the ones the scorecard flags with a known bounded divergence or
//! open finding. Every scenario prints its measured max at test time
//! so drift stays visible.
//!
//! Divergences beyond physical reasonableness are FINDINGS to report,
//! not to hide. If a bound needs to grow to cover a real divergence,
//! the growth belongs in the scorecard (`docs/differential.md`) with
//! a note explaining why.
//!
//! # Debug dump
//!
//! Set `NEWT_DIFFERENTIAL_DUMP=1` when running these tests to print
//! the per-sample per-component error tape (qpos and qvel) — useful
//! when a tolerance needs to be understood or re-set after a fixture
//! regen. Example:
//!
//! ```text
//! NEWT_DIFFERENTIAL_DUMP=1 cargo test --test differential -- --nocapture
//! ```
//!
//! Fixtures live in `tests/references/*.bin` and are read directly
//! from disk (no Python/MuJoCo at test time; CI runs this on Linux
//! with no MuJoCo).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use newt::geom::SolRef;
use newt::joint::JointKind;
use newt::json::{self, Value};
use newt::math::{Quat, Vec3};
use newt::model::Scene;
use newt::solver::SolverMode;
use newt::world::{Integrator, World};

// ---------------------------------------------------------------------------
// tolerances (MEASURED-then-stated; see docs/differential.md)
// ---------------------------------------------------------------------------

/// Per-scenario tolerance windows. `qpos` and `qvel` are absolute L∞
/// bounds applied component-wise to `|newt - mujoco|` at every sampled
/// step. Energy scenarios use a separate [`EnergyTolerance`] instead.
///
/// Numbers here are the ACTUAL observed max divergence times
/// per-scenario headroom — ~2× for the clean scenarios; up to ~5× for
/// scenarios with a known bounded divergence or an open finding (see
/// the inline notes on each branch and `docs/differential.md`).
/// Update when fixtures regen, and mirror to the scorecard.
struct Tolerance {
    qpos: f64,
    qvel: f64,
}

fn tolerance(name: &str) -> Tolerance {
    // Bounds are AUTHORED after observation. Default headroom is ~2×
    // the observed max; a scenario may use more when its verdict in
    // the scorecard is "bounded divergence" or "open finding" (both
    // documented per-branch). Any bound more than ~5× observation
    // without such a note is a smell — either the measurement was
    // wrong or a real divergence is being papered over. See
    // docs/differential.md for the per-scenario "observed vs bound"
    // table.
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
        // First-bounce transient dominates. The exact reference-row form
        // matches the source equations; RK4 still holds solver forces across
        // stages while MuJoCo reevaluates constraints at each stage.
        // Observed max qpos 5.69e-3 (bounce apex), qvel 4.72e-1
        // (bounce recovery).
        "sphere_drop" => Tolerance {
            qpos: 1.0e-2,
            qvel: 1.0,
        },
        // Solref-sweep companion at tc=0.010 (stiffer). The exact
        // reference-row form removes the impedance-form finding. The
        // RK4 transient remains an integration-semantics residual.
        // Observed max qpos 5.24e-2 m (first-bounce apex),
        // qvel 6.25e-1 m/s.
        "sphere_drop_stiff" => Tolerance {
            qpos: 8.0e-2,
            qvel: 8.0e-1,
        },
        // Solref-sweep companion at tc=0.050 (softer). The RK4 residual
        // grows because the soft contact force varies more within a step.
        // Observed max qpos 4.55e-2 m, qvel 3.80e-1 m/s.
        "sphere_drop_soft" => Tolerance {
            qpos: 7.0e-2,
            qvel: 8.0e-1,
        },
        // Post-NEWT-14 box-box full-manifold fix: the stack now holds
        // (see docs/differential.md, box_stack row). Observed max
        // qpos 1.05e-2 m (first-bounce transient), qvel 1.03e-1 m/s.
        // Bound leaves ~5× headroom on qpos and ~5× on qvel — enough
        // to survive integrator noise but tight enough to catch any
        // regression that lets the stack drift more than a centimetre.
        "box_stack" => Tolerance {
            qpos: 5.0e-2,
            qvel: 5.0e-1,
        },
        // Limit-force impulse profile differs between newt PGS and
        // MuJoCo PGS; drift accumulates each swing.
        // Observed max qpos 1.08e-1 rad, qvel 9.10e-1 rad/s.
        "joint_limit_swing" => Tolerance {
            qpos: 1.5e-1,
            qvel: 1.2,
        },
        // Free-root + hinge; proves the per-joint qpos/qvel remap by
        // sweeping every free-root layout slot with a nonzero initial
        // orientation, angular velocity, linear velocity, and hinge
        // rate. Divergence is at f32-quant scale — a mapping bug
        // (wrong quat-slot order, missed body-vs-world linear frame
        // rotation) would show up on the order of the initial
        // velocities themselves, not the 1e-6 seen here.
        // Observed max qpos 2.64e-6, qvel 2.77e-6.
        "floating_base" => Tolerance {
            qpos: 6.0e-6,
            qvel: 6.0e-6,
        },
        // v2 tier 2: velocity actuator on slide (cart) + free hinge (pole).
        // Both engines use gain=fixed(kv), bias=affine(0,0,-kv) semantics
        // and evaluate identically inside RK4; component divergence sits
        // at f32-quant scale over the 4 s horizon. A semantic mismatch
        // (wrong bias sign, missed gear, ctrl-vs-vel swap) would blow
        // divergence to O(0.1 m or rad).
        // Observed max qpos 2.70e-7, qvel 5.77e-7.
        "velocity_cartpole" => Tolerance {
            qpos: 1.0e-6,
            qvel: 2.0e-6,
        },
        // v2 tier 2: filter activation on a single-hinge pendulum.
        // Newt integrates activation with forward Euler at the RK4 step
        // boundary (ZOH within the step — see src/actuator.rs);
        // MuJoCo integrates activation through the same RK4 stages as
        // the mechanical state. The residual scales with dt/tau (0.1
        // here) and the ctrl step amplitude; parity remains bounded
        // and steady after the filter settles. Called out as a
        // bounded-divergence row in docs/differential.md.
        // Observed max qpos 2.56e-4 rad, qvel 1.34e-3 rad/s.
        "filtered_motor_pendulum" => Tolerance {
            qpos: 6.0e-4,
            qvel: 3.0e-3,
        },
        // v2 tier 3 (tendons): the tendon Jacobian is a linear map on
        // qdot in both engines and the passive spring/damper enters
        // through Jᵀ identically. Divergence should sit at f32-quant
        // scale — a mapping bug (wrong sign, dropped chain-rule) would
        // blow this by orders. Bounds ~2× observation.
        // Observed max qpos 5.60e-8, qvel 3.65e-7.
        "tendon_coupled" => Tolerance {
            qpos: 2.0e-7,
            qvel: 1.0e-6,
        },
        // Observed max qpos 1.26e-7, qvel 1.35e-6 — the wrap-arc math
        // adds one atan2 and two asin, giving a modest f32-quant
        // inflation over the fixed-tendon path but still well below
        // any semantic-mismatch threshold.
        "tendon_wrap" => Tolerance {
            qpos: 3.0e-7,
            qvel: 3.0e-6,
        },
        "mocap_rangefinder" => Tolerance {
            qpos: 1.0e-6,
            qvel: 1.0e-6,
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

struct SensorFixture {
    samples: Vec<Vec<f64>>,
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

fn read_sensor_fixture(name: &str) -> SensorFixture {
    let path = references_dir().join(format!("{name}_sensors.bin"));
    let bytes = fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{name}: cannot read sensor fixture {}: {e}; regenerate with tools/capture_mujoco.py",
            path.display()
        )
    });
    let mut c = Cursor::new(&bytes);
    let n_samples = c.u32() as usize;
    let dim = c.u32() as usize;
    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut sample = Vec::with_capacity(dim);
        for _ in 0..dim {
            sample.push(c.f64());
        }
        samples.push(sample);
    }
    assert_eq!(
        c.remaining(),
        0,
        "{name}: sensor fixture has trailing bytes"
    );
    SensorFixture { samples }
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
    /// "state" (default) or "energy". Energy scenarios also read
    /// `<name>_energy.bin` and compare newt vs MuJoCo total-energy
    /// drift instead of per-component state divergence (which would
    /// fail for chaotic trajectories over a long horizon).
    check_kind: CheckKind,
    compare_sensors: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum CheckKind {
    State,
    Energy,
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
    let check_kind = match get("check_kind") {
        None => CheckKind::State,
        Some(Value::String(s)) => match s.as_str() {
            "state" => CheckKind::State,
            "energy" => CheckKind::Energy,
            other => panic!("unknown check_kind {other:?}"),
        },
        Some(other) => panic!("check_kind must be a string, got {}", other.type_name()),
    };
    let compare_sensors = match get("compare_sensors") {
        None => false,
        Some(Value::Bool(value)) => value,
        Some(other) => panic!("compare_sensors must be boolean, got {}", other.type_name()),
    };
    ScenarioSpec {
        name,
        mjcf,
        n_steps,
        stride,
        init_qpos,
        init_qvel,
        actuator_targets,
        check_kind,
        compare_sensors,
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

fn object_value(object: &[(String, Value)], key: &str) -> Value {
    object
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| panic!("fixture object missing {key}"))
}

fn expect_object(v: Value) -> Vec<(String, Value)> {
    match v {
        Value::Object(object) => object,
        other => panic!("expected object, got {}", other.type_name()),
    }
}

// ---------------------------------------------------------------------------
// initial-state application (MuJoCo layout → newt state)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MuJoCo <-> newt state remap
// ---------------------------------------------------------------------------
//
// Free-body scenarios: newt holds `world.bodies[i]` with world-frame
// position, orientation quaternion (x,y,z,w), world-frame linear
// velocity, and body-frame angular velocity. MuJoCo's freejoint qpos is
// (px,py,pz, qw,qx,qy,qz) and qvel is (vx,vy,vz world, ωx,ωy,ωz body).
// One quat-slot reorder; velocity slots already match.
//
// Tree scenarios: newt visits `world.trees` in add order; each tree's
// links are topologically ordered with the root first. Every link
// carries one JointKind that decides its slot count and layout:
//
//   Free root   nq=7  newt (px,py,pz, qx,qy,qz,qw)  | MJ (px,py,pz, qw,qx,qy,qz)
//               nv=6  newt (ωx,ωy,ωz body,           | MJ (vx,vy,vz world,
//                          vx,vy,vz body)             |     ωx,ωy,ωz body)
//   Ball        nq=4  newt (qx,qy,qz,qw)             | MJ (qw,qx,qy,qz)
//               nv=3  newt (ωx,ωy,ωz body)           | MJ same
//   Hinge/Slide nq=1  scalar (identical)             | scalar (identical)
//   Fixed       nq=0                                 | (no MJ slot either)
//
// The free-root linear-velocity frame swap (body <-> world) uses the
// root's current orientation. This is the trap the reviewer flagged:
// no shipped v0/v1 scenario has a moving free-root tree, so a naive
// copy would silently pass here today but break the biped validation.
// `floating_base` proves the remap works.

fn apply_init_qpos(world: &mut World, qpos: &[f64]) {
    let values = qpos.iter().map(|value| *value as f32).collect::<Vec<_>>();
    world.apply_mujoco_qpos(&values);
}

fn apply_init_qvel(world: &mut World, qvel: &[f64]) {
    let values = qvel.iter().map(|value| *value as f32).collect::<Vec<_>>();
    world.apply_mujoco_qvel(&values);
}

fn extract_qpos(world: &World) -> Vec<f64> {
    let mut out = Vec::new();
    for body in &world.bodies {
        out.push(body.position.x as f64);
        out.push(body.position.y as f64);
        out.push(body.position.z as f64);
        // newt (x,y,z,w) -> MJ (w,x,y,z).
        out.push(body.orientation.w as f64);
        out.push(body.orientation.x as f64);
        out.push(body.orientation.y as f64);
        out.push(body.orientation.z as f64);
    }
    for tree in &world.trees {
        for i in 0..tree.links.len() {
            let q_off = tree.q_offset[i];
            match tree.links[i].joint {
                JointKind::Free => {
                    out.push(tree.q[q_off] as f64);
                    out.push(tree.q[q_off + 1] as f64);
                    out.push(tree.q[q_off + 2] as f64);
                    // (qx, qy, qz, qw) -> (qw, qx, qy, qz)
                    out.push(tree.q[q_off + 6] as f64);
                    out.push(tree.q[q_off + 3] as f64);
                    out.push(tree.q[q_off + 4] as f64);
                    out.push(tree.q[q_off + 5] as f64);
                }
                JointKind::Ball { .. } => {
                    out.push(tree.q[q_off + 3] as f64);
                    out.push(tree.q[q_off] as f64);
                    out.push(tree.q[q_off + 1] as f64);
                    out.push(tree.q[q_off + 2] as f64);
                }
                JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                    out.push(tree.q[q_off] as f64);
                }
                JointKind::Fixed => {}
            }
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
        for i in 0..tree.links.len() {
            let v_off = tree.v_offset[i];
            match tree.links[i].joint {
                JointKind::Free => {
                    // Newt (ω_body, v_body) -> MJ (v_world, ω_body).
                    let q_off = tree.q_offset[i];
                    let root_ori = Quat::new(
                        tree.q[q_off + 3],
                        tree.q[q_off + 4],
                        tree.q[q_off + 5],
                        tree.q[q_off + 6],
                    );
                    let v_body = Vec3::new(
                        tree.qdot[v_off + 3],
                        tree.qdot[v_off + 4],
                        tree.qdot[v_off + 5],
                    );
                    let v_world = root_ori.rotate(v_body);
                    out.push(v_world.x as f64);
                    out.push(v_world.y as f64);
                    out.push(v_world.z as f64);
                    out.push(tree.qdot[v_off] as f64);
                    out.push(tree.qdot[v_off + 1] as f64);
                    out.push(tree.qdot[v_off + 2] as f64);
                }
                JointKind::Ball { .. } => {
                    out.push(tree.qdot[v_off] as f64);
                    out.push(tree.qdot[v_off + 1] as f64);
                    out.push(tree.qdot[v_off + 2] as f64);
                }
                JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                    out.push(tree.qdot[v_off] as f64);
                }
                JointKind::Fixed => {}
            }
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

fn compare_sensor_samples(name: &str, fixture: &SensorFixture, samples: &[Vec<f64>]) {
    assert_eq!(
        samples.len(),
        fixture.samples.len(),
        "{name}: sensor sample count mismatch (newt {}, fixture {})",
        samples.len(),
        fixture.samples.len()
    );
    let mut max_error = 0.0f64;
    let mut max_at = (0usize, 0usize);
    for (sample_idx, (actual, expected)) in samples.iter().zip(&fixture.samples).enumerate() {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{name}: sensor dimension mismatch at sample {sample_idx}"
        );
        for (component, (a, b)) in actual.iter().zip(expected).enumerate() {
            let error = (a - b).abs();
            if error > max_error {
                max_error = error;
                max_at = (sample_idx, component);
            }
        }
    }
    println!(
        "differential[{name}] sensors: max_err={max_error:.3e} at sample {}, component {}",
        max_at.0, max_at.1
    );
    assert!(
        max_error <= 2.0e-6,
        "{name}: sensor error {max_error:.3e} exceeds 2e-6 at sample {}, component {}",
        max_at.0,
        max_at.1
    );
}

/// Number of leading free bodies in the qpos layout for a scenario. Used
/// to canonicalize quaternion sign for those slots.
fn scenario_free_body_slots(name: &str) -> usize {
    match name {
        "ballistic" | "tumble" | "sphere_drop" | "sphere_drop_stiff" | "sphere_drop_soft" => 1,
        "box_stack" => 3,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// energy comparison
// ---------------------------------------------------------------------------

/// Total mechanical energy of newt's world (KE + PE) in world
/// coordinates. Sums every free body and every tree link. PE reference
/// is z = 0 (matches MuJoCo's convention).
fn newt_total_energy(world: &World) -> f64 {
    let g = -world.gravity.z as f64; // magnitude of gravitational acceleration
    let mut e = 0.0f64;
    for body in &world.bodies {
        let ke = body.kinetic_energy() as f64;
        let pe = (body.mass as f64) * g * (body.position.z as f64);
        e += ke + pe;
    }
    for tree in &world.trees {
        e += tree_energy(tree, g);
    }
    e
}

/// Total mechanical energy of one tree in world coordinates.
///
/// Walks the tree once in topological order, propagating world-frame COM
/// linear velocity and body-frame angular velocity through the joint
/// subspace at each link. Uses newt's math and forward-kinematics
/// primitives but does NOT call ABA, so a bug in ABA cannot help this
/// helper pass. Same shape as `tree_energy` in `joints_chain_energy.rs`.
fn tree_energy(tree: &newt::tree::Tree, g: f64) -> f64 {
    let poses = newt::tree::forward_kinematics(tree);
    let n = tree.links.len();
    let mut v_lin_world = vec![Vec3::ZERO; n];
    let mut w_world = vec![Vec3::ZERO; n];
    let mut w_body = vec![Vec3::ZERO; n];
    // Root: Fixed or Free. Fixed root has zero velocity by construction;
    // Free root's velocity comes straight from qdot at its v_offset
    // (angular body + linear body).
    if let JointKind::Free = tree.links[0].joint {
        let v_off = tree.v_offset[0];
        let q_off = tree.q_offset[0];
        let root_ori = Quat::new(
            tree.q[q_off + 3],
            tree.q[q_off + 4],
            tree.q[q_off + 5],
            tree.q[q_off + 6],
        );
        let w_body0 = Vec3::new(tree.qdot[v_off], tree.qdot[v_off + 1], tree.qdot[v_off + 2]);
        let v_body0 = Vec3::new(
            tree.qdot[v_off + 3],
            tree.qdot[v_off + 4],
            tree.qdot[v_off + 5],
        );
        w_body[0] = w_body0;
        w_world[0] = root_ori.rotate(w_body0);
        v_lin_world[0] = root_ori.rotate(v_body0);
    }
    for i in 1..n {
        let link = &tree.links[i];
        let parent = link.parent.expect("non-root link has parent");
        let (child_pos, child_ori) = poses[i];
        let (parent_pos, _) = poses[parent];
        match link.joint {
            JointKind::Hinge { axis, .. } => {
                let axis_world = poses[parent].1.rotate(axis);
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let w_i_world = w_world[parent] + axis_world * qdot_i;
                w_world[i] = w_i_world;
                w_body[i] = child_ori.inverse_rotate(w_i_world);
                let joint_world =
                    parent_pos + poses[parent].1.rotate(link.joint_offset_in_parent.0);
                let v_parent_at_child_com =
                    v_lin_world[parent] + w_world[parent].cross(child_pos - parent_pos);
                v_lin_world[i] =
                    v_parent_at_child_com + (axis_world * qdot_i).cross(child_pos - joint_world);
            }
            JointKind::Fixed => {
                w_world[i] = w_world[parent];
                w_body[i] = child_ori.inverse_rotate(w_world[i]);
                v_lin_world[i] =
                    v_lin_world[parent] + w_world[parent].cross(child_pos - parent_pos);
            }
            JointKind::Slide { .. } | JointKind::Ball { .. } | JointKind::Free => {
                // No energy scenario in this ticket exercises these on a
                // non-root link. Leaving them zero would falsely report
                // conservation, so panic loudly if we ever add one.
                panic!(
                    "tree_energy encountered unsupported non-root joint kind {:?} at link {i}; \
                     extend the helper before authoring an energy scenario that uses it",
                    link.joint
                );
            }
        }
    }
    let mut e = 0.0f64;
    for i in 0..n {
        let link = &tree.links[i];
        let (com, _) = poses[i];
        let lin_ke = 0.5 * (link.mass as f64) * v_lin_world[i].dot(v_lin_world[i]) as f64;
        let iw = link.inertia_body * w_body[i];
        let rot_ke = 0.5 * w_body[i].dot(iw) as f64;
        let pe = (link.mass as f64) * g * (com.z as f64);
        e += lin_ke + rot_ke + pe;
    }
    e
}

/// Read a `<name>_energy.bin` sidecar (per-sample (kin, pot) f64 pairs)
/// and return the total energy per sample. Panics with a helpful
/// message if the sidecar is missing.
fn read_mujoco_totals(name: &str, expected_samples: usize) -> Vec<f64> {
    let path = references_dir().join(format!("{name}_energy.bin"));
    let bytes = fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{name}: cannot read energy sidecar {}: {e}; regenerate with tools/capture_mujoco.py",
            path.display()
        )
    });
    assert_eq!(
        bytes.len(),
        expected_samples * 16,
        "{name}: energy sidecar has {} bytes; expected {expected_samples} samples * 16",
        bytes.len(),
    );
    let mut out = Vec::with_capacity(expected_samples);
    for i in 0..expected_samples {
        let off = i * 16;
        let kin = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let pot = f64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
        out.push(kin + pot);
    }
    out
}

// ---------------------------------------------------------------------------
// per-scenario harness
// ---------------------------------------------------------------------------

fn run_scenario(spec: &ScenarioSpec) -> Divergence {
    run_scenario_with(spec, None, "")
}

fn run_scenario_with(
    spec: &ScenarioSpec,
    integrator: Option<Integrator>,
    fixture_suffix: &str,
) -> Divergence {
    run_scenario_with_solver(spec, integrator, None, fixture_suffix)
}

fn run_scenario_with_solver(
    spec: &ScenarioSpec,
    integrator: Option<Integrator>,
    solver: Option<SolverMode>,
    fixture_suffix: &str,
) -> Divergence {
    let mjcf_path = references_dir().join(&spec.mjcf);
    let src = fs::read_to_string(&mjcf_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", mjcf_path.display()));
    let mut scene: Scene = newt::mjcf::load_mjcf_str(&src)
        .unwrap_or_else(|e| panic!("{} newt-load failed: {e}", spec.name));
    if let Some(integrator) = integrator {
        scene.world.integrator = integrator;
    }
    if let Some(solver) = solver {
        scene.world.solver.mode = solver;
    }

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

    let fixture_path = references_dir().join(format!("{}{}.bin", spec.name, fixture_suffix));
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

    let sensor_fixture = spec
        .compare_sensors
        .then(|| read_sensor_fixture(&spec.name));
    let mut newt_sensor_samples = Vec::new();
    if spec.compare_sensors {
        scene.world.evaluate_sensors(&[]);
        newt_sensor_samples.push(
            scene
                .world
                .sensors
                .data
                .iter()
                .map(|value| *value as f64)
                .collect(),
        );
    }

    // Sample step 0 (initial state, post-init).
    let mut newt_samples = Vec::with_capacity(fixture.samples.len());
    newt_samples.push((extract_qpos(&scene.world), extract_qvel(&scene.world)));
    for step in 1..=spec.n_steps {
        scene.world.step();
        if step % spec.stride == 0 {
            newt_samples.push((extract_qpos(&scene.world), extract_qvel(&scene.world)));
            if spec.compare_sensors {
                newt_sensor_samples.push(
                    scene
                        .world
                        .sensors
                        .data
                        .iter()
                        .map(|value| *value as f64)
                        .collect(),
                );
            }
        }
    }
    if let Some(sensor_fixture) = sensor_fixture {
        compare_sensor_samples(&spec.name, &sensor_fixture, &newt_sensor_samples);
    }
    compare_and_measure(&spec.name, &fixture, &newt_samples)
}

struct EnergyReport {
    newt_drift: f64,
    mujoco_drift: f64,
    /// Max |drift_newt - drift_mujoco| over samples.
    drift_gap: f64,
}

fn run_energy_scenario(spec: &ScenarioSpec) -> EnergyReport {
    assert_eq!(spec.check_kind, CheckKind::Energy);
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

    let n_samples_expected = (spec.n_steps / spec.stride) as usize + 1;
    let mj_totals = read_mujoco_totals(&spec.name, n_samples_expected);
    let mut newt_totals = Vec::with_capacity(n_samples_expected);
    newt_totals.push(newt_total_energy(&scene.world));
    for step in 1..=spec.n_steps {
        scene.world.step();
        if step % spec.stride == 0 {
            newt_totals.push(newt_total_energy(&scene.world));
        }
    }
    assert_eq!(
        newt_totals.len(),
        mj_totals.len(),
        "{}: sample count mismatch (newt {}, mujoco {})",
        spec.name,
        newt_totals.len(),
        mj_totals.len()
    );

    let newt_ref = newt_totals[0];
    let mj_ref = mj_totals[0];
    let mut newt_drift = 0.0f64;
    let mut mujoco_drift = 0.0f64;
    let mut drift_gap = 0.0f64;
    for (n, m) in newt_totals.iter().zip(mj_totals.iter()) {
        let dn = (n - newt_ref).abs();
        let dm = (m - mj_ref).abs();
        newt_drift = newt_drift.max(dn);
        mujoco_drift = mujoco_drift.max(dm);
        drift_gap = drift_gap.max((dn - dm).abs());
    }
    EnergyReport {
        newt_drift,
        mujoco_drift,
        drift_gap,
    }
}

/// Tolerances for `double_pendulum_energy`. Measured then set.
///
/// MuJoCo's RK4 drift over 5 s is at 1e-8 scale (single-precision-free
/// integrator on a f64 state). newt's RK4 uses f32 state and quaternion
/// renormalization at RK4 stage boundaries, so its drift is larger by
/// several orders. Both should remain SMALL relative to the total
/// energy scale (~31 J for this scene) and their difference in
/// magnitude is bounded — a mapping or energy-formula bug would blow
/// either drift or the gap by orders.
struct EnergyTolerance {
    newt_drift: f64,
    mujoco_drift: f64,
    drift_gap: f64,
}

fn energy_tolerance(name: &str) -> EnergyTolerance {
    match name {
        // Observed newt_drift 2.76e-6, mujoco_drift 3.38e-8,
        // drift_gap 2.74e-6. Bounds ~2× observation. A mapping or
        // energy-formula bug would blow either drift or the gap by
        // orders (10⁻² scale) — we would notice immediately.
        "double_pendulum_energy" => EnergyTolerance {
            newt_drift: 6.0e-6,
            mujoco_drift: 8.0e-8,
            drift_gap: 6.0e-6,
        },
        other => panic!("no energy tolerance for {other:?}"),
    }
}

fn assert_energy_bounds(scenario: &str, report: &EnergyReport) {
    let tol = energy_tolerance(scenario);
    println!(
        "differential[{scenario}] energy: newt_drift={:.3e} (bound {:.2e}) \
         mujoco_drift={:.3e} (bound {:.2e}) drift_gap={:.3e} (bound {:.2e})",
        report.newt_drift,
        tol.newt_drift,
        report.mujoco_drift,
        tol.mujoco_drift,
        report.drift_gap,
        tol.drift_gap,
    );
    assert!(
        report.newt_drift <= tol.newt_drift,
        "{scenario}: newt energy drift {:.3e} exceeds bound {:.2e}",
        report.newt_drift,
        tol.newt_drift,
    );
    assert!(
        report.mujoco_drift <= tol.mujoco_drift,
        "{scenario}: mujoco energy drift {:.3e} exceeds bound {:.2e} \
         (regenerate the fixture)",
        report.mujoco_drift,
        tol.mujoco_drift,
    );
    assert!(
        report.drift_gap <= tol.drift_gap,
        "{scenario}: |newt_drift - mujoco_drift| {:.3e} exceeds bound {:.2e}",
        report.drift_gap,
        tol.drift_gap,
    );
}

fn assert_within_tolerance(scenario: &str, d: &Divergence) {
    assert_within_bounds(scenario, d, tolerance(scenario));
}

fn assert_within_bounds(scenario: &str, d: &Divergence, tol: Tolerance) {
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

fn matched_integrator_tolerance(name: &str) -> Tolerance {
    match name {
        // These bounds are measured against the 2026-08-15 MuJoCo 3.11.0
        // Euler captures. They are intentionally separate from the older
        // RK4-reference scorecard rows.
        "ballistic" => Tolerance {
            qpos: 3.0e-5,
            qvel: 8.0e-5,
        },
        "double_pendulum" => Tolerance {
            qpos: 4.0e-7,
            qvel: 1.0e-6,
        },
        "sphere_drop" => Tolerance {
            qpos: 1.5e-2,
            qvel: 9.0e-1,
        },
        "box_stack" => Tolerance {
            qpos: 2.0e-2,
            qvel: 5.0e-1,
        },
        other => panic!("no matched-integrator tolerance for {other:?}"),
    }
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
fn differential_sphere_drop_stiff() {
    let d = run_scenario(&scenario("sphere_drop_stiff"));
    assert_within_tolerance("sphere_drop_stiff", &d);
}

#[test]
fn differential_sphere_drop_soft() {
    let d = run_scenario(&scenario("sphere_drop_soft"));
    assert_within_tolerance("sphere_drop_soft", &d);
}

/// The steady-state Euler assertion for the in-window solref sweep.
/// Component-wise RK4 divergence is an integration-semantics residual;
/// this test compares the final sample from the committed RK4 fixtures.
///
/// The MuJoCo reference z is loaded from the fixture; newt is
/// stepped fresh. Both should have long since settled by t=3s
/// (post-bounce, r_dot near zero) so the compare is a true
/// steady-state comparison.
#[test]
fn sphere_drop_steady_state_penetration_matches_mujoco() {
    for (name, gap_um) in [
        ("sphere_drop_stiff", 10.0f64),
        ("sphere_drop", 10.0),
        ("sphere_drop_soft", 30.0),
    ] {
        let spec = scenario(name);
        let mjcf_path = references_dir().join(&spec.mjcf);
        let src = fs::read_to_string(&mjcf_path).unwrap();
        let mut scene: newt::model::Scene = newt::mjcf::load_mjcf_str(&src).unwrap();
        for step in 1..=spec.n_steps {
            scene.world.step();
            let _ = step;
        }
        let newt_z = scene.world.bodies[0].position.z as f64;
        let fixture = read_fixture(&references_dir().join(format!("{name}.bin")));
        let mj_final_qpos = &fixture.samples.last().unwrap().qpos;
        let mj_z = mj_final_qpos[2];
        let gap_m = (newt_z - mj_z).abs();
        let gap_bound_m = gap_um * 1.0e-6;
        println!(
            "{name} steady-state z: newt={newt_z:.9} mj={mj_z:.9} gap={:.2}μm (bound {gap_um:.0}μm)",
            gap_m * 1e6
        );
        assert!(
            gap_m <= gap_bound_m,
            "{name}: steady-state penetration gap {:.2}μm exceeds bound {gap_um:.0}μm; \
             newt_z={newt_z:.9}, mj_z={mj_z:.9}",
            gap_m * 1e6,
        );
    }
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

#[test]
fn differential_floating_base() {
    let d = run_scenario(&scenario("floating_base"));
    assert_within_tolerance("floating_base", &d);
}

#[test]
fn differential_tree_chain_contact_newton_euler() {
    let d = run_scenario_with_solver(
        &scenario("tree_chain_contact"),
        Some(Integrator::Euler),
        Some(SolverMode::Newton),
        "_newton_euler",
    );
    assert_within_bounds(
        "tree_chain_contact Newton Euler",
        &d,
        Tolerance {
            qpos: 8.0e-2,
            qvel: 1.3,
        },
    );
}

#[test]
fn differential_tree_chain_contact_pgs_euler() {
    let d = run_scenario_with_solver(
        &scenario("tree_chain_contact"),
        Some(Integrator::Euler),
        Some(SolverMode::Pgs),
        "_pgs_euler",
    );
    assert_within_bounds(
        "tree_chain_contact PGS Euler",
        &d,
        Tolerance {
            qpos: 7.0e-2,
            qvel: 0.7,
        },
    );
}

#[test]
fn differential_double_pendulum_energy() {
    let report = run_energy_scenario(&scenario("double_pendulum_energy"));
    assert_energy_bounds("double_pendulum_energy", &report);
}

#[test]
fn differential_velocity_cartpole() {
    let d = run_scenario(&scenario("velocity_cartpole"));
    assert_within_tolerance("velocity_cartpole", &d);
}

#[test]
fn differential_filtered_motor_pendulum() {
    let d = run_scenario(&scenario("filtered_motor_pendulum"));
    assert_within_tolerance("filtered_motor_pendulum", &d);
}

#[test]
fn differential_tendon_coupled() {
    let d = run_scenario(&scenario("tendon_coupled"));
    assert_within_tolerance("tendon_coupled", &d);
}

#[test]
fn differential_tendon_wrap() {
    let d = run_scenario(&scenario("tendon_wrap"));
    assert_within_tolerance("tendon_wrap", &d);
}

#[test]
fn differential_mocap_rangefinder() {
    let d = run_scenario(&scenario("mocap_rangefinder"));
    assert_within_tolerance("mocap_rangefinder", &d);
}

#[test]
fn differential_matched_euler_rows() {
    for name in ["ballistic", "double_pendulum", "sphere_drop", "box_stack"] {
        let d = run_scenario_with(&scenario(name), Some(Integrator::Euler), "_euler");
        assert_within_bounds(name, &d, matched_integrator_tolerance(name));
    }
}

#[test]
fn differential_matched_implicitfast_filtered_pendulum() {
    let d = run_scenario_with(
        &scenario("filtered_motor_pendulum"),
        Some(Integrator::ImplicitFast),
        "_implicitfast",
    );
    assert_within_bounds(
        "filtered_motor_pendulum implicitfast",
        &d,
        Tolerance {
            qpos: 3.0e-7,
            qvel: 1.0e-6,
        },
    );
}

#[test]
fn differential_matched_newton_euler_rows() {
    for name in ["sphere_drop", "box_stack", "joint_limit_swing"] {
        let d = run_scenario_with_solver(
            &scenario(name),
            Some(Integrator::Euler),
            Some(SolverMode::Newton),
            "_newton_euler",
        );
        // These bounds are measured from the captured MuJoCo 3.11.0 Newton
        // fixtures. Newton is a new parity row, so keep the bounds separate
        // from the PGS scorecard rows.
        let bound = match name {
            "sphere_drop" => Tolerance {
                qpos: 2.0e-2,
                qvel: 1.2,
            },
            "box_stack" => Tolerance {
                qpos: 5.0e-2,
                qvel: 1.0,
            },
            "joint_limit_swing" => Tolerance {
                qpos: 2.0e-1,
                qvel: 1.5,
            },
            _ => unreachable!(),
        };
        assert_within_bounds(&format!("{name} Newton Euler"), &d, bound);
    }
}

#[test]
fn permanent_constraint_factor_diagnostics_match_mujoco() {
    let source = fs::read_to_string(references_dir().join("constraint_factor_diagnostics.json"))
        .expect("constraint factor fixture missing");
    let root = expect_object(json::parse(&source).expect("constraint factor fixture is invalid"));
    assert_eq!(expect_string(object_value(&root, "mujoco")), "3.11.0");
    let records = match object_value(&root, "records") {
        Value::Array(records) => records,
        other => panic!("records must be an array, got {}", other.type_name()),
    };
    assert_eq!(records.len(), 6);
    for value in records {
        let record = expect_object(value);
        let tc = expect_f64(object_value(&record, "tc"));
        let dampratio = expect_f64(object_value(&record, "dampratio"));
        let dt = expect_f64(object_value(&record, "dt"));
        let positions = expect_f64_vec(object_value(&record, "efc_pos"));
        let velocities = expect_f64_vec(object_value(&record, "efc_vel"));
        let margins = expect_f64_vec(object_value(&record, "efc_margin"));
        let diag_a = expect_f64_vec(object_value(&record, "efc_diagA"));
        let reference = expect_f64_vec(object_value(&record, "efc_aref"));
        let regularization = expect_f64_vec(object_value(&record, "efc_R"));
        let kbip = match object_value(&record, "efc_KBIP") {
            Value::Array(rows) => rows.into_iter().map(expect_f64_vec).collect::<Vec<_>>(),
            other => panic!("efc_KBIP must be an array, got {}", other.type_name()),
        };
        assert_eq!(positions.len(), 4);
        let solref = SolRef::new(tc as f32, dampratio as f32);
        let effective_tc = tc.max(2.0 * dt);
        for row in 0..4 {
            let position = positions[row] as f32 - margins[row] as f32;
            let velocity = velocities[row] as f32;
            let impedance_value = newt::solver::impedance(position, newt::solver::SolImp::DEFAULT);
            let expected_b = 2.0 / (newt::solver::SolImp::DEFAULT.dmax * effective_tc as f32);
            let expected_k_eff = impedance_value
                / (newt::solver::SolImp::DEFAULT.dmax
                    * newt::solver::SolImp::DEFAULT.dmax
                    * effective_tc as f32
                    * effective_tc as f32
                    * dampratio as f32
                    * dampratio as f32);
            let expected_aref = newt::solver::reference_accel(
                position,
                velocity,
                solref,
                newt::solver::SolImp::DEFAULT,
            );
            let expected_r = (1.0 - impedance_value) / impedance_value * diag_a[row] as f32;
            assert!((kbip[row][2] as f32 - impedance_value).abs() < 2.0e-6);
            assert!((kbip[row][1] as f32 - expected_b).abs() < 2.0e-4);
            assert!((kbip[row][0] as f32 * impedance_value - expected_k_eff).abs() < 2.0e-1);
            assert!((reference[row] as f32 - expected_aref).abs() < 1.0e-3);
            assert!((regularization[row] as f32 - expected_r).abs() < 1.0e-5);
        }
    }

    let tree_records = match object_value(&root, "tree_records") {
        Value::Array(records) => records,
        other => panic!("tree_records must be an array, got {}", other.type_name()),
    };
    assert_eq!(tree_records.len(), 1);
    let tree = expect_object(tree_records.into_iter().next().unwrap());
    let tree_diag = expect_f64_vec(object_value(&tree, "efc_diagA"));
    let tree_r = expect_f64_vec(object_value(&tree, "efc_R"));
    let tree_kbip = match object_value(&tree, "efc_KBIP") {
        Value::Array(rows) => rows.into_iter().map(expect_f64_vec).collect::<Vec<_>>(),
        other => panic!("tree efc_KBIP must be an array, got {}", other.type_name()),
    };
    assert_eq!(tree_diag.len(), 4);
    let normal_diag_approx = tree_diag[0] / (2.0 * 0.6f64 * 0.6);
    let normal_r = (1.0 - tree_kbip[0][2]) / tree_kbip[0][2] * normal_diag_approx;
    let expected_rpy = 2.0 * 0.6f64 * 0.6 * normal_r;
    for row in 0..4 {
        assert!((tree_diag[row] - tree_diag[0]).abs() < 1.0e-7);
        assert!((tree_r[row] - expected_rpy).abs() < 1.0e-7);
        assert!((tree_kbip[row][0] - tree_kbip[0][0]).abs() < 1.0e-7);
    }
}

#[test]
fn permanent_matched_euler_solref_sweep() {
    let source = fs::read_to_string(references_dir().join("solref_sweep_euler.json"))
        .expect("Euler sweep fixture missing");
    let root = expect_object(json::parse(&source).expect("Euler sweep fixture is invalid"));
    assert_eq!(expect_string(object_value(&root, "mujoco")), "3.11.0");
    let records = match object_value(&root, "records") {
        Value::Array(records) => records,
        other => panic!("records must be an array, got {}", other.type_name()),
    };
    let bounds_um = [10.0, 20.0, 50.0, 6_000.0, 35_000.0, 70_000.0, 4_000.0];
    assert_eq!(records.len(), bounds_um.len());
    for (value, gap_um) in records.into_iter().zip(bounds_um) {
        let record = expect_object(value);
        let tc = expect_f64(object_value(&record, "tc"));
        let dampratio = expect_f64(object_value(&record, "dampratio"));
        let steps = expect_u32(object_value(&record, "steps"));
        let expected_qpos = expect_f64_vec(object_value(&record, "qpos"));
        let mjcf = fs::read_to_string(references_dir().join("sphere_drop.xml")).unwrap();
        let mut scene = newt::mjcf::load_mjcf_str(&mjcf).unwrap();
        for geom in &mut scene.world.geoms {
            geom.solref = SolRef::new(tc as f32, dampratio as f32);
        }
        scene.world.integrator = Integrator::Euler;
        scene.world.solver.mode = SolverMode::Pgs;
        for _ in 0..steps {
            scene.world.step();
        }
        let gap_um_observed =
            (scene.world.bodies[0].position.z as f64 - expected_qpos[2]).abs() * 1.0e6;
        println!(
            "Euler solref tc={tc:.3} dr={dampratio:.1}: z gap={gap_um_observed:.3}μm (bound {gap_um:.0}μm)"
        );
        assert!(
            gap_um_observed <= gap_um,
            "Euler solref tc={tc} dr={dampratio}: gap {gap_um_observed:.3}μm exceeds {gap_um:.0}μm"
        );
    }
}

#[test]
fn permanent_rk4_out_of_window_solref_rows_have_separate_bounds() {
    let source = fs::read_to_string(references_dir().join("solref_sweep_rk4.json"))
        .expect("RK4 sweep fixture missing");
    let root = expect_object(json::parse(&source).expect("RK4 sweep fixture is invalid"));
    assert_eq!(expect_string(object_value(&root, "mujoco")), "3.11.0");
    let records = match object_value(&root, "records") {
        Value::Array(records) => records,
        other => panic!("records must be an array, got {}", other.type_name()),
    };
    let bounds_um = [100.0, 4_000.0];
    assert_eq!(records.len(), bounds_um.len());
    for (value, gap_um) in records.into_iter().zip(bounds_um) {
        let record = expect_object(value);
        let tc = expect_f64(object_value(&record, "tc"));
        let dampratio = expect_f64(object_value(&record, "dampratio"));
        let steps = expect_u32(object_value(&record, "steps"));
        let expected_qpos = expect_f64_vec(object_value(&record, "qpos"));
        let mjcf = fs::read_to_string(references_dir().join("sphere_drop.xml")).unwrap();
        let mut scene = newt::mjcf::load_mjcf_str(&mjcf).unwrap();
        for geom in &mut scene.world.geoms {
            geom.solref = SolRef::new(tc as f32, dampratio as f32);
        }
        for _ in 0..steps {
            scene.world.step();
        }
        let gap_um_observed =
            (scene.world.bodies[0].position.z as f64 - expected_qpos[2]).abs() * 1.0e6;
        println!(
            "RK4 solref tc={tc:.3} dr={dampratio:.1}: z gap={gap_um_observed:.3}μm (bound {gap_um:.0}μm)"
        );
        assert!(
            gap_um_observed <= gap_um,
            "RK4 solref tc={tc} dr={dampratio}: gap {gap_um_observed:.3}μm exceeds {gap_um:.0}μm"
        );
    }
}
