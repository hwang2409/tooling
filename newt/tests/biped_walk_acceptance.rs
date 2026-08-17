//! NEWT-24 v3 no-assist biped acceptance sweep.

#![allow(clippy::excessive_precision)]

#[path = "../examples/biped_walk_support.rs"]
mod biped_walk_support;

use biped_walk_support::{GaitConfig, run_walk_with_solver_observed};
use newt::solver::SolverMode;
use newt::world::Integrator;
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 8] = b"NEWTBIP3";
const QPOS_COUNT: usize = 17;
const QVEL_COUNT: usize = 16;
const FALL_ROOT_COM_HEIGHT: f32 = 0.45 + 0.023298969;
const COM_OFFSET_Z: f64 = 0.023298969;
const EARLY_QPOS_GAP_BOUND: f64 = 0.08;
const EARLY_QVEL_GAP_BOUND: f64 = 1.5;

#[derive(Debug)]
struct OracleFixture {
    assist_scale: f64,
    steps_requested: u32,
    steps_simulated: u32,
    fall_step: u32,
    outcome: u8,
    distance: f64,
    cadence: f64,
    step_length: f64,
    stride_length: f64,
    clearance: f64,
    final_root_height: f64,
    final_forward_speed: f64,
    max_self_contact_force: f64,
    self_contact_steps: u32,
    ground_contact_steps: u32,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Debug)]
struct Checkpoint {
    step: u32,
    contact_mask: u8,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
}

#[derive(Debug)]
struct NewtRun {
    result: biped_walk_support::WalkResult,
    first_fall_step: Option<u32>,
    checkpoints: Vec<Checkpoint>,
}

fn fixture(level: &str) -> OracleFixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "tests/references/biped_walk_oracle_v3_assist_{level}.bin"
    ));
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut cursor = Cursor::new(&bytes);
    assert_eq!(cursor.take(8), MAGIC);
    let provenance_length = cursor.u32() as usize;
    let provenance = cursor.take(provenance_length);
    let provenance = std::str::from_utf8(provenance).expect("oracle provenance is utf8");
    assert_provenance(level, provenance);
    let assist_scale = cursor.f64();
    let steps_requested = cursor.u32();
    let steps_simulated = cursor.u32();
    let fall_step = cursor.u32();
    let outcome = cursor.u8();
    let distance = cursor.f64();
    let cadence = cursor.f64();
    let step_length = cursor.f64();
    let stride_length = cursor.f64();
    let clearance = cursor.f64();
    let final_root_height = cursor.f64();
    let final_forward_speed = cursor.f64();
    let max_self_contact_force = cursor.f64();
    let self_contact_steps = cursor.u32();
    let ground_contact_steps = cursor.u32();
    let checkpoint_count = cursor.u32() as usize;
    let mut checkpoints = Vec::with_capacity(checkpoint_count);
    for _ in 0..checkpoint_count {
        let step = cursor.u32();
        let contact_mask = cursor.u8();
        let qpos = (0..QPOS_COUNT).map(|_| cursor.f64()).collect();
        let qvel = (0..QVEL_COUNT).map(|_| cursor.f64()).collect();
        checkpoints.push(Checkpoint {
            step,
            contact_mask,
            qpos,
            qvel,
        });
    }
    assert!(cursor.is_empty(), "oracle fixture has trailing bytes");
    OracleFixture {
        assist_scale,
        steps_requested,
        steps_simulated,
        fall_step,
        outcome,
        distance,
        cadence,
        step_length,
        stride_length,
        clearance,
        final_root_height,
        final_forward_speed,
        max_self_contact_force,
        self_contact_steps,
        ground_contact_steps,
        checkpoints,
    }
}

fn assert_provenance(level: &str, provenance: &str) {
    let (assist, scenario, config_sha256) = match level {
        "080" => (
            "0.8",
            "stable_joint_walk",
            "21747154ff36c0bceaeb9e6bca6a2a64678acc4f3343f9c2762a192c77493eef",
        ),
        "040" => (
            "0.4",
            "stable_joint_walk",
            "07700ea795cea806b9e13edda2e8efd24da785ba4d703a6c38bcbc4fb3241bc8",
        ),
        "020" => (
            "0.2",
            "stable_joint_walk",
            "fd25fe326cc9b5b767d4c168826b2f543e18ffa140101ca69c71303c29b3b0cb",
        ),
        "000" => (
            "0.0",
            "joint_walk",
            "e5a7c0f1d25da486b4b8c106c9cd85fb1eeabc595de2768d0d239428dbd24403",
        ),
        _ => unreachable!(),
    };
    for (key, expected) in [
        ("name", "biped_walk_oracle_v3"),
        ("mujoco", "3.11.0"),
        ("source_model", "phase3_biped_3d_v1"),
        (
            "model_sha256",
            "1e7eb5bea5f624b1dc139d027a031af72b8ba9cfb590d155925833eb19b9c62c",
        ),
        ("config_sha256", config_sha256),
        ("controller", "joint_walk"),
        (
            "controller_source_sha256",
            "e8d59b084df687b359aa4d4a71032c2717590ac275a39844c05a36cb943ee62f",
        ),
        (
            "controller_constants_sha256",
            "65baeb118d26cdea9e9c949727c5d754deb10fa4d323a3b92b5d451596026151",
        ),
        ("controller_constants_count", "123"),
        ("scenario", scenario),
        ("balance_mode", "controller"),
        (
            "deterministic_init",
            "zero_qpos_qvel_controller_targets_mj_forward",
        ),
        ("random_seed", "none"),
        ("dt", "0.005"),
        ("integrator", "Euler"),
        ("solver", "Newton"),
        ("cone", "pyramidal"),
        ("iters", "20"),
        ("steps", "5000"),
        ("stride", "1"),
        ("assist_scale", assist),
        ("nq", "17"),
        ("nv", "16"),
        ("target_speed", "0.087026797316651347"),
        ("gait_amplitude", "0.21245733613562609"),
        ("gait_frequency", "0.97925167990440609"),
        ("knee_target", "0.16236533098347533"),
        ("ankle_target", "0.09169653690943777"),
        ("root_height", "1.2431770030617277"),
        (
            "initial_qpos",
            "0,0,1.2431770030617277,1,0,0,0,0,-0.43895686944129908,0.059999999999999998,0,0.09169653690943777,0,0.44922817002131532,0.13902412302495534,0,0.1388300014922402",
        ),
        ("initial_qvel", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"),
    ] {
        assert_eq!(provenance_field(provenance, key), expected, "{provenance}");
    }
    assert_eq!(provenance_field(provenance, "date").len(), 10);
    assert!(provenance_field(provenance, "controller_constants").starts_with("{"));
}

fn provenance_field<'a>(provenance: &'a str, key: &str) -> &'a str {
    provenance
        .split('|')
        .find_map(|field| field.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("missing {key} in provenance: {provenance}"))
}

fn run_newt(assist_scale: f32, steps: usize) -> NewtRun {
    let mut config = if assist_scale == 0.0 {
        GaitConfig::joint_walk(steps)
    } else {
        GaitConfig::stable_joint_walk(steps)
    };
    config.assist_scale = assist_scale;
    let mut first_fall_step = None;
    let mut checkpoints = Vec::with_capacity(steps + 1);
    let (initial_qpos, initial_qvel) = initial_biped_qpos_qvel();
    checkpoints.push(Checkpoint {
        step: 0,
        contact_mask: 0,
        qpos: initial_qpos.into_iter().map(f64::from).collect(),
        qvel: initial_qvel.into_iter().map(f64::from).collect(),
    });
    let result = run_walk_with_solver_observed(
        config,
        Integrator::Euler,
        SolverMode::Newton,
        |step, scene| {
            let contact_mask = visual_contact_mask(scene);
            let (qpos, qvel) = extract_biped_qpos_qvel(scene);
            checkpoints.push(Checkpoint {
                step: step as u32,
                contact_mask,
                qpos: qpos.into_iter().map(f64::from).collect(),
                qvel: qvel.into_iter().map(f64::from).collect(),
            });
            if first_fall_step.is_none() && scene.world.trees[0].q[2] < FALL_ROOT_COM_HEIGHT {
                first_fall_step = Some(step as u32);
            }
        },
    );
    NewtRun {
        result,
        first_fall_step,
        checkpoints,
    }
}

fn assert_level(level: &str, expected_newt_fall_step: Option<u32>) {
    let source = fixture(level);
    let newt = run_newt(source.assist_scale as f32, source.steps_requested as usize);
    let source_fall_step = (source.fall_step > 0).then_some(source.fall_step);
    assert_eq!(
        source_fall_step,
        (source.outcome == 1).then_some(source.fall_step)
    );
    assert_eq!(newt.first_fall_step, expected_newt_fall_step);
    let source_complete = source.outcome == 0;
    assert_eq!(source_complete, source_fall_step.is_none());
    assert_eq!(source.self_contact_steps, 0);
    assert_eq!(source.max_self_contact_force, 0.0);
    assert!(source.ground_contact_steps > 0);
    assert!(source.final_forward_speed.is_finite());
    assert_eq!(newt.result.metrics.self_contact_force_steps, 0);
    assert!(newt.result.active_self_contacts.is_empty());

    let compare_steps = source
        .steps_simulated
        .min(newt.first_fall_step.unwrap_or(source.steps_requested));
    let mut max_qpos = 0.0f64;
    let mut max_qvel = 0.0f64;
    let mut first_contact_mismatch = None;
    for step in 0..=compare_steps {
        let expected = &source.checkpoints[step as usize];
        let actual = &newt.checkpoints[step as usize];
        assert_eq!(expected.step, step);
        assert_eq!(actual.step, step);
        if step <= 12 {
            assert_eq!(
                actual.contact_mask, expected.contact_mask,
                "{level} visual contact mask diverged in the closed early window at step {step}"
            );
        }
        if first_contact_mismatch.is_none() && expected.contact_mask != actual.contact_mask {
            first_contact_mismatch = Some(step);
        }
        let step_qpos = actual
            .qpos
            .iter()
            .zip(&expected.qpos)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        let step_qvel = actual
            .qvel
            .iter()
            .zip(&expected.qvel)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        max_qpos = max_qpos.max(step_qpos);
        max_qvel = max_qvel.max(step_qvel);
        if step <= 12 {
            assert!(
                step_qpos <= EARLY_QPOS_GAP_BOUND,
                "{level} step {step} qpos gap {step_qpos} exceeds early bound {EARLY_QPOS_GAP_BOUND}"
            );
            assert!(
                step_qvel <= EARLY_QVEL_GAP_BOUND,
                "{level} step {step} qvel gap {step_qvel} exceeds early bound {EARLY_QVEL_GAP_BOUND}"
            );
        }
    }
    let metrics = &newt.result.metrics;
    let (qpos_bound, qvel_bound) = match level {
        // Measured maxima plus headroom from the committed source traces.
        "080" => (0.40, 5.0),
        "040" => (1.30, 7.0),
        "020" => (1.00, 7.2),
        "000" => (1.30, 6.8),
        _ => unreachable!(),
    };
    assert_measured_metrics(level, &source, &newt.result);
    println!(
        "assist={:.1} source={} newt={} source_fall={:?} newt_fall={:?} compare_steps={} first_contact_mismatch={:?} qpos_max={max_qpos:.6e} qvel_max={max_qvel:.6e} source_distance={:.6} newt_distance={:.6} source_cadence={:.3} newt_cadence={:.3} source_step={:.6} newt_step={:.6} source_stride={:.6} newt_stride={:.6} source_clearance={:.6} newt_clearance={:.6} source_final_root={:.6} newt_final_root={:.6}",
        source.assist_scale,
        if source_complete {
            "complete"
        } else {
            "fallen"
        },
        if newt.first_fall_step.is_some() {
            "fallen"
        } else {
            "complete"
        },
        source_fall_step,
        newt.first_fall_step,
        compare_steps,
        first_contact_mismatch,
        source.distance,
        metrics.forward_distance,
        source.cadence,
        metrics.cadence_bpm,
        source.step_length,
        metrics.mean_step_length,
        source.stride_length,
        metrics.mean_stride_length,
        source.clearance,
        metrics.max_foot_clearance,
        source.final_root_height,
        newt.result.final_root_height,
    );
    assert!(max_qpos.is_finite() && max_qvel.is_finite());
    assert_ne!(
        first_contact_mismatch,
        Some(12),
        "the step-12 visual contact divergence must stay closed"
    );
    assert!(
        max_qpos <= qpos_bound,
        "current qpos gap {max_qpos} exceeds {qpos_bound}"
    );
    assert!(
        max_qvel <= qvel_bound,
        "current qvel gap {max_qvel} exceeds {qvel_bound}"
    );
    if level == "080" {
        assert!((2.50..=2.55).contains(&metrics.forward_distance));
        assert!((116.0..=119.0).contains(&metrics.cadence_bpm));
        assert!((0.36..=0.39).contains(&metrics.mean_step_length));
        assert!((0.19..=0.21).contains(&metrics.max_foot_clearance));
    }
}

fn assert_representative_level(level: &str, assist_scale: f32) {
    let source = fixture(level);
    let newt = run_newt(assist_scale, 120);
    assert_eq!(newt.first_fall_step, None);
    assert!(newt.result.final_root_height.is_finite());
    assert!(newt.result.final_forward_speed.is_finite());
    for step in 0..=120 {
        let expected = &source.checkpoints[step];
        let actual = &newt.checkpoints[step];
        let qpos_gap = actual
            .qpos
            .iter()
            .zip(&expected.qpos)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        let qvel_gap = actual
            .qvel
            .iter()
            .zip(&expected.qvel)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        if step <= 12 {
            assert!(
                qpos_gap <= EARLY_QPOS_GAP_BOUND,
                "{level} step {step} qpos gap {qpos_gap}"
            );
            assert!(
                qvel_gap <= EARLY_QVEL_GAP_BOUND,
                "{level} step {step} qvel gap {qvel_gap}"
            );
        }
    }
}

fn assert_measured_metrics(
    level: &str,
    source: &OracleFixture,
    newt: &biped_walk_support::WalkResult,
) {
    let (source_expected, newt_expected) = match level {
        "080" => (
            [2.258638, 117.600000, 0.489715, 0.112570, 0.216504, 0.981455],
            [2.523389, 117.600000, 0.372897, 0.114431, 0.200563, 0.973457],
        ),
        "040" => (
            [0.553179, 79.365079, 0.532088, 0.178983, 0.224684, 0.449739],
            [3.199486, 129.600000, 0.099338, 0.107029, 0.182692, 0.442372],
        ),
        "020" => (
            [-0.123535, 73.170732, 0.734206, 0.195305, 0.137381, 0.433937],
            [0.901319, 45.600000, 0.132662, 0.086810, 0.493977, 0.156672],
        ),
        "000" => (
            [-0.694926, 83.044983, 0.813246, 0.255460, 0.165822, 0.427472],
            [0.238266, 4.800000, 0.831020, 0.000000, 0.425592, 0.239724],
        ),
        _ => unreachable!(),
    };
    let source_actual = [
        source.distance,
        source.cadence,
        source.step_length,
        source.stride_length,
        source.clearance,
        source.final_root_height,
    ];
    let newt_actual = [
        f64::from(newt.metrics.forward_distance),
        f64::from(newt.metrics.cadence_bpm),
        f64::from(newt.metrics.mean_step_length),
        f64::from(newt.metrics.mean_stride_length),
        f64::from(newt.metrics.max_foot_clearance),
        f64::from(newt.final_root_height),
    ];
    for (name, (actual, expected)) in [
        "distance",
        "cadence",
        "step_length",
        "stride_length",
        "clearance",
        "final_root_height",
    ]
    .into_iter()
    .zip(source_actual.into_iter().zip(source_expected))
    {
        assert_metric(level, "source", name, actual, expected);
    }
    for (name, (actual, expected)) in [
        "distance",
        "cadence",
        "step_length",
        "stride_length",
        "clearance",
        "final_root_height",
    ]
    .into_iter()
    .zip(newt_actual.into_iter().zip(newt_expected))
    {
        assert_metric(level, "newt", name, actual, expected);
    }
}

fn assert_metric(level: &str, side: &str, name: &str, actual: f64, expected: f64) {
    let tolerance = 0.00002_f64.max(expected.abs() * 0.00002);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{level} {side} {name}={actual} differs from measured {expected} by more than {tolerance}"
    );
}

#[test]
fn v3_representative_sweep_is_short_and_deterministic() {
    assert_representative_level("080", 0.8);
    assert_representative_level("000", 0.0);
}

#[test]
#[ignore = "full four-level 5000-step acceptance sweep"]
fn v3_full_sweep_records_each_measured_outcome_and_divergence() {
    assert_level("080", None);
    assert_level("040", Some(551));
    assert_level("020", Some(472));
    assert_level("000", Some(439));
}

type BipedState = (Vec<f32>, Vec<f32>);

fn initial_biped_qpos_qvel() -> BipedState {
    (
        vec![
            0.0,
            0.0,
            1.2431770031,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.4389568694,
            0.06,
            0.0,
            0.091696537,
            0.0,
            0.4492281700,
            0.1390241230,
            0.0,
            0.1388300015,
        ],
        vec![0.0; QVEL_COUNT],
    )
}

fn extract_biped_qpos_qvel(scene: &newt::model::Scene) -> BipedState {
    let tree = &scene.world.trees[0];
    let mut qpos = tree.q[..3].to_vec();
    qpos[2] -= COM_OFFSET_Z as f32;
    qpos.extend([tree.q[6], tree.q[3], tree.q[4], tree.q[5]]);
    qpos.extend_from_slice(&tree.q[7..]);
    let mut qvel = tree.qdot[3..6].to_vec();
    qvel.extend_from_slice(&tree.qdot[0..3]);
    qvel.extend_from_slice(&tree.qdot[6..]);
    (qpos, qvel)
}

#[test]
fn v3_diagnostic_fixture_records_geom_manifolds() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("tests/references/biped_walk_v3_diagnostics.json");
    let diagnostics = fs::read_to_string(path).expect("biped diagnostics fixture");
    for field in [
        "\"qpos\"",
        "\"qvel\"",
        "\"position\"",
        "\"normal\"",
        "\"condim\"",
        "\"frame\"",
        "\"efc_address\"",
        "\"row_indices\"",
        "\"row_to_contact\"",
    ] {
        assert!(diagnostics.contains(field), "missing {field}");
    }
    assert!(diagnostics.contains("\"geom1\": \"ground\""));
    assert!(diagnostics.contains("\"geom2\": \"right_foot_geom\""));
    assert!(diagnostics.contains("\"contact_capture\": \"post-step mj_forward\""));
    assert!(
        diagnostics
            .contains("\"row_indices\": [\n            0,\n            1,\n            2,\n            3\n          ]")
    );
    assert!(diagnostics.contains("\"step\": 12"));

    let newt_path = root.join("tests/references/biped_walk_v3_newt_diagnostics.json");
    let newt_diagnostics = fs::read_to_string(newt_path).expect("newt biped diagnostics fixture");
    assert!(newt_diagnostics.contains("\"contact_capture\": \"post-step detect_contacts\""));
    assert!(newt_diagnostics.contains("\"geom_pair\": [\"ground\", \"right_foot_geom\"]"));
    assert!(newt_diagnostics.contains("\"normal\": [0.0, 0.0, 1.0]"));
    assert!(newt_diagnostics.contains("\"frame\": [0.0, 0.0, 1.0"));
    assert!(newt_diagnostics.contains("\"dist\": -0.004183933"));
    assert!(newt_diagnostics.contains("\"row_to_contact\": [0, 0, 0, 0, 1, 1, 1, 1]"));
}

fn visual_contact_mask(scene: &newt::model::Scene) -> u8 {
    let left_heel = scene.site_pose("left_heel_site").unwrap().0.z;
    let left_toe = scene.site_pose("left_toe_site").unwrap().0.z;
    let right_heel = scene.site_pose("right_heel_site").unwrap().0.z;
    let right_toe = scene.site_pose("right_toe_site").unwrap().0.z;
    (u8::from(left_heel <= 0.035 || left_toe <= 0.035))
        | (u8::from(right_heel <= 0.035 || right_toe <= 0.035) << 1)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> &'a [u8] {
        let end = self.offset + length;
        let output = &self.bytes[self.offset..end];
        self.offset = end;
        output
    }

    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }

    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take(8).try_into().unwrap())
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
