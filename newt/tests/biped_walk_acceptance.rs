//! NEWT-24 v3 no-assist biped acceptance sweep.

#![allow(clippy::excessive_precision)]

#[path = "../examples/biped_walk_support.rs"]
mod biped_walk_support;

use biped_walk_support::{GaitConfig, run_walk_with_solver_phase_observed};
use newt::contact::Contact;
use newt::geom::{GeomShape, geom_world_pose};
use newt::json::{self, Value};
use newt::math::Vec3;
use newt::mjcf::load_mjcf_path;
use newt::model::Scene;
use newt::solver::SolverMode;
use newt::tree::forward_kinematics;
use newt::world::{Integrator, SolverPhaseDiagnostics};
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 8] = b"NEWTBIP3";
const QPOS_COUNT: usize = 17;
const QVEL_COUNT: usize = 16;
const FALL_ROOT_COM_HEIGHT: f32 = 0.45 + 0.023298969;
const COM_OFFSET_Z: f64 = 0.023298969;
const EARLY_QPOS_GAP_BOUND: f64 = 0.08;
const EARLY_QVEL_GAP_BOUND: f64 = 1.5;
const STATE_INJECTION_POSITION_BOUND: f64 = 5e-4;
const STATE_INJECTION_DEPTH_BOUND: f64 = 5e-6;

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
    visual_contact_mask: u8,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
}

#[derive(Debug)]
struct NewtRun {
    result: biped_walk_support::WalkResult,
    first_fall_step: Option<u32>,
    checkpoints: Vec<Checkpoint>,
    phase_checkpoints: Vec<PhaseCheckpoint>,
}

#[derive(Debug)]
struct PhaseCheckpoint {
    step: u32,
    visual_contact_mask: u8,
    solver_contact_mask: u8,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    contacts: Vec<PhaseContact>,
    row_to_contact: Vec<usize>,
}

#[derive(Debug)]
struct PhaseContact {
    geom_pair: [String; 2],
    position: [f64; 3],
    dist: f64,
    row_indices: Vec<usize>,
}

fn solver_phase_fixture() -> Vec<PhaseCheckpoint> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    parse_phase_fixture(
        &root.join("tests/references/biped_walk_v3_diagnostics.json"),
        "mujoco",
        "solver phase before mj_step; post-step mj_forward separately",
    )
}

fn newt_solver_phase_fixture() -> Vec<PhaseCheckpoint> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    parse_phase_fixture(
        &root.join("tests/references/biped_walk_v3_newt_diagnostics.json"),
        "newt",
        "solver phase before world.step; post-step geometry omitted",
    )
}

fn parse_phase_fixture(path: &Path, engine: &str, capture: &str) -> Vec<PhaseCheckpoint> {
    let source =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let object = expect_object(json::parse(&source).expect("biped diagnostics JSON is valid"));
    assert_eq!(expect_string(object_value(&object, "engine")), engine);
    if engine == "mujoco" {
        assert_eq!(expect_string(object_value(&object, "mujoco")), "3.11.0");
    }
    assert_eq!(
        expect_string(object_value(&object, "contact_capture")),
        capture
    );
    match object_value(&object, "records") {
        Value::Array(records) => records
            .into_iter()
            .map(|record| {
                let record = expect_object(record);
                let phase = expect_object(object_value(&record, "solver_phase"));
                assert_eq!(
                    expect_string(object_value(&phase, "phase")),
                    "solver_phase_pre_step"
                );
                parse_phase_checkpoint(&phase)
            })
            .collect(),
        other => panic!(
            "diagnostic records must be an array, got {}",
            other.type_name()
        ),
    }
}

fn parse_phase_checkpoint(object: &[(String, Value)]) -> PhaseCheckpoint {
    let contacts = match object_value(object, "contacts") {
        Value::Array(contacts) => contacts
            .into_iter()
            .map(|contact| {
                let contact = expect_object(contact);
                let pair = [
                    expect_string(object_value(&contact, "geom1")),
                    expect_string(object_value(&contact, "geom2")),
                ];
                let position = expect_f64_vec(object_value(&contact, "position"));
                assert_eq!(position.len(), 3);
                let row_indices = expect_usize_vec(object_value(&contact, "row_indices"));
                PhaseContact {
                    geom_pair: pair,
                    position: [position[0], position[1], position[2]],
                    dist: expect_f64(object_value(&contact, "dist")),
                    row_indices,
                }
            })
            .collect(),
        other => panic!("contacts must be an array, got {}", other.type_name()),
    };
    let row_to_contact = match object_value(object, "row_to_contact") {
        Value::Array(rows) => rows
            .into_iter()
            .map(|row| {
                let row = expect_object(row);
                expect_usize(object_value(&row, "contact_index"))
            })
            .collect(),
        other => panic!(
            "source row_to_contact must be an array, got {}",
            other.type_name()
        ),
    };
    let qpos = expect_f64_vec(object_value(object, "qpos"));
    let qvel = expect_f64_vec(object_value(object, "qvel"));
    PhaseCheckpoint {
        step: expect_u32(object_value(object, "step")),
        visual_contact_mask: expect_u8(object_value(object, "visual_contact_mask")),
        solver_contact_mask: expect_u8(object_value(object, "solver_contact_mask")),
        qpos,
        qvel,
        contacts,
        row_to_contact,
    }
}

fn phase_checkpoint_from_newt(step: usize, scene: &newt::model::Scene) -> PhaseCheckpoint {
    let phase = scene
        .world
        .solver_phase_diagnostics()
        .expect("solver phase capture enabled");
    let (qpos, qvel) = extract_solver_phase_qpos_qvel(phase);
    let contacts = phase
        .contacts
        .iter()
        .enumerate()
        .map(|(contact_index, contact)| {
            phase_contact_from_newt(scene, contact_index, contact, &phase.row_to_contact)
        })
        .collect();
    PhaseCheckpoint {
        step: step as u32,
        visual_contact_mask: visual_contact_mask(scene),
        solver_contact_mask: solver_contact_mask(scene, &phase.contacts),
        qpos: qpos.into_iter().map(f64::from).collect(),
        qvel: qvel.into_iter().map(f64::from).collect(),
        contacts,
        row_to_contact: phase.row_to_contact.clone(),
    }
}

fn extract_solver_phase_qpos_qvel(phase: &SolverPhaseDiagnostics) -> BipedState {
    let mut qpos = phase.qpos[..3].to_vec();
    qpos[2] -= COM_OFFSET_Z as f32;
    qpos.extend([phase.qpos[6], phase.qpos[3], phase.qpos[4], phase.qpos[5]]);
    qpos.extend_from_slice(&phase.qpos[7..]);
    let mut qvel = phase.qvel[3..6].to_vec();
    qvel.extend_from_slice(&phase.qvel[0..3]);
    qvel.extend_from_slice(&phase.qvel[6..]);
    (qpos, qvel)
}

fn inject_biped_qpos(scene: &mut Scene, qpos: &[f64]) {
    assert_eq!(qpos.len(), QPOS_COUNT);
    let tree = &mut scene.world.trees[0];
    tree.q[..3]
        .iter_mut()
        .zip(&qpos[..3])
        .for_each(|(target, source)| *target = *source as f32);
    tree.q[2] += COM_OFFSET_Z as f32;
    tree.q[3] = qpos[4] as f32;
    tree.q[4] = qpos[5] as f32;
    tree.q[5] = qpos[6] as f32;
    tree.q[6] = qpos[3] as f32;
    tree.q[7..]
        .iter_mut()
        .zip(&qpos[7..])
        .for_each(|(target, source)| *target = *source as f32);
    tree.qdot.fill(0.0);
}

fn newt_contacts_at_injected_qpos(qpos: &[f64]) -> Vec<PhaseContact> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut scene =
        load_mjcf_path(root.join("models/biped-walk.xml")).expect("biped-walk MJCF must load");
    scene.world.integrator = Integrator::Euler;
    scene.world.solver.mode = SolverMode::Newton;
    scene.world.solver.cone = newt::solver::ConeKind::Pyramidal;
    scene.world.set_solver_phase_capture(true);
    inject_biped_qpos(&mut scene, qpos);
    scene.world.capture_solver_phase();
    let phase = scene
        .world
        .solver_phase_diagnostics()
        .expect("solver phase capture enabled")
        .clone();
    phase
        .contacts
        .iter()
        .enumerate()
        .map(|(contact_index, contact)| {
            phase_contact_from_newt(&scene, contact_index, contact, &phase.row_to_contact)
        })
        .collect()
}

fn biped_foot_min_signed_distance(scene: &mut Scene, qpos: &[f64], foot: &str) -> f64 {
    inject_biped_qpos(scene, qpos);
    let geom_index = scene.geoms_by_name[foot];
    let geom = scene.world.geoms[geom_index];
    let (tree_index, link_index) = geom.link.expect("biped foot geom is link-attached");
    let (link_position, link_orientation) =
        forward_kinematics(&scene.world.trees[tree_index])[link_index];
    let pose = geom_world_pose(&geom, link_position, link_orientation);
    let half_extents = match geom.shape {
        GeomShape::Box { half_extents } => half_extents,
        shape => panic!("expected box foot geom, got {shape:?}"),
    };
    let mut minimum = f32::INFINITY;
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                minimum = minimum.min(
                    pose.point_to_world(Vec3::new(
                        sx * half_extents.x,
                        sy * half_extents.y,
                        sz * half_extents.z,
                    ))
                    .z,
                );
            }
        }
    }
    f64::from(minimum)
}

fn phase_contact_from_newt(
    scene: &newt::model::Scene,
    contact_index: usize,
    contact: &Contact,
    row_to_contact: &[usize],
) -> PhaseContact {
    let pair = [
        geom_name(scene, contact.geom_a),
        geom_name(scene, contact.geom_b),
    ];
    let row_indices = row_to_contact
        .iter()
        .enumerate()
        .filter_map(|(row, &mapped_contact)| (mapped_contact == contact_index).then_some(row))
        .collect();
    PhaseContact {
        geom_pair: pair,
        position: [
            f64::from(contact.position_world.x),
            f64::from(contact.position_world.y),
            f64::from(contact.position_world.z),
        ],
        dist: -f64::from(contact.penetration),
        row_indices,
    }
}

fn geom_name(scene: &newt::model::Scene, index: usize) -> String {
    scene
        .geoms_by_name
        .iter()
        .find_map(|(name, &candidate)| (candidate == index).then_some(name.clone()))
        .unwrap_or_else(|| format!("<unnamed:{index}>"))
}

fn solver_contact_mask(scene: &newt::model::Scene, contacts: &[Contact]) -> u8 {
    let mut mask = 0;
    for contact in contacts {
        let names = [
            geom_name(scene, contact.geom_a),
            geom_name(scene, contact.geom_b),
        ];
        if names.iter().any(|name| name == "left_foot_geom") {
            mask |= 1;
        }
        if names.iter().any(|name| name == "right_foot_geom") {
            mask |= 2;
        }
    }
    mask
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
        let visual_contact_mask = cursor.u8();
        let qpos = (0..QPOS_COUNT).map(|_| cursor.f64()).collect();
        let qvel = (0..QVEL_COUNT).map(|_| cursor.f64()).collect();
        checkpoints.push(Checkpoint {
            step,
            visual_contact_mask,
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
    let mut phase_checkpoints = Vec::with_capacity(steps + 1);
    let (initial_qpos, initial_qvel) = initial_biped_qpos_qvel();
    checkpoints.push(Checkpoint {
        step: 0,
        visual_contact_mask: 0,
        qpos: initial_qpos.into_iter().map(f64::from).collect(),
        qvel: initial_qvel.into_iter().map(f64::from).collect(),
    });
    let result = run_walk_with_solver_phase_observed(
        config,
        Integrator::Euler,
        SolverMode::Newton,
        |step, scene| {
            phase_checkpoints.push(phase_checkpoint_from_newt(step, scene));
            if (1..steps).contains(&step) {
                let visual_contact_mask = visual_contact_mask(scene);
                let (qpos, qvel) = extract_biped_qpos_qvel(scene);
                checkpoints.push(Checkpoint {
                    step: step as u32,
                    visual_contact_mask,
                    qpos: qpos.into_iter().map(f64::from).collect(),
                    qvel: qvel.into_iter().map(f64::from).collect(),
                });
            }
        },
    );
    let final_step = steps;
    checkpoints.push(checkpoint_from_trace(final_step, &result.trace));
    for (step, state) in result.trace.chunks_exact(33).enumerate() {
        if state[2] < FALL_ROOT_COM_HEIGHT {
            first_fall_step = Some((step + 1) as u32);
            break;
        }
    }
    NewtRun {
        result,
        first_fall_step,
        checkpoints,
        phase_checkpoints,
    }
}

fn checkpoint_from_trace(step: usize, trace: &[f32]) -> Checkpoint {
    assert!(step > 0);
    let state = &trace[(step - 1) * 33..step * 33];
    let mut qpos = state[..3].to_vec();
    qpos[2] -= COM_OFFSET_Z as f32;
    qpos.extend([state[6], state[3], state[4], state[5]]);
    qpos.extend_from_slice(&state[7..17]);
    let mut qvel = state[20..23].to_vec();
    qvel.extend_from_slice(&state[17..20]);
    qvel.extend_from_slice(&state[23..]);
    Checkpoint {
        step: step as u32,
        visual_contact_mask: 0,
        qpos: qpos.into_iter().map(f64::from).collect(),
        qvel: qvel.into_iter().map(f64::from).collect(),
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
    let mut first_visual_mask_mismatch = None;
    for step in 0..=compare_steps {
        let expected = &source.checkpoints[step as usize];
        let actual = &newt.checkpoints[step as usize];
        assert_eq!(expected.step, step);
        assert_eq!(actual.step, step);
        if step <= 12 {
            assert_eq!(
                actual.visual_contact_mask, expected.visual_contact_mask,
                "{level} visual contact mask diverged in the closed early window at step {step}"
            );
        }
        if first_visual_mask_mismatch.is_none()
            && expected.visual_contact_mask != actual.visual_contact_mask
        {
            first_visual_mask_mismatch = Some(step);
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
        "assist={:.1} source={} newt={} source_fall={:?} newt_fall={:?} compare_steps={} first_visual_mask_mismatch={:?} qpos_max={max_qpos:.6e} qvel_max={max_qvel:.6e} source_distance={:.6} newt_distance={:.6} source_cadence={:.3} newt_cadence={:.3} source_step={:.6} newt_step={:.6} source_stride={:.6} newt_stride={:.6} source_clearance={:.6} newt_clearance={:.6} source_final_root={:.6} newt_final_root={:.6}",
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
        first_visual_mask_mismatch,
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
        first_visual_mask_mismatch,
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
    let source = solver_phase_fixture();
    let recorded_newt = newt_solver_phase_fixture();
    assert_phase_fixture_shape(&source, 41);
    assert_phase_fixture_shape(&recorded_newt, 41);
    assert_eq!(source[18].visual_contact_mask, 2);
    assert_eq!(source[18].solver_contact_mask, 0);
    assert_eq!(recorded_newt[18].visual_contact_mask, 2);
    assert_eq!(recorded_newt[18].solver_contact_mask, 0);
    let step25 = &source[25];
    assert_eq!(step25.visual_contact_mask, 2);
    assert_eq!(step25.solver_contact_mask, 2);
    assert_eq!(step25.contacts.len(), 1);
    assert_eq!(
        step25.contacts[0].geom_pair,
        ["ground".to_string(), "right_foot_geom".to_string()]
    );
    assert!((step25.contacts[0].dist + 0.00030427783267333863).abs() < 1e-12);
    assert_eq!(step25.contacts[0].row_indices, vec![0, 1, 2, 3]);

    let structural_mismatches = compare_phase_fixtures("mujoco/newt", &source, &recorded_newt, 36);
    // The phase labels are now aligned to the contact set consumed by each
    // Euler/Newton step. The fresh comparison still measures two structural
    // mismatches, so keep the complete measured window visible.
    assert_eq!(
        structural_mismatches,
        vec![25, 35],
        "solver-phase mismatch window changed; update the fixture and diagnosis"
    );

    let newt = run_newt(0.4, 36);
    assert_eq!(newt.phase_checkpoints.len(), 37);
    assert_eq!(
        compare_phase_fixtures(
            "fixture/live newt",
            &recorded_newt,
            &newt.phase_checkpoints,
            36
        ),
        Vec::<usize>::new(),
        "the parsed newt solver-phase fixture must match the live capture"
    );
    let mut first_qpos_bound_exceed = None;
    let mut first_qvel_bound_exceed = None;
    for (step, (expected, actual)) in source
        .iter()
        .zip(&newt.phase_checkpoints)
        .take(37)
        .enumerate()
    {
        let qpos_gap = max_gap(&actual.qpos, &expected.qpos);
        let qvel_gap = max_gap(&actual.qvel, &expected.qvel);
        if qpos_gap > EARLY_QPOS_GAP_BOUND && first_qpos_bound_exceed.is_none() {
            first_qpos_bound_exceed = Some(step);
        }
        if qvel_gap > EARLY_QVEL_GAP_BOUND && first_qvel_bound_exceed.is_none() {
            first_qvel_bound_exceed = Some(step);
        }
    }
    println!(
        "solver_phase_first_bound_exceed qpos={first_qpos_bound_exceed:?} qvel={first_qvel_bound_exceed:?} structural={structural_mismatches:?}"
    );
}

#[test]
fn state_injected_contact_detection_matches_mujoco_onsets() {
    let source = solver_phase_fixture();
    for step in [25, 35] {
        let expected = &source[step].contacts;
        let actual = newt_contacts_at_injected_qpos(&source[step].qpos);
        assert_eq!(
            actual.len(),
            expected.len(),
            "injected step {step} contact count"
        );
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_eq!(
                actual.geom_pair, expected.geom_pair,
                "injected step {step} pair"
            );
            assert_eq!(
                actual.row_indices, expected.row_indices,
                "injected step {step} rows"
            );
            let position_gap = max_gap(&actual.position, &expected.position);
            let depth_gap = (actual.dist - expected.dist).abs();
            println!(
                "state_injection step={step} pair={:?} position_gap={position_gap:.9e} depth_gap={depth_gap:.9e}",
                actual.geom_pair
            );
            assert!(
                position_gap <= STATE_INJECTION_POSITION_BOUND,
                "injected step {step} position expected={:?} actual={:?}",
                expected.position,
                actual.position
            );
            assert!(
                depth_gap <= STATE_INJECTION_DEPTH_BOUND,
                "injected step {step} depth expected={} actual={}",
                expected.dist,
                actual.dist
            );
        }
        println!(
            "state_injection step={step} contacts={} pairs={:?}",
            actual.len(),
            actual
                .iter()
                .map(|contact| contact.geom_pair.clone())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn biped_contact_onset_distances_are_measured() {
    let source = solver_phase_fixture();
    let recorded_newt = newt_solver_phase_fixture();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut source_scene =
        load_mjcf_path(root.join("models/biped-walk.xml")).expect("biped-walk MJCF must load");
    let mut newt_scene =
        load_mjcf_path(root.join("models/biped-walk.xml")).expect("biped-walk MJCF must load");
    for step in 20..=26 {
        let source_left =
            biped_foot_min_signed_distance(&mut source_scene, &source[step].qpos, "left_foot_geom");
        let source_right = biped_foot_min_signed_distance(
            &mut source_scene,
            &source[step].qpos,
            "right_foot_geom",
        );
        let newt_left = biped_foot_min_signed_distance(
            &mut newt_scene,
            &recorded_newt[step].qpos,
            "left_foot_geom",
        );
        let newt_right = biped_foot_min_signed_distance(
            &mut newt_scene,
            &recorded_newt[step].qpos,
            "right_foot_geom",
        );
        println!(
            "onset_distance step={step} source_left={source_left:.9e} source_right={source_right:.9e} newt_left={newt_left:.9e} newt_right={newt_right:.9e}"
        );
        assert!(source_left.is_finite());
        assert!(source_right.is_finite());
        assert!(newt_left.is_finite());
        assert!(newt_right.is_finite());
    }
}

fn assert_phase_fixture_shape(checkpoints: &[PhaseCheckpoint], expected_len: usize) {
    assert_eq!(checkpoints.len(), expected_len);
    for (step, checkpoint) in checkpoints.iter().enumerate() {
        assert_eq!(checkpoint.step, step as u32);
        assert_eq!(checkpoint.qpos.len(), QPOS_COUNT);
        assert_eq!(checkpoint.qvel.len(), QVEL_COUNT);
        assert_eq!(
            checkpoint.row_to_contact.len(),
            checkpoint.contacts.len() * 4
        );
        assert_contact_rows(checkpoint);
    }
}

fn compare_phase_fixtures(
    label: &str,
    expected: &[PhaseCheckpoint],
    actual: &[PhaseCheckpoint],
    last_step: usize,
) -> Vec<usize> {
    let mut structural_mismatches = Vec::new();
    for step in 0..=last_step {
        let expected_checkpoint = &expected[step];
        let actual_checkpoint = &actual[step];
        assert_eq!(actual_checkpoint.step, step as u32, "{label} step number");
        assert_contact_rows(actual_checkpoint);
        let structural_match = phase_structure_matches(expected_checkpoint, actual_checkpoint);
        if !structural_match {
            structural_mismatches.push(step);
        }
        if structural_match {
            for (expected_contact, actual_contact) in expected_checkpoint
                .contacts
                .iter()
                .zip(&actual_checkpoint.contacts)
            {
                assert!(
                    max_gap(&actual_contact.position, &expected_contact.position) <= 0.02,
                    "{label} step {step} contact position gap exceeds 0.02"
                );
                assert!(
                    (actual_contact.dist - expected_contact.dist).abs() <= 0.02,
                    "{label} step {step} contact depth gap exceeds 0.02"
                );
            }
        }
        println!(
            "solver_phase_fixture={label} step={step} visual_masks={}/{} solver_masks={}/{} contacts={}/{} rows={}/{} structural_match={structural_match}",
            expected_checkpoint.visual_contact_mask,
            actual_checkpoint.visual_contact_mask,
            expected_checkpoint.solver_contact_mask,
            actual_checkpoint.solver_contact_mask,
            expected_checkpoint.contacts.len(),
            actual_checkpoint.contacts.len(),
            expected_checkpoint.row_to_contact.len(),
            actual_checkpoint.row_to_contact.len(),
        );
    }
    structural_mismatches
}

fn phase_structure_matches(expected: &PhaseCheckpoint, actual: &PhaseCheckpoint) -> bool {
    expected.solver_contact_mask == actual.solver_contact_mask
        && expected.row_to_contact == actual.row_to_contact
        && expected.contacts.len() == actual.contacts.len()
        && expected
            .contacts
            .iter()
            .zip(&actual.contacts)
            .all(|(expected, actual)| {
                expected.geom_pair == actual.geom_pair && expected.row_indices == actual.row_indices
            })
}

fn assert_contact_rows(checkpoint: &PhaseCheckpoint) {
    assert_eq!(
        checkpoint.row_to_contact.len(),
        checkpoint.contacts.len() * 4
    );
    for (contact_index, contact) in checkpoint.contacts.iter().enumerate() {
        assert_eq!(
            contact.row_indices,
            (contact_index * 4..contact_index * 4 + 4).collect::<Vec<_>>()
        );
    }
    for (row, &contact_index) in checkpoint.row_to_contact.iter().enumerate() {
        assert_eq!(contact_index, row / 4);
    }
}

fn max_gap(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn expect_string(v: Value) -> String {
    match v {
        Value::String(value) => value,
        other => panic!("expected string, got {}", other.type_name()),
    }
}

fn expect_u8(v: Value) -> u8 {
    expect_usize(v) as u8
}

fn expect_u32(v: Value) -> u32 {
    expect_usize(v) as u32
}

fn expect_usize(v: Value) -> usize {
    match v {
        Value::Number(value) => value as usize,
        other => panic!("expected number, got {}", other.type_name()),
    }
}

fn expect_f64(v: Value) -> f64 {
    match v {
        Value::Number(value) => value,
        other => panic!("expected number, got {}", other.type_name()),
    }
}

fn expect_f64_vec(v: Value) -> Vec<f64> {
    match v {
        Value::Array(values) => values.into_iter().map(expect_f64).collect(),
        other => panic!("expected array, got {}", other.type_name()),
    }
}

fn expect_usize_vec(v: Value) -> Vec<usize> {
    match v {
        Value::Array(values) => values.into_iter().map(expect_usize).collect(),
        other => panic!("expected array, got {}", other.type_name()),
    }
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
