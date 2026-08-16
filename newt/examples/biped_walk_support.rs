//! Shared NEWT-18 walker controller, metrics, and rollout harness.
//!
//! This file stays under `examples/` so the engine crate remains a pure
//! physics library. The integration tests include it by path.

#![allow(dead_code)]
#![allow(clippy::excessive_precision)]

use std::path::Path;

use newt::contact::Contact;
use newt::math::{self, Quat, Vec3};
use newt::mjcf::load_mjcf_path;
use newt::model::Scene;
use newt::solver::{ConeKind, SolverMode};
use newt::tree::Tree;
use newt::world::Integrator;

// Cargo also discovers top-level files in `examples/` as binaries. The
// implementation is included as a module by `biped_walk.rs` and tests.
#[allow(dead_code)]
fn main() {}

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/biped-walk.xml");
const DT: f32 = 0.005;
const COM_OFFSET_Z: f32 = 0.023298969;
const STANCE_FRACTION: f32 = 0.54;
const FOOT_CONTACT_HEIGHT: f32 = 0.035;
const CONTACT_DEBOUNCE: f32 = 0.16;
const SELF_FORCE_EPSILON: f32 = 1e-3;
const ROBOT_TOUCH_SENSORS: [&str; 9] = [
    "chest_touch",
    "pelvis_touch",
    "head_touch",
    "left_thigh_touch",
    "left_shin_touch",
    "left_foot_touch",
    "right_thigh_touch",
    "right_shin_touch",
    "right_foot_touch",
];

/// Values are copied from `biped/mujoco_biped.py`.
///
/// Source line references are recorded in `docs/biped-walk.md`. The source
/// tuned these values for its `joint_walk` controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaitConfig {
    pub steps: usize,
    pub target_speed: f32,
    pub assist_scale: f32,
    pub gait_amplitude: f32,
    pub gait_frequency: f32,
    pub knee_target: f32,
    pub ankle_target: f32,
    pub root_height: f32,
}

impl GaitConfig {
    pub const fn joint_walk(steps: usize) -> Self {
        Self {
            steps,
            target_speed: 0.0870267973,
            assist_scale: 0.0,
            gait_amplitude: 0.2124573361,
            gait_frequency: 0.9792516799,
            knee_target: 0.1623653310,
            ankle_target: 0.0916965369,
            root_height: 1.2431770031,
        }
    }

    pub const fn stable_joint_walk(steps: usize) -> Self {
        let mut config = Self::joint_walk(steps);
        config.assist_scale = 0.8;
        config
    }

    pub const fn newt_root_height(self) -> f32 {
        self.root_height + COM_OFFSET_Z
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LegState {
    was_swing: bool,
    capture_adjust: f32,
    contact: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ContactEvents {
    pub left: bool,
    pub right: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GaitMetrics {
    pub step_count: usize,
    pub cadence_bpm: f32,
    pub mean_step_length: f32,
    pub mean_stride_length: f32,
    pub left_duty_factor: f32,
    pub right_duty_factor: f32,
    pub max_foot_clearance: f32,
    pub forward_distance: f32,
    pub self_contact_force_steps: usize,
    pub max_self_contact_force: f32,
    pub ground_contact_force_steps: usize,
}

#[derive(Clone, Debug, Default)]
struct MetricsAccumulator {
    previous_contacts: ContactEvents,
    stance_time: [f32; 2],
    swing_time: [f32; 2],
    last_event_time: [f32; 2],
    last_same_side_x: [Option<f32>; 2],
    previous_event_x: Option<f32>,
    step_lengths: Vec<f32>,
    stride_lengths: Vec<f32>,
    step_count: usize,
    max_foot_clearance: f32,
    initial_root_x: f32,
    final_root_x: f32,
    self_contact_force_steps: usize,
    max_self_contact_force: f32,
    ground_contact_force_steps: usize,
}

impl MetricsAccumulator {
    fn new(initial_root_x: f32, initial_contacts: ContactEvents) -> Self {
        Self {
            previous_contacts: initial_contacts,
            last_event_time: [-1.0, -1.0],
            initial_root_x,
            final_root_x: initial_root_x,
            ..Self::default()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        contacts: ContactEvents,
        time: f32,
        root_x: f32,
        left_x: f32,
        right_x: f32,
        left_clearance: f32,
        right_clearance: f32,
        self_contact_force: f32,
    ) {
        self.final_root_x = root_x;
        self.max_foot_clearance = self
            .max_foot_clearance
            .max(left_clearance)
            .max(right_clearance);

        for (side, current) in [contacts.left, contacts.right].into_iter().enumerate() {
            if current {
                self.stance_time[side] += DT;
            } else {
                self.swing_time[side] += DT;
            }
            let previous = [self.previous_contacts.left, self.previous_contacts.right][side];
            if current && !previous && time - self.last_event_time[side] >= CONTACT_DEBOUNCE {
                let x = [left_x, right_x][side];
                self.step_count += 1;
                self.last_event_time[side] = time;
                if let Some(previous_x) = self.previous_event_x {
                    self.step_lengths.push((x - previous_x).abs());
                }
                if let Some(previous_x) = self.last_same_side_x[side] {
                    self.stride_lengths.push((x - previous_x).abs());
                }
                self.previous_event_x = Some(x);
                self.last_same_side_x[side] = Some(x);
            }
        }
        self.previous_contacts = contacts;
        if self_contact_force > SELF_FORCE_EPSILON {
            self.self_contact_force_steps += 1;
            self.max_self_contact_force = self.max_self_contact_force.max(self_contact_force);
        }
        if contacts.left || contacts.right {
            self.ground_contact_force_steps += 1;
        }
    }

    fn finish(&self, duration: f32) -> GaitMetrics {
        let left_total = self.stance_time[0] + self.swing_time[0];
        let right_total = self.stance_time[1] + self.swing_time[1];
        GaitMetrics {
            step_count: self.step_count,
            cadence_bpm: if duration > 0.0 {
                self.step_count as f32 / duration * 60.0
            } else {
                0.0
            },
            mean_step_length: mean(&self.step_lengths),
            mean_stride_length: mean(&self.stride_lengths),
            left_duty_factor: ratio(self.stance_time[0], left_total),
            right_duty_factor: ratio(self.stance_time[1], right_total),
            max_foot_clearance: self.max_foot_clearance,
            forward_distance: self.final_root_x - self.initial_root_x,
            self_contact_force_steps: self.self_contact_force_steps,
            max_self_contact_force: self.max_self_contact_force,
            ground_contact_force_steps: self.ground_contact_force_steps,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WalkResult {
    pub config: GaitConfig,
    pub metrics: GaitMetrics,
    pub final_root_height: f32,
    pub final_forward_speed: f32,
    pub final_contacts: ContactEvents,
    pub trace: Vec<f32>,
    pub active_self_contacts: Vec<Contact>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GaitWindowSample {
    pub root_x: f32,
    pub root_z: f32,
    pub forward_speed: f32,
    pub left_clearance: f32,
    pub right_clearance: f32,
    pub left_contact: bool,
    pub right_contact: bool,
}

pub fn gait_window_sample(scene: &Scene) -> GaitWindowSample {
    GaitWindowSample {
        root_x: scene.world.trees[0].q[0],
        root_z: scene.world.trees[0].q[2],
        forward_speed: scene.world.trees[0].qdot[3],
        left_clearance: foot_clearance(scene, "left"),
        right_clearance: foot_clearance(scene, "right"),
        left_contact: foot_in_contact(scene, "left"),
        right_contact: foot_in_contact(scene, "right"),
    }
}

/// Port of the source `JointWalkController` target path.
pub struct JointWalkController {
    config: GaitConfig,
    phase_time: f32,
    last_timing_time: f32,
    legs: [LegState; 2],
    contacts: ContactEvents,
    initial_root_x: f32,
}

impl JointWalkController {
    // Source: `mujoco_biped.py`, JointWalkController class, lines 659-716.
    const REACH_BASE: f32 = 0.4390370608;
    const REACH_SPEED: f32 = 0.1278056195;
    const SWING_SPEED_GAIN: f32 = 0.377833;
    const STANCE_SPEED_GAIN: f32 = 0.1287270752;
    const PITCH_RATE_GAIN: f32 = 0.4477714341;
    const KNEE_LIFT: f32 = 1.4261730851;
    const SWING_KNEE_PREP: f32 = 0.2360448194;
    const SWING_ANKLE_LIFT: f32 = 0.1395586483;
    const SWING_ANKLE_BIAS: f32 = 0.1464497849;
    const STANCE_KNEE_UNLOAD: f32 = 0.15372;
    const TOE_OFF_START: f32 = 0.5640276785;
    const TOE_OFF_KNEE: f32 = 0.1570649901;
    const TOE_OFF_ANKLE: f32 = 0.0567808386;
    const SPEED_REGULATION_DEADBAND: f32 = 0.1473283639;
    const SPEED_REGULATION_GAIN: f32 = 0.5;
    const SPEED_REGULATION_LIMIT: f32 = 0.5458675878;
    const SPEED_REACH_GAIN: f32 = 0.2971386604;
    const SPEED_KNEE_GAIN: f32 = 0.9344992602;
    const SPEED_ANKLE_GAIN: f32 = 1.3701300514;
    const CONTACT_TIMING_GAIN: f32 = 0.8316505101;
    const CONTACT_SWING_HOLD_PROGRESS: f32 = 0.8362770946;
    const CONTACT_STANCE_EXTENSION_PROGRESS: f32 = 0.445887;
    const CONTACT_LANDING_PROGRESS: f32 = 0.92;
    const CONTACT_AIRBORNE_SLOWDOWN: f32 = 0.2734851087;
    const CONTACT_MIN_PHASE_RATE: f32 = 0.8064195407;
    const CONTACT_MAX_PHASE_RATE: f32 = 1.25;
    const CONTACT_LIFTOFF_GAIN: f32 = 0.1240132361;
    const CONTACT_LIFTOFF_PROGRESS_LIMIT: f32 = 0.226123;
    const CONTACT_LIFTOFF_SPEED_LIMIT: f32 = 0.3189261751;
    const CONTACT_LIFTOFF_SPEED_WIDTH: f32 = 0.300493;
    const CONTACT_LIFTOFF_KNEE_BOOST: f32 = 0.3226540996;
    const CONTACT_LIFTOFF_ANKLE_BOOST: f32 = 0.1103209537;
    const CONTACT_LIFTOFF_HIP_BOOST: f32 = 0.0943665035;
    const CAPTURE_SPEED_GAIN: f32 = 0.2733305025;
    const CAPTURE_POSITION_GAIN: f32 = 0.0520265404;
    const CAPTURE_LIMIT: f32 = 0.1880413122;
    const BRAKE_FORWARD_SPEED: f32 = 0.3222767983;
    const BRAKE_HIP: f32 = -0.1861610547;
    const BRAKE_KNEE: f32 = 1.0333017467;
    const BRAKE_ANKLE: f32 = 0.2107117518;
    const BRAKE_BLEND_WIDTH: f32 = 0.3909524244;
    const HIGH_SPEED_BRAKE_THRESHOLD: f32 = 0.7;
    const HIGH_SPEED_BRAKE_BLEND_WIDTH: f32 = 0.3629127943;
    const HIGH_SPEED_BRAKE_HIP: f32 = 0.1753383860;
    const HIGH_SPEED_BRAKE_KNEE: f32 = 0.1287504570;
    const HIGH_SPEED_BRAKE_ANKLE: f32 = 0.6429862135;

    fn new(scene: &Scene, config: GaitConfig) -> Self {
        let initial_root_x = scene.world.trees[0].q[0];
        Self {
            config,
            phase_time: 0.0,
            last_timing_time: 0.0,
            legs: [LegState {
                contact: true,
                ..LegState::default()
            }; 2],
            contacts: ContactEvents {
                left: true,
                right: true,
            },
            initial_root_x,
        }
    }

    fn observe_contacts(&mut self, scene: &Scene) {
        let left = foot_in_contact(scene, "left");
        let right = foot_in_contact(scene, "right");
        self.contacts = ContactEvents { left, right };
        self.legs[0].contact = left;
        self.legs[1].contact = right;
    }

    fn apply_balance(&self, scene: &mut Scene, time: f32) {
        let tree = &scene.world.trees[0];
        let q = &tree.q;
        let qdot = &tree.qdot;
        let x_error = self.config.target_speed * time - q[0];
        let speed_error = self.config.target_speed - qdot[3];
        let y_error = -q[1];
        let height_error = self.config.newt_root_height() - q[2];
        let force_x = clamp(42.0 * x_error + 82.0 * speed_error, -75.0, 75.0);
        let force_y = clamp(90.0 * y_error - 35.0 * qdot[4], -35.0, 35.0);
        let force_z = clamp(240.0 * height_error - 70.0 * qdot[5], -90.0, 260.0);
        let orientation = Quat::new(q[3], q[4], q[5], q[6]);
        let up = orientation.rotate(Vec3::Z);
        let torque_x = clamp(135.0 * up.y - 24.0 * qdot[0], -95.0, 95.0);
        let torque_y = clamp(-135.0 * up.x - 24.0 * qdot[1], -95.0, 95.0);
        let torque_z = clamp(-12.0 * qdot[2], -28.0, 28.0);
        let scale = self.config.assist_scale;
        scene.world.trees[0].set_link_wrench(
            0,
            Vec3::new(force_x * scale, force_y * scale, force_z * scale),
            Vec3::new(torque_x * scale, torque_y * scale, torque_z * scale),
        );
    }

    fn before_step(&mut self, tree: &Tree, time: f32) {
        let left = self.commanded_leg_cycle(time, 0.0, 0);
        let right = self.commanded_leg_cycle(time, 0.5, 1);
        let contact_count = self.contacts.left as u8 + self.contacts.right as u8;
        let phase_rate = if contact_count == 0 {
            clamp(
                1.0 - Self::CONTACT_TIMING_GAIN * Self::CONTACT_AIRBORNE_SLOWDOWN,
                Self::CONTACT_MIN_PHASE_RATE,
                Self::CONTACT_MAX_PHASE_RATE,
            )
        } else {
            1.0
        };
        let elapsed = (time - self.last_timing_time).max(0.0);
        self.last_timing_time = time;
        self.phase_time += elapsed * phase_rate;
        let _ = (left, right, tree, time);
    }

    fn targets(&mut self, tree: &Tree, time: f32) -> [f32; 10] {
        let speed_regulation = self.speed_regulation(tree);
        let brake = self.brake_intensity(tree);
        let high_speed = clamp(
            (tree.qdot[3].abs() - Self::HIGH_SPEED_BRAKE_THRESHOLD)
                / Self::HIGH_SPEED_BRAKE_BLEND_WIDTH.max(1e-6),
            0.0,
            1.0,
        );
        let brake_hip = blend(Self::BRAKE_HIP, Self::HIGH_SPEED_BRAKE_HIP, high_speed);
        let brake_knee = blend(Self::BRAKE_KNEE, Self::HIGH_SPEED_BRAKE_KNEE, high_speed);
        let brake_ankle = blend(Self::BRAKE_ANKLE, Self::HIGH_SPEED_BRAKE_ANKLE, high_speed);
        let left = self.leg_targets(tree, time, 0.0, 0, speed_regulation);
        let right = self.leg_targets(tree, time, 0.5, 1, speed_regulation);
        let mut out = [0.0; 10];
        out[0] = self.roll_target(tree, true);
        out[1] = blend(left.0, brake_hip, brake);
        out[2] = blend(left.1, brake_knee, brake);
        out[3] = self.roll_target(tree, false);
        out[4] = blend(left.2, brake_ankle, brake);
        out[5] = -out[0];
        out[6] = blend(right.0, brake_hip, brake);
        out[7] = blend(right.1, brake_knee, brake);
        out[8] = -out[3];
        out[9] = blend(right.2, brake_ankle, brake);
        out
    }

    fn leg_targets(
        &mut self,
        tree: &Tree,
        time: f32,
        offset: f32,
        side: usize,
        speed_regulation: f32,
    ) -> (f32, f32, f32) {
        let (swing, progress) = self.commanded_leg_cycle(time, offset, side);
        let eased = ease_cycle(progress);
        let direction = if self.config.target_speed >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let reach = clamp(
            Self::REACH_BASE + Self::REACH_SPEED * self.config.target_speed.abs(),
            0.08,
            0.55,
        );
        let mut capture_adjust = self.legs[side].capture_adjust;
        if swing && !self.legs[side].was_swing {
            capture_adjust = self.capture_adjust(tree, time);
            self.legs[side].capture_adjust = capture_adjust;
        }
        self.legs[side].was_swing = swing;
        let front_hip =
            -direction * (reach + Self::SPEED_REACH_GAIN * speed_regulation) + capture_adjust;
        let back_hip = direction * reach;
        let speed_error = self.config.target_speed - tree.qdot[3];
        let pitch_feedback = Self::PITCH_RATE_GAIN * tree.qdot[1];
        let (mut hip, mut knee, mut ankle) = if swing {
            let lift = math::sin(math::PI * clamp(progress, 0.0, 1.0));
            (
                back_hip * (1.0 - eased)
                    + front_hip * eased
                    + Self::SWING_SPEED_GAIN * speed_error
                    + pitch_feedback,
                self.config.knee_target
                    + Self::KNEE_LIFT * lift
                    + Self::SWING_KNEE_PREP * (1.0 - eased)
                    + Self::SPEED_KNEE_GAIN * speed_regulation,
                self.config.ankle_target
                    - Self::SWING_ANKLE_LIFT * lift
                    - Self::SWING_ANKLE_BIAS * direction
                    + Self::SPEED_ANKLE_GAIN * speed_regulation,
            )
        } else {
            let toe_off = clamp(
                (progress - Self::TOE_OFF_START) / (1.0 - Self::TOE_OFF_START).max(0.05),
                0.0,
                1.0,
            );
            (
                front_hip * (1.0 - eased)
                    + back_hip * eased
                    + Self::STANCE_SPEED_GAIN * speed_error
                    + pitch_feedback,
                (self.config.knee_target - Self::STANCE_KNEE_UNLOAD
                    + Self::TOE_OFF_KNEE * toe_off
                    + Self::SPEED_KNEE_GAIN * speed_regulation)
                    .max(0.06),
                self.config.ankle_target
                    + (Self::TOE_OFF_ANKLE - Self::SPEED_ANKLE_GAIN * speed_regulation).max(0.0)
                        * toe_off,
            )
        };
        let other = 1 - side;
        let liftoff = if swing
            && self.legs[side].contact
            && self.legs[other].contact
            && progress < Self::CONTACT_LIFTOFF_PROGRESS_LIMIT
        {
            let signed_speed = direction * tree.qdot[3];
            let speed_weight = clamp(
                (Self::CONTACT_LIFTOFF_SPEED_LIMIT - signed_speed)
                    / Self::CONTACT_LIFTOFF_SPEED_WIDTH.max(1e-6),
                0.0,
                1.0,
            );
            Self::CONTACT_LIFTOFF_GAIN
                * speed_weight
                * (1.0
                    - 0.45
                        * clamp(
                            progress / Self::CONTACT_LIFTOFF_PROGRESS_LIMIT.max(0.05),
                            0.0,
                            1.0,
                        ))
        } else {
            0.0
        };
        hip -= direction * Self::CONTACT_LIFTOFF_HIP_BOOST * liftoff;
        knee += Self::CONTACT_LIFTOFF_KNEE_BOOST * liftoff;
        ankle -= Self::CONTACT_LIFTOFF_ANKLE_BOOST * liftoff;
        (hip, knee, ankle)
    }

    fn commanded_leg_cycle(&self, time: f32, offset: f32, side: usize) -> (bool, f32) {
        let cycle = positive_mod(self.phase_time * self.config.gait_frequency + offset, 1.0);
        let planned = if cycle < STANCE_FRACTION {
            (false, cycle / STANCE_FRACTION)
        } else {
            (true, (cycle - STANCE_FRACTION) / (1.0 - STANCE_FRACTION))
        };
        let contact = [self.contacts.left, self.contacts.right][side];
        let other_contact = [self.contacts.left, self.contacts.right][1 - side];
        let timing_gain = clamp(Self::CONTACT_TIMING_GAIN, 0.0, 1.0);
        if timing_gain <= 0.0 {
            return planned;
        }
        if planned.0 && contact && !other_contact && planned.1 < Self::CONTACT_SWING_HOLD_PROGRESS {
            if timing_gain >= 0.95 {
                return (false, 1.0);
            }
            return (true, planned.1 * (1.0 - timing_gain));
        }
        if !planned.0 && !contact && planned.1 < Self::CONTACT_STANCE_EXTENSION_PROGRESS {
            if timing_gain >= 0.95 {
                return (true, Self::CONTACT_LANDING_PROGRESS);
            }
            return (false, planned.1 * (1.0 - timing_gain));
        }
        let _ = time;
        planned
    }

    fn capture_adjust(&self, tree: &Tree, time: f32) -> f32 {
        let position_error = self.config.target_speed * time - tree.q[0];
        let speed_error = self.config.target_speed - tree.qdot[3];
        let up = root_up(tree);
        clamp(
            Self::CAPTURE_SPEED_GAIN * speed_error - Self::CAPTURE_POSITION_GAIN * position_error,
            -Self::CAPTURE_LIMIT,
            Self::CAPTURE_LIMIT,
        ) + up.x * 0.0
    }

    fn speed_regulation(&self, tree: &Tree) -> f32 {
        let direction = if self.config.target_speed >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let overspeed = (direction * (tree.qdot[3] - self.config.target_speed)
            - Self::SPEED_REGULATION_DEADBAND)
            .max(0.0);
        clamp(
            Self::SPEED_REGULATION_GAIN * overspeed,
            0.0,
            Self::SPEED_REGULATION_LIMIT,
        )
    }

    fn brake_intensity(&self, tree: &Tree) -> f32 {
        let direction = if self.config.target_speed >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let speed = direction * tree.qdot[3];
        clamp(
            (speed - Self::BRAKE_FORWARD_SPEED) / Self::BRAKE_BLEND_WIDTH.max(1e-6),
            0.0,
            1.0,
        )
        .max(clamp(
            (speed - Self::HIGH_SPEED_BRAKE_THRESHOLD)
                / Self::HIGH_SPEED_BRAKE_BLEND_WIDTH.max(1e-6),
            0.0,
            1.0,
        ))
    }

    fn roll_target(&self, tree: &Tree, left: bool) -> f32 {
        let up = root_up(tree);
        let lateral = clamp(
            -1.5354146 * up.y - 0.3160794 * tree.qdot[0],
            -0.31202857,
            0.31202857,
        );
        if left { lateral } else { -lateral }
    }

    fn apply_targets(&self, scene: &mut Scene, targets: [f32; 10]) {
        let names = [
            "left_hip_roll",
            "left_hip",
            "left_knee",
            "left_ankle_roll",
            "left_ankle",
            "right_hip_roll",
            "right_hip",
            "right_knee",
            "right_ankle_roll",
            "right_ankle",
        ];
        for (joint, target) in names.into_iter().zip(targets) {
            let actuator = scene.actuators_by_name[&format!("{joint}_motor")];
            scene.world.trees[actuator.0].set_actuator_target(actuator.1, target);
        }
    }
}

/// Run one deterministic rollout using the loaded walker model.
pub fn run_walk(config: GaitConfig) -> WalkResult {
    run_walk_observed(config, |_, _| {})
}

/// Run the walker with an explicit engine integrator. The default
/// `run_walk` path keeps the model's configured RK4 selection unchanged.
pub fn run_walk_with_integrator(config: GaitConfig, integrator: Integrator) -> WalkResult {
    run_walk_observed_with_integrator(config, integrator, |_, _| {})
}

/// Run the walker with an explicit integrator and constraint solver. This is
/// used by the Newton acceptance anchor; the default helpers keep their
/// established PGS configuration. Tree contacts remain on the penalty path.
pub fn run_walk_with_solver(
    config: GaitConfig,
    integrator: Integrator,
    solver: SolverMode,
) -> WalkResult {
    run_walk_observed_from_path_with_options(
        Path::new(MODEL_PATH),
        config,
        Some(integrator),
        Some(solver),
        |_, _| {},
    )
}

pub fn run_walk_from_path(path: &Path, config: GaitConfig) -> WalkResult {
    run_walk_observed_from_path(path, config, |_, _| {})
}

/// Return the commanded ten-joint targets before each physics step.
///
/// The trace follows the source loop: step zero sees time zero, then each
/// later sample sees the state after the prior physics step.
pub fn controller_target_trace(config: GaitConfig, steps: usize) -> Vec<[f32; 10]> {
    let mut scene = load_mjcf_path(Path::new(MODEL_PATH)).expect("biped-walk MJCF must load");
    scene.world.trees[0].set_free_root_pose(
        Vec3::new(0.0, 0.0, config.newt_root_height()),
        Quat::IDENTITY,
    );
    scene.world.trees[0].qdot.fill(0.0);
    let initial_targets = [
        0.0,
        -0.4389568694,
        0.06,
        0.0,
        config.ankle_target,
        0.0,
        0.4492281700,
        0.1390241230,
        0.0,
        0.1388300015,
    ];
    set_hinge_pose(&mut scene, initial_targets);
    scene.world.evaluate_sensors(&[]);
    let mut controller = JointWalkController::new(&scene, config);
    controller.observe_contacts(&scene);
    let mut trace = Vec::with_capacity(steps);
    for step in 0..steps {
        let time = step as f32 * DT;
        controller.apply_balance(&mut scene, time);
        let tree_snapshot = scene.world.trees[0].clone();
        controller.before_step(&tree_snapshot, time);
        let targets = controller.targets(&tree_snapshot, time);
        trace.push(targets);
        controller.apply_targets(&mut scene, targets);
        scene.world.step();
        controller.observe_contacts(&scene);
    }
    trace
}

pub fn run_walk_observed<F>(config: GaitConfig, observer: F) -> WalkResult
where
    F: FnMut(usize, &Scene),
{
    run_walk_observed_from_path(Path::new(MODEL_PATH), config, observer)
}

fn run_walk_observed_from_path<F>(path: &Path, config: GaitConfig, observer: F) -> WalkResult
where
    F: FnMut(usize, &Scene),
{
    run_walk_observed_from_path_with_options(path, config, None, None, observer)
}

fn run_walk_observed_with_integrator<F>(
    config: GaitConfig,
    integrator: Integrator,
    observer: F,
) -> WalkResult
where
    F: FnMut(usize, &Scene),
{
    run_walk_observed_from_path_with_options(
        Path::new(MODEL_PATH),
        config,
        Some(integrator),
        None,
        observer,
    )
}

fn run_walk_observed_from_path_with_options<F>(
    path: &Path,
    config: GaitConfig,
    integrator: Option<Integrator>,
    solver: Option<SolverMode>,
    mut observer: F,
) -> WalkResult
where
    F: FnMut(usize, &Scene),
{
    let mut scene = load_mjcf_path(path).expect("biped-walk MJCF must load");
    if let Some(integrator) = integrator {
        scene.world.integrator = integrator;
    }
    if let Some(solver) = solver {
        scene.world.solver.mode = solver;
        if solver == SolverMode::Newton {
            scene.world.solver.cone = ConeKind::Pyramidal;
        }
    }
    let root_height = config.newt_root_height();
    scene.world.trees[0].set_free_root_pose(Vec3::new(0.0, 0.0, root_height), Quat::IDENTITY);
    scene.world.trees[0].qdot.fill(0.0);
    let initial_targets = [
        0.0,
        -0.4389568694,
        0.06,
        0.0,
        config.ankle_target,
        0.0,
        0.4492281700,
        0.1390241230,
        0.0,
        0.1388300015,
    ];
    set_hinge_pose(&mut scene, initial_targets);
    scene.world.evaluate_sensors(&[]);
    let mut controller = JointWalkController::new(&scene, config);
    controller.observe_contacts(&scene);
    controller.initial_root_x = scene.world.trees[0].q[0];
    let initial_contacts = controller.contacts;
    let mut metrics = MetricsAccumulator::new(controller.initial_root_x, initial_contacts);
    let mut trace = Vec::with_capacity(config.steps * 16);

    for step in 0..config.steps {
        let time = step as f32 * DT;
        controller.apply_balance(&mut scene, time);
        let tree_snapshot = scene.world.trees[0].clone();
        controller.before_step(&tree_snapshot, time);
        let targets = controller.targets(&tree_snapshot, time);
        controller.apply_targets(&mut scene, targets);
        scene.world.step();
        controller.observe_contacts(&scene);
        let tree = &scene.world.trees[0];
        let left_clearance = foot_clearance(&scene, "left");
        let right_clearance = foot_clearance(&scene, "right");
        let left_x = foot_center_x(&scene, "left");
        let right_x = foot_center_x(&scene, "right");
        let self_contact_force = self_contact_force(&scene);
        metrics.observe(
            controller.contacts,
            (step + 1) as f32 * DT,
            tree.q[0],
            left_x,
            right_x,
            left_clearance,
            right_clearance,
            self_contact_force,
        );
        observer(step + 1, &scene);
        trace.extend_from_slice(&tree.q);
        trace.extend_from_slice(&tree.qdot);
    }

    let tree = &scene.world.trees[0];
    let duration = config.steps as f32 * DT;
    let active_self_contacts = active_self_contacts(&scene);
    WalkResult {
        config,
        metrics: metrics.finish(duration),
        final_root_height: tree.q[2],
        final_forward_speed: tree.qdot[3],
        final_contacts: controller.contacts,
        trace,
        active_self_contacts,
    }
}

pub fn trace_bytes(trace: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(trace.len() * 4);
    for value in trace {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

pub fn active_self_contacts(scene: &Scene) -> Vec<Contact> {
    let ground = scene.geoms_by_name["ground"];
    scene
        .world
        .detect_contacts()
        .into_iter()
        .filter(|contact| contact.geom_a != ground && contact.geom_b != ground)
        .collect()
}

fn self_contact_force(scene: &Scene) -> f32 {
    let robot_force: f32 = ROBOT_TOUCH_SENSORS
        .iter()
        .filter_map(|name| scene.sensors_by_name.get(*name))
        .filter_map(|&sensor| scene.world.sensor(sensor).and_then(|values| values.first()))
        .copied()
        .sum();
    let ground_force = scene
        .sensors_by_name
        .get("ground_touch")
        .and_then(|&sensor| scene.world.sensor(sensor))
        .and_then(|values| values.first())
        .copied()
        .unwrap_or(0.0);
    ((robot_force - ground_force).max(0.0)) * 0.5
}

fn set_hinge_pose(scene: &mut Scene, targets: [f32; 10]) {
    let names = [
        "left_hip_roll",
        "left_hip",
        "left_knee",
        "left_ankle_roll",
        "left_ankle",
        "right_hip_roll",
        "right_hip",
        "right_knee",
        "right_ankle_roll",
        "right_ankle",
    ];
    for (joint, target) in names.into_iter().zip(targets) {
        let (tree_idx, actuator_idx) = scene.actuators_by_name[&format!("{joint}_motor")];
        let link = scene.world.trees[tree_idx].actuators[actuator_idx].link_idx;
        scene.world.trees[0].set_hinge_angle(link, target);
    }
}

fn foot_center_x(scene: &Scene, side: &str) -> f32 {
    let heel = scene.site_pose(&format!("{side}_heel_site")).unwrap().0;
    let toe = scene.site_pose(&format!("{side}_toe_site")).unwrap().0;
    (heel.x + toe.x) * 0.5
}

fn foot_clearance(scene: &Scene, side: &str) -> f32 {
    let heel = scene.site_pose(&format!("{side}_heel_site")).unwrap().0.z;
    let toe = scene.site_pose(&format!("{side}_toe_site")).unwrap().0.z;
    0.0f32.max(heel.min(toe))
}

fn foot_in_contact(scene: &Scene, side: &str) -> bool {
    let heel = scene.site_pose(&format!("{side}_heel_site")).unwrap().0.z;
    let toe = scene.site_pose(&format!("{side}_toe_site")).unwrap().0.z;
    heel <= FOOT_CONTACT_HEIGHT || toe <= FOOT_CONTACT_HEIGHT
}

fn root_up(tree: &Tree) -> Vec3 {
    Quat::new(tree.q[3], tree.q[4], tree.q[5], tree.q[6]).rotate(Vec3::Z)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn clamp(value: f32, low: f32, high: f32) -> f32 {
    value.max(low).min(high)
}

fn blend(a: f32, b: f32, amount: f32) -> f32 {
    a * (1.0 - amount) + b * amount
}

fn ease_cycle(progress: f32) -> f32 {
    0.5 - 0.5 * math::cos(math::PI * clamp(progress, 0.0, 1.0))
}

fn positive_mod(value: f32, modulus: f32) -> f32 {
    let remainder = value - (value / modulus).floor() * modulus;
    if remainder < 0.0 {
        remainder + modulus
    } else {
        remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_constructed_contact_sequence_matches_gait_formulas() {
        let mut tracker = MetricsAccumulator::new(0.0, ContactEvents::default());
        let mut time = DT;
        let sample = |tracker: &mut MetricsAccumulator,
                      time: &mut f32,
                      contacts: ContactEvents,
                      left_x: f32,
                      right_x: f32| {
            tracker.observe(contacts, *time, 0.0, left_x, right_x, 0.08, 0.02, 0.0);
            *time += DT;
        };

        sample(&mut tracker, &mut time, ContactEvents::default(), 0.0, 0.0);
        sample(
            &mut tracker,
            &mut time,
            ContactEvents {
                left: true,
                right: false,
            },
            0.0,
            0.0,
        );
        for _ in 0..34 {
            sample(
                &mut tracker,
                &mut time,
                ContactEvents {
                    left: true,
                    right: false,
                },
                0.0,
                0.0,
            );
        }
        sample(
            &mut tracker,
            &mut time,
            ContactEvents {
                left: true,
                right: true,
            },
            0.0,
            0.1,
        );
        for _ in 0..34 {
            sample(
                &mut tracker,
                &mut time,
                ContactEvents {
                    left: true,
                    right: true,
                },
                0.0,
                0.1,
            );
        }
        sample(
            &mut tracker,
            &mut time,
            ContactEvents {
                left: false,
                right: true,
            },
            0.1,
            0.1,
        );
        for _ in 0..34 {
            sample(
                &mut tracker,
                &mut time,
                ContactEvents {
                    left: false,
                    right: true,
                },
                0.1,
                0.1,
            );
        }
        sample(
            &mut tracker,
            &mut time,
            ContactEvents {
                left: true,
                right: true,
            },
            0.2,
            0.1,
        );

        let metrics = tracker.finish(time - DT);
        assert_eq!(metrics.step_count, 3);
        assert!((metrics.mean_step_length - 0.1).abs() < 1e-6);
        assert!((metrics.mean_stride_length - 0.2).abs() < 1e-6);
        assert!(metrics.cadence_bpm > 0.0);
        assert!(metrics.left_duty_factor > 0.0 && metrics.left_duty_factor < 1.0);
        assert!(metrics.right_duty_factor > 0.0 && metrics.right_duty_factor < 1.0);
        assert!((metrics.max_foot_clearance - 0.08).abs() < 1e-6);
    }
}
