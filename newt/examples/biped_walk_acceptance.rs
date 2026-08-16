//! Print the matched Newton/Euler biped acceptance sweep.

mod biped_walk_support;

use biped_walk_support::{GaitConfig, run_walk_with_solver_observed};
use newt::solver::SolverMode;
use newt::world::Integrator;

const ASSIST_LEVELS: [f32; 4] = [0.8, 0.4, 0.2, 0.0];
const FALL_ROOT_COM_HEIGHT: f32 = 0.47329897;

fn main() {
    let requested = std::env::args().skip(1).collect::<Vec<_>>();
    let levels = if let Some(value) = requested
        .windows(2)
        .find(|args| args[0] == "--assist-scale")
        .map(|args| args[1].parse::<f32>().expect("assist scale"))
    {
        vec![value]
    } else {
        ASSIST_LEVELS.to_vec()
    };
    for assist_scale in levels {
        let mut config = if assist_scale == 0.0 {
            GaitConfig::joint_walk(5000)
        } else {
            GaitConfig::stable_joint_walk(5000)
        };
        config.assist_scale = assist_scale;
        let mut first_fall_step = None;
        let result = run_walk_with_solver_observed(
            config,
            Integrator::Euler,
            SolverMode::Newton,
            |step, scene| {
                if first_fall_step.is_none() && scene.world.trees[0].q[2] < FALL_ROOT_COM_HEIGHT {
                    first_fall_step = Some(step);
                }
            },
        );
        let outcome = if first_fall_step.is_some() {
            "fallen"
        } else {
            "complete"
        };
        println!(
            "assist={assist_scale:.1} outcome={outcome} fall_step={:?} distance={:.6} cadence={:.3} step_length={:.6} stride_length={:.6} clearance={:.6} final_root_height={:.6} self_contact_steps={} max_self_contact_force={:.6}",
            first_fall_step,
            result.metrics.forward_distance,
            result.metrics.cadence_bpm,
            result.metrics.mean_step_length,
            result.metrics.mean_stride_length,
            result.metrics.max_foot_clearance,
            result.final_root_height,
            result.metrics.self_contact_force_steps,
            result.metrics.max_self_contact_force,
        );
    }
}
