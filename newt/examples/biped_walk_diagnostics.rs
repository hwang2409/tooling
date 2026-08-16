//! Print first-divergence contact and NEWT-23 row diagnostics for the biped.

mod biped_walk_support;

use biped_walk_support::{GaitConfig, gait_window_sample, run_walk_with_solver_observed};
use newt::solver::{ConeKind, SolverMode, solve_tree_contacts};
use newt::world::Integrator;

fn main() {
    let mut config = GaitConfig::stable_joint_walk(40);
    config.assist_scale = 0.4;
    run_walk_with_solver_observed(
        config,
        Integrator::Euler,
        SolverMode::Newton,
        |step, scene| {
            if !matches!(step, 17 | 25 | 36) {
                return;
            }
            let sample = gait_window_sample(scene);
            let contacts = scene.world.detect_contacts();
            let solution = solve_tree_contacts(
                &scene.world.bodies,
                &scene.world.trees,
                &scene.world.geoms,
                &contacts,
                scene.world.gravity,
                scene.world.dt,
                ConeKind::Pyramidal,
                scene.world.solver.iterations,
                true,
                Some(false),
            );
            let qfrc_max = solution.tree_qfrc[0]
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0, f32::max);
            let row_max = solution
                .row_diagnostics
                .iter()
                .map(|row| row.reference_accel.abs())
                .fold(0.0, f32::max);
            println!(
                "step={step} contact_mask={} contacts={} rows={} qfrc_max={qfrc_max:.6} qfrc={:?} row_aref_max={row_max:.6} root=({:.6},{:.6},{:.6})",
                sample.left_contact as u8 | ((sample.right_contact as u8) << 1),
                contacts.len(),
                solution.row_diagnostics.len(),
                solution.tree_qfrc[0],
                scene.world.trees[0].q[0],
                scene.world.trees[0].q[1],
                scene.world.trees[0].q[2],
            );
        },
    );
}
