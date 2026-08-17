//! Print first-divergence contact and NEWT-23 row diagnostics for the biped.

mod biped_walk_support;

use biped_walk_support::{GaitConfig, gait_window_sample, run_walk_with_solver_observed};
use newt::solver::{ConeKind, SolverMode, solve_tree_contacts};
use newt::world::{self, Integrator};

const COM_OFFSET_Z: f32 = 0.023_298_97;

fn main() {
    println!(
        "step=0 contact_mask=0 contacts=0 rows=0 row_to_contact=[] root_com=(0.000000,0.000000,{:.6})",
        GaitConfig::stable_joint_walk(40).root_height
    );
    let mut config = GaitConfig::stable_joint_walk(40);
    config.assist_scale = 0.4;
    run_walk_with_solver_observed(
        config,
        Integrator::Euler,
        SolverMode::Newton,
        |step, scene| {
            if !(1..=12).contains(&step) && !matches!(step, 17 | 25 | 36) {
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
            let row_to_contact: Vec<usize> = contacts
                .iter()
                .enumerate()
                .flat_map(|(contact_index, contact)| {
                    let condim = scene.world.geoms[contact.geom_a]
                        .condim
                        .min(scene.world.geoms[contact.geom_b].condim);
                    let rows = if condim == 3 { 4 } else { condim as usize };
                    std::iter::repeat_n(contact_index, rows)
                })
                .collect();
            println!(
                "step={step} contact_mask={} contacts={} rows={} row_to_contact={row_to_contact:?} qfrc_max={qfrc_max:.6} qfrc={:?} row_aref_max={row_max:.6} root_com=({:.6},{:.6},{:.6})",
                sample.left_contact as u8 | ((sample.right_contact as u8) << 1),
                contacts.len(),
                solution.row_diagnostics.len(),
                solution.tree_qfrc[0],
                scene.world.trees[0].q[0],
                scene.world.trees[0].q[1],
                scene.world.trees[0].q[2] - COM_OFFSET_Z,
            );
            for (contact_index, contact) in contacts.iter().enumerate() {
                let geom_a = geom_name(scene, contact.geom_a);
                let geom_b = geom_name(scene, contact.geom_b);
                let condim = scene.world.geoms[contact.geom_a]
                    .condim
                    .min(scene.world.geoms[contact.geom_b].condim);
                // MuJoCo records its frame from geom1 to geom2. Newt's
                // internal normal points from geom_b into geom_a, so invert
                // it only in this oracle-facing diagnostic.
                let normal = -contact.normal_world;
                let (t1, t2) = world::tangent_basis(contact.normal_world);
                let row_indices = row_to_contact
                    .iter()
                    .enumerate()
                    .filter_map(|(row, &mapped)| (mapped == contact_index).then_some(row))
                    .collect::<Vec<_>>();
                println!(
                    " contact={contact_index} geom_pair={geom_a}/{geom_b} position={:?} normal={:?} condim={condim} frame=[{:?},{:?},{:?}] rows={:?} dist={:.9}",
                    contact.position_world,
                    normal,
                    normal,
                    t1,
                    t2,
                    row_indices,
                    -contact.penetration,
                );
            }
        },
    );
}

fn geom_name(scene: &newt::model::Scene, index: usize) -> &str {
    scene
        .geoms_by_name
        .iter()
        .find_map(|(name, &candidate)| (candidate == index).then_some(name.as_str()))
        .unwrap_or("<unnamed>")
}
