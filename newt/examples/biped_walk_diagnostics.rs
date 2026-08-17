//! Print first-divergence contact and NEWT-23 row diagnostics for the biped.

mod biped_walk_support;

use biped_walk_support::{GaitConfig, gait_window_sample, run_walk_with_solver_observed};
use newt::contact::Contact;
use newt::math::Vec3;
use newt::solver::SolverMode;
use newt::world::{self, Integrator, SolverPhaseDiagnostics};

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
            let phase = scene
                .world
                .solver_phase_diagnostics()
                .expect("solver phase capture enabled by the walk runner");
            print_solver_phase(step, scene, phase);

            // Keep post-step geometry separate. It is useful for visual
            // support timing, but it was not the state solved for this step.
            let post_contacts = scene.world.detect_contacts();
            let post_rows = row_mapping(scene, &post_contacts);
            print_contact_set(
                "post_step_geometry",
                step,
                scene.world.trees[0].q[2] - COM_OFFSET_Z,
                scene,
                &post_contacts,
                &post_rows,
                &[],
                &[],
            );
            if step == 1 {
                println!(
                    " visual_contact_mask={} solver_contact_mask={}",
                    sample.left_contact as u8 | ((sample.right_contact as u8) << 1),
                    solver_contact_mask(scene, &phase.contacts),
                );
            }
        },
    );
}

fn print_solver_phase(step: usize, scene: &newt::model::Scene, phase: &SolverPhaseDiagnostics) {
    let root = Vec3::new(phase.qpos[0], phase.qpos[1], phase.qpos[2] - COM_OFFSET_Z);
    print_contact_set(
        "solver_phase",
        step,
        root.z,
        scene,
        &phase.contacts,
        &phase.row_to_contact,
        phase
            .tree_qfrc
            .first()
            .map_or(&[][..], |qfrc| qfrc.as_slice()),
        &phase.row_diagnostics,
    );
    println!(
        "solver_phase_state step={step} contact_mask={} root_com=({:.6},{:.6},{:.6}) qpos={:?} qvel={:?}",
        solver_contact_mask(scene, &phase.contacts),
        root.x,
        root.y,
        root.z,
        phase.qpos,
        phase.qvel,
    );
}

#[allow(clippy::too_many_arguments)]
fn print_contact_set(
    label: &str,
    step: usize,
    root_z: f32,
    scene: &newt::model::Scene,
    contacts: &[Contact],
    row_to_contact: &[usize],
    qfrc: &[f32],
    row_diagnostics: &[newt::solver::ConstraintRowDiagnostic],
) {
    let qfrc_max = qfrc.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let row_max = row_diagnostics
        .iter()
        .map(|row| row.reference_accel.abs())
        .fold(0.0, f32::max);
    println!(
        "{label} step={step} contacts={} rows={} row_to_contact={row_to_contact:?} qfrc_max={qfrc_max:.6} qfrc={qfrc:?} row_aref_max={row_max:.6} root_com_z={root_z:.6}",
        contacts.len(),
        row_to_contact.len(),
    );
    for (contact_index, contact) in contacts.iter().enumerate() {
        let geom_a = geom_name(scene, contact.geom_a);
        let geom_b = geom_name(scene, contact.geom_b);
        let condim = scene.world.geoms[contact.geom_a]
            .condim
            .min(scene.world.geoms[contact.geom_b].condim);
        // MuJoCo records its frame from geom1 to geom2. Newt's internal
        // normal points from geom_b into geom_a, so invert it here.
        let normal = -contact.normal_world;
        let (t1, t2) = world::tangent_basis(contact.normal_world);
        let row_indices = row_to_contact
            .iter()
            .enumerate()
            .filter_map(|(row, &mapped)| (mapped == contact_index).then_some(row))
            .collect::<Vec<_>>();
        println!(
            " {label}_contact={contact_index} geom_pair={geom_a}/{geom_b} position={:?} normal={:?} condim={condim} frame=[{:?},{:?},{:?}] rows={row_indices:?} dist={:.9}",
            contact.position_world, normal, normal, t1, t2, -contact.penetration,
        );
    }
}

fn row_mapping(scene: &newt::model::Scene, contacts: &[Contact]) -> Vec<usize> {
    contacts
        .iter()
        .enumerate()
        .flat_map(|(contact_index, contact)| {
            let condim = scene.world.geoms[contact.geom_a]
                .condim
                .min(scene.world.geoms[contact.geom_b].condim);
            let rows = if condim == 3 { 4 } else { condim as usize };
            std::iter::repeat_n(contact_index, rows)
        })
        .collect()
}

fn solver_contact_mask(scene: &newt::model::Scene, contacts: &[Contact]) -> u8 {
    let mut mask = 0;
    for contact in contacts {
        for (bit, name) in [(1, "left_foot_geom"), (2, "right_foot_geom")] {
            if geom_name(scene, contact.geom_a) == name || geom_name(scene, contact.geom_b) == name
            {
                mask |= bit;
            }
        }
    }
    mask
}

fn geom_name(scene: &newt::model::Scene, index: usize) -> &str {
    scene
        .geoms_by_name
        .iter()
        .find_map(|(name, &candidate)| (candidate == index).then_some(name.as_str()))
        .unwrap_or("<unnamed>")
}
