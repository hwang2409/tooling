//! Capture the biped solver-phase records used by the parity acceptance test.

mod biped_walk_support;

use biped_walk_support::{GaitConfig, run_walk_with_solver_phase_observed};
use newt::contact::Contact;
use newt::solver::SolverMode;
use newt::world::{Integrator, SolverPhaseDiagnostics};

const COM_OFFSET_Z: f32 = 0.023_298_97;

fn main() {
    let mut config = GaitConfig::stable_joint_walk(40);
    config.assist_scale = 0.4;
    let mut records = Vec::new();
    run_walk_with_solver_phase_observed(
        config,
        Integrator::Euler,
        SolverMode::Newton,
        |step, scene| {
            let phase = scene
                .world
                .solver_phase_diagnostics()
                .expect("solver phase capture enabled by the walk runner");
            records.push(render_record(step, scene, phase));
        },
    );
    println!(
        "{{\"engine\":\"newt\",\"integrator\":\"Euler\",\"solver\":\"Newton\",\"cone\":\"pyramidal\",\"iterations\":20,\"assist_scale\":0.4,\"steps\":40,\"contact_capture\":\"solver phase before world.step; post-step geometry omitted\",\"records\":[{}]}}",
        records.join(",")
    );
}

fn render_record(
    step: usize,
    scene: &newt::model::Scene,
    phase: &SolverPhaseDiagnostics,
) -> String {
    let (qpos, qvel) = extract_qpos_qvel(phase);
    let contacts = phase
        .contacts
        .iter()
        .enumerate()
        .map(|(index, contact)| render_contact(scene, index, contact, &phase.row_to_contact))
        .collect::<Vec<_>>();
    format!(
        "{{\"step\":{step},\"solver_phase\":{{\"step\":{step},\"phase\":\"solver_phase_pre_step\",\"visual_contact_mask\":{},\"solver_contact_mask\":{},\"qpos\":{},\"qvel\":{},\"contacts\":[{}],\"row_to_contact\":{}}}}}",
        visual_contact_mask(scene),
        solver_contact_mask(scene, &phase.contacts),
        json_f32_vec(&qpos),
        json_f32_vec(&qvel),
        contacts.join(","),
        json_row_mapping(&phase.row_to_contact),
    )
}

fn render_contact(
    scene: &newt::model::Scene,
    contact_index: usize,
    contact: &Contact,
    row_to_contact: &[usize],
) -> String {
    let row_indices = row_to_contact
        .iter()
        .enumerate()
        .filter_map(|(row, &mapped)| (mapped == contact_index).then_some(row))
        .collect::<Vec<_>>();
    format!(
        "{{\"geom1\":\"{}\",\"geom2\":\"{}\",\"position\":{},\"dist\":{:.9},\"row_indices\":{}}}",
        geom_name(scene, contact.geom_a),
        geom_name(scene, contact.geom_b),
        json_f32_vec(&[
            contact.position_world.x,
            contact.position_world.y,
            contact.position_world.z,
        ]),
        -contact.penetration,
        json_usize_vec(&row_indices),
    )
}

fn extract_qpos_qvel(phase: &SolverPhaseDiagnostics) -> (Vec<f32>, Vec<f32>) {
    let mut qpos = phase.qpos[..3].to_vec();
    qpos[2] -= COM_OFFSET_Z;
    qpos.extend([phase.qpos[6], phase.qpos[3], phase.qpos[4], phase.qpos[5]]);
    qpos.extend_from_slice(&phase.qpos[7..]);
    let mut qvel = phase.qvel[3..6].to_vec();
    qvel.extend_from_slice(&phase.qvel[0..3]);
    qvel.extend_from_slice(&phase.qvel[6..]);
    (qpos, qvel)
}

fn visual_contact_mask(scene: &newt::model::Scene) -> u8 {
    let left_heel = scene.site_pose("left_heel_site").unwrap().0.z;
    let left_toe = scene.site_pose("left_toe_site").unwrap().0.z;
    let right_heel = scene.site_pose("right_heel_site").unwrap().0.z;
    let right_toe = scene.site_pose("right_toe_site").unwrap().0.z;
    u8::from(left_heel <= 0.035 || left_toe <= 0.035)
        | (u8::from(right_heel <= 0.035 || right_toe <= 0.035) << 1)
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

fn json_f32_vec(values: &[f32]) -> String {
    let mut output = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&format!("{value:.9}"));
    }
    output.push(']');
    output
}

fn json_usize_vec(values: &[usize]) -> String {
    let mut output = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output.push(']');
    output
}

fn json_row_mapping(values: &[usize]) -> String {
    let mut output = String::from("[");
    for (row, contact_index) in values.iter().enumerate() {
        if row > 0 {
            output.push(',');
        }
        output.push_str(&format!(
            "{{\"row\":{row},\"contact_index\":{contact_index}}}"
        ));
    }
    output.push(']');
    output
}
