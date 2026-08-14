//! Deterministic NEWT-18 no-assist parameter sweep.

mod biped_walk_support;

use biped_walk_support::{GaitConfig, run_walk};

const SPEED_SCALES: [f32; 3] = [0.8, 1.0, 1.2];
const AMPLITUDE_SCALES: [f32; 3] = [0.9, 1.0, 1.1];
const FREQUENCY_SCALES: [f32; 3] = [0.9, 1.0, 1.1];

fn main() {
    let mut best: Option<(f32, f32, f32, f32)> = None;
    for speed_scale in SPEED_SCALES {
        for amplitude_scale in AMPLITUDE_SCALES {
            for frequency_scale in FREQUENCY_SCALES {
                let mut config = GaitConfig::joint_walk(1000);
                config.target_speed *= speed_scale;
                config.gait_amplitude *= amplitude_scale;
                config.gait_frequency *= frequency_scale;
                let result = run_walk(config);
                let distance = result.metrics.forward_distance;
                println!(
                    "run speed_scale={speed_scale:.1} amplitude_scale={amplitude_scale:.1} frequency_scale={frequency_scale:.1} distance={distance:.6} final_height={:.6} self_contacts={}",
                    result.final_root_height,
                    result.active_self_contacts.len(),
                );
                if best.is_none_or(|current| distance > current.0) {
                    best = Some((distance, speed_scale, amplitude_scale, frequency_scale));
                }
            }
        }
    }
    let (distance, speed_scale, amplitude_scale, frequency_scale) =
        best.expect("sweep has candidates");
    println!(
        "best distance={distance:.6} speed_scale={speed_scale:.1} amplitude_scale={amplitude_scale:.1} frequency_scale={frequency_scale:.1}"
    );
}
