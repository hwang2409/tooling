//! Humanoid arm showcase: a three-dof shoulder, muscle elbow, and two-dof
//! wrist perform a smooth reach-up, reach-across, and settle sequence.
//!
//! The model is loaded from `models/arm_humanoid.json`. The biceps and
//! triceps share the elbow hinge with opposite gears, so the rendered muscle
//! activation shows the delayed force onset of the muscle dynamics.
//!
//! Run:
//! ```text
//! cargo run --release --example arm_humanoid -- --frames 1800
//! cargo run --release --example arm_humanoid -- --frames 1800 --still /tmp/arm.ppm
//! ```

mod showcase_support;

use chimy2::fb::Framebuffer;
use chimy2::math::Vec3 as CVec3;
use newt::math::{Quat, Vec3};
use newt::model::{Scene, load_from_path};
use newt::tree::{forward_kinematics, rk4_step};
use std::path::PathBuf;

struct Args {
    frames: usize,
    out: PathBuf,
    size: (usize, usize),
    model: PathBuf,
    still: Option<PathBuf>,
    frames_dir: Option<PathBuf>,
}

fn args() -> Args {
    let mut value = Args {
        frames: 1800,
        out: PathBuf::from("newt-arm-humanoid.mp4"),
        size: (640, 360),
        model: PathBuf::from("models/arm_humanoid.json"),
        still: None,
        frames_dir: None,
    };
    let mut input = std::env::args().skip(1);
    while let Some(flag) = input.next() {
        match flag.as_str() {
            "--frames" => value.frames = input.next().unwrap().parse().unwrap(),
            "--out" => value.out = PathBuf::from(input.next().unwrap()),
            "--model" => value.model = PathBuf::from(input.next().unwrap()),
            "--size" => {
                let dimensions = input.next().unwrap();
                let (width, height) = dimensions.split_once('x').expect("--size WxH");
                value.size = (width.parse().unwrap(), height.parse().unwrap());
            }
            "--still" => value.still = Some(PathBuf::from(input.next().unwrap())),
            "--frames-dir" => value.frames_dir = Some(PathBuf::from(input.next().unwrap())),
            other => panic!("unknown argument: {other}"),
        }
    }
    value
}

const UP: [f32; 7] = [0.0, -0.20, 0.05, -0.80, 0.08, -0.10, 0.06];
const ACROSS: [f32; 7] = [0.85, -0.80, -0.18, -1.05, 0.30, 0.18, 0.12];
const SETTLE: [f32; 7] = [0.18, -0.48, 0.0, -0.36, 0.0, 0.0, 0.0];

fn target_for(frame: usize, frames: usize) -> ([f32; 7], (f32, f32), &'static str) {
    let phase = frames / 3;
    if frame < phase {
        (UP, (0.82, 0.12), "reach up")
    } else if frame < phase * 2 {
        (ACROSS, (0.88, 0.10), "reach across")
    } else {
        (SETTLE, (0.28, 0.28), "settle")
    }
}

fn set_targets(scene: &mut Scene, tree_idx: usize, targets: [f32; 7], muscle: (f32, f32)) {
    let names = [
        "shoulder_yaw_servo",
        "shoulder_pitch_servo",
        "shoulder_roll_servo",
        "biceps",
        "triceps",
        "wrist_yaw_servo",
        "wrist_pitch_servo",
    ];
    for (index, name) in names.iter().enumerate() {
        let (actuator_tree, actuator) = *scene
            .actuators_by_name
            .get(*name)
            .unwrap_or_else(|| panic!("missing actuator {name}"));
        assert_eq!(actuator_tree, tree_idx);
        let target = match index {
            3 => muscle.0,
            4 => muscle.1,
            _ => targets[index],
        };
        scene.world.trees[tree_idx].set_actuator_target(actuator, target);
    }
}

fn segment(poses: &[(Vec3, Quat)], link: usize, half_length: f32) -> (Vec3, Vec3) {
    let (center, orientation) = poses[link];
    let axis = orientation.rotate(Vec3::new(0.0, 0.0, half_length));
    (center - axis, center + axis)
}

fn render_frame(
    scene: &Scene,
    tree_idx: usize,
    width: usize,
    height: usize,
    step: usize,
    phase: &str,
) -> Framebuffer {
    let tree = &scene.world.trees[tree_idx];
    let poses = forward_kinematics(tree);
    let mut items = Vec::new();

    let torso = showcase_support::Material::new(CVec3::new(0.10, 0.14, 0.19), 0.05, 0.72);
    let skin = showcase_support::Material::new(CVec3::new(0.72, 0.31, 0.17), 0.0, 0.52);
    let skin_light = showcase_support::Material::new(CVec3::new(0.92, 0.52, 0.28), 0.0, 0.46);
    let joint = showcase_support::Material::new(CVec3::new(0.12, 0.18, 0.24), 0.25, 0.30);
    let biceps_level = tree.actuators[3].act;
    let triceps_level = tree.actuators[4].act;
    let biceps = showcase_support::Material::new(
        CVec3::new(0.38 + 0.45 * biceps_level, 0.04, 0.03),
        0.05,
        0.38,
    );
    let triceps = showcase_support::Material::new(
        CVec3::new(0.16 + 0.35 * triceps_level, 0.02, 0.03),
        0.05,
        0.42,
    );

    items.push(showcase_support::item(
        showcase_support::sphere_mesh(18, 12),
        showcase_support::transform(
            poses[0].0 + Vec3::new(0.0, 0.0, 0.08),
            Quat::IDENTITY,
            CVec3::new(0.28, 0.20, 0.48),
        ),
        torso,
    ));
    showcase_support::add_marker(&mut items, poses[1].0, 0.12, joint);
    let (upper_a, upper_b) = segment(&poses, 3, 0.20);
    let (fore_a, fore_b) = segment(&poses, 4, 0.16);
    showcase_support::add_capsule(&mut items, upper_a, upper_b, 0.095, skin);
    showcase_support::add_capsule(&mut items, fore_a, fore_b, 0.078, skin_light);

    let upper_axis = (upper_b - upper_a).normalize();
    let upper_ori = poses[3].1;
    for (offset, material) in [
        (Vec3::new(0.065, 0.0, 0.01), biceps),
        (Vec3::new(-0.065, 0.0, -0.01), triceps),
    ] {
        let start = upper_a + upper_ori.rotate(offset) + upper_axis * 0.035;
        let end = upper_b + upper_ori.rotate(offset) - upper_axis * 0.045;
        showcase_support::add_capsule(&mut items, start, end, 0.035, material);
    }

    showcase_support::add_marker(&mut items, fore_b, 0.095, joint);
    let wrist = poses[6].0;
    let hand_tip = wrist + poses[6].1.rotate(Vec3::new(0.0, 0.0, 0.18));
    showcase_support::add_capsule(&mut items, wrist, hand_tip, 0.085, skin_light);
    showcase_support::add_marker(&mut items, hand_tip, 0.075, skin_light);

    showcase_support::render_items(
        &items,
        showcase_support::composition("humanoid"),
        width,
        height,
        &format!(
            "humanoid arm  |  {phase}  |  step {step:04}  |  biceps {:.2}  triceps {:.2}",
            biceps_level, triceps_level
        ),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let value = args();
    let (width, height) = value.size;
    let mut scene = load_from_path(&value.model)
        .unwrap_or_else(|error| panic!("load {}: {error}", value.model.display()));
    let tree_idx = *scene
        .trees_by_name
        .get("humanoid_arm")
        .expect("model must define humanoid_arm");
    let tree = &mut scene.world.trees[tree_idx];
    tree.set_hinge_angle(1, 0.18);
    tree.set_hinge_angle(2, -0.52);
    tree.set_hinge_angle(3, 0.04);
    tree.set_hinge_angle(4, -0.35);
    tree.set_hinge_angle(5, 0.08);
    tree.set_hinge_angle(6, -0.06);

    let first = render_frame(&scene, tree_idx, width, height, 0, "reach up");
    if let Some(path) = value.still {
        chimy2::demo::write_ppm(path, &first)?;
        return Ok(());
    }
    if let Some(directory) = value.frames_dir {
        std::fs::create_dir_all(&directory)?;
        chimy2::demo::write_ppm(directory.join("frame-00.ppm"), &first)?;
        for step in 0..value.frames {
            let (targets, muscle, phase) = target_for(step, value.frames);
            set_targets(&mut scene, tree_idx, targets, muscle);
            rk4_step(
                &mut scene.world.trees[tree_idx],
                scene.world.gravity,
                scene.world.dt,
                |_| vec![(Vec3::ZERO, Vec3::ZERO); 8],
            );
            if step + 1 == value.frames {
                chimy2::demo::write_ppm(
                    directory.join("frame-final.ppm"),
                    &render_frame(&scene, tree_idx, width, height, step + 1, phase),
                )?;
            }
        }
        return Ok(());
    }

    let mut writer = showcase_support::VideoWriter::new(&value.out)?;
    writer.push(&first)?;
    for step in 0..value.frames {
        let (targets, muscle, phase) = target_for(step, value.frames);
        set_targets(&mut scene, tree_idx, targets, muscle);
        rk4_step(
            &mut scene.world.trees[tree_idx],
            scene.world.gravity,
            scene.world.dt,
            |_| vec![(Vec3::ZERO, Vec3::ZERO); 8],
        );
        if (step + 1) % showcase_support::SIM_STEPS_PER_VIDEO_FRAME == 0 || step + 1 == value.frames
        {
            writer.push(&render_frame(
                &scene,
                tree_idx,
                width,
                height,
                step + 1,
                phase,
            ))?;
        }
    }
    writer.finish()?;
    println!(
        "wrote {} ({} simulation steps, {} fps, {:.2}x simulation speed)",
        value.out.display(),
        value.frames,
        showcase_support::VIDEO_FPS,
        showcase_support::video_speed_factor(scene.world.dt),
    );
    Ok(())
}
