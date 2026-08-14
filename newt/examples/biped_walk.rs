//! NEWT-18 biped walking showcase.
//!
//! The demo runs the source `stable_joint_walk` preset by default. It writes
//! a full-run MP4 and prints the same gait metrics used by the integration
//! tests.

mod biped_walk_support;
mod showcase_support;

use std::path::PathBuf;

use biped_walk_support::{GaitConfig, gait_window_sample, run_walk_observed};
use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};
use newt::math::Vec3;
use newt::model::Scene;
use newt::tree::forward_kinematics;

struct Args {
    steps: usize,
    assist_scale: f32,
    out: PathBuf,
    still: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    wireframe: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        steps: 5000,
        assist_scale: 0.8,
        out: PathBuf::from("newt-biped-walk.mp4"),
        still: None,
        out_dir: None,
        wireframe: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--steps" => args.steps = it.next().expect("--steps value").parse().unwrap(),
            "--no-assist" => args.assist_scale = 0.0,
            "--out" => args.out = PathBuf::from(it.next().expect("output path")),
            "--still" => args.still = Some(PathBuf::from(it.next().expect("still path"))),
            "--out-dir" | "--frames-dir" => {
                args.out_dir = Some(PathBuf::from(it.next().expect("output directory value")))
            }
            "--wireframe" => args.wireframe = true,
            _ => panic!("unknown argument: {arg}"),
        }
    }
    args
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    if let Some(out_dir) = &args.out_dir {
        std::fs::create_dir_all(out_dir)?;
    }
    let mut config = if args.assist_scale == 0.0 {
        GaitConfig::joint_walk(args.steps)
    } else {
        GaitConfig::stable_joint_walk(args.steps)
    };
    config.assist_scale = args.assist_scale;
    let frame_stride = (args.steps / 8).max(1);
    let video_mode = !args.wireframe && args.still.is_none() && args.out_dir.is_none();
    let mut video = video_mode
        .then(|| showcase_support::VideoWriter::new(&args.out))
        .transpose()?;
    let out_dir = args.out_dir.clone();
    let mut window_start_step = 0;
    let mut window_start_x = 0.0;
    let mut window_max_clearance: f32 = 0.0;
    let result = run_walk_observed(config, |step, scene| {
        let sample = gait_window_sample(scene);
        if step == 1 {
            window_start_x = sample.root_x;
        }
        window_max_clearance = window_max_clearance
            .max(sample.left_clearance)
            .max(sample.right_clearance);
        if video_mode {
            if step % showcase_support::SIM_STEPS_PER_VIDEO_FRAME == 0 || step == args.steps {
                video
                    .as_mut()
                    .expect("video writer")
                    .push(&render_frame_buffer(scene))
                    .expect("write video frame");
            }
        } else if let Some(path) = &args.still {
            if step == args.steps {
                write_ppm(path, &render_frame_buffer(scene)).expect("write still frame");
            }
        } else if step % frame_stride == 0 || step == args.steps {
            let frame = (step / frame_stride).min(8);
            let path = out_dir
                .as_ref()
                .expect("frame directory")
                .join(format!("frame-{frame:02}.ppm"));
            if args.wireframe {
                render_wireframe(scene, &path).expect("write PPM frame");
            } else {
                render_frame(scene, &path).expect("write PPM frame");
            }
            println!(
                "frame {frame:02}: steps={} assist_scale={:.1} window_distance={:.4} window_clearance={:.4} root_x={:.4} root_z={:.4} speed={:.4} contacts=({},{})",
                step - window_start_step,
                config.assist_scale,
                sample.root_x - window_start_x,
                window_max_clearance,
                sample.root_x,
                sample.root_z,
                sample.forward_speed,
                sample.left_contact,
                sample.right_contact,
            );
            window_start_step = step;
            window_start_x = sample.root_x;
            window_max_clearance = 0.0;
        }
    });
    if let Some(writer) = video {
        writer.finish()?;
        println!(
            "wrote {} (960x640, {} fps, {} sim steps per video frame)",
            args.out.display(),
            showcase_support::VIDEO_FPS,
            showcase_support::SIM_STEPS_PER_VIDEO_FRAME,
        );
    }
    println!(
        "walk: assist_scale={:.1} steps={} distance={:.4} cadence={:.2} bpm step_length={:.4} stride_length={:.4} duty=({:.4},{:.4}) clearance={:.4} self_contact_steps={} max_self_contact_force={:.6} final_root_height={:.4} final_forward_speed={:.4}",
        config.assist_scale,
        config.steps,
        result.metrics.forward_distance,
        result.metrics.cadence_bpm,
        result.metrics.mean_step_length,
        result.metrics.mean_stride_length,
        result.metrics.left_duty_factor,
        result.metrics.right_duty_factor,
        result.metrics.max_foot_clearance,
        result.metrics.self_contact_force_steps,
        result.metrics.max_self_contact_force,
        result.final_root_height,
        result.final_forward_speed,
    );
    Ok(())
}

fn render_wireframe(
    scene: &Scene,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let width = 800;
    let height = 500;
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 16, 24));
    let tree = &scene.world.trees[0];
    let poses = forward_kinematics(tree);
    let root = poses[0].0;
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        width as f32 / height as f32,
        0.05,
        100.0,
    ) * Mat4::look_at(
        CVec3::new(root.x + 1.6, root.y - 2.1, root.z + 1.25),
        CVec3::new(root.x, root.y, 0.85),
        CVec3::new(0.0, 0.0, 1.0),
    );
    for i in -12..=12 {
        let x = root.x + i as f32 * 0.25;
        let y0 = -1.5;
        let y1 = 1.5;
        draw_world_line(
            &mut fb,
            camera,
            Vec3::new(x, y0, 0.0),
            Vec3::new(x, y1, 0.0),
            width,
            height,
            argb8888(0xff, 42, 52, 64),
        );
        draw_world_line(
            &mut fb,
            camera,
            Vec3::new(root.x - 3.0, i as f32 * 0.25, 0.0),
            Vec3::new(root.x + 3.0, i as f32 * 0.25, 0.0),
            width,
            height,
            argb8888(0xff, 42, 52, 64),
        );
    }
    for (idx, link) in tree.links.iter().enumerate().skip(1) {
        if let Some(parent) = link.parent {
            draw_world_line(
                &mut fb,
                camera,
                poses[parent].0,
                poses[idx].0,
                width,
                height,
                argb8888(0xff, 226, 232, 240),
            );
        }
        draw_world_point(
            &mut fb,
            camera,
            poses[idx].0,
            width,
            height,
            argb8888(0xff, 86, 180, 233),
        );
    }
    draw_world_point(
        &mut fb,
        camera,
        poses[0].0,
        width,
        height,
        argb8888(0xff, 244, 180, 0),
    );
    for side in ["left", "right"] {
        for site in ["heel", "toe"] {
            if let Some((position, _)) = scene.site_pose(&format!("{side}_{site}_site")) {
                draw_world_point(
                    &mut fb,
                    camera,
                    position,
                    width,
                    height,
                    argb8888(0xff, 239, 68, 68),
                );
            }
        }
    }
    write_ppm(path, &fb)?;
    Ok(())
}

fn render_frame_buffer(scene: &Scene) -> Framebuffer {
    let poses = forward_kinematics(&scene.world.trees[0]);
    let root = poses[0].0;
    let items = showcase_support::world_items(&scene.world);
    showcase_support::render_items(
        &items,
        showcase_support::Composition::new(
            showcase_support::to_cvec(root + Vec3::new(0.0, 0.0, 0.45)),
            showcase_support::to_cvec(root + Vec3::new(2.4, -3.4, 2.0)),
            chimy2::math::Vec3::new(0.1, 0.55, 1.0),
        ),
        960,
        640,
        "biped walk  |  follow camera  |  step sequence",
    )
}

fn render_frame(scene: &Scene, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    write_ppm(path, &render_frame_buffer(scene))?;
    Ok(())
}

fn project(camera: Mat4, point: Vec3, width: usize, height: usize) -> Option<(i32, i32)> {
    let clip = camera * Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc_z = clip.z / clip.w;
    if !(-1.0..=1.0).contains(&ndc_z) {
        return None;
    }
    Some((
        ((clip.x / clip.w * 0.5 + 0.5) * width as f32) as i32,
        ((1.0 - (clip.y / clip.w * 0.5 + 0.5)) * height as f32) as i32,
    ))
}

fn draw_world_point(
    fb: &mut Framebuffer,
    camera: Mat4,
    point: Vec3,
    width: usize,
    height: usize,
    color: u32,
) {
    if let Some((x, y)) = project(camera, point, width, height) {
        for dx in -3..=3 {
            for dy in -3..=3 {
                let px = x + dx;
                let py = y + dy;
                if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
                    fb.put_pixel(px as usize, py as usize, color);
                }
            }
        }
    }
}

fn draw_world_line(
    fb: &mut Framebuffer,
    camera: Mat4,
    a: Vec3,
    b: Vec3,
    width: usize,
    height: usize,
    color: u32,
) {
    let (Some((mut x0, mut y0)), Some((x1, y1))) = (
        project(camera, a, width, height),
        project(camera, b, width, height),
    ) else {
        return;
    };
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        for px_offset in -1..=1 {
            for py_offset in -1..=1 {
                let px = x0 + px_offset;
                let py = y0 + py_offset;
                if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
                    fb.put_pixel(px as usize, py as usize, color);
                }
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x0 += sx;
        }
        if twice <= dx {
            error += dx;
            y0 += sy;
        }
    }
}
