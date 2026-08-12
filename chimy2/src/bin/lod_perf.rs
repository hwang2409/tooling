use chimy2::demo::{LodScene, build_lod_scene, render_lod_scene};
use chimy2::fb::Framebuffer;
use chimy2::pipeline::Pipeline;
use chimy2::shaders::BlinnPhongShader;
use std::time::Instant;

fn render_forced_level_zero(framebuffer: &mut Framebuffer, scene: &LodScene) {
    framebuffer.clear(chimy2::fb::argb8888(255, 8, 10, 18));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.set_thread_count(1);
    pipeline.render(framebuffer, |frame, target| {
        for uniforms in &scene.uniforms {
            frame.draw_lod_mesh_level(target, &scene.mesh, uniforms, 0);
        }
    });
}

fn main() {
    const WIDTH: usize = 320;
    const HEIGHT: usize = 240;
    const WARMUP: usize = 5;
    const SAMPLES: usize = 20;
    let build_start = Instant::now();
    let scene = build_lod_scene(WIDTH as f32 / HEIGHT as f32);
    let selected_levels = scene
        .models
        .iter()
        .map(|model| {
            scene
                .mesh
                .select(
                    scene.camera.view_matrix(),
                    scene.projection,
                    *model,
                    WIDTH,
                    HEIGHT,
                )
                .level()
        })
        .collect::<Vec<_>>();
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let mut lod_target = Framebuffer::new(WIDTH, HEIGHT);
    let mut level_zero_target = Framebuffer::new(WIDTH, HEIGHT);
    for _ in 0..WARMUP {
        render_lod_scene(&mut lod_target, &scene);
        render_forced_level_zero(&mut level_zero_target, &scene);
    }
    let lod_start = Instant::now();
    for _ in 0..SAMPLES {
        render_lod_scene(&mut lod_target, &scene);
    }
    let lod_ms = lod_start.elapsed().as_secs_f64() * 1000.0 / SAMPLES as f64;
    let level_zero_start = Instant::now();
    for _ in 0..SAMPLES {
        render_forced_level_zero(&mut level_zero_target, &scene);
    }
    let level_zero_ms = level_zero_start.elapsed().as_secs_f64() * 1000.0 / SAMPLES as f64;
    println!("build_ms={build_ms:.3}");
    println!("selected_levels={selected_levels:?}");
    println!("lod_on_ms_per_frame={lod_ms:.3}");
    println!("forced_level0_ms_per_frame={level_zero_ms:.3}");
    println!("lod_on_over_level0={:.3}", lod_ms / level_zero_ms);
}
