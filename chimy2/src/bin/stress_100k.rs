//! 100k-triangle stress scene that logs render-time fps to stdout.
//!
//! Two numbers get reported: render-only fps (time inside `Pipeline::render`)
//! and present-inclusive fps (wall-clock between successive callback starts).
//! The gap is the event loop plus the softbuffer present, which allocates on
//! macOS every frame. See PR #6 for the criterion baseline this demo mirrors.

use chimy2::camera::{Camera, OrbitController};
use chimy2::demo::{DemoArgs, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use std::time::{Duration, Instant};

const SPHERE_RINGS: usize = 224;
const SPHERE_SEGMENTS: usize = 224;
const PERIODIC_WINDOW: Duration = Duration::from_secs(1);

struct FpsReporter {
    triangle_count: usize,
    frame_count: usize,
    render_ms_total: f64,
    render_ms_max: f64,
    wall_start: Option<Instant>,
    window_start: Option<Instant>,
    window_frames: usize,
    window_render_ms: f64,
    last_width: usize,
    last_height: usize,
}

impl FpsReporter {
    fn new(triangle_count: usize) -> Self {
        Self {
            triangle_count,
            frame_count: 0,
            render_ms_total: 0.0,
            render_ms_max: 0.0,
            wall_start: None,
            window_start: None,
            window_frames: 0,
            window_render_ms: 0.0,
            last_width: 0,
            last_height: 0,
        }
    }

    fn observe(&mut self, render_ms: f64, width: usize, height: usize) {
        let now = Instant::now();
        if self.wall_start.is_none() {
            self.wall_start = Some(now);
            self.window_start = Some(now);
        }
        self.frame_count += 1;
        self.render_ms_total += render_ms;
        if render_ms > self.render_ms_max {
            self.render_ms_max = render_ms;
        }
        self.window_frames += 1;
        self.window_render_ms += render_ms;
        self.last_width = width;
        self.last_height = height;

        let window_elapsed = self
            .window_start
            .map(|start| now - start)
            .unwrap_or_default();
        if window_elapsed >= PERIODIC_WINDOW {
            let render_avg_ms = self.window_render_ms / self.window_frames as f64;
            let render_fps = 1000.0 / render_avg_ms.max(f64::EPSILON);
            let wall_fps = self.window_frames as f64 / window_elapsed.as_secs_f64();
            println!(
                "frame {}: render {:.2} ms avg ({:.1} fps render-only), \
                 {:.1} fps present-inclusive over {} frames at {}x{}",
                self.frame_count,
                render_avg_ms,
                render_fps,
                wall_fps,
                self.window_frames,
                width,
                height,
            );
            self.window_start = Some(now);
            self.window_frames = 0;
            self.window_render_ms = 0.0;
        }
    }
}

impl Drop for FpsReporter {
    fn drop(&mut self) {
        if self.frame_count == 0 {
            println!("stress_100k: no frames rendered");
            return;
        }
        let render_avg_ms = self.render_ms_total / self.frame_count as f64;
        let render_fps = 1000.0 / render_avg_ms.max(f64::EPSILON);
        let wall_seconds = self
            .wall_start
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let wall_fps = if wall_seconds > 0.0 {
            self.frame_count as f64 / wall_seconds
        } else {
            0.0
        };
        println!(
            "stress_100k: {} frames of {} triangles at {}x{}, \
             render {:.2} ms avg / {:.2} ms max ({:.1} fps render-only), \
             {:.1} fps present-inclusive over {:.2}s",
            self.frame_count,
            self.triangle_count,
            self.last_width,
            self.last_height,
            render_avg_ms,
            self.render_ms_max,
            render_fps,
            wall_fps,
            wall_seconds,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let mesh = uv_sphere(1.5, SPHERE_RINGS, SPHERE_SEGMENTS);
    let triangle_count = mesh.indices().len();
    println!("stress_100k: rendering {triangle_count} triangles per frame");
    let mut reporter = FpsReporter::new(triangle_count);

    run_demo(
        "chimy2 stress 100k",
        1280,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let orbit = OrbitController::new(Vec3::ZERO, 4.5, elapsed * 0.25, 0.15);
            let camera: Camera = orbit.camera(1.05, aspect, 0.1, 100.0);

            let model = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.35);
            let uniforms = BlinnPhongUniforms::new(
                model,
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.10, 0.11, 0.14),
                Vec3::new(0.86, 0.48, 0.20),
                Vec3::new(0.9, 0.9, 0.9),
                48.0,
                camera.position,
                DirectionalLight::new(
                    Vec3::new(0.4, 0.75, 0.9).normalize(),
                    Vec3::new(1.05, 1.00, 0.90),
                ),
                PointLight::new(
                    Vec3::new(-2.0, 2.5, 2.5),
                    Vec3::new(0.40, 0.60, 1.10),
                    1.0,
                    0.08,
                    0.02,
                ),
            );

            framebuffer.clear(argb8888(255, 8, 10, 16));
            let render_start = Instant::now();
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw_mesh(target, &mesh, &uniforms);
            });
            let render_ms = render_start.elapsed().as_secs_f64() * 1000.0;
            reporter.observe(render_ms, framebuffer.width, framebuffer.height);
        },
    )
}
