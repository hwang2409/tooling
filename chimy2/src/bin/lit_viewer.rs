use chimy2::camera::{FlyController, FlyInput, OrbitController};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::present::{InputState, run_with_input};
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use winit::keyboard::KeyCode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut path = format!("{}/assets/icosahedron.obj", env!("CARGO_MANIFEST_DIR"));
    let mut max_frames = None;
    while let Some(argument) = args.next() {
        if argument == "--frames" {
            let value = args.next().ok_or("--frames needs a number")?;
            max_frames = Some(value.parse::<usize>()?);
        } else if argument.starts_with("--") {
            return Err(format!("unknown argument: {argument}").into());
        } else {
            path = argument;
        }
    }

    let mesh = Mesh::load(path)?;
    let mut orbit = OrbitController::new(Vec3::ZERO, 5.0, 0.0, 0.0);
    let initial_camera = orbit.camera(1.0, 4.0 / 3.0, 0.1, 100.0);
    let mut fly = FlyController::new(initial_camera);
    let mut fly_mode = false;
    let mut was_toggle_down = false;
    let mut previous_elapsed = 0.0;

    run_with_input(
        "chimy2 lit viewer",
        800,
        600,
        max_frames,
        move |framebuffer: &mut Framebuffer, elapsed, input: &InputState| {
            let delta = (elapsed - previous_elapsed).clamp(0.0, 0.1);
            previous_elapsed = elapsed;
            let toggle_down = input.is_down(KeyCode::KeyF);
            if toggle_down && !was_toggle_down {
                fly_mode = !fly_mode;
                if fly_mode {
                    fly.yaw = orbit.yaw;
                    fly.pitch = orbit.pitch;
                    fly.camera = orbit.camera(
                        1.0,
                        framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32,
                        0.1,
                        100.0,
                    );
                }
            }
            was_toggle_down = toggle_down;

            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = if fly_mode {
                let forward =
                    input.is_down(KeyCode::KeyW) as i32 - input.is_down(KeyCode::KeyS) as i32;
                let right =
                    input.is_down(KeyCode::KeyD) as i32 - input.is_down(KeyCode::KeyA) as i32;
                let up =
                    input.is_down(KeyCode::Space) as i32 - input.is_down(KeyCode::ShiftLeft) as i32;
                let yaw = input.is_down(KeyCode::ArrowRight) as i32
                    - input.is_down(KeyCode::ArrowLeft) as i32;
                let pitch = input.is_down(KeyCode::ArrowDown) as i32
                    - input.is_down(KeyCode::ArrowUp) as i32;
                fly.step(
                    FlyInput {
                        forward: forward as f32,
                        right: right as f32,
                        up: up as f32,
                        yaw: yaw as f32,
                        pitch: pitch as f32,
                    },
                    delta,
                );
                fly.camera.aspect = aspect;
                fly.camera
            } else {
                let yaw = input.is_down(KeyCode::ArrowRight) as i32
                    - input.is_down(KeyCode::ArrowLeft) as i32;
                let pitch = input.is_down(KeyCode::ArrowDown) as i32
                    - input.is_down(KeyCode::ArrowUp) as i32;
                orbit.step(yaw as f32 * delta, pitch as f32 * delta, 0.0);
                orbit.camera(1.0, aspect, 0.1, 100.0)
            };

            framebuffer.clear(argb8888(255, 12, 16, 24));
            let uniforms = BlinnPhongUniforms::new(
                Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.5),
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.02, 0.02, 0.02),
                Vec3::new(0.78, 0.42, 0.12),
                Vec3::new(0.9, 0.9, 0.9),
                32.0,
                camera.position,
                DirectionalLight::new(
                    Vec3::new(elapsed.sin() * 0.5, 0.7, 1.0).normalize(),
                    Vec3::new(0.8, 0.85, 1.0),
                ),
                PointLight::new(
                    Vec3::new(2.0, 2.0, 3.0),
                    Vec3::new(1.0, 0.45, 0.2),
                    1.0,
                    0.08,
                    0.02,
                ),
            );
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw_mesh(target, &mesh, &uniforms);
            });
        },
    )
}
