use chimy2::camera::{FlyController, FlyInput, OrbitController};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::{Texture, WrapMode};
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::present::{InputState, run_with_input};
use chimy2::shaders::{
    DirectionalLight, PointLight, TextureFilter, TexturedBlinnPhongShader,
    TexturedBlinnPhongUniforms,
};
use winit::keyboard::KeyCode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut texture_path = format!("{}/assets/checker.qoi", env!("CARGO_MANIFEST_DIR"));
    let mut max_frames = None;
    while let Some(argument) = args.next() {
        if argument == "--frames" {
            let value = args.next().ok_or("--frames needs a number")?;
            max_frames = Some(value.parse::<usize>()?);
        } else if argument.starts_with("--") {
            return Err(format!("unknown argument: {argument}").into());
        } else {
            texture_path = argument;
        }
    }

    let mesh = Mesh::parse(TEXTURED_CUBE)?;
    let texture = Texture::load(texture_path)?.with_wrap_mode(WrapMode::Repeat);
    let mut orbit = OrbitController::new(Vec3::ZERO, 5.0, 0.0, 0.0);
    let mut fly = FlyController::new(orbit.camera(1.0, 4.0 / 3.0, 0.1, 100.0));
    let mut fly_mode = false;
    let mut was_toggle_down = false;
    let mut previous_elapsed = 0.0;

    run_with_input(
        "chimy2 textured viewer",
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
            let lighting = chimy2::shaders::BlinnPhongUniforms::new(
                Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.5),
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.03, 0.03, 0.03),
                Vec3::new(0.9, 0.9, 0.9),
                Vec3::new(0.8, 0.8, 0.8),
                32.0,
                camera.position,
                DirectionalLight::new(
                    Vec3::new(-0.4, 0.7, 1.0).normalize(),
                    Vec3::new(0.9, 0.95, 1.0),
                ),
                PointLight::new(
                    Vec3::new(2.0, 2.0, 3.0),
                    Vec3::new(1.0, 0.45, 0.2),
                    1.0,
                    0.08,
                    0.02,
                ),
            );
            let uniforms =
                TexturedBlinnPhongUniforms::new(lighting, &texture, TextureFilter::Bilinear);
            let mut pipeline = Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
            pipeline.draw_mesh(framebuffer, &mesh, &uniforms);
        },
    )
}

const TEXTURED_CUBE: &str = concat!(
    "# cube with one uv square per face\n",
    "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n",
    "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\n",
    "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n",
    "f 1/1 2/2 3/3 4/4\n",
    "f 5/1 8/4 7/3 6/2\n",
    "f 1/1 5/2 6/3 2/4\n",
    "f 2/1 6/2 7/3 3/4\n",
    "f 3/1 7/2 8/3 4/4\n",
    "f 5/1 1/2 4/3 8/4\n",
);
