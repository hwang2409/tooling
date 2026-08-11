//! Perspective-correctness showcase: a checker plane receding to the horizon.
//!
//! An affine-only interpolator swims the checker as it goes down the plane; the
//! raster core divides varyings by clip w before interpolating, so the tiles
//! stay square. A low camera makes the effect obvious.

use chimy2::camera::{Camera, OrbitController};
use chimy2::demo::{DemoArgs, checkerboard_texture, plane_xz, run_demo};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::WrapMode;
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{TextureFilter, TexturedShader, TexturedUniforms};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let floor = plane_xz(200.0, 200.0, 80, 32.0);
    let texture = checkerboard_texture(128, 2).with_wrap_mode(WrapMode::Repeat);

    run_demo(
        "chimy2 perspective floor",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let yaw = 0.4 + (elapsed * 0.08).sin() * 0.2;
            let orbit = OrbitController::new(Vec3::new(0.0, 0.0, 0.0), 4.5, yaw, 0.22);
            let camera: Camera = orbit.camera(1.05, aspect, 0.1, 400.0);

            let transform = camera.projection_matrix()
                * camera.view_matrix()
                * Mat4::translate(Vec3::new(0.0, -0.6, 0.0));
            let uniforms = TexturedUniforms::new(transform, &texture, TextureFilter::Bilinear);

            framebuffer.clear(argb8888(255, 22, 30, 42));
            let mut pipeline = Pipeline::new(TexturedShader, TexturedShader);
            pipeline.draw_mesh(framebuffer, &floor, &uniforms);
        },
    )
}
