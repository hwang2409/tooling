//! Lit post-processing demo.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::postfx::{BloomPass, FxaaPass, PostChain, VignettePass};
use chimy2::present::InputState;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};

fn cycling_chain(phase: usize) -> PostChain {
    let mut chain = PostChain::new();
    if phase & 1 != 0 {
        chain.push(BloomPass);
    }
    if phase & 2 != 0 {
        chain.push(FxaaPass);
    }
    if phase & 4 != 0 {
        chain.push(VignettePass);
    }
    chain
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let cycle = !args.bloom && !args.fxaa && !args.vignette;
    let mesh = uv_sphere(1.35, 28, 56);

    run_demo(
        "chimy2 postfx",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = OrbitController::new(Vec3::ZERO, 4.6, elapsed * 0.24, 0.2)
                .camera(1.0, aspect, 0.1, 100.0);
            let model = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.35);
            let uniforms = BlinnPhongUniforms::new(
                model,
                camera.view_matrix(),
                camera.projection_matrix(),
                Vec3::new(0.06, 0.07, 0.10),
                Vec3::new(0.95, 0.42, 0.16),
                Vec3::new(1.0, 0.85, 0.65),
                32.0,
                camera.position,
                DirectionalLight::new(
                    Vec3::new(-0.35, 0.75, 1.0).normalize(),
                    Vec3::new(1.0, 0.92, 0.82),
                ),
                PointLight::new(
                    Vec3::new(2.0, 2.0, 3.0),
                    Vec3::new(1.2, 0.38, 0.12),
                    1.0,
                    0.04,
                    0.01,
                ),
            );

            framebuffer.clear(argb8888(255, 5, 7, 13));
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw_mesh(target, &mesh, &uniforms);
            });
            if cycle {
                cycling_chain(((elapsed * 0.75) as usize) % 8).apply(framebuffer);
            }
        },
    )
}
