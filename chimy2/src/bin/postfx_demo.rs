//! Lit post-processing demo.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, cube_with_uvs, plane_xz, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::postfx::{BloomPass, FxaaPass, PostChain, SsaoPass, VignettePass};
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

fn lighting_uniform(
    camera: chimy2::camera::Camera,
    model: Mat4,
    ambient: Vec3,
    diffuse: Vec3,
    specular: Vec3,
    shininess: f32,
) -> BlinnPhongUniforms {
    BlinnPhongUniforms::new(
        model,
        camera.view_matrix(),
        camera.projection_matrix(),
        ambient,
        diffuse,
        specular,
        shininess,
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
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let ssao_enabled = args.ssao;
    let cycle = !args.bloom && !args.fxaa && !args.vignette && !ssao_enabled;
    let mesh = uv_sphere(1.35, 28, 56);
    let ground = plane_xz(8.0, 8.0, 1, 1.0);
    let cube = cube_with_uvs(0.62);

    run_demo(
        "chimy2 postfx",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, _: &InputState| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let camera = if ssao_enabled {
                OrbitController::new(Vec3::new(0.0, -0.45, 0.0), 5.4, elapsed * 0.16, 0.24)
                    .camera(1.0, aspect, 0.1, 100.0)
            } else {
                OrbitController::new(Vec3::ZERO, 4.6, elapsed * 0.24, 0.2)
                    .camera(1.0, aspect, 0.1, 100.0)
            };
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
            let ssao_uniforms = (
                lighting_uniform(
                    camera,
                    Mat4::translate(Vec3::new(0.0, -1.05, 0.0)),
                    Vec3::new(0.035, 0.04, 0.055),
                    Vec3::new(0.34, 0.38, 0.45),
                    Vec3::new(0.12, 0.14, 0.18),
                    16.0,
                ),
                lighting_uniform(
                    camera,
                    Mat4::translate(Vec3::new(-0.78, -0.43, 0.0)),
                    Vec3::new(0.08, 0.035, 0.025),
                    Vec3::new(0.85, 0.18, 0.07),
                    Vec3::new(1.0, 0.8, 0.55),
                    32.0,
                ),
                lighting_uniform(
                    camera,
                    Mat4::translate(Vec3::new(0.78, -0.43, -0.1)),
                    Vec3::new(0.025, 0.05, 0.08),
                    Vec3::new(0.08, 0.34, 0.82),
                    Vec3::new(0.55, 0.75, 1.0),
                    32.0,
                ),
            );
            let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
            if ssao_enabled {
                pipeline.set_post_chain(
                    PostChain::new().with_pass(SsaoPass::new(camera.projection_matrix())),
                );
            }
            pipeline.render(framebuffer, |frame, target| {
                if ssao_enabled {
                    frame.draw_mesh(target, &ground, &ssao_uniforms.0);
                    frame.draw_mesh(target, &cube, &ssao_uniforms.1);
                    frame.draw_mesh(target, &cube, &ssao_uniforms.2);
                } else {
                    frame.draw_mesh(target, &mesh, &uniforms);
                }
            });
            if cycle {
                cycling_chain(((elapsed * 0.75) as usize) % 8).apply(framebuffer);
            }
        },
    )
}
