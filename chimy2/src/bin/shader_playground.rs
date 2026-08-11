//! Headless-friendly showcase for the six chimy2 shader-pack materials.
//!
//! Pass `--shader toon|psx|dither|fog|normals|wireframe`. Press `n` in the
//! window to cycle modes. All six materials use the same expanded mesh and
//! the same raster pipeline.

use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo, uv_sphere};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::present::InputState;
use chimy2::shaders::{
    DitherShader, DitherUniforms, FogShader, FogUniforms, NormalsShader, NormalsUniforms,
    PsxShader, PsxUniforms, ShaderPackVertex, ToonShader, ToonUniforms, WireframeShader,
    WireframeUniforms, expand_mesh_with_barycentrics,
};
use std::error::Error;
use winit::keyboard::KeyCode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShaderMode {
    Toon,
    Psx,
    Dither,
    Fog,
    Normals,
    Wireframe,
}

impl ShaderMode {
    const ALL: [Self; 6] = [
        Self::Toon,
        Self::Psx,
        Self::Dither,
        Self::Fog,
        Self::Normals,
        Self::Wireframe,
    ];

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "toon" => Ok(Self::Toon),
            "psx" => Ok(Self::Psx),
            "dither" => Ok(Self::Dither),
            "fog" => Ok(Self::Fog),
            "normals" => Ok(Self::Normals),
            "wireframe" | "wire" => Ok(Self::Wireframe),
            _ => Err(format!(
                "unknown shader {value}; use toon, psx, dither, fog, normals, or wireframe"
            )),
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|&mode| mode == self).unwrap_or(0)
    }
}

fn parse_args() -> Result<(ShaderMode, DemoArgs), String> {
    let mut shader = ShaderMode::Toon;
    let mut demo_arguments = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--shader" {
            let value = arguments
                .next()
                .ok_or_else(|| "--shader needs a name".to_string())?;
            shader = ShaderMode::parse(&value)?;
        } else {
            demo_arguments.push(argument);
            if demo_arguments
                .last()
                .is_some_and(|arg| arg == "--frames" || arg == "--screenshot" || arg == "--size")
            {
                let value = arguments
                    .next()
                    .ok_or_else(|| "flag needs a value".to_string())?;
                demo_arguments.push(value);
            }
        }
    }
    Ok((shader, DemoArgs::parse(demo_arguments.into_iter())?))
}

fn draw_mode(
    mode: ShaderMode,
    framebuffer: &mut Framebuffer,
    vertices: &[ShaderPackVertex],
    triangles: &[[usize; 3]],
    model: Mat4,
    view: Mat4,
    projection: Mat4,
) {
    let target_size = (framebuffer.width as u32, framebuffer.height as u32);
    match mode {
        ShaderMode::Toon => {
            let uniforms = ToonUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.84, 0.30, 0.10),
                Vec3::new(-0.4, 0.8, 1.0),
            );
            let mut pipeline = Pipeline::new(ToonShader, ToonShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw(target, vertices, triangles, &uniforms);
            });
        }
        ShaderMode::Psx => {
            let uniforms = PsxUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.18, 0.66, 0.92),
                target_size,
            );
            let mut pipeline = Pipeline::new(PsxShader, PsxShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw(target, vertices, triangles, &uniforms);
            });
        }
        ShaderMode::Dither => {
            let uniforms = DitherUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.95, 0.62, 0.16),
                target_size,
            );
            let mut pipeline = Pipeline::new(DitherShader, DitherShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw(target, vertices, triangles, &uniforms);
            });
        }
        ShaderMode::Fog => {
            let uniforms = FogUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.18, 0.72, 0.34),
                Vec3::new(0.12, 0.16, 0.28),
                3.0,
                6.0,
            );
            let mut pipeline = Pipeline::new(FogShader, FogShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw(target, vertices, triangles, &uniforms);
            });
        }
        ShaderMode::Normals => {
            let uniforms = NormalsUniforms::new(model, view, projection);
            let mut pipeline = Pipeline::new(NormalsShader, NormalsShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw(target, vertices, triangles, &uniforms);
            });
        }
        ShaderMode::Wireframe => {
            let uniforms = WireframeUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.08, 0.10, 0.16),
                Vec3::new(0.95, 0.72, 0.12),
            );
            let mut pipeline = Pipeline::new(WireframeShader, WireframeShader);
            pipeline.render(framebuffer, |frame, target| {
                frame.draw(target, vertices, triangles, &uniforms);
            });
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (initial_mode, args) = parse_args()?;
    let mesh = uv_sphere(1.35, 24, 48);
    let (vertices, triangles) = expand_mesh_with_barycentrics(&mesh);
    let mut mode = initial_mode;
    let mut was_next_down = false;

    run_demo(
        "chimy2 shader playground",
        960,
        720,
        args,
        move |framebuffer: &mut Framebuffer, elapsed: f32, input: &InputState| {
            let next_down = input.is_down(KeyCode::KeyN);
            if next_down && !was_next_down {
                let next = (mode.index() + 1) % ShaderMode::ALL.len();
                mode = ShaderMode::ALL[next];
            }
            was_next_down = next_down;

            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            let orbit = OrbitController::new(Vec3::ZERO, 4.4, elapsed * 0.28, 0.24);
            let camera = orbit.camera(1.05, aspect, 0.1, 100.0);
            let model = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), elapsed * 0.45);

            framebuffer.clear(argb8888(255, 8, 10, 16));
            draw_mode(
                mode,
                framebuffer,
                &vertices,
                &triangles,
                model,
                camera.view_matrix(),
                camera.projection_matrix(),
            );
        },
    )
}
