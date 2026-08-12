//! A 300-instance field rendered through one mesh submission.

use chimy2::demo::{DemoArgs, build_instancing_scene, render_instancing_scene, run_demo};
use chimy2::fb::Framebuffer;
use chimy2::present::InputState;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args = DemoArgs::from_env()?;
    let (width, height) = args.size.unwrap_or((960, 640));
    let scene = build_instancing_scene(width as f32 / height as f32);
    run_demo(
        "chimy2 mesh instancing",
        960,
        640,
        args,
        move |framebuffer: &mut Framebuffer, _: f32, _: &InputState| {
            render_instancing_scene(framebuffer, &scene);
        },
    )
}
