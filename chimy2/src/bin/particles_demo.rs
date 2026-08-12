//! Deterministic CPU fountain rendered as camera-facing instanced billboards.

use chimy2::demo::{DemoArgs, build_fountain_scene, render_fountain_scene, run_demo};
use chimy2::fb::Framebuffer;
use chimy2::present::InputState;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args = DemoArgs::from_env()?;
    let (width, height) = args.size.unwrap_or((960, 640));
    let mut scene = build_fountain_scene(width as f32 / height as f32);
    run_demo(
        "chimy2 particles",
        960,
        640,
        args,
        move |framebuffer: &mut Framebuffer, _: f32, _: &InputState| {
            render_fountain_scene(framebuffer, &mut scene);
        },
    )
}
