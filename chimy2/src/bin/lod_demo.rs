use chimy2::demo::{DemoArgs, build_lod_scene, render_lod_scene, run_demo};
use chimy2::fb::Framebuffer;
use chimy2::present::InputState;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let args = DemoArgs::from_env()?;
    let (width, height) = args.size.unwrap_or((960, 640));
    let scene = build_lod_scene(width as f32 / height as f32);
    run_demo(
        "chimy2 mesh lod",
        width,
        height,
        args,
        move |framebuffer: &mut Framebuffer, _: f32, _: &InputState| {
            render_lod_scene(framebuffer, &scene);
        },
    )
}
