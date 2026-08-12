use chimy2::demo::{DemoArgs, build_morph_scene, render_morph_scene, run_demo};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::from_env()?;
    let mut scene = None;
    let mut scene_size = (0, 0);
    run_demo(
        "chimy2 morph targets",
        960,
        640,
        args,
        |framebuffer, elapsed, _input| {
            if scene_size != (framebuffer.width, framebuffer.height) {
                let aspect = framebuffer.width as f32 / framebuffer.height as f32;
                scene = Some(build_morph_scene(aspect));
                scene_size = (framebuffer.width, framebuffer.height);
            }
            let phase = elapsed * 1.4;
            let weights = [
                phase.sin() * 0.5 + 0.5,
                (phase + std::f32::consts::FRAC_PI_2).sin() * 0.5 + 0.5,
            ];
            render_morph_scene(
                framebuffer,
                scene.as_ref().expect("scene initialized"),
                weights,
            );
        },
    )
}
