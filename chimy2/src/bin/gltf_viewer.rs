use chimy2::camera::OrbitController;
use chimy2::demo::{DemoArgs, run_demo};
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::gltf::{GltfAsset, submit_gltf_draws};
use chimy2::math::Vec3;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut path = PathBuf::from(format!("{}/assets/arm.gltf", env!("CARGO_MANIFEST_DIR")));
    let mut demo_arguments = Vec::new();
    while let Some(argument) = arguments.next() {
        if argument == "--frames" || argument == "--screenshot" || argument == "--size" {
            demo_arguments.push(argument);
            demo_arguments.push(arguments.next().ok_or("option needs a value")?);
        } else if argument.starts_with("--") {
            return Err(format!("unknown argument: {argument}").into());
        } else {
            path = PathBuf::from(argument);
        }
    }
    let args = DemoArgs::parse(demo_arguments.into_iter())?;
    let asset = GltfAsset::load(path)?;
    let animation = (!asset.animations.is_empty()).then_some(0);
    let mut orbit = OrbitController::new(Vec3::new(0.5, 0.5, 0.0), 3.5, 0.0, 0.0);
    run_demo(
        "chimy2 gltf viewer",
        800,
        600,
        args,
        move |framebuffer: &mut Framebuffer, elapsed, _| {
            let aspect = framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32;
            orbit.step(0.2 / 60.0, 0.0, 0.0);
            let camera = orbit.camera(1.0, aspect, 0.1, 100.0);
            framebuffer.clear(argb8888(255, 12, 16, 24));
            let draws = asset
                .scene_draws(asset.default_scene, animation, elapsed)
                .expect("valid glTF scene");
            submit_gltf_draws(
                framebuffer,
                &asset,
                &draws,
                camera.view_matrix(),
                camera.projection_matrix(),
                camera.position,
            )
            .expect("valid glTF material");
        },
    )?;
    Ok(())
}
