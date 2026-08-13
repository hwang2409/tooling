use chimy2::demo::write_ppm;
use chimy2::fb::Framebuffer;
use chimy2::present::run_with_input;
use chimy2::scene::Scene;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut scene_path = None;
    let mut screenshot = None;
    let mut size = (960_u32, 640_u32);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--scene" => {
                scene_path = Some(PathBuf::from(
                    arguments.next().ok_or("--scene needs a path")?,
                ))
            }
            "--screenshot" => {
                screenshot = Some(PathBuf::from(
                    arguments.next().ok_or("--screenshot needs a path")?,
                ))
            }
            "--size" => {
                let value = arguments.next().ok_or("--size needs WxH")?;
                let (width, height) = value.split_once(['x', 'X']).ok_or("--size needs WxH")?;
                size = (width.parse()?, height.parse()?);
                if size.0 == 0 || size.1 == 0 {
                    return Err("--size dimensions must be non-zero".into());
                }
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let scene_path = scene_path.ok_or("--scene needs a path")?;
    let scene = Scene::load(&scene_path)?;
    let root = scene_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    if let Some(path) = screenshot {
        let mut framebuffer = Framebuffer::new(size.0 as usize, size.1 as usize);
        scene.render(&mut framebuffer, &root)?;
        write_ppm(path, &framebuffer)?;
        return Ok(());
    }
    run_with_input(
        "chimy2 scene viewer",
        size.0,
        size.1,
        None,
        move |framebuffer, _, _| {
            scene
                .render(framebuffer, &root)
                .expect("scene render failed");
        },
    )?;
    Ok(())
}
