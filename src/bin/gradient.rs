use chimy2::fb::{Framebuffer, argb8888};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut max_frames = None;
    while let Some(argument) = args.next() {
        if argument == "--frames" {
            let value = args.next().ok_or("--frames needs a number")?;
            max_frames = Some(value.parse::<usize>()?);
        } else {
            return Err(format!("unknown argument: {argument}").into());
        }
    }

    chimy2::present::run(
        "chimy2 gradient",
        800,
        600,
        max_frames,
        |framebuffer: &mut Framebuffer, elapsed| {
            for y in 0..framebuffer.height {
                for x in 0..framebuffer.width {
                    let red = ((x as f32 / framebuffer.width.max(1) as f32) * 255.0) as u8;
                    let green = ((y as f32 / framebuffer.height.max(1) as f32) * 255.0) as u8;
                    let blue = (((elapsed.sin() * 0.5 + 0.5) * 255.0) as u8)
                        .saturating_add((x % 32) as u8);
                    framebuffer.put_pixel(x, y, argb8888(255, red, green, blue));
                }
            }
        },
    )
}
