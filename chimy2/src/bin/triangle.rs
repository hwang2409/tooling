use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{FlatColorShader, FlatColorUniforms};

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

    let triangle_a = [
        Vec4::new(-0.65, -0.55, 0.0, 1.0),
        Vec4::new(0.65, -0.55, 0.0, 1.0),
        Vec4::new(0.0, 0.7, 0.0, 1.0),
    ];
    let triangle_b = [
        Vec4::new(-0.65, -0.55, 0.0, 1.0),
        Vec4::new(0.65, -0.55, 0.0, 1.0),
        Vec4::new(0.0, 0.7, 0.0, 1.0),
    ];
    let indices = [[0, 1, 2]];

    chimy2::present::run(
        "chimy2 triangle",
        800,
        600,
        max_frames,
        move |framebuffer: &mut Framebuffer, elapsed| {
            framebuffer.clear(argb8888(255, 12, 16, 24));
            let angle = elapsed * 1.5;
            let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);

            let first = FlatColorUniforms::new(
                Mat4::translate(Vec3::new(-0.18, 0.0, 0.18))
                    * Mat4::rotate(Vec3::new(0.0, 0.0, 1.0), angle),
                argb8888(255, 235, 76, 76),
            );
            let second = FlatColorUniforms::new(
                Mat4::translate(Vec3::new(0.18, 0.0, -0.18))
                    * Mat4::rotate(Vec3::new(0.0, 0.0, 1.0), -angle * 0.8),
                argb8888(255, 76, 150, 235),
            );
            pipeline.render(framebuffer, |frame, target| {
                frame.draw(target, &triangle_a, &indices, &first);
                frame.draw(target, &triangle_b, &indices, &second);
            });
        },
    )
}
