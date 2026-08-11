use chimy2::image::{ColorSpace, Texture, encode_qoi};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    let checker = Texture::new(
        4,
        4,
        (0..16)
            .map(|index| {
                let x = index % 4;
                let y = index / 4;
                if (x + y) % 2 == 0 {
                    [235, 235, 235, 255]
                } else {
                    [30, 35, 45, 255]
                }
            })
            .collect(),
    )?;
    let gradient = Texture::new(
        4,
        4,
        (0..16)
            .map(|index| {
                let x = index % 4;
                let y = index / 4;
                [(x * 85) as u8, (y * 85) as u8, 180, 255]
            })
            .collect(),
    )?;
    let normal_bump = Texture::new_with_color_space(
        64,
        64,
        (0..64 * 64)
            .map(|index| {
                let x = index % 64;
                let y = index / 64;
                let u = x as f32 / 64.0;
                let v = y as f32 / 64.0;
                let dx = (u * std::f32::consts::TAU * 3.0).cos()
                    * (v * std::f32::consts::TAU * 2.0).cos()
                    * 0.32;
                let dy = -(u * std::f32::consts::TAU * 3.0).sin()
                    * (v * std::f32::consts::TAU * 2.0).sin()
                    * 0.32;
                let normal = chimy2::math::Vec3::new(-dx, -dy, 1.0).normalize();
                [
                    ((normal.x * 0.5 + 0.5) * 255.0).round() as u8,
                    ((normal.y * 0.5 + 0.5) * 255.0).round() as u8,
                    ((normal.z * 0.5 + 0.5) * 255.0).round() as u8,
                    255,
                ]
            })
            .collect(),
        ColorSpace::Linear,
    )?;
    for (name, texture) in [
        ("checker", checker),
        ("gradient", gradient),
        ("normal_bump", normal_bump),
    ] {
        fs::write(assets.join(format!("{name}.qoi")), encode_qoi(&texture)?)?;
        let mut ppm = format!("P6\n{} {}\n255\n", texture.width(), texture.height()).into_bytes();
        for pixel in texture.pixels() {
            ppm.extend_from_slice(&pixel[..3]);
        }
        fs::write(assets.join(format!("{name}.ppm")), ppm)?;
    }
    Ok(())
}
