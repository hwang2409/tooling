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
    let sky_colors = [
        [210, 70, 55],
        [55, 105, 220],
        [55, 180, 100],
        [205, 155, 45],
        [155, 70, 205],
        [45, 185, 195],
    ];
    let sky_faces = sky_colors
        .into_iter()
        .enumerate()
        .map(|(face, horizon)| {
            let texture = Texture::new(
                32,
                32,
                (0..32 * 32)
                    .map(|index| {
                        let x = index % 32;
                        let y = index / 32;
                        let vertical = y as f32 / 31.0;
                        let sun_distance =
                            ((x as f32 / 31.0 - 0.72).powi(2) + (vertical - 0.24).powi(2)).sqrt();
                        let sun = (1.0 - sun_distance / 0.16).clamp(0.0, 1.0);
                        let shade = 1.0 - vertical * 0.38;
                        [
                            (horizon[0] as f32 * shade + 255.0 * sun).min(255.0) as u8,
                            (horizon[1] as f32 * shade + 235.0 * sun).min(255.0) as u8,
                            (horizon[2] as f32 * shade + 150.0 * sun).min(255.0) as u8,
                            255,
                        ]
                    })
                    .collect(),
            )?;
            let name = ["px", "nx", "py", "ny", "pz", "nz"][face];
            Ok((format!("skybox_{name}"), texture))
        })
        .collect::<Result<Vec<_>, chimy2::image::ImageError>>()?;
    for (name, texture) in [
        ("checker", checker),
        ("gradient", gradient),
        ("normal_bump", normal_bump),
    ]
    .into_iter()
    .map(|(name, texture)| (name.to_string(), texture))
    .chain(sky_faces)
    {
        fs::write(assets.join(format!("{name}.qoi")), encode_qoi(&texture)?)?;
        let mut ppm = format!("P6\n{} {}\n255\n", texture.width(), texture.height()).into_bytes();
        for pixel in texture.pixels() {
            ppm.extend_from_slice(&pixel[..3]);
        }
        fs::write(assets.join(format!("{name}.ppm")), ppm)?;
    }
    Ok(())
}
