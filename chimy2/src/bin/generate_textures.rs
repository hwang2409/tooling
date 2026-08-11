use chimy2::image::{Texture, encode_qoi};
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
    for (name, texture) in [("checker", checker), ("gradient", gradient)] {
        fs::write(assets.join(format!("{name}.qoi")), encode_qoi(&texture)?)?;
        let mut ppm = format!("P6\n{} {}\n255\n", texture.width(), texture.height()).into_bytes();
        for pixel in texture.pixels() {
            ppm.extend_from_slice(&pixel[..3]);
        }
        fs::write(assets.join(format!("{name}.ppm")), ppm)?;
    }
    Ok(())
}
