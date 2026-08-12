//! Shared entry point for the M7 demo binaries.
//!
//! Every demo takes the same three flags:
//!
//! - `--frames N` — exit after `N` frames. In windowed mode the exit is
//!   governed by [`crate::present::run_with_input`]. In screenshot mode this
//!   drives the number of synthetic ticks used to advance the demo before the
//!   final frame is written.
//! - `--screenshot <path.ppm>` — render headless and write the framebuffer as
//!   a binary PPM. Combines with `--frames` to freeze motion at a pose.
//! - `--size WxH` — override the demo's default window size. Screenshot mode
//!   uses this size directly for the offscreen framebuffer.
//! - `--ssaa` — enable 2x linear-light supersampling in demos that opt in.
//! - `--bloom`, `--fxaa`, `--vignette`, `--ssao` — enable post-processing passes.
//! - `--hdr` — keep linear HDR values through the render and bloom stages.
//! - `--exposure X` — scale linear HDR values before ACES tonemapping.
//! - `--ibl` — use the load-time image-based lighting GGX variant.

use crate::fb::Framebuffer;
use crate::image::Texture;
use crate::math::{Mat4, Vec2, Vec3};
use crate::mesh::{Mesh, MeshVertex};
use crate::postfx::{AcesTonemapPass, BloomPass, FxaaPass, PostChain, SsaoPass, VignettePass};
use crate::present::{InputState, run_with_input};
use std::error::Error;
use std::f32::consts::{PI, TAU};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const SCREENSHOT_FPS: f32 = 60.0;

#[derive(Clone, Debug, PartialEq)]
pub struct DemoArgs {
    pub frames: Option<usize>,
    pub screenshot: Option<PathBuf>,
    pub size: Option<(u32, u32)>,
    pub ssaa: bool,
    pub bloom: bool,
    pub fxaa: bool,
    pub vignette: bool,
    pub ssao: bool,
    pub hdr: bool,
    pub ibl: bool,
    pub exposure: f32,
}

impl Default for DemoArgs {
    fn default() -> Self {
        Self {
            frames: None,
            screenshot: None,
            size: None,
            ssaa: false,
            bloom: false,
            fxaa: false,
            vignette: false,
            ssao: false,
            hdr: false,
            ibl: false,
            exposure: 1.0,
        }
    }
}

impl DemoArgs {
    pub fn parse<I>(mut source: I) -> Result<Self, String>
    where
        I: Iterator<Item = String>,
    {
        let mut args = Self::default();
        while let Some(argument) = source.next() {
            match argument.as_str() {
                "--frames" => {
                    let value = source
                        .next()
                        .ok_or_else(|| "--frames needs a number".to_string())?;
                    let frames = value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --frames value {value}"))?;
                    args.frames = Some(frames);
                }
                "--screenshot" => {
                    let value = source
                        .next()
                        .ok_or_else(|| "--screenshot needs a path".to_string())?;
                    args.screenshot = Some(PathBuf::from(value));
                }
                "--size" => {
                    let value = source
                        .next()
                        .ok_or_else(|| "--size needs WxH".to_string())?;
                    args.size = Some(parse_size(&value)?);
                }
                "--ssaa" => args.ssaa = true,
                "--bloom" => args.bloom = true,
                "--fxaa" => args.fxaa = true,
                "--vignette" => args.vignette = true,
                "--ssao" => args.ssao = true,
                "--hdr" => args.hdr = true,
                "--ibl" => args.ibl = true,
                "--exposure" => {
                    let value = source
                        .next()
                        .ok_or_else(|| "--exposure needs a number".to_string())?;
                    args.exposure = value
                        .parse::<f32>()
                        .map_err(|_| format!("invalid --exposure value {value}"))?;
                }
                other => {
                    return Err(format!("unexpected argument: {other}"));
                }
            }
        }
        Ok(args)
    }

    pub fn from_env() -> Result<Self, String> {
        Self::parse(std::env::args().skip(1))
    }

    pub fn post_chain(&self) -> PostChain {
        let mut chain = PostChain::new();
        // SSAO needs the frame projection. Demos that render through a
        // pipeline add it there with `post_chain_with_projection` so it runs
        // before these color-only passes.
        if self.bloom {
            chain.push(BloomPass);
        }
        if self.hdr {
            if self.vignette {
                chain.push(VignettePass);
            }
            chain.push(AcesTonemapPass::new(self.exposure));
            if self.fxaa {
                chain.push(FxaaPass);
            }
        } else {
            if self.fxaa {
                chain.push(FxaaPass);
            }
            if self.vignette {
                chain.push(VignettePass);
            }
        }
        chain
    }

    /// Builds a chain with SSAO first, using the projection for this frame.
    pub fn post_chain_with_projection(&self, projection: Mat4) -> PostChain {
        let mut chain = PostChain::new();
        if self.ssao {
            chain.push(SsaoPass::new(projection));
        }
        if self.bloom {
            chain.push(BloomPass);
        }
        if self.hdr {
            if self.vignette {
                chain.push(VignettePass);
            }
            chain.push(AcesTonemapPass::new(self.exposure));
            if self.fxaa {
                chain.push(FxaaPass);
            }
        } else {
            if self.fxaa {
                chain.push(FxaaPass);
            }
            if self.vignette {
                chain.push(VignettePass);
            }
        }
        chain
    }
}

fn parse_size(value: &str) -> Result<(u32, u32), String> {
    let (width, height) = value
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("--size wants WxH, got {value}"))?;
    let width = width
        .parse::<u32>()
        .map_err(|_| format!("invalid width in --size {value}"))?;
    let height = height
        .parse::<u32>()
        .map_err(|_| format!("invalid height in --size {value}"))?;
    if width == 0 || height == 0 {
        return Err(format!("--size dimensions must be non-zero, got {value}"));
    }
    Ok((width, height))
}

pub fn run_demo<F>(
    title: &str,
    default_width: u32,
    default_height: u32,
    args: DemoArgs,
    mut draw: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut(&mut Framebuffer, f32, &InputState),
{
    let (width, height) = args.size.unwrap_or((default_width, default_height));
    let post_chain = args.post_chain();
    if let Some(path) = args.screenshot.as_ref() {
        let frames = args.frames.unwrap_or(60).max(1);
        let mut framebuffer = Framebuffer::new(width as usize, height as usize);
        let input = InputState::default();
        for frame in 0..frames {
            let elapsed = frame as f32 / SCREENSHOT_FPS;
            draw(&mut framebuffer, elapsed, &input);
            post_chain.apply(&mut framebuffer);
        }
        write_ppm(path, &framebuffer)?;
        Ok(())
    } else {
        run_with_input(
            title,
            width,
            height,
            args.frames,
            move |framebuffer, elapsed, input| {
                draw(framebuffer, elapsed, input);
                post_chain.apply(framebuffer);
            },
        )
    }
}

pub fn write_ppm(path: impl AsRef<Path>, framebuffer: &Framebuffer) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "P6")?;
    writeln!(writer, "{} {}", framebuffer.width, framebuffer.height)?;
    writeln!(writer, "255")?;
    let mut row = Vec::with_capacity(framebuffer.width * 3);
    for y in 0..framebuffer.height {
        row.clear();
        let start = y * framebuffer.width;
        for &pixel in &framebuffer.color[start..start + framebuffer.width] {
            let [_, red, green, blue] = pixel.to_be_bytes();
            row.push(red);
            row.push(green);
            row.push(blue);
        }
        writer.write_all(&row)?;
    }
    writer.flush()?;
    Ok(())
}

/// UV sphere with per-vertex normals and spherical UVs.
///
/// `segments` is the number of longitude slices; `rings` is the number of
/// latitude bands. The seam duplicates its longitude column so texturing is
/// continuous.
pub fn uv_sphere(radius: f32, rings: usize, segments: usize) -> Mesh {
    let rings = rings.max(2);
    let segments = segments.max(3);
    let mut vertices = Vec::with_capacity((rings + 1) * (segments + 1));
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = v * PI;
        let (sin_phi, cos_phi) = phi.sin_cos();
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let theta = u * TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let normal = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            vertices.push(MeshVertex::new(
                normal * radius,
                Some(Vec2::new(u, v)),
                Some(normal),
            ));
        }
    }
    let mut triangles = Vec::with_capacity(rings * segments * 2);
    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * stride + segment;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            triangles.push([a, b, c]);
            triangles.push([b, d, c]);
        }
    }
    Mesh::new(vertices, triangles)
}

/// XZ plane centred at the origin with a resolution grid.
///
/// `uv_repeat` multiplies texture coordinates. A value of `1.0` maps the whole
/// texture across the plane. Larger values tile the texture, and pair with
/// [`crate::image::WrapMode::Repeat`] on the sampled texture.
pub fn plane_xz(width: f32, depth: f32, subdivisions: usize, uv_repeat: f32) -> Mesh {
    let subdivisions = subdivisions.max(1);
    let mut vertices = Vec::with_capacity((subdivisions + 1) * (subdivisions + 1));
    let half_width = width * 0.5;
    let half_depth = depth * 0.5;
    for row in 0..=subdivisions {
        let v = row as f32 / subdivisions as f32;
        let z = -half_depth + v * depth;
        for column in 0..=subdivisions {
            let u = column as f32 / subdivisions as f32;
            let x = -half_width + u * width;
            vertices.push(MeshVertex::new(
                Vec3::new(x, 0.0, z),
                Some(Vec2::new(u * uv_repeat, v * uv_repeat)),
                Some(Vec3::new(0.0, 1.0, 0.0)),
            ));
        }
    }
    let stride = subdivisions + 1;
    let mut triangles = Vec::with_capacity(subdivisions * subdivisions * 2);
    for row in 0..subdivisions {
        for column in 0..subdivisions {
            let a = row * stride + column;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            triangles.push([a, c, b]);
            triangles.push([b, c, d]);
        }
    }
    Mesh::new(vertices, triangles)
}

/// Axis-aligned cube with per-face UVs and outward normals.
pub fn cube_with_uvs(half_extent: f32) -> Mesh {
    let e = half_extent;
    let mut vertices = Vec::with_capacity(24);
    let mut triangles = Vec::with_capacity(12);
    let faces: [(Vec3, Vec3, Vec3, Vec3, Vec3); 6] = [
        (
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-e, -e, e),
            Vec3::new(e, -e, e),
            Vec3::new(e, e, e),
            Vec3::new(-e, e, e),
        ),
        (
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(e, -e, -e),
            Vec3::new(-e, -e, -e),
            Vec3::new(-e, e, -e),
            Vec3::new(e, e, -e),
        ),
        (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(e, -e, e),
            Vec3::new(e, -e, -e),
            Vec3::new(e, e, -e),
            Vec3::new(e, e, e),
        ),
        (
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(-e, -e, -e),
            Vec3::new(-e, -e, e),
            Vec3::new(-e, e, e),
            Vec3::new(-e, e, -e),
        ),
        (
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-e, e, e),
            Vec3::new(e, e, e),
            Vec3::new(e, e, -e),
            Vec3::new(-e, e, -e),
        ),
        (
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(-e, -e, -e),
            Vec3::new(e, -e, -e),
            Vec3::new(e, -e, e),
            Vec3::new(-e, -e, e),
        ),
    ];
    for (normal, a, b, c, d) in faces {
        let base = vertices.len();
        vertices.push(MeshVertex::new(a, Some(Vec2::new(0.0, 1.0)), Some(normal)));
        vertices.push(MeshVertex::new(b, Some(Vec2::new(1.0, 1.0)), Some(normal)));
        vertices.push(MeshVertex::new(c, Some(Vec2::new(1.0, 0.0)), Some(normal)));
        vertices.push(MeshVertex::new(d, Some(Vec2::new(0.0, 0.0)), Some(normal)));
        triangles.push([base, base + 1, base + 2]);
        triangles.push([base, base + 2, base + 3]);
    }
    Mesh::new(vertices, triangles)
}

/// Painterly banded planet texture for the hero scene.
///
/// The pattern is a soft latitude stripe plus a slow longitude modulation and
/// a small pseudo-noise term. It reads well on a UV sphere without a real
/// noise function.
pub fn banded_texture(size: usize) -> Texture {
    let size = size.max(2);
    let mut pixels = Vec::with_capacity(size * size);
    let base_warm = Vec3::new(0.96, 0.72, 0.44);
    let base_cool = Vec3::new(0.32, 0.50, 0.72);
    let accent = Vec3::new(0.98, 0.94, 0.86);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let latitude = (v * PI).sin();
            let stripe = ((v * 6.5).sin() * 0.5 + 0.5).powf(1.4);
            let swirl = ((u * TAU * 3.0 + (v * PI * 2.0).sin() * 1.5).sin() * 0.5 + 0.5) * 0.35;
            let hazy_bright =
                (((u * 71.0 + v * 113.0).sin() * (u * 37.0 + v * 41.0).cos()) * 0.5 + 0.5) * 0.15;
            let mixed = base_cool + (base_warm - base_cool) * stripe;
            let mixed = mixed + (accent - mixed) * swirl * latitude;
            let mixed = Vec3::new(
                (mixed.x + hazy_bright).min(1.0),
                (mixed.y + hazy_bright).min(1.0),
                (mixed.z + hazy_bright).min(1.0),
            );
            pixels.push([
                (mixed.x * 255.0) as u8,
                (mixed.y * 255.0) as u8,
                (mixed.z * 255.0) as u8,
                255,
            ]);
        }
    }
    Texture::new(size, size, pixels).expect("procedural texture has matching dimensions")
}

/// High-contrast checkerboard for perspective-correct interpolation demos.
pub fn checkerboard_texture(size: usize, tiles: usize) -> Texture {
    let size = size.max(2);
    let tiles = tiles.max(1);
    let tile_size = size / tiles;
    let mut pixels = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let tx = x / tile_size.max(1);
            let ty = y / tile_size.max(1);
            let dark = (tx + ty) % 2 == 0;
            let border_x = (x % tile_size.max(1)) < 2 || (x % tile_size.max(1)) >= tile_size - 2;
            let border_y = (y % tile_size.max(1)) < 2 || (y % tile_size.max(1)) >= tile_size - 2;
            let border = border_x || border_y;
            let pixel = if border {
                [40, 55, 70, 255]
            } else if dark {
                [235, 232, 220, 255]
            } else {
                [30, 40, 55, 255]
            };
            pixels.push(pixel);
        }
    }
    Texture::new(size, size, pixels).expect("procedural texture has matching dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::argb8888;

    #[test]
    fn parses_frames_screenshot_and_size() {
        let args = DemoArgs::parse(
            [
                "--frames",
                "12",
                "--screenshot",
                "shot.ppm",
                "--size",
                "320x200",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(args.frames, Some(12));
        assert_eq!(args.screenshot, Some(PathBuf::from("shot.ppm")));
        assert_eq!(args.size, Some((320, 200)));
    }

    #[test]
    fn parses_postfx_flags_in_chain_order() {
        let args = DemoArgs::parse(
            ["--bloom", "--fxaa", "--vignette"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(args.bloom && args.fxaa && args.vignette);
        assert_eq!(args.post_chain().len(), 3);
    }

    #[test]
    fn parses_hdr_and_exposure() {
        let args =
            DemoArgs::parse(["--hdr", "--exposure", "2.0"].into_iter().map(String::from)).unwrap();
        assert!(args.hdr);
        assert_eq!(args.exposure, 2.0);
        assert_eq!(args.post_chain().len(), 1);
    }

    #[test]
    fn parses_ibl() {
        let args = DemoArgs::parse(["--ibl"].into_iter().map(String::from)).unwrap();
        assert!(args.ibl);
    }

    #[test]
    fn parser_rejects_bare_positional_and_unknown_flag() {
        let bare = DemoArgs::parse(["asset.obj"].into_iter().map(String::from)).unwrap_err();
        assert!(bare.contains("unexpected argument"), "got {bare}");
        let flag = DemoArgs::parse(["--nope"].into_iter().map(String::from)).unwrap_err();
        assert!(flag.contains("unexpected argument"), "got {flag}");
    }

    #[test]
    fn size_rejects_zero_and_junk() {
        assert!(parse_size("0x600").is_err());
        assert!(parse_size("800x0").is_err());
        assert!(parse_size("nope").is_err());
    }

    #[test]
    fn write_ppm_round_trips_through_the_ppm_decoder() {
        let mut framebuffer = Framebuffer::new(2, 2);
        framebuffer.color = vec![
            argb8888(255, 10, 20, 30),
            argb8888(255, 40, 50, 60),
            argb8888(255, 70, 80, 90),
            argb8888(255, 100, 110, 120),
        ];
        let path = std::env::temp_dir().join("chimy2_demo_ppm_test.ppm");
        write_ppm(&path, &framebuffer).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let texture = crate::image::Texture::from_ppm(&bytes).unwrap();
        assert_eq!(texture.width(), 2);
        assert_eq!(texture.height(), 2);
        assert_eq!(
            texture.pixels(),
            vec![
                [10, 20, 30, 255],
                [40, 50, 60, 255],
                [70, 80, 90, 255],
                [100, 110, 120, 255],
            ]
        );
        std::fs::remove_file(path).ok();
    }
}
