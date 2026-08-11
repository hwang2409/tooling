//! Deterministic framebuffer post-processing.
//!
//! The renderer stores encoded sRGB colors in `Framebuffer`. A post chain
//! starts by decoding those colors into one linear float buffer. Each pass
//! declares its working color space. Conversion happens only at an explicit
//! pass boundary, and the final buffer is encoded once into the framebuffer.
//! The fixed render order is: render -> SSAA downsample -> post chain -> present.

use crate::fb::{Framebuffer, argb8888_linear};
use crate::image::{srgb_to_linear, srgb_to_linear_u8};

/// The color space used by a post-processing pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostColorSpace {
    /// Linear-light RGB. Bloom and vignette use this space.
    Linear,
    /// Encoded sRGB RGB. FXAA uses this space by design because its luma
    /// contrast follows the values seen by the display.
    EncodedSrgb,
}

/// A float post-processing image in one declared color space.
#[derive(Clone, Debug, PartialEq)]
pub struct PostBuffer {
    pub width: usize,
    pub height: usize,
    /// Pixels are `[alpha, red, green, blue]`, normalized to `0.0..=1.0`.
    pub pixels: Vec<[f32; 4]>,
    pub color_space: PostColorSpace,
}

impl PostBuffer {
    pub fn new(width: usize, height: usize, color_space: PostColorSpace) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0.0; 4]; width.saturating_mul(height)],
            color_space,
        }
    }

    fn from_framebuffer(framebuffer: &Framebuffer) -> Self {
        let mut buffer = Self::new(
            framebuffer.width,
            framebuffer.height,
            PostColorSpace::Linear,
        );
        for (destination, &pixel) in buffer.pixels.iter_mut().zip(&framebuffer.color) {
            let [alpha, red, green, blue] = pixel.to_be_bytes();
            *destination = [
                f32::from(alpha) / 255.0,
                srgb_to_linear_u8(red),
                srgb_to_linear_u8(green),
                srgb_to_linear_u8(blue),
            ];
        }
        buffer
    }

    fn convert_to(&self, color_space: PostColorSpace) -> Self {
        if self.color_space == color_space {
            return self.clone();
        }
        let mut converted = Self::new(self.width, self.height, color_space);
        for (destination, &source) in converted.pixels.iter_mut().zip(&self.pixels) {
            *destination = match color_space {
                PostColorSpace::Linear => [
                    source[0],
                    srgb_to_linear(source[1]),
                    srgb_to_linear(source[2]),
                    srgb_to_linear(source[3]),
                ],
                PostColorSpace::EncodedSrgb => [
                    source[0],
                    linear_to_encoded(source[1]),
                    linear_to_encoded(source[2]),
                    linear_to_encoded(source[3]),
                ],
            };
        }
        converted
    }

    fn write_to_framebuffer(&self, framebuffer: &mut Framebuffer) {
        for (destination, &source) in framebuffer.color.iter_mut().zip(&self.pixels) {
            *destination = match self.color_space {
                PostColorSpace::Linear => {
                    argb8888_linear(source[0], [source[1], source[2], source[3]])
                }
                PostColorSpace::EncodedSrgb => {
                    let alpha = (source[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                    let red = (source[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                    let green = (source[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                    let blue = (source[3].clamp(0.0, 1.0) * 255.0).round() as u8;
                    u32::from_be_bytes([alpha, red, green, blue])
                }
            };
        }
    }
}

/// Encodes a linear value to normalized sRGB without quantizing to u8.
/// Quantization is deferred until `write_to_framebuffer`.
fn linear_to_encoded(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// A pure framebuffer-to-framebuffer post pass.
pub trait PostPass: Send + Sync {
    fn color_space(&self) -> PostColorSpace;

    fn apply(&self, input: &PostBuffer, output: &mut PostBuffer);
}

/// An ordered, opt-in collection of post-processing passes.
#[derive(Default)]
pub struct PostChain {
    passes: Vec<Box<dyn PostPass>>,
}

impl PostChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push<P>(&mut self, pass: P)
    where
        P: PostPass + 'static,
    {
        self.passes.push(Box::new(pass));
    }

    pub fn with_pass<P>(mut self, pass: P) -> Self
    where
        P: PostPass + 'static,
    {
        self.push(pass);
        self
    }

    pub fn len(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    pub fn clear(&mut self) {
        self.passes.clear();
    }

    /// Runs the chain after rendering and before presentation.
    pub fn apply(&self, framebuffer: &mut Framebuffer) {
        if self.passes.is_empty() {
            return;
        }
        let mut current = PostBuffer::from_framebuffer(framebuffer);
        for pass in &self.passes {
            let input = current.convert_to(pass.color_space());
            let mut output = PostBuffer::new(input.width, input.height, pass.color_space());
            pass.apply(&input, &mut output);
            current = output;
        }
        current.write_to_framebuffer(framebuffer);
    }
}

pub const BLOOM_THRESHOLD: f32 = 0.70;
pub const BLOOM_STRENGTH: f32 = 0.65;

/// Normalized five-pair Gaussian taps for sigma 2.0 and radius 4.
///
/// The center tap is followed by distances one through four. The full
/// separable kernel is `center + 2 * (tap1 + ... + tap4)` and sums to one.
pub const BLOOM_GAUSSIAN_KERNEL: [f32; 5] =
    [0.20416369, 0.18017382, 0.12383154, 0.06628225, 0.02763055];

/// Returns the linear threshold extraction for one RGB pixel.
pub fn bloom_threshold_extract(rgb: [f32; 3], threshold: f32) -> [f32; 3] {
    let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    if luminance > threshold { rgb } else { [0.0; 3] }
}

/// Linear bloom: threshold, clamped separable Gaussian blur, then additive
/// recombination. The threshold is 0.70 luminance; sigma is 2.0; radius is 4;
/// boundaries clamp to the nearest pixel; recombination strength is 0.65.
#[derive(Clone, Copy, Debug, Default)]
pub struct BloomPass;

impl PostPass for BloomPass {
    fn color_space(&self) -> PostColorSpace {
        PostColorSpace::Linear
    }

    fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
        debug_assert_eq!(input.color_space, PostColorSpace::Linear);
        debug_assert_eq!(output.color_space, PostColorSpace::Linear);
        let mut extracted = vec![[0.0; 3]; input.pixels.len()];
        for (destination, pixel) in extracted.iter_mut().zip(&input.pixels) {
            *destination = bloom_threshold_extract([pixel[1], pixel[2], pixel[3]], BLOOM_THRESHOLD);
        }

        let mut horizontal = vec![[0.0; 3]; input.pixels.len()];
        for y in 0..input.height {
            for x in 0..input.width {
                let index = y * input.width + x;
                let center = extracted[index];
                for channel in 0..3 {
                    horizontal[index][channel] += center[channel] * BLOOM_GAUSSIAN_KERNEL[0];
                }
                for distance in 1..BLOOM_GAUSSIAN_KERNEL.len() {
                    let weight = BLOOM_GAUSSIAN_KERNEL[distance];
                    let left = extracted[y * input.width + x.saturating_sub(distance)];
                    let right = extracted[y * input.width
                        + x.saturating_add(distance)
                            .min(input.width.saturating_sub(1))];
                    for channel in 0..3 {
                        horizontal[index][channel] += (left[channel] + right[channel]) * weight;
                    }
                }
            }
        }

        for y in 0..input.height {
            for x in 0..input.width {
                let index = y * input.width + x;
                let mut blur = [0.0; 3];
                let center = horizontal[index];
                for channel in 0..3 {
                    blur[channel] += center[channel] * BLOOM_GAUSSIAN_KERNEL[0];
                }
                for distance in 1..BLOOM_GAUSSIAN_KERNEL.len() {
                    let weight = BLOOM_GAUSSIAN_KERNEL[distance];
                    let top = horizontal[y.saturating_sub(distance) * input.width + x];
                    let bottom = horizontal[y
                        .saturating_add(distance)
                        .min(input.height.saturating_sub(1))
                        * input.width
                        + x];
                    for channel in 0..3 {
                        blur[channel] += (top[channel] + bottom[channel]) * weight;
                    }
                }
                let source = input.pixels[index];
                output.pixels[index] = [
                    source[0],
                    (source[1] + blur[0] * BLOOM_STRENGTH).min(1.0),
                    (source[2] + blur[1] * BLOOM_STRENGTH).min(1.0),
                    (source[3] + blur[2] * BLOOM_STRENGTH).min(1.0),
                ];
            }
        }
    }
}

pub const FXAA_EDGE_THRESHOLD: f32 = 0.08;

/// Encoded-space edge contrast used by the simplified FXAA pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct FxaaPass;

impl PostPass for FxaaPass {
    fn color_space(&self) -> PostColorSpace {
        PostColorSpace::EncodedSrgb
    }

    fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
        debug_assert_eq!(input.color_space, PostColorSpace::EncodedSrgb);
        debug_assert_eq!(output.color_space, PostColorSpace::EncodedSrgb);
        for y in 0..input.height {
            for x in 0..input.width {
                let index = y * input.width + x;
                let center = input.pixels[index];
                let north = input.pixels[y.saturating_sub(1) * input.width + x];
                let south = input.pixels
                    [y.saturating_add(1).min(input.height.saturating_sub(1)) * input.width + x];
                let west = input.pixels[y * input.width + x.saturating_sub(1)];
                let east = input.pixels
                    [y * input.width + x.saturating_add(1).min(input.width.saturating_sub(1))];
                let luma =
                    |pixel: [f32; 4]| 0.2126 * pixel[1] + 0.7152 * pixel[2] + 0.0722 * pixel[3];
                let center_luma = luma(center);
                let minimum = center_luma
                    .min(luma(north))
                    .min(luma(south))
                    .min(luma(west))
                    .min(luma(east));
                let maximum = center_luma
                    .max(luma(north))
                    .max(luma(south))
                    .max(luma(west))
                    .max(luma(east));
                let contrast = maximum - minimum;
                let threshold = FXAA_EDGE_THRESHOLD * maximum.max(0.1);
                if contrast <= threshold {
                    output.pixels[index] = center;
                    continue;
                }
                let horizontal = (luma(north) + luma(south) - 2.0 * center_luma).abs();
                let vertical = (luma(west) + luma(east) - 2.0 * center_luma).abs();
                let neighbor = if horizontal >= vertical {
                    [
                        (north[1] + south[1]) * 0.5,
                        (north[2] + south[2]) * 0.5,
                        (north[3] + south[3]) * 0.5,
                    ]
                } else {
                    [
                        (west[1] + east[1]) * 0.5,
                        (west[2] + east[2]) * 0.5,
                        (west[3] + east[3]) * 0.5,
                    ]
                };
                let blend = ((contrast - threshold) / contrast.max(f32::EPSILON)).clamp(0.0, 0.75);
                output.pixels[index] = [
                    center[0],
                    center[1] + (neighbor[0] - center[1]) * blend,
                    center[2] + (neighbor[1] - center[2]) * blend,
                    center[3] + (neighbor[2] - center[3]) * blend,
                ];
            }
        }
    }
}

pub const VIGNETTE_STRENGTH: f32 = 0.65;
pub const VIGNETTE_FALLOFF: f32 = 2.0;

/// Returns the radial linear-light vignette factor. The center is one. A
/// corner is `1 - VIGNETTE_STRENGTH` because the normalized radius is one.
pub fn vignette_factor(x: usize, y: usize, width: usize, height: usize) -> f32 {
    let nx = if width <= 1 {
        0.0
    } else {
        x as f32 / (width - 1) as f32 * 2.0 - 1.0
    };
    let ny = if height <= 1 {
        0.0
    } else {
        y as f32 / (height - 1) as f32 * 2.0 - 1.0
    };
    let radius = ((nx * nx + ny * ny) * 0.5).sqrt().min(1.0);
    1.0 - VIGNETTE_STRENGTH * radius.powf(VIGNETTE_FALLOFF)
}

/// Linear radial darkening with a quadratic falloff and 0.65 edge strength.
#[derive(Clone, Copy, Debug, Default)]
pub struct VignettePass;

impl PostPass for VignettePass {
    fn color_space(&self) -> PostColorSpace {
        PostColorSpace::Linear
    }

    fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
        debug_assert_eq!(input.color_space, PostColorSpace::Linear);
        debug_assert_eq!(output.color_space, PostColorSpace::Linear);
        for y in 0..input.height {
            for x in 0..input.width {
                let index = y * input.width + x;
                let factor = vignette_factor(x, y, input.width, input.height);
                let source = input.pixels[index];
                output.pixels[index] = [
                    source[0],
                    source[1] * factor,
                    source[2] * factor,
                    source[3] * factor,
                ];
            }
        }
    }
}

pub fn bloom() -> BloomPass {
    BloomPass
}

pub fn fxaa() -> FxaaPass {
    FxaaPass
}

pub fn vignette() -> VignettePass {
    VignettePass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct IdentityPass(PostColorSpace);

    impl PostPass for IdentityPass {
        fn color_space(&self) -> PostColorSpace {
            self.0
        }

        fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
            output.pixels.clone_from(&input.pixels);
        }
    }

    #[derive(Clone, Copy)]
    struct SetLinear(f32);

    impl PostPass for SetLinear {
        fn color_space(&self) -> PostColorSpace {
            PostColorSpace::Linear
        }

        fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
            output.pixels.clone_from(&input.pixels);
            for pixel in &mut output.pixels {
                pixel[1] = self.0;
            }
        }
    }

    #[derive(Clone, Copy)]
    struct AddLinear(f32);

    impl PostPass for AddLinear {
        fn color_space(&self) -> PostColorSpace {
            PostColorSpace::Linear
        }

        fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
            output.pixels.clone_from(&input.pixels);
            for pixel in &mut output.pixels {
                pixel[1] += self.0;
            }
        }
    }

    #[test]
    fn gaussian_kernel_weights_sum_to_one() {
        let sum = BLOOM_GAUSSIAN_KERNEL[0]
            + 2.0 * BLOOM_GAUSSIAN_KERNEL[1..].iter().copied().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum is {sum}");
        assert_eq!(BLOOM_GAUSSIAN_KERNEL[0], 0.20416369);
        assert_eq!(BLOOM_GAUSSIAN_KERNEL[4], 0.02763055);
    }

    #[test]
    fn threshold_extracts_known_bright_pixel() {
        assert_eq!(
            bloom_threshold_extract([1.0, 0.8, 0.6], BLOOM_THRESHOLD),
            [1.0, 0.8, 0.6]
        );
        assert_eq!(
            bloom_threshold_extract([0.2, 0.2, 0.2], BLOOM_THRESHOLD),
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn vignette_factor_is_one_at_center_and_hand_computed_at_corner() {
        assert_eq!(vignette_factor(1, 1, 3, 3), 1.0);
        assert_eq!(vignette_factor(0, 0, 3, 3), 1.0 - VIGNETTE_STRENGTH);
    }

    #[test]
    fn identity_chain_is_byte_transparent() {
        let mut framebuffer = Framebuffer::new(3, 2);
        framebuffer.color = vec![
            0x00000000, 0x10204080, 0x80abcdef, 0xff010203, 0xff7f8081, 0xffffffff,
        ];
        let expected = framebuffer.color.clone();
        let chain = PostChain::new()
            .with_pass(IdentityPass(PostColorSpace::Linear))
            .with_pass(IdentityPass(PostColorSpace::EncodedSrgb))
            .with_pass(IdentityPass(PostColorSpace::Linear));
        chain.apply(&mut framebuffer);
        assert_eq!(framebuffer.color, expected);
    }

    #[test]
    fn encoded_identity_between_linear_passes_is_byte_transparent() {
        let mut direct = Framebuffer::new(1, 1);
        let mut bridged = direct.clone();
        let direct_chain = PostChain::new()
            .with_pass(SetLinear(0.0002))
            .with_pass(AddLinear(0.017));
        let bridged_chain = PostChain::new()
            .with_pass(SetLinear(0.0002))
            .with_pass(IdentityPass(PostColorSpace::EncodedSrgb))
            .with_pass(AddLinear(0.017));
        direct_chain.apply(&mut direct);
        bridged_chain.apply(&mut bridged);
        assert_eq!(bridged.color, direct.color);
    }

    #[test]
    fn fxaa_blends_across_a_vertical_edge() {
        let mut framebuffer = Framebuffer::new(4, 3);
        for y in 0..3 {
            for x in 2..4 {
                framebuffer.color[y * 4 + x] = 0xffff_ffff;
            }
        }
        let before = framebuffer.color.clone();
        PostChain::new().with_pass(FxaaPass).apply(&mut framebuffer);
        assert_ne!(framebuffer.color[5], before[5]);
    }

    #[test]
    fn empty_chain_does_not_touch_bytes() {
        let mut framebuffer = Framebuffer::new(1, 1);
        framebuffer.color[0] = 0x12345678;
        PostChain::new().apply(&mut framebuffer);
        assert_eq!(framebuffer.color[0], 0x12345678);
    }
}
