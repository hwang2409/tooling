//! Deterministic ordered dithering for the float->8-bit output boundary.
//!
//! Banding on smooth gradients is the perceptual cost of an 8-bit sRGB
//! framebuffer. An 8x8 Bayer matrix breaks each u8 step into 64 spatial
//! phases, so a linear-light gradient that would round to the same u8 for
//! many pixels instead alternates between neighboring u8 values with a
//! high-frequency pattern the eye reads as a continuous ramp.
//!
//! Determinism is load-bearing: the CI wasm-check job compares byte output
//! across macOS and Linux, so the pattern lookup is a fixed integer table
//! and the offset is a rational scalar. There is no libm, no randomness,
//! and no hash-map iteration in this module.
//!
//! The offset is applied AFTER sRGB encoding and BEFORE the u8 round.
//! Perceptual banding lives in encoded sRGB space, so dithering there is
//! the correct place to break the visible staircase.

/// Classic 8x8 ordered Bayer matrix. Values 0..=63 with the standard
/// recursive halftone construction: each cell holds the phase at which its
/// spatial position first crosses a threshold in a uniform 64-level ramp.
pub const BAYER_8X8: [[u8; 8]; 8] = [
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
];

/// Amplitude of the dither offset in u8 steps.
///
/// One full u8 step spreads the pattern across the entire 8-bit
/// quantization interval, which is the maximum useful amplitude and
/// exactly the ticket's "~1/255" spec at the sRGB output.
pub const DITHER_AMPLITUDE: f32 = 1.0;

/// Returns the signed dither offset for pixel `(x, y)` in u8 units.
///
/// The output range is `[-0.5, +0.5) * DITHER_AMPLITUDE`. Adding this to
/// `encoded_srgb * 255.0` and rounding produces the classic 8-bit
/// ordered dither: pixels with the same input alternate between the two
/// bracketing u8 values in a 64-cell repeating tile.
#[inline]
pub fn bayer_offset(x: usize, y: usize) -> f32 {
    let value = BAYER_8X8[y & 7][x & 7];
    // (value + 0.5) / 64 - 0.5 = (value - 31.5) / 64 gives a symmetric offset
    // that treats the 64 phases as bin centers between -0.5 and +0.5.
    (f32::from(value) - 31.5) / 64.0 * DITHER_AMPLITUDE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_matrix_covers_every_value_zero_through_sixty_three_exactly_once() {
        let mut seen = [false; 64];
        for row in &BAYER_8X8 {
            for &value in row {
                assert!(value < 64, "value {value} out of range");
                assert!(!seen[value as usize], "duplicate value {value}");
                seen[value as usize] = true;
            }
        }
        assert!(seen.iter().all(|&flag| flag));
    }

    #[test]
    fn bayer_offset_range_is_centered_around_zero() {
        let mut minimum: f32 = 0.0;
        let mut maximum: f32 = 0.0;
        let mut sum: f32 = 0.0;
        for y in 0..8 {
            for x in 0..8 {
                let offset = bayer_offset(x, y);
                minimum = minimum.min(offset);
                maximum = maximum.max(offset);
                sum += offset;
            }
        }
        // The 64 phases are symmetric around zero: mean is exactly zero.
        assert!(sum.abs() < 1e-4, "mean {sum} is not zero");
        // Symmetric range around zero, magnitude just under half a u8 step.
        assert!((maximum + minimum).abs() < 1e-4);
        assert!(maximum > 0.49 && maximum < 0.5);
    }

    #[test]
    fn bayer_offset_is_periodic_and_deterministic() {
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(bayer_offset(x, y), bayer_offset(x + 8, y));
                assert_eq!(bayer_offset(x, y), bayer_offset(x, y + 8));
                assert_eq!(bayer_offset(x, y), bayer_offset(x + 16, y + 24));
            }
        }
    }
}
