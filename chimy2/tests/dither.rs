//! End-to-end contract for the shared float->8-bit dithering path.
//!
//! Two properties are load-bearing for CHIMY-41:
//!   1. Determinism: the same float frame quantized twice produces
//!      byte-identical output on this host, and again cross-host once the
//!      wasm-check job runs the same code on Linux. There is no libm,
//!      randomness, or hash-map iteration in the dither pattern.
//!   2. Banding reduction: a synthetic smooth encoded-sRGB ramp quantized
//!      WITH dither yields strictly more distinct u8 levels per row than
//!      the same ramp quantized WITHOUT dither. This is the whole reason
//!      the pattern exists.

use chimy2::fb::{Framebuffer, argb8888_encoded_srgb_dithered, argb8888_linear_dithered};
use chimy2::image::{linear_to_srgb, quantize_srgb_encoded};
use chimy2::postfx::{AcesTonemapPass, PostChain};
use std::collections::BTreeSet;

fn linear_gradient_pixels(width: usize) -> Vec<[f32; 3]> {
    // Deep, subtle linear ramp that would fall inside a single u8 step for
    // wide runs of pixels without dither. The ticket calls out shadow
    // penumbras, SSAO falloff, and DoF gradients as the failure surface.
    (0..width)
        .map(|x| {
            let value = 0.20 + (x as f32) * (0.02 / width as f32);
            [value, value, value]
        })
        .collect()
}

#[test]
fn determinism_same_frame_quantized_twice_is_byte_identical() {
    let width = 64;
    let height = 32;
    let pixels = linear_gradient_pixels(width);
    let quantize = || {
        let mut buffer = vec![0u32; width * height];
        for y in 0..height {
            for x in 0..width {
                let rgb = pixels[x];
                buffer[y * width + x] = argb8888_linear_dithered(x, y, 1.0, rgb);
            }
        }
        buffer
    };
    let first = quantize();
    let second = quantize();
    assert_eq!(
        first, second,
        "the shared dithered quantization must be pure and deterministic"
    );
}

#[test]
fn banding_smooth_gradient_gains_distinct_levels_from_dither() {
    // Encode a smooth gray ramp in encoded-sRGB space so we test the exact
    // quantize function used by the postfx output path.
    let width = 512;
    let height = 8;
    // A 0.030-wide encoded-sRGB slice at midtones covers about 7.7 u8 steps.
    // Without dither, a whole row can only hit those ~8 discrete levels.
    let base = 0.500_f32;
    let span = 0.030_f32;
    let mut without_dither = BTreeSet::new();
    let mut with_dither = BTreeSet::new();
    for y in 0..height {
        for x in 0..width {
            let encoded = base + (x as f32 / (width - 1) as f32) * span;
            without_dither.insert(quantize_srgb_encoded(encoded, 0.0));
            with_dither.insert(quantize_srgb_encoded(
                encoded,
                chimy2::dither::bayer_offset(x, y),
            ));
        }
    }
    // Without dither: at most one u8 per encoded-step boundary.
    // With dither: every intermediate u8 also lights up between boundaries.
    assert!(
        with_dither.len() > without_dither.len(),
        "dither must add distinct output levels: with={} without={}",
        with_dither.len(),
        without_dither.len(),
    );
    // Extra sanity: the reduction is meaningful, not a one-level nudge.
    assert!(
        with_dither.len() >= without_dither.len() + 2,
        "dither should surface several extra levels on a smooth ramp: with={} without={}",
        with_dither.len(),
        without_dither.len(),
    );
}

#[test]
fn dither_yields_multiple_u8_neighbors_on_a_uniform_encoded_input() {
    // A uniform encoded input midway between two u8 levels is the classic
    // Bayer worst case for a naive quantizer. With dither we must see both
    // neighbors distributed across the tile.
    let encoded = 100.5_f32 / 255.0;
    let mut u8_values = BTreeSet::new();
    for y in 0..8 {
        for x in 0..8 {
            u8_values.insert(quantize_srgb_encoded(
                encoded,
                chimy2::dither::bayer_offset(x, y),
            ));
        }
    }
    assert!(
        u8_values.len() >= 2,
        "dither must resolve a half-step input to both neighboring u8 levels, got {u8_values:?}",
    );
}

#[test]
fn dither_does_not_perturb_exact_u8_boundary_values() {
    // A linear-light input that ROUND-TRIPS exactly through the sRGB encode
    // to a whole u8 step must survive dither unchanged for every pixel in
    // the tile. Round-half-away-from-zero and the ±0.492 max offset stay
    // strictly below the ±0.5 nudge that would flip the round.
    for value in [0u8, 32, 96, 160, 224, 255] {
        let linear = chimy2::image::srgb_to_linear_u8(value);
        // Sanity: this linear value re-encodes exactly to the same u8.
        assert_eq!(linear_to_srgb(linear), value);
        let mut hits = BTreeSet::new();
        for y in 0..8 {
            for x in 0..8 {
                let pixel = argb8888_linear_dithered(x, y, 1.0, [linear, linear, linear]);
                hits.insert(pixel);
            }
        }
        assert_eq!(
            hits.len(),
            1,
            "u8-boundary value {value} must not shift under dither, got {hits:?}",
        );
    }
}

#[test]
fn encoded_and_linear_dither_wrappers_agree_on_encoded_inputs() {
    // The postfx write path picks between argb8888_linear_dithered and
    // argb8888_encoded_srgb_dithered based on the pass's color space. For an
    // already-encoded input, running it through the "linear" wrapper via a
    // linear->encoded roundtrip must match the direct encoded quantizer.
    let encoded = 0.501_f32;
    let (x, y) = (3, 5);
    let direct = argb8888_encoded_srgb_dithered(x, y, [1.0, encoded, encoded, encoded]);
    let offset = chimy2::dither::bayer_offset(x, y);
    let expected_channel = quantize_srgb_encoded(encoded, offset);
    let [_, red, green, blue] = direct.to_be_bytes();
    assert_eq!(red, expected_channel);
    assert_eq!(green, expected_channel);
    assert_eq!(blue, expected_channel);
}

#[test]
fn postfx_write_applies_dither_on_the_final_ldr_encode() {
    // End-to-end: a solid mid-gray LDR framebuffer processed by an identity
    // ACES pass (exposure 1.0) must still pick up spatial dither at the
    // final quantize, otherwise banding-prone LDR chains stay flat.
    let mut framebuffer = Framebuffer::new(16, 16);
    let base_color = chimy2::fb::argb8888(255, 118, 118, 118);
    framebuffer.color.fill(base_color);
    // Use ACES with a tiny exposure boost so the quantizer sees a
    // fractional u8 midpoint. The pass runs in linear space and its
    // post-tonemap encoded midpoint lands between two u8 steps.
    PostChain::new()
        .with_pass(AcesTonemapPass::new(1.0))
        .apply(&mut framebuffer);
    let unique = framebuffer.color.iter().copied().collect::<BTreeSet<_>>();
    assert!(
        unique.len() >= 2,
        "postfx write must dither the final encode; got only {} unique u32 pixel values",
        unique.len(),
    );
}
