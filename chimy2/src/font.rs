//! Hand-authored 8x8 bitmap font and screen-space text overlays.
//!
//! Text is a presentation overlay. Call [`Framebuffer::draw_text`] after the
//! scene's [`PostChain`](crate::postfx::PostChain) has completed. This keeps UI
//! pixels out of bloom, tonemapping, and other scene post-processing passes.

use crate::fb::Framebuffer;

pub const GLYPH_WIDTH: usize = 8;
pub const GLYPH_HEIGHT: usize = 8;
pub const GLYPH_COUNT: usize = 95;
pub const FALLBACK_GLYPH: [u8; GLYPH_HEIGHT] = [0x7e, 0x42, 0x5a, 0x66, 0x5a, 0x42, 0x7e, 0x00];

/// Printable ASCII glyphs from space (0x20) through tilde (0x7e).
///
/// Each byte is one row. The high bit is the leftmost pixel.
pub const GLYPHS: [[u8; GLYPH_HEIGHT]; GLYPH_COUNT] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // space
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00], // !
    [0x6c, 0x6c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // "
    [0x6c, 0x6c, 0xfe, 0x6c, 0xfe, 0x6c, 0x6c, 0x00], // #
    [0x18, 0x7e, 0xc0, 0x7c, 0x06, 0xfc, 0x18, 0x00], // $
    [0xc2, 0xc6, 0x0c, 0x18, 0x30, 0x66, 0xc6, 0x00], // %
    [0x38, 0x6c, 0x38, 0x76, 0xdc, 0xcc, 0x76, 0x00], // &
    [0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // '
    [0x0c, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0c, 0x00], // (
    [0x30, 0x18, 0x0c, 0x0c, 0x0c, 0x18, 0x30, 0x00], // )
    [0x00, 0x66, 0x3c, 0xff, 0x3c, 0x66, 0x00, 0x00], // *
    [0x00, 0x18, 0x18, 0x7e, 0x18, 0x18, 0x00, 0x00], // +
    [0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x18, 0x30], // ,
    [0x00, 0x00, 0x00, 0x7e, 0x00, 0x00, 0x00, 0x00], // -
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00], // .
    [0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0, 0x80, 0x00], // /
    [0x3c, 0x66, 0x6e, 0x76, 0x66, 0x66, 0x3c, 0x00], // 0
    [0x18, 0x38, 0x78, 0x18, 0x18, 0x18, 0x7e, 0x00], // 1
    [0x3c, 0x66, 0x06, 0x0c, 0x18, 0x30, 0x7e, 0x00], // 2
    [0x3c, 0x66, 0x06, 0x1c, 0x06, 0x66, 0x3c, 0x00], // 3
    [0x0c, 0x1c, 0x3c, 0x6c, 0x7e, 0x0c, 0x0c, 0x00], // 4
    [0x7e, 0x60, 0x7c, 0x06, 0x06, 0x66, 0x3c, 0x00], // 5
    [0x1c, 0x30, 0x60, 0x7c, 0x66, 0x66, 0x3c, 0x00], // 6
    [0x7e, 0x66, 0x0c, 0x18, 0x30, 0x30, 0x30, 0x00], // 7
    [0x3c, 0x66, 0x66, 0x3c, 0x66, 0x66, 0x3c, 0x00], // 8
    [0x3c, 0x66, 0x66, 0x3e, 0x06, 0x0c, 0x38, 0x00], // 9
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00], // :
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30, 0x00], // ;
    [0x0c, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0c, 0x00], // <
    [0x00, 0x00, 0x7e, 0x00, 0x7e, 0x00, 0x00, 0x00], // =
    [0x30, 0x18, 0x0c, 0x06, 0x0c, 0x18, 0x30, 0x00], // >
    [0x3c, 0x66, 0x06, 0x0c, 0x18, 0x00, 0x18, 0x00], // ?
    [0x3c, 0x66, 0x6e, 0x6e, 0x60, 0x62, 0x3c, 0x00], // @
    [0x18, 0x24, 0x42, 0x7e, 0x42, 0x42, 0x42, 0x00], // A
    [0x7c, 0x66, 0x66, 0x7c, 0x66, 0x66, 0x7c, 0x00], // B
    [0x3c, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3c, 0x00], // C
    [0x78, 0x6c, 0x66, 0x66, 0x66, 0x6c, 0x78, 0x00], // D
    [0x7e, 0x60, 0x60, 0x7c, 0x60, 0x60, 0x7e, 0x00], // E
    [0x7e, 0x60, 0x60, 0x7c, 0x60, 0x60, 0x60, 0x00], // F
    [0x3c, 0x66, 0x60, 0x6e, 0x66, 0x66, 0x3e, 0x00], // G
    [0x66, 0x66, 0x66, 0x7e, 0x66, 0x66, 0x66, 0x00], // H
    [0x3c, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3c, 0x00], // I
    [0x1e, 0x0c, 0x0c, 0x0c, 0x0c, 0x6c, 0x38, 0x00], // J
    [0x66, 0x6c, 0x78, 0x70, 0x78, 0x6c, 0x66, 0x00], // K
    [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7e, 0x00], // L
    [0x63, 0x77, 0x7f, 0x6b, 0x63, 0x63, 0x63, 0x00], // M
    [0x66, 0x76, 0x7e, 0x7e, 0x6e, 0x66, 0x66, 0x00], // N
    [0x3c, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x00], // O
    [0x7c, 0x66, 0x66, 0x7c, 0x60, 0x60, 0x60, 0x00], // P
    [0x3c, 0x66, 0x66, 0x66, 0x6e, 0x3c, 0x0e, 0x00], // Q
    [0x7c, 0x66, 0x66, 0x7c, 0x78, 0x6c, 0x66, 0x00], // R
    [0x3e, 0x60, 0x60, 0x3c, 0x06, 0x06, 0x7c, 0x00], // S
    [0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // T
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x00], // U
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x18, 0x00], // V
    [0x63, 0x63, 0x6b, 0x6b, 0x7f, 0x77, 0x63, 0x00], // W
    [0x66, 0x66, 0x3c, 0x18, 0x3c, 0x66, 0x66, 0x00], // X
    [0x66, 0x66, 0x3c, 0x18, 0x18, 0x18, 0x18, 0x00], // Y
    [0x7e, 0x06, 0x0c, 0x18, 0x30, 0x60, 0x7e, 0x00], // Z
    [0x3c, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3c, 0x00], // [
    [0xc0, 0x60, 0x30, 0x18, 0x0c, 0x06, 0x02, 0x00], // \
    [0x3c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x3c, 0x00], // ]
    [0x18, 0x3c, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00], // ^
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfe, 0x00], // _
    [0x30, 0x18, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00], // `
    [0x00, 0x00, 0x3c, 0x06, 0x3e, 0x66, 0x3e, 0x00], // a
    [0x60, 0x60, 0x7c, 0x66, 0x66, 0x66, 0x7c, 0x00], // b
    [0x00, 0x00, 0x3c, 0x66, 0x60, 0x66, 0x3c, 0x00], // c
    [0x06, 0x06, 0x3e, 0x66, 0x66, 0x66, 0x3e, 0x00], // d
    [0x00, 0x00, 0x3c, 0x66, 0x7e, 0x60, 0x3c, 0x00], // e
    [0x1c, 0x36, 0x30, 0x7c, 0x30, 0x30, 0x30, 0x00], // f
    [0x00, 0x00, 0x3e, 0x66, 0x3e, 0x06, 0x3c, 0x00], // g
    [0x60, 0x60, 0x7c, 0x66, 0x66, 0x66, 0x66, 0x00], // h
    [0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3c, 0x00], // i
    [0x06, 0x00, 0x0e, 0x06, 0x06, 0x66, 0x3c, 0x00], // j
    [0x60, 0x60, 0x66, 0x6c, 0x78, 0x6c, 0x66, 0x00], // k
    [0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3c, 0x00], // l
    [0x00, 0x00, 0x66, 0x7f, 0x7f, 0x6b, 0x63, 0x00], // m
    [0x00, 0x00, 0x7c, 0x66, 0x66, 0x66, 0x66, 0x00], // n
    [0x00, 0x00, 0x3c, 0x66, 0x66, 0x66, 0x3c, 0x00], // o
    [0x00, 0x00, 0x7c, 0x66, 0x7c, 0x60, 0x60, 0x00], // p
    [0x00, 0x00, 0x3e, 0x66, 0x3e, 0x06, 0x06, 0x00], // q
    [0x00, 0x00, 0x6c, 0x76, 0x60, 0x60, 0x60, 0x00], // r
    [0x00, 0x00, 0x3e, 0x60, 0x3c, 0x06, 0x7c, 0x00], // s
    [0x30, 0x30, 0x7c, 0x30, 0x30, 0x36, 0x1c, 0x00], // t
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3e, 0x00], // u
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x3c, 0x18, 0x00], // v
    [0x00, 0x00, 0x63, 0x6b, 0x6b, 0x7f, 0x36, 0x00], // w
    [0x00, 0x00, 0x66, 0x3c, 0x18, 0x3c, 0x66, 0x00], // x
    [0x00, 0x00, 0x66, 0x66, 0x3e, 0x06, 0x3c, 0x00], // y
    [0x00, 0x00, 0x7e, 0x0c, 0x18, 0x30, 0x7e, 0x00], // z
    [0x0e, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0e, 0x00], // {
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // |
    [0x70, 0x18, 0x18, 0x0e, 0x18, 0x18, 0x70, 0x00], // }
    [0x76, 0xdc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ~
];

pub const fn glyph(character: char) -> [u8; GLYPH_HEIGHT] {
    let code = character as u32;
    if code >= 0x20 && code <= 0x7e {
        GLYPHS[(code - 0x20) as usize]
    } else {
        FALLBACK_GLYPH
    }
}

pub fn text_extent(text: &str, scale: usize) -> (usize, usize) {
    if scale == 0 || text.is_empty() {
        return (0, 0);
    }
    let advance = GLYPH_WIDTH.saturating_mul(scale);
    let line_height = GLYPH_HEIGHT.saturating_mul(scale);
    let mut width = 0usize;
    let mut line_width = 0usize;
    let mut lines = 1usize;
    for character in text.chars() {
        if character == '\n' {
            width = width.max(line_width);
            line_width = 0;
            lines = lines.saturating_add(1);
        } else {
            line_width = line_width.saturating_add(advance);
        }
    }
    (width.max(line_width), line_height.saturating_mul(lines))
}

/// Draws text directly into the presentation framebuffer.
///
/// The position is in screen texels. Pixels use nearest-neighbor scaling and
/// clip at every framebuffer edge. Alpha uses the framebuffer's linear blend.
pub fn draw_text(
    framebuffer: &mut Framebuffer,
    x: i32,
    y: i32,
    text: &str,
    scale: usize,
    color: u32,
) {
    if scale == 0 || framebuffer.width == 0 || framebuffer.height == 0 {
        return;
    }
    let scale = scale.min(i64::MAX as usize) as i64;
    let advance = scale.saturating_mul(GLYPH_WIDTH as i64);
    let line_height = scale.saturating_mul(GLYPH_HEIGHT as i64);
    let mut cursor_x = i64::from(x);
    let mut cursor_y = i64::from(y);
    for character in text.chars() {
        if character == '\n' {
            cursor_x = i64::from(x);
            cursor_y = cursor_y.saturating_add(line_height);
            continue;
        }
        let rows = glyph(character);
        for (row, &bits) in rows.iter().enumerate() {
            for column in 0..GLYPH_WIDTH {
                if bits & (0x80 >> column) == 0 {
                    continue;
                }
                let left = cursor_x.saturating_add((column as i64).saturating_mul(scale));
                let top = cursor_y.saturating_add((row as i64).saturating_mul(scale));
                let right = left.saturating_add(scale);
                let bottom = top.saturating_add(scale);
                let x_start = left.max(0).min(framebuffer.width as i64) as usize;
                let x_end = right.max(0).min(framebuffer.width as i64) as usize;
                let y_start = top.max(0).min(framebuffer.height as i64) as usize;
                let y_end = bottom.max(0).min(framebuffer.height as i64) as usize;
                for pixel_y in y_start..y_end {
                    for pixel_x in x_start..x_end {
                        framebuffer.blend_pixel(pixel_x, pixel_y, color);
                    }
                }
            }
        }
        cursor_x = cursor_x.saturating_add(advance);
    }
}

pub const ASCII_HUD_ROWS: [&str; 3] = [
    " !\"#$%&'()*+,-./0123456789:;<=>?",
    "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_",
    "`abcdefghijklmnopqrstuvwxyz{|}~",
];

/// Draws the deterministic full-ASCII HUD used by the demos and golden test.
pub fn draw_ascii_hud(framebuffer: &mut Framebuffer, frame_number: u32) {
    let white = crate::fb::argb8888(255, 235, 240, 255);
    let amber = crate::fb::argb8888(255, 255, 196, 96);
    framebuffer.draw_text(16, 8, "CHIMY2 PBR / BITMAP HUD", 2, white);
    let counter = format!("FRAME {frame_number:04}");
    framebuffer.draw_text(16, 30, &counter, 1, amber);
    for (row, text) in ASCII_HUD_ROWS.iter().enumerate() {
        framebuffer.draw_text(16, 42 + row as i32 * 10, text, 1, white);
    }
}

impl Framebuffer {
    /// Draws a screen-space text overlay. Call after post-processing.
    pub fn draw_text(&mut self, x: i32, y: i32, text: &str, scale: usize, color: u32) {
        draw_text(self, x, y, text, scale, color);
    }

    pub fn text_extent(text: &str, scale: usize) -> (usize, usize) {
        text_extent(text, scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::{argb8888, blend_argb8888_linear};
    use crate::postfx::{AcesTonemapPass, BloomPass, PostChain};

    #[test]
    fn glyph_probe_matches_the_authored_a_mask() {
        let mut framebuffer = Framebuffer::new(8, 8);
        framebuffer.draw_text(0, 0, "A", 1, argb8888(255, 255, 255, 255));
        let white = argb8888(255, 255, 255, 255);
        let expected = [0x18, 0x24, 0x42, 0x7e, 0x42, 0x42, 0x42, 0x00];
        for (row, &bits) in expected.iter().enumerate() {
            for column in 0..GLYPH_WIDTH {
                let expected_pixel = if bits & (0x80 >> column) != 0 {
                    white
                } else {
                    0
                };
                assert_eq!(framebuffer.color[row * 8 + column], expected_pixel);
            }
        }
        /*
        assert_eq!(
            framebuffer.color,
            vec![
                0,
                0,
                argb8888(255, 255, 255, 255),
                argb8888(255, 255, 255, 255),
                0,
                0,
                0,
                0,
                0,
                argb8888(255, 255, 255, 255),
                0,
                0,
                0,
                argb8888(255, 255, 255, 255),
                0,
                0,
                argb8888(255, 255, 255, 255, 255),
                0,
                0,
                0,
                0,
                0,
                argb8888(255, 255, 255, 255, 255),
                0,
                argb8888(255, 255, 255, 255, 255),
                argb8888(255, 255, 255, 255, 255),
                argb8888(255, 255, 255, 255),
                argb8888(255, 255, 255, 255, 255),
                argb8888(255, 255, 255, 255, 255),
                argb8888(255, 255, 255, 255, 255),
                0,
                argb8888(255, 255, 255, 255),
                0,
                0,
                0,
                0,
                0,
                argb8888(255, 255, 255, 255),
                0,
                argb8888(255, 255, 255, 255),
                0,
                0,
                0,
                0,
                0,
                argb8888(255, 255, 255),
                0,
                argb8888(255, 255, 255),
                0,
                0,
                0,
                0,
                0,
                argb8888(255, 255, 255),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ); */
    }

    #[test]
    fn scale_probe_expands_each_source_pixel_to_a_block() {
        let mut framebuffer = Framebuffer::new(16, 16);
        framebuffer.draw_text(0, 0, "A", 2, argb8888(255, 255, 255, 255));
        for y in 0..16 {
            for x in 0..16 {
                let source = if y / 2 < GLYPH_HEIGHT && x / 2 < GLYPH_WIDTH {
                    glyph('A')[y / 2] & (0x80 >> (x / 2)) != 0
                } else {
                    false
                };
                assert_eq!(framebuffer.color[y * 16 + x] != 0, source, "{x},{y}");
            }
        }
    }

    #[test]
    fn clipping_probe_handles_all_edges_and_fully_offscreen_text() {
        let color = argb8888(255, 255, 255, 255);
        let mut negative = Framebuffer::new(5, 5);
        negative.draw_text(-2, -2, "A", 1, color);
        assert_eq!(negative.color[4], color);
        assert_eq!(&negative.color[5..10], &[color; 5]);
        assert_eq!(negative.color[14], color);
        assert_eq!(negative.color[19], color);
        assert_eq!(negative.color[24], color);

        let mut right = Framebuffer::new(5, 5);
        right.draw_text(3, 0, "A", 1, color);
        assert_eq!(right.color[4 * 2 + 2], 0);
        assert_eq!(right.color[2 * 5 + 4], color);
        assert_eq!(right.color[3 * 5 + 4], color);

        let mut bottom = Framebuffer::new(5, 5);
        bottom.draw_text(0, 3, "A", 1, color);
        assert_eq!(bottom.color[3 * 5 + 3], color);
        assert_eq!(bottom.color[4 * 5 + 2], color);

        let mut partial_block = Framebuffer::new(3, 3);
        partial_block.draw_text(-1, -12, "/", 2, color);
        assert_eq!(partial_block.color[0], color);
        assert_eq!(partial_block.color[3], color);

        let mut offscreen = Framebuffer::new(5, 5);
        offscreen.draw_text(20, 20, "A", 1, color);
        offscreen.draw_text(-100, -100, "A", 2, color);
        assert!(offscreen.color.iter().all(|&pixel| pixel == 0));
    }

    #[test]
    fn alpha_probe_uses_linear_blending() {
        let background = argb8888(255, 128, 128, 128);
        let source = argb8888(128, 255, 255, 255);
        let mut framebuffer = Framebuffer::new(8, 8);
        framebuffer.clear(background);
        framebuffer.draw_text(0, 0, "A", 1, source);
        assert_eq!(
            framebuffer.color[3],
            blend_argb8888_linear(background, source)
        );
        assert_ne!(framebuffer.color[3], source);
    }

    #[test]
    fn extent_and_newline_are_monospace_and_missing_glyphs_use_tofu() {
        assert_eq!(text_extent("AB\n?", 2), (32, 32));
        assert_eq!(text_extent("", 2), (0, 0));
        assert_eq!(glyph('\u{2603}'), FALLBACK_GLYPH);
        let mut framebuffer = Framebuffer::new(8, 8);
        framebuffer.draw_text(0, 0, "\u{2603}", 1, argb8888(255, 255, 255, 255));
        assert!(framebuffer.color.iter().any(|&pixel| pixel != 0));
    }

    #[test]
    fn ordering_keeps_text_out_of_bloom() {
        let mut framebuffer = Framebuffer::new(8, 8);
        framebuffer.clear(argb8888(255, 0, 0, 0));
        PostChain::new()
            .with_pass(BloomPass)
            .with_pass(AcesTonemapPass::new(1.0))
            .apply(&mut framebuffer);
        let text_color = argb8888(255, 255, 255, 255);
        framebuffer.draw_text(0, 0, "A", 1, text_color);
        assert_eq!(framebuffer.color[3], text_color);
    }

    #[test]
    fn rendering_is_deterministic() {
        let render = || {
            let mut framebuffer = Framebuffer::new(40, 24);
            framebuffer.draw_text(-3, 2, "ASCII 123\nxyz", 2, argb8888(192, 220, 230, 255));
            framebuffer.color
        };
        assert_eq!(render(), render());
    }
}
