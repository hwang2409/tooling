//! Hand-written PPM and QOI image codecs plus CPU texture sampling.
//!
//! Texture coordinates use texel centers. A coordinate `u` maps to
//! `u * width - 0.5` in texel space before filtering. Nearest sampling then
//! selects `floor(u * width)` after wrapping. Bilinear sampling uses the four
//! neighboring texels around that center-space position. Pixel values are raw
//! 8-bit values; no sRGB or gamma conversion is applied.

use crate::math::Vec2;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

const QOI_HEADER_SIZE: usize = 14;
const QOI_END_MARKER: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 1];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageError {
    pub offset: usize,
    pub message: String,
}

impl ImageError {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl Display for ImageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ImageError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrapMode {
    Repeat,
    ClampToEdge,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Texture {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[u8; 4]>,
    pub wrap_mode: WrapMode,
}

impl Texture {
    pub fn new(width: usize, height: usize, pixels: Vec<[u8; 4]>) -> Result<Self, ImageError> {
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| ImageError::new(0, "texture dimensions overflow the pixel count"))?;
        if expected != pixels.len() {
            return Err(ImageError::new(
                0,
                format!(
                    "texture has {} pixels, expected {expected} for {width}x{height}",
                    pixels.len()
                ),
            ));
        }
        if width == 0 || height == 0 {
            return Err(ImageError::new(0, "texture dimensions must be non-zero"));
        }
        Ok(Self {
            width,
            height,
            pixels,
            wrap_mode: WrapMode::Repeat,
        })
    }

    pub fn from_ppm(bytes: &[u8]) -> Result<Self, ImageError> {
        decode_ppm(bytes)
    }

    pub fn from_qoi(bytes: &[u8]) -> Result<Self, ImageError> {
        decode_qoi(bytes)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ImageError> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .map_err(|error| ImageError::new(0, format!("{}: {error}", path.display())))?;
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("ppm") => Self::from_ppm(&bytes),
            Some("qoi") => Self::from_qoi(&bytes),
            _ => Err(ImageError::new(
                0,
                format!("unsupported image extension: {}", path.display()),
            )),
        }
    }

    pub fn with_wrap_mode(mut self, wrap_mode: WrapMode) -> Self {
        self.wrap_mode = wrap_mode;
        self
    }

    pub fn set_wrap_mode(&mut self, wrap_mode: WrapMode) {
        self.wrap_mode = wrap_mode;
    }

    pub fn pixel(&self, x: usize, y: usize) -> [u8; 4] {
        self.pixels[y * self.width + x]
    }

    pub fn sample_nearest(&self, uv: Vec2) -> [u8; 4] {
        self.sample_nearest_with_wrap(uv, self.wrap_mode)
    }

    pub fn sample_nearest_with_wrap(&self, uv: Vec2, wrap_mode: WrapMode) -> [u8; 4] {
        let u = wrap_coordinate(uv.x, wrap_mode);
        let v = wrap_coordinate(uv.y, wrap_mode);
        let x = nearest_index(u, self.width);
        let y = nearest_index(v, self.height);
        self.pixel(x, y)
    }

    pub fn sample_bilinear(&self, uv: Vec2) -> [u8; 4] {
        self.sample_bilinear_with_wrap(uv, self.wrap_mode)
    }

    pub fn sample_bilinear_with_wrap(&self, uv: Vec2, wrap_mode: WrapMode) -> [u8; 4] {
        let u = wrap_coordinate(uv.x, wrap_mode);
        let v = wrap_coordinate(uv.y, wrap_mode);
        let x = u * self.width as f32 - 0.5;
        let y = v * self.height as f32 - 0.5;
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let tx = x - x.floor();
        let ty = y - y.floor();
        let p00 = self.pixel_wrapped(x0, y0, wrap_mode);
        let p10 = self.pixel_wrapped(x0 + 1, y0, wrap_mode);
        let p01 = self.pixel_wrapped(x0, y0 + 1, wrap_mode);
        let p11 = self.pixel_wrapped(x0 + 1, y0 + 1, wrap_mode);
        let mut result = [0; 4];
        for channel in 0..4 {
            let top = p00[channel] as f32 * (1.0 - tx) + p10[channel] as f32 * tx;
            let bottom = p01[channel] as f32 * (1.0 - tx) + p11[channel] as f32 * tx;
            result[channel] = (top * (1.0 - ty) + bottom * ty).round() as u8;
        }
        result
    }

    fn pixel_wrapped(&self, x: isize, y: isize, wrap_mode: WrapMode) -> [u8; 4] {
        let x = wrap_index(x, self.width, wrap_mode);
        let y = wrap_index(y, self.height, wrap_mode);
        self.pixel(x, y)
    }
}

fn wrap_coordinate(value: f32, wrap_mode: WrapMode) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    match wrap_mode {
        WrapMode::Repeat => value.rem_euclid(1.0),
        WrapMode::ClampToEdge => value.clamp(0.0, 1.0),
    }
}

fn nearest_index(coordinate: f32, size: usize) -> usize {
    ((coordinate * size as f32).floor() as usize).min(size - 1)
}

fn wrap_index(index: isize, size: usize, wrap_mode: WrapMode) -> usize {
    match wrap_mode {
        WrapMode::Repeat => index.rem_euclid(size as isize) as usize,
        WrapMode::ClampToEdge => index.clamp(0, size as isize - 1) as usize,
    }
}

pub fn decode_ppm(bytes: &[u8]) -> Result<Texture, ImageError> {
    let mut cursor = 0;
    let magic = next_ppm_token(bytes, &mut cursor)?;
    if magic != b"P6" {
        return Err(ImageError::new(0, "PPM magic must be P6"));
    }
    let width = parse_ppm_dimension(next_ppm_token(bytes, &mut cursor)?, cursor)?;
    let height = parse_ppm_dimension(next_ppm_token(bytes, &mut cursor)?, cursor)?;
    let max_value = parse_ppm_dimension(next_ppm_token(bytes, &mut cursor)?, cursor)?;
    if max_value != 255 {
        return Err(ImageError::new(cursor, "PPM max value must be 255"));
    }
    let separator = bytes
        .get(cursor)
        .copied()
        .ok_or_else(|| ImageError::new(cursor, "missing PPM raster separator"))?;
    if !separator.is_ascii_whitespace() {
        return Err(ImageError::new(cursor, "missing PPM raster separator"));
    }
    cursor += 1;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| ImageError::new(cursor, "PPM dimensions overflow the pixel count"))?;
    let payload_size = pixel_count
        .checked_mul(3)
        .ok_or_else(|| ImageError::new(cursor, "PPM payload size overflows usize"))?;
    let end = cursor
        .checked_add(payload_size)
        .ok_or_else(|| ImageError::new(cursor, "PPM payload offset overflows usize"))?;
    if end > bytes.len() {
        return Err(ImageError::new(
            bytes.len(),
            format!("truncated PPM payload: need {payload_size} bytes"),
        ));
    }
    if end != bytes.len() {
        return Err(ImageError::new(end, "trailing bytes after PPM payload"));
    }
    let pixels = bytes[cursor..end]
        .chunks_exact(3)
        .map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
        .collect();
    Texture::new(width, height, pixels)
}

fn next_ppm_token(bytes: &[u8], cursor: &mut usize) -> Result<Vec<u8>, ImageError> {
    skip_ppm_header_space(bytes, cursor);
    let start = *cursor;
    while let Some(&byte) = bytes.get(*cursor) {
        if byte.is_ascii_whitespace() || byte == b'#' {
            break;
        }
        *cursor += 1;
    }
    if start == *cursor {
        return Err(ImageError::new(*cursor, "missing PPM header token"));
    }
    Ok(bytes[start..*cursor].to_vec())
}

fn skip_ppm_header_space(bytes: &[u8], cursor: &mut usize) {
    loop {
        while bytes
            .get(*cursor)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            *cursor += 1;
        }
        if bytes.get(*cursor) != Some(&b'#') {
            break;
        }
        while bytes.get(*cursor).is_some_and(|byte| *byte != b'\n') {
            *cursor += 1;
        }
    }
}

fn parse_ppm_dimension(token: Vec<u8>, offset: usize) -> Result<usize, ImageError> {
    let text = String::from_utf8_lossy(&token);
    text.parse::<usize>()
        .map_err(|_| ImageError::new(offset.saturating_sub(token.len()), "invalid PPM number"))
}

pub fn decode_qoi(bytes: &[u8]) -> Result<Texture, ImageError> {
    if bytes.len() < QOI_HEADER_SIZE + QOI_END_MARKER.len() {
        return Err(ImageError::new(0, "truncated QOI header or end marker"));
    }
    if &bytes[0..4] != b"qoif" {
        return Err(ImageError::new(0, "QOI magic must be qoif"));
    }
    let width = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let height = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if width == 0 || height == 0 {
        return Err(ImageError::new(4, "QOI dimensions must be non-zero"));
    }
    let channels = bytes[12];
    if channels != 3 && channels != 4 {
        return Err(ImageError::new(12, "QOI channels must be 3 or 4"));
    }
    if bytes[13] > 1 {
        return Err(ImageError::new(13, "QOI colorspace must be 0 or 1"));
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| ImageError::new(4, "QOI dimensions overflow the pixel count"))?;
    let maximum_encoded_pixels = bytes
        .len()
        .saturating_sub(QOI_HEADER_SIZE + QOI_END_MARKER.len())
        .saturating_mul(62);
    if pixel_count > maximum_encoded_pixels {
        return Err(ImageError::new(
            4,
            "QOI dimensions exceed the encoded stream",
        ));
    }
    let mut pixels = Vec::with_capacity(pixel_count);
    let mut index = [[0u8; 4]; 64];
    let mut previous = [0, 0, 0, 255];
    let mut cursor = QOI_HEADER_SIZE;
    while pixels.len() < pixel_count {
        let opcode_offset = cursor;
        let opcode = *bytes
            .get(cursor)
            .ok_or_else(|| ImageError::new(cursor, "truncated QOI opcode"))?;
        cursor += 1;
        match opcode {
            0xfe => {
                let rgb = bytes.get(cursor..cursor + 3).ok_or_else(|| {
                    ImageError::new(opcode_offset, "truncated QOI_OP_RGB payload")
                })?;
                previous[0] = rgb[0];
                previous[1] = rgb[1];
                previous[2] = rgb[2];
                cursor += 3;
                index[qoi_hash(previous)] = previous;
                pixels.push(previous);
            }
            0xff => {
                let rgba = bytes.get(cursor..cursor + 4).ok_or_else(|| {
                    ImageError::new(opcode_offset, "truncated QOI_OP_RGBA payload")
                })?;
                previous = [rgba[0], rgba[1], rgba[2], rgba[3]];
                cursor += 4;
                index[qoi_hash(previous)] = previous;
                pixels.push(previous);
            }
            opcode if opcode & 0xc0 == 0x00 => {
                previous = index[(opcode & 0x3f) as usize];
                pixels.push(previous);
            }
            opcode if opcode & 0xc0 == 0x40 => {
                previous[0] = previous[0].wrapping_add(((opcode >> 4) & 0x03).wrapping_sub(2));
                previous[1] = previous[1].wrapping_add(((opcode >> 2) & 0x03).wrapping_sub(2));
                previous[2] = previous[2].wrapping_add((opcode & 0x03).wrapping_sub(2));
                index[qoi_hash(previous)] = previous;
                pixels.push(previous);
            }
            opcode if opcode & 0xc0 == 0x80 => {
                let second = *bytes.get(cursor).ok_or_else(|| {
                    ImageError::new(opcode_offset, "truncated QOI_OP_LUMA payload")
                })?;
                cursor += 1;
                let green_delta = (opcode & 0x3f).wrapping_sub(32);
                let red_delta = ((second >> 4) & 0x0f)
                    .wrapping_sub(8)
                    .wrapping_add(green_delta);
                let blue_delta = (second & 0x0f).wrapping_sub(8).wrapping_add(green_delta);
                previous[0] = previous[0].wrapping_add(red_delta);
                previous[1] = previous[1].wrapping_add(green_delta);
                previous[2] = previous[2].wrapping_add(blue_delta);
                index[qoi_hash(previous)] = previous;
                pixels.push(previous);
            }
            opcode => {
                let run_length = (opcode & 0x3f) as usize + 1;
                let remaining = pixel_count - pixels.len();
                if run_length > remaining {
                    return Err(ImageError::new(
                        opcode_offset,
                        "QOI_OP_RUN writes past the end of the image",
                    ));
                }
                index[qoi_hash(previous)] = previous;
                pixels.extend(std::iter::repeat_n(previous, run_length));
            }
        }
    }
    let marker_end = cursor
        .checked_add(QOI_END_MARKER.len())
        .ok_or_else(|| ImageError::new(cursor, "QOI end marker offset overflows usize"))?;
    if marker_end > bytes.len() {
        return Err(ImageError::new(cursor, "truncated QOI end marker"));
    }
    if bytes[cursor..marker_end] != QOI_END_MARKER {
        return Err(ImageError::new(cursor, "invalid QOI end marker"));
    }
    if marker_end != bytes.len() {
        return Err(ImageError::new(
            marker_end,
            "trailing bytes after QOI end marker",
        ));
    }
    Texture::new(width, height, pixels)
}

pub fn encode_qoi(texture: &Texture) -> Result<Vec<u8>, ImageError> {
    let width = u32::try_from(texture.width)
        .map_err(|_| ImageError::new(4, "texture width does not fit in QOI"))?;
    let height = u32::try_from(texture.height)
        .map_err(|_| ImageError::new(8, "texture height does not fit in QOI"))?;
    let channels = if texture.pixels.iter().all(|pixel| pixel[3] == 255) {
        3
    } else {
        4
    };
    let mut output = Vec::with_capacity(QOI_HEADER_SIZE + texture.pixels.len() * 5 + 8);
    output.extend_from_slice(b"qoif");
    output.extend_from_slice(&width.to_be_bytes());
    output.extend_from_slice(&height.to_be_bytes());
    output.push(channels);
    output.push(0);

    let mut index = [[0u8; 4]; 64];
    let mut previous = [0, 0, 0, 255];
    let mut run = 0usize;
    for &pixel in &texture.pixels {
        if pixel == previous {
            run += 1;
            if run == 62 {
                output.push(0xc0 | (run as u8 - 1));
                index[qoi_hash(previous)] = previous;
                run = 0;
            }
            continue;
        }
        if run != 0 {
            output.push(0xc0 | (run as u8 - 1));
            index[qoi_hash(previous)] = previous;
            run = 0;
        }
        let hash = qoi_hash(pixel);
        if index[hash] == pixel {
            output.push(hash as u8);
            previous = pixel;
            continue;
        }
        if pixel[3] == previous[3] {
            let dr = pixel[0].wrapping_sub(previous[0]) as i8;
            let dg = pixel[1].wrapping_sub(previous[1]) as i8;
            let db = pixel[2].wrapping_sub(previous[2]) as i8;
            if (-2..=1).contains(&dr) && (-2..=1).contains(&dg) && (-2..=1).contains(&db) {
                output
                    .push(0x40 | (((dr + 2) as u8) << 4) | ((dg + 2) as u8) << 2 | (db + 2) as u8);
            } else {
                let dr_dg = i16::from(dr) - i16::from(dg);
                let db_dg = i16::from(db) - i16::from(dg);
                if (-32..=31).contains(&dg)
                    && (-8..=7).contains(&dr_dg)
                    && (-8..=7).contains(&db_dg)
                {
                    output.push(0x80 | (dg + 32) as u8);
                    output.push(((dr_dg + 8) as u8) << 4 | (db_dg + 8) as u8);
                } else {
                    output.push(0xfe);
                    output.extend_from_slice(&pixel[..3]);
                }
            }
        } else {
            output.push(0xff);
            output.extend_from_slice(&pixel);
        }
        index[hash] = pixel;
        previous = pixel;
    }
    if run != 0 {
        output.push(0xc0 | (run as u8 - 1));
    }
    output.extend_from_slice(&QOI_END_MARKER);
    Ok(output)
}

fn qoi_hash(pixel: [u8; 4]) -> usize {
    (pixel[0] as usize * 3 + pixel[1] as usize * 5 + pixel[2] as usize * 7 + pixel[3] as usize * 11)
        % 64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qoi_header(width: u32, height: u32, channels: u8) -> Vec<u8> {
        let mut bytes = b"qoif".to_vec();
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[channels, 0]);
        bytes
    }

    fn with_marker(mut bytes: Vec<u8>) -> Vec<u8> {
        bytes.extend_from_slice(&QOI_END_MARKER);
        bytes
    }

    #[test]
    fn decodes_every_qoi_opcode_with_hand_written_bytes() {
        // RGB: fe 01 02 03 -> [1, 2, 3, 255].
        // INDEX: 00 -> index slot 0, which is the initial [0, 0, 0, 0].
        // DIFF: 7f -> dr=1, dg=1, db=1, from [0, 0, 0, 0].
        // LUMA: a1 98 -> dg=1, dr=2, db=1, from [1, 1, 1, 0].
        // RGBA: ff 09 08 07 06 -> [9, 8, 7, 6].
        // RUN: c1 -> repeat the current pixel twice.
        let mut bytes = qoi_header(7, 1, 4);
        bytes.extend_from_slice(&[
            0xfe, 1, 2, 3, 0x00, 0x7f, 0xa1, 0x98, 0xff, 9, 8, 7, 6, 0xc1,
        ]);
        let texture = decode_qoi(&with_marker(bytes)).unwrap();
        assert_eq!(
            texture.pixels,
            vec![
                [1, 2, 3, 255],
                [0, 0, 0, 0],
                [1, 1, 1, 0],
                [3, 2, 2, 0],
                [9, 8, 7, 6],
                [9, 8, 7, 6],
                [9, 8, 7, 6],
            ]
        );
    }

    #[test]
    fn qoi_round_trip_preserves_rgba_pixels() {
        let texture = Texture::new(
            4,
            2,
            vec![
                [1, 2, 3, 255],
                [1, 2, 3, 255],
                [4, 8, 12, 250],
                [4, 8, 12, 250],
                [30, 20, 10, 1],
                [31, 21, 11, 2],
                [32, 22, 12, 3],
                [33, 23, 13, 4],
            ],
        )
        .unwrap();
        let encoded = encode_qoi(&texture).unwrap();
        assert_eq!(decode_qoi(&encoded).unwrap().pixels, texture.pixels);
    }

    #[test]
    fn qoi_round_trip_handles_wrapped_channel_deltas() {
        let texture = Texture::new(
            7,
            1,
            vec![
                // Successive pairs cover all four (dr - dg, db - dg) signs.
                [10, 10, 10, 255],
                [10, 11, 10, 255],
                [10, 12, 12, 255],
                [12, 13, 12, 255],
                [14, 14, 14, 255],
                [255, 0, 255, 255],
                [0, 128, 0, 255],
            ],
        )
        .unwrap();
        let encoded = encode_qoi(&texture).unwrap();
        assert_eq!(decode_qoi(&encoded).unwrap().pixels, texture.pixels);
    }

    #[test]
    fn qoi_run_updates_the_index_for_a_following_index_opcode() {
        let mut bytes = qoi_header(2, 1, 4);
        bytes.extend_from_slice(&[0xc0, 0x35]);
        let texture = decode_qoi(&with_marker(bytes)).unwrap();
        assert_eq!(texture.pixels, vec![[0, 0, 0, 255], [0, 0, 0, 255]]);
    }

    #[test]
    fn malformed_qoi_returns_errors_without_panicking() {
        let mut header = qoi_header(1, 1, 3);
        header.push(0xfe);
        assert!(decode_qoi(&header).is_err());
        let mut run = qoi_header(1, 1, 3);
        run.push(0xff);
        run.extend_from_slice(&[1, 2, 3, 4]);
        run.push(0xc1);
        assert!(decode_qoi(&with_marker(run)).is_err());
        let mut marker = qoi_header(1, 1, 3);
        marker.extend_from_slice(&[0xfe, 1, 2, 3]);
        marker.extend_from_slice(&[0; 8]);
        assert!(decode_qoi(&marker).is_err());
        let mut bad_magic = qoi_header(1, 1, 3);
        bad_magic[0] = b'x';
        assert!(decode_qoi(&bad_magic).is_err());
        let mut bad_channels = qoi_header(1, 1, 2);
        bad_channels.extend_from_slice(&[0xfe, 1, 2, 3]);
        assert!(decode_qoi(&with_marker(bad_channels)).is_err());
    }

    #[test]
    fn malformed_ppm_returns_errors_with_truncation_and_header_context() {
        assert!(decode_ppm(b"P3\n1 1\n255\n\0\0\0").is_err());
        assert!(decode_ppm(b"P6\n2 1\n255\n\0\0").is_err());
        assert!(decode_ppm(b"P6\n1 1\n16\n\0\0\0").is_err());
    }

    #[test]
    fn bilinear_sampling_uses_texel_centers_and_weighted_average() {
        let texture = Texture::new(
            2,
            2,
            vec![
                [0, 0, 0, 255],
                [100, 0, 0, 255],
                [0, 100, 0, 255],
                [100, 100, 0, 255],
            ],
        )
        .unwrap()
        .with_wrap_mode(WrapMode::ClampToEdge);
        // u=.6, v=.4 maps to x=.7, y=.3. The hand-computed weighted result
        // is red=70 and green=30.
        assert_eq!(
            texture.sample_bilinear(Vec2::new(0.6, 0.4)),
            [70, 30, 0, 255]
        );
    }

    #[test]
    fn wrap_modes_handle_edges_and_out_of_range_coordinates() {
        let texture = Texture::new(2, 1, vec![[10, 0, 0, 255], [20, 0, 0, 255]]).unwrap();
        assert_eq!(
            texture.sample_nearest_with_wrap(Vec2::new(0.0, 0.0), WrapMode::Repeat),
            [10, 0, 0, 255]
        );
        assert_eq!(
            texture.sample_nearest_with_wrap(Vec2::new(1.0, 0.0), WrapMode::Repeat),
            [10, 0, 0, 255]
        );
        assert_eq!(
            texture.sample_nearest_with_wrap(Vec2::new(-0.1, 0.0), WrapMode::Repeat),
            [20, 0, 0, 255]
        );
        assert_eq!(
            texture.sample_nearest_with_wrap(Vec2::new(1.1, 0.0), WrapMode::Repeat),
            [10, 0, 0, 255]
        );
        assert_eq!(
            texture.sample_nearest_with_wrap(Vec2::new(1.0, 0.0), WrapMode::ClampToEdge),
            [20, 0, 0, 255]
        );
        assert_eq!(
            texture.sample_nearest_with_wrap(Vec2::new(-0.1, 0.0), WrapMode::ClampToEdge),
            [10, 0, 0, 255]
        );
    }

    #[test]
    fn ppm_p6_decodes_comments_and_rgb_payload() {
        let bytes = b"P6\n# test\n2 1\n255\n\x01\x02\x03\x04\x05\x06";
        assert_eq!(
            decode_ppm(bytes).unwrap().pixels,
            vec![[1, 2, 3, 255], [4, 5, 6, 255]]
        );
    }

    #[test]
    fn ppm_p6_keeps_newline_and_hash_bytes_in_the_raster() {
        for first_byte in *b"\n#" {
            let mut bytes = b"P6\n1 1\n255\n".to_vec();
            bytes.extend_from_slice(&[first_byte, 2, 3]);
            assert_eq!(
                decode_ppm(&bytes).unwrap().pixels,
                vec![[first_byte, 2, 3, 255]]
            );
        }
    }
}
