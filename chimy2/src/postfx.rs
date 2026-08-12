//! Deterministic framebuffer post-processing.
//!
//! The renderer stores encoded sRGB colors in `Framebuffer`. HDR mode also
//! stores a linear sidecar. Each pass declares its working color space.
//! The HDR order is render -> SSAA -> SSAO -> DoF -> bloom -> vignette -> ACES ->
//! encode -> FXAA. LDR chains retain their existing pass order and byte output.
//! SSAO and DoF must run before tonemapping and bloom because they read linear
//! scene color and depth. SSAO modulates the full color, not only the ambient term.
//! Ambient-only modulation needs a separate ambient buffer and is out of scope.

use crate::fb::{Framebuffer, argb8888_linear};
use crate::image::{srgb_to_linear, srgb_to_linear_u8};
use crate::math::{Mat4, Vec3, Vec4};

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
    hdr: bool,
}

impl PostBuffer {
    pub fn new(width: usize, height: usize, color_space: PostColorSpace) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0.0; 4]; width.saturating_mul(height)],
            color_space,
            hdr: false,
        }
    }

    fn from_framebuffer(framebuffer: &Framebuffer) -> Self {
        let mut buffer = Self::new(
            framebuffer.width,
            framebuffer.height,
            PostColorSpace::Linear,
        );
        if let Some(linear) = framebuffer.linear_pixels() {
            buffer.pixels.copy_from_slice(linear);
            buffer.hdr = true;
        } else {
            for (destination, &pixel) in buffer.pixels.iter_mut().zip(&framebuffer.color) {
                let [alpha, red, green, blue] = pixel.to_be_bytes();
                *destination = [
                    f32::from(alpha) / 255.0,
                    srgb_to_linear_u8(red),
                    srgb_to_linear_u8(green),
                    srgb_to_linear_u8(blue),
                ];
            }
        }
        buffer
    }

    fn convert_to(&self, color_space: PostColorSpace) -> Self {
        if self.color_space == color_space {
            return self.clone();
        }
        let mut converted = Self::new(self.width, self.height, color_space);
        converted.hdr = self.hdr;
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
        for (index, &source) in self.pixels.iter().enumerate() {
            framebuffer.color[index] = match self.color_space {
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
            if let Some(linear) = framebuffer.linear_pixels_mut() {
                linear[index] = match self.color_space {
                    PostColorSpace::Linear => source,
                    PostColorSpace::EncodedSrgb => [
                        source[0],
                        srgb_to_linear(source[1]),
                        srgb_to_linear(source[2]),
                        srgb_to_linear(source[3]),
                    ],
                };
            }
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

    /// Returns true when this pass has no effect and must be skipped before
    /// color-space conversion, preserving exact framebuffer bytes.
    fn is_noop(&self) -> bool {
        false
    }

    /// Applies a pass with access to the rendered framebuffer.
    ///
    /// Most passes only need the color buffer and use [`Self::apply`]. A pass
    /// such as SSAO can override this hook to read depth without widening the
    /// raster or shader interfaces.
    fn apply_with_framebuffer(
        &self,
        input: &PostBuffer,
        output: &mut PostBuffer,
        _framebuffer: &Framebuffer,
    ) {
        self.apply(input, output);
    }
}

/// An ordered, opt-in collection of post-processing passes.
///
/// Push [`SsaoPass`] and [`DofPass`] before [`BloomPass`] and
/// [`AcesTonemapPass`]. These depth passes need the linear scene before later
/// passes change the color space or add display-space light.
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
            if pass.is_noop() {
                continue;
            }
            let input = current.convert_to(pass.color_space());
            let mut output = PostBuffer::new(input.width, input.height, pass.color_space());
            output.hdr = input.hdr;
            pass.apply_with_framebuffer(&input, &mut output, framebuffer);
            current = output;
        }
        current.write_to_framebuffer(framebuffer);
    }
}

/// Number of deterministic hemisphere samples used by [`SsaoPass`].
pub const SSAO_SAMPLE_COUNT: usize = 16;
pub const SSAO_DEFAULT_RADIUS: f32 = 0.55;
pub const SSAO_DEFAULT_BIAS: f32 = 0.025;
pub const SSAO_DEFAULT_STRENGTH: f32 = 1.0;
pub const SSAO_DEFAULT_RANGE: f32 = 0.9;
pub const SSAO_DEFAULT_BLUR_DEPTH_THRESHOLD: f32 = 0.35;

const SSAO_BLUR_TAPS: [(isize, f32); 3] = [(0, 0.5), (-1, 0.25), (1, 0.25)];

/// Screen-space ambient occlusion for a perspective-rendered depth buffer.
///
/// The pass reconstructs view-space positions by applying the inverse of the
/// exact projection matrix used by the renderer. It estimates a normal from
/// the lower-delta screen-space position neighbors, samples a fixed
/// deterministic hemisphere, range-checks depth comparisons, and applies an
/// edge-aware separable blur to occlusion only.
///
/// SSAO runs in linear color space and multiplies the complete color. It does
/// not isolate ambient lighting. That needs a separate ambient buffer and is
/// out of scope for this pass.
#[derive(Clone, Debug, PartialEq)]
pub struct SsaoPass {
    projection: Mat4,
    inverse_projection: Mat4,
    projection_valid: bool,
    radius: f32,
    bias: f32,
    strength: f32,
    range: f32,
    blur_depth_threshold: f32,
}

/// Reconstructs one view-space position from one raster depth sample.
///
/// This is the shared perspective-depth reconstruction used by both SSAO and
/// depth of field. Keeping one implementation also keeps their linear depth
/// interpretation identical.
pub fn reconstruct_view_position(
    inverse_projection: Mat4,
    x: usize,
    y: usize,
    depth: f32,
    width: usize,
    height: usize,
) -> Option<Vec3> {
    if width == 0 || height == 0 || x >= width || y >= height || !depth.is_finite() {
        return None;
    }
    let ndc_x = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((y as f32 + 0.5) / height as f32) * 2.0;
    let clip = inverse_projection * Vec4::new(ndc_x, ndc_y, depth, 1.0);
    if !clip.w.is_finite() || clip.w.abs() <= f32::EPSILON {
        return None;
    }
    let position = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
    if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() {
        return None;
    }
    Some(position)
}

impl SsaoPass {
    /// Creates a pass from the projection matrix used by the camera.
    ///
    /// The matrix is sanitized at this boundary. Non-finite matrices and
    /// singular matrices fall back to identity, which makes the pass a safe
    /// no-op for an invalid projection rather than producing invalid pixels.
    pub fn new(projection: Mat4) -> Self {
        let (projection, inverse_projection, projection_valid) = sanitize_projection(projection);
        Self {
            projection,
            inverse_projection,
            projection_valid,
            radius: SSAO_DEFAULT_RADIUS,
            bias: SSAO_DEFAULT_BIAS,
            strength: SSAO_DEFAULT_STRENGTH,
            range: SSAO_DEFAULT_RANGE,
            blur_depth_threshold: SSAO_DEFAULT_BLUR_DEPTH_THRESHOLD,
        }
    }

    pub const fn projection(&self) -> Mat4 {
        self.projection
    }

    pub fn set_projection(&mut self, projection: Mat4) {
        let (projection, inverse_projection, projection_valid) = sanitize_projection(projection);
        self.projection = projection;
        self.inverse_projection = inverse_projection;
        self.projection_valid = projection_valid;
    }

    pub const fn radius(&self) -> f32 {
        self.radius
    }

    pub fn set_radius(&mut self, radius: f32) {
        self.radius = sanitize_positive(radius, SSAO_DEFAULT_RADIUS, 1000.0);
    }

    pub const fn bias(&self) -> f32 {
        self.bias
    }

    pub fn set_bias(&mut self, bias: f32) {
        self.bias = sanitize_nonnegative(bias, SSAO_DEFAULT_BIAS, 1000.0);
    }

    pub const fn strength(&self) -> f32 {
        self.strength
    }

    pub fn set_strength(&mut self, strength: f32) {
        self.strength = sanitize_unit(strength, SSAO_DEFAULT_STRENGTH);
    }

    pub const fn range(&self) -> f32 {
        self.range
    }

    pub fn set_range(&mut self, range: f32) {
        self.range = sanitize_positive(range, SSAO_DEFAULT_RANGE, 1000.0);
    }

    pub const fn blur_depth_threshold(&self) -> f32 {
        self.blur_depth_threshold
    }

    pub fn set_blur_depth_threshold(&mut self, threshold: f32) {
        self.blur_depth_threshold =
            sanitize_positive(threshold, SSAO_DEFAULT_BLUR_DEPTH_THRESHOLD, 1000.0);
    }

    /// Reconstructs one pixel's view-space position from framebuffer depth.
    ///
    /// Depth is the post-divide NDC z written by the rasterizer. The inverse
    /// projection performs the required non-linear perspective linearization.
    pub fn reconstruct_view_position(
        &self,
        x: usize,
        y: usize,
        depth: f32,
        width: usize,
        height: usize,
    ) -> Option<Vec3> {
        reconstruct_view_position(self.inverse_projection, x, y, depth, width, height)
    }

    /// Computes the unblurred occlusion buffer from the framebuffer depth.
    pub fn raw_occlusion_buffer(&self, framebuffer: &Framebuffer) -> Vec<f32> {
        let length = framebuffer.width.saturating_mul(framebuffer.height);
        if !self.projection_valid {
            return vec![0.0; length];
        }
        let positions = self.reconstructed_positions(framebuffer);
        let mut raw = vec![0.0; length];
        for y in 0..framebuffer.height {
            for x in 0..framebuffer.width {
                let index = y * framebuffer.width + x;
                let Some(center) = positions[index] else {
                    continue;
                };
                let normal = self.estimate_normal(&positions, x, y, center, framebuffer.width);
                raw[index] = self.sample_occlusion(framebuffer, &positions, center, normal);
            }
        }
        raw
    }

    /// Computes the blurred occlusion buffer from the framebuffer depth.
    ///
    /// This is the same production path used by [`Self::apply_with_framebuffer`].
    /// A caller can use it to inspect or compare deterministic occlusion data.
    pub fn occlusion_buffer(&self, framebuffer: &Framebuffer) -> Vec<f32> {
        let raw = self.raw_occlusion_buffer(framebuffer);
        if !self.projection_valid {
            return raw;
        }
        let positions = self.reconstructed_positions(framebuffer);
        self.blur_occlusion(&raw, &positions, framebuffer.width, framebuffer.height)
    }

    /// Applies SSAO directly to a framebuffer through the production path.
    pub fn apply_to_framebuffer(&self, framebuffer: &mut Framebuffer) {
        if !self.projection_valid {
            return;
        }
        let input = PostBuffer::from_framebuffer(framebuffer);
        let mut output = PostBuffer::new(input.width, input.height, PostColorSpace::Linear);
        output.hdr = input.hdr;
        self.apply_with_framebuffer(&input, &mut output, framebuffer);
        output.write_to_framebuffer(framebuffer);
    }

    fn reconstructed_positions(&self, framebuffer: &Framebuffer) -> Vec<Option<Vec3>> {
        let length = framebuffer.width.saturating_mul(framebuffer.height);
        let mut positions = vec![None; length];
        for y in 0..framebuffer.height {
            for x in 0..framebuffer.width {
                let index = y * framebuffer.width + x;
                let Some(&depth) = framebuffer.depth.get(index) else {
                    continue;
                };
                if depth >= 1.0 {
                    continue;
                }
                positions[index] = reconstruct_view_position(
                    self.inverse_projection,
                    x,
                    y,
                    depth,
                    framebuffer.width,
                    framebuffer.height,
                );
            }
        }
        positions
    }

    fn estimate_normal(
        &self,
        positions: &[Option<Vec3>],
        x: usize,
        y: usize,
        center: Vec3,
        width: usize,
    ) -> Vec3 {
        let neighbor = |nx: usize, ny: usize| positions[ny * width + nx];
        let left = (x > 0).then(|| neighbor(x - 1, y)).flatten();
        let right = (x + 1 < width).then(|| neighbor(x + 1, y)).flatten();
        let height = positions.len().checked_div(width).unwrap_or(0);
        let up = (y > 0).then(|| neighbor(x, y - 1)).flatten();
        let down = (y + 1 < height).then(|| neighbor(x, y + 1)).flatten();
        let dx = choose_delta(center, left, right);
        let dy = choose_delta(center, up, down);
        let mut normal = dx.cross(dy).normalize();
        if normal.length() == 0.0 || !normal.x.is_finite() {
            return Vec3::new(0.0, 0.0, 1.0);
        }
        if normal.dot(-center) < 0.0 {
            normal = -normal;
        }
        normal
    }

    fn sample_occlusion(
        &self,
        framebuffer: &Framebuffer,
        positions: &[Option<Vec3>],
        center: Vec3,
        normal: Vec3,
    ) -> f32 {
        let tangent_up = if normal.z.abs() < 0.9 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let tangent = tangent_up.cross(normal).normalize();
        let bitangent = normal.cross(tangent).normalize();
        let mut occluded = 0.0;
        let mut considered = 0.0;
        for sample_index in 0..SSAO_SAMPLE_COUNT {
            let fraction = (sample_index as f32 + 0.5) / SSAO_SAMPLE_COUNT as f32;
            let angle = (sample_index as f32 * 0.618_033_95).fract() * std::f32::consts::TAU;
            let radial = fraction.sqrt();
            let local = Vec3::new(
                angle.cos() * radial,
                angle.sin() * radial,
                (1.0 - fraction).sqrt(),
            );
            let direction = tangent * local.x + bitangent * local.y + normal * local.z;
            let sample_position = center + direction * self.radius;
            let Some((sample_x, sample_y)) =
                self.project_to_pixel(sample_position, framebuffer.width, framebuffer.height)
            else {
                continue;
            };
            let sample_index = sample_y * framebuffer.width + sample_x;
            let Some(sample_surface) = positions.get(sample_index).copied().flatten() else {
                continue;
            };
            let depth_delta = (center.z - sample_surface.z).abs();
            if depth_delta > self.range {
                continue;
            }
            considered += 1.0;
            if sample_surface.z > sample_position.z + self.bias {
                occluded += 1.0;
            }
        }
        if considered == 0.0 {
            0.0
        } else {
            occluded / considered
        }
    }

    fn project_to_pixel(
        &self,
        position: Vec3,
        width: usize,
        height: usize,
    ) -> Option<(usize, usize)> {
        if width == 0 || height == 0 {
            return None;
        }
        let clip = self.projection * Vec4::new(position.x, position.y, position.z, 1.0);
        if !clip.w.is_finite() || clip.w <= f32::EPSILON {
            return None;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        if !ndc_x.is_finite() || !ndc_y.is_finite() {
            return None;
        }
        let pixel_x = ((ndc_x + 1.0) * 0.5 * width as f32).floor() as isize;
        let pixel_y = ((1.0 - ndc_y) * 0.5 * height as f32).floor() as isize;
        if pixel_x < 0 || pixel_y < 0 || pixel_x >= width as isize || pixel_y >= height as isize {
            None
        } else {
            Some((pixel_x as usize, pixel_y as usize))
        }
    }

    fn blur_occlusion(
        &self,
        raw: &[f32],
        positions: &[Option<Vec3>],
        width: usize,
        height: usize,
    ) -> Vec<f32> {
        let mut horizontal = vec![0.0; raw.len()];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                horizontal[index] = self.blur_pixel(raw, positions, x, y, width, true);
            }
        }
        let mut blurred = vec![0.0; raw.len()];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                blurred[index] = self.blur_pixel(&horizontal, positions, x, y, width, false);
            }
        }
        blurred
    }

    fn blur_pixel(
        &self,
        values: &[f32],
        positions: &[Option<Vec3>],
        x: usize,
        y: usize,
        width: usize,
        horizontal: bool,
    ) -> f32 {
        let center_index = y * width + x;
        let height = positions.len().checked_div(width).unwrap_or(0);
        let Some(center_position) = positions[center_index] else {
            return 0.0;
        };
        // The pre-guard protects the cross-axis edge. Axis taps are checked
        // below so each separable blur direction keeps its own load-bearing
        // depth threshold.
        let perpendicular_neighbors = if horizontal {
            [
                (Some(x), y.checked_sub(1)),
                (Some(x), y.checked_add(1).filter(|&value| value < height)),
            ]
        } else {
            [
                (x.checked_sub(1), Some(y)),
                (x.checked_add(1).filter(|&value| value < width), Some(y)),
            ]
        };
        for (nx, ny) in perpendicular_neighbors {
            let (Some(nx), Some(ny)) = (nx, ny) else {
                continue;
            };
            let neighbor_index = ny * width + nx;
            let Some(neighbor_position) = positions[neighbor_index] else {
                continue;
            };
            if (neighbor_position.z - center_position.z).abs() > self.blur_depth_threshold {
                return values[center_index];
            }
        }
        let mut total = 0.0;
        let mut weight_total = 0.0;
        for (offset, weight) in SSAO_BLUR_TAPS {
            let coordinate = if horizontal {
                x as isize + offset
            } else {
                y as isize + offset
            };
            let limit = if horizontal { width } else { height };
            if coordinate < 0 || coordinate >= limit as isize {
                continue;
            }
            let (nx, ny) = if horizontal {
                (coordinate as usize, y)
            } else {
                (x, coordinate as usize)
            };
            let index = ny * width + nx;
            let Some(neighbor_position) = positions[index] else {
                continue;
            };
            if (neighbor_position.z - center_position.z).abs() > self.blur_depth_threshold {
                continue;
            }
            total += values[index] * weight;
            weight_total += weight;
        }
        if weight_total == 0.0 {
            0.0
        } else {
            total / weight_total
        }
    }
}

/// Fixed focal length used by the renderer's view-space thin-lens model.
/// Aperture is a texel-scaled strength, so the standard CoC equation produces
/// a pixel radius directly:
/// `aperture * |focal_length * (focus - depth)| /
/// (depth * (focus - focal_length))`.
pub const DOF_FOCAL_LENGTH: f32 = 1.0;
pub const DOF_DEFAULT_FOCUS_DISTANCE: f32 = 4.0;
pub const DOF_DEFAULT_APERTURE: f32 = 6.0;
pub const DOF_DEFAULT_MAX_COC_RADIUS: f32 = 8.0;
pub const DOF_MAX_COC_RADIUS: f32 = 64.0;
pub const DOF_SAMPLE_COUNT: usize = 24;
const DOF_FOCAL_COC_EPSILON: f32 = 1.0e-4;
const DOF_MAX_GATHER_SAMPLES: usize = DOF_SAMPLE_COUNT * 2 + 1;

struct DofGather<'a> {
    input: &'a PostBuffer,
    depths: &'a [Option<f32>],
    center_depth: f32,
    center_radius: f32,
    sampled_indices: [usize; DOF_MAX_GATHER_SAMPLES],
    sampled_count: usize,
    total: [f32; 3],
    weight_total: f32,
}

impl<'a> DofGather<'a> {
    fn new(
        input: &'a PostBuffer,
        depths: &'a [Option<f32>],
        index: usize,
        source: [f32; 4],
        center_depth: f32,
        center_radius: f32,
    ) -> Self {
        let mut sampled_indices = [0; DOF_MAX_GATHER_SAMPLES];
        sampled_indices[0] = index;
        Self {
            input,
            depths,
            center_depth,
            center_radius,
            sampled_indices,
            sampled_count: 1,
            total: [source[1], source[2], source[3]],
            weight_total: 1.0,
        }
    }
}

const DOF_DISC_KERNEL: [(f32, f32); DOF_SAMPLE_COUNT] = [
    (0.125, 0.0),
    (-0.125, 0.0),
    (0.0, 0.125),
    (0.0, -0.125),
    (0.35, 0.0),
    (-0.35, 0.0),
    (0.0, 0.35),
    (0.0, -0.35),
    (0.25, 0.25),
    (-0.25, 0.25),
    (0.25, -0.25),
    (-0.25, -0.25),
    (0.6, 0.0),
    (-0.6, 0.0),
    (0.0, 0.6),
    (0.0, -0.6),
    (0.42, 0.42),
    (-0.42, 0.42),
    (0.42, -0.42),
    (-0.42, -0.42),
    (0.9, 0.0),
    (-0.9, 0.0),
    (0.0, 0.9),
    (0.0, -0.9),
];

/// Thin-lens depth of field in linear color space.
///
/// This is a single-pass gather with scatter-as-gather coverage. A tap can
/// contribute when its own CoC covers the center, so defocused foreground
/// color can spread over sharp background pixels. Full near-field scatter and
/// its separate foreground layer are out of scope.
#[derive(Clone, Debug, PartialEq)]
pub struct DofPass {
    projection: Mat4,
    inverse_projection: Mat4,
    projection_valid: bool,
    focus_distance: f32,
    aperture: f32,
    max_coc_radius: f32,
}

impl DofPass {
    /// Creates a pass from the exact projection matrix used by the camera.
    pub fn new(projection: Mat4) -> Self {
        let (projection, inverse_projection, projection_valid) = sanitize_projection(projection);
        Self {
            projection,
            inverse_projection,
            projection_valid,
            focus_distance: DOF_DEFAULT_FOCUS_DISTANCE,
            aperture: DOF_DEFAULT_APERTURE,
            max_coc_radius: DOF_DEFAULT_MAX_COC_RADIUS,
        }
    }

    pub const fn projection(&self) -> Mat4 {
        self.projection
    }

    pub fn set_projection(&mut self, projection: Mat4) {
        let (projection, inverse_projection, projection_valid) = sanitize_projection(projection);
        self.projection = projection;
        self.inverse_projection = inverse_projection;
        self.projection_valid = projection_valid;
    }

    pub const fn focus_distance(&self) -> f32 {
        self.focus_distance
    }

    pub fn set_focus_distance(&mut self, distance: f32) {
        self.focus_distance = sanitize_positive(distance, DOF_DEFAULT_FOCUS_DISTANCE, 1000.0)
            .max(DOF_FOCAL_LENGTH + f32::EPSILON);
    }

    pub const fn aperture(&self) -> f32 {
        self.aperture
    }

    pub fn set_aperture(&mut self, aperture: f32) {
        self.aperture = sanitize_nonnegative(aperture, DOF_DEFAULT_APERTURE, 1000.0);
    }

    pub const fn max_coc_radius(&self) -> f32 {
        self.max_coc_radius
    }

    pub fn set_max_coc_radius(&mut self, radius: f32) {
        self.max_coc_radius =
            sanitize_nonnegative(radius, DOF_DEFAULT_MAX_COC_RADIUS, DOF_MAX_COC_RADIUS);
    }

    /// Returns the clamped CoC radius in texels for positive linear depth.
    pub fn circle_of_confusion(&self, depth: f32) -> f32 {
        if !depth.is_finite() || depth <= 0.0 || self.aperture == 0.0 {
            return 0.0;
        }
        let numerator = DOF_FOCAL_LENGTH * (self.focus_distance - depth).abs();
        let denominator = depth * (self.focus_distance - DOF_FOCAL_LENGTH);
        if denominator <= 0.0 || !denominator.is_finite() {
            return 0.0;
        }
        (self.aperture * numerator / denominator).clamp(0.0, self.max_coc_radius)
    }

    pub fn coc_radius(&self, depth: f32) -> f32 {
        self.circle_of_confusion(depth)
    }

    fn accumulate_sample(
        &self,
        gather: &mut DofGather<'_>,
        sample_x: usize,
        sample_y: usize,
        coverage: f32,
    ) {
        if coverage <= 0.0 || !coverage.is_finite() {
            return;
        }
        let sample_index = sample_y * gather.input.width + sample_x;
        if gather.sampled_indices[..gather.sampled_count].contains(&sample_index) {
            return;
        }
        let Some(sample_depth) = gather.depths[sample_index] else {
            return;
        };
        let sample_radius = self.circle_of_confusion(sample_depth);
        let depth_delta = sample_depth - gather.center_depth;
        // A background tap with a larger CoC cannot bleed onto a center
        // surface with less blur. Foreground taps remain eligible, which
        // is the useful gather approximation for foreground defocus.
        if depth_delta > 0.0 && sample_radius > gather.center_radius {
            return;
        }
        let depth_weight = if depth_delta > 0.0 {
            gather.center_radius / (gather.center_radius + depth_delta).max(f32::EPSILON)
        } else {
            1.0
        };
        let sample_weight = depth_weight * coverage;
        if sample_weight <= 0.0 || !sample_weight.is_finite() {
            return;
        }
        gather.sampled_indices[gather.sampled_count] = sample_index;
        gather.sampled_count += 1;
        let sample = gather.input.pixels[sample_index];
        gather.total[0] += sample[1] * sample_weight;
        gather.total[1] += sample[2] * sample_weight;
        gather.total[2] += sample[3] * sample_weight;
        gather.weight_total += sample_weight;
    }

    /// Applies DoF directly through the same production hook used by
    /// [`PostChain`].
    pub fn apply_to_framebuffer(&self, framebuffer: &mut Framebuffer) {
        if self.is_noop() {
            return;
        }
        let input = PostBuffer::from_framebuffer(framebuffer);
        let mut output = PostBuffer::new(input.width, input.height, PostColorSpace::Linear);
        output.hdr = input.hdr;
        self.apply_with_framebuffer(&input, &mut output, framebuffer);
        output.write_to_framebuffer(framebuffer);
    }

    fn reconstructed_depths(&self, framebuffer: &Framebuffer) -> Vec<Option<f32>> {
        let length = framebuffer.width.saturating_mul(framebuffer.height);
        let mut depths = vec![None; length];
        for y in 0..framebuffer.height {
            for x in 0..framebuffer.width {
                let index = y * framebuffer.width + x;
                let Some(&depth) = framebuffer.depth.get(index) else {
                    continue;
                };
                if depth >= 1.0 {
                    continue;
                }
                let Some(position) = reconstruct_view_position(
                    self.inverse_projection,
                    x,
                    y,
                    depth,
                    framebuffer.width,
                    framebuffer.height,
                ) else {
                    continue;
                };
                let linear_depth = -position.z;
                if linear_depth.is_finite() && linear_depth > 0.0 {
                    depths[index] = Some(linear_depth);
                }
            }
        }
        depths
    }

    fn gather_pixel(
        &self,
        input: &PostBuffer,
        depths: &[Option<f32>],
        x: usize,
        y: usize,
    ) -> [f32; 4] {
        let index = y * input.width + x;
        let source = input.pixels[index];
        let Some(center_depth) = depths[index] else {
            return source;
        };
        let center_radius = self.circle_of_confusion(center_depth);
        let mut gather = DofGather::new(input, depths, index, source, center_depth, center_radius);
        // Fractional CoCs blend with a one-texel gather. This keeps subpixel
        // blur visible without pretending that a fractional pixel is sampleable.
        let center_search_radius = if center_radius < 1.0 {
            1.0
        } else {
            center_radius
        };
        for &(offset_x, offset_y) in &DOF_DISC_KERNEL {
            let sample_x = (x as f32 + offset_x * center_search_radius).round() as isize;
            let sample_y = (y as f32 + offset_y * center_search_radius).round() as isize;
            let sample_x = sample_x.clamp(0, input.width.saturating_sub(1) as isize) as usize;
            let sample_y = sample_y.clamp(0, input.height.saturating_sub(1) as isize) as usize;
            let sample_index = sample_y * input.width + sample_x;
            let Some(_) = depths[sample_index] else {
                continue;
            };
            let sample_offset_x = sample_x as f32 - x as f32;
            let sample_offset_y = sample_y as f32 - y as f32;
            let sample_distance =
                (sample_offset_x * sample_offset_x + sample_offset_y * sample_offset_y).sqrt();
            let center_coverage = if center_radius <= DOF_FOCAL_COC_EPSILON {
                0.0
            } else if center_radius < 1.0 {
                1.0
            } else {
                f32::from(sample_distance <= center_radius + 0.5)
            };
            self.accumulate_sample(&mut gather, sample_x, sample_y, center_coverage);
        }

        // A sharp center still needs to search far enough to find a
        // defocused neighbor whose own CoC covers this pixel. This second
        // phase uses the pass-wide clamped radius, then shares the sample set
        // with the center gather so a tap contributes only once.
        let neighbor_search_radius = self.max_coc_radius.max(1.0);
        for &(offset_x, offset_y) in &DOF_DISC_KERNEL {
            let sample_x = (x as f32 + offset_x * neighbor_search_radius).round() as isize;
            let sample_y = (y as f32 + offset_y * neighbor_search_radius).round() as isize;
            let sample_x = sample_x.clamp(0, input.width.saturating_sub(1) as isize) as usize;
            let sample_y = sample_y.clamp(0, input.height.saturating_sub(1) as isize) as usize;
            let sample_index = sample_y * input.width + sample_x;
            let Some(sample_depth) = depths[sample_index] else {
                continue;
            };
            let sample_radius = self.circle_of_confusion(sample_depth);
            let sample_offset_x = sample_x as f32 - x as f32;
            let sample_offset_y = sample_y as f32 - y as f32;
            let sample_distance =
                (sample_offset_x * sample_offset_x + sample_offset_y * sample_offset_y).sqrt();
            let sample_coverage = if sample_radius <= DOF_FOCAL_COC_EPSILON {
                0.0
            } else {
                f32::from(sample_distance <= sample_radius + 0.5)
            };
            self.accumulate_sample(&mut gather, sample_x, sample_y, sample_coverage);
        }
        let gathered = [
            gather.total[0] / gather.weight_total,
            gather.total[1] / gather.weight_total,
            gather.total[2] / gather.weight_total,
        ];
        let blend = if center_radius <= DOF_FOCAL_COC_EPSILON {
            1.0
        } else {
            center_radius.min(1.0)
        };
        [
            source[0],
            source[1] + (gathered[0] - source[1]) * blend,
            source[2] + (gathered[1] - source[2]) * blend,
            source[3] + (gathered[2] - source[3]) * blend,
        ]
    }
}

impl PostPass for DofPass {
    fn color_space(&self) -> PostColorSpace {
        PostColorSpace::Linear
    }

    fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
        output.pixels.clone_from(&input.pixels);
    }

    fn is_noop(&self) -> bool {
        !self.projection_valid || self.aperture == 0.0 || self.max_coc_radius == 0.0
    }

    fn apply_with_framebuffer(
        &self,
        input: &PostBuffer,
        output: &mut PostBuffer,
        framebuffer: &Framebuffer,
    ) {
        debug_assert_eq!(input.color_space, PostColorSpace::Linear);
        debug_assert_eq!(output.color_space, PostColorSpace::Linear);
        let depths = self.reconstructed_depths(framebuffer);
        for y in 0..input.height {
            for x in 0..input.width {
                let index = y * input.width + x;
                output.pixels[index] = self.gather_pixel(input, &depths, x, y);
            }
        }
    }
}

pub fn dof(projection: Mat4) -> DofPass {
    DofPass::new(projection)
}

impl PostPass for SsaoPass {
    fn color_space(&self) -> PostColorSpace {
        PostColorSpace::Linear
    }

    fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
        // A depth-aware invocation comes through `apply_with_framebuffer`.
        // Keep the direct color-only trait call safe and unsurprising.
        output.pixels.clone_from(&input.pixels);
    }

    fn is_noop(&self) -> bool {
        !self.projection_valid
    }

    fn apply_with_framebuffer(
        &self,
        input: &PostBuffer,
        output: &mut PostBuffer,
        framebuffer: &Framebuffer,
    ) {
        debug_assert_eq!(input.color_space, PostColorSpace::Linear);
        debug_assert_eq!(output.color_space, PostColorSpace::Linear);
        let occlusion = self.occlusion_buffer(framebuffer);
        for (index, (destination, &source)) in
            output.pixels.iter_mut().zip(&input.pixels).enumerate()
        {
            let factor = 1.0 - self.strength * occlusion.get(index).copied().unwrap_or(0.0);
            *destination = [
                source[0],
                source[1] * factor,
                source[2] * factor,
                source[3] * factor,
            ];
        }
    }
}

pub fn ssao(projection: Mat4) -> SsaoPass {
    SsaoPass::new(projection)
}

fn choose_delta(center: Vec3, first: Option<Vec3>, second: Option<Vec3>) -> Vec3 {
    match (first, second) {
        (Some(first), Some(second)) => {
            if (center - first).length() <= (second - center).length() {
                center - first
            } else {
                second - center
            }
        }
        (Some(first), None) => center - first,
        (None, Some(second)) => second - center,
        (None, None) => Vec3::ZERO,
    }
}

fn sanitize_projection(projection: Mat4) -> (Mat4, Mat4, bool) {
    if projection.data.iter().all(|value| value.is_finite()) {
        if let Some(inverse) = projection.inverse() {
            return (projection, inverse, true);
        }
    }
    (Mat4::IDENTITY, Mat4::IDENTITY, false)
}

fn sanitize_positive(value: f32, fallback: f32, maximum: f32) -> f32 {
    if value.is_nan() {
        fallback
    } else if value.is_finite() {
        value.clamp(f32::EPSILON, maximum)
    } else if value.is_sign_positive() {
        maximum
    } else {
        fallback
    }
}

fn sanitize_nonnegative(value: f32, fallback: f32, maximum: f32) -> f32 {
    if value.is_nan() {
        fallback
    } else if value.is_finite() {
        value.clamp(0.0, maximum)
    } else if value.is_sign_positive() {
        maximum
    } else {
        fallback
    }
}

fn sanitize_unit(value: f32, fallback: f32) -> f32 {
    if value.is_nan() {
        fallback
    } else if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else if value.is_sign_positive() {
        1.0
    } else {
        0.0
    }
}

pub const BLOOM_THRESHOLD: f32 = 0.70;
pub const HDR_BLOOM_THRESHOLD: f32 = 1.0;
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
            let threshold = if input.hdr {
                HDR_BLOOM_THRESHOLD
            } else {
                BLOOM_THRESHOLD
            };
            *destination = bloom_threshold_extract([pixel[1], pixel[2], pixel[3]], threshold);
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
                let rgb = [
                    source[1] + blur[0] * BLOOM_STRENGTH,
                    source[2] + blur[1] * BLOOM_STRENGTH,
                    source[3] + blur[2] * BLOOM_STRENGTH,
                ];
                output.pixels[index] = [
                    source[0],
                    if input.hdr { rgb[0] } else { rgb[0].min(1.0) },
                    if input.hdr { rgb[1] } else { rgb[1].min(1.0) },
                    if input.hdr { rgb[2] } else { rgb[2].min(1.0) },
                ];
            }
        }
    }
}

/// Narkowicz's fitted ACES approximation. Exposure scales linear HDR values
/// before the fit, and the result stays linear until the final sRGB encode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcesTonemapPass {
    exposure: f32,
}

impl AcesTonemapPass {
    pub fn new(exposure: f32) -> Self {
        Self {
            exposure: sanitize_exposure(exposure),
        }
    }

    pub const fn exposure(self) -> f32 {
        self.exposure
    }
}

/// Maximum exposure accepted at the post-processing boundary.
pub const MAX_EXPOSURE: f32 = 100.0;

pub fn sanitize_exposure(exposure: f32) -> f32 {
    if exposure.is_nan() {
        1.0
    } else if exposure.is_finite() {
        exposure.clamp(0.0, MAX_EXPOSURE)
    } else if exposure.is_sign_positive() {
        MAX_EXPOSURE
    } else {
        1.0
    }
}

/// The fitted curve reaches its display ceiling near x=7.25. This cutover
/// avoids overflow in the quadratic terms for larger finite HDR values.
pub const ACES_SATURATION_CUTOFF: f32 = 8.0;

pub fn aces_tonemap(value: f32) -> f32 {
    let value = if value.is_nan() {
        return 0.0;
    } else if value.is_infinite() {
        return if value.is_sign_positive() { 1.0 } else { 0.0 };
    } else {
        value
    };
    if value >= ACES_SATURATION_CUTOFF {
        return 1.0;
    }
    let numerator = value * (2.51 * value + 0.03);
    let denominator = value * (2.43 * value + 0.59) + 0.14;
    if denominator > 0.0 {
        (numerator / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

impl PostPass for AcesTonemapPass {
    fn color_space(&self) -> PostColorSpace {
        PostColorSpace::Linear
    }

    fn apply(&self, input: &PostBuffer, output: &mut PostBuffer) {
        debug_assert_eq!(input.color_space, PostColorSpace::Linear);
        debug_assert_eq!(output.color_space, PostColorSpace::Linear);
        output.hdr = input.hdr;
        for (destination, &source) in output.pixels.iter_mut().zip(&input.pixels) {
            *destination = [
                source[0],
                aces_tonemap(source[1] * self.exposure),
                aces_tonemap(source[2] * self.exposure),
                aces_tonemap(source[3] * self.exposure),
            ];
        }
    }
}

pub fn aces(exposure: f32) -> AcesTonemapPass {
    AcesTonemapPass::new(exposure)
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

    #[test]
    fn aces_fit_matches_hand_computed_samples() {
        // x=0: numerator=0, so the fitted curve returns 0.
        assert_eq!(aces_tonemap(0.0), 0.0);
        // x=0.18: 0.18*(2.51*0.18+0.03)=0.086724;
        // denominator=0.18*(2.43*0.18+0.59)+0.14=0.324932;
        // 0.086724/0.324932=0.2668987.
        assert!((aces_tonemap(0.18) - 0.2668987).abs() < 1e-6);
        // x=1: numerator=2.54, denominator=3.16, result=0.8037975.
        assert!((aces_tonemap(1.0) - 0.8037975).abs() < 1e-6);
        // x=10: 251.3/249.04=1.009..., then the display bound gives 1.
        assert_eq!(aces_tonemap(10.0), 1.0);
    }

    #[test]
    fn aces_extreme_inputs_saturate_without_nan() {
        for value in [1.0e20, f32::MAX, f32::INFINITY] {
            let result = aces_tonemap(value);
            assert_eq!(result, 1.0);
            assert!(result.is_finite());
        }
    }

    #[test]
    fn exposure_sanitization_has_a_finite_sane_bound() {
        assert_eq!(sanitize_exposure(1.0e20), MAX_EXPOSURE);
        assert_eq!(sanitize_exposure(f32::INFINITY), MAX_EXPOSURE);
        assert_eq!(sanitize_exposure(f32::NAN), 1.0);
    }

    #[test]
    fn aces_exposure_scales_linear_input_before_tonemap() {
        let mut framebuffer = Framebuffer::new(1, 1);
        framebuffer.set_hdr(true);
        framebuffer.linear_pixels_mut().unwrap()[0] = [1.0, 0.5, 0.5, 0.5];
        PostChain::new()
            .with_pass(AcesTonemapPass::new(2.0))
            .apply(&mut framebuffer);
        let red = framebuffer.linear_pixels().unwrap()[0][1];
        assert!((red - aces_tonemap(1.0)).abs() < 1e-6);
    }
}
