//! The triangle rasterizer used by both serial and tile-binned paths.
//!
//! Rasterization uses screen-space barycentric coordinates for coverage and
//! depth. Varyings use perspective-corrected weights in the core.

use crate::fb::Framebuffer;
use crate::math::{Vec3, Vec4};
use crate::pipeline::{SampleDerivatives, SamplingVaryings, Varyings};
use std::cell::RefCell;

#[path = "raster_simd.rs"]
mod raster_simd;

#[derive(Clone, Copy, Debug)]
pub enum FragmentColor {
    Encoded(u32),
    Linear([f32; 4]),
}

pub trait FragmentOutput {
    fn write(self, framebuffer: &mut Framebuffer, index: usize, blend: bool);
}

/// Supplies an optional depth value to the shared depth-only raster kernel.
pub trait DepthVaryings: Varyings {
    fn depth(vertices: &[ScreenVertex<Self>; 3], weights: Vec3, inverse_w: Vec3) -> f32;
}

impl FragmentOutput for u32 {
    fn write(self, framebuffer: &mut Framebuffer, index: usize, blend: bool) {
        if let Some(color) = framebuffer.color.get_mut(index) {
            *color = if blend {
                crate::fb::blend_argb8888_linear(*color, self)
            } else {
                self
            };
        }
    }
}

impl FragmentOutput for FragmentColor {
    fn write(self, framebuffer: &mut Framebuffer, index: usize, blend: bool) {
        match self {
            Self::Encoded(color) => color.write(framebuffer, index, blend),
            Self::Linear(color) => crate::fb::write_linear_pixel(framebuffer, index, color, blend),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterState {
    pub depth_test: bool,
    pub depth_write: bool,
    pub color_write: bool,
    pub blend: bool,
}

impl RasterState {
    pub const OPAQUE: Self = Self {
        depth_test: true,
        depth_write: true,
        color_write: true,
        blend: false,
    };

    pub const TRANSPARENT: Self = Self {
        depth_test: true,
        depth_write: false,
        color_write: true,
        blend: true,
    };

    pub const DEPTH_ONLY: Self = Self {
        depth_test: true,
        depth_write: true,
        color_write: false,
        blend: false,
    };
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScreenVertex<V> {
    pub position: Vec3,
    pub inverse_w: f32,
    pub varyings: V,
}

impl<V> ScreenVertex<V> {
    pub const fn new(position: Vec3, varyings: V) -> Self {
        Self {
            position,
            inverse_w: 1.0,
            varyings,
        }
    }

    pub const fn with_inverse_w(position: Vec3, inverse_w: f32, varyings: V) -> Self {
        Self {
            position,
            inverse_w,
            varyings,
        }
    }
}

/// Converts affine screen-space weights into perspective-correct weights.
///
/// Each vertex contributes its screen weight divided by clip `w`. The result
/// is normalized so the corrected weights still sum to one.
pub fn perspective_correct_weights(weights: Vec3, inverse_w: Vec3) -> Vec3 {
    let weighted_inverse_w = weights * inverse_w;
    let sum = weighted_inverse_w.x + weighted_inverse_w.y + weighted_inverse_w.z;
    if sum == 0.0 || !sum.is_finite() {
        weights
    } else {
        weighted_inverse_w / sum
    }
}

/// Converts clip coordinates to pixel coordinates.
///
/// Pixel centers are at `(x + 0.5, y + 0.5)`. The viewport maps NDC +Y to
/// the top of the framebuffer because framebuffer rows increase downward.
pub fn viewport_transform(clip: Vec4, width: usize, height: usize) -> Option<Vec3> {
    if clip.w == 0.0 || !clip.x.is_finite() || !clip.y.is_finite() || !clip.z.is_finite() {
        return None;
    }
    let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
    if !ndc.x.is_finite() || !ndc.y.is_finite() || !ndc.z.is_finite() {
        return None;
    }
    Some(Vec3::new(
        (ndc.x + 1.0) * 0.5 * width as f32,
        (1.0 - ndc.y) * 0.5 * height as f32,
        ndc.z,
    ))
}

fn edge(a: Vec3, b: Vec3, point: Vec3) -> f32 {
    (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
}

// With y increasing down the screen, an edge is top-left when it points up,
// or is horizontal and points right. This is the D3D-style top-left rule.
fn is_top_left(a: Vec3, b: Vec3) -> bool {
    let dy = b.y - a.y;
    let dx = b.x - a.x;
    dy < 0.0 || (dy == 0.0 && dx > 0.0)
}

fn inside(value: f32, top_left: bool) -> bool {
    value > 0.0 || (value == 0.0 && top_left)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PixelRect {
    pub min_x: i32,
    pub max_x: i32,
    pub min_y: i32,
    pub max_y: i32,
}

impl PixelRect {
    pub const fn empty() -> Self {
        Self {
            min_x: 1,
            max_x: 0,
            min_y: 1,
            max_y: 0,
        }
    }

    pub const fn is_empty(self) -> bool {
        self.min_x > self.max_x || self.min_y > self.max_y
    }
}

#[derive(Default)]
struct RasterRow {
    // The row keeps each hot scalar in one contiguous lane array. This is
    // the shared input for scalar fallback and the four-lane depth dispatch.
    covered: Vec<u8>,
    indices: Vec<usize>,
    depth: Vec<f32>,
    weights_x: Vec<f32>,
    weights_y: Vec<f32>,
    weights_z: Vec<f32>,
}

impl RasterRow {
    fn resize(&mut self, length: usize) {
        self.covered.resize(length, 0);
        self.covered.fill(0);
        self.indices.resize(length, 0);
        self.depth.resize(length, 0.0);
        self.weights_x.resize(length, 0.0);
        self.weights_y.resize(length, 0.0);
        self.weights_z.resize(length, 0.0);
    }
}

thread_local! {
    static RASTER_ROW: RefCell<RasterRow> = RefCell::new(RasterRow::default());
}

#[inline]
fn simd_enabled() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::env::var_os("CHIMY_NO_SIMD").is_none()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub(crate) fn triangle_pixel_rect<V>(
    vertices: &[ScreenVertex<V>; 3],
    width: usize,
    height: usize,
) -> PixelRect {
    if width == 0 || height == 0 {
        return PixelRect::empty();
    }

    let min_x = vertices
        .iter()
        .map(|vertex| vertex.position.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = vertices
        .iter()
        .map(|vertex| vertex.position.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = vertices
        .iter()
        .map(|vertex| vertex.position.y)
        .fold(f32::INFINITY, f32::min);
    let max_y = vertices
        .iter()
        .map(|vertex| vertex.position.y)
        .fold(f32::NEG_INFINITY, f32::max);

    PixelRect {
        min_x: ((min_x - 0.5).ceil() as i32).clamp(0, width as i32 - 1),
        max_x: ((max_x - 0.5).floor() as i32).clamp(0, width as i32 - 1),
        min_y: ((min_y - 0.5).ceil() as i32).clamp(0, height as i32 - 1),
        max_y: ((max_y - 0.5).floor() as i32).clamp(0, height as i32 - 1),
    }
}

/// Rasterizes one triangle and invokes `fragment` for every passing pixel.
///
/// Zero-area triangles are culled. The bounding box is clamped to the
/// framebuffer. The top-left fill rule gives shared edges to one triangle.
/// Depth uses OpenGL-style NDC: `-1` is near and `+1` is far. NDC z is already
/// post-divide, so its screen-space affine interpolation is the correct depth
/// interpolation. Varyings use perspective-corrected interpolation instead.
/// The framebuffer clears to `1`; a fragment passes only when its depth is
/// strictly smaller.
pub fn rasterize_triangle<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    fragment: F,
) where
    V: Varyings,
    F: FnMut(V) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect(
        framebuffer,
        vertices,
        PixelRect {
            min_x: 0,
            max_x: framebuffer.width.saturating_sub(1) as i32,
            min_y: 0,
            max_y: framebuffer.height.saturating_sub(1) as i32,
        },
        fragment,
    )
}

pub fn rasterize_triangle_with_state<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    state: RasterState,
    fragment: F,
) where
    V: Varyings,
    F: FnMut(V) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect_with_state(
        framebuffer,
        vertices,
        PixelRect {
            min_x: 0,
            max_x: framebuffer.width.saturating_sub(1) as i32,
            min_y: 0,
            max_y: framebuffer.height.saturating_sub(1) as i32,
        },
        state,
        fragment,
    )
}

/// Rasterizes one triangle while updating depth without writing color.
pub fn rasterize_triangle_depth<V>(framebuffer: &mut Framebuffer, vertices: [ScreenVertex<V>; 3])
where
    V: Varyings,
{
    rasterize_triangle_depth_in_rect(
        framebuffer,
        vertices,
        PixelRect {
            min_x: 0,
            max_x: framebuffer.width.saturating_sub(1) as i32,
            min_y: 0,
            max_y: framebuffer.height.saturating_sub(1) as i32,
        },
    );
}

/// Rasterizes one triangle and passes sampler-facing derivatives with each
/// fragment. Only sampling varyings opt into this channel.
pub fn rasterize_triangle_with_sampling<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    fragment: F,
) where
    V: SamplingVaryings,
    F: FnMut(V, SampleDerivatives) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect_with_sampling(
        framebuffer,
        vertices,
        PixelRect {
            min_x: 0,
            max_x: framebuffer.width.saturating_sub(1) as i32,
            min_y: 0,
            max_y: framebuffer.height.saturating_sub(1) as i32,
        },
        fragment,
    );
}

pub fn rasterize_triangle_with_sampling_state<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    state: RasterState,
    fragment: F,
) where
    V: SamplingVaryings,
    F: FnMut(V, SampleDerivatives) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect_with_sampling_state(
        framebuffer,
        vertices,
        PixelRect {
            min_x: 0,
            max_x: framebuffer.width.saturating_sub(1) as i32,
            min_y: 0,
            max_y: framebuffer.height.saturating_sub(1) as i32,
        },
        state,
        fragment,
    )
}

/// Rasterizes one triangle inside a caller-owned pixel rectangle.
///
/// The serial path passes the full framebuffer rectangle. A parallel worker
/// passes its tile-local framebuffer, so its scan is clamped to that tile.
/// The per-pixel coverage, depth, interpolation, and fragment code are shared
/// by both paths. Each tile has one worker owner, so no two workers write the
/// same color or depth element.
pub(crate) fn rasterize_triangle_in_rect<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    rect: PixelRect,
    fragment: F,
) where
    V: Varyings,
    F: FnMut(V) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect_core(
        framebuffer,
        vertices,
        rect,
        RasterState::OPAQUE,
        simd_enabled(),
        |vertices, weights, _| interpolated_depth(vertices, weights),
        |vertices, weights, inverse_w, _, _| {
            V::lerp3(
                &vertices[0].varyings,
                &vertices[1].varyings,
                &vertices[2].varyings,
                perspective_correct_weights(weights, inverse_w),
            )
        },
        fragment,
    );
}

pub(crate) fn rasterize_triangle_in_rect_with_state<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    rect: PixelRect,
    state: RasterState,
    fragment: F,
) where
    V: Varyings,
    F: FnMut(V) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect_core(
        framebuffer,
        vertices,
        rect,
        state,
        simd_enabled(),
        |vertices, weights, _| interpolated_depth(vertices, weights),
        |vertices, weights, inverse_w, _, _| {
            V::lerp3(
                &vertices[0].varyings,
                &vertices[1].varyings,
                &vertices[2].varyings,
                perspective_correct_weights(weights, inverse_w),
            )
        },
        fragment,
    );
}

pub(crate) fn rasterize_triangle_depth_in_rect<V>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    rect: PixelRect,
) where
    V: Varyings,
{
    rasterize_triangle_in_rect_core(
        framebuffer,
        vertices,
        rect,
        RasterState::DEPTH_ONLY,
        simd_enabled(),
        |vertices, weights, _| interpolated_depth(vertices, weights),
        |vertices, weights, inverse_w, _, _| {
            V::lerp3(
                &vertices[0].varyings,
                &vertices[1].varyings,
                &vertices[2].varyings,
                perspective_correct_weights(weights, inverse_w),
            )
        },
        |_| 0,
    );
}

pub(crate) fn rasterize_triangle_depth_with_varyings_in_rect<V>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    rect: PixelRect,
) where
    V: DepthVaryings,
{
    rasterize_triangle_in_rect_core(
        framebuffer,
        vertices,
        rect,
        RasterState::DEPTH_ONLY,
        simd_enabled(),
        V::depth,
        |vertices, weights, inverse_w, _, _| {
            V::lerp3(
                &vertices[0].varyings,
                &vertices[1].varyings,
                &vertices[2].varyings,
                perspective_correct_weights(weights, inverse_w),
            )
        },
        |_| 0,
    );
}

pub(crate) fn rasterize_triangle_in_rect_with_sampling<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    rect: PixelRect,
    mut fragment: F,
) where
    V: SamplingVaryings,
    F: FnMut(V, SampleDerivatives) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect_core(
        framebuffer,
        vertices,
        rect,
        RasterState::OPAQUE,
        simd_enabled(),
        |vertices, weights, _| interpolated_depth(vertices, weights),
        |vertices, weights, inverse_w, ddx_weights, ddy_weights| {
            let varyings = V::lerp3(
                &vertices[0].varyings,
                &vertices[1].varyings,
                &vertices[2].varyings,
                perspective_correct_weights(weights, inverse_w),
            );
            (
                varyings,
                texture_derivatives(vertices, weights, inverse_w, ddx_weights, ddy_weights),
            )
        },
        |(varyings, derivatives)| fragment(varyings, derivatives),
    );
}

pub(crate) fn rasterize_triangle_in_rect_with_sampling_state<V, F, O>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    rect: PixelRect,
    state: RasterState,
    mut fragment: F,
) where
    V: SamplingVaryings,
    F: FnMut(V, SampleDerivatives) -> O,
    O: FragmentOutput,
{
    rasterize_triangle_in_rect_core(
        framebuffer,
        vertices,
        rect,
        state,
        simd_enabled(),
        |vertices, weights, _| interpolated_depth(vertices, weights),
        |vertices, weights, inverse_w, ddx_weights, ddy_weights| {
            let varyings = V::lerp3(
                &vertices[0].varyings,
                &vertices[1].varyings,
                &vertices[2].varyings,
                perspective_correct_weights(weights, inverse_w),
            );
            (
                varyings,
                texture_derivatives(vertices, weights, inverse_w, ddx_weights, ddy_weights),
            )
        },
        |(varyings, derivatives)| fragment(varyings, derivatives),
    );
}

#[allow(clippy::too_many_arguments)]
fn rasterize_triangle_in_rect_core<V, Input, Interpolate, Fragment, O, Depth>(
    framebuffer: &mut Framebuffer,
    mut vertices: [ScreenVertex<V>; 3],
    rect: PixelRect,
    state: RasterState,
    use_simd: bool,
    mut depth_value: Depth,
    mut interpolate: Interpolate,
    mut fragment: Fragment,
) where
    V: Varyings,
    Depth: FnMut(&[ScreenVertex<V>; 3], Vec3, Vec3) -> f32,
    Interpolate: FnMut(&[ScreenVertex<V>; 3], Vec3, Vec3, Vec3, Vec3) -> Input,
    Fragment: FnMut(Input) -> O,
    O: FragmentOutput,
{
    let mut area = edge(
        vertices[0].position,
        vertices[1].position,
        vertices[2].position,
    );
    if !area.is_finite() || area == 0.0 {
        return;
    }
    if area < 0.0 {
        vertices.swap(1, 2);
        area = -area;
    }

    if framebuffer.width == 0 || framebuffer.height == 0 || rect.is_empty() {
        return;
    }

    let bounds = triangle_pixel_rect(&vertices, framebuffer.width, framebuffer.height);
    let min_x = bounds.min_x.max(rect.min_x);
    let max_x = bounds.max_x.min(rect.max_x);
    let min_y = bounds.min_y.max(rect.min_y);
    let max_y = bounds.max_y.min(rect.max_y);
    if min_x > max_x || min_y > max_y {
        return;
    }

    let top_left_0 = is_top_left(vertices[1].position, vertices[2].position);
    let top_left_1 = is_top_left(vertices[2].position, vertices[0].position);
    let top_left_2 = is_top_left(vertices[0].position, vertices[1].position);

    let inverse_w_vector = inverse_w(&vertices);
    let ddx_weights = Vec3::new(
        -(vertices[2].position.y - vertices[1].position.y) / area,
        -(vertices[0].position.y - vertices[2].position.y) / area,
        -(vertices[1].position.y - vertices[0].position.y) / area,
    );
    let ddy_weights = Vec3::new(
        (vertices[2].position.x - vertices[1].position.x) / area,
        (vertices[0].position.x - vertices[2].position.x) / area,
        (vertices[1].position.x - vertices[0].position.x) / area,
    );
    let row_length = (max_x - min_x + 1) as usize;
    RASTER_ROW.with_borrow_mut(|row| {
        for y in min_y..=max_y {
            row.resize(row_length);
            for x in min_x..=max_x {
                let row_offset = (x - min_x) as usize;
                let point = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let weights = barycentric_weights(&vertices, point, area);
                if !covered(weights, area, top_left_0, top_left_1, top_left_2) {
                    continue;
                }

                let depth = depth_value(&vertices, weights, inverse_w_vector);
                let Ok(x) = usize::try_from(x) else { continue };
                let Ok(y) = usize::try_from(y) else { continue };
                let Some(index) = y
                    .checked_mul(framebuffer.width)
                    .and_then(|row| row.checked_add(x))
                else {
                    continue;
                };
                row.covered[row_offset] = 1;
                row.indices[row_offset] = index;
                row.depth[row_offset] = depth;
                row.weights_x[row_offset] = weights.x;
                row.weights_y[row_offset] = weights.y;
                row.weights_z[row_offset] = weights.z;
            }

            let mut offset = 0;
            while offset < row_length {
                let lane_count = (row_length - offset).min(4);
                let simd_mask = if use_simd && lane_count == 4 {
                    // Coverage and depth use four-lane NEON. Generic varying
                    // interpolation and fragment closures remain scalar because
                    // their types are shader-defined and may sample or blend.
                    let first_index = row.indices[offset];
                    let contiguous = (0..4).all(|lane| {
                        first_index
                            .checked_add(lane)
                            .is_some_and(|expected| row.indices[offset + lane] == expected)
                    });
                    if contiguous {
                        raster_simd::depth_mask(
                            &row.covered,
                            &row.depth,
                            &framebuffer.depth,
                            offset,
                            first_index,
                            state.depth_test,
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };

                let mut process_lane = |lane: usize, depth_already_passed: bool| {
                    if row.covered[lane] == 0 {
                        return;
                    }
                    let index = row.indices[lane];
                    let depth = row.depth[lane];
                    let Some(buffer_depth) = framebuffer.depth.get(index) else {
                        return;
                    };
                    if state.depth_test && !depth_already_passed && depth >= *buffer_depth {
                        return;
                    }
                    let weights = Vec3::new(
                        row.weights_x[lane],
                        row.weights_y[lane],
                        row.weights_z[lane],
                    );
                    let input = interpolate(
                        &vertices,
                        weights,
                        inverse_w_vector,
                        ddx_weights,
                        ddy_weights,
                    );
                    if state.depth_write {
                        let Some(buffer_depth) = framebuffer.depth.get_mut(index) else {
                            return;
                        };
                        *buffer_depth = depth;
                    }
                    if state.color_write {
                        fragment(input).write(framebuffer, index, state.blend);
                    }
                };

                if let Some(mask) = simd_mask {
                    for lane in 0..4 {
                        if mask & (1 << lane) != 0 {
                            process_lane(offset + lane, true);
                        }
                    }
                } else {
                    for lane in offset..offset + lane_count {
                        process_lane(lane, false);
                    }
                }
                offset += lane_count;
            }
        }
    });
}

fn barycentric_weights<V>(vertices: &[ScreenVertex<V>; 3], point: Vec3, area: f32) -> Vec3 {
    Vec3::new(
        edge(vertices[1].position, vertices[2].position, point) / area,
        edge(vertices[2].position, vertices[0].position, point) / area,
        edge(vertices[0].position, vertices[1].position, point) / area,
    )
}

fn covered(weights: Vec3, area: f32, top_left_0: bool, top_left_1: bool, top_left_2: bool) -> bool {
    inside(weights.x * area, top_left_0)
        && inside(weights.y * area, top_left_1)
        && inside(weights.z * area, top_left_2)
}

fn interpolated_depth<V>(vertices: &[ScreenVertex<V>; 3], weights: Vec3) -> f32 {
    vertices[0].position.z * weights.x
        + vertices[1].position.z * weights.y
        + vertices[2].position.z * weights.z
}

fn inverse_w<V>(vertices: &[ScreenVertex<V>; 3]) -> Vec3 {
    Vec3::new(
        vertices[0].inverse_w,
        vertices[1].inverse_w,
        vertices[2].inverse_w,
    )
}

fn texture_derivatives<V: SamplingVaryings>(
    vertices: &[ScreenVertex<V>; 3],
    weights: Vec3,
    inverse_w: Vec3,
    ddx_weights: Vec3,
    ddy_weights: Vec3,
) -> SampleDerivatives {
    let coordinates = [
        vertices[0].varyings.texture_coordinates(),
        vertices[1].varyings.texture_coordinates(),
        vertices[2].varyings.texture_coordinates(),
    ];
    let q = weights.x * inverse_w.x + weights.y * inverse_w.y + weights.z * inverse_w.z;
    let numerator = coordinates[0] * (weights.x * inverse_w.x)
        + coordinates[1] * (weights.y * inverse_w.y)
        + coordinates[2] * (weights.z * inverse_w.z);
    let derivative = |gradient: Vec3| {
        let dq = gradient.x * inverse_w.x + gradient.y * inverse_w.y + gradient.z * inverse_w.z;
        let dn = coordinates[0] * (gradient.x * inverse_w.x)
            + coordinates[1] * (gradient.y * inverse_w.y)
            + coordinates[2] * (gradient.z * inverse_w.z);
        (dn * q - numerator * dq) / (q * q)
    };
    SampleDerivatives {
        ddx: derivative(ddx_weights),
        ddy: derivative(ddy_weights),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::argb8888;
    use crate::pipeline::ColorVarying;

    fn triangle(z: f32) -> [ScreenVertex<()>; 3] {
        [
            ScreenVertex::new(Vec3::new(1.0, 1.0, z), ()),
            ScreenVertex::new(Vec3::new(5.0, 1.0, z), ()),
            ScreenVertex::new(Vec3::new(1.0, 5.0, z), ()),
        ]
    }

    #[test]
    fn viewport_maps_ndc_to_pixel_coordinates() {
        assert_eq!(
            viewport_transform(Vec4::new(-1.0, 1.0, -1.0, 1.0), 8, 6),
            Some(Vec3::new(0.0, 0.0, -1.0))
        );
        assert_eq!(
            viewport_transform(Vec4::new(1.0, -1.0, 1.0, 1.0), 8, 6),
            Some(Vec3::new(8.0, 6.0, 1.0))
        );
    }

    #[test]
    fn depth_test_keeps_nearer_triangle() {
        let mut framebuffer = Framebuffer::new(6, 6);
        rasterize_triangle(&mut framebuffer, triangle(0.5), |_| {
            argb8888(255, 255, 0, 0)
        });
        rasterize_triangle(&mut framebuffer, triangle(-0.5), |_| {
            argb8888(255, 0, 255, 0)
        });
        assert!(framebuffer.color.contains(&0xff00ff00));
        assert!(!framebuffer.color.contains(&0xffff0000));
    }

    #[test]
    fn raster_interpolates_varyings_at_pixel_center() {
        let mut framebuffer = Framebuffer::new(6, 6);
        rasterize_triangle(
            &mut framebuffer,
            [
                ScreenVertex::new(
                    Vec3::new(1.0, 1.0, 0.0),
                    ColorVarying::new(Vec4::new(1.0, 0.0, 0.0, 1.0)),
                ),
                ScreenVertex::new(
                    Vec3::new(5.0, 1.0, 0.0),
                    ColorVarying::new(Vec4::new(0.0, 1.0, 0.0, 1.0)),
                ),
                ScreenVertex::new(
                    Vec3::new(1.0, 5.0, 0.0),
                    ColorVarying::new(Vec4::new(0.0, 0.0, 1.0, 1.0)),
                ),
            ],
            |varyings| {
                let color = varyings.color;
                argb8888(
                    255,
                    (color.x * 255.0).round() as u8,
                    (color.y * 255.0).round() as u8,
                    (color.z * 255.0).round() as u8,
                )
            },
        );

        // At pixel center (2.5, 2.5), the hand-calculated weights are
        // (1/4, 3/8, 3/8), so RGB rounds to (64, 96, 96).
        assert_eq!(framebuffer.color[2 * framebuffer.width + 2], 0xff406060);
    }

    #[test]
    fn perspective_weights_match_hand_computed_triangle() {
        let affine = Vec3::new(0.25, 0.5, 0.25);
        let corrected = perspective_correct_weights(affine, Vec3::new(1.0, 0.5, 0.25));
        let denominator = 0.25 + 0.25 + 0.0625;
        assert!((corrected.x - 0.25 / denominator).abs() < 1e-6);
        assert!((corrected.y - 0.25 / denominator).abs() < 1e-6);
        assert!((corrected.z - 0.0625 / denominator).abs() < 1e-6);
    }

    #[test]
    fn raster_perspective_corrects_varying_at_pixel_center() {
        let mut framebuffer = Framebuffer::new(6, 6);
        rasterize_triangle(
            &mut framebuffer,
            [
                ScreenVertex::with_inverse_w(
                    Vec3::new(1.0, 1.0, 0.0),
                    1.0,
                    ColorVarying::new(Vec4::new(1.0, 0.0, 0.0, 1.0)),
                ),
                ScreenVertex::with_inverse_w(
                    Vec3::new(5.0, 1.0, 0.0),
                    0.5,
                    ColorVarying::new(Vec4::new(0.0, 1.0, 0.0, 1.0)),
                ),
                ScreenVertex::with_inverse_w(
                    Vec3::new(1.0, 5.0, 0.0),
                    0.25,
                    ColorVarying::new(Vec4::new(0.0, 0.0, 1.0, 1.0)),
                ),
            ],
            |varyings| {
                let color = varyings.color;
                argb8888(
                    255,
                    (color.x * 255.0).round() as u8,
                    (color.y * 255.0).round() as u8,
                    (color.z * 255.0).round() as u8,
                )
            },
        );

        // Affine weights are (1/4, 3/8, 3/8). Dividing by clip w gives
        // (1/4, 3/16, 3/32), then normalizing gives (8/17, 6/17, 3/17).
        assert_eq!(framebuffer.color[2 * framebuffer.width + 2], 0xff785a2d);
    }

    #[test]
    fn degenerate_triangle_is_culled() {
        let mut framebuffer = Framebuffer::new(4, 4);
        rasterize_triangle(
            &mut framebuffer,
            [
                ScreenVertex::new(Vec3::new(1.0, 1.0, 0.0), ()),
                ScreenVertex::new(Vec3::new(2.0, 2.0, 0.0), ()),
                ScreenVertex::new(Vec3::new(3.0, 3.0, 0.0), ()),
            ],
            |_| 0xffff_ffff,
        );
        assert_eq!(framebuffer.color, vec![0; 16]);
    }

    fn render_gradient(width: usize, height: usize, use_simd: bool) -> Framebuffer {
        let mut framebuffer = Framebuffer::new(width, height);
        let vertices = [
            ScreenVertex::new(
                Vec3::new(0.0, 0.0, 0.0),
                ColorVarying::new(Vec4::new(1.0, 0.0, 0.0, 1.0)),
            ),
            ScreenVertex::new(
                Vec3::new(width as f32, 0.0, 0.0),
                ColorVarying::new(Vec4::new(0.0, 1.0, 0.0, 1.0)),
            ),
            ScreenVertex::new(
                Vec3::new(0.0, height as f32, 0.0),
                ColorVarying::new(Vec4::new(0.0, 0.0, 1.0, 1.0)),
            ),
        ];
        rasterize_triangle_in_rect_core(
            &mut framebuffer,
            vertices,
            PixelRect {
                min_x: 0,
                max_x: width.saturating_sub(1) as i32,
                min_y: 0,
                max_y: height.saturating_sub(1) as i32,
            },
            RasterState::OPAQUE,
            use_simd,
            |vertices, weights, _| interpolated_depth(vertices, weights),
            |vertices, weights, inverse_w, _, _| {
                ColorVarying::lerp3(
                    &vertices[0].varyings,
                    &vertices[1].varyings,
                    &vertices[2].varyings,
                    perspective_correct_weights(weights, inverse_w),
                )
            },
            |varyings| {
                let color = varyings.color;
                argb8888(
                    255,
                    (color.x * 255.0).round() as u8,
                    (color.y * 255.0).round() as u8,
                    (color.z * 255.0).round() as u8,
                )
            },
        );
        framebuffer
    }

    #[test]
    fn simd_and_scalar_modes_match_for_non_multiple_widths() {
        for width in [1, 63, 65] {
            assert_eq!(
                render_gradient(width, 7, false),
                render_gradient(width, 7, true),
                "raster modes differ at width {width}"
            );
        }
    }

    #[test]
    fn empty_depth_buffer_falls_back_without_writing_color() {
        raster_simd::reset_depth_fallbacks();
        let mut framebuffer = Framebuffer::new(8, 4);
        framebuffer.depth.clear();
        rasterize_triangle_in_rect_core(
            &mut framebuffer,
            [
                ScreenVertex::new(Vec3::new(-100.0, 1.0, 0.0), ()),
                ScreenVertex::new(Vec3::new(100.0, 1.0, 0.0), ()),
                ScreenVertex::new(Vec3::new(0.0, 2.0, 0.0), ()),
            ],
            PixelRect {
                min_x: 0,
                max_x: 7,
                min_y: 0,
                max_y: 3,
            },
            RasterState::OPAQUE,
            true,
            |vertices, weights, _| interpolated_depth(vertices, weights),
            |_, _, _, _, _| (),
            |_| argb8888(255, 255, 255, 255),
        );
        assert!(framebuffer.color.iter().all(|&color| color == 0));
        assert!(raster_simd::depth_fallbacks() > 0);
    }

    #[test]
    fn short_last_row_falls_back_and_preserves_missing_pixel() {
        raster_simd::reset_depth_fallbacks();
        let mut framebuffer = Framebuffer::new(4, 2);
        framebuffer.depth.pop();
        rasterize_triangle_in_rect_core(
            &mut framebuffer,
            [
                ScreenVertex::new(Vec3::new(-100.0, 1.0, 0.0), ()),
                ScreenVertex::new(Vec3::new(100.0, 1.0, 0.0), ()),
                ScreenVertex::new(Vec3::new(0.0, 2.0, 0.0), ()),
            ],
            PixelRect {
                min_x: 0,
                max_x: 3,
                min_y: 0,
                max_y: 1,
            },
            RasterState::OPAQUE,
            true,
            |vertices, weights, _| interpolated_depth(vertices, weights),
            |_, _, _, _, _| (),
            |_| argb8888(255, 255, 255, 255),
        );
        assert!(framebuffer.color[..7].contains(&0xffff_ffff));
        assert_eq!(framebuffer.color[7], 0);
        assert!(raster_simd::depth_fallbacks() > 0);
    }
}
