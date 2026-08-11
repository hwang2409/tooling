//! AArch64 NEON backend for the rasterizer's coverage and depth loop.

use super::{PixelRect, ScreenVertex, rasterize_pixel, write_pixel};
use crate::fb::Framebuffer;
use crate::math::Vec3;
use crate::pipeline::Varyings;
use core::arch::aarch64::*;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
static DEPTH_FALLBACKS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct RasterSpan {
    area: f32,
    x: i32,
    y: i32,
    max_x: i32,
    top_left: [bool; 3],
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rasterize_prepared<V, Input, Interpolate, Fragment>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    area: f32,
    bounds: PixelRect,
    top_left: [bool; 3],
    ddx_weights: Vec3,
    ddy_weights: Vec3,
    inverse_w: Vec3,
    mut interpolate: Interpolate,
    mut fragment: Fragment,
) where
    V: Varyings,
    Interpolate: FnMut(&[ScreenVertex<V>; 3], Vec3, Vec3, Vec3, Vec3) -> Input,
    Fragment: FnMut(Input) -> u32,
{
    for y in bounds.min_y..=bounds.max_y {
        let mut x = bounds.min_x;
        while bounds.max_x - x + 1 >= 4 {
            let span = RasterSpan {
                area,
                x,
                y,
                max_x: bounds.max_x,
                top_left,
            };
            let depth_ptr = (y as usize)
                .checked_mul(framebuffer.width)
                .and_then(|row| row.checked_add(x as usize))
                .and_then(|index| {
                    index
                        .checked_add(4)
                        .and_then(|end| framebuffer.depth.get(index..end))
                        .map(|depth_span| depth_span.as_ptr())
                });
            if let Some(depth_ptr) = depth_ptr {
                unsafe {
                    // SAFETY: depth_ptr comes from a checked four-element
                    // depth slice. The prepared bounds contain every lane.
                    rasterize_quad(
                        framebuffer,
                        &vertices,
                        span,
                        depth_ptr,
                        inverse_w,
                        ddx_weights,
                        ddy_weights,
                        &mut interpolate,
                        &mut fragment,
                    );
                }
            } else {
                #[cfg(test)]
                DEPTH_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                rasterize_scalar_quad(
                    framebuffer,
                    &vertices,
                    span,
                    inverse_w,
                    ddx_weights,
                    ddy_weights,
                    &mut interpolate,
                    &mut fragment,
                );
            }
            x = x.saturating_add(4);
        }
        while x <= bounds.max_x {
            rasterize_pixel(
                framebuffer,
                &vertices,
                area,
                x,
                y,
                top_left,
                inverse_w,
                ddx_weights,
                ddy_weights,
                &mut interpolate,
                &mut fragment,
            );
            x = x.saturating_add(1);
        }
    }
}

#[cfg(test)]
pub(super) fn reset_depth_fallbacks() {
    DEPTH_FALLBACKS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn depth_fallbacks() -> usize {
    DEPTH_FALLBACKS.load(Ordering::Relaxed)
}

#[allow(clippy::too_many_arguments)]
fn rasterize_scalar_quad<V, Input, Interpolate, Fragment>(
    framebuffer: &mut Framebuffer,
    vertices: &[ScreenVertex<V>; 3],
    span: RasterSpan,
    inverse_w: Vec3,
    ddx_weights: Vec3,
    ddy_weights: Vec3,
    interpolate: &mut Interpolate,
    fragment: &mut Fragment,
) where
    V: Varyings,
    Interpolate: FnMut(&[ScreenVertex<V>; 3], Vec3, Vec3, Vec3, Vec3) -> Input,
    Fragment: FnMut(Input) -> u32,
{
    let scalar_end = span.x.saturating_add(3).min(span.max_x);
    for scalar_x in span.x..=scalar_end {
        rasterize_pixel(
            framebuffer,
            vertices,
            span.area,
            scalar_x,
            span.y,
            span.top_left,
            inverse_w,
            ddx_weights,
            ddy_weights,
            interpolate,
            fragment,
        );
    }
}

/// # Safety
/// `depth_ptr` must point to four initialized f32 values for this span.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn rasterize_quad<V, Input, Interpolate, Fragment>(
    framebuffer: &mut Framebuffer,
    vertices: &[ScreenVertex<V>; 3],
    span: RasterSpan,
    depth_ptr: *const f32,
    inverse_w: Vec3,
    ddx_weights: Vec3,
    ddy_weights: Vec3,
    interpolate: &mut Interpolate,
    fragment: &mut Fragment,
) where
    V: Varyings,
    Interpolate: FnMut(&[ScreenVertex<V>; 3], Vec3, Vec3, Vec3, Vec3) -> Input,
    Fragment: FnMut(Input) -> u32,
{
    let xs = [
        span.x as f32 + 0.5,
        (span.x + 1) as f32 + 0.5,
        (span.x + 2) as f32 + 0.5,
        (span.x + 3) as f32 + 0.5,
    ];
    let mut passing_lanes = [0_u32; 4];
    let mut depth_lanes = [0.0_f32; 4];
    let mut weights_x_lanes = [0.0_f32; 4];
    let mut weights_y_lanes = [0.0_f32; 4];
    let mut weights_z_lanes = [0.0_f32; 4];
    unsafe {
        // SAFETY: xs and depth_ptr each contain four valid lanes. The
        // destination arrays contain four writable lanes. NEON uses no fused
        // operations here, matching the scalar multiply-then-add order.
        let px = vld1q_f32(xs.as_ptr());
        let py = vdupq_n_f32(span.y as f32 + 0.5);
        let area_v = vdupq_n_f32(span.area);
        let a0 = vertices[1].position;
        let b0 = vertices[2].position;
        let edge_0 = vsubq_f32(
            vmulq_f32(vdupq_n_f32(b0.x - a0.x), vsubq_f32(py, vdupq_n_f32(a0.y))),
            vmulq_f32(vdupq_n_f32(b0.y - a0.y), vsubq_f32(px, vdupq_n_f32(a0.x))),
        );
        let a1 = vertices[2].position;
        let b1 = vertices[0].position;
        let edge_1 = vsubq_f32(
            vmulq_f32(vdupq_n_f32(b1.x - a1.x), vsubq_f32(py, vdupq_n_f32(a1.y))),
            vmulq_f32(vdupq_n_f32(b1.y - a1.y), vsubq_f32(px, vdupq_n_f32(a1.x))),
        );
        let a2 = vertices[0].position;
        let b2 = vertices[1].position;
        let edge_2 = vsubq_f32(
            vmulq_f32(vdupq_n_f32(b2.x - a2.x), vsubq_f32(py, vdupq_n_f32(a2.y))),
            vmulq_f32(vdupq_n_f32(b2.y - a2.y), vsubq_f32(px, vdupq_n_f32(a2.x))),
        );
        let weights_x = vdivq_f32(edge_0, area_v);
        let weights_y = vdivq_f32(edge_1, area_v);
        let weights_z = vdivq_f32(edge_2, area_v);
        let scaled_0 = vmulq_f32(weights_x, area_v);
        let positive_0 = vcgtq_f32(scaled_0, vdupq_n_f32(0.0));
        let inside_0 = if span.top_left[0] {
            vorrq_u32(positive_0, vceqq_f32(scaled_0, vdupq_n_f32(0.0)))
        } else {
            positive_0
        };
        let scaled_1 = vmulq_f32(weights_y, area_v);
        let positive_1 = vcgtq_f32(scaled_1, vdupq_n_f32(0.0));
        let inside_1 = if span.top_left[1] {
            vorrq_u32(positive_1, vceqq_f32(scaled_1, vdupq_n_f32(0.0)))
        } else {
            positive_1
        };
        let scaled_2 = vmulq_f32(weights_z, area_v);
        let positive_2 = vcgtq_f32(scaled_2, vdupq_n_f32(0.0));
        let inside_2 = if span.top_left[2] {
            vorrq_u32(positive_2, vceqq_f32(scaled_2, vdupq_n_f32(0.0)))
        } else {
            positive_2
        };
        let coverage = vandq_u32(vandq_u32(inside_0, inside_1), inside_2);
        let depth = vaddq_f32(
            vaddq_f32(
                vmulq_n_f32(weights_x, vertices[0].position.z),
                vmulq_n_f32(weights_y, vertices[1].position.z),
            ),
            vmulq_n_f32(weights_z, vertices[2].position.z),
        );
        let buffer_depth = vld1q_f32(depth_ptr);
        let passes_depth = vmvnq_u32(vcgeq_f32(depth, buffer_depth));
        let passing = vandq_u32(coverage, passes_depth);
        vst1q_u32(passing_lanes.as_mut_ptr(), passing);
        vst1q_f32(depth_lanes.as_mut_ptr(), depth);
        vst1q_f32(weights_x_lanes.as_mut_ptr(), weights_x);
        vst1q_f32(weights_y_lanes.as_mut_ptr(), weights_y);
        vst1q_f32(weights_z_lanes.as_mut_ptr(), weights_z);
    }
    for lane in 0..4 {
        if passing_lanes[lane] != 0 {
            let Some(index) = (span.y as usize)
                .checked_mul(framebuffer.width)
                .and_then(|row| row.checked_add((span.x + lane as i32) as usize))
            else {
                continue;
            };
            write_pixel(
                framebuffer,
                vertices,
                index,
                depth_lanes[lane],
                Vec3::new(
                    weights_x_lanes[lane],
                    weights_y_lanes[lane],
                    weights_z_lanes[lane],
                ),
                inverse_w,
                ddx_weights,
                ddy_weights,
                interpolate,
                fragment,
            );
        }
    }
}
