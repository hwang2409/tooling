//! AArch64 NEON backend for the rasterizer's coverage and depth loop.

use super::{PixelRect, ScreenVertex, rasterize_pixel, write_pixel};
use crate::fb::Framebuffer;
use crate::math::Vec3;
use crate::pipeline::Varyings;
use core::arch::aarch64::*;

pub(super) fn rasterize_prepared<V, F>(
    framebuffer: &mut Framebuffer,
    vertices: [ScreenVertex<V>; 3],
    area: f32,
    bounds: PixelRect,
    rect: PixelRect,
    mut fragment: F,
) where
    V: Varyings,
    F: FnMut(V) -> u32,
{
    let min_x = bounds.min_x.max(rect.min_x);
    let max_x = bounds.max_x.min(rect.max_x);
    let min_y = bounds.min_y.max(rect.min_y);
    let max_y = bounds.max_y.min(rect.max_y);
    let top_left = [
        super::is_top_left(vertices[1].position, vertices[2].position),
        super::is_top_left(vertices[2].position, vertices[0].position),
        super::is_top_left(vertices[0].position, vertices[1].position),
    ];

    for y in min_y..=max_y {
        let mut x = min_x;
        while max_x - x + 1 >= 4 {
            unsafe {
                // SAFETY: the prepared rectangle guarantees four valid lanes
                // in both framebuffer dimensions and all SIMD inputs are f32.
                rasterize_quad(framebuffer, &vertices, area, x, y, top_left, &mut fragment);
            }
            x = x.saturating_add(4);
        }
        while x <= max_x {
            let Some((depth, weights)) =
                rasterize_pixel(framebuffer, &vertices, area, x, y, top_left)
            else {
                x = x.saturating_add(1);
                continue;
            };
            write_pixel(framebuffer, &vertices, x, y, depth, weights, &mut fragment);
            x = x.saturating_add(1);
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn rasterize_quad<V, F>(
    framebuffer: &mut Framebuffer,
    vertices: &[ScreenVertex<V>; 3],
    area: f32,
    x: i32,
    y: i32,
    top_left: [bool; 3],
    fragment: &mut F,
) where
    V: Varyings,
    F: FnMut(V) -> u32,
{
    let xs = [
        x as f32 + 0.5,
        (x + 1) as f32 + 0.5,
        (x + 2) as f32 + 0.5,
        (x + 3) as f32 + 0.5,
    ];
    let pixel_index = y as usize * framebuffer.width + x as usize;
    let mut passing_lanes = [0_u32; 4];
    let mut depth_lanes = [0.0_f32; 4];
    let mut weights_x_lanes = [0.0_f32; 4];
    let mut weights_y_lanes = [0.0_f32; 4];
    let mut weights_z_lanes = [0.0_f32; 4];
    unsafe {
        // SAFETY: xs and the framebuffer span contain four valid lanes. The
        // destination arrays contain four writable lanes. NEON uses no fused
        // operations here, matching the scalar multiply-then-add order.
        let px = vld1q_f32(xs.as_ptr());
        let py = vdupq_n_f32(y as f32 + 0.5);
        let area_v = vdupq_n_f32(area);
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
        let inside_0 = if top_left[0] {
            vorrq_u32(positive_0, vceqq_f32(scaled_0, vdupq_n_f32(0.0)))
        } else {
            positive_0
        };
        let scaled_1 = vmulq_f32(weights_y, area_v);
        let positive_1 = vcgtq_f32(scaled_1, vdupq_n_f32(0.0));
        let inside_1 = if top_left[1] {
            vorrq_u32(positive_1, vceqq_f32(scaled_1, vdupq_n_f32(0.0)))
        } else {
            positive_1
        };
        let scaled_2 = vmulq_f32(weights_z, area_v);
        let positive_2 = vcgtq_f32(scaled_2, vdupq_n_f32(0.0));
        let inside_2 = if top_left[2] {
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
        let buffer_depth = vld1q_f32(framebuffer.depth.as_ptr().add(pixel_index));
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
            write_pixel(
                framebuffer,
                vertices,
                x + lane as i32,
                y,
                depth_lanes[lane],
                Vec3::new(
                    weights_x_lanes[lane],
                    weights_y_lanes[lane],
                    weights_z_lanes[lane],
                ),
                fragment,
            );
        }
    }
}
