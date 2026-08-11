//! Directional shadow-map targets, sampling, and depth-pass shaders.

use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec2, Vec4};
use crate::mesh::MeshVertex;
use crate::pipeline::{VertexOutput, VertexStage};
use std::sync::Arc;

/// A read-only depth texture produced by a directional-light pass.
///
/// Depth keeps the renderer's OpenGL-style NDC range: `-1` is near and `+1`
/// is far. Sampling outside the map is handled as fully lit by the compare.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowMap {
    width: usize,
    height: usize,
    depth: Arc<[f32]>,
}

impl ShadowMap {
    pub const PCF_WEIGHTS: [f32; 9] = [1.0 / 9.0; 9];

    pub fn from_framebuffer(framebuffer: &Framebuffer) -> Self {
        let expected = framebuffer.width.saturating_mul(framebuffer.height);
        let mut depth = vec![1.0; expected];
        let count = expected.min(framebuffer.depth.len());
        depth[..count].copy_from_slice(&framebuffer.depth[..count]);
        Self {
            width: framebuffer.width,
            height: framebuffer.height,
            depth: Arc::from(depth),
        }
    }

    pub fn from_depth(width: usize, height: usize, depth: Vec<f32>) -> Result<Self, String> {
        let expected = width
            .checked_mul(height)
            .ok_or_else(|| "shadow-map dimensions overflow".to_string())?;
        if width == 0 || height == 0 {
            return Err("shadow-map dimensions must be non-zero".to_string());
        }
        if depth.len() != expected {
            return Err(format!(
                "shadow map has {} depth values, expected {expected}",
                depth.len()
            ));
        }
        Ok(Self {
            width,
            height,
            depth: Arc::from(depth),
        })
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns the nearest depth sample at a clamped UV coordinate.
    pub fn sample_depth(&self, uv: Vec2) -> f32 {
        let x = (uv.x.clamp(0.0, 1.0) * self.width as f32).floor() as usize;
        let y = (uv.y.clamp(0.0, 1.0) * self.height as f32).floor() as usize;
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        self.depth[y * self.width + x]
    }

    /// Compares one receiver depth with one shadow-map depth after bias.
    pub fn depth_visible(receiver_depth: f32, shadow_depth: f32, bias: f32) -> bool {
        receiver_depth - bias <= shadow_depth
    }

    /// Applies a uniform 3x3 percentage-closer filter.
    pub fn visibility_3x3(&self, uv: Vec2, receiver_depth: f32, bias: f32) -> f32 {
        if !(0.0..=1.0).contains(&uv.x)
            || !(0.0..=1.0).contains(&uv.y)
            || !receiver_depth.is_finite()
            || !(-1.0..=1.0).contains(&receiver_depth)
        {
            return 1.0;
        }
        let center_x = (uv.x * self.width as f32).floor() as isize;
        let center_y = (uv.y * self.height as f32).floor() as isize;
        let mut visibility = 0.0;
        for (index, (offset_y, offset_x)) in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        .into_iter()
        .enumerate()
        {
            let x = (center_x + offset_x).clamp(0, self.width as isize - 1) as usize;
            let y = (center_y + offset_y).clamp(0, self.height as isize - 1) as usize;
            let shadow_depth = self.depth[y * self.width + x];
            if Self::depth_visible(receiver_depth, shadow_depth, bias) {
                visibility += Self::PCF_WEIGHTS[index];
            }
        }
        visibility
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowDepthUniforms {
    model: Mat4,
    light_view_projection: Mat4,
    transform: Mat4,
}

impl ShadowDepthUniforms {
    pub fn new(model: Mat4, light_view_projection: Mat4) -> Self {
        Self {
            model,
            light_view_projection,
            transform: light_view_projection * model,
        }
    }

    pub const fn model(&self) -> Mat4 {
        self.model
    }

    pub const fn light_view_projection(&self) -> Mat4 {
        self.light_view_projection
    }

    pub const fn transform(&self) -> Mat4 {
        self.transform
    }

    pub fn set_model(&mut self, model: Mat4) {
        self.model = model;
        self.rebuild_transform();
    }

    pub fn set_light_view_projection(&mut self, light_view_projection: Mat4) {
        self.light_view_projection = light_view_projection;
        self.rebuild_transform();
    }

    fn rebuild_transform(&mut self) {
        self.transform = self.light_view_projection * self.model;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowDepthShader;

impl VertexStage<MeshVertex, ShadowDepthUniforms> for ShadowDepthShader {
    type Varyings = ();

    fn run(&self, vertex: &MeshVertex, uniforms: &ShadowDepthUniforms) -> VertexOutput<()> {
        VertexOutput::new(
            uniforms.transform()
                * Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0),
            (),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_compare_applies_bias_at_known_depths() {
        assert!(ShadowMap::depth_visible(0.51, 0.5, 0.02));
        assert!(!ShadowMap::depth_visible(0.53, 0.5, 0.02));
    }

    #[test]
    fn pcf_weights_sum_to_one() {
        let sum: f32 = ShadowMap::PCF_WEIGHTS.into_iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn light_space_transform_matches_hand_computed_point() {
        let light_view_projection = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 5.0);
        let uniforms = ShadowDepthUniforms::new(
            Mat4::translate(crate::math::Vec3::new(1.0, 0.0, 0.0)),
            light_view_projection,
        );
        let shader = ShadowDepthShader;
        let output = shader.run(
            &MeshVertex {
                position: crate::math::Vec3::new(0.0, 0.0, -1.0),
                texcoord: None,
                normal: None,
            },
            &uniforms,
        );
        assert_eq!(output.clip_position, Vec4::new(0.5, 0.0, -1.0, 1.0));
    }
}
