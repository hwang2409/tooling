//! Directional shadow-map targets, sampling, and depth-pass shaders.

use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec2, Vec4};
use crate::mesh::MeshVertex;
use crate::pipeline::{VertexOutput, VertexStage};
use std::sync::Arc;

/// Builds a directional-light view from a direction that points toward light.
pub fn directional_light_view(
    light_direction: crate::math::Vec3,
    target: crate::math::Vec3,
    distance: f32,
    up: crate::math::Vec3,
) -> Mat4 {
    Mat4::look_at(target + light_direction.normalize() * distance, target, up)
}

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

    pub fn from_framebuffer(framebuffer: &Framebuffer) -> Result<Self, String> {
        Self::from_depth(
            framebuffer.width,
            framebuffer.height,
            framebuffer.depth.clone(),
        )
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
            let x = center_x + offset_x;
            let y = center_y + offset_y;
            if x < 0 || x >= self.width as isize || y < 0 || y >= self.height as isize {
                visibility += Self::PCF_WEIGHTS[index];
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            let shadow_depth = self.depth[y * self.width + x];
            if Self::depth_visible(receiver_depth, shadow_depth, bias) {
                visibility += Self::PCF_WEIGHTS[index];
            }
        }
        visibility
    }
}

/// Complete directional shadow state used by the lighting shader.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowState {
    light_view_projection: Mat4,
    shadow_map: ShadowMap,
    constant_bias: f32,
    slope_bias: f32,
}

impl ShadowState {
    pub fn new(light_view_projection: Mat4, shadow_map: ShadowMap) -> Self {
        Self {
            light_view_projection,
            shadow_map,
            constant_bias: 0.002,
            slope_bias: 0.02,
        }
    }

    pub const fn light_view_projection(&self) -> Mat4 {
        self.light_view_projection
    }

    pub fn shadow_map(&self) -> &ShadowMap {
        &self.shadow_map
    }

    pub const fn bias(&self) -> (f32, f32) {
        (self.constant_bias, self.slope_bias)
    }

    pub fn set_bias(&mut self, constant: f32, slope: f32) {
        self.constant_bias = constant.max(0.0);
        self.slope_bias = slope.max(0.0);
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
    fn pcf_counts_outside_taps_as_lit() {
        let shadow_map = ShadowMap::from_depth(1, 1, vec![-1.0]).unwrap();
        let visibility = shadow_map.visibility_3x3(Vec2::new(0.5, 0.5), 0.0, 0.0);
        assert!((visibility - 8.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn zero_sized_framebuffer_cannot_create_shadow_map() {
        let framebuffer = Framebuffer::new(0, 0);
        let error = ShadowMap::from_framebuffer(&framebuffer).unwrap_err();
        assert!(error.contains("non-zero"));
    }

    #[test]
    fn point_toward_light_has_nearer_light_ndc_depth() {
        let direction = crate::math::Vec3::new(0.0, 0.0, 1.0);
        let target = crate::math::Vec3::ZERO;
        let light_view_projection = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 20.0)
            * directional_light_view(
                direction,
                target,
                8.0,
                crate::math::Vec3::new(0.0, 1.0, 0.0),
            );
        let point_at_target = light_view_projection * Vec4::new(target.x, target.y, target.z, 1.0);
        let point_toward_light =
            light_view_projection * Vec4::new(target.x, target.y, target.z + 1.0, 1.0);
        assert!((point_at_target.z - (-5.0 / 19.0)).abs() < 1e-6);
        assert!((point_toward_light.z - (-7.0 / 19.0)).abs() < 1e-6);
        assert!(point_toward_light.z < point_at_target.z);
    }

    #[test]
    fn shadow_depth_set_model_rebuilds_transform_immediately() {
        let mut uniforms = ShadowDepthUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY);
        let model = Mat4::translate(crate::math::Vec3::new(1.0, 2.0, 3.0));
        uniforms.set_model(model);
        assert_eq!(uniforms.transform(), Mat4::IDENTITY * model);
    }

    #[test]
    fn shadow_depth_set_light_view_projection_rebuilds_transform_immediately() {
        let mut uniforms = ShadowDepthUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY);
        let light_view_projection = Mat4::scale(crate::math::Vec3::new(2.0, 3.0, 4.0));
        uniforms.set_light_view_projection(light_view_projection);
        assert_eq!(uniforms.transform(), light_view_projection * Mat4::IDENTITY);
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
                tangent: None,
            },
            &uniforms,
        );
        assert_eq!(output.clip_position, Vec4::new(0.5, 0.0, -1.0, 1.0));
    }
}
