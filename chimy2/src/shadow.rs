//! Directional shadow-map targets, sampling, and depth-pass shaders.
//!
//! Single-map directional shadows support opt-in percentage-closer soft
//! shadows (PCSS), following Fernando, "Percentage-Closer Soft Shadows",
//! NVIDIA, 2005. PCSS is not applied to cascaded maps or cube shadows.
//!
//! The directional adaptation measures blocker and receiver distances from
//! the light projection's near plane. This is an explicit light-depth origin,
//! and keeps the denominator translation-invariant for an orthographic light.

use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::pipeline::{Instance, InstanceUniforms, Pipeline, Varyings, VertexOutput, VertexStage};
use crate::raster::{DepthVaryings, ScreenVertex, perspective_correct_weights};
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
    const PCSS_BLOCKER_SAMPLES: [Vec2; 16] = [
        Vec2::new(-0.942, -0.399),
        Vec2::new(-0.751, 0.271),
        Vec2::new(-0.527, -0.844),
        Vec2::new(-0.305, 0.689),
        Vec2::new(-0.088, -0.176),
        Vec2::new(0.097, 0.932),
        Vec2::new(0.244, -0.604),
        Vec2::new(0.416, 0.153),
        Vec2::new(0.571, -0.925),
        Vec2::new(0.694, 0.554),
        Vec2::new(0.827, -0.196),
        Vec2::new(0.932, 0.802),
        Vec2::new(-0.881, 0.735),
        Vec2::new(-0.637, -0.612),
        Vec2::new(-0.216, 0.382),
        Vec2::new(0.011, -0.947),
    ];
    const PCSS_FILTER_SAMPLES: [Vec2; 16] = [
        Vec2::new(-0.942, -0.399),
        Vec2::new(-0.751, 0.271),
        Vec2::new(-0.527, -0.844),
        Vec2::new(-0.305, 0.689),
        Vec2::new(-0.088, -0.176),
        Vec2::new(0.097, 0.932),
        Vec2::new(0.244, -0.604),
        Vec2::new(0.416, 0.153),
        Vec2::new(0.571, -0.925),
        Vec2::new(0.694, 0.554),
        Vec2::new(0.827, -0.196),
        Vec2::new(0.932, 0.802),
        Vec2::new(-0.881, 0.735),
        Vec2::new(-0.637, -0.612),
        Vec2::new(-0.216, 0.382),
        Vec2::new(0.011, -0.947),
    ];

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

    /// Applies PCSS to a single directional map.
    ///
    /// The map stores NDC depth, but blocker and receiver distances are
    /// reconstructed as linear light-space units. For an orthographic light
    /// matrix, the row-2 scale converts NDC depth to distance from the light.
    /// The light size is in world units. Row-norm extents convert it to UV;
    /// this uses matrix rows because [`Mat4`] stores columns.
    pub fn visibility_pcss(
        &self,
        uv: Vec2,
        receiver_depth: f32,
        light_view_projection: Mat4,
        parameters: PcssShadowParameters,
    ) -> f32 {
        let PcssShadowParameters {
            constant_bias,
            slope_bias,
            normal_dot_light,
            light_size,
            light_depth_origin,
        } = parameters;
        if !(0.0..=1.0).contains(&uv.x)
            || !(0.0..=1.0).contains(&uv.y)
            || !receiver_depth.is_finite()
            || !(-1.0..=1.0).contains(&receiver_depth)
            || !light_size.is_finite()
            || light_size <= 0.0
        {
            let bias = constant_bias.max(slope_bias * (1.0 - normal_dot_light));
            return self.visibility_3x3(uv, receiver_depth, bias);
        }
        let Some(receiver_linear_depth) =
            linear_light_depth(light_view_projection, receiver_depth, light_depth_origin)
        else {
            return 1.0;
        };
        let row_x = matrix_row_scale(light_view_projection, 0);
        let row_y = matrix_row_scale(light_view_projection, 1);
        let row_z = matrix_row_scale(light_view_projection, 2);
        if !row_x.is_finite()
            || !row_y.is_finite()
            || !row_z.is_finite()
            || row_x <= f32::EPSILON
            || row_y <= f32::EPSILON
            || row_z <= f32::EPSILON
        {
            return 1.0;
        }

        // A world-unit light radius projects to half an NDC span, then to UV.
        // This is the world-units-to-UV conversion used by both search stages.
        let search_radius_uv = pcss_search_radius_uv(light_size, light_view_projection);
        let base_bias = constant_bias.max(slope_bias * (1.0 - normal_dot_light));
        let receiver_bias = base_bias.max(0.0) / row_z;
        let mut blocker_depth_sum = 0.0;
        let mut blocker_count = 0_u32;
        for offset in Self::PCSS_BLOCKER_SAMPLES {
            let sample_uv = uv + offset * search_radius_uv;
            let sample_depth = self.sample_depth_clamped(sample_uv);
            let Some(sample_linear_depth) =
                linear_light_depth(light_view_projection, sample_depth, light_depth_origin)
            else {
                continue;
            };
            if sample_linear_depth < receiver_linear_depth - receiver_bias {
                blocker_depth_sum += sample_linear_depth;
                blocker_count += 1;
            }
        }
        // No blocker means no penumbra. This early-out also prevents empty
        // regions from becoming artificial blockers at depth zero.
        if blocker_count == 0 {
            return 1.0;
        }
        let average_blocker_depth = blocker_depth_sum / blocker_count as f32;
        if !average_blocker_depth.is_finite() {
            return 1.0;
        }

        // Fernando's PCSS estimate is linear in light-space depth. NDC depth
        // is nonlinear for perspective projections, so it is not used here.
        let penumbra_width = if average_blocker_depth > 0.0 {
            pcss_penumbra_width(receiver_linear_depth, average_blocker_depth, light_size)
        } else {
            f32::MAX
        };
        let raw_radius_x = (penumbra_width * row_x * self.width as f32 * 0.5).max(2.0);
        let raw_radius_y = (penumbra_width * row_y * self.height as f32 * 0.5).max(2.0);
        let center_x = (uv.x * self.width as f32).floor() as isize;
        let center_y = (uv.y * self.height as f32).floor() as isize;
        let radius_x =
            raw_radius_x.min(center_x.min(self.width as isize - 1 - center_x).max(0) as f32);
        let radius_y =
            raw_radius_y.min(center_y.min(self.height as isize - 1 - center_y).max(0) as f32);
        let filter_radius_uv =
            Vec2::new(radius_x / self.width as f32, radius_y / self.height as f32);
        // The slope term grows with the filter radius. A wide filter needs a
        // wider world-space receiver offset to avoid reintroducing acne.
        let kernel_scale = radius_x.max(radius_y).max(1.0);
        let scaled_bias =
            pcss_filter_bias(constant_bias, slope_bias, normal_dot_light, kernel_scale);
        let mut visibility = 0.0;
        for offset in Self::PCSS_FILTER_SAMPLES {
            let sample_uv = uv + offset * filter_radius_uv;
            let sample_depth = self.sample_depth_clamped(sample_uv);
            if Self::depth_visible(receiver_depth, sample_depth, scaled_bias) {
                visibility += 1.0;
            }
        }
        visibility / Self::PCSS_FILTER_SAMPLES.len() as f32
    }

    fn sample_depth_clamped(&self, uv: Vec2) -> f32 {
        let x = (uv.x.clamp(0.0, 1.0) * self.width as f32).floor() as usize;
        let y = (uv.y.clamp(0.0, 1.0) * self.height as f32).floor() as usize;
        self.depth[y.min(self.height - 1) * self.width + x.min(self.width - 1)]
    }
}

/// Runtime parameters for one single-map PCSS visibility query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcssShadowParameters {
    pub constant_bias: f32,
    pub slope_bias: f32,
    pub normal_dot_light: f32,
    pub light_size: f32,
    /// Distance origin in light view units. Zero means the light near plane.
    pub light_depth_origin: f32,
}

/// Complete directional shadow state used by the lighting shader.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowState {
    light_view_projection: Mat4,
    shadow_map: ShadowMap,
    cascades: Option<CascadeShadowState>,
    constant_bias: f32,
    slope_bias: f32,
    light_size: f32,
    light_depth_origin: f32,
}

impl ShadowState {
    pub fn new(light_view_projection: Mat4, shadow_map: ShadowMap) -> Self {
        Self {
            light_view_projection,
            shadow_map,
            cascades: None,
            constant_bias: 0.002,
            slope_bias: 0.02,
            light_size: 0.0,
            light_depth_origin: 0.0,
        }
    }

    /// Creates a directional shadow state backed by cascaded maps.
    pub fn from_cascades(cascades: CascadeShadowState) -> Self {
        Self {
            light_view_projection: cascades.light_view_projections[0],
            shadow_map: cascades.shadow_maps[0]
                .clone()
                .expect("a cascaded state has at least two maps"),
            cascades: Some(cascades),
            constant_bias: 0.002,
            slope_bias: 0.02,
            light_size: 0.0,
            light_depth_origin: 0.0,
        }
    }

    pub const fn light_view_projection(&self) -> Mat4 {
        self.light_view_projection
    }

    pub fn shadow_map(&self) -> &ShadowMap {
        &self.shadow_map
    }

    pub fn cascades(&self) -> Option<&CascadeShadowState> {
        self.cascades.as_ref()
    }

    pub const fn is_cascaded(&self) -> bool {
        self.cascades.is_some()
    }

    pub const fn bias(&self) -> (f32, f32) {
        (self.constant_bias, self.slope_bias)
    }

    /// Returns the world-unit directional emitter radius used by PCSS.
    /// Zero keeps the exact legacy 3x3 PCF path. Cascaded maps ignore it.
    pub const fn light_size(&self) -> f32 {
        self.light_size
    }

    /// Returns the explicit light-view distance origin used by PCSS.
    pub const fn light_depth_origin(&self) -> f32 {
        self.light_depth_origin
    }

    pub fn set_bias(&mut self, constant: f32, slope: f32) {
        self.constant_bias = sanitize_bias(constant);
        self.slope_bias = sanitize_bias(slope);
        if let Some(cascades) = self.cascades.as_mut() {
            cascades.set_bias(constant, slope);
        }
    }

    /// Enables PCSS for a single directional map. Invalid or negative sizes
    /// sanitize to zero. Cascaded maps retain their existing PCF behavior.
    pub fn set_light_size(&mut self, light_size: f32) {
        self.light_size = sanitize_light_size(light_size);
    }

    /// Sets the PCSS distance origin in light-view units. Zero is the light
    /// projection near plane and is translation-invariant for an orthographic
    /// directional projection.
    pub fn set_light_depth_origin(&mut self, origin: f32) {
        self.light_depth_origin = sanitize_light_depth_origin(origin);
    }

    pub(crate) fn visibility(&self, uv: Vec2, receiver_depth: f32, normal_dot_light: f32) -> f32 {
        let (constant_bias, slope_bias) = self.bias();
        self.shadow_map.visibility_pcss(
            uv,
            receiver_depth,
            self.light_view_projection,
            PcssShadowParameters {
                constant_bias,
                slope_bias,
                normal_dot_light,
                light_size: self.light_size,
                light_depth_origin: self.light_depth_origin,
            },
        )
    }
}

fn matrix_row_scale(matrix: Mat4, row: usize) -> f32 {
    let x = matrix.data[row];
    let y = matrix.data[row + 4];
    let z = matrix.data[row + 8];
    (x * x + y * y + z * z).sqrt()
}

fn linear_light_depth(matrix: Mat4, ndc_depth: f32, light_depth_origin: f32) -> Option<f32> {
    let scale = matrix_row_scale(matrix, 2);
    if !scale.is_finite()
        || scale <= f32::EPSILON
        || !ndc_depth.is_finite()
        || !light_depth_origin.is_finite()
    {
        return None;
    }
    // For an orthographic projection, (ndc + 1) / scale is view-space
    // distance from the near plane. Do not use the combined matrix
    // translation: it changes when the scene and light move together.
    let depth = (ndc_depth + 1.0) / scale + light_depth_origin;
    depth.is_finite().then_some(depth)
}

fn pcss_search_radius_uv(light_size: f32, matrix: Mat4) -> Vec2 {
    let row_x = matrix_row_scale(matrix, 0);
    let row_y = matrix_row_scale(matrix, 1);
    Vec2::new(light_size * row_x * 0.5, light_size * row_y * 0.5)
}

fn pcss_penumbra_width(receiver_depth: f32, blocker_depth: f32, light_size: f32) -> f32 {
    if !receiver_depth.is_finite() || !blocker_depth.is_finite() || blocker_depth <= 0.0 {
        return 0.0;
    }
    ((receiver_depth - blocker_depth) * light_size / blocker_depth).max(0.0)
}

fn pcss_filter_bias(
    constant_bias: f32,
    slope_bias: f32,
    normal_dot_light: f32,
    kernel_scale: f32,
) -> f32 {
    constant_bias
        .max(slope_bias.max(0.0) * (1.0 - normal_dot_light.clamp(0.0, 1.0)) * kernel_scale.max(1.0))
}

/// CSM implementation is kept in the focused `csm` module.
#[cfg(test)]
pub(crate) use crate::csm::fit_cascade_light_projection_with_casters;
pub use crate::csm::{
    CascadeShadowConfig, CascadeShadowState, MAX_CASCADES, cascade_index,
    fit_cascade_light_projection, frustum_slice_corners, practical_split_depths,
    render_cascade_shadow_maps, render_cascade_shadow_maps_instanced,
    render_cascade_shadow_maps_instanced_with_config, render_cascade_shadow_maps_with_config,
    snap_ortho_origin,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowDepthUniforms {
    model: Mat4,
    light_view_projection: Mat4,
    transform: Mat4,
    light_position: Vec3,
    far_plane: f32,
    linear_depth: bool,
}

impl ShadowDepthUniforms {
    pub fn new(model: Mat4, light_view_projection: Mat4) -> Self {
        Self {
            model,
            light_view_projection,
            transform: light_view_projection * model,
            light_position: Vec3::ZERO,
            far_plane: 1.0,
            linear_depth: false,
        }
    }

    pub fn new_cube(
        model: Mat4,
        light_view_projection: Mat4,
        light_position: Vec3,
        far_plane: f32,
    ) -> Self {
        let mut uniforms = Self::new(model, light_view_projection);
        uniforms.light_position = sanitize_position(light_position);
        uniforms.far_plane = sanitize_far_plane(far_plane, sanitize_near_plane(0.01));
        uniforms.linear_depth = true;
        uniforms
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

impl InstanceUniforms for ShadowDepthUniforms {
    fn set_instance_model(&mut self, model: Mat4) {
        self.set_model(model);
    }

    fn apply_instance_tint(&mut self, _: Vec4) {}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ShadowDepthShader;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowDepthVaryings {
    light_vector_over_far: Vec3,
    linear_depth: bool,
}

impl Varyings for ShadowDepthVaryings {
    fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
        Self {
            light_vector_over_far: a.light_vector_over_far * weights.x
                + b.light_vector_over_far * weights.y
                + c.light_vector_over_far * weights.z,
            linear_depth: a.linear_depth,
        }
    }
}

impl DepthVaryings for ShadowDepthVaryings {
    fn depth(vertices: &[ScreenVertex<Self>; 3], weights: Vec3, inverse_w: Vec3) -> f32 {
        if !vertices[0].varyings.linear_depth {
            return vertices[0].position.z * weights.x
                + vertices[1].position.z * weights.y
                + vertices[2].position.z * weights.z;
        }
        let weights = perspective_correct_weights(weights, inverse_w);
        let vector = vertices[0].varyings.light_vector_over_far * weights.x
            + vertices[1].varyings.light_vector_over_far * weights.y
            + vertices[2].varyings.light_vector_over_far * weights.z;
        normalized_radial_distance(vector)
    }
}

impl VertexStage<MeshVertex, ShadowDepthUniforms> for ShadowDepthShader {
    type Varyings = ShadowDepthVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &ShadowDepthUniforms,
    ) -> VertexOutput<ShadowDepthVaryings> {
        let local_position = Vec4::new(
            vertex.position().x,
            vertex.position().y,
            vertex.position().z,
            1.0,
        );
        let world_position = uniforms.model * local_position;
        VertexOutput::new(
            uniforms.transform() * local_position,
            ShadowDepthVaryings {
                light_vector_over_far: if uniforms.linear_depth {
                    Vec3::new(
                        (world_position.x - uniforms.light_position.x) / uniforms.far_plane,
                        (world_position.y - uniforms.light_position.y) / uniforms.far_plane,
                        (world_position.z - uniforms.light_position.z) / uniforms.far_plane,
                    )
                } else {
                    Vec3::ZERO
                },
                linear_depth: uniforms.linear_depth,
            },
        )
    }
}

/// The six faces use the renderer's right-handed, camera-forward `-Z` view.
/// The Y faces use Z as their up vector to keep the cube orientation stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeShadowFace {
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ,
}

impl CubeShadowFace {
    pub const ALL: [Self; 6] = [
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveY,
        Self::NegativeY,
        Self::PositiveZ,
        Self::NegativeZ,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::PositiveX => 0,
            Self::NegativeX => 1,
            Self::PositiveY => 2,
            Self::NegativeY => 3,
            Self::PositiveZ => 4,
            Self::NegativeZ => 5,
        }
    }

    pub const fn direction(self) -> Vec3 {
        match self {
            Self::PositiveX => Vec3::new(1.0, 0.0, 0.0),
            Self::NegativeX => Vec3::new(-1.0, 0.0, 0.0),
            Self::PositiveY => Vec3::new(0.0, 1.0, 0.0),
            Self::NegativeY => Vec3::new(0.0, -1.0, 0.0),
            Self::PositiveZ => Vec3::new(0.0, 0.0, 1.0),
            Self::NegativeZ => Vec3::new(0.0, 0.0, -1.0),
        }
    }

    pub const fn up(self) -> Vec3 {
        match self {
            Self::PositiveX | Self::NegativeX | Self::PositiveZ | Self::NegativeZ => {
                Vec3::new(0.0, -1.0, 0.0)
            }
            Self::PositiveY => Vec3::new(0.0, 0.0, 1.0),
            Self::NegativeY => Vec3::new(0.0, 0.0, -1.0),
        }
    }
}

/// Returns the six 90-degree view-projection matrices in [`CubeShadowFace`]
/// order. Near and far are sanitized at the uniform boundary.
pub fn cube_face_view_projections(
    light_position: Vec3,
    near_plane: f32,
    far_plane: f32,
) -> [Mat4; 6] {
    let light_position = sanitize_position(light_position);
    let near_plane = sanitize_near_plane(near_plane);
    let far_plane = sanitize_far_plane(far_plane, near_plane);
    let projection = Mat4::perspective(std::f32::consts::FRAC_PI_2, 1.0, near_plane, far_plane);
    std::array::from_fn(|index| {
        let face = CubeShadowFace::ALL[index];
        projection * Mat4::look_at(light_position, light_position + face.direction(), face.up())
    })
}

/// A six-face depth target with normalized linear light distance.
///
/// Directional maps keep NDC depth because their projection is orthographic.
/// Cube maps use perspective faces, so linear distance gives uniform precision
/// and one face-independent comparison for every direction.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeShadowMap {
    size: usize,
    near_plane: f32,
    far_plane: f32,
    faces: [Arc<[f32]>; 6],
}

impl CubeShadowMap {
    pub const PCF_WEIGHTS: [f32; 9] = [1.0 / 9.0; 9];

    /// Creates a map from six normalized linear-distance faces.
    pub fn from_depth(
        size: usize,
        near_plane: f32,
        far_plane: f32,
        faces: [Vec<f32>; 6],
    ) -> Result<Self, String> {
        if size == 0 {
            return Err("cube shadow-map size must be non-zero".to_string());
        }
        let expected = size
            .checked_mul(size)
            .ok_or_else(|| "cube shadow-map dimensions overflow".to_string())?;
        let near_plane = sanitize_near_plane(near_plane);
        let far_plane = sanitize_far_plane(far_plane, near_plane);
        let mut stored = std::array::from_fn(|_| Arc::<[f32]>::from(Vec::<f32>::new()));
        for (index, face) in faces.into_iter().enumerate() {
            if face.len() != expected {
                return Err(format!(
                    "cube shadow face {index} has {} depth values, expected {expected}",
                    face.len()
                ));
            }
            stored[index] = Arc::from(face.into_iter().map(sanitize_distance).collect::<Vec<_>>());
        }
        Ok(Self {
            size,
            near_plane,
            far_plane,
            faces: stored,
        })
    }

    /// Creates a map from the depth-only pipeline's normalized radial faces.
    pub fn from_framebuffers(
        faces: [Framebuffer; 6],
        near_plane: f32,
        far_plane: f32,
    ) -> Result<Self, String> {
        let size = faces[0].width;
        if size == 0 || faces[0].height != size {
            return Err("cube shadow faces must be non-zero and square".to_string());
        }
        let near_plane = sanitize_near_plane(near_plane);
        let far_plane = sanitize_far_plane(far_plane, near_plane);
        let mut converted = std::array::from_fn(|_| Vec::new());
        for (index, face) in faces.into_iter().enumerate() {
            if face.width != size || face.height != size {
                return Err("cube shadow faces must have matching dimensions".to_string());
            }
            converted[index] = face.depth.into_iter().map(sanitize_distance).collect();
        }
        Self::from_depth(size, near_plane, far_plane, converted)
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    pub const fn near_plane(&self) -> f32 {
        self.near_plane
    }

    pub const fn far_plane(&self) -> f32 {
        self.far_plane
    }

    pub fn sample_depth(&self, face: CubeShadowFace, uv: Vec2) -> f32 {
        let coordinate = |value: f32| (value.clamp(0.0, 1.0) * self.size as f32).floor() as usize;
        let x = coordinate(uv.x).min(self.size - 1);
        let y = coordinate(uv.y).min(self.size - 1);
        self.faces[face.index()][y * self.size + x]
    }

    /// Applies a clamped 3x3 PCF filter. Taps never wrap to another face.
    pub fn visibility_3x3(
        &self,
        face: CubeShadowFace,
        uv: Vec2,
        receiver_distance: f32,
        bias: f32,
    ) -> f32 {
        if !(0.0..=1.0).contains(&uv.x)
            || !(0.0..=1.0).contains(&uv.y)
            || !receiver_distance.is_finite()
        {
            return 1.0;
        }
        let center_x = (uv.x * self.size as f32).floor() as isize;
        let center_y = (uv.y * self.size as f32).floor() as isize;
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
            let x = (center_x + offset_x).clamp(0, self.size as isize - 1) as usize;
            let y = (center_y + offset_y).clamp(0, self.size as isize - 1) as usize;
            if receiver_distance - bias <= self.faces[face.index()][y * self.size + x] {
                visibility += Self::PCF_WEIGHTS[index];
            }
        }
        visibility
    }
}

/// Complete point-light cube-shadow state used by both shader families.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeShadowState {
    light_position: Vec3,
    face_view_projections: [Mat4; 6],
    shadow_map: CubeShadowMap,
    constant_bias: f32,
    slope_bias: f32,
}

impl CubeShadowState {
    pub fn new(light_position: Vec3, shadow_map: CubeShadowMap) -> Self {
        let mut state = Self {
            light_position: sanitize_position(light_position),
            face_view_projections: [Mat4::IDENTITY; 6],
            shadow_map,
            constant_bias: 0.002,
            slope_bias: 0.02,
        };
        state.rebuild_face_view_projections();
        state
    }

    pub const fn light_position(&self) -> Vec3 {
        self.light_position
    }

    pub const fn face_view_projections(&self) -> [Mat4; 6] {
        self.face_view_projections
    }

    pub fn shadow_map(&self) -> &CubeShadowMap {
        &self.shadow_map
    }

    pub const fn bias(&self) -> (f32, f32) {
        (self.constant_bias, self.slope_bias)
    }

    /// Replaces the captured light position and all six faces together.
    pub fn replace_capture(&mut self, light_position: Vec3, shadow_map: CubeShadowMap) {
        self.light_position = sanitize_position(light_position);
        self.shadow_map = shadow_map;
        self.rebuild_face_view_projections();
    }

    pub fn set_bias(&mut self, constant: f32, slope: f32) {
        self.constant_bias = sanitize_bias(constant);
        self.slope_bias = sanitize_bias(slope);
    }

    pub fn visibility(&self, point: Vec3, normal: Vec3) -> f32 {
        let to_point = point - self.light_position;
        let distance = to_point.length();
        if !distance.is_finite() || distance <= 0.0 {
            return 1.0;
        }
        let face = cube_face_for_direction(to_point);
        let clip =
            self.face_view_projections[face.index()] * Vec4::new(point.x, point.y, point.z, 1.0);
        if clip.w <= 0.0 || !clip.w.is_finite() {
            return 1.0;
        }
        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        let uv = Vec2::new((ndc.x + 1.0) * 0.5, (1.0 - ndc.y) * 0.5);
        let receiver_distance = distance / self.shadow_map.far_plane;
        if !ndc.x.is_finite()
            || !ndc.y.is_finite()
            || !receiver_distance.is_finite()
            || receiver_distance > 1.0
        {
            return 1.0;
        }
        let light_direction = (-to_point).normalize();
        let normal_dot_light = normal.normalize().dot(light_direction).clamp(0.0, 1.0);
        let bias = self
            .constant_bias
            .max(self.slope_bias * (1.0 - normal_dot_light));
        self.shadow_map
            .visibility_3x3(face, uv, receiver_distance, bias)
    }

    fn rebuild_face_view_projections(&mut self) {
        self.face_view_projections = cube_face_view_projections(
            self.light_position,
            self.shadow_map.near_plane,
            self.shadow_map.far_plane,
        );
    }
}

/// Renders all six faces through [`Pipeline::draw_mesh_depth`].
pub fn render_cube_shadow_map(
    light_position: Vec3,
    near_plane: f32,
    far_plane: f32,
    size: usize,
    meshes: &[(&Mesh, Mat4)],
) -> Result<CubeShadowMap, String> {
    if size == 0 {
        return Err("cube shadow-map size must be non-zero".to_string());
    }
    let near_plane = sanitize_near_plane(near_plane);
    let far_plane = sanitize_far_plane(far_plane, near_plane);
    let matrices = cube_face_view_projections(light_position, near_plane, far_plane);
    let mut faces = std::array::from_fn(|_| Framebuffer::new(size, size));
    let mut pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    for (index, face) in faces.iter_mut().enumerate() {
        face.clear(0);
        for &(mesh, model) in meshes {
            pipeline.draw_mesh_depth_with_varyings(
                face,
                mesh,
                &ShadowDepthUniforms::new_cube(model, matrices[index], light_position, far_plane),
            );
        }
    }
    CubeShadowMap::from_framebuffers(faces, near_plane, far_plane)
}

/// Renders all six faces for mesh groups that share one material and use
/// instances.
pub fn render_cube_shadow_map_instanced(
    light_position: Vec3,
    near_plane: f32,
    far_plane: f32,
    size: usize,
    meshes: &[(&Mesh, &[Instance])],
) -> Result<CubeShadowMap, String> {
    if size == 0 {
        return Err("cube shadow-map size must be non-zero".to_string());
    }
    let near_plane = sanitize_near_plane(near_plane);
    let far_plane = sanitize_far_plane(far_plane, near_plane);
    let matrices = cube_face_view_projections(light_position, near_plane, far_plane);
    let mut faces = std::array::from_fn(|_| Framebuffer::new(size, size));
    let mut pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    for (index, face) in faces.iter_mut().enumerate() {
        face.clear(0);
        for &(mesh, instances) in meshes {
            pipeline.draw_mesh_depth_instanced_with_varyings(
                face,
                mesh,
                &ShadowDepthUniforms::new_cube(
                    Mat4::IDENTITY,
                    matrices[index],
                    light_position,
                    far_plane,
                ),
                instances,
            );
        }
    }
    CubeShadowMap::from_framebuffers(faces, near_plane, far_plane)
}

fn cube_face_for_direction(direction: Vec3) -> CubeShadowFace {
    let absolute = Vec3::new(direction.x.abs(), direction.y.abs(), direction.z.abs());
    if absolute.x >= absolute.y && absolute.x >= absolute.z {
        if direction.x >= 0.0 {
            CubeShadowFace::PositiveX
        } else {
            CubeShadowFace::NegativeX
        }
    } else if absolute.y >= absolute.z {
        if direction.y >= 0.0 {
            CubeShadowFace::PositiveY
        } else {
            CubeShadowFace::NegativeY
        }
    } else if direction.z >= 0.0 {
        CubeShadowFace::PositiveZ
    } else {
        CubeShadowFace::NegativeZ
    }
}

pub(crate) fn sanitize_position(position: Vec3) -> Vec3 {
    Vec3::new(
        finite_or_zero(position.x),
        finite_or_zero(position.y),
        finite_or_zero(position.z),
    )
}

fn normalized_radial_distance(light_vector_over_far: Vec3) -> f32 {
    // The vector is normalized by the far plane. Its length is the ray-length
    // factor that converts face-axis depth into radial light distance.
    let ray_length_factor = light_vector_over_far.length();
    ray_length_factor.clamp(0.0, 1.0)
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

pub(crate) fn sanitize_near_plane(value: f32) -> f32 {
    const MIN_NEAR_PLANE: f32 = 0.0001;
    const MAX_NEAR_PLANE: f32 = 10_000.0;
    if value.is_finite() && value > 0.0 {
        value.clamp(MIN_NEAR_PLANE, MAX_NEAR_PLANE)
    } else {
        0.01_f32.clamp(MIN_NEAR_PLANE, MAX_NEAR_PLANE)
    }
}

pub(crate) fn sanitize_far_plane(value: f32, near_plane: f32) -> f32 {
    const MAX_FAR_PLANE: f32 = 100_000.0;
    if value.is_finite() && value > near_plane {
        value.min(MAX_FAR_PLANE)
    } else {
        (near_plane + near_plane.max(1.0) * 0.01).min(MAX_FAR_PLANE)
    }
}

fn sanitize_distance(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

pub(crate) fn sanitize_bias(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

pub(crate) fn sanitize_light_size(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

pub(crate) fn sanitize_light_depth_origin(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::math::Quat;

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }

    fn test_camera() -> Camera {
        Camera::new(
            Vec3::new(0.0, 1.0, 5.0),
            Quat::IDENTITY,
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            100.0,
        )
    }

    #[test]
    fn practical_split_formula_matches_hand_computation() {
        let splits = practical_split_depths(0.1, 100.0, 3, 0.5);
        assert_close(splits[0], 17.2);
        assert_close(splits[1], 38.35);
        assert_close(splits[2], 100.0);
        let asymmetric = practical_split_depths(0.1, 100.0, 3, 0.25);
        assert_close(asymmetric[0], 25.3);
    }

    #[test]
    fn cascade_config_sanitizes_mutators_immediately() {
        let mut config = CascadeShadowConfig::default();
        config.set_cascade_count(1);
        assert_eq!(config.cascade_count(), 2);
        config.set_cascade_count(99);
        assert_eq!(config.cascade_count(), MAX_CASCADES);
        config.set_lambda(-1.0);
        assert_eq!(config.lambda(), 0.0);
        config.set_lambda(2.0);
        assert_eq!(config.lambda(), 1.0);
        config.set_lambda(f32::NAN);
        assert_eq!(config.lambda(), 0.5);
    }

    #[test]
    fn practical_splits_sanitize_degenerate_camera_planes() {
        let splits = practical_split_depths(f32::NAN, -1.0, 3, 0.5);
        assert!(splits[..3].iter().all(|value| value.is_finite()));
        assert!(splits[0] > 0.0);
        assert!(splits[2] > splits[0]);
    }

    #[test]
    fn cascade_selection_is_deterministic_at_boundaries() {
        let splits = practical_split_depths(0.1, 100.0, 3, 0.5);
        assert_eq!(cascade_index(splits[0] - 1e-4, &splits[..3]), 0);
        assert_eq!(cascade_index(splits[0], &splits[..3]), 0);
        assert_eq!(cascade_index(splits[0] + 1e-4, &splits[..3]), 1);
        assert_eq!(cascade_index(splits[1], &splits[..3]), 1);
        assert_eq!(cascade_index(splits[1] + 1e-4, &splits[..3]), 2);
    }

    #[test]
    fn frustum_slice_fit_contains_corners_and_is_tight() {
        let camera = test_camera();
        let corners = frustum_slice_corners(camera, 1.0, 10.0);
        let projection =
            fit_cascade_light_projection(camera, Vec3::new(0.6, 1.0, 0.4), 1.0, 10.0, 128);
        assert!(corners.iter().all(|corner| {
            let clip = projection * Vec4::new(corner.x, corner.y, corner.z, 1.0);
            let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
            ndc.x.abs() <= 1.0 + 1e-5 && ndc.y.abs() <= 1.0 + 1e-5 && ndc.z.abs() <= 1.0 + 1e-5
        }));
        let max_xy = corners
            .iter()
            .map(|corner| {
                let clip = projection * Vec4::new(corner.x, corner.y, corner.z, 1.0);
                (clip.x / clip.w).abs().max((clip.y / clip.w).abs())
            })
            .fold(0.0, f32::max);
        assert!(max_xy > 0.98);

        let whole = frustum_slice_corners(camera, camera.near, camera.far);
        let whole_width = whole
            .iter()
            .map(|corner| corner.x)
            .fold(f32::NEG_INFINITY, f32::max)
            - whole
                .iter()
                .map(|corner| corner.x)
                .fold(f32::INFINITY, f32::min);
        let slice_width = corners
            .iter()
            .map(|corner| corner.x)
            .fold(f32::NEG_INFINITY, f32::max)
            - corners
                .iter()
                .map(|corner| corner.x)
                .fold(f32::INFINITY, f32::min);
        assert!(slice_width < whole_width * 0.2);
    }

    #[test]
    fn texel_snapping_moves_in_whole_texel_steps() {
        let texel = 0.25;
        let first = snap_ortho_origin(1.01, texel);
        let second = snap_ortho_origin(1.10, texel);
        let third = snap_ortho_origin(1.26, texel);
        assert_eq!(first, second);
        assert_eq!(third - second, texel);
    }

    #[test]
    fn caster_relevant_depth_bounds_keep_a_tall_caster_inside() {
        let camera = test_camera();
        let caster = Vec3::new(0.0, 50.0, -3.0);
        let projection = fit_cascade_light_projection_with_casters(
            camera,
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
            2.0,
            64,
            &[caster],
        );
        let clip = projection * Vec4::new(caster.x, caster.y, caster.z, 1.0);
        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        assert!(ndc.z.abs() <= 1.0 + 1e-5, "caster clipped at {ndc:?}");
    }

    #[test]
    fn production_snap_keeps_static_visibility_grid_for_subtexel_camera_motion() {
        let camera = test_camera();
        let ground = crate::demo::plane_xz(20.0, 20.0, 2, 1.0);
        let occluder = crate::demo::cube_with_uvs(0.5);
        let occluder_model = Mat4::translate(Vec3::new(0.0, 0.5, -2.0));
        let meshes = [(&ground, Mat4::IDENTITY), (&occluder, occluder_model)];
        let config = CascadeShadowConfig::new(2, 0.5);
        let first = render_cascade_shadow_maps_with_config(
            camera,
            Vec3::new(0.6, 1.0, 0.4),
            config,
            64,
            &meshes,
        )
        .unwrap();
        let moved_camera = Camera::new(
            camera.position + Vec3::new(0.001, 0.0, 0.0),
            camera.orientation,
            camera.fov_y,
            camera.aspect,
            camera.near,
            camera.far,
        );
        let second = render_cascade_shadow_maps_with_config(
            moved_camera,
            Vec3::new(0.6, 1.0, 0.4),
            config,
            64,
            &meshes,
        )
        .unwrap();
        let receiver = Vec3::new(-0.3, 0.0, -2.2);
        let first_visibility = first.visibility(
            receiver,
            5.0,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.6, 1.0, 0.4),
        );
        let second_visibility = second.visibility(
            receiver,
            5.0,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.6, 1.0, 0.4),
        );
        for x in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5] {
            for z in [-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5] {
                let point = Vec3::new(x, 0.0, z);
                let first_grid = first.visibility(
                    point,
                    5.0 - z,
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(0.6, 1.0, 0.4),
                );
                let second_grid = second.visibility(
                    point,
                    5.0 - z,
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(0.6, 1.0, 0.4),
                );
                assert!(
                    (first_grid - second_grid).abs() < 1e-6,
                    "grid moved at {point:?}"
                );
            }
        }
        assert!((first_visibility - second_visibility).abs() < 1e-6);
    }

    #[test]
    fn all_cascade_boundaries_blend_without_a_visibility_jump() {
        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            100.0,
        );
        let maps = vec![
            ShadowMap::from_depth(1, 1, vec![-1.0]).unwrap(),
            ShadowMap::from_depth(1, 1, vec![1.0]).unwrap(),
            ShadowMap::from_depth(1, 1, vec![-1.0]).unwrap(),
        ];
        let state = CascadeShadowState::new(camera, Vec3::new(0.4, 1.0, 0.2), maps).unwrap();
        for (index, &boundary) in state.split_depths()[..2].iter().enumerate() {
            let width = state.blend_widths()[index];
            let lower = if index == 0 {
                camera.near
            } else {
                state.split_depths()[index - 1]
            };
            let upper = state.split_depths()[index + 1];
            let probe_width = (boundary - lower).min(upper - boundary) * 0.1;
            assert!(width > 0.0);
            let step = probe_width * 0.01;
            let mut previous = state.visibility(
                Vec3::new(0.0, 0.0, -(boundary - probe_width)),
                boundary - probe_width,
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.4, 1.0, 0.2),
            );
            for offset in 1..=100 {
                let depth = boundary - width + step * offset as f32;
                let current = state.visibility(
                    Vec3::new(0.0, 0.0, -depth),
                    depth,
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(0.4, 1.0, 0.2),
                );
                assert!((current - previous).abs() < 0.05, "boundary {index} jump");
                previous = current;
            }
        }
    }

    #[test]
    fn config_count_and_lambda_reach_production_rendering() {
        let camera = test_camera();
        let config_two = CascadeShadowConfig::new(2, 0.0);
        let config_two_logarithmic = CascadeShadowConfig::new(2, 1.0);
        let config_four = CascadeShadowConfig::new(4, 1.0);
        let two = render_cascade_shadow_maps_with_config(
            camera,
            Vec3::new(0.5, 1.0, 0.25),
            config_two,
            8,
            &[],
        )
        .unwrap();
        let two_logarithmic = render_cascade_shadow_maps_with_config(
            camera,
            Vec3::new(0.5, 1.0, 0.25),
            config_two_logarithmic,
            8,
            &[],
        )
        .unwrap();
        let four = render_cascade_shadow_maps_with_config(
            camera,
            Vec3::new(0.5, 1.0, 0.25),
            config_four,
            8,
            &[],
        )
        .unwrap();
        assert_eq!(two.cascade_count(), 2);
        assert_eq!(four.cascade_count(), 4);
        assert_ne!(two.split_depths()[0], four.split_depths()[0]);
        assert_ne!(
            two.light_view_projections()[0],
            two_logarithmic.light_view_projections()[0]
        );
    }

    #[test]
    fn extreme_finite_csm_inputs_are_capped_before_projection() {
        let camera = Camera::new(
            Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            Quat::new(f32::MAX, f32::MAX, f32::MAX, f32::MAX),
            f32::MAX,
            f32::MAX,
            f32::MAX,
            f32::MAX,
        );
        let projection = fit_cascade_light_projection(
            camera,
            Vec3::new(f32::MAX, f32::MAX, f32::MAX),
            0.1,
            10.0,
            64,
        );
        assert!(projection.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn production_resolution_discriminator_favors_the_near_cascade() {
        let camera = Camera::new(
            Vec3::new(0.0, 0.5, 8.0),
            Quat::IDENTITY,
            0.9,
            1.0,
            0.1,
            40.0,
        );
        let occluder = crate::demo::cube_with_uvs(0.5);
        let caster_model =
            Mat4::translate(Vec3::new(0.0, 0.5, 7.0)) * Mat4::scale(Vec3::new(0.08, 1.0, 2.0));
        let meshes = [(&occluder, caster_model)];
        let light_direction = Vec3::new(0.8, 1.0, 0.0).normalize();
        let cascades = render_cascade_shadow_maps_with_config(
            camera,
            light_direction,
            CascadeShadowConfig::new(2, 1.0),
            8,
            &meshes,
        )
        .unwrap();
        let single_matrix = fit_cascade_light_projection(camera, light_direction, 0.1, 40.0, 16);
        let mut single_target = Framebuffer::new(16, 8);
        single_target.clear(0);
        let mut depth_pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
        for &(mesh, model) in &meshes {
            depth_pipeline.draw_mesh_depth_with_varyings(
                &mut single_target,
                mesh,
                &ShadowDepthUniforms::new(model, single_matrix),
            );
        }
        let single_map = ShadowMap::from_framebuffer(&single_target).unwrap();
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let receiver = Vec3::new(-0.8, 0.0, 7.0);
        let view_depth = 8.0 - receiver.z;
        let csm_visibility = cascades.visibility(receiver, view_depth, normal, light_direction);
        let single_clip = single_matrix * Vec4::new(receiver.x, receiver.y, receiver.z, 1.0);
        let single_ndc = Vec3::new(
            single_clip.x / single_clip.w,
            single_clip.y / single_clip.w,
            single_clip.z / single_clip.w,
        );
        let single_visibility = single_map.visibility_3x3(
            Vec2::new((single_ndc.x + 1.0) * 0.5, (1.0 - single_ndc.y) * 0.5),
            single_ndc.z,
            0.002,
        );
        assert!(csm_visibility < 0.5);
        assert!(single_visibility > 0.5);
    }

    #[test]
    fn cascade_visibility_uses_the_selected_map() {
        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            20.0,
        );
        let near_map = ShadowMap::from_depth(1, 1, vec![-1.0]).unwrap();
        let far_map = ShadowMap::from_depth(1, 1, vec![1.0]).unwrap();
        let state =
            CascadeShadowState::new(camera, Vec3::new(0.0, 1.0, 0.0), vec![near_map, far_map])
                .unwrap();
        let near_visibility = state.visibility(
            Vec3::new(0.0, 0.0, -1.0),
            1.0,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let far_visibility = state.visibility(
            Vec3::new(0.0, 0.0, -10.0),
            10.0,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!(near_visibility < far_visibility);
    }

    #[test]
    fn uniform_occlusion_is_continuous_across_a_cascade_boundary() {
        let camera = Camera::new(
            Vec3::ZERO,
            Quat::IDENTITY,
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.1,
            20.0,
        );
        let maps = vec![
            ShadowMap::from_depth(4, 4, vec![0.0; 16]).unwrap(),
            ShadowMap::from_depth(4, 4, vec![0.0; 16]).unwrap(),
        ];
        let state = CascadeShadowState::new(camera, Vec3::new(0.0, 1.0, 0.0), maps).unwrap();
        let boundary = state.split_depths()[0];
        let before = state.visibility(
            Vec3::new(0.0, 0.0, -boundary + 1e-4),
            boundary - 1e-4,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let after = state.visibility(
            Vec3::new(0.0, 0.0, -boundary - 1e-4),
            boundary + 1e-4,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!((before - after).abs() <= 1.0 / 16.0);
    }

    #[test]
    fn cube_face_matrices_keep_axis_centers_and_tangent_orientation() {
        let light = Vec3::new(1.0, 2.0, 3.0);
        let matrices = cube_face_view_projections(light, 0.1, 20.0);
        for face in CubeShadowFace::ALL {
            let direction = face.direction();
            let expected_up = match face {
                CubeShadowFace::PositiveY => Vec3::new(0.0, 0.0, 1.0),
                CubeShadowFace::NegativeY => Vec3::new(0.0, 0.0, -1.0),
                CubeShadowFace::PositiveX
                | CubeShadowFace::NegativeX
                | CubeShadowFace::PositiveZ
                | CubeShadowFace::NegativeZ => Vec3::new(0.0, -1.0, 0.0),
            };
            let up = face.up();
            assert_eq!(up, expected_up);
            let right = direction.cross(up).normalize();
            let center = matrices[face.index()]
                * Vec4::new(
                    light.x + direction.x * 2.0,
                    light.y + direction.y * 2.0,
                    light.z + direction.z * 2.0,
                    1.0,
                );
            assert_close(center.x / center.w, 0.0);
            assert_close(center.y / center.w, 0.0);
            let tangent = light + direction * 2.0 + right * 0.25;
            let tangent = matrices[face.index()] * Vec4::new(tangent.x, tangent.y, tangent.z, 1.0);
            assert!(tangent.x / tangent.w > 0.0);
            let vertical = light + direction * 2.0 + up * 0.25;
            let vertical =
                matrices[face.index()] * Vec4::new(vertical.x, vertical.y, vertical.z, 1.0);
            assert!(vertical.y / vertical.w > 0.0);
        }
    }

    #[test]
    fn cube_map_keeps_linear_depth_at_far_range() {
        let near = 0.1;
        let far = 1000.0;
        let mut framebuffer = Framebuffer::new(1, 1);
        framebuffer.depth[0] = 0.999;
        let mut framebuffers = std::array::from_fn(|_| Framebuffer::new(1, 1));
        framebuffers[CubeShadowFace::PositiveZ.index()] = framebuffer;
        let map = CubeShadowMap::from_framebuffers(framebuffers, near, far).unwrap();
        assert_close(
            map.sample_depth(CubeShadowFace::PositiveZ, Vec2::new(0.5, 0.5)),
            0.999,
        );
        let mut far_framebuffer = Framebuffer::new(1, 1);
        far_framebuffer.depth[0] = 1.0;
        let mut far_faces = std::array::from_fn(|_| Framebuffer::new(1, 1));
        far_faces[CubeShadowFace::PositiveZ.index()] = far_framebuffer;
        let far_map = CubeShadowMap::from_framebuffers(far_faces, near, far).unwrap();
        assert_eq!(
            far_map.sample_depth(CubeShadowFace::PositiveZ, Vec2::new(0.5, 0.5)),
            1.0
        );
    }

    #[test]
    fn production_capture_keeps_linear_depth_near_and_at_far() {
        let capture = |distance: f32| {
            let plane = crate::demo::plane_xz(20.0, 20.0, 1, 1.0);
            let model = Mat4::translate(Vec3::new(0.0, 0.0, distance))
                * Mat4::rotate(Vec3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2);
            render_cube_shadow_map(Vec3::ZERO, 0.1, 1000.0, 1024, &[(&plane, model)]).unwrap()
        };

        let near_far_map = capture(999.0);
        assert_close(
            near_far_map.sample_depth(CubeShadowFace::PositiveZ, Vec2::new(0.5, 0.5)),
            0.999,
        );
        let exact_far_map = capture(1000.0);
        assert_eq!(
            exact_far_map.sample_depth(CubeShadowFace::PositiveZ, Vec2::new(0.5, 0.5)),
            1.0
        );
    }

    #[test]
    fn cube_map_pcf_clamps_at_face_edges_without_wrap() {
        let map = CubeShadowMap::from_depth(
            2,
            0.1,
            10.0,
            std::array::from_fn(|_| vec![0.1, 0.1, 0.1, 0.1]),
        )
        .unwrap();
        assert_eq!(
            map.visibility_3x3(CubeShadowFace::PositiveX, Vec2::ZERO, 0.5, 0.0),
            0.0
        );
    }

    #[test]
    fn cube_shadow_parameters_sanitize_immediately() {
        let map =
            CubeShadowMap::from_depth(1, 0.0, -1.0, std::array::from_fn(|_| vec![1.0])).unwrap();
        assert!(map.near_plane() > 0.0);
        assert!(map.far_plane() > map.near_plane());
        let mut state = CubeShadowState::new(Vec3::new(f32::NAN, f32::INFINITY, 1.0), map);
        assert_eq!(state.light_position(), Vec3::new(0.0, 0.0, 1.0));
        let moved_map = state.shadow_map().clone();
        state.replace_capture(Vec3::new(f32::NEG_INFINITY, 2.0, f32::NAN), moved_map);
        assert_eq!(state.light_position(), Vec3::new(0.0, 2.0, 0.0));
        assert!(
            state
                .face_view_projections()
                .iter()
                .all(|matrix| matrix.data.iter().all(|value| value.is_finite()))
        );

        let extreme =
            CubeShadowMap::from_depth(1, f32::MAX, f32::MAX, std::array::from_fn(|_| vec![1.0]))
                .unwrap();
        assert!(extreme.near_plane().is_finite());
        assert!(extreme.far_plane().is_finite());
        assert!(extreme.far_plane() > extreme.near_plane());
    }

    #[test]
    fn extreme_cube_face_matrices_are_finite() {
        let matrices = cube_face_view_projections(Vec3::ZERO, f32::MAX, f32::MAX);
        assert!(
            matrices
                .iter()
                .all(|matrix| matrix.data.iter().all(|value| value.is_finite()))
        );
    }

    #[test]
    fn cube_shadow_render_uses_the_depth_pipeline_for_an_occluder() {
        let occluder = crate::demo::cube_with_uvs(0.5);
        let light = Vec3::ZERO;
        let map = render_cube_shadow_map(
            light,
            0.1,
            10.0,
            64,
            &[(&occluder, Mat4::translate(Vec3::new(0.0, 0.0, 2.0)))],
        )
        .unwrap();
        let state = CubeShadowState::new(light, map);
        assert!(state.visibility(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0)) < 0.5);
        assert!(state.visibility(Vec3::new(2.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0)) > 0.5);
    }

    #[test]
    fn off_axis_capture_uses_radial_distance_for_shadow_sampling() {
        let occluder = crate::demo::cube_with_uvs(0.35);
        let light = Vec3::ZERO;
        let center = Vec3::new(0.8, 0.0, 2.0);
        let map = render_cube_shadow_map(
            light,
            0.1,
            10.0,
            64,
            &[(&occluder, Mat4::translate(center))],
        )
        .unwrap();
        let state = CubeShadowState::new(light, map);
        let shadowed = center.normalize() * 3.0;
        let tangent = Vec3::new(-center.z, 0.0, center.x).normalize();
        let front_surface = center - center.normalize() * 0.35;
        let front_visibility = state.visibility(front_surface, -front_surface.normalize());
        assert!(
            front_visibility > 0.1,
            "off-axis occluder self-shadowed: {front_visibility}"
        );
        assert!(state.visibility(shadowed, -shadowed.normalize()) < 0.5);
        assert!(state.visibility(shadowed + tangent, -shadowed.normalize()) > 0.5);
    }

    #[test]
    fn capture_replacement_rebuilds_position_and_faces_as_one_state() {
        let occluder = crate::demo::cube_with_uvs(0.5);
        let first_light = Vec3::new(0.0, 0.0, -2.0);
        let moved_light = Vec3::new(0.0, 0.0, 2.0);
        let first_map =
            render_cube_shadow_map(first_light, 0.1, 10.0, 64, &[(&occluder, Mat4::IDENTITY)])
                .unwrap();
        let moved_map =
            render_cube_shadow_map(moved_light, 0.1, 10.0, 64, &[(&occluder, Mat4::IDENTITY)])
                .unwrap();
        let mut moved = CubeShadowState::new(first_light, first_map);
        moved.replace_capture(moved_light, moved_map.clone());
        let fresh = CubeShadowState::new(moved_light, moved_map);
        let receiver = Vec3::new(0.0, 0.0, -3.0);
        assert_eq!(
            moved.visibility(receiver, Vec3::new(0.0, 0.0, 1.0)),
            fresh.visibility(receiver, Vec3::new(0.0, 0.0, 1.0))
        );
    }

    #[test]
    fn front_facing_receiver_does_not_get_full_slope_bias() {
        let map =
            CubeShadowMap::from_depth(1, 0.1, 10.0, std::array::from_fn(|_| vec![0.5])).unwrap();
        let mut state = CubeShadowState::new(Vec3::ZERO, map);
        state.set_bias(0.0, 0.02);
        assert!(state.visibility(Vec3::new(0.0, 0.0, 5.1), Vec3::new(0.0, 0.0, -1.0)) < 0.5);
    }

    #[test]
    fn cube_shadow_render_and_sampling_cover_all_six_directions() {
        let occluder = crate::demo::cube_with_uvs(0.35);
        let light = Vec3::ZERO;
        let models = CubeShadowFace::ALL.map(|face| {
            let direction = face.direction();
            (&occluder, Mat4::translate(direction * 2.0))
        });
        let map = render_cube_shadow_map(light, 0.1, 10.0, 64, &models).unwrap();
        let state = CubeShadowState::new(light, map);
        for face in CubeShadowFace::ALL {
            let direction = face.direction();
            assert!(
                state.visibility(direction * 3.0, -direction) < 0.5,
                "face {:?} did not cast a shadow",
                face
            );
        }
    }

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
    fn pcss_penumbra_formula_uses_linear_light_depth() {
        let projection = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 11.0);
        let blocker = linear_light_depth(projection, -0.6, 0.0).unwrap();
        let receiver = linear_light_depth(projection, -0.2, 0.0).unwrap();
        assert!((blocker - 2.0).abs() < 1e-6);
        assert!((receiver - 4.0).abs() < 1e-6);
        assert!((pcss_penumbra_width(receiver, blocker, 1.5) - 1.5).abs() < 1e-6);
        let translated = projection * Mat4::translate(Vec3::new(7.0, -2.0, 5.0));
        assert_eq!(
            linear_light_depth(translated, -0.6, 0.0),
            linear_light_depth(projection, -0.6, 0.0)
        );
    }

    #[test]
    fn pcss_world_light_size_uses_projection_row_norms_for_uv() {
        let matrix = Mat4::new([
            2.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let radius = pcss_search_radius_uv(2.0, matrix);
        assert!((radius.x - 5.0_f32.sqrt()).abs() < 1e-6);
        assert!((radius.y - 5.0).abs() < 1e-6);
    }

    #[test]
    fn pcss_zero_light_size_is_exact_legacy_pcf() {
        let mut depths = vec![0.4; 9];
        depths[4] = -0.4;
        let map = ShadowMap::from_depth(3, 3, depths).unwrap();
        let matrix = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 10.0);
        let uv = Vec2::new(0.5, 0.5);
        let legacy = map.visibility_3x3(uv, 0.0, 0.0);
        let pcss = map.visibility_pcss(
            uv,
            0.0,
            matrix,
            PcssShadowParameters {
                constant_bias: 0.0,
                slope_bias: 0.0,
                normal_dot_light: 0.5,
                light_size: 0.0,
                light_depth_origin: 0.0,
            },
        );
        assert_eq!(legacy, 8.0 / 9.0);
        assert_eq!(pcss, 8.0 / 9.0);
    }

    #[test]
    fn pcss_zero_blockers_is_fully_lit() {
        let mut depths = vec![1.0; 32 * 4];
        for y in 0..4 {
            depths[y * 32 + 4] = -0.9;
        }
        let map = ShadowMap::from_depth(32, 4, depths).unwrap();
        let matrix = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 10.0);
        assert_eq!(
            map.visibility_pcss(
                Vec2::new(0.9, 0.5),
                -0.5,
                matrix,
                PcssShadowParameters {
                    constant_bias: 0.0,
                    slope_bias: 0.0,
                    normal_dot_light: 1.0,
                    light_size: 1.0,
                    light_depth_origin: 0.0,
                },
            ),
            1.0
        );
    }

    #[test]
    fn pcss_kernel_clamps_at_map_edge_without_wrap() {
        let mut depths = vec![0.4; 8];
        depths[0] = -0.8;
        let map = ShadowMap::from_depth(8, 1, depths).unwrap();
        let matrix = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 10.0);
        let visibility = map.visibility_pcss(
            Vec2::new(0.01, 0.5),
            -0.6,
            matrix,
            PcssShadowParameters {
                constant_bias: 0.0,
                slope_bias: 0.0,
                normal_dot_light: 1.0,
                light_size: 3.0,
                light_depth_origin: 0.0,
            },
        );
        assert_eq!(visibility, 0.0);
    }

    #[test]
    fn pcss_light_size_sanitizes_immediately() {
        let map = ShadowMap::from_depth(1, 1, vec![0.0]).unwrap();
        let mut state = ShadowState::new(Mat4::IDENTITY, map);
        state.set_light_size(-1.0);
        assert_eq!(state.light_size(), 0.0);
        state.set_light_size(f32::NAN);
        assert_eq!(state.light_size(), 0.0);
        state.set_light_size(f32::INFINITY);
        assert_eq!(state.light_size(), 0.0);
        state.set_light_size(2.0);
        assert_eq!(state.light_size(), 2.0);
        state.set_light_depth_origin(f32::NAN);
        assert_eq!(state.light_depth_origin(), 0.0);
        state.set_light_depth_origin(3.0);
        assert_eq!(state.light_depth_origin(), 3.0);
    }

    #[test]
    fn pcss_bias_scales_with_wide_kernel_on_a_slope() {
        let base = pcss_filter_bias(0.0, 0.002, 0.0, 1.0);
        let wide = pcss_filter_bias(0.0, 0.002, 0.0, 16.0);
        assert!((base - 0.002).abs() < 1e-6);
        assert!((wide - 0.032).abs() < 1e-6);
    }

    #[test]
    fn pcss_sampling_is_byte_deterministic() {
        let map = ShadowMap::from_depth(32, 32, vec![-0.5; 1024]).unwrap();
        let matrix = Mat4::orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 10.0);
        let parameters = PcssShadowParameters {
            constant_bias: 0.0,
            slope_bias: 0.02,
            normal_dot_light: 0.3,
            light_size: 1.0,
            light_depth_origin: 0.0,
        };
        let first = map.visibility_pcss(Vec2::new(0.45, 0.6), 0.0, matrix, parameters);
        let second = map.visibility_pcss(Vec2::new(0.45, 0.6), 0.0, matrix, parameters);
        assert_eq!(first.to_bits(), second.to_bits());
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
            &MeshVertex::new(crate::math::Vec3::new(0.0, 0.0, -1.0), None, None),
            &uniforms,
        );
        assert_eq!(output.clip_position, Vec4::new(0.5, 0.0, -1.0, 1.0));
    }
}
