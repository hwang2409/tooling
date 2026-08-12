//! Directional shadow-map targets, sampling, and depth-pass shaders.

use crate::camera::Camera;
use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::pipeline::{Pipeline, Varyings, VertexOutput, VertexStage};
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
    cascades: Option<CascadeShadowState>,
    constant_bias: f32,
    slope_bias: f32,
}

impl ShadowState {
    pub fn new(light_view_projection: Mat4, shadow_map: ShadowMap) -> Self {
        Self {
            light_view_projection,
            shadow_map,
            cascades: None,
            constant_bias: 0.002,
            slope_bias: 0.02,
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

    pub fn set_bias(&mut self, constant: f32, slope: f32) {
        self.constant_bias = sanitize_bias(constant);
        self.slope_bias = sanitize_bias(slope);
        if let Some(cascades) = self.cascades.as_mut() {
            cascades.set_bias(constant, slope);
        }
    }
}

/// Maximum number of directional-light cascades.
pub const MAX_CASCADES: usize = 4;

/// User-facing cascade count and split blend settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeShadowConfig {
    cascade_count: usize,
    lambda: f32,
}

impl Default for CascadeShadowConfig {
    fn default() -> Self {
        Self {
            cascade_count: 3,
            lambda: 0.5,
        }
    }
}

impl CascadeShadowConfig {
    pub fn new(cascade_count: usize, lambda: f32) -> Self {
        let mut config = Self::default();
        config.set_cascade_count(cascade_count);
        config.set_lambda(lambda);
        config
    }

    pub const fn cascade_count(self) -> usize {
        self.cascade_count
    }

    pub const fn lambda(self) -> f32 {
        self.lambda
    }

    pub fn set_cascade_count(&mut self, cascade_count: usize) {
        self.cascade_count = cascade_count.clamp(2, MAX_CASCADES);
    }

    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = if lambda.is_finite() {
            lambda.clamp(0.0, 1.0)
        } else {
            0.5
        };
    }
}

/// Stores the split bounds, light projections, and maps for a cascaded shadow.
///
/// Splits use the practical scheme. For cascade `i`,
/// `c_log = n * (f / n)^(i / N)`, `c_uni = n + (f - n) * (i / N)`, and
/// `split = lambda * c_log + (1 - lambda) * c_uni`. The default lambda is 0.5.
#[derive(Clone, Debug, PartialEq)]
pub struct CascadeShadowState {
    cascade_count: usize,
    split_depths: [f32; MAX_CASCADES],
    light_view_projections: [Mat4; MAX_CASCADES],
    shadow_maps: [Option<ShadowMap>; MAX_CASCADES],
    constant_bias: f32,
    slope_bias: f32,
}

impl CascadeShadowState {
    pub fn new(
        camera: Camera,
        light_direction: Vec3,
        shadow_maps: Vec<ShadowMap>,
    ) -> Result<Self, String> {
        Self::with_config(
            camera,
            light_direction,
            shadow_maps,
            CascadeShadowConfig::default(),
        )
    }

    pub fn with_lambda(
        camera: Camera,
        light_direction: Vec3,
        shadow_maps: Vec<ShadowMap>,
        lambda: f32,
    ) -> Result<Self, String> {
        let mut config = CascadeShadowConfig::default();
        config.set_cascade_count(shadow_maps.len());
        config.set_lambda(lambda);
        Self::with_config(camera, light_direction, shadow_maps, config)
    }

    pub fn with_config(
        camera: Camera,
        light_direction: Vec3,
        shadow_maps: Vec<ShadowMap>,
        config: CascadeShadowConfig,
    ) -> Result<Self, String> {
        let count = shadow_maps.len();
        if !(2..=MAX_CASCADES).contains(&count) {
            return Err("cascades must contain between 2 and 4 maps".to_string());
        }
        let mut maps: [Option<ShadowMap>; MAX_CASCADES] = std::array::from_fn(|_| None);
        for (slot, map) in maps.iter_mut().zip(shadow_maps) {
            *slot = Some(map);
        }
        let camera = sanitize_camera(camera);
        let split_depths = practical_split_depths(camera.near, camera.far, count, config.lambda());
        let light_view_projections = std::array::from_fn(|index| {
            if index < count {
                fit_cascade_light_projection(
                    camera,
                    light_direction,
                    if index == 0 {
                        camera.near
                    } else {
                        split_depths[index - 1]
                    },
                    split_depths[index],
                    maps[index].as_ref().map_or(1, ShadowMap::width),
                )
            } else {
                Mat4::IDENTITY
            }
        });
        Ok(Self {
            cascade_count: count,
            split_depths,
            light_view_projections,
            shadow_maps: maps,
            constant_bias: 0.002,
            slope_bias: 0.02,
        })
    }

    pub fn cascade_count(&self) -> usize {
        self.cascade_count
    }

    pub fn split_depths(&self) -> &[f32] {
        &self.split_depths[..self.cascade_count]
    }

    pub fn light_view_projections(&self) -> &[Mat4] {
        &self.light_view_projections[..self.cascade_count]
    }

    pub fn shadow_maps(&self) -> impl Iterator<Item = &ShadowMap> {
        self.shadow_maps[..self.cascade_count]
            .iter()
            .filter_map(Option::as_ref)
    }

    pub fn select_cascade(&self, view_depth: f32) -> usize {
        cascade_index(view_depth, self.split_depths())
    }

    pub const fn bias(&self) -> (f32, f32) {
        (self.constant_bias, self.slope_bias)
    }

    pub fn set_bias(&mut self, constant: f32, slope: f32) {
        self.constant_bias = sanitize_bias(constant);
        self.slope_bias = sanitize_bias(slope);
    }

    /// Samples the selected cascade with the same 3x3 PCF and slope bias as a
    /// single directional map. View depth is positive distance along camera -Z.
    pub fn visibility(
        &self,
        world_position: Vec3,
        view_depth: f32,
        normal: Vec3,
        light_direction: Vec3,
    ) -> f32 {
        if !view_depth.is_finite() {
            return 1.0;
        }
        let index = self.select_cascade(view_depth);
        let map = self.shadow_maps[index]
            .as_ref()
            .expect("cascade map exists");
        let clip = self.light_view_projections[index]
            * Vec4::new(world_position.x, world_position.y, world_position.z, 1.0);
        if clip.w <= 0.0 || !clip.w.is_finite() {
            return 1.0;
        }
        let ndc = Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        let uv = Vec2::new((ndc.x + 1.0) * 0.5, (1.0 - ndc.y) * 0.5);
        let normal_dot_light = normal
            .normalize()
            .dot(light_direction.normalize())
            .clamp(0.0, 1.0);
        let bias = self
            .constant_bias
            .max(self.slope_bias * (1.0 - normal_dot_light));
        map.visibility_3x3(uv, ndc.z, bias)
    }
}

/// Computes practical split far bounds for `count` cascades.
pub fn practical_split_depths(
    near: f32,
    far: f32,
    count: usize,
    lambda: f32,
) -> [f32; MAX_CASCADES] {
    let near = sanitize_near_plane(near);
    let far = sanitize_far_plane(far, near);
    let count = count.clamp(2, MAX_CASCADES);
    let lambda = if lambda.is_finite() {
        lambda.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let mut splits = [far; MAX_CASCADES];
    for index in 1..=count {
        let fraction = index as f32 / count as f32;
        let logarithmic = near * (far / near).powf(fraction);
        let uniform = near + (far - near) * fraction;
        splits[index - 1] = lambda * logarithmic + (1.0 - lambda) * uniform;
    }
    splits
}

/// Selects the first cascade whose far bound contains `view_depth`.
pub fn cascade_index(view_depth: f32, split_depths: &[f32]) -> usize {
    if split_depths.is_empty() {
        return 0;
    }
    split_depths
        .iter()
        .position(|&split| view_depth <= split)
        .unwrap_or(split_depths.len() - 1)
}

/// Returns the eight world-space corners of a perspective frustum slice.
pub fn frustum_slice_corners(camera: Camera, slice_near: f32, slice_far: f32) -> [Vec3; 8] {
    let camera = sanitize_camera(camera);
    let near = slice_near.clamp(camera.near, camera.far);
    let far = slice_far.clamp(near, camera.far);
    let inverse = camera.view_projection().inverse().unwrap_or(Mat4::IDENTITY);
    let near_ndc = ndc_depth_for_view_depth(near, camera.near, camera.far);
    let far_ndc = ndc_depth_for_view_depth(far, camera.near, camera.far);
    let mut corners = [Vec3::ZERO; 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        let depth = if index < 4 { near_ndc } else { far_ndc };
        let x = if index % 2 == 0 { -1.0 } else { 1.0 };
        let y = if index % 4 < 2 { -1.0 } else { 1.0 };
        let point = inverse * Vec4::new(x, y, depth, 1.0);
        *corner = Vec3::new(point.x / point.w, point.y / point.w, point.z / point.w);
    }
    corners
}

/// Fits a snapped orthographic light projection around one frustum slice.
pub fn fit_cascade_light_projection(
    camera: Camera,
    light_direction: Vec3,
    slice_near: f32,
    slice_far: f32,
    shadow_map_size: usize,
) -> Mat4 {
    let camera = sanitize_camera(camera);
    let corners = frustum_slice_corners(camera, slice_near, slice_far);
    let center = corners
        .iter()
        .copied()
        .fold(Vec3::ZERO, |sum, value| sum + value)
        / corners.len() as f32;
    let direction = safe_direction(light_direction);
    let up = safe_up(direction);
    let distance = corners
        .iter()
        .map(|corner| (*corner - center).length())
        .fold(1.0, f32::max)
        + 1.0;
    let view = directional_light_view(direction, center, distance, up);
    let light_corners = corners.map(|corner| {
        let point = view * Vec4::new(corner.x, corner.y, corner.z, 1.0);
        Vec3::new(point.x, point.y, point.z)
    });
    let mut min = light_corners[0];
    let mut max = light_corners[0];
    for point in light_corners.iter().skip(1) {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        min.z = min.z.min(point.z);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
        max.z = max.z.max(point.z);
    }
    let raw_extent_x = (max.x - min.x).max(0.001);
    let raw_extent_y = (max.y - min.y).max(0.001);
    let texels = shadow_map_size.max(1) as f32;
    let texel_x = raw_extent_x / texels;
    let texel_y = raw_extent_y / texels;
    // One texel of guard band keeps the snapped box enclosing its extrema.
    let extent_x = raw_extent_x + texel_x;
    let extent_y = raw_extent_y + texel_y;
    let center_x = snap_ortho_origin((min.x + max.x) * 0.5, texel_x);
    let center_y = snap_ortho_origin((min.y + max.y) * 0.5, texel_y);
    let depth_margin = 1.0;
    let near = (-(max.z) - depth_margin).max(0.001);
    let far = (-min.z + depth_margin).max(near + 0.001);
    Mat4::orthographic(
        center_x - extent_x * 0.5,
        center_x + extent_x * 0.5,
        center_y - extent_y * 0.5,
        center_y + extent_y * 0.5,
        near,
        far,
    ) * view
}

/// Snaps an orthographic origin to the shadow-map texel grid.
pub fn snap_ortho_origin(origin: f32, texel_size: f32) -> f32 {
    if !origin.is_finite() || !texel_size.is_finite() || texel_size <= 0.0 {
        return 0.0;
    }
    (origin / texel_size).round() * texel_size
}

/// Renders all cascades through the existing depth-only pipeline.
pub fn render_cascade_shadow_maps(
    camera: Camera,
    light_direction: Vec3,
    cascade_count: usize,
    map_size: usize,
    meshes: &[(&Mesh, Mat4)],
) -> Result<CascadeShadowState, String> {
    let count = cascade_count.clamp(2, MAX_CASCADES);
    if map_size == 0 {
        return Err("cascade shadow-map size must be non-zero".to_string());
    }
    let camera = sanitize_camera(camera);
    let split_depths = practical_split_depths(camera.near, camera.far, count, 0.5);
    let mut maps = Vec::with_capacity(count);
    let mut pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    for index in 0..count {
        let slice_near = if index == 0 {
            camera.near
        } else {
            split_depths[index - 1]
        };
        let matrix = fit_cascade_light_projection(
            camera,
            light_direction,
            slice_near,
            split_depths[index],
            map_size,
        );
        let mut target = Framebuffer::new(map_size, map_size);
        target.clear(0);
        for &(mesh, model) in meshes {
            pipeline.draw_mesh_depth_with_varyings(
                &mut target,
                mesh,
                &ShadowDepthUniforms::new(model, matrix),
            );
        }
        maps.push(ShadowMap::from_framebuffer(&target)?);
    }
    CascadeShadowState::new(camera, light_direction, maps)
}

fn ndc_depth_for_view_depth(depth: f32, near: f32, far: f32) -> f32 {
    let a = (far + near) / (near - far);
    let b = (2.0 * far * near) / (near - far);
    (-a * depth + b) / depth
}

fn sanitize_camera(mut camera: Camera) -> Camera {
    camera.position = sanitize_position(camera.position);
    camera.fov_y = if camera.fov_y.is_finite() {
        camera.fov_y.clamp(0.01, std::f32::consts::PI - 0.01)
    } else {
        std::f32::consts::FRAC_PI_4
    };
    camera.aspect = if camera.aspect.is_finite() && camera.aspect > 0.0 {
        camera.aspect
    } else {
        1.0
    };
    camera.near = sanitize_near_plane(camera.near);
    camera.far = sanitize_far_plane(camera.far, camera.near);
    camera
}

fn safe_direction(direction: Vec3) -> Vec3 {
    let direction = sanitize_position(direction);
    if direction.length() > 0.0 {
        direction.normalize()
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    }
}

fn safe_up(direction: Vec3) -> Vec3 {
    if direction.y.abs() > 0.95 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    }
}

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

fn sanitize_position(position: Vec3) -> Vec3 {
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

fn sanitize_near_plane(value: f32) -> f32 {
    const MIN_NEAR_PLANE: f32 = 0.0001;
    const MAX_NEAR_PLANE: f32 = 10_000.0;
    if value.is_finite() && value > 0.0 {
        value.clamp(MIN_NEAR_PLANE, MAX_NEAR_PLANE)
    } else {
        0.01_f32.clamp(MIN_NEAR_PLANE, MAX_NEAR_PLANE)
    }
}

fn sanitize_far_plane(value: f32, near_plane: f32) -> f32 {
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

fn sanitize_bias(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
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
        assert!(max_xy > 0.8);

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
