//! Cascaded directional shadow maps.

use crate::camera::Camera;
use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::mesh::Mesh;
use crate::pipeline::Pipeline;
use crate::shadow::{
    ShadowDepthShader, ShadowDepthUniforms, ShadowMap, sanitize_bias, sanitize_far_plane,
    sanitize_near_plane, sanitize_position,
};

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
    blend_widths: [f32; MAX_CASCADES - 1],
    pub(crate) light_view_projections: [Mat4; MAX_CASCADES],
    pub(crate) shadow_maps: [Option<ShadowMap>; MAX_CASCADES],
    constant_bias: f32,
    slope_bias: f32,
}

impl CascadeShadowState {
    pub fn new(
        camera: Camera,
        light_direction: Vec3,
        shadow_maps: Vec<ShadowMap>,
    ) -> Result<Self, String> {
        let mut config = CascadeShadowConfig::default();
        config.set_cascade_count(shadow_maps.len());
        Self::with_config(camera, light_direction, shadow_maps, config)
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
        let count = config.cascade_count();
        if !(2..=MAX_CASCADES).contains(&count) || shadow_maps.len() != count {
            return Err("cascade config count must match 2 to 4 shadow maps".to_string());
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
        Self::from_parts(camera, maps, config, light_view_projections)
    }

    fn with_config_and_projections(
        camera: Camera,
        shadow_maps: Vec<ShadowMap>,
        config: CascadeShadowConfig,
        light_view_projections: [Mat4; MAX_CASCADES],
    ) -> Result<Self, String> {
        let count = config.cascade_count();
        if !(2..=MAX_CASCADES).contains(&count) || shadow_maps.len() != count {
            return Err("cascade config count must match 2 to 4 shadow maps".to_string());
        }
        let mut maps: [Option<ShadowMap>; MAX_CASCADES] = std::array::from_fn(|_| None);
        for (slot, map) in maps.iter_mut().zip(shadow_maps) {
            *slot = Some(map);
        }
        Self::from_parts(
            sanitize_camera(camera),
            maps,
            config,
            light_view_projections,
        )
    }

    fn from_parts(
        camera: Camera,
        maps: [Option<ShadowMap>; MAX_CASCADES],
        config: CascadeShadowConfig,
        light_view_projections: [Mat4; MAX_CASCADES],
    ) -> Result<Self, String> {
        let count = config.cascade_count();
        let split_depths = practical_split_depths(camera.near, camera.far, count, config.lambda());
        let blend_widths = cascade_blend_widths(camera.near, &split_depths, count);
        Ok(Self {
            cascade_count: count,
            split_depths,
            blend_widths,
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

    pub fn blend_widths(&self) -> &[f32] {
        &self.blend_widths[..self.cascade_count.saturating_sub(1)]
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
        let current = self.sample_cascade(index, world_position, normal, light_direction);
        if index == 0 {
            return current;
        }
        let boundary = self.split_depths[index - 1];
        let width = self.blend_widths[index - 1];
        if width <= 0.0 || view_depth <= boundary - width {
            return current;
        }
        let previous = self.sample_cascade(index - 1, world_position, normal, light_direction);
        let amount = ((view_depth - (boundary - width)) / width).clamp(0.0, 1.0);
        previous * (1.0 - amount) + current * amount
    }

    fn sample_cascade(
        &self,
        index: usize,
        world_position: Vec3,
        normal: Vec3,
        light_direction: Vec3,
    ) -> f32 {
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
        // The depth offset scales with this cascade's texel footprint. This
        // keeps the slope term stable as cascade resolution and extent vary.
        let texel_depth = 2.0 / map.height().max(1) as f32;
        let depth_offset = texel_depth * 0.1;
        let bias = self
            .constant_bias
            .max(self.slope_bias * (1.0 - normal_dot_light) + depth_offset);
        map.visibility_3x3(uv, ndc.z, bias)
    }
}

fn cascade_blend_widths(
    near: f32,
    splits: &[f32; MAX_CASCADES],
    count: usize,
) -> [f32; MAX_CASCADES - 1] {
    let mut widths = [0.0; MAX_CASCADES - 1];
    for index in 0..count.saturating_sub(1) {
        let lower = if index == 0 { near } else { splits[index - 1] };
        let upper = splits[index + 1];
        widths[index] = (splits[index] - lower).min(upper - splits[index]).max(0.0) * 0.1;
    }
    widths
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
    fit_cascade_light_projection_with_casters(
        camera,
        light_direction,
        slice_near,
        slice_far,
        shadow_map_size,
        &[],
    )
}

pub(crate) fn fit_cascade_light_projection_with_casters(
    camera: Camera,
    light_direction: Vec3,
    slice_near: f32,
    slice_far: f32,
    shadow_map_size: usize,
    caster_points: &[Vec3],
) -> Mat4 {
    let camera = sanitize_camera(camera);
    let corners = frustum_slice_corners(camera, slice_near, slice_far);
    let direction = safe_direction(light_direction);
    let up = safe_up(direction);
    // The basis is fixed by the light, not by the camera. Snap its world-space
    // origin before building the translated view matrix. This keeps a static
    // scene on the same texel grid during sub-texel camera motion.
    let basis = Mat4::look_at(Vec3::ZERO, -direction, up);
    let basis_corners = corners.map(|corner| transform_point(basis, corner));
    let center = basis_corners
        .iter()
        .copied()
        .fold(Vec3::ZERO, |sum, value| sum + value)
        / basis_corners.len() as f32;
    let mut min = basis_corners[0];
    let mut max = basis_corners[0];
    for point in basis_corners.iter().skip(1) {
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
    let snapped_basis_origin = Vec3::new(center_x, center_y, center.z);
    let snapped_origin = basis.inverse().map_or(Vec3::ZERO, |inverse| {
        transform_point(inverse, snapped_basis_origin)
    });
    let distance = corners
        .iter()
        .chain(caster_points.iter())
        .map(|point| (*point - snapped_origin).length())
        .filter(|value| value.is_finite())
        .fold(1.0, f32::max)
        + 1.0;
    let light_origin = snapped_origin + direction * distance;
    let view = basis * Mat4::translate(-light_origin);
    let mut depth_points = corners.to_vec();
    depth_points.extend(caster_points.iter().copied());
    let light_points: Vec<Vec3> = depth_points
        .into_iter()
        .filter_map(|point| finite_point(transform_point(view, point)))
        .collect();
    let mut depth_min = light_points.first().copied().unwrap_or(Vec3::ZERO).z;
    let mut depth_max = depth_min;
    for point in light_points.iter().skip(1) {
        depth_min = depth_min.min(point.z);
        depth_max = depth_max.max(point.z);
    }
    // The padding scales with the light-space scene extent. It covers raster
    // edge rounding without imposing a fixed-depth clipping margin.
    let depth_padding = ((depth_max - depth_min).abs() * 0.01).max(0.001);
    let near = (-depth_max - depth_padding).max(0.001);
    let far = (-depth_min + depth_padding).max(near + 0.001);
    Mat4::orthographic(
        -extent_x * 0.5,
        extent_x * 0.5,
        -extent_y * 0.5,
        extent_y * 0.5,
        near,
        far,
    ) * view
}

fn transform_point(matrix: Mat4, point: Vec3) -> Vec3 {
    let transformed = matrix * Vec4::new(point.x, point.y, point.z, 1.0);
    if transformed.w == 0.0 || !transformed.w.is_finite() {
        return Vec3::new(f32::NAN, f32::NAN, f32::NAN);
    }
    Vec3::new(
        transformed.x / transformed.w,
        transformed.y / transformed.w,
        transformed.z / transformed.w,
    )
}

fn finite_point(point: Vec3) -> Option<Vec3> {
    if point.x.is_finite() && point.y.is_finite() && point.z.is_finite() {
        Some(Vec3::new(
            point
                .x
                .clamp(-MAX_GEOMETRY_COORDINATE, MAX_GEOMETRY_COORDINATE),
            point
                .y
                .clamp(-MAX_GEOMETRY_COORDINATE, MAX_GEOMETRY_COORDINATE),
            point
                .z
                .clamp(-MAX_GEOMETRY_COORDINATE, MAX_GEOMETRY_COORDINATE),
        ))
    } else {
        None
    }
}

const MAX_GEOMETRY_COORDINATE: f32 = 10_000.0;

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
    render_cascade_shadow_maps_with_config(
        camera,
        light_direction,
        CascadeShadowConfig::new(cascade_count, 0.5),
        map_size,
        meshes,
    )
}

/// Renders cascades with one configuration shared by split calculation,
/// fitting, map rendering, and the returned sampling state.
pub fn render_cascade_shadow_maps_with_config(
    camera: Camera,
    light_direction: Vec3,
    config: CascadeShadowConfig,
    map_size: usize,
    meshes: &[(&Mesh, Mat4)],
) -> Result<CascadeShadowState, String> {
    let count = config.cascade_count();
    if map_size == 0 {
        return Err("cascade shadow-map size must be non-zero".to_string());
    }
    let camera = sanitize_camera(camera);
    let split_depths = practical_split_depths(camera.near, camera.far, count, config.lambda());
    let caster_points = meshes
        .iter()
        .flat_map(|(mesh, model)| {
            mesh.vertices()
                .iter()
                .filter_map(|vertex| finite_point(transform_point(*model, vertex.position())))
        })
        .collect::<Vec<_>>();
    let mut maps = Vec::with_capacity(count);
    let mut light_view_projections = [Mat4::IDENTITY; MAX_CASCADES];
    let mut pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    for index in 0..count {
        let slice_near = if index == 0 {
            camera.near
        } else {
            split_depths[index - 1]
        };
        let matrix = fit_cascade_light_projection_with_casters(
            camera,
            light_direction,
            slice_near,
            split_depths[index],
            map_size,
            &caster_points,
        );
        light_view_projections[index] = matrix;
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
    CascadeShadowState::with_config_and_projections(camera, maps, config, light_view_projections)
}

fn ndc_depth_for_view_depth(depth: f32, near: f32, far: f32) -> f32 {
    let a = (far + near) / (near - far);
    let b = (2.0 * far * near) / (near - far);
    (-a * depth + b) / depth
}

fn sanitize_camera(mut camera: Camera) -> Camera {
    camera.position = finite_point(sanitize_position(camera.position)).unwrap_or(Vec3::ZERO);
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
    let direction = finite_point(sanitize_position(direction)).unwrap_or(Vec3::ZERO);
    let length = direction.length();
    if length.is_finite() && length > 0.0 {
        direction / length
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
