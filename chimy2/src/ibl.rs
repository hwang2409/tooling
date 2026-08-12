//! Deterministic image-based lighting precomputation.
//!
//! The source cube is sampled in linear space. Irradiance stores the
//! cosine-normalized integral `integral(L cos(theta) d_omega) / PI`, so a
//! constant environment remains constant. Prefilter levels use deterministic
//! GGX importance samples. The split-sum BRDF is a deterministic 2D lookup
//! table using the Karis Smith-Schlick visibility approximation.

use crate::math::{Vec2, Vec3};
use crate::skybox::{CubeFace, CubeTexture};
use std::f32::consts::PI;

const MIN_ROUGHNESS: f32 = 0.001;

/// Knobs for the load-time IBL bake.
///
/// The defaults use 16x16 irradiance faces, 32x32 prefiltered faces over five
/// roughness levels, 64x64 BRDF pixels, and fixed sample counts. Increase the
/// counts or face sizes for a higher quality bake. No setting uses randomness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IblSettings {
    pub irradiance_size: usize,
    pub irradiance_samples: usize,
    pub prefilter_size: usize,
    pub prefilter_levels: usize,
    pub prefilter_samples: usize,
    pub brdf_size: usize,
    pub brdf_samples: usize,
}

impl Default for IblSettings {
    fn default() -> Self {
        Self {
            irradiance_size: 16,
            irradiance_samples: 64,
            prefilter_size: 32,
            prefilter_levels: 5,
            prefilter_samples: 64,
            brdf_size: 64,
            brdf_samples: 64,
        }
    }
}

impl IblSettings {
    /// Small settings for unit tests and hand inspection.
    pub const fn test() -> Self {
        Self {
            irradiance_size: 2,
            irradiance_samples: 16,
            prefilter_size: 2,
            prefilter_levels: 3,
            prefilter_samples: 16,
            brdf_size: 8,
            brdf_samples: 32,
        }
    }

    fn sanitized(self) -> Self {
        Self {
            irradiance_size: self.irradiance_size.max(1),
            irradiance_samples: self.irradiance_samples.max(1),
            prefilter_size: self.prefilter_size.max(1),
            prefilter_levels: self.prefilter_levels.max(1),
            prefilter_samples: self.prefilter_samples.max(1),
            brdf_size: self.brdf_size.max(1),
            brdf_samples: self.brdf_samples.max(1),
        }
    }
}

/// One RGB floating-point cube map with the same face order as [`CubeTexture`].
#[derive(Clone, Debug, PartialEq)]
pub struct FloatCube {
    size: usize,
    faces: [Vec<Vec3>; 6],
}

impl FloatCube {
    /// Decodes a cube into floating-point RGB and applies a linear intensity.
    /// The scale allows an 8-bit sky asset to drive an HDR lighting test.
    pub fn from_cube_texture(environment: &CubeTexture, intensity: f32) -> Self {
        let intensity = if intensity.is_finite() {
            intensity.max(0.0)
        } else {
            1.0
        };
        let size = environment.size();
        let faces = std::array::from_fn(|face_index| {
            let texture = environment.face(cube_face(face_index));
            (0..size * size)
                .map(|index| {
                    let uv = Vec2::new(
                        ((index % size) as f32 + 0.5) / size as f32,
                        ((index / size) as f32 + 0.5) / size as f32,
                    );
                    let sample = texture.sample_linear_nearest(uv);
                    Vec3::new(sample[0], sample[1], sample[2]) * intensity
                })
                .collect()
        });
        Self::new(size, faces)
    }

    fn new(size: usize, faces: [Vec<Vec3>; 6]) -> Self {
        debug_assert!(size > 0);
        debug_assert!(faces.iter().all(|face| face.len() == size * size));
        Self { size, faces }
    }

    /// Builds a constant floating-point cube.
    pub fn constant(size: usize, color: Vec3) -> Self {
        let size = size.max(1);
        Self::new(
            std::cmp::max(size, 1),
            std::array::from_fn(|_| vec![color; size * size]),
        )
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns a face pixel in `+X, -X, +Y, -Y, +Z, -Z` order.
    pub fn pixel(&self, face: CubeFace, x: usize, y: usize) -> Vec3 {
        self.faces[face_index(face)][y.min(self.size - 1) * self.size + x.min(self.size - 1)]
    }

    /// Samples the cube with clamped bilinear filtering.
    pub fn sample(&self, direction: Vec3) -> Vec3 {
        let (face, uv) = CubeTexture::face_and_uv(direction);
        self.sample_face(face, uv)
    }

    fn sample_face(&self, face: CubeFace, uv: Vec2) -> Vec3 {
        let u = uv.x.clamp(0.0, 1.0);
        let v = uv.y.clamp(0.0, 1.0);
        let x = u * self.size as f32 - 0.5;
        let y = v * self.size as f32 - 0.5;
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let tx = x - x.floor();
        let ty = y - y.floor();
        let p00 = self.pixel_signed(face, x0, y0);
        let p10 = self.pixel_signed(face, x0 + 1, y0);
        let p01 = self.pixel_signed(face, x0, y0 + 1);
        let p11 = self.pixel_signed(face, x0 + 1, y0 + 1);
        let top = p00 * (1.0 - tx) + p10 * tx;
        let bottom = p01 * (1.0 - tx) + p11 * tx;
        top * (1.0 - ty) + bottom * ty
    }

    fn pixel_signed(&self, face: CubeFace, x: isize, y: isize) -> Vec3 {
        let x = x.clamp(0, self.size as isize - 1) as usize;
        let y = y.clamp(0, self.size as isize - 1) as usize;
        self.pixel(face, x, y)
    }
}

/// A roughness-indexed environment cube chain.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefilteredEnvironment {
    levels: Vec<FloatCube>,
}

impl PrefilteredEnvironment {
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn level(&self, index: usize) -> Option<&FloatCube> {
        self.levels.get(index)
    }

    /// Samples between roughness levels using linear interpolation.
    pub fn sample(&self, direction: Vec3, roughness: f32) -> Vec3 {
        let last = self.levels.len().saturating_sub(1);
        if last == 0 {
            return self.levels[0].sample(direction);
        }
        let position = roughness.clamp(0.0, 1.0) * last as f32;
        let lower = position.floor() as usize;
        let upper = (lower + 1).min(last);
        let amount = position - lower as f32;
        let a = self.levels[lower].sample(direction);
        let b = self.levels[upper].sample(direction);
        a * (1.0 - amount) + b * amount
    }
}

/// The precomputed split-sum environment BRDF lookup.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentBrdfLut {
    size: usize,
    pixels: Vec<Vec2>,
}

impl EnvironmentBrdfLut {
    pub const fn size(&self) -> usize {
        self.size
    }

    pub fn pixel(&self, x: usize, y: usize) -> Vec2 {
        self.pixels[y.min(self.size - 1) * self.size + x.min(self.size - 1)]
    }

    /// Samples by `N dot V` on x and roughness on y.
    pub fn sample(&self, n_dot_v: f32, roughness: f32) -> Vec2 {
        let x = n_dot_v.clamp(0.0, 1.0) * self.size as f32 - 0.5;
        let y = roughness.clamp(0.0, 1.0) * self.size as f32 - 0.5;
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let tx = x - x.floor();
        let ty = y - y.floor();
        let p00 = self.pixel_signed(x0, y0);
        let p10 = self.pixel_signed(x0 + 1, y0);
        let p01 = self.pixel_signed(x0, y0 + 1);
        let p11 = self.pixel_signed(x0 + 1, y0 + 1);
        let top = p00 * (1.0 - tx) + p10 * tx;
        let bottom = p01 * (1.0 - tx) + p11 * tx;
        top * (1.0 - ty) + bottom * ty
    }

    fn pixel_signed(&self, x: isize, y: isize) -> Vec2 {
        let x = x.clamp(0, self.size as isize - 1) as usize;
        let y = y.clamp(0, self.size as isize - 1) as usize;
        self.pixel(x, y)
    }
}

/// All load-time maps required by the IBL GGX shader.
#[derive(Clone, Debug, PartialEq)]
pub struct IblMaps {
    pub irradiance: FloatCube,
    pub prefiltered: PrefilteredEnvironment,
    pub brdf_lut: EnvironmentBrdfLut,
    pub settings: IblSettings,
}

impl IblMaps {
    /// Bakes the default 16/32/64 maps from the linear-sampled source cube.
    pub fn from_environment(environment: &CubeTexture) -> Self {
        Self::from_environment_with_settings(environment, IblSettings::default())
    }

    /// Bakes all maps once. Every sample is derived from its integer index.
    pub fn from_environment_with_settings(
        environment: &CubeTexture,
        settings: IblSettings,
    ) -> Self {
        Self::from_sampler(|direction| rgb(environment.sample(direction)), settings)
    }

    /// Bakes from an existing floating-point cube, including values above 1.0.
    /// This is the HDR source path when an application owns decoded float data.
    pub fn from_float_environment(environment: &FloatCube, settings: IblSettings) -> Self {
        Self::from_sampler(|direction| environment.sample(direction), settings)
    }

    fn from_sampler<S>(sample: S, settings: IblSettings) -> Self
    where
        S: Fn(Vec3) -> Vec3 + Copy,
    {
        let settings = settings.sanitized();
        let irradiance = bake_irradiance(sample, settings);
        let prefiltered = bake_prefiltered(sample, settings);
        let brdf_lut = bake_brdf_lut(settings);
        Self {
            irradiance,
            prefiltered,
            brdf_lut,
            settings,
        }
    }
}

fn bake_irradiance<S>(sample: S, settings: IblSettings) -> FloatCube
where
    S: Fn(Vec3) -> Vec3 + Copy,
{
    let size = settings.irradiance_size;
    let faces = std::array::from_fn(|face_index| {
        let face = cube_face(face_index);
        (0..size * size)
            .map(|index| {
                let normal = direction_for_texel(face, index % size, index / size, size);
                integrate_irradiance(sample, normal, settings.irradiance_samples)
            })
            .collect()
    });
    FloatCube::new(size, faces)
}

fn integrate_irradiance<S>(sample: S, normal: Vec3, sample_count: usize) -> Vec3
where
    S: Fn(Vec3) -> Vec3 + Copy,
{
    let (tangent, bitangent) = basis(normal);
    let mut sum = Vec3::ZERO;
    for sample_index in 0..sample_count {
        let xi = hammersley(sample_index, sample_count);
        let radius = xi.x.sqrt();
        let phi = 2.0 * PI * xi.y;
        let local = Vec3::new(radius * phi.cos(), radius * phi.sin(), (1.0 - xi.x).sqrt());
        let direction = (tangent * local.x + bitangent * local.y + normal * local.z).normalize();
        sum = sum + sample(direction);
    }
    sum / sample_count as f32
}

fn bake_prefiltered<S>(sample: S, settings: IblSettings) -> PrefilteredEnvironment
where
    S: Fn(Vec3) -> Vec3 + Copy,
{
    let mut levels = Vec::with_capacity(settings.prefilter_levels);
    for level in 0..settings.prefilter_levels {
        let roughness = if settings.prefilter_levels == 1 {
            0.0
        } else {
            level as f32 / (settings.prefilter_levels - 1) as f32
        };
        let size = settings.prefilter_size;
        let faces = std::array::from_fn(|face_index| {
            let face = cube_face(face_index);
            (0..size * size)
                .map(|index| {
                    let normal = direction_for_texel(face, index % size, index / size, size);
                    let view = normal;
                    let (tangent, bitangent) = basis(normal);
                    let mut sum = Vec3::ZERO;
                    let mut weight_sum = 0.0;
                    for sample_index in 0..settings.prefilter_samples {
                        let xi = hammersley(sample_index, settings.prefilter_samples);
                        let half_vector =
                            importance_sample_ggx(xi, roughness, normal, tangent, bitangent);
                        let light =
                            (half_vector * (2.0 * view.dot(half_vector)) - view).normalize();
                        let n_dot_l = normal.dot(light).max(0.0);
                        if n_dot_l > 0.0 {
                            sum = sum + sample(light) * n_dot_l;
                            weight_sum += n_dot_l;
                        }
                    }
                    if weight_sum > 0.0 {
                        sum / weight_sum
                    } else {
                        sample(normal)
                    }
                })
                .collect()
        });
        levels.push(FloatCube::new(size, faces));
    }
    PrefilteredEnvironment { levels }
}

fn bake_brdf_lut(settings: IblSettings) -> EnvironmentBrdfLut {
    let size = settings.brdf_size;
    let pixels = (0..size * size)
        .map(|index| {
            let x = index % size;
            let y = index / size;
            let n_dot_v = if size == 1 {
                1.0
            } else {
                x as f32 / (size - 1) as f32
            };
            let roughness = if size == 1 {
                0.0
            } else {
                y as f32 / (size - 1) as f32
            };
            integrate_brdf(n_dot_v, roughness, settings.brdf_samples)
        })
        .collect();
    EnvironmentBrdfLut { size, pixels }
}

fn integrate_brdf(n_dot_v: f32, roughness: f32, sample_count: usize) -> Vec2 {
    let n_dot_v = n_dot_v.clamp(0.0, 1.0);
    let view = Vec3::new((1.0 - n_dot_v * n_dot_v).sqrt(), 0.0, n_dot_v);
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let tangent = Vec3::new(1.0, 0.0, 0.0);
    let bitangent = Vec3::new(0.0, 1.0, 0.0);
    let mut result = Vec2::ZERO;
    for sample_index in 0..sample_count {
        let half_vector = importance_sample_ggx(
            hammersley(sample_index, sample_count),
            roughness,
            normal,
            tangent,
            bitangent,
        );
        let light = (half_vector * (2.0 * view.dot(half_vector)) - view).normalize();
        let n_dot_l = light.z.max(0.0);
        let n_dot_h = half_vector.z.max(0.0);
        let v_dot_h = view.dot(half_vector).max(0.0);
        if n_dot_l > 0.0 && n_dot_h > 0.0 && n_dot_v > 0.0 {
            let visibility = smith_visibility(n_dot_l, n_dot_v, roughness) * v_dot_h
                / (n_dot_h * n_dot_v).max(1.0e-6);
            let fresnel = (1.0 - v_dot_h).powi(5);
            result = result + Vec2::new((1.0 - fresnel) * visibility, fresnel * visibility);
        }
    }
    result / sample_count as f32
}

fn smith_visibility(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    // Karis's IBL remap uses alpha = roughness^2, unlike the direct-light
    // remap. This keeps grazing-angle split-sum energy from getting too dark.
    let k = roughness.clamp(0.0, 1.0).powi(2) * 0.5;
    let one = |n_dot: f32| n_dot / (n_dot * (1.0 - k) + k).max(1.0e-6);
    one(n_dot_l) * one(n_dot_v)
}

fn importance_sample_ggx(
    xi: Vec2,
    roughness: f32,
    normal: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
) -> Vec3 {
    // Roughness is perceptual. GGX alpha is roughness squared, then the NDF
    // formula uses alpha squared again.
    let alpha = roughness.clamp(0.0, 1.0).max(MIN_ROUGHNESS).powi(2);
    let alpha_squared = alpha * alpha;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (alpha_squared - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    (tangent * (sin_theta * phi.cos()) + bitangent * (sin_theta * phi.sin()) + normal * cos_theta)
        .normalize()
}

fn hammersley(index: usize, count: usize) -> Vec2 {
    Vec2::new(
        (index as f32 + 0.5) / count as f32,
        radical_inverse_vdc(index as u32),
    )
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_left(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

fn direction_for_texel(face: CubeFace, x: usize, y: usize, size: usize) -> Vec3 {
    let u = ((x as f32 + 0.5) / size as f32) * 2.0 - 1.0;
    let v = ((y as f32 + 0.5) / size as f32) * 2.0 - 1.0;
    match face {
        CubeFace::PositiveX => Vec3::new(1.0, -v, -u),
        CubeFace::NegativeX => Vec3::new(-1.0, -v, u),
        CubeFace::PositiveY => Vec3::new(u, 1.0, v),
        CubeFace::NegativeY => Vec3::new(u, -1.0, -v),
        CubeFace::PositiveZ => Vec3::new(u, -v, 1.0),
        CubeFace::NegativeZ => Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

fn basis(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.z.abs() < 0.999 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let tangent = up.cross(normal).normalize();
    let bitangent = normal.cross(tangent).normalize();
    (tangent, bitangent)
}

fn rgb(sample: [f32; 4]) -> Vec3 {
    Vec3::new(sample[0], sample[1], sample[2])
}

fn cube_face(index: usize) -> CubeFace {
    match index {
        0 => CubeFace::PositiveX,
        1 => CubeFace::NegativeX,
        2 => CubeFace::PositiveY,
        3 => CubeFace::NegativeY,
        4 => CubeFace::PositiveZ,
        _ => CubeFace::NegativeZ,
    }
}

fn face_index(face: CubeFace) -> usize {
    match face {
        CubeFace::PositiveX => 0,
        CubeFace::NegativeX => 1,
        CubeFace::PositiveY => 2,
        CubeFace::NegativeY => 3,
        CubeFace::PositiveZ => 4,
        CubeFace::NegativeZ => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{ColorSpace, Texture};

    fn constant_environment(color: [u8; 4]) -> CubeTexture {
        CubeTexture::new(std::array::from_fn(|_| {
            Texture::new_with_color_space(4, 4, vec![color; 16], ColorSpace::Linear).unwrap()
        }))
        .unwrap()
    }

    #[test]
    fn constant_environment_irradiance_is_constant() {
        let maps = IblMaps::from_environment_with_settings(
            &constant_environment([64, 128, 255, 255]),
            IblSettings::test(),
        );
        let expected = Vec3::new(64.0 / 255.0, 128.0 / 255.0, 1.0);
        for direction in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let actual = maps.irradiance.sample(direction);
            assert!((actual.x - expected.x).abs() < 1.0e-5);
            assert!((actual.y - expected.y).abs() < 1.0e-5);
            assert!((actual.z - expected.z).abs() < 1.0e-5);
        }
    }

    #[test]
    fn constant_environment_prefilter_mip_zero_is_constant() {
        let maps = IblMaps::from_environment_with_settings(
            &constant_environment([32, 96, 200, 255]),
            IblSettings::test(),
        );
        let expected = Vec3::new(32.0 / 255.0, 96.0 / 255.0, 200.0 / 255.0);
        let mip_zero = maps.prefiltered.level(0).unwrap();
        let actual = mip_zero.sample(Vec3::new(0.2, -0.8, 0.5));
        assert!((actual.x - expected.x).abs() < 1.0e-5);
        assert!((actual.y - expected.y).abs() < 1.0e-5);
        assert!((actual.z - expected.z).abs() < 1.0e-5);
    }

    #[test]
    fn karis_brdf_normal_incidence_at_zero_roughness_is_one_zero() {
        let maps = IblMaps::from_environment_with_settings(
            &constant_environment([255, 255, 255, 255]),
            IblSettings::test(),
        );
        let anchor = maps.brdf_lut.sample(1.0, 0.0);
        assert!((anchor.x - 1.0).abs() < 1.0e-5);
        assert!(anchor.y.abs() < 1.0e-5);
    }

    #[test]
    fn karis_prefilter_uses_squared_perceptual_roughness() {
        let half_vector = importance_sample_ggx(
            Vec2::new(0.5, 0.5),
            0.5,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!((half_vector.z - 0.9701425).abs() < 1.0e-5);
    }

    #[test]
    fn karis_ibl_visibility_anchor_at_grazing_zero_roughness() {
        let anchor = integrate_brdf(0.5, 0.0, 64);
        assert!((anchor.x - 0.96875).abs() < 1.0e-5);
        assert!((anchor.y - 0.03125).abs() < 1.0e-5);
    }

    #[test]
    fn two_hemisphere_irradiance_probe_matches_analytic_values() {
        let top_white = |direction: Vec3| {
            if direction.z > 0.0 {
                Vec3::new(1.0, 1.0, 1.0)
            } else {
                Vec3::ZERO
            }
        };
        let tilted = Vec3::new(0.57735026, 0.0, 0.8164966);
        let actual_tilted = integrate_irradiance(top_white, tilted, 1_048_576).x;
        let expected_tilted = 0.9082483;
        assert!((actual_tilted - expected_tilted).abs() < 2.0e-5);
        let actual_up = integrate_irradiance(top_white, Vec3::new(0.0, 0.0, 1.0), 64).x;
        assert!((actual_up - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn bake_is_byte_stable_for_the_same_environment() {
        let environment = constant_environment([17, 83, 191, 255]);
        let first = IblMaps::from_environment_with_settings(&environment, IblSettings::test());
        let second = IblMaps::from_environment_with_settings(&environment, IblSettings::test());
        assert_eq!(first, second);
    }

    #[test]
    fn floating_point_source_preserves_hdr_values_above_one() {
        let source = FloatCube::constant(2, Vec3::new(2.0, 1.5, 4.0));
        let maps = IblMaps::from_float_environment(&source, IblSettings::test());
        let actual = maps.irradiance.sample(Vec3::new(0.0, 0.0, 1.0));
        assert!((actual.x - 2.0).abs() < 1.0e-5);
        assert!((actual.y - 1.5).abs() < 1.0e-5);
        assert!((actual.z - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn cube_texture_float_decode_applies_linear_hdr_intensity() {
        let source = constant_environment([128, 128, 128, 255]);
        let float_source = FloatCube::from_cube_texture(&source, 2.0);
        let sample = float_source.sample(Vec3::new(1.0, 0.0, 0.0));
        let expected = 2.0 * 128.0 / 255.0;
        assert!((sample.x - expected).abs() < 1.0e-5);
        assert!((sample.y - expected).abs() < 1.0e-5);
        assert!((sample.z - expected).abs() < 1.0e-5);
    }
}
