use crate::camera::Camera;
use crate::math::{Mat4, Quat, Vec3, Vec4};
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneError(pub String);

impl Display for SceneError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SceneError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraConfig {
    pub position: Vec3,
    pub target: Vec3,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl CameraConfig {
    pub fn camera(self, aspect: f32) -> Camera {
        let direction = (self.target - self.position).normalize();
        let yaw = (-direction.x).atan2(-direction.z);
        let pitch = (-direction.y).clamp(-1.0, 1.0).asin();
        let orientation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), yaw)
            * Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -pitch);
        Camera::new(
            self.position,
            orientation,
            self.fov,
            aspect,
            self.near,
            self.far,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SkyboxConfig {
    pub px: PathBuf,
    pub nx: PathBuf,
    pub py: PathBuf,
    pub ny: PathBuf,
    pub pz: PathBuf,
    pub nz: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IblConfig {
    pub intensity: f32,
    pub irradiance_size: usize,
    pub prefilter_size: usize,
    pub prefilter_levels: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentConfig {
    pub background: Vec4,
    pub skybox: Option<SkyboxConfig>,
    pub ibl: Option<IblConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightType {
    Directional,
    Point,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowType {
    Basic,
    Csm,
    Pcss,
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowConfig {
    pub kind: ShadowType,
    pub map_size: usize,
    pub cascades: usize,
    pub lambda: f32,
    pub light_size: f32,
    pub bias: f32,
    pub slope_bias: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightConfig {
    pub kind: LightType,
    pub direction: Vec3,
    pub position: Vec3,
    pub color: Vec3,
    pub constant_attenuation: f32,
    pub linear_attenuation: f32,
    pub quadratic_attenuation: f32,
    pub shadow: Option<ShadowConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialType {
    BlinnPhong,
    Ggx,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialConfig {
    pub kind: MaterialType,
    pub ambient: Vec3,
    pub diffuse: Vec3,
    pub specular: Vec3,
    pub shininess: f32,
    pub alpha: f32,
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for MaterialConfig {
    fn default() -> Self {
        Self {
            kind: MaterialType::BlinnPhong,
            ambient: Vec3::new(0.03, 0.03, 0.03),
            diffuse: Vec3::new(0.75, 0.75, 0.75),
            specular: Vec3::new(0.2, 0.2, 0.2),
            shininess: 24.0,
            alpha: 1.0,
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformConfig {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

impl TransformConfig {
    pub fn matrix(self) -> Mat4 {
        Mat4::translate(self.position)
            * Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), self.rotation.y)
            * Mat4::rotate(Vec3::new(1.0, 0.0, 0.0), self.rotation.x)
            * Mat4::rotate(Vec3::new(0.0, 0.0, 1.0), self.rotation.z)
            * Mat4::scale(self.scale)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstanceSource {
    Transforms(Vec<TransformConfig>),
    Grid {
        dimensions: [usize; 3],
        spacing: Vec3,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstancingConfig {
    pub count: usize,
    pub source: InstanceSource,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LodConfig {
    pub ratios: Vec<f32>,
    pub thresholds: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectConfig {
    pub mesh: PathBuf,
    pub material: MaterialConfig,
    pub transform: TransformConfig,
    pub instancing: Option<InstancingConfig>,
    pub lod: Option<LodConfig>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PostFxConfig {
    Ssao {
        radius: f32,
        bias: f32,
        strength: f32,
        range: f32,
        blur_depth_threshold: f32,
    },
    Dof {
        focus_distance: f32,
        aperture: f32,
        max_coc_radius: f32,
    },
    Bloom,
    Fxaa,
    Vignette,
    Aces {
        exposure: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleConfig {
    pub position: Vec3,
    pub emission_rate: usize,
    pub lifetime_steps: usize,
    pub initial_velocity: Vec3,
    pub velocity_variation: Vec3,
    pub gravity: Vec3,
    pub drag: f32,
    pub capacity: usize,
    /// Number of fixed timesteps to advance before the screenshot renders.
    /// A single-frame `scene_viewer --screenshot` otherwise sees the emitter
    /// mid-emission-cycle; warming past `2 * lifetime_steps` gives the steady
    /// state a fountain would show.
    pub warmup_steps: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HudLine {
    pub text: String,
    pub x: i32,
    pub y: i32,
    pub scale: usize,
    pub color: Vec4,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub camera: CameraConfig,
    pub environment: EnvironmentConfig,
    pub lights: Vec<LightConfig>,
    pub objects: Vec<ObjectConfig>,
    pub postfx: Vec<PostFxConfig>,
    pub particles: Vec<ParticleConfig>,
    pub hud: Vec<HudLine>,
}
