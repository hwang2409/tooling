//! Strict, data-driven scene loading for the software renderer.
//!
//! The scene format deliberately uses the local JSON parser.  It keeps scene
//! order in vectors, rejects unknown keys, and reports every error path.

use crate::camera::Camera;
use crate::csm::{CascadeShadowConfig, render_cascade_shadow_maps_with_config};
use crate::fb::{Framebuffer, argb8888};
use crate::gltf::GltfAsset;
use crate::ibl::{FloatCube, IblMaps, IblSettings};
use crate::json::Value;
use crate::math::{Mat4, Quat, Vec3, Vec4};
use crate::mesh::{LodMesh, Mesh};
use crate::particles::{ParticleEmitter, ParticleSystem, billboard_quad};
use crate::pipeline::{
    FragmentStage, Instance, InstanceUniforms, Pipeline, VertexOutput, VertexStage,
};
use crate::postfx::{
    AcesTonemapPass, BloomPass, DofPass, FxaaPass, PostChain, SsaoPass, VignettePass,
};
use crate::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, BlinnPhongVaryings, CookTorranceShader,
    CookTorranceUniforms, DirectionalLight, IblCookTorranceShader, IblCookTorranceUniforms,
    PointLight, TextureFilter, TexturedShader, TexturedUniforms,
};
use crate::shadow::{CubeShadowState, render_cube_shadow_map};
use crate::skybox::CubeTexture;
use std::f32::consts::PI;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

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

#[derive(Clone)]
enum SceneUniforms<'a> {
    Blinn(BlinnPhongUniforms),
    Ggx(CookTorranceUniforms),
    IblGgx(IblCookTorranceUniforms<'a>),
}

#[derive(Clone, Copy, Debug, Default)]
struct SceneShader;

impl<'a> VertexStage<crate::mesh::MeshVertex, SceneUniforms<'a>> for SceneShader {
    type Varyings = BlinnPhongVaryings;

    fn run(
        &self,
        vertex: &crate::mesh::MeshVertex,
        uniforms: &SceneUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        match uniforms {
            SceneUniforms::Blinn(value) => VertexStage::run(&BlinnPhongShader, vertex, value),
            SceneUniforms::Ggx(value) => VertexStage::run(&CookTorranceShader, vertex, value),
            SceneUniforms::IblGgx(value) => VertexStage::run(&IblCookTorranceShader, vertex, value),
        }
    }
}

impl<'a> FragmentStage<BlinnPhongVaryings, SceneUniforms<'a>> for SceneShader {
    fn run(&self, varyings: &BlinnPhongVaryings, uniforms: &SceneUniforms<'a>) -> u32 {
        match uniforms {
            SceneUniforms::Blinn(value) => FragmentStage::run(&BlinnPhongShader, varyings, value),
            SceneUniforms::Ggx(value) => FragmentStage::run(&CookTorranceShader, varyings, value),
            SceneUniforms::IblGgx(value) => {
                FragmentStage::run(&IblCookTorranceShader, varyings, value)
            }
        }
    }

    fn run_linear(&self, varyings: &BlinnPhongVaryings, uniforms: &SceneUniforms<'a>) -> [f32; 4] {
        match uniforms {
            SceneUniforms::Blinn(value) => BlinnPhongShader.run_linear(varyings, value),
            SceneUniforms::Ggx(value) => CookTorranceShader.run_linear(varyings, value),
            SceneUniforms::IblGgx(value) => IblCookTorranceShader.run_linear(varyings, value),
        }
    }

    fn is_opaque(&self, uniforms: &SceneUniforms<'a>) -> bool {
        match uniforms {
            SceneUniforms::Blinn(value) => BlinnPhongShader.is_opaque(value),
            SceneUniforms::Ggx(value) => CookTorranceShader.is_opaque(value),
            SceneUniforms::IblGgx(value) => IblCookTorranceShader.is_opaque(value),
        }
    }

    fn model_view(&self, uniforms: &SceneUniforms<'a>) -> Option<Mat4> {
        match uniforms {
            SceneUniforms::Blinn(value) => BlinnPhongShader.model_view(value),
            SceneUniforms::Ggx(value) => CookTorranceShader.model_view(value),
            SceneUniforms::IblGgx(value) => IblCookTorranceShader.model_view(value),
        }
    }

    fn culling_transform(&self, uniforms: &SceneUniforms<'a>) -> Option<Mat4> {
        match uniforms {
            SceneUniforms::Blinn(value) => BlinnPhongShader.culling_transform(value),
            SceneUniforms::Ggx(value) => CookTorranceShader.culling_transform(value),
            SceneUniforms::IblGgx(value) => IblCookTorranceShader.culling_transform(value),
        }
    }
}

impl<'a> InstanceUniforms for SceneUniforms<'a> {
    fn set_instance_model(&mut self, model: Mat4) {
        match self {
            SceneUniforms::Blinn(value) => value.set_model(model),
            SceneUniforms::Ggx(value) => value.lighting.set_model(model),
            SceneUniforms::IblGgx(value) => value.lighting.lighting.set_model(model),
        }
    }

    fn apply_instance_tint(&mut self, tint: Vec4) {
        match self {
            SceneUniforms::Blinn(value) => {
                value.set_diffuse_color(value.diffuse_color() * tint_rgb(tint));
            }
            SceneUniforms::Ggx(value) => {
                value.set_base_color_linear(value.base_color() * tint_rgb(tint));
            }
            SceneUniforms::IblGgx(value) => {
                let base = value.base_color() * tint_rgb(tint);
                value.lighting.set_base_color_linear(base);
            }
        }
    }

    fn for_instance(&self, instance: &Instance) -> Self {
        match self {
            SceneUniforms::Blinn(value) => SceneUniforms::Blinn(value.for_instance(instance)),
            SceneUniforms::Ggx(value) => {
                let mut result = value.clone();
                result.lighting.set_model(instance.model());
                if let Some(tint) = instance.tint() {
                    result.set_base_color_linear(result.base_color() * tint_rgb(tint));
                    result.set_alpha(result.lighting.alpha * tint.w);
                }
                SceneUniforms::Ggx(result)
            }
            SceneUniforms::IblGgx(value) => {
                let mut result = value.clone();
                result.lighting.lighting.set_model(instance.model());
                if let Some(tint) = instance.tint() {
                    result
                        .lighting
                        .set_base_color_linear(result.base_color() * tint_rgb(tint));
                    result
                        .lighting
                        .set_alpha(result.lighting.lighting.alpha * tint.w);
                }
                SceneUniforms::IblGgx(result)
            }
        }
    }
}

impl<'a> SceneUniforms<'a> {
    fn set_directional_cascaded_shadow(
        &mut self,
        light: DirectionalLight,
        cascades: crate::shadow::CascadeShadowState,
    ) {
        match self {
            SceneUniforms::Blinn(value) => value.set_directional_cascaded_shadow(light, cascades),
            SceneUniforms::Ggx(value) => value
                .lighting
                .set_directional_cascaded_shadow(light, cascades),
            SceneUniforms::IblGgx(value) => value
                .lighting
                .lighting
                .set_directional_cascaded_shadow(light, cascades),
        }
    }

    fn set_point_shadow(&mut self, index: usize, shadow: CubeShadowState) {
        match self {
            SceneUniforms::Blinn(value) => {
                let _ = value.set_point_light_shadow(index, Some(shadow));
            }
            SceneUniforms::Ggx(value) => {
                let _ = value.lighting.set_point_light_shadow(index, Some(shadow));
            }
            SceneUniforms::IblGgx(value) => {
                let _ = value
                    .lighting
                    .lighting
                    .set_point_light_shadow(index, Some(shadow));
            }
        }
    }
}

impl Scene {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(source: &str) -> Result<Self, SceneError> {
        let value =
            crate::json::parse(source).map_err(|error| SceneError(format!("scene: {error}")))?;
        parse_scene(&value)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, SceneError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path)
            .map_err(|error| SceneError(format!("scene: {}: {error}", path.display())))?;
        let scene = Self::from_str(&source)?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        scene.validate_assets(root)?;
        Ok(scene)
    }

    pub fn validate_assets(&self, root: impl AsRef<Path>) -> Result<(), SceneError> {
        let root = root.as_ref();
        for (index, object) in self.objects.iter().enumerate() {
            let path = root.join(&object.mesh);
            if !path.is_file() {
                return Err(SceneError(format!(
                    "objects[{index}].mesh: asset does not exist: {}",
                    path.display()
                )));
            }
        }
        if let Some(skybox) = &self.environment.skybox {
            for (name, path) in [
                ("px", &skybox.px),
                ("nx", &skybox.nx),
                ("py", &skybox.py),
                ("ny", &skybox.ny),
                ("pz", &skybox.pz),
                ("nz", &skybox.nz),
            ] {
                if !root.join(path).is_file() {
                    return Err(SceneError(format!(
                        "environment.skybox.{name}: asset does not exist: {}",
                        root.join(path).display()
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn post_chain(&self, projection: Mat4) -> PostChain {
        let mut chain = PostChain::new();
        for pass in &self.postfx {
            match *pass {
                PostFxConfig::Ssao {
                    radius,
                    bias,
                    strength,
                    range,
                    blur_depth_threshold,
                } => {
                    let mut value = SsaoPass::new(projection);
                    value.set_radius(radius);
                    value.set_bias(bias);
                    value.set_strength(strength);
                    value.set_range(range);
                    value.set_blur_depth_threshold(blur_depth_threshold);
                    chain.push(value);
                }
                PostFxConfig::Dof {
                    focus_distance,
                    aperture,
                    max_coc_radius,
                } => {
                    let mut value = DofPass::new(projection);
                    value.set_focus_distance(focus_distance);
                    value.set_aperture(aperture);
                    value.set_max_coc_radius(max_coc_radius);
                    chain.push(value);
                }
                PostFxConfig::Bloom => chain.push(BloomPass),
                PostFxConfig::Fxaa => chain.push(FxaaPass),
                PostFxConfig::Vignette => chain.push(VignettePass),
                PostFxConfig::Aces { exposure } => chain.push(AcesTonemapPass::new(exposure)),
            }
        }
        chain
    }

    /// Renders objects in file order through one production pipeline.
    pub fn render(
        &self,
        framebuffer: &mut Framebuffer,
        asset_root: impl AsRef<Path>,
    ) -> Result<(), SceneError> {
        let root = asset_root.as_ref();
        self.validate_assets(root)?;
        let camera = self
            .camera
            .camera(framebuffer.width.max(1) as f32 / framebuffer.height.max(1) as f32);
        let background = self.environment.background;
        framebuffer.clear(argb8888(
            channel(background.w),
            channel(background.x),
            channel(background.y),
            channel(background.z),
        ));
        let directional = self
            .lights
            .iter()
            .find(|light| light.kind == LightType::Directional)
            .copied();
        let point = self
            .lights
            .iter()
            .find(|light| light.kind == LightType::Point)
            .copied();
        let skybox = load_skybox(&self.environment, root)?;
        let ibl_maps = build_ibl_maps(&self.environment, skybox.as_ref());
        let mut meshes = Vec::with_capacity(self.objects.len());
        let mut lods: Vec<Option<LodMesh>> = Vec::with_capacity(self.objects.len());
        let mut uniforms: Vec<SceneUniforms<'_>> = Vec::with_capacity(self.objects.len());
        let mut instances: Vec<Option<Vec<Instance>>> = Vec::with_capacity(self.objects.len());
        for (object_index, object) in self.objects.iter().enumerate() {
            let path = root.join(&object.mesh);
            let mesh = load_mesh(&path, &format!("objects[{object_index}].mesh"))?;
            let model = object.transform.matrix();
            let directional_light = directional.map_or_else(
                || DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO),
                |light| DirectionalLight::new(light.direction, light.color),
            );
            let point_light = point.map_or_else(
                || PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
                |light| {
                    PointLight::new(
                        light.position,
                        light.color,
                        light.constant_attenuation,
                        light.linear_attenuation,
                        light.quadratic_attenuation,
                    )
                },
            );
            let mut lighting = BlinnPhongUniforms::new(
                model,
                camera.view_matrix(),
                camera.projection_matrix(),
                object.material.ambient,
                object.material.diffuse,
                object.material.specular,
                object.material.shininess,
                camera.position,
                directional_light,
                point_light,
            );
            lighting.set_alpha(object.material.alpha);
            lighting.clear_directional_lights();
            lighting.clear_point_lights();
            for light in &self.lights {
                match light.kind {
                    LightType::Directional => {
                        let _ = lighting.add_directional_light(DirectionalLight::new(
                            light.direction,
                            light.color,
                        ));
                    }
                    LightType::Point => {
                        let _ = lighting.add_point_light(PointLight::new(
                            light.position,
                            light.color,
                            light.constant_attenuation,
                            light.linear_attenuation,
                            light.quadratic_attenuation,
                        ));
                    }
                }
            }
            let object_instances = object
                .instancing
                .as_ref()
                .map(|config| build_instances(config, model));
            let lod = object.lod.as_ref().map(|config| {
                let mut value = LodMesh::with_ratios(mesh.clone(), &config.ratios);
                if !config.thresholds.is_empty() {
                    value.set_thresholds(config.thresholds.clone());
                }
                value
            });
            let value = match object.material.kind {
                MaterialType::BlinnPhong => SceneUniforms::Blinn(lighting),
                MaterialType::Ggx => match ibl_maps.as_ref() {
                    Some(maps) => SceneUniforms::IblGgx(IblCookTorranceUniforms::new(
                        lighting,
                        maps,
                        object.material.diffuse,
                        object.material.metallic,
                        object.material.roughness,
                    )),
                    None => SceneUniforms::Ggx(CookTorranceUniforms::new(
                        lighting,
                        object.material.diffuse,
                        object.material.metallic,
                        object.material.roughness,
                    )),
                },
            };
            meshes.push(mesh);
            lods.push(lod);
            uniforms.push(value);
            instances.push(object_instances);
        }
        let shadow_meshes: Vec<(&Mesh, Mat4)> = meshes
            .iter()
            .zip(self.objects.iter())
            .map(|(mesh, object)| (mesh, object.transform.matrix()))
            .collect();
        if let Some(light) = directional
            && let Some(shadow) = light.shadow
            && shadow.kind == ShadowType::Csm
        {
            let cascades = render_cascade_shadow_maps_with_config(
                camera,
                light.direction,
                CascadeShadowConfig::new(shadow.cascades, shadow.lambda),
                shadow.map_size.max(1),
                &shadow_meshes,
            )
            .map_err(|error| SceneError(format!("lights.directional.shadow: {error}")))?;
            let directional_light = DirectionalLight::new(light.direction, light.color);
            for value in &mut uniforms {
                value.set_directional_cascaded_shadow(directional_light, cascades.clone());
            }
        }
        if let Some((index, light)) = self
            .lights
            .iter()
            .enumerate()
            .find(|(_, light)| light.kind == LightType::Point)
            && let Some(shadow) = light.shadow
            && shadow.kind == ShadowType::Cube
        {
            let map = render_cube_shadow_map(
                light.position,
                shadow.near,
                shadow.far,
                shadow.map_size.max(1),
                &shadow_meshes,
            )
            .map_err(|error| SceneError(format!("lights[{index}].shadow: {error}")))?;
            let state = CubeShadowState::new(light.position, map);
            let point_index = self.lights[..index]
                .iter()
                .filter(|light| light.kind == LightType::Point)
                .count();
            for value in &mut uniforms {
                value.set_point_shadow(point_index, state.clone());
            }
        }
        let mut pipeline = Pipeline::new(SceneShader, SceneShader);
        pipeline.set_hdr(
            self.postfx
                .iter()
                .any(|pass| matches!(pass, PostFxConfig::Aces { .. })),
        );
        pipeline.set_post_chain(self.post_chain(camera.projection_matrix()));
        pipeline.render(framebuffer, |frame, target| {
            if let Some(cube) = skybox.as_ref() {
                frame.draw_skybox(target, cube, camera);
            }
            for index in 0..meshes.len() {
                if let Some(list) = instances[index].as_ref() {
                    frame.draw_mesh_instanced(target, &meshes[index], &uniforms[index], list);
                } else if let Some(lod) = lods[index].as_ref() {
                    frame.draw_lod_mesh(
                        target,
                        lod,
                        &uniforms[index],
                        camera,
                        camera.projection_matrix(),
                        scene_uniform_model(&uniforms[index]),
                    );
                } else {
                    frame.draw_mesh(target, &meshes[index], &uniforms[index]);
                }
            }
        });
        render_particles(self, framebuffer, camera)?;
        for line in &self.hud {
            framebuffer.draw_text(
                line.x,
                line.y,
                &line.text,
                line.scale,
                argb8888(
                    channel(line.color.w),
                    channel(line.color.x),
                    channel(line.color.y),
                    channel(line.color.z),
                ),
            );
        }
        Ok(())
    }
}

fn render_particles(
    scene: &Scene,
    framebuffer: &mut Framebuffer,
    camera: Camera,
) -> Result<(), SceneError> {
    if scene.particles.is_empty() {
        return Ok(());
    }
    let texture = crate::image::Texture::new(1, 1, vec![[255, 255, 255, 220]])
        .map_err(|error| SceneError(format!("particles.texture: {error}")))?;
    let quad = billboard_quad();
    let mut draws = Vec::with_capacity(scene.particles.len());
    for config in &scene.particles {
        let mut emitter = ParticleEmitter::new(
            config.position,
            config.emission_rate,
            config.lifetime_steps,
            config.initial_velocity,
        );
        emitter.set_velocity_variation(config.velocity_variation);
        emitter.set_gravity(config.gravity);
        emitter.set_drag(config.drag);
        let mut system = ParticleSystem::new(emitter, config.capacity);
        system.step();
        let instances = system.billboard_instances(camera, 0.16, Vec4::new(0.75, 0.9, 1.0, 0.9));
        let uniforms = TexturedUniforms::new(
            camera.projection_matrix() * camera.view_matrix(),
            &texture,
            TextureFilter::Bilinear,
        );
        draws.push((instances, uniforms));
    }
    let mut pipeline = Pipeline::new(TexturedShader, TexturedShader);
    pipeline.render(framebuffer, |frame, target| {
        for (instances, uniforms) in &draws {
            frame.draw_mesh_instanced_with_sampling(target, &quad, uniforms, instances);
        }
    });
    Ok(())
}

impl std::str::FromStr for Scene {
    type Err = SceneError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Scene::from_str(source)
    }
}

fn build_instances(config: &InstancingConfig, model: Mat4) -> Vec<Instance> {
    let mut result = Vec::with_capacity(config.count);
    match &config.source {
        InstanceSource::Transforms(transforms) => {
            for transform in transforms.iter().take(config.count) {
                result.push(Instance::new(model * transform.matrix()));
            }
        }
        InstanceSource::Grid {
            dimensions,
            spacing,
        } => {
            for z in 0..dimensions[2] {
                for y in 0..dimensions[1] {
                    for x in 0..dimensions[0] {
                        if result.len() == config.count {
                            return result;
                        }
                        let offset = Vec3::new(
                            x as f32 * spacing.x,
                            y as f32 * spacing.y,
                            z as f32 * spacing.z,
                        );
                        result.push(Instance::new(model * Mat4::translate(offset)));
                    }
                }
            }
        }
    }
    result
}

fn load_mesh(path: &Path, error_path: &str) -> Result<Mesh, SceneError> {
    if path
        .extension()
        .is_some_and(|extension| extension == "gltf")
    {
        let asset =
            GltfAsset::load(path).map_err(|error| SceneError(format!("{error_path}: {error}")))?;
        return asset
            .meshes
            .first()
            .and_then(|mesh| mesh.primitives.first())
            .map(|primitive| primitive.mesh.clone())
            .ok_or_else(|| SceneError(format!("{error_path}: glTF contains no mesh primitives")));
    }
    Mesh::load(path).map_err(|error| SceneError(format!("{error_path}: {error}")))
}

fn load_skybox(
    environment: &EnvironmentConfig,
    root: &Path,
) -> Result<Option<CubeTexture>, SceneError> {
    let Some(config) = environment.skybox.as_ref() else {
        return Ok(None);
    };
    let faces = [
        &config.px, &config.nx, &config.py, &config.ny, &config.pz, &config.nz,
    ]
    .iter()
    .map(|path| {
        crate::image::Texture::load(root.join(path))
            .map_err(|error| SceneError(format!("environment.skybox: {error}")))
    })
    .collect::<Result<Vec<_>, _>>()?;
    let faces: [crate::image::Texture; 6] = faces
        .try_into()
        .map_err(|_| SceneError("environment.skybox: expected six faces".to_string()))?;
    Ok(Some(CubeTexture::new(faces).map_err(|error| {
        SceneError(format!("environment.skybox: {error}"))
    })?))
}

fn build_ibl_maps(
    environment: &EnvironmentConfig,
    skybox: Option<&CubeTexture>,
) -> Option<IblMaps> {
    let config = environment.ibl?;
    let cube = skybox?;
    let defaults = IblSettings::default();
    let source = FloatCube::from_cube_texture(cube, config.intensity);
    Some(IblMaps::from_float_environment(
        &source,
        IblSettings {
            irradiance_size: config.irradiance_size,
            irradiance_samples: defaults.irradiance_samples,
            prefilter_size: config.prefilter_size,
            prefilter_levels: config.prefilter_levels,
            prefilter_samples: defaults.prefilter_samples,
            brdf_size: defaults.brdf_size,
            brdf_samples: defaults.brdf_samples,
        },
    ))
}

const fn tint_rgb(tint: Vec4) -> Vec3 {
    Vec3::new(tint.x, tint.y, tint.z)
}

fn scene_uniform_model(uniforms: &SceneUniforms<'_>) -> Mat4 {
    match uniforms {
        SceneUniforms::Blinn(value) => value.model(),
        SceneUniforms::Ggx(value) => value.lighting.model(),
        SceneUniforms::IblGgx(value) => value.lighting.lighting.model(),
    }
}

fn channel(value: f32) -> u8 {
    (if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    } * 255.0)
        .round() as u8
}

fn parse_scene(value: &Value) -> Result<Scene, SceneError> {
    let mut object = Fields::new(
        value,
        "scene",
        &[
            "camera",
            "environment",
            "lights",
            "objects",
            "postfx",
            "particles",
            "hud",
        ],
    )?;
    let camera = parse_camera(object.required("camera")?)?;
    let environment = object
        .optional("environment")
        .map_or_else(|| Ok(default_environment()), parse_environment)?;
    let lights = parse_array(object.optional("lights"), "lights", parse_light)?;
    let objects = parse_array(object.optional("objects"), "objects", parse_object)?;
    let postfx = parse_array(object.optional("postfx"), "postfx", parse_postfx)?;
    let particles = parse_array(object.optional("particles"), "particles", parse_particle)?;
    let hud = parse_array(object.optional("hud"), "hud", parse_hud)?;
    Ok(Scene {
        camera,
        environment,
        lights,
        objects,
        postfx,
        particles,
        hud,
    })
}

fn default_environment() -> EnvironmentConfig {
    EnvironmentConfig {
        background: Vec4::new(0.015, 0.02, 0.04, 1.0),
        skybox: None,
        ibl: None,
    }
}

fn parse_camera(value: &Value) -> Result<CameraConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        "camera",
        &["position", "target", "fov", "near", "far"],
    )?;
    let position = vec3(fields.required("position")?, "camera.position")?;
    let target = vec3(fields.required("target")?, "camera.target")?;
    let fov = scalar(fields.required("fov")?, "camera.fov")?.clamp(0.01, PI - 0.01);
    let near = scalar(fields.required("near")?, "camera.near")?.max(0.0001);
    let far = scalar(fields.required("far")?, "camera.far")?.max(near + 0.0001);
    Ok(CameraConfig {
        position,
        target,
        fov,
        near,
        far,
    })
}

fn parse_environment(value: &Value) -> Result<EnvironmentConfig, SceneError> {
    let mut fields = Fields::new(value, "environment", &["background", "skybox", "ibl"])?;
    let background = vec4(
        fields.optional("background").unwrap_or(&Value::Array(vec![
            Value::Number(0.0),
            Value::Number(0.0),
            Value::Number(0.0),
            Value::Number(1.0),
        ])),
        "environment.background",
    )?;
    let skybox = fields.optional("skybox").map(parse_skybox).transpose()?;
    let ibl = fields.optional("ibl").map(parse_ibl).transpose()?;
    Ok(EnvironmentConfig {
        background,
        skybox,
        ibl,
    })
}

fn parse_skybox(value: &Value) -> Result<SkyboxConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        "environment.skybox",
        &["px", "nx", "py", "ny", "pz", "nz"],
    )?;
    Ok(SkyboxConfig {
        px: path_value(fields.required("px")?, "environment.skybox.px")?,
        nx: path_value(fields.required("nx")?, "environment.skybox.nx")?,
        py: path_value(fields.required("py")?, "environment.skybox.py")?,
        ny: path_value(fields.required("ny")?, "environment.skybox.ny")?,
        pz: path_value(fields.required("pz")?, "environment.skybox.pz")?,
        nz: path_value(fields.required("nz")?, "environment.skybox.nz")?,
    })
}

fn parse_ibl(value: &Value) -> Result<IblConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        "environment.ibl",
        &[
            "intensity",
            "irradiance_size",
            "prefilter_size",
            "prefilter_levels",
        ],
    )?;
    Ok(IblConfig {
        intensity: optional_scalar(&mut fields, "intensity", 1.0)?.max(0.0),
        irradiance_size: optional_usize(&mut fields, "irradiance_size", 16)?.max(1),
        prefilter_size: optional_usize(&mut fields, "prefilter_size", 32)?.max(1),
        prefilter_levels: optional_usize(&mut fields, "prefilter_levels", 5)?.max(1),
    })
}

fn parse_light(value: &Value, path: &str) -> Result<LightConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "direction",
            "position",
            "color",
            "constant",
            "linear",
            "quadratic",
            "shadow",
        ],
    )?;
    let kind = match string(fields.required("type")?, &format!("{path}.type"))? {
        ref value if value == "directional" => LightType::Directional,
        ref value if value == "point" => LightType::Point,
        value => {
            return Err(SceneError(format!(
                "{path}.type: unknown light type {value}"
            )));
        }
    };
    let direction = optional_vec3(
        &mut fields,
        "direction",
        Vec3::new(0.0, -1.0, 0.0),
        &format!("{path}.direction"),
    )?;
    let position = optional_vec3(
        &mut fields,
        "position",
        Vec3::ZERO,
        &format!("{path}.position"),
    )?;
    let color = optional_vec3(
        &mut fields,
        "color",
        Vec3::new(1.0, 1.0, 1.0),
        &format!("{path}.color"),
    )?;
    let shadow = fields
        .optional("shadow")
        .map(|value| parse_shadow(value, &format!("{path}.shadow")))
        .transpose()?;
    Ok(LightConfig {
        kind,
        direction,
        position,
        color,
        constant_attenuation: optional_scalar(&mut fields, "constant", 1.0)?.max(0.0),
        linear_attenuation: optional_scalar(&mut fields, "linear", 0.0)?.max(0.0),
        quadratic_attenuation: optional_scalar(&mut fields, "quadratic", 0.0)?.max(0.0),
        shadow,
    })
}

fn parse_shadow(value: &Value, path: &str) -> Result<ShadowConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "map_size",
            "cascades",
            "lambda",
            "light_size",
            "bias",
            "slope_bias",
            "near",
            "far",
        ],
    )?;
    let kind = match string(fields.required("type")?, &format!("{path}.type"))? {
        ref value if value == "basic" => ShadowType::Basic,
        ref value if value == "csm" => ShadowType::Csm,
        ref value if value == "pcss" => ShadowType::Pcss,
        ref value if value == "cube" => ShadowType::Cube,
        value => {
            return Err(SceneError(format!(
                "{path}.type: unknown shadow type {value}"
            )));
        }
    };
    let near = optional_scalar(&mut fields, "near", 0.1)?.max(0.0001);
    Ok(ShadowConfig {
        kind,
        map_size: optional_usize(&mut fields, "map_size", 1024)?.max(1),
        cascades: optional_usize(&mut fields, "cascades", 3)?.clamp(2, 4),
        lambda: optional_scalar(&mut fields, "lambda", 0.5)?.clamp(0.0, 1.0),
        light_size: optional_scalar(&mut fields, "light_size", 0.0)?.max(0.0),
        bias: optional_scalar(&mut fields, "bias", 0.002)?.max(0.0),
        slope_bias: optional_scalar(&mut fields, "slope_bias", 0.02)?.max(0.0),
        near,
        far: optional_scalar(&mut fields, "far", 100.0)?.max(near + 0.0001),
    })
}

fn parse_object(value: &Value, path: &str) -> Result<ObjectConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &["mesh", "material", "transform", "instancing", "lod"],
    )?;
    let material = fields
        .optional("material")
        .map(|value| parse_material(value, &format!("{path}.material")))
        .transpose()?
        .unwrap_or_default();
    let transform = fields
        .optional("transform")
        .map(|value| parse_transform(value, &format!("{path}.transform")))
        .transpose()?
        .unwrap_or_default();
    Ok(ObjectConfig {
        mesh: path_value(fields.required("mesh")?, &format!("{path}.mesh"))?,
        material,
        transform,
        instancing: fields
            .optional("instancing")
            .map(|value| parse_instancing(value, &format!("{path}.instancing")))
            .transpose()?,
        lod: fields
            .optional("lod")
            .map(|value| parse_lod(value, &format!("{path}.lod")))
            .transpose()?,
    })
}

fn parse_material(value: &Value, path: &str) -> Result<MaterialConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "ambient",
            "diffuse",
            "specular",
            "shininess",
            "alpha",
            "metallic",
            "roughness",
        ],
    )?;
    let mut result = MaterialConfig::default();
    if let Some(value) = fields.optional("type") {
        result.kind = match string(value, &format!("{path}.type"))?.as_str() {
            "blinn_phong" => MaterialType::BlinnPhong,
            "ggx" => MaterialType::Ggx,
            other => {
                return Err(SceneError(format!(
                    "{path}.type: unknown material type {other}"
                )));
            }
        };
    }
    result.ambient = optional_vec3(
        &mut fields,
        "ambient",
        result.ambient,
        &format!("{path}.ambient"),
    )?;
    result.diffuse = optional_vec3(
        &mut fields,
        "diffuse",
        result.diffuse,
        &format!("{path}.diffuse"),
    )?;
    result.specular = optional_vec3(
        &mut fields,
        "specular",
        result.specular,
        &format!("{path}.specular"),
    )?;
    result.shininess = optional_scalar(&mut fields, "shininess", result.shininess)?.max(0.0);
    result.alpha = optional_scalar(&mut fields, "alpha", result.alpha)?.clamp(0.0, 1.0);
    result.metallic = optional_scalar(&mut fields, "metallic", result.metallic)?.clamp(0.0, 1.0);
    result.roughness =
        optional_scalar(&mut fields, "roughness", result.roughness)?.clamp(0.001, 1.0);
    Ok(result)
}

fn parse_transform(value: &Value, path: &str) -> Result<TransformConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["position", "rotation", "scale"])?;
    Ok(TransformConfig {
        position: optional_vec3(
            &mut fields,
            "position",
            Vec3::ZERO,
            &format!("{path}.position"),
        )?,
        rotation: optional_vec3(
            &mut fields,
            "rotation",
            Vec3::ZERO,
            &format!("{path}.rotation"),
        )?,
        scale: optional_vec3(
            &mut fields,
            "scale",
            Vec3::new(1.0, 1.0, 1.0),
            &format!("{path}.scale"),
        )?,
    })
}

fn parse_instancing(value: &Value, path: &str) -> Result<InstancingConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["count", "transforms", "grid"])?;
    let count = usize_value(fields.required("count")?, &format!("{path}.count"))?;
    let source = if let Some(value) = fields.optional("transforms") {
        InstanceSource::Transforms(parse_array(
            Some(value),
            &format!("{path}.transforms"),
            parse_transform_at,
        )?)
    } else if let Some(value) = fields.optional("grid") {
        let mut grid = Fields::new(value, &format!("{path}.grid"), &["dimensions", "spacing"])?;
        let dimensions = usize3(
            grid.required("dimensions")?,
            &format!("{path}.grid.dimensions"),
        )?;
        let spacing = vec3(grid.required("spacing")?, &format!("{path}.grid.spacing"))?;
        InstanceSource::Grid {
            dimensions,
            spacing,
        }
    } else {
        return Err(SceneError(format!("{path}: expected transforms or grid")));
    };
    Ok(InstancingConfig { count, source })
}

fn parse_lod(value: &Value, path: &str) -> Result<LodConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["ratios", "thresholds"])?;
    Ok(LodConfig {
        ratios: floats(fields.required("ratios")?, &format!("{path}.ratios"))?,
        thresholds: floats(
            fields.required("thresholds")?,
            &format!("{path}.thresholds"),
        )?,
    })
}

fn parse_postfx(value: &Value, path: &str) -> Result<PostFxConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "radius",
            "bias",
            "strength",
            "range",
            "blur_depth_threshold",
            "focus_distance",
            "aperture",
            "max_coc_radius",
            "exposure",
        ],
    )?;
    match string(fields.required("type")?, &format!("{path}.type"))?.as_str() {
        "ssao" => Ok(PostFxConfig::Ssao {
            radius: optional_scalar(&mut fields, "radius", 0.55)?,
            bias: optional_scalar(&mut fields, "bias", 0.025)?,
            strength: optional_scalar(&mut fields, "strength", 1.0)?,
            range: optional_scalar(&mut fields, "range", 0.9)?,
            blur_depth_threshold: optional_scalar(&mut fields, "blur_depth_threshold", 0.35)?,
        }),
        "dof" => Ok(PostFxConfig::Dof {
            focus_distance: optional_scalar(&mut fields, "focus_distance", 4.0)?,
            aperture: optional_scalar(&mut fields, "aperture", 6.0)?,
            max_coc_radius: optional_scalar(&mut fields, "max_coc_radius", 8.0)?,
        }),
        "bloom" => Ok(PostFxConfig::Bloom),
        "fxaa" => Ok(PostFxConfig::Fxaa),
        "vignette" => Ok(PostFxConfig::Vignette),
        "aces" => Ok(PostFxConfig::Aces {
            exposure: optional_scalar(&mut fields, "exposure", 1.0)?,
        }),
        other => Err(SceneError(format!(
            "{path}.type: unknown postfx pass {other}"
        ))),
    }
}

fn parse_particle(value: &Value, path: &str) -> Result<ParticleConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "position",
            "emission_rate",
            "lifetime_steps",
            "initial_velocity",
            "velocity_variation",
            "gravity",
            "drag",
            "capacity",
        ],
    )?;
    Ok(ParticleConfig {
        position: optional_vec3(
            &mut fields,
            "position",
            Vec3::ZERO,
            &format!("{path}.position"),
        )?,
        emission_rate: optional_usize(&mut fields, "emission_rate", 0)?,
        lifetime_steps: optional_usize(&mut fields, "lifetime_steps", 60)?.max(1),
        initial_velocity: optional_vec3(
            &mut fields,
            "initial_velocity",
            Vec3::ZERO,
            &format!("{path}.initial_velocity"),
        )?,
        velocity_variation: optional_vec3(
            &mut fields,
            "velocity_variation",
            Vec3::ZERO,
            &format!("{path}.velocity_variation"),
        )?,
        gravity: optional_vec3(
            &mut fields,
            "gravity",
            Vec3::ZERO,
            &format!("{path}.gravity"),
        )?,
        drag: optional_scalar(&mut fields, "drag", 0.0)?.clamp(0.0, 1.0),
        capacity: optional_usize(&mut fields, "capacity", 128)?,
    })
}

fn parse_hud(value: &Value, path: &str) -> Result<HudLine, SceneError> {
    let mut fields = Fields::new(value, path, &["text", "x", "y", "scale", "color"])?;
    Ok(HudLine {
        text: string(fields.required("text")?, &format!("{path}.text"))?,
        x: integer(fields.required("x")?, &format!("{path}.x"))?,
        y: integer(fields.required("y")?, &format!("{path}.y"))?,
        scale: optional_usize(&mut fields, "scale", 1)?.max(1),
        color: vec4(fields.required("color")?, &format!("{path}.color"))?,
    })
}

fn parse_transform_at(value: &Value, path: &str) -> Result<TransformConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["position", "rotation", "scale"])?;
    Ok(TransformConfig {
        position: optional_vec3(
            &mut fields,
            "position",
            Vec3::ZERO,
            &format!("{path}.position"),
        )?,
        rotation: optional_vec3(
            &mut fields,
            "rotation",
            Vec3::ZERO,
            &format!("{path}.rotation"),
        )?,
        scale: optional_vec3(
            &mut fields,
            "scale",
            Vec3::new(1.0, 1.0, 1.0),
            &format!("{path}.scale"),
        )?,
    })
}

struct Fields<'a> {
    path: String,
    values: Vec<(String, &'a Value)>,
}

impl<'a> Fields<'a> {
    fn new(value: &'a Value, path: &str, allowed: &[&str]) -> Result<Self, SceneError> {
        let Value::Object(values) = value else {
            return Err(SceneError(format!("{path}: expected object")));
        };
        for (key, _) in values {
            if !allowed.contains(&key.as_str()) {
                return Err(SceneError(format!("{path}.{key}: unknown field")));
            }
            if values.iter().filter(|(other, _)| other == key).count() > 1 {
                return Err(SceneError(format!("{path}.{key}: duplicate field")));
            }
        }
        Ok(Self {
            path: path.to_string(),
            values: values
                .iter()
                .map(|(key, value)| (key.clone(), value))
                .collect(),
        })
    }
    fn required(&mut self, key: &str) -> Result<&'a Value, SceneError> {
        self.optional(key)
            .ok_or_else(|| SceneError(format!("{}.{}: missing required field", self.path, key)))
    }
    fn optional(&mut self, key: &str) -> Option<&'a Value> {
        self.values
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| *value)
    }
}

fn parse_array<T>(
    value: Option<&Value>,
    path: &str,
    parse: impl Fn(&Value, &str) -> Result<T, SceneError>,
) -> Result<Vec<T>, SceneError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Value::Array(values) = value else {
        return Err(SceneError(format!("{path}: expected array")));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| parse(value, &format!("{path}[{index}]")))
        .collect()
}
fn scalar(value: &Value, path: &str) -> Result<f32, SceneError> {
    let Value::Number(value) = value else {
        return Err(SceneError(format!("{path}: expected number")));
    };
    let value = *value as f32;
    Ok(if value.is_finite() {
        value
    } else if value.is_sign_positive() {
        f32::MAX
    } else {
        f32::MIN
    })
}
fn optional_scalar(fields: &mut Fields<'_>, key: &str, default: f32) -> Result<f32, SceneError> {
    fields.optional(key).map_or(Ok(default), |value| {
        scalar(value, &format!("{}.{}", fields.path, key))
    })
}
fn integer(value: &Value, path: &str) -> Result<i32, SceneError> {
    let value = scalar(value, path)?;
    if value.fract() != 0.0 {
        return Err(SceneError(format!("{path}: expected integer")));
    }
    Ok(value.clamp(i32::MIN as f32, i32::MAX as f32) as i32)
}
fn usize_value(value: &Value, path: &str) -> Result<usize, SceneError> {
    let value = scalar(value, path)?;
    if value < 0.0 || value.fract() != 0.0 {
        return Err(SceneError(format!("{path}: expected non-negative integer")));
    }
    Ok((value as u64).min(usize::MAX as u64) as usize)
}
fn optional_usize(fields: &mut Fields<'_>, key: &str, default: usize) -> Result<usize, SceneError> {
    fields.optional(key).map_or(Ok(default), |value| {
        usize_value(value, &format!("{}.{}", fields.path, key))
    })
}
fn string(value: &Value, path: &str) -> Result<String, SceneError> {
    let Value::String(value) = value else {
        return Err(SceneError(format!("{path}: expected string")));
    };
    Ok(value.clone())
}
fn path_value(value: &Value, path: &str) -> Result<PathBuf, SceneError> {
    Ok(PathBuf::from(string(value, path)?))
}
fn vec3(value: &Value, path: &str) -> Result<Vec3, SceneError> {
    let values = array_values(value, path, 3)?;
    Ok(Vec3::new(
        scalar(&values[0], &format!("{path}[0]"))?,
        scalar(&values[1], &format!("{path}[1]"))?,
        scalar(&values[2], &format!("{path}[2]"))?,
    ))
}
fn vec4(value: &Value, path: &str) -> Result<Vec4, SceneError> {
    let values = array_values(value, path, 4)?;
    Ok(Vec4::new(
        scalar(&values[0], &format!("{path}[0]"))?,
        scalar(&values[1], &format!("{path}[1]"))?,
        scalar(&values[2], &format!("{path}[2]"))?,
        scalar(&values[3], &format!("{path}[3]"))?,
    ))
}
fn optional_vec3(
    fields: &mut Fields<'_>,
    key: &str,
    default: Vec3,
    path: &str,
) -> Result<Vec3, SceneError> {
    fields
        .optional(key)
        .map_or(Ok(default), |value| vec3(value, path))
}
fn array_values<'a>(
    value: &'a Value,
    path: &str,
    expected: usize,
) -> Result<&'a [Value], SceneError> {
    let Value::Array(values) = value else {
        return Err(SceneError(format!("{path}: expected {expected} numbers")));
    };
    if values.len() != expected {
        return Err(SceneError(format!("{path}: expected {expected} numbers")));
    }
    Ok(values)
}
fn floats(value: &Value, path: &str) -> Result<Vec<f32>, SceneError> {
    let Value::Array(values) = value else {
        return Err(SceneError(format!("{path}: expected array")));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| scalar(value, &format!("{path}[{index}]")))
        .collect()
}
fn usize3(value: &Value, path: &str) -> Result<[usize; 3], SceneError> {
    let values = array_values(value, path, 3)?;
    Ok([
        usize_value(&values[0], &format!("{path}[0]"))?,
        usize_value(&values[1], &format!("{path}[1]"))?,
        usize_value(&values[2], &format!("{path}[2]"))?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":100},"objects":[]}"#;

    #[test]
    fn parses_sections_and_rejects_unknown_fields() {
        let scene = Scene::from_str(MINIMAL).unwrap();
        assert_eq!(scene.camera.position, Vec3::new(0.0, 0.0, 5.0));
        let error = Scene::from_str(r#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":100,"oops":1}}"#).unwrap_err();
        assert!(error.0.contains("camera.oops"));
    }

    #[test]
    fn sanitizes_postfx_values_at_production_setters() {
        let scene = Scene::from_str(r#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":100},"postfx":[{"type":"ssao","radius":-1,"strength":9,"range":-2}] }"#).unwrap();
        let chain = scene.post_chain(Mat4::IDENTITY);
        assert_eq!(chain.len(), 1);
    }
}
