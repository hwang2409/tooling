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
use crate::math::{Mat4, Vec3, Vec4};
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
use crate::shadow::{
    CubeShadowState, ShadowDepthShader, ShadowDepthUniforms, ShadowMap, ShadowState,
    directional_light_view, render_cube_shadow_map,
};
use crate::shadow::{sanitize_bias, sanitize_far_plane, sanitize_light_size, sanitize_near_plane};
use crate::skybox::CubeTexture;
use std::f32::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

mod assets;
mod config;
mod parse;
mod render;

pub use config::*;
use render::{SceneShader, SceneUniforms};
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
        let mut shadow_meshes = Vec::new();
        for (index, (mesh, object)) in meshes.iter().zip(self.objects.iter()).enumerate() {
            if let Some(list) = instances[index].as_ref() {
                shadow_meshes.extend(list.iter().map(|instance| (mesh, instance.model())));
            } else {
                shadow_meshes.push((mesh, object.transform.matrix()));
            }
        }
        if let Some(light) = directional
            && let Some(shadow) = light.shadow
        {
            let directional_light = DirectionalLight::new(light.direction, light.color);
            match shadow.kind {
                ShadowType::Csm => {
                    let mut cascades = render_cascade_shadow_maps_with_config(
                        camera,
                        light.direction,
                        CascadeShadowConfig::new(shadow.cascades, shadow.lambda),
                        shadow.map_size.max(1),
                        &shadow_meshes,
                    )
                    .map_err(|error| SceneError(format!("lights.directional.shadow: {error}")))?;
                    cascades.set_bias(shadow.bias, shadow.slope_bias);
                    for value in &mut uniforms {
                        value.set_directional_cascaded_shadow(directional_light, cascades.clone());
                    }
                }
                ShadowType::Basic | ShadowType::Pcss => {
                    let mut state =
                        render_directional_shadow(camera, light.direction, shadow, &shadow_meshes)?;
                    state.set_bias(shadow.bias, shadow.slope_bias);
                    if shadow.kind == ShadowType::Pcss {
                        state.set_light_size(shadow.light_size);
                        state.set_light_depth_origin(shadow.near);
                    }
                    for value in &mut uniforms {
                        value.set_directional_shadow(directional_light, Some(state.clone()));
                    }
                }
                ShadowType::Cube => {}
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
            let mut state = CubeShadowState::new(light.position, map);
            state.set_bias(shadow.bias, shadow.slope_bias);
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

impl std::str::FromStr for Scene {
    type Err = SceneError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Scene::from_str(source)
    }
}

use assets::{build_ibl_maps, build_instances, load_mesh, load_skybox};
const fn tint_rgb(tint: Vec4) -> Vec3 {
    Vec3::new(tint.x, tint.y, tint.z)
}

use render::{render_directional_shadow, render_particles, scene_uniform_model};
fn channel(value: f32) -> u8 {
    (if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    } * 255.0)
        .round() as u8
}

use parse::parse_scene;
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
