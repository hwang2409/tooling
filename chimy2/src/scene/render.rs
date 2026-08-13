use super::*;
pub(super) fn render_particles(
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
        let steps = config.warmup_steps.max(1);
        system.step_n(steps);
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

pub(super) fn render_directional_shadow(
    camera: Camera,
    direction: Vec3,
    config: ShadowConfig,
    meshes: &[(&Mesh, Mat4)],
) -> Result<ShadowState, SceneError> {
    let extent = camera.far.max(10.0);
    let target = Vec3::ZERO;
    let up = if direction.y.abs() > 0.95 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let view = directional_light_view(direction, target, extent * 2.0, up);
    let projection = Mat4::orthographic(-extent, extent, -extent, extent, config.near, config.far);
    let light_view_projection = projection * view;
    let mut target = Framebuffer::new(config.map_size.max(1), config.map_size.max(1));
    target.clear(0);
    let mut pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    for &(mesh, model) in meshes {
        pipeline.draw_mesh_depth_with_varyings(
            &mut target,
            mesh,
            &ShadowDepthUniforms::new(model, light_view_projection),
        );
    }
    let map = ShadowMap::from_framebuffer(&target)
        .map_err(|error| SceneError(format!("lights.directional.shadow: {error}")))?;
    Ok(ShadowState::new(light_view_projection, map))
}

pub(super) fn scene_uniform_model(uniforms: &SceneUniforms<'_>) -> Mat4 {
    match uniforms {
        SceneUniforms::Blinn(value) => value.model(),
        SceneUniforms::Ggx(value) => value.lighting.model(),
        SceneUniforms::IblGgx(value) => value.lighting.lighting.model(),
    }
}
#[derive(Clone)]
pub(super) enum SceneUniforms<'a> {
    Blinn(BlinnPhongUniforms),
    Ggx(CookTorranceUniforms),
    IblGgx(IblCookTorranceUniforms<'a>),
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SceneShader;

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
    pub(super) fn set_directional_shadow(
        &mut self,
        light: DirectionalLight,
        state: Option<ShadowState>,
    ) {
        match self {
            SceneUniforms::Blinn(value) => value.set_directional_shadow(light, state),
            SceneUniforms::Ggx(value) => value.lighting.set_directional_shadow(light, state),
            SceneUniforms::IblGgx(value) => {
                value.lighting.lighting.set_directional_shadow(light, state)
            }
        }
    }

    pub(super) fn set_directional_cascaded_shadow(
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

    pub(super) fn set_point_shadow(&mut self, index: usize, shadow: CubeShadowState) {
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
