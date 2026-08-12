use chimy2::camera::Camera;
use chimy2::csm::{CascadeShadowConfig, render_cascade_shadow_maps_instanced_with_config};
use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Quat, Vec3, Vec4};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::{Instance, InstanceUniforms, Pipeline};
use chimy2::postfx::{PostChain, SsaoPass};
use chimy2::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, MeshShader, MeshUniforms, PointLight,
};
use chimy2::shadow::{render_cube_shadow_map, render_cube_shadow_map_instanced};

fn triangle() -> Mesh {
    Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(-0.7, -0.6, 0.0), None, None),
            MeshVertex::new(Vec3::new(0.7, -0.6, 0.0), None, None),
            MeshVertex::new(Vec3::new(0.0, 0.7, 0.0), None, None),
        ],
        vec![[0, 1, 2]],
    )
}

fn transparent_pair() -> Mesh {
    Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(-0.7, -0.6, -0.25), None, None),
            MeshVertex::new(Vec3::new(0.7, -0.6, -0.25), None, None),
            MeshVertex::new(Vec3::new(0.0, 0.7, -0.25), None, None),
            MeshVertex::new(Vec3::new(-0.7, -0.6, 0.25), None, None),
            MeshVertex::new(Vec3::new(0.7, -0.6, 0.25), None, None),
            MeshVertex::new(Vec3::new(0.0, 0.7, 0.25), None, None),
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    )
}

fn render_mixed_scene(instanced: bool) -> Framebuffer {
    let mesh = transparent_pair();
    let instances = [
        Instance::with_tint(
            Mat4::translate(Vec3::new(-0.2, 0.0, 0.45)),
            Vec4::new(0.8, 1.0, 0.8, 0.7),
        ),
        Instance::with_tint(
            Mat4::translate(Vec3::new(0.2, 0.0, -0.45)),
            Vec4::new(1.0, 0.7, 0.8, 0.8),
        ),
    ];
    let view = Mat4::IDENTITY;
    let projection = Mat4::IDENTITY;
    let mut base = MeshUniforms::new(Mat4::IDENTITY, view, projection, 0xFF5070A0);
    base.set_alpha(0.5);
    let mut other = MeshUniforms::new(
        Mat4::translate(Vec3::new(0.0, 0.0, 0.45)),
        view,
        projection,
        0xFFB07040,
    );
    other.set_alpha(0.5);
    let individual_uniforms = instances.map(|instance| {
        let mut uniforms = base;
        uniforms.set_model(instance.model());
        uniforms.apply_instance_tint(instance.tint().unwrap());
        uniforms
    });
    let mut framebuffer = Framebuffer::new(40, 32);
    framebuffer.clear(0xFF101820);
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.set_thread_count(1);
    pipeline.render(&mut framebuffer, |frame, target| {
        if instanced {
            frame.draw_mesh_instanced(target, &mesh, &base, &instances);
        } else {
            for uniforms in &individual_uniforms {
                frame.draw_mesh(target, &mesh, uniforms);
            }
        }
        frame.draw_mesh(target, &mesh, &other);
    });
    framebuffer
}

fn render_ssaa_ssao_scene(instanced: bool) -> Framebuffer {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 48;
    let mesh = triangle();
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 4.0),
        Quat::IDENTITY,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        20.0,
    );
    let projection = camera.projection_matrix();
    let instances = [
        Instance::new(Mat4::translate(Vec3::new(-0.55, 0.0, -2.0))),
        Instance::new(Mat4::translate(Vec3::new(0.55, 0.0, -2.7))),
    ];
    let base = MeshUniforms::new(Mat4::IDENTITY, camera.view_matrix(), projection, 0xFF5070A0);
    let individual_uniforms = instances.map(|instance| base.for_instance(&instance));
    let mut ssao = SsaoPass::new(projection);
    ssao.set_radius(0.9);
    ssao.set_range(10.0);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.set_ssaa_scale(2);
    pipeline.set_post_chain(PostChain::new().with_pass(ssao));
    pipeline.render(&mut framebuffer, |frame, target| {
        if instanced {
            frame.draw_mesh_instanced(target, &mesh, &base, &instances);
        } else {
            for uniforms in &individual_uniforms {
                frame.draw_mesh(target, &mesh, uniforms);
            }
        }
    });
    framebuffer
}

#[test]
fn instanced_submission_matches_interleaved_individual_draws() {
    assert_eq!(
        render_mixed_scene(true).color,
        render_mixed_scene(false).color
    );
}

#[test]
fn instanced_submission_matches_individual_draws_with_ssaa_and_ssao() {
    let instanced = render_ssaa_ssao_scene(true);
    let individual = render_ssaa_ssao_scene(false);
    assert_eq!(instanced.color, individual.color);
    assert_eq!(instanced.depth, individual.depth);
}

#[test]
fn empty_and_single_instance_cases_match_the_existing_path() {
    let mesh = triangle();
    let base = MeshUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, 0xFF406080);
    let mut empty = Framebuffer::new(24, 24);
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.render(&mut empty, |frame, target| {
        frame.draw_mesh_instanced(target, &mesh, &base, &[]);
    });
    let mut clear = Framebuffer::new(24, 24);
    pipeline.render(&mut clear, |_, _| {});
    assert_eq!(empty.color, clear.color);
    assert_eq!(empty.depth, clear.depth);

    let instance = Instance::new(Mat4::translate(Vec3::new(0.1, 0.0, 0.0)));
    let mut instanced = Framebuffer::new(24, 24);
    pipeline.render(&mut instanced, |frame, target| {
        frame.draw_mesh_instanced(target, &mesh, &base, &[instance]);
    });
    let mut individual = Framebuffer::new(24, 24);
    let mut uniforms = base;
    uniforms.set_model(instance.model());
    pipeline.render(&mut individual, |frame, target| {
        frame.draw_mesh(target, &mesh, &uniforms);
    });
    assert_eq!(instanced.color, individual.color);
    assert_eq!(instanced.depth, individual.depth);
}

#[test]
fn malformed_instance_values_are_sanitized_at_the_setter() {
    let mut instance = Instance::with_tint(
        Mat4::new([f32::NAN; 16]),
        Vec4::new(f32::INFINITY, -1.0, 0.5, f32::NAN),
    );
    assert_eq!(instance.model(), Mat4::IDENTITY);
    assert_eq!(instance.tint(), Some(Vec4::new(1.0, 0.0, 0.5, 1.0)));
    instance.set_model(Mat4::new([f32::INFINITY; 16]));
    instance.set_tint(Some(Vec4::new(f32::NAN, 2.0, -2.0, 0.25)));
    assert_eq!(instance.model(), Mat4::IDENTITY);
    assert_eq!(instance.tint(), Some(Vec4::new(1.0, 1.0, 0.0, 0.25)));
}

#[test]
fn instanced_depth_matches_individual_depth_for_cube_and_csm() {
    let mesh = triangle();
    let models = [
        Mat4::translate(Vec3::new(-0.25, 0.0, -1.5)),
        Mat4::translate(Vec3::new(0.25, 0.0, -2.5)),
    ];
    let instances = models.map(Instance::new);
    let individual = [(&mesh, models[0]), (&mesh, models[1])];
    let grouped = [(&mesh, instances.as_slice())];
    let cube_a = render_cube_shadow_map(Vec3::ZERO, 0.1, 8.0, 32, &individual).unwrap();
    let cube_b = render_cube_shadow_map_instanced(Vec3::ZERO, 0.1, 8.0, 32, &grouped).unwrap();
    assert_eq!(cube_a, cube_b);

    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 4.0),
        Quat::IDENTITY,
        std::f32::consts::FRAC_PI_3,
        1.0,
        0.1,
        20.0,
    );
    let config = CascadeShadowConfig::new(2, 0.5);
    let csm_a = chimy2::csm::render_cascade_shadow_maps_with_config(
        camera,
        Vec3::new(0.4, 1.0, 0.3),
        config,
        32,
        &individual,
    )
    .unwrap();
    let csm_b = render_cascade_shadow_maps_instanced_with_config(
        camera,
        Vec3::new(0.4, 1.0, 0.3),
        config,
        32,
        &grouped,
    )
    .unwrap();
    assert_eq!(csm_a, csm_b);
}

#[test]
fn non_uniform_instance_lighting_matches_individual_draw() {
    let mesh = Mesh::new(
        vec![
            MeshVertex::new(
                Vec3::new(-0.3, -0.3, 0.0),
                None,
                Some(Vec3::new(0.6, 0.8, 0.0)),
            ),
            MeshVertex::new(
                Vec3::new(0.3, -0.3, 0.0),
                None,
                Some(Vec3::new(0.6, 0.8, 0.0)),
            ),
            MeshVertex::new(
                Vec3::new(0.0, 0.3, 0.0),
                None,
                Some(Vec3::new(0.6, 0.8, 0.0)),
            ),
        ],
        vec![[0, 1, 2]],
    );
    let instances = [
        Instance::new(
            Mat4::translate(Vec3::new(-0.35, 0.0, 0.0)) * Mat4::scale(Vec3::new(2.0, 1.0, 1.0)),
        ),
        Instance::new(
            Mat4::translate(Vec3::new(0.35, 0.0, 0.0)) * Mat4::scale(Vec3::new(1.0, 3.0, 1.0)),
        ),
    ];
    let base = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Vec3::ZERO,
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::ZERO,
        1.0,
        Vec3::new(0.0, 0.0, 2.0),
        DirectionalLight::new(Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    );
    let mut instanced = Framebuffer::new(48, 32);
    let mut individual = Framebuffer::new(48, 32);
    let mut instanced_pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    let mut individual_pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    instanced_pipeline.render(&mut instanced, |frame, target| {
        frame.draw_mesh_instanced(target, &mesh, &base, &instances);
    });
    let individual_uniforms = instances.map(|instance| {
        let mut uniforms = base.clone();
        uniforms.set_model(instance.model());
        uniforms
    });
    individual_pipeline.render(&mut individual, |frame, target| {
        for uniforms in &individual_uniforms {
            frame.draw_mesh(target, &mesh, uniforms);
        }
    });
    assert_eq!(instanced.color, individual.color);
}
