use crate::demo::uv_sphere;
use crate::fb::Framebuffer;
use crate::gltf::{MorphTarget, blend_morph_targets};
use crate::math::{Mat4, Vec3};
use crate::mesh::Mesh;
use crate::pipeline::Pipeline;
use crate::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use std::f32::consts::PI;

/// Shared CPU morph scene for the interactive demo and fixed-weight golden.
/// Morph targets are deliberately kept outside LodMesh; LOD interaction is
/// not part of the morph-target subset.
#[derive(Clone, Debug, PartialEq)]
pub struct MorphScene {
    pub base: Mesh,
    pub targets: Vec<MorphTarget>,
    pub camera: crate::camera::Camera,
    pub lighting: BlinnPhongUniforms,
}

pub fn build_morph_scene(aspect: f32) -> MorphScene {
    let camera = crate::camera::Camera::new(
        Vec3::new(0.0, 0.15, 4.2),
        crate::math::Quat::IDENTITY,
        PI / 3.0,
        aspect.max(0.01),
        0.1,
        20.0,
    );
    let base = uv_sphere(1.15, 16, 24);
    let inflate = base
        .vertices()
        .iter()
        .map(|vertex| vertex.position() * 0.22)
        .collect();
    let spike = base
        .vertices()
        .iter()
        .map(|vertex| Vec3::new(0.0, vertex.position().y.max(0.0) * 0.9, 0.0))
        .collect();
    let targets = vec![
        MorphTarget::new(inflate, None).expect("inflate target matches the base mesh"),
        MorphTarget::new(spike, None).expect("spike target matches the base mesh"),
    ];
    let lighting = BlinnPhongUniforms::new_with_linear_colors(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.08, 0.08, 0.08),
        Vec3::new(0.76, 0.32, 0.12),
        Vec3::new(0.35, 0.35, 0.35),
        24.0,
        camera.position,
        DirectionalLight::new(
            Vec3::new(-0.45, 0.7, 0.5).normalize(),
            Vec3::new(1.0, 0.94, 0.86),
        ),
        PointLight::new(
            Vec3::new(2.0, 2.5, 3.0),
            Vec3::new(0.35, 0.4, 0.55),
            1.0,
            0.05,
            0.01,
        ),
    );
    MorphScene {
        base,
        targets,
        camera,
        lighting,
    }
}

pub fn render_morph_scene(framebuffer: &mut Framebuffer, scene: &MorphScene, weights: [f32; 2]) {
    framebuffer.clear(crate::fb::argb8888(255, 8, 10, 18));
    let mesh = blend_morph_targets(&scene.base, &scene.targets, &weights)
        .expect("shared morph scene has matching target data");
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.set_thread_count(1);
    pipeline.render(framebuffer, |frame, target| {
        frame.draw_mesh(target, &mesh, &scene.lighting);
    });
}
