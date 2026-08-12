//! Shared pole-and-slab scene for the directional PCSS demo and golden.

use crate::demo::{cube_with_uvs, plane_xz};
use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec3};
use crate::mesh::Mesh;
use crate::pipeline::Pipeline;
use crate::shadow::{ShadowDepthShader, ShadowDepthUniforms, ShadowMap, directional_light_view};

pub const PCSS_SHADOW_SIZE: usize = 256;

pub struct PcssShadowScene {
    pub ground: Mesh,
    pub pole: Mesh,
    pub slab: Mesh,
    pub pole_model: Mat4,
    pub slab_model: Mat4,
    pub light_direction: Vec3,
    pub light_view_projection: Mat4,
    pub shadow_map: ShadowMap,
    pub camera_position: Vec3,
    pub target: Vec3,
}

/// Builds the same geometry and directional shadow map used by the demo and
/// golden. The pole tip receives a broad penumbra, while the slab stays near
/// the ground and shows contact hardening.
pub fn build_pcss_shadow_scene() -> PcssShadowScene {
    let ground = plane_xz(14.0, 14.0, 8, 1.0);
    let pole = cube_with_uvs(1.0);
    let slab = cube_with_uvs(1.0);
    let pole_model =
        Mat4::translate(Vec3::new(-1.4, 2.0, 0.0)) * Mat4::scale(Vec3::new(0.55, 2.0, 0.55));
    let slab_model =
        Mat4::translate(Vec3::new(1.4, 0.2, 0.0)) * Mat4::scale(Vec3::new(1.4, 0.2, 1.2));
    let target = Vec3::new(0.0, 1.0, 0.0);
    let light_direction = Vec3::new(0.65, 1.0, -0.45).normalize();
    let light_view_projection = Mat4::orthographic(-8.0, 8.0, -8.0, 8.0, 1.0, 24.0)
        * directional_light_view(light_direction, target, 12.0, Vec3::new(0.0, 1.0, 0.0));
    let mut shadow_target = Framebuffer::new(PCSS_SHADOW_SIZE, PCSS_SHADOW_SIZE);
    shadow_target.clear(0);
    let mut depth_pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
    depth_pipeline.draw_mesh_depth(
        &mut shadow_target,
        &ground,
        &ShadowDepthUniforms::new(Mat4::IDENTITY, light_view_projection),
    );
    depth_pipeline.draw_mesh_depth(
        &mut shadow_target,
        &pole,
        &ShadowDepthUniforms::new(pole_model, light_view_projection),
    );
    depth_pipeline.draw_mesh_depth(
        &mut shadow_target,
        &slab,
        &ShadowDepthUniforms::new(slab_model, light_view_projection),
    );
    let shadow_map = ShadowMap::from_framebuffer(&shadow_target).expect("shadow target has size");
    PcssShadowScene {
        ground,
        pole,
        slab,
        pole_model,
        slab_model,
        light_direction,
        light_view_projection,
        shadow_map,
        camera_position: Vec3::new(8.0, 5.8, 9.0),
        target,
    }
}
