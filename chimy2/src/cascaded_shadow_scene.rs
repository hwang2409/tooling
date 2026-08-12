use crate::camera::Camera;
use crate::demo::{cube_with_uvs, plane_xz};
use crate::math::{Mat4, Quat, Vec3};
use crate::mesh::Mesh;

pub struct CascadedShadowScene {
    pub ground: Mesh,
    pub object: Mesh,
    pub object_models: [Mat4; 8],
    pub camera: Camera,
    pub light_direction: Vec3,
}

pub fn build_cascaded_shadow_scene(aspect: f32) -> CascadedShadowScene {
    CascadedShadowScene {
        ground: plane_xz(28.0, 82.0, 16, 1.0),
        object: cube_with_uvs(0.8),
        object_models: std::array::from_fn(|index| {
            let z = -4.0 - index as f32 * 8.0;
            let x = if index % 2 == 0 { -1.8 } else { 1.8 };
            Mat4::translate(Vec3::new(x, 0.8, z))
                * Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), index as f32 * 0.31)
        }),
        camera: Camera::new(
            Vec3::new(8.0, 5.0, 12.0),
            Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.26)
                * Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -0.16),
            0.82,
            aspect.max(0.001),
            0.1,
            70.0,
        ),
        light_direction: Vec3::new(0.65, 1.0, 0.45).normalize(),
    }
}
