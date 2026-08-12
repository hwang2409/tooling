use crate::demo::{cube_with_uvs, plane_xz};
use crate::math::{Mat4, Vec3};
use crate::mesh::Mesh;

pub struct PointShadowScene {
    pub floor: Mesh,
    pub wall: Mesh,
    pub occluder: Mesh,
    pub wall_models: [Mat4; 4],
    pub occluder_models: [Mat4; 3],
    pub light_position: Vec3,
}

pub fn build_point_shadow_scene() -> PointShadowScene {
    PointShadowScene {
        floor: plane_xz(10.0, 10.0, 8, 1.0),
        wall: cube_with_uvs(1.0),
        occluder: cube_with_uvs(1.0),
        wall_models: [
            Mat4::translate(Vec3::new(0.0, 4.8, 0.0)) * Mat4::scale(Vec3::new(5.0, 0.15, 5.0)),
            Mat4::translate(Vec3::new(-4.8, 2.3, 0.0)) * Mat4::scale(Vec3::new(0.15, 2.5, 5.0)),
            Mat4::translate(Vec3::new(4.8, 2.3, 0.0)) * Mat4::scale(Vec3::new(0.15, 2.5, 5.0)),
            Mat4::translate(Vec3::new(0.0, 2.3, -4.8)) * Mat4::scale(Vec3::new(5.0, 2.5, 0.15)),
        ],
        occluder_models: [
            Mat4::translate(Vec3::new(-1.8, 0.9, -0.7)) * Mat4::scale(Vec3::new(0.8, 0.9, 0.8)),
            Mat4::translate(Vec3::new(1.4, 1.2, -1.8)) * Mat4::scale(Vec3::new(1.0, 1.2, 0.65)),
            Mat4::translate(Vec3::new(1.8, 0.55, 1.5)) * Mat4::scale(Vec3::new(0.65, 0.55, 1.0)),
        ],
        light_position: Vec3::new(0.0, 3.6, 0.4),
    }
}
