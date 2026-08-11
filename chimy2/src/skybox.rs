//! Cube-map sampling and the frame-level skybox pass.
//!
//! Face images use the following convention. `u` increases to the right and
//! `v` increases toward the bottom of each image. The dominant direction axis
//! selects the face, then the table maps the direction to face UV coordinates:
//!
//! | face | `u` before remapping | `v` before remapping |
//! | --- | --- | --- |
//! | +X | `-z / abs(x)` | `-y / abs(x)` |
//! | -X | ` z / abs(x)` | `-y / abs(x)` |
//! | +Y | ` x / abs(y)` | ` z / abs(y)` |
//! | -Y | ` x / abs(y)` | `-z / abs(y)` |
//! | +Z | ` x / abs(z)` | `-y / abs(z)` |
//! | -Z | `-x / abs(z)` | `-y / abs(z)` |
//!
//! Each value is remapped from `[-1, 1]` to `[0, 1]`. Face edges clamp during
//! bilinear filtering. This avoids sampling a different face, but leaves the
//! usual small cubemap seam when neighboring faces do not match.

use crate::camera::Camera;
use crate::fb::{Framebuffer, argb8888_linear};
use crate::image::{ColorSpace, Texture, WrapMode};
use crate::math::{Vec2, Vec3, Vec4};
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeFace {
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ,
}

impl CubeFace {
    const fn index(self) -> usize {
        match self {
            Self::PositiveX => 0,
            Self::NegativeX => 1,
            Self::PositiveY => 2,
            Self::NegativeY => 3,
            Self::PositiveZ => 4,
            Self::NegativeZ => 5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubeTextureError {
    message: String,
}

impl CubeTextureError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for CubeTextureError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CubeTextureError {}

#[derive(Clone, Debug, PartialEq)]
pub struct CubeTexture {
    faces: [Texture; 6],
    size: usize,
    color_space: ColorSpace,
}

impl CubeTexture {
    /// Builds a cube texture in `+X, -X, +Y, -Y, +Z, -Z` order.
    pub fn new(mut faces: [Texture; 6]) -> Result<Self, CubeTextureError> {
        let first = &faces[0];
        if first.width() != first.height() {
            return Err(CubeTextureError::new("cube texture faces must be square"));
        }
        if first.width() == 0 {
            return Err(CubeTextureError::new("cube texture faces must be non-zero"));
        }
        let size = first.width();
        let color_space = first.color_space();
        for (index, face) in faces.iter().enumerate().skip(1) {
            if face.width() != first.width() || face.height() != first.height() {
                return Err(CubeTextureError::new(format!(
                    "cube face {index} has dimensions {}x{}, expected {}x{}",
                    face.width(),
                    face.height(),
                    first.width(),
                    first.height()
                )));
            }
            if face.color_space() != first.color_space() {
                return Err(CubeTextureError::new(format!(
                    "cube face {index} has a different color space"
                )));
            }
        }
        for face in &mut faces {
            face.set_wrap_mode(WrapMode::ClampToEdge);
        }
        Ok(Self {
            size,
            color_space,
            faces,
        })
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    pub const fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    pub fn face(&self, face: CubeFace) -> &Texture {
        &self.faces[face.index()]
    }

    /// Samples the cube map into linear RGB and alpha.
    pub fn sample(&self, direction: Vec3) -> [f32; 4] {
        let (face, uv) = face_and_uv(direction);
        self.face(face).sample_linear_bilinear(uv)
    }

    pub fn face_and_uv(direction: Vec3) -> (CubeFace, Vec2) {
        face_and_uv(direction)
    }
}

// Kept separate from `CubeTexture::sample` so tests can assert the hand table
// without depending on texture filtering or color conversion.
fn face_and_uv(direction: Vec3) -> (CubeFace, Vec2) {
    let direction = if direction.x.is_finite() && direction.y.is_finite() && direction.z.is_finite()
    {
        direction
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let x = direction.x.abs();
    let y = direction.y.abs();
    let z = direction.z.abs();
    let (face, u, v, denominator) = if x >= y && x >= z {
        if direction.x >= 0.0 {
            (CubeFace::PositiveX, -direction.z, -direction.y, x)
        } else {
            (CubeFace::NegativeX, direction.z, -direction.y, x)
        }
    } else if y >= z {
        if direction.y >= 0.0 {
            (CubeFace::PositiveY, direction.x, direction.z, y)
        } else {
            (CubeFace::NegativeY, direction.x, -direction.z, y)
        }
    } else if direction.z >= 0.0 {
        (CubeFace::PositiveZ, direction.x, -direction.y, z)
    } else {
        (CubeFace::NegativeZ, -direction.x, -direction.y, z)
    };
    if denominator == 0.0 {
        return (CubeFace::PositiveX, Vec2::new(0.5, 0.5));
    }
    (
        face,
        Vec2::new((u / denominator + 1.0) * 0.5, (v / denominator + 1.0) * 0.5),
    )
}

/// Returns the world-space ray through a pixel center.
///
/// The ray uses the inverse view-projection at far clip depth. Subtracting the
/// camera position removes view translation, so moving the camera cannot move
/// the sky. Rotation still changes the ray direction.
pub fn skybox_ray(
    camera: Camera,
    pixel_x: usize,
    pixel_y: usize,
    width: usize,
    height: usize,
) -> Vec3 {
    if width == 0 || height == 0 {
        return camera.forward();
    }
    let ndc_x = ((pixel_x as f32 + 0.5) / width as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((pixel_y as f32 + 0.5) / height as f32) * 2.0;
    let clip_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let inverse = (camera.projection_matrix() * camera.view_matrix())
        .inverse()
        .unwrap_or(Mat4::IDENTITY);
    let world_far = inverse * clip_far;
    let world_far = if world_far.w == 0.0 {
        Vec3::new(world_far.x, world_far.y, world_far.z)
    } else {
        Vec3::new(
            world_far.x / world_far.w,
            world_far.y / world_far.w,
            world_far.z / world_far.w,
        )
    };
    (world_far - camera.position).normalize()
}

/// Draws the sky after opaque commands and before transparent commands.
///
/// The pass tests against the cleared far depth and never writes depth. Opaque
/// geometry therefore remains visible, while transparent geometry can blend
/// over the sky in the normal queued order.
pub fn render_skybox(framebuffer: &mut Framebuffer, camera: Camera, cube: &CubeTexture) {
    for y in 0..framebuffer.height {
        for x in 0..framebuffer.width {
            let index = y * framebuffer.width + x;
            if framebuffer.depth[index] < 1.0 {
                continue;
            }
            let sample = cube.sample(skybox_ray(
                camera,
                x,
                y,
                framebuffer.width,
                framebuffer.height,
            ));
            framebuffer.color[index] =
                argb8888_linear(sample[3], [sample[0], sample[1], sample[2]]);
        }
    }
}

use crate::math::Mat4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Texture;
    use crate::math::{Quat, Vec3};
    use std::f32::consts::FRAC_PI_2;

    fn cube() -> CubeTexture {
        CubeTexture::new(std::array::from_fn(|index| {
            Texture::new(2, 2, vec![[index as u8 * 30 + 10, 0, 0, 255]; 4]).unwrap()
        }))
        .unwrap()
    }

    #[test]
    fn axis_directions_hit_documented_face_centers() {
        let cases = [
            (Vec3::new(1.0, 0.0, 0.0), CubeFace::PositiveX),
            (Vec3::new(-1.0, 0.0, 0.0), CubeFace::NegativeX),
            (Vec3::new(0.0, 1.0, 0.0), CubeFace::PositiveY),
            (Vec3::new(0.0, -1.0, 0.0), CubeFace::NegativeY),
            (Vec3::new(0.0, 0.0, 1.0), CubeFace::PositiveZ),
            (Vec3::new(0.0, 0.0, -1.0), CubeFace::NegativeZ),
        ];
        for (direction, expected_face) in cases {
            let (face, uv) = CubeTexture::face_and_uv(direction);
            assert_eq!(face, expected_face);
            assert_eq!(uv, Vec2::new(0.5, 0.5));
        }
    }

    #[test]
    fn off_axis_direction_uses_hand_computed_positive_x_uv() {
        let (face, uv) = CubeTexture::face_and_uv(Vec3::new(1.0, -0.5, -0.25));
        assert_eq!(face, CubeFace::PositiveX);
        assert!((uv.x - 0.625).abs() < 1e-6);
        assert!((uv.y - 0.75).abs() < 1e-6);
    }

    #[test]
    fn sample_decodes_each_face_and_clamps_edges() {
        let cube = cube();
        let sample = cube.sample(Vec3::new(1.0, -0.5, -0.25));
        assert!((sample[0] - crate::image::srgb_to_linear_u8(10)).abs() < 1e-6);
        assert_eq!(
            cube.face(CubeFace::PositiveX).wrap_mode,
            WrapMode::ClampToEdge
        );
    }

    #[test]
    fn construction_rejects_mismatched_face_space() {
        let faces = [
            Texture::new(2, 2, vec![[0, 0, 0, 255]; 4]).unwrap(),
            Texture::new_with_color_space(2, 2, vec![[0, 0, 0, 255]; 4], ColorSpace::Linear)
                .unwrap(),
            Texture::new(2, 2, vec![[0, 0, 0, 255]; 4]).unwrap(),
            Texture::new(2, 2, vec![[0, 0, 0, 255]; 4]).unwrap(),
            Texture::new(2, 2, vec![[0, 0, 0, 255]; 4]).unwrap(),
            Texture::new(2, 2, vec![[0, 0, 0, 255]; 4]).unwrap(),
        ];
        assert!(CubeTexture::new(faces).is_err());
    }

    #[test]
    fn skybox_ray_is_translation_invariant() {
        let first = Camera::new(
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_2),
            1.0,
            1.5,
            0.1,
            100.0,
        );
        let second = Camera::new(
            Vec3::new(100.0, -20.0, 3.0),
            first.orientation,
            first.fov_y,
            first.aspect,
            first.near,
            first.far,
        );
        let a = skybox_ray(first, 7, 3, 16, 10);
        let b = skybox_ray(second, 7, 3, 16, 10);
        assert!((a.x - b.x).abs() < 1e-5);
        assert!((a.y - b.y).abs() < 1e-5);
        assert!((a.z - b.z).abs() < 1e-5);
    }
}
