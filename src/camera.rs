//! Quaternion camera and deterministic orbit/fly controllers.

use crate::math::{Mat4, Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub orientation: Quat,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub const fn new(
        position: Vec3,
        orientation: Quat,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            orientation,
            fov_y,
            aspect,
            near,
            far,
        }
    }

    pub fn view_matrix(self) -> Mat4 {
        self.orientation.conjugate().normalized().to_mat4() * Mat4::translate(-self.position)
    }

    pub fn projection_matrix(self) -> Mat4 {
        Mat4::perspective(self.fov_y, self.aspect, self.near, self.far)
    }

    pub fn view_projection(self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn forward(self) -> Vec3 {
        self.orientation.rotate_vec3(Vec3::new(0.0, 0.0, -1.0))
    }

    pub fn right(self) -> Vec3 {
        self.orientation.rotate_vec3(Vec3::new(1.0, 0.0, 0.0))
    }

    pub fn up(self) -> Vec3 {
        self.orientation.rotate_vec3(Vec3::new(0.0, 1.0, 0.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitController {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
}

impl OrbitController {
    pub fn new(target: Vec3, distance: f32, yaw: f32, pitch: f32) -> Self {
        Self {
            target,
            distance: distance.max(0.001),
            yaw,
            pitch,
            min_distance: 0.1,
            max_distance: 1000.0,
            min_pitch: -1.55,
            max_pitch: 1.55,
        }
    }

    pub fn step(&mut self, yaw_delta: f32, pitch_delta: f32, zoom_delta: f32) {
        self.yaw += yaw_delta;
        self.pitch = (self.pitch + pitch_delta).clamp(self.min_pitch, self.max_pitch);
        self.distance = (self.distance + zoom_delta).clamp(self.min_distance, self.max_distance);
    }

    pub fn position(self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        self.target
            + Vec3::new(
                self.yaw.sin() * cos_pitch * self.distance,
                self.pitch.sin() * self.distance,
                self.yaw.cos() * cos_pitch * self.distance,
            )
    }

    pub fn camera(self, fov_y: f32, aspect: f32, near: f32, far: f32) -> Camera {
        let orientation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), self.yaw)
            * Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -self.pitch);
        Camera::new(self.position(), orientation, fov_y, aspect, near, far)
    }
}

pub type OrbitCamera = OrbitController;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlyInput {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyController {
    pub camera: Camera,
    pub move_speed: f32,
    pub look_speed: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl FlyController {
    pub fn new(camera: Camera) -> Self {
        let forward = camera
            .orientation
            .normalized()
            .rotate_vec3(Vec3::new(0.0, 0.0, -1.0));
        Self {
            camera,
            move_speed: 3.0,
            look_speed: 1.0,
            yaw: (-forward.x).atan2(-forward.z),
            pitch: (-forward.y).clamp(-1.0, 1.0).asin(),
        }
    }

    pub fn step(&mut self, input: FlyInput, delta_seconds: f32) {
        self.yaw += input.yaw * self.look_speed * delta_seconds;
        self.pitch =
            (self.pitch + input.pitch * self.look_speed * delta_seconds).clamp(-1.55, 1.55);
        self.camera.orientation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), self.yaw)
            * Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -self.pitch);
        let local_move =
            Vec3::new(input.right, input.up, -input.forward) * (self.move_speed * delta_seconds);
        self.camera.position =
            self.camera.position + self.camera.orientation.rotate_vec3(local_move);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec4;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    fn assert_close(actual: Vec4, expected: Vec4) {
        assert!((actual.x - expected.x).abs() < 1e-5);
        assert!((actual.y - expected.y).abs() < 1e-5);
        assert!((actual.z - expected.z).abs() < 1e-5);
        assert!((actual.w - expected.w).abs() < 1e-5);
    }

    #[test]
    fn identity_orientation_view_matches_hand_computed_pose() {
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Quat::IDENTITY,
            FRAC_PI_2,
            1.0,
            0.1,
            100.0,
        );
        assert_close(
            camera.view_matrix() * Vec4::new(1.0, 2.0, 3.0, 1.0),
            Vec4::new(1.0, 2.0, -2.0, 1.0),
        );
    }

    #[test]
    fn orbit_step_updates_angles_and_distance() {
        let mut orbit = OrbitController::new(Vec3::ZERO, 4.0, 0.0, 0.0);
        orbit.step(FRAC_PI_2, 0.0, -1.0);
        assert_eq!(orbit.distance, 3.0);
        let position = orbit.position();
        assert!((position.x - 3.0).abs() < 1e-5);
        assert!(position.z.abs() < 1e-5);
    }

    #[test]
    fn fly_step_moves_forward_in_camera_space() {
        let camera = Camera::new(Vec3::ZERO, Quat::IDENTITY, FRAC_PI_2, 1.0, 0.1, 100.0);
        let mut fly = FlyController::new(camera);
        fly.move_speed = 2.0;
        fly.step(
            FlyInput {
                forward: 1.0,
                ..FlyInput::default()
            },
            0.5,
        );
        assert_eq!(fly.camera.position, Vec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn fly_step_without_input_preserves_rotated_camera_orientation() {
        let orientation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_4);
        let camera = Camera::new(Vec3::ZERO, orientation, FRAC_PI_2, 1.0, 0.1, 100.0);
        let mut fly = FlyController::new(camera);
        fly.step(FlyInput::default(), 1.0);
        assert_close(
            Vec4::new(
                fly.camera.orientation.x,
                fly.camera.orientation.y,
                fly.camera.orientation.z,
                fly.camera.orientation.w,
            ),
            Vec4::new(orientation.x, orientation.y, orientation.z, orientation.w),
        );
    }
}
