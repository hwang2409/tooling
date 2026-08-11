//! Small, dependency-free math types for the renderer.
//!
//! `Mat4` uses column-major storage. The element at row `r`, column `c` is
//! stored at `c * 4 + r`, and vectors are column vectors multiplied on the
//! right. Translation therefore lives in the fourth column.
//!
//! The coordinate system is right-handed. `look_at` maps the camera's forward
//! direction to -Z. Perspective projection uses OpenGL-style NDC depth in
//! `[-1, +1]`. Orthographic and perspective projections use this same NDC
//! depth convention. Quaternions use Hamilton multiplication and active
//! rotations.

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let length = self.length();
        if length == 0.0 { self } else { self / length }
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Neg for Vec2 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Vec2 {
        rhs * self
    }
}

impl Mul for Vec2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl Div for Vec2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let length = self.length();
        if length == 0.0 { self } else { self / length }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl Mul for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Div for Vec3 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let length = self.length();
        if length == 0.0 { self } else { self / length }
    }
}

impl Add for Vec4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Sub for Vec4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl Neg for Vec4 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Vec4 {
        rhs * self
    }
}

impl Mul for Vec4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
            self.w * rhs.w,
        )
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl Div for Vec4 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::new(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
            self.w / rhs.w,
        )
    }
}

/// A column-major 3x3 matrix for linear transforms such as normal matrices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub data: [f32; 9],
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };

    pub const fn new(data: [f32; 9]) -> Self {
        Self { data }
    }

    pub fn get(self, row: usize, column: usize) -> f32 {
        self.data[column * 3 + row]
    }

    pub fn transpose(self) -> Self {
        Self::new([
            self.get(0, 0),
            self.get(0, 1),
            self.get(0, 2),
            self.get(1, 0),
            self.get(1, 1),
            self.get(1, 2),
            self.get(2, 0),
            self.get(2, 1),
            self.get(2, 2),
        ])
    }

    /// Returns the inverse, or `None` for a singular matrix.
    pub fn inverse(self) -> Option<Self> {
        let a = self.get(0, 0);
        let b = self.get(0, 1);
        let c = self.get(0, 2);
        let d = self.get(1, 0);
        let e = self.get(1, 1);
        let f = self.get(1, 2);
        let g = self.get(2, 0);
        let h = self.get(2, 1);
        let i = self.get(2, 2);
        let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if determinant == 0.0 || !determinant.is_finite() {
            return None;
        }
        let inverse_determinant = 1.0 / determinant;
        Some(Self::new([
            (e * i - f * h) * inverse_determinant,
            (f * g - d * i) * inverse_determinant,
            (d * h - e * g) * inverse_determinant,
            (c * h - b * i) * inverse_determinant,
            (a * i - c * g) * inverse_determinant,
            (b * g - a * h) * inverse_determinant,
            (b * f - c * e) * inverse_determinant,
            (c * d - a * f) * inverse_determinant,
            (a * e - b * d) * inverse_determinant,
        ]))
    }
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.data[0] * rhs.x + self.data[3] * rhs.y + self.data[6] * rhs.z,
            self.data[1] * rhs.x + self.data[4] * rhs.y + self.data[7] * rhs.z,
            self.data[2] * rhs.x + self.data[5] * rhs.y + self.data[8] * rhs.z,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub data: [f32; 16],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        data: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };

    pub const fn new(data: [f32; 16]) -> Self {
        Self { data }
    }

    pub fn get(self, row: usize, column: usize) -> f32 {
        self.data[column * 4 + row]
    }

    pub fn translate(offset: Vec3) -> Self {
        let mut matrix = Self::IDENTITY;
        matrix.data[12] = offset.x;
        matrix.data[13] = offset.y;
        matrix.data[14] = offset.z;
        matrix
    }

    pub fn scale(factors: Vec3) -> Self {
        let mut matrix = Self::IDENTITY;
        matrix.data[0] = factors.x;
        matrix.data[5] = factors.y;
        matrix.data[10] = factors.z;
        matrix
    }

    pub fn upper_left3(self) -> Mat3 {
        Mat3::new([
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[4],
            self.data[5],
            self.data[6],
            self.data[8],
            self.data[9],
            self.data[10],
        ])
    }

    /// Computes the inverse-transpose of the model matrix's linear part.
    pub fn normal_matrix(self) -> Option<Mat3> {
        self.upper_left3().inverse().map(Mat3::transpose)
    }

    /// Returns the inverse, or `None` for a singular matrix.
    pub fn inverse(self) -> Option<Self> {
        let mut augmented = [[0.0_f32; 8]; 4];
        for (row, values) in augmented.iter_mut().enumerate() {
            for (column, value) in values[..4].iter_mut().enumerate() {
                *value = self.get(row, column);
            }
            values[row + 4] = 1.0;
        }

        for column in 0..4 {
            let pivot = (column..4).max_by(|&left, &right| {
                augmented[left][column]
                    .abs()
                    .total_cmp(&augmented[right][column].abs())
            })?;
            if augmented[pivot][column] == 0.0 || !augmented[pivot][column].is_finite() {
                return None;
            }
            augmented.swap(column, pivot);

            let divisor = augmented[column][column];
            for value in &mut augmented[column] {
                *value /= divisor;
            }
            let pivot_row = augmented[column];
            for (row, values) in augmented.iter_mut().enumerate() {
                if row == column {
                    continue;
                }
                let factor = values[column];
                for (index, value) in values.iter_mut().enumerate() {
                    *value -= factor * pivot_row[index];
                }
            }
        }

        let mut inverse = [0.0; 16];
        for row in 0..4 {
            for column in 0..4 {
                inverse[column * 4 + row] = augmented[row][column + 4];
            }
        }
        if inverse.iter().all(|value| value.is_finite()) {
            Some(Self::new(inverse))
        } else {
            None
        }
    }

    pub fn rotate(axis: Vec3, angle: f32) -> Self {
        Quat::from_axis_angle(axis, angle).to_mat4()
    }

    /// Builds a right-handed perspective projection matrix.
    ///
    /// The preconditions are `0 < fov_y < PI`, `aspect > 0`, and
    /// `0 < near < far`.
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        debug_assert!(fov_y > 0.0 && fov_y < std::f32::consts::PI);
        debug_assert!(aspect > 0.0);
        debug_assert!(near > 0.0 && near < far);
        let focal = 1.0 / (fov_y * 0.5).tan();
        Self::new([
            focal / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            focal,
            0.0,
            0.0,
            0.0,
            0.0,
            (far + near) / (near - far),
            -1.0,
            0.0,
            0.0,
            (2.0 * far * near) / (near - far),
            0.0,
        ])
    }

    /// Builds a right-handed orthographic projection matrix.
    ///
    /// The preconditions are `left < right`, `bottom < top`, and `near < far`.
    /// View-space `z = -near` maps to NDC `z = -1`, and `z = -far` maps to
    /// NDC `z = +1`, matching [`Self::perspective`].
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        debug_assert!(left < right && bottom < top && near < far);
        Self::new([
            2.0 / (right - left),
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / (top - bottom),
            0.0,
            0.0,
            0.0,
            0.0,
            -2.0 / (far - near),
            0.0,
            -(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            -(far + near) / (far - near),
            1.0,
        ])
    }

    /// Builds a right-handed view matrix with camera forward along -Z.
    ///
    /// The preconditions are `eye != target` and a non-zero cross product of
    /// the forward direction and `up`.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward_direction = target - eye;
        debug_assert!(forward_direction.length() > 0.0);
        let forward = forward_direction.normalize();
        debug_assert!(forward.cross(up).length() > 0.0);
        let right = forward.cross(up).normalize();
        let corrected_up = right.cross(forward);
        Self::new([
            right.x,
            corrected_up.x,
            -forward.x,
            0.0,
            right.y,
            corrected_up.y,
            -forward.y,
            0.0,
            right.z,
            corrected_up.z,
            -forward.z,
            0.0,
            -right.dot(eye),
            -corrected_up.dot(eye),
            forward.dot(eye),
            1.0,
        ])
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul for Mat4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut result = [0.0; 16];
        for column in 0..4 {
            for row in 0..4 {
                result[column * 4 + row] = (0..4)
                    .map(|index| self.get(row, index) * rhs.get(index, column))
                    .sum();
            }
        }
        Self::new(result)
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4::new(
            self.data[0] * rhs.x
                + self.data[4] * rhs.y
                + self.data[8] * rhs.z
                + self.data[12] * rhs.w,
            self.data[1] * rhs.x
                + self.data[5] * rhs.y
                + self.data[9] * rhs.z
                + self.data[13] * rhs.w,
            self.data[2] * rhs.x
                + self.data[6] * rhs.y
                + self.data[10] * rhs.z
                + self.data[14] * rhs.w,
            self.data[3] * rhs.x
                + self.data[7] * rhs.y
                + self.data[11] * rhs.z
                + self.data[15] * rhs.w,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        debug_assert!(axis.length() > 0.0);
        let half = angle * 0.5;
        let axis = axis.normalize();
        let scale = half.sin();
        Self::new(axis.x * scale, axis.y * scale, axis.z * scale, half.cos())
    }

    pub fn to_mat4(self) -> Mat4 {
        let q = self.normalize();
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        Mat4::new([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
            0.0,
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
            0.0,
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])
    }

    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    pub fn rotate_vec3(self, vector: Vec3) -> Vec3 {
        let rotation = self.normalized();
        let pure = Self::new(vector.x, vector.y, vector.z, 0.0);
        let rotated = rotation * pure * rotation.conjugate();
        Vec3::new(rotated.x, rotated.y, rotated.z)
    }

    pub fn normalized(self) -> Self {
        let length = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if length == 0.0 {
            Self::IDENTITY
        } else {
            self / length
        }
    }

    fn normalize(self) -> Self {
        self.normalized()
    }
}

impl Mul for Quat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        )
    }
}

impl Mul<f32> for Quat {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Div<f32> for Quat {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn assert_vec(actual: [f32; 4], expected: [f32; 4]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(2.0, -1.0);
        assert_eq!(a + b, Vec2::new(5.0, 3.0));
        assert_eq!(a - b, Vec2::new(1.0, 5.0));
        assert_eq!(-b, Vec2::new(-2.0, 1.0));
        assert_eq!(a * 2.0, Vec2::new(6.0, 8.0));
        assert_eq!(a * b, Vec2::new(6.0, -4.0));
        assert_eq!(a / b, Vec2::new(1.5, -4.0));
        assert_eq!(a.dot(b), 2.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.normalize(), Vec2::new(0.6, 0.8));
    }

    #[test]
    fn vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -2.0, 1.0);
        assert_eq!(a + b, Vec3::new(5.0, 0.0, 4.0));
        assert_eq!(a - b, Vec3::new(-3.0, 4.0, 2.0));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a * b, Vec3::new(4.0, -4.0, 3.0));
        assert_eq!(a / b, Vec3::new(0.25, -1.0, 3.0));
        assert_eq!(a.dot(b), 3.0);
        assert_eq!(a.cross(b), Vec3::new(8.0, 11.0, -10.0));
        assert!((a.length() - 14.0_f32.sqrt()).abs() < 1e-5);
        assert_vec(
            [a.normalize().x, a.normalize().y, a.normalize().z, 0.0],
            [
                1.0 / 14.0_f32.sqrt(),
                2.0 / 14.0_f32.sqrt(),
                3.0 / 14.0_f32.sqrt(),
                0.0,
            ],
        );
    }

    #[test]
    fn vec4_operations() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(2.0, 1.0, -1.0, 2.0);
        assert_eq!(a + b, Vec4::new(3.0, 3.0, 2.0, 6.0));
        assert_eq!(a - b, Vec4::new(-1.0, 1.0, 4.0, 2.0));
        assert_eq!(-a, Vec4::new(-1.0, -2.0, -3.0, -4.0));
        assert_eq!(a * 2.0, Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(a * b, Vec4::new(2.0, 2.0, -3.0, 8.0));
        assert_eq!(a / b, Vec4::new(0.5, 2.0, -3.0, 2.0));
        assert_eq!(a.dot(b), 9.0);
        assert!((a.length() - 30.0_f32.sqrt()).abs() < 1e-5);
        assert_vec(
            [
                a.normalize().x,
                a.normalize().y,
                a.normalize().z,
                a.normalize().w,
            ],
            [
                1.0 / 30.0_f32.sqrt(),
                2.0 / 30.0_f32.sqrt(),
                3.0 / 30.0_f32.sqrt(),
                4.0 / 30.0_f32.sqrt(),
            ],
        );
    }

    #[test]
    fn mat4_constructors_and_products() {
        let translate = Mat4::translate(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(
            translate * Vec4::new(1.0, 1.0, 1.0, 1.0),
            Vec4::new(3.0, 4.0, 5.0, 1.0)
        );
        assert_eq!(
            Mat4::scale(Vec3::new(2.0, 3.0, 4.0)) * Vec4::new(1.0, 1.0, 1.0, 1.0),
            Vec4::new(2.0, 3.0, 4.0, 1.0)
        );
        assert_eq!(Mat4::IDENTITY * translate, translate);
        let combined = translate * Mat4::scale(Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(
            combined * Vec4::new(1.0, 1.0, 1.0, 1.0),
            Vec4::new(4.0, 5.0, 6.0, 1.0)
        );
        let rotated =
            Mat4::rotate(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2) * Vec4::new(1.0, 0.0, 0.0, 1.0);
        assert_vec(
            [rotated.x, rotated.y, rotated.z, rotated.w],
            [0.0, 1.0, 0.0, 1.0],
        );
        let projection = Mat4::perspective(FRAC_PI_2, 2.0, 1.0, 11.0);
        assert_vec(
            [
                projection.get(0, 0),
                projection.get(1, 1),
                projection.get(2, 2),
                projection.get(3, 2),
            ],
            [0.5, 1.0, -1.2, -1.0],
        );

        // For z = -near, clip z = (-1.2)(-1) - 2.2 = -1 and w = 1.
        // For z = -far, clip z = (-1.2)(-11) - 2.2 = 11 and w = 11.
        let near_clip = projection * Vec4::new(0.0, 0.0, -1.0, 1.0);
        assert_vec(
            [near_clip.x, near_clip.y, near_clip.z, near_clip.w],
            [0.0, 0.0, -1.0, 1.0],
        );
        let far_clip = projection * Vec4::new(0.0, 0.0, -11.0, 1.0);
        assert_vec(
            [far_clip.x, far_clip.y, far_clip.z, far_clip.w],
            [0.0, 0.0, 11.0, 11.0],
        );

        let orthographic = Mat4::orthographic(-2.0, 6.0, -3.0, 5.0, 1.0, 11.0);
        assert_vec(
            [
                orthographic.get(0, 0),
                orthographic.get(1, 1),
                orthographic.get(2, 2),
                orthographic.get(3, 3),
            ],
            [0.25, 0.25, -0.2, 1.0],
        );
        assert_eq!(
            orthographic * Vec4::new(-2.0, -3.0, -1.0, 1.0),
            Vec4::new(-1.0, -1.0, -1.0, 1.0)
        );
        assert_eq!(
            orthographic * Vec4::new(6.0, 5.0, -11.0, 1.0),
            Vec4::new(1.0, 1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn look_at_constructor() {
        let view = Mat4::look_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert_eq!(
            view * Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 0.0, -5.0, 1.0)
        );

        let diagonal = Mat4::look_at(
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 2.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let sqrt_5 = 5.0_f32.sqrt();
        // forward = (1, 2, 2) / 3 and the non-perpendicular up vector gives:
        // right = (-2, 0, 1) / sqrt(5), corrected_up = (-2, 5, -4) / (3 * sqrt(5)).
        assert_vec(
            [
                diagonal.get(0, 0),
                diagonal.get(0, 1),
                diagonal.get(0, 2),
                0.0,
            ],
            [-2.0 / sqrt_5, 0.0, 1.0 / sqrt_5, 0.0],
        );
        assert_vec(
            [
                diagonal.get(1, 0),
                diagonal.get(1, 1),
                diagonal.get(1, 2),
                0.0,
            ],
            [
                -2.0 / (3.0 * sqrt_5),
                5.0 / (3.0 * sqrt_5),
                -4.0 / (3.0 * sqrt_5),
                0.0,
            ],
        );
        assert_vec(
            [
                diagonal.get(2, 0),
                diagonal.get(2, 1),
                diagonal.get(2, 2),
                0.0,
            ],
            [-1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, 0.0],
        );
    }

    #[test]
    fn mat4_general_product() {
        let left = Mat4::new([
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,
        ]);
        let right = Mat4::new([
            2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 4.0, 5.0, 2.0,
        ]);
        // Standard row products give rows [8, 7, 11, 34], [24, 19, 27, 90],
        // [40, 31, 43, 146], and [56, 43, 59, 202].
        assert_eq!(
            left * right,
            Mat4::new([
                8.0, 24.0, 40.0, 56.0, 7.0, 19.0, 31.0, 43.0, 11.0, 27.0, 43.0, 59.0, 34.0, 90.0,
                146.0, 202.0,
            ])
        );
    }

    #[test]
    fn normal_matrix_uses_inverse_transpose_for_non_uniform_scale() {
        let model = Mat4::scale(Vec3::new(2.0, 3.0, 4.0));
        let normal_matrix = model.normal_matrix().expect("scale is invertible");
        let transformed = normal_matrix * Vec3::new(1.0, 1.0, 1.0);
        assert_vec(
            [transformed.x, transformed.y, transformed.z, 0.0],
            [0.5, 1.0 / 3.0, 0.25, 0.0],
        );
    }

    #[test]
    fn singular_mat3_has_no_inverse() {
        let matrix = Mat3::new([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 0.0]);
        assert!(matrix.inverse().is_none());
    }

    #[test]
    fn quaternion_operations() {
        let quarter_turn = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
        assert_vec(
            [
                quarter_turn.x,
                quarter_turn.y,
                quarter_turn.z,
                quarter_turn.w,
            ],
            [0.0, 0.0, 2.0_f32.sqrt() / 2.0, 2.0_f32.sqrt() / 2.0],
        );
        assert_eq!(quarter_turn * Quat::IDENTITY, quarter_turn);
        let rotated = quarter_turn.to_mat4() * Vec4::new(1.0, 0.0, 0.0, 1.0);
        assert_vec(
            [rotated.x, rotated.y, rotated.z, rotated.w],
            [0.0, 1.0, 0.0, 1.0],
        );
        let full_turn = quarter_turn * quarter_turn * quarter_turn * quarter_turn;
        assert_vec(
            [full_turn.x, full_turn.y, full_turn.z, full_turn.w],
            [0.0, 0.0, 0.0, -1.0],
        );
        assert_eq!(
            Mat4::rotate(Vec3::new(1.0, 0.0, 0.0), PI),
            Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), PI).to_mat4()
        );

        let x_turn = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), FRAC_PI_2);
        let y_turn = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_2);
        let xy = x_turn * y_turn;
        let yx = y_turn * x_turn;
        // Hamilton products are (.5, .5, .5, .5) and (.5, .5, -.5, .5).
        assert_vec([xy.x, xy.y, xy.z, xy.w], [0.5, 0.5, 0.5, 0.5]);
        assert_vec([yx.x, yx.y, yx.z, yx.w], [0.5, 0.5, -0.5, 0.5]);
        // Active xy maps (1, 2, 3) to (3, 1, 2); active yx maps it to (2, -3, -1).
        assert_eq!(
            xy.to_mat4() * Vec4::new(1.0, 2.0, 3.0, 1.0),
            Vec4::new(3.0, 1.0, 2.0, 1.0)
        );
        assert_eq!(
            yx.to_mat4() * Vec4::new(1.0, 2.0, 3.0, 1.0),
            Vec4::new(2.0, -3.0, -1.0, 1.0)
        );
    }

    #[test]
    fn quaternion_vector_rotation_is_scale_invariant() {
        let rotation = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), FRAC_PI_2);
        let vector = Vec3::new(1.0, 0.0, 0.0);
        let rotated = rotation.rotate_vec3(vector);
        let scaled_rotated = (rotation * 2.0).rotate_vec3(vector);

        assert_vec([rotated.x, rotated.y, rotated.z, 0.0], [0.0, 1.0, 0.0, 0.0]);
        assert_vec(
            [scaled_rotated.x, scaled_rotated.y, scaled_rotated.z, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        );
        assert_eq!(rotated, scaled_rotated);
    }
}
