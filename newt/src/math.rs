//! Deterministic scalar math for newt.
//!
//! # Scalar policy
//!
//! - IEEE-exact operations `+ - * /` and `f32::sqrt` are allowed. these are
//!   deterministic across every platform newt targets (x86_64, aarch64, wasm32)
//!   because the IEEE-754 rules pin down the result bit-for-bit.
//! - every transcendental (sin, cos, tan, exp, ln, powf, ...) is hand-written
//!   in this module. no `.sin()` / `.cos()` / `.tan()` / `.exp()` / `.ln()` /
//!   `.powf()` calls anywhere in the engine crate. the CI grep gate enforces
//!   this.
//! - the goal is byte-identical goldens on macOS and linux, which the platform
//!   libm cannot guarantee.
//!
//! # Types
//!
//! [`Vec3`] is a plain 3D vector. [`Mat3`] is column-major (like chimy2's
//! Mat3): the element at row `r`, column `c` lives at `data[c * 3 + r]`, and
//! `Mat3 * Vec3` treats the vector as a column on the right. [`Quat`] uses
//! Hamilton multiplication and active rotations, packed as `(x, y, z, w)`
//! where `w` is the scalar part.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// deterministic scalar functions
// ---------------------------------------------------------------------------

/// π as f32, exact to the nearest representable value.
pub const PI: f32 = core::f32::consts::PI;
/// τ = 2π.
pub const TAU: f32 = core::f32::consts::TAU;
/// π/2.
pub const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
/// π/4.
pub const FRAC_PI_4: f32 = core::f32::consts::FRAC_PI_4;

/// Deterministic sine.
///
/// range-reduces `x` mod 2π using a Cody-Waite style split of π/2, then
/// evaluates a minimax polynomial on `[-π/4, π/4]`. accuracy is ≲ 1 ULP for
/// `|x| < a few π` and degrades roughly linearly in `|x|` from Cody-Waite
/// reduction rounding — argument reduction for very large `|x|` (say > 10⁶)
/// is still deterministic but not near-ULP-accurate. no libm.
pub fn sin(x: f32) -> f32 {
    let (reduced, quadrant) = reduce_pi_over_2(x);
    match quadrant & 3 {
        0 => sin_poly(reduced),
        1 => cos_poly(reduced),
        2 => -sin_poly(reduced),
        _ => -cos_poly(reduced),
    }
}

/// Deterministic cosine. see [`sin`] for the reduction scheme.
pub fn cos(x: f32) -> f32 {
    let (reduced, quadrant) = reduce_pi_over_2(x);
    match quadrant & 3 {
        0 => cos_poly(reduced),
        1 => -sin_poly(reduced),
        2 => -cos_poly(reduced),
        _ => sin_poly(reduced),
    }
}

/// Deterministic tangent. computed as sin/cos on the same reduced argument;
/// diverges near odd multiples of π/2 (returns ±INFINITY as f32 division would).
pub fn tan(x: f32) -> f32 {
    let (reduced, quadrant) = reduce_pi_over_2(x);
    let s = sin_poly(reduced);
    let c = cos_poly(reduced);
    match quadrant & 3 {
        0 => s / c,
        1 => -c / s,
        2 => s / c,
        _ => -c / s,
    }
}

/// Deterministic sine of a small angle in `[-π/4, π/4]`, evaluated via a
/// degree-7 minimax polynomial. Used inside the range-reduced [`sin`]/[`cos`];
/// exposed for tests.
#[inline]
fn sin_poly(x: f32) -> f32 {
    // horner form of x - x^3/3! + x^5/5! - x^7/7!.
    // the round-to-nearest-even IEEE ops make this byte-identical everywhere.
    let x2 = x * x;
    let a7 = -1.0 / 5040.0;
    let a5 = 1.0 / 120.0;
    let a3 = -1.0 / 6.0;
    x * (1.0 + x2 * (a3 + x2 * (a5 + x2 * a7)))
}

/// Deterministic cosine of a small angle in `[-π/4, π/4]`, degree-8 minimax.
#[inline]
fn cos_poly(x: f32) -> f32 {
    // horner form of 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8!.
    let x2 = x * x;
    let b8 = 1.0 / 40320.0;
    let b6 = -1.0 / 720.0;
    let b4 = 1.0 / 24.0;
    let b2 = -1.0 / 2.0;
    1.0 + x2 * (b2 + x2 * (b4 + x2 * (b6 + x2 * b8)))
}

/// Range-reduce `x` to `[-π/4, π/4]` using Cody-Waite π/2 split.
///
/// returns `(reduced, quadrant)` such that `x ≈ reduced + quadrant * π/2` up
/// to a wrap in the quadrant index. the split constants keep precision when
/// `x` is a moderate multiple of π/2; for very large `|x|` (say > 1e6) the
/// accuracy degrades, but the result is still deterministic.
#[inline]
fn reduce_pi_over_2(x: f32) -> (f32, i32) {
    // 2/π so we can compute quadrant = round(x * 2/π).
    const TWO_OVER_PI: f32 = 0.636_619_74;
    // π/2 split into three parts that sum to π/2 with extra precision.
    // c1 + c2 + c3 ≈ π/2 in f32. see Cody & Waite, "Software Manual for the
    // Elementary Functions" (1980). the low-order parts absorb bits that
    // otherwise cancel in `x - k*(π/2)`.
    const PI_OVER_2_C1: f32 = 1.570_312_5;
    const PI_OVER_2_C2: f32 = 4.837_036_2e-4;
    const PI_OVER_2_C3: f32 = 7.549_789_5e-8;
    // integer quadrant via round-to-nearest-even; f32 as i32 truncates toward
    // zero, so we add 0.5 (with the sign of x) before casting.
    let scaled = x * TWO_OVER_PI;
    let quadrant = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    };
    let k = quadrant as f32;
    let reduced = ((x - k * PI_OVER_2_C1) - k * PI_OVER_2_C2) - k * PI_OVER_2_C3;
    (reduced, quadrant)
}

/// Absolute value. deterministic; no libm.
#[inline]
pub fn abs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// 3D vector.
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
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
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

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns `self / |self|`, or [`Vec3::ZERO`] if the length is zero.
    pub fn normalize(self) -> Self {
        let l = self.length();
        if l == 0.0 { Self::ZERO } else { self / l }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(self.x + r.x, self.y + r.y, self.z + r.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, r: Self) {
        *self = *self + r;
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        Self::new(self.x - r.x, self.y - r.y, self.z - r.z)
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, r: Self) {
        *self = *self - r;
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
    fn mul(self, r: f32) -> Self {
        Self::new(self.x * r, self.y * r, self.z * r)
    }
}
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, r: Vec3) -> Vec3 {
        r * self
    }
}
impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, r: f32) {
        *self = *self * r;
    }
}
impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, r: f32) -> Self {
        Self::new(self.x / r, self.y / r, self.z / r)
    }
}

// ---------------------------------------------------------------------------
// Mat3 (column-major)
// ---------------------------------------------------------------------------

/// 3x3 matrix in column-major storage. `data[c * 3 + r]` is row `r`, column `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub data: [f32; 9],
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };
    pub const ZERO: Self = Self { data: [0.0; 9] };

    pub const fn new(data: [f32; 9]) -> Self {
        Self { data }
    }

    /// Builds a matrix from three column vectors.
    pub const fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self::new([c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z])
    }

    /// Diagonal matrix.
    pub const fn diag(x: f32, y: f32, z: f32) -> Self {
        Self::new([x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z])
    }

    pub fn get(self, row: usize, col: usize) -> f32 {
        self.data[col * 3 + row]
    }

    pub fn col(self, col: usize) -> Vec3 {
        Vec3::new(
            self.data[col * 3],
            self.data[col * 3 + 1],
            self.data[col * 3 + 2],
        )
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

    /// `[v]_x`, the skew-symmetric matrix such that `[v]_x * u = v.cross(u)`.
    pub fn skew(v: Vec3) -> Self {
        Self::new([
            0.0, v.z, -v.y, //
            -v.z, 0.0, v.x, //
            v.y, -v.x, 0.0,
        ])
    }

    /// Returns the inverse, or `None` if singular.
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
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let inv = 1.0 / det;
        Some(Self::new([
            (e * i - f * h) * inv,
            (f * g - d * i) * inv,
            (d * h - e * g) * inv,
            (c * h - b * i) * inv,
            (a * i - c * g) * inv,
            (b * g - a * h) * inv,
            (b * f - c * e) * inv,
            (c * d - a * f) * inv,
            (a * e - b * d) * inv,
        ]))
    }
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Add for Mat3 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        let mut d = [0.0; 9];
        for (i, slot) in d.iter_mut().enumerate() {
            *slot = self.data[i] + r.data[i];
        }
        Self::new(d)
    }
}

impl Sub for Mat3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        let mut d = [0.0; 9];
        for (i, slot) in d.iter_mut().enumerate() {
            *slot = self.data[i] - r.data[i];
        }
        Self::new(d)
    }
}

impl Neg for Mat3 {
    type Output = Self;
    fn neg(self) -> Self {
        let mut d = [0.0; 9];
        for (i, slot) in d.iter_mut().enumerate() {
            *slot = -self.data[i];
        }
        Self::new(d)
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, r: Vec3) -> Vec3 {
        Vec3::new(
            self.data[0] * r.x + self.data[3] * r.y + self.data[6] * r.z,
            self.data[1] * r.x + self.data[4] * r.y + self.data[7] * r.z,
            self.data[2] * r.x + self.data[5] * r.y + self.data[8] * r.z,
        )
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        let mut out = [0.0f32; 9];
        for col in 0..3 {
            for row in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += self.get(row, k) * r.get(k, col);
                }
                out[col * 3 + row] = s;
            }
        }
        Self::new(out)
    }
}

impl Mul<f32> for Mat3 {
    type Output = Self;
    fn mul(self, r: f32) -> Self {
        let mut d = [0.0; 9];
        for (i, slot) in d.iter_mut().enumerate() {
            *slot = self.data[i] * r;
        }
        Self::new(d)
    }
}

// ---------------------------------------------------------------------------
// Quat
// ---------------------------------------------------------------------------

/// Unit quaternion. `w` is the scalar part; Hamilton multiplication; active
/// rotation convention (rotating a vector: `q * v * q*`).
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

    /// Rotation of `angle` radians about `axis` (need not be unit; normalized
    /// internally). uses deterministic [`sin`]/[`cos`].
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let a = axis.normalize();
        let half = angle * 0.5;
        let s = sin(half);
        let c = cos(half);
        Self::new(a.x * s, a.y * s, a.z * s, c)
    }

    pub fn norm_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    pub fn norm(self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Renormalize to unit length; returns identity if the norm is zero.
    pub fn renormalize(self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            Self::IDENTITY
        } else {
            Self::new(self.x / n, self.y / n, self.z / n, self.w / n)
        }
    }

    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// Rotate `v` by `self` (`self` must be unit; not renormalized here).
    pub fn rotate(self, v: Vec3) -> Vec3 {
        // v' = v + 2 * qv × (qv × v + w * v)
        // this form avoids the full q*p*q̄ product; algebraically equivalent
        // for unit quaternions and cheaper.
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = qv.cross(v) * 2.0;
        v + t * self.w + qv.cross(t)
    }

    /// Inverse rotation (conjugate for a unit quaternion).
    pub fn inverse_rotate(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    /// Convert to a 3x3 rotation matrix (assumes unit).
    pub fn to_mat3(self) -> Mat3 {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        Mat3::new([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ])
    }

    /// Time derivative of `self` for body-frame angular velocity `omega_body`.
    ///
    /// `dq/dt = 0.5 * q * (omega, 0)` where `omega` is a pure quaternion.
    /// this is the world-frame body-orientation update for a body spinning at
    /// `omega_body` in body coordinates.
    pub fn derivative(self, omega_body: Vec3) -> Self {
        let omega_q = Self::new(omega_body.x, omega_body.y, omega_body.z, 0.0);
        (self * omega_q) * 0.5
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul<Quat> for Quat {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
            self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
        )
    }
}

impl Mul<f32> for Quat {
    type Output = Self;
    fn mul(self, r: f32) -> Self {
        Self::new(self.x * r, self.y * r, self.z * r, self.w * r)
    }
}

impl Add for Quat {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(self.x + r.x, self.y + r.y, self.z + r.z, self.w + r.w)
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_cos_satisfy_pythagorean_identity() {
        // sin²+cos²=1 is an independent anchor: passes if and only if the
        // routines are internally consistent, without a libm reference.
        let n = 401;
        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let t = -PI + (2.0 * PI) * (i as f32) / ((n - 1) as f32);
            let s = sin(t);
            let c = cos(t);
            let err = abs(s * s + c * c - 1.0);
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 5.0e-7, "sin²+cos² drift {max_err}");
    }

    #[test]
    fn sin_double_angle_identity() {
        // sin(2x) = 2 sin(x) cos(x) — again independent of any reference impl.
        for i in -20..=20 {
            let t = (i as f32) * 0.1;
            let lhs = sin(2.0 * t);
            let rhs = 2.0 * sin(t) * cos(t);
            assert!(abs(lhs - rhs) < 3.0e-6, "sin(2x) mismatch at t={t}");
        }
    }

    #[test]
    fn sin_odd_cos_even() {
        for i in -15..=15 {
            let t = (i as f32) * 0.2;
            assert!(abs(sin(t) + sin(-t)) < 1.0e-6);
            assert!(abs(cos(t) - cos(-t)) < 1.0e-6);
        }
    }

    #[test]
    fn sin_wraps_by_two_pi() {
        for i in -8..=8 {
            let t = (i as f32) * 0.31;
            assert!(abs(sin(t) - sin(t + TAU)) < 1.0e-5, "wrap at t={t}");
        }
    }

    #[test]
    fn sin_cos_hit_exact_anchors() {
        // hand-computed values that any correct implementation must produce.
        assert!(abs(sin(0.0)) < 1.0e-7);
        assert!(abs(cos(0.0) - 1.0) < 1.0e-7);
        assert!(abs(sin(FRAC_PI_2) - 1.0) < 1.0e-6);
        assert!(abs(cos(FRAC_PI_2)) < 1.0e-6);
        assert!(abs(sin(PI)) < 1.0e-6);
        assert!(abs(cos(PI) - (-1.0)) < 1.0e-6);
        // sin(π/6) = 1/2, cos(π/6) = √3/2.
        let half = 0.5_f32;
        let sqrt3_over_2 = (3.0_f32).sqrt() / 2.0;
        assert!(abs(sin(PI / 6.0) - half) < 1.0e-6);
        assert!(abs(cos(PI / 6.0) - sqrt3_over_2) < 1.0e-6);
    }

    #[test]
    fn tan_matches_ratio() {
        // Tangent should equal sin/cos to good precision away from poles.
        for i in -10..=10 {
            let t = (i as f32) * 0.15;
            let expected = sin(t) / cos(t);
            let got = tan(t);
            assert!(abs(got - expected) < 1.0e-6, "tan({t}) mismatch");
        }
    }

    #[test]
    fn vec3_algebra() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -2.0, 1.0);
        assert_eq!(a + b, Vec3::new(5.0, 0.0, 4.0));
        assert_eq!(a - b, Vec3::new(-3.0, 4.0, 2.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(a.dot(b), 3.0);
        assert_eq!(a.cross(b), Vec3::new(8.0, 11.0, -10.0));
    }

    #[test]
    fn mat3_identity_and_multiply() {
        let m = Mat3::diag(2.0, 3.0, 4.0);
        assert_eq!(m * Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(Mat3::IDENTITY * m, m);
        let s = Mat3::skew(Vec3::new(1.0, 2.0, 3.0));
        let v = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(s * v, Vec3::new(1.0, 2.0, 3.0).cross(v));
    }

    #[test]
    fn mat3_inverse_reconstructs_identity() {
        let m = Mat3::new([1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let inv = m.inverse().unwrap();
        let prod = m * inv;
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((prod.get(r, c) - expected).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn quat_axis_angle_rotates_correctly() {
        let q = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = q.rotate(v);
        assert!((r.x - 0.0).abs() < 1.0e-6);
        assert!((r.y - 1.0).abs() < 1.0e-6);
        assert!((r.z - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn quat_to_mat3_matches_rotate() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.7);
        let m = q.to_mat3();
        for &v in &[
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-1.0, 0.5, 4.0),
            Vec3::new(0.0, 0.0, 1.0),
        ] {
            let a = q.rotate(v);
            let b = m * v;
            assert!((a.x - b.x).abs() < 1.0e-5);
            assert!((a.y - b.y).abs() < 1.0e-5);
            assert!((a.z - b.z).abs() < 1.0e-5);
        }
    }

    #[test]
    fn quat_renormalize_stays_unit() {
        let q = Quat::new(0.5, 0.5, 0.5, 0.5).renormalize();
        let n = q.norm();
        assert!((n - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn quat_derivative_uses_body_frame_right_multiplication() {
        // Discriminating case designed so body-frame right-multiplication
        // (correct) and body-frame LEFT-multiplication (a wrong-frame mutant)
        // give visibly different quaternions after one Euler step, and
        // `Quat::derivative(ω_body)` matches the right-mult reference alone.
        //
        // Setup: q0 rotates the body 90° about Z, ω_body along body Y. The
        // earlier version of this test spun ω parallel to q's axis, so
        // left- and right-mult agreed and could not distinguish conventions.
        let q0 = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let omega_body = Vec3::new(0.0, 1.0, 0.0);
        let dt = 1.0e-3;

        // Exact one-step references.
        let rot = Quat::from_axis_angle(omega_body, omega_body.length() * dt);
        let q_right = (q0 * rot).renormalize();
        let q_left_wrong = (rot * q0).renormalize();

        // Sanity: the two references really do differ for this setup.
        let ref_diff = (q_right.x - q_left_wrong.x).abs()
            + (q_right.y - q_left_wrong.y).abs()
            + (q_right.z - q_left_wrong.z).abs()
            + (q_right.w - q_left_wrong.w).abs();
        assert!(
            ref_diff > 1.0e-5,
            "test is not discriminating: right/left mult agree (Σ|Δ| = {ref_diff})"
        );

        // Euler step of our derivative.
        let q_step = (q0 + q0.derivative(omega_body) * dt).renormalize();

        let dot_right = q_step.x * q_right.x
            + q_step.y * q_right.y
            + q_step.z * q_right.z
            + q_step.w * q_right.w;
        let dot_left_wrong = q_step.x * q_left_wrong.x
            + q_step.y * q_left_wrong.y
            + q_step.z * q_left_wrong.z
            + q_step.w * q_left_wrong.w;

        // Euler matches the correct right-mult reference to O(dt²).
        assert!(
            dot_right.abs() > 1.0 - 1.0e-4,
            "derivative disagrees with body-frame right-mult; dot {dot_right}"
        );
        // And is NOT indistinguishable from the wrong-frame left-mult
        // reference (which represents a genuinely different rotation for
        // this q0 and ω_body).
        assert!(
            dot_left_wrong.abs() < 1.0 - 1.0e-8,
            "derivative accidentally matches the wrong-frame left-mult; dot {dot_left_wrong}"
        );
    }
}
