#![allow(clippy::needless_range_loop)]

use crate::joint::JointKind;
use crate::math::{Mat3, Quat, Vec3};
use crate::spatial::Mat6;
use crate::tree::{Link, Tree, xup_for_link};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Dual {
    pub value: f32,
    pub derivative: f32,
}

impl Dual {
    pub(crate) fn zero() -> Self {
        Self {
            value: 0.0,
            derivative: 0.0,
        }
    }
}

pub(crate) trait Scalar: Copy {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f32(value: f32) -> Self;
    fn value(self) -> f32;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn quaternion_matrix(x: Self, y: Self, z: Self, w: Self) -> [[Self; 3]; 3];
    fn mat3_mul_vec(matrix: [[Self; 3]; 3], vector: [Self; 3]) -> [Self; 3];
    fn mat3_transpose_mul_vec(matrix: [[Self; 3]; 3], vector: [Self; 3]) -> [Self; 3];
}

impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f32(value: f32) -> Self {
        value
    }
    fn value(self) -> f32 {
        self
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
    fn sin(self) -> Self {
        crate::math::sin(self)
    }
    fn cos(self) -> Self {
        crate::math::cos(self)
    }
    fn quaternion_matrix(x: Self, y: Self, z: Self, w: Self) -> [[Self; 3]; 3] {
        let matrix = Quat::new(x, y, z, w).to_mat3();
        [
            [matrix.get(0, 0), matrix.get(0, 1), matrix.get(0, 2)],
            [matrix.get(1, 0), matrix.get(1, 1), matrix.get(1, 2)],
            [matrix.get(2, 0), matrix.get(2, 1), matrix.get(2, 2)],
        ]
    }
    fn mat3_mul_vec(matrix: [[Self; 3]; 3], vector: [Self; 3]) -> [Self; 3] {
        [
            matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
            matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
            matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
        ]
    }
    fn mat3_transpose_mul_vec(matrix: [[Self; 3]; 3], vector: [Self; 3]) -> [Self; 3] {
        [
            matrix[0][0] * vector[0] + matrix[1][0] * vector[1] + matrix[2][0] * vector[2],
            matrix[0][1] * vector[0] + matrix[1][1] * vector[1] + matrix[2][1] * vector[2],
            matrix[0][2] * vector[0] + matrix[1][2] * vector[1] + matrix[2][2] * vector[2],
        ]
    }
}

impl Scalar for Dual {
    fn zero() -> Self {
        Self {
            value: 0.0,
            derivative: 0.0,
        }
    }
    fn one() -> Self {
        Self {
            value: 1.0,
            derivative: 0.0,
        }
    }
    fn from_f32(value: f32) -> Self {
        Self {
            value,
            derivative: 0.0,
        }
    }
    fn value(self) -> f32 {
        self.value
    }
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivative: self.derivative + rhs.derivative,
        }
    }
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            derivative: self.derivative - rhs.derivative,
        }
    }
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivative: self.derivative * rhs.value + self.value * rhs.derivative,
        }
    }
    fn div(self, rhs: Self) -> Self {
        let denominator = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            derivative: (self.derivative * rhs.value - self.value * rhs.derivative) / denominator,
        }
    }
    fn sin(self) -> Self {
        Self {
            value: crate::math::sin(self.value),
            derivative: crate::math::cos(self.value) * self.derivative,
        }
    }
    fn cos(self) -> Self {
        Self {
            value: crate::math::cos(self.value),
            derivative: -crate::math::sin(self.value) * self.derivative,
        }
    }
    fn quaternion_matrix(x: Self, y: Self, z: Self, w: Self) -> [[Self; 3]; 3] {
        let two = Self::from_f32(2.0);
        [
            [
                Self::one().sub(two.mul(y.mul(y).add(z.mul(z)))),
                two.mul(x.mul(y).sub(z.mul(w))),
                two.mul(x.mul(z).add(y.mul(w))),
            ],
            [
                two.mul(x.mul(y).add(z.mul(w))),
                Self::one().sub(two.mul(x.mul(x).add(z.mul(z)))),
                two.mul(y.mul(z).sub(x.mul(w))),
            ],
            [
                two.mul(x.mul(z).sub(y.mul(w))),
                two.mul(y.mul(z).add(x.mul(w))),
                Self::one().sub(two.mul(x.mul(x).add(y.mul(y)))),
            ],
        ]
    }
    fn mat3_mul_vec(matrix: [[Self; 3]; 3], vector: [Self; 3]) -> [Self; 3] {
        let mut out = [Self::zero(); 3];
        for row in 0..3 {
            for col in 0..3 {
                out[row] = out[row].add(matrix[row][col].mul(vector[col]));
            }
        }
        out
    }
    fn mat3_transpose_mul_vec(matrix: [[Self; 3]; 3], vector: [Self; 3]) -> [Self; 3] {
        let mut out = [Self::zero(); 3];
        for col in 0..3 {
            for row in 0..3 {
                out[col] = out[col].add(matrix[row][col].mul(vector[row]));
            }
        }
        out
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GVec3<S: Scalar> {
    pub(crate) x: S,
    pub(crate) y: S,
    pub(crate) z: S,
}

impl<S: Scalar> GVec3<S> {
    pub(crate) fn zero() -> Self {
        Self {
            x: S::zero(),
            y: S::zero(),
            z: S::zero(),
        }
    }
    pub(crate) fn from_vec3(v: Vec3) -> Self {
        Self {
            x: S::from_f32(v.x),
            y: S::from_f32(v.y),
            z: S::from_f32(v.z),
        }
    }
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x.add(rhs.x),
            y: self.y.add(rhs.y),
            z: self.z.add(rhs.z),
        }
    }
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x.sub(rhs.x),
            y: self.y.sub(rhs.y),
            z: self.z.sub(rhs.z),
        }
    }
    fn scale(self, rhs: S) -> Self {
        Self {
            x: self.x.mul(rhs),
            y: self.y.mul(rhs),
            z: self.z.mul(rhs),
        }
    }
    fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y.mul(rhs.z).sub(self.z.mul(rhs.y)),
            y: self.z.mul(rhs.x).sub(self.x.mul(rhs.z)),
            z: self.x.mul(rhs.y).sub(self.y.mul(rhs.x)),
        }
    }
    fn dot(self, rhs: Self) -> S {
        self.x
            .mul(rhs.x)
            .add(self.y.mul(rhs.y))
            .add(self.z.mul(rhs.z))
    }
}

#[derive(Clone, Copy, Debug)]
struct GMotion<S: Scalar> {
    angular: GVec3<S>,
    linear: GVec3<S>,
}

impl<S: Scalar> GMotion<S> {
    pub(crate) fn zero() -> Self {
        Self {
            angular: GVec3::zero(),
            linear: GVec3::zero(),
        }
    }
    fn add(self, rhs: Self) -> Self {
        Self {
            angular: self.angular.add(rhs.angular),
            linear: self.linear.add(rhs.linear),
        }
    }
    fn scale(self, rhs: S) -> Self {
        Self {
            angular: self.angular.scale(rhs),
            linear: self.linear.scale(rhs),
        }
    }
    fn cross_motion(self, rhs: Self) -> Self {
        Self {
            angular: self.angular.cross(rhs.angular),
            linear: self
                .angular
                .cross(rhs.linear)
                .add(self.linear.cross(rhs.angular)),
        }
    }
    fn cross_force(self, rhs: GForce<S>) -> GForce<S> {
        GForce {
            torque: self
                .angular
                .cross(rhs.torque)
                .add(self.linear.cross(rhs.linear)),
            linear: self.angular.cross(rhs.linear),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GForce<S: Scalar> {
    pub(crate) torque: GVec3<S>,
    pub(crate) linear: GVec3<S>,
}

impl<S: Scalar> GForce<S> {
    pub(crate) fn zero() -> Self {
        Self {
            torque: GVec3::zero(),
            linear: GVec3::zero(),
        }
    }
    fn add(self, rhs: Self) -> Self {
        Self {
            torque: self.torque.add(rhs.torque),
            linear: self.linear.add(rhs.linear),
        }
    }
    fn sub(self, rhs: Self) -> Self {
        Self {
            torque: self.torque.sub(rhs.torque),
            linear: self.linear.sub(rhs.linear),
        }
    }
    fn scale(self, rhs: S) -> Self {
        Self {
            torque: self.torque.scale(rhs),
            linear: self.linear.scale(rhs),
        }
    }
}

fn force_as_motion<S: Scalar>(force: GForce<S>) -> GMotion<S> {
    GMotion {
        angular: force.torque,
        linear: force.linear,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GMat3<S: Scalar> {
    pub(crate) data: [[S; 3]; 3],
}

impl<S: Scalar> GMat3<S> {
    fn zero() -> Self {
        Self {
            data: [[S::zero(); 3]; 3],
        }
    }
    fn identity() -> Self {
        let mut out = Self::zero();
        for i in 0..3 {
            out.data[i][i] = S::one();
        }
        out
    }
    pub(crate) fn from_mat3(m: Mat3) -> Self {
        let mut out = Self::zero();
        for row in 0..3 {
            for col in 0..3 {
                out.data[row][col] = S::from_f32(m.get(row, col));
            }
        }
        out
    }
    fn mul_vec(self, v: GVec3<S>) -> GVec3<S> {
        let out = S::mat3_mul_vec(self.data, [v.x, v.y, v.z]);
        GVec3 {
            x: out[0],
            y: out[1],
            z: out[2],
        }
    }
    pub(crate) fn transpose_mul_vec(self, v: GVec3<S>) -> GVec3<S> {
        let out = S::mat3_transpose_mul_vec(self.data, [v.x, v.y, v.z]);
        GVec3 {
            x: out[0],
            y: out[1],
            z: out[2],
        }
    }
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::zero();
        for row in 0..3 {
            for col in 0..3 {
                for k in 0..3 {
                    out.data[row][col] =
                        out.data[row][col].add(self.data[row][k].mul(rhs.data[k][col]));
                }
            }
        }
        out
    }
    fn inverse(self) -> Self {
        let a = self.data;
        let d = a[0][0].mul(a[1][1]).sub(a[0][1].mul(a[1][0]));
        let det = a[0][0]
            .mul(a[1][1].mul(a[2][2]).sub(a[1][2].mul(a[2][1])))
            .sub(a[0][1].mul(a[1][0].mul(a[2][2]).sub(a[1][2].mul(a[2][0]))))
            .add(a[0][2].mul(a[1][0].mul(a[2][1]).sub(a[1][1].mul(a[2][0]))));
        let inv = S::one().div(det);
        Self {
            data: [
                [
                    a[1][1].mul(a[2][2]).sub(a[1][2].mul(a[2][1])).mul(inv),
                    a[0][2].mul(a[2][1]).sub(a[0][1].mul(a[2][2])).mul(inv),
                    a[0][1].mul(a[1][2]).sub(a[0][2].mul(a[1][1])).mul(inv),
                ],
                [
                    a[1][2].mul(a[2][0]).sub(a[1][0].mul(a[2][2])).mul(inv),
                    a[0][0].mul(a[2][2]).sub(a[0][2].mul(a[2][0])).mul(inv),
                    a[0][2].mul(a[1][0]).sub(a[0][0].mul(a[1][2])).mul(inv),
                ],
                [
                    a[1][0].mul(a[2][1]).sub(a[1][1].mul(a[2][0])).mul(inv),
                    a[0][1].mul(a[2][0]).sub(a[0][0].mul(a[2][1])).mul(inv),
                    d.mul(inv),
                ],
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GXform<S: Scalar> {
    rotation: GMat3<S>,
    translation: GVec3<S>,
}

impl<S: Scalar> GXform<S> {
    fn identity() -> Self {
        Self {
            rotation: GMat3::identity(),
            translation: GVec3::zero(),
        }
    }
    fn motion(self, m: GMotion<S>) -> GMotion<S> {
        let angular = self.rotation.mul_vec(m.angular);
        GMotion {
            angular,
            linear: self
                .rotation
                .mul_vec(m.linear)
                .add(self.translation.cross(angular)),
        }
    }
    fn transpose_force(self, f: GForce<S>) -> GForce<S> {
        let linear = self.rotation.transpose_mul_vec(f.linear);
        let torque = self
            .rotation
            .transpose_mul_vec(f.torque.sub(self.translation.cross(f.linear)));
        GForce { torque, linear }
    }
}

#[derive(Clone, Copy, Debug)]
struct GMat6<S: Scalar> {
    rows: [[S; 6]; 6],
}

impl<S: Scalar> GMat6<S> {
    fn zero() -> Self {
        Self {
            rows: [[S::zero(); 6]; 6],
        }
    }
    fn from_mat6(m: Mat6) -> Self {
        let mut out = Self::zero();
        for row in 0..6 {
            for col in 0..6 {
                out.rows[row][col] = S::from_f32(m.rows[row][col]);
            }
        }
        out
    }
    fn add(self, rhs: Self) -> Self {
        let mut out = self;
        for row in 0..6 {
            for col in 0..6 {
                out.rows[row][col] = out.rows[row][col].add(rhs.rows[row][col]);
            }
        }
        out
    }
    fn sub(self, rhs: Self) -> Self {
        let mut out = self;
        for row in 0..6 {
            for col in 0..6 {
                out.rows[row][col] = out.rows[row][col].sub(rhs.rows[row][col]);
            }
        }
        out
    }
    fn times_motion(self, m: GMotion<S>) -> GForce<S> {
        let values = [
            m.angular.x,
            m.angular.y,
            m.angular.z,
            m.linear.x,
            m.linear.y,
            m.linear.z,
        ];
        let mut out = [S::zero(); 6];
        for row in 0..6 {
            for (col, value) in values.iter().enumerate() {
                out[row] = out[row].add(self.rows[row][col].mul(*value));
            }
        }
        GForce {
            torque: GVec3 {
                x: out[0],
                y: out[1],
                z: out[2],
            },
            linear: GVec3 {
                x: out[3],
                y: out[4],
                z: out[5],
            },
        }
    }
    fn outer(force: GForce<S>, motion: GMotion<S>) -> Self {
        let rows = [
            force.torque.x,
            force.torque.y,
            force.torque.z,
            force.linear.x,
            force.linear.y,
            force.linear.z,
        ];
        let cols = [
            motion.angular.x,
            motion.angular.y,
            motion.angular.z,
            motion.linear.x,
            motion.linear.y,
            motion.linear.z,
        ];
        let mut out = Self::zero();
        for row in 0..6 {
            for col in 0..6 {
                out.rows[row][col] = rows[row].mul(cols[col]);
            }
        }
        out
    }
    fn pull_back(self, x: GXform<S>) -> Self {
        let basis = [
            GMotion {
                angular: GVec3 {
                    x: S::one(),
                    y: S::zero(),
                    z: S::zero(),
                },
                linear: GVec3::zero(),
            },
            GMotion {
                angular: GVec3 {
                    x: S::zero(),
                    y: S::one(),
                    z: S::zero(),
                },
                linear: GVec3::zero(),
            },
            GMotion {
                angular: GVec3 {
                    x: S::zero(),
                    y: S::zero(),
                    z: S::one(),
                },
                linear: GVec3::zero(),
            },
            GMotion {
                angular: GVec3::zero(),
                linear: GVec3 {
                    x: S::one(),
                    y: S::zero(),
                    z: S::zero(),
                },
            },
            GMotion {
                angular: GVec3::zero(),
                linear: GVec3 {
                    x: S::zero(),
                    y: S::one(),
                    z: S::zero(),
                },
            },
            GMotion {
                angular: GVec3::zero(),
                linear: GVec3 {
                    x: S::zero(),
                    y: S::zero(),
                    z: S::one(),
                },
            },
        ];
        let mut out = Self::zero();
        for (col, motion) in basis.into_iter().enumerate() {
            let transformed = x.motion(motion);
            let force = self.times_motion(transformed);
            let pulled = x.transpose_force(force);
            let values = [
                pulled.torque.x,
                pulled.torque.y,
                pulled.torque.z,
                pulled.linear.x,
                pulled.linear.y,
                pulled.linear.z,
            ];
            for row in 0..6 {
                out.rows[row][col] = values[row];
            }
        }
        out
    }
    fn solve(self, rhs: GForce<S>) -> GMotion<S> {
        let values = [
            rhs.torque.x,
            rhs.torque.y,
            rhs.torque.z,
            rhs.linear.x,
            rhs.linear.y,
            rhs.linear.z,
        ];
        let mut a = [[S::zero(); 7]; 6];
        for row in 0..6 {
            for col in 0..6 {
                a[row][col] = self.rows[row][col];
            }
            a[row][6] = values[row];
        }
        for col in 0..6 {
            let pivot = (col..6)
                .max_by(|row_a, row_b| {
                    a[*row_a][col]
                        .value()
                        .abs()
                        .total_cmp(&a[*row_b][col].value().abs())
                })
                .unwrap();
            a.swap(col, pivot);
            let scale = a[col][col];
            let inv = S::one().div(scale);
            let pivot_row = a[col];
            for row in 0..6 {
                if row != col {
                    let factor = a[row][col].mul(inv);
                    for k in col..7 {
                        a[row][k] = a[row][k].sub(factor.mul(pivot_row[k]));
                    }
                }
            }
        }
        GMotion {
            angular: GVec3 {
                x: a[0][6].div(a[0][0]),
                y: a[1][6].div(a[1][1]),
                z: a[2][6].div(a[2][2]),
            },
            linear: GVec3 {
                x: a[3][6].div(a[3][3]),
                y: a[4][6].div(a[4][4]),
                z: a[5][6].div(a[5][5]),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SharedAbaWorkspace<S: Scalar> {
    xup: Vec<GXform<S>>,
    s: Vec<GMotion<S>>,
    s3: Vec<[GMotion<S>; 3]>,
    v: Vec<GMotion<S>>,
    c: Vec<GMotion<S>>,
    ia: Vec<GMat6<S>>,
    pa: Vec<GForce<S>>,
    ia_s: Vec<GForce<S>>,
    d: Vec<S>,
    d3_inv: Vec<GMat3<S>>,
    tau: Vec<S>,
    tau3: Vec<GVec3<S>>,
    a: Vec<GMotion<S>>,
}

impl<S: Scalar> SharedAbaWorkspace<S> {
    pub(crate) fn new(n: usize) -> Self {
        let zero_s3 = [GMotion::zero(); 3];
        Self {
            xup: vec![GXform::identity(); n],
            s: vec![GMotion::zero(); n],
            s3: vec![zero_s3; n],
            v: vec![GMotion::zero(); n],
            c: vec![GMotion::zero(); n],
            ia: vec![GMat6::zero(); n],
            pa: vec![GForce::zero(); n],
            ia_s: vec![GForce::zero(); n],
            d: vec![S::zero(); n],
            d3_inv: vec![GMat3::zero(); n],
            tau: vec![S::zero(); n],
            tau3: vec![GVec3::zero(); n],
            a: vec![GMotion::zero(); n],
        }
    }
}

pub(crate) struct JointForces<'a, S: Scalar> {
    pub scalar: &'a [S],
    pub ball: &'a [GVec3<S>],
    pub free: GForce<S>,
}

pub(crate) fn orientations<S: Scalar>(tree: &Tree, q: &[S]) -> Vec<GMat3<S>> {
    let mut out = vec![GMat3::identity(); tree.links.len()];
    for i in 0..tree.links.len() {
        let link = &tree.links[i];
        out[i] = match link.joint {
            JointKind::Free => {
                let off = tree.q_offset[i] + 3;
                quaternion_matrix(q[off], q[off + 1], q[off + 2], q[off + 3])
            }
            JointKind::Fixed => {
                let relative =
                    link.joint_offset_in_parent.1 * link.joint_offset_in_child.1.conjugate();
                let value = GMat3::from_mat3(relative.to_mat3());
                link.parent.map_or(value, |parent| out[parent].mul(value))
            }
            JointKind::Hinge { axis, .. } => {
                let rotation = axis_rotation::<S>(axis, q[tree.q_offset[i]]);
                out[link.parent.expect("hinge parent")].mul(rotation)
            }
            JointKind::Slide { .. } => link.parent.map_or(GMat3::identity(), |parent| out[parent]),
            JointKind::Ball { .. } => {
                let off = tree.q_offset[i];
                out[link.parent.expect("ball parent")].mul(quaternion_matrix(
                    q[off],
                    q[off + 1],
                    q[off + 2],
                    q[off + 3],
                ))
            }
        };
    }
    out
}

fn quaternion_matrix<S: Scalar>(x: S, y: S, z: S, w: S) -> GMat3<S> {
    GMat3 {
        data: S::quaternion_matrix(x, y, z, w),
    }
}

fn axis_rotation<S: Scalar>(axis: Vec3, angle: S) -> GMat3<S> {
    let axis = axis.normalize();
    let x = S::from_f32(axis.x);
    let y = S::from_f32(axis.y);
    let z = S::from_f32(axis.z);
    let half = angle.mul(S::from_f32(0.5));
    let s = half.sin();
    let c = half.cos();
    quaternion_matrix(x.mul(s), y.mul(s), z.mul(s), c)
}

fn transform<S: Scalar>(link: &Link, kind: JointKind, q: S) -> GXform<S> {
    match kind {
        JointKind::Hinge { axis, .. } => {
            let axis = axis.normalize();
            let half = S::zero().sub(q).mul(S::from_f32(0.5));
            let s = half.sin();
            let c = half.cos();
            let rotation = quaternion_matrix(
                S::from_f32(axis.x).mul(s),
                S::from_f32(axis.y).mul(s),
                S::from_f32(axis.z).mul(s),
                c,
            );
            let r_pj = GVec3::from_vec3(link.joint_offset_in_parent.0);
            let r_jc = GVec3::from_vec3(link.joint_offset_in_child.0);
            GXform {
                rotation,
                translation: r_jc.sub(rotation.mul_vec(r_pj)),
            }
        }
        JointKind::Slide { axis, .. } => {
            let r_pj = GVec3::from_vec3(link.joint_offset_in_parent.0);
            let r_jc = GVec3::from_vec3(link.joint_offset_in_child.0);
            GXform {
                rotation: GMat3::identity(),
                translation: r_jc.sub(r_pj).sub(GVec3::from_vec3(axis).scale(q)),
            }
        }
        JointKind::Free => GXform::identity(),
        JointKind::Fixed => {
            let value = xup_for_link(link, 0.0);
            GXform {
                rotation: GMat3::from_mat3(value.rot_a_to_b),
                translation: GVec3::from_vec3(value.translation_a_in_b),
            }
        }
        JointKind::Ball { .. } => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run<S: Scalar>(
    tree: &Tree,
    q: &[S],
    qdot: &[S],
    external_body: &[GForce<S>],
    forces: JointForces<'_, S>,
    damping_mass: &[S],
    workspace: &mut SharedAbaWorkspace<S>,
    output: &mut [S],
) {
    let n = tree.links.len();
    for i in 0..n {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Free => {
                let off = tree.v_offset[i];
                workspace.xup[i] = GXform::identity();
                workspace.v[i] = GMotion {
                    angular: GVec3 {
                        x: qdot[off],
                        y: qdot[off + 1],
                        z: qdot[off + 2],
                    },
                    linear: GVec3 {
                        x: qdot[off + 3],
                        y: qdot[off + 4],
                        z: qdot[off + 5],
                    },
                };
                workspace.c[i] = GMotion::zero();
            }
            JointKind::Fixed => {
                workspace.xup[i] = {
                    let x = xup_for_link(link, 0.0);
                    GXform {
                        rotation: GMat3::from_mat3(x.rot_a_to_b),
                        translation: GVec3::from_vec3(x.translation_a_in_b),
                    }
                };
                let parent_v = link
                    .parent
                    .map_or(GMotion::zero(), |parent| workspace.v[parent]);
                workspace.v[i] = workspace.xup[i].motion(parent_v);
                workspace.c[i] = GMotion::zero();
            }
            JointKind::Hinge { axis, .. } | JointKind::Slide { axis, .. } => {
                let parent = link.parent.expect("joint parent");
                let xup = transform(link, link.joint, q[tree.q_offset[i]]);
                workspace.xup[i] = xup;
                workspace.s[i] = match link.joint {
                    JointKind::Hinge { .. } => {
                        let r = GVec3::from_vec3(link.joint_offset_in_child.0);
                        GMotion {
                            angular: GVec3::from_vec3(axis),
                            linear: r.cross(GVec3::from_vec3(axis)),
                        }
                    }
                    _ => GMotion {
                        angular: GVec3::zero(),
                        linear: GVec3::from_vec3(axis),
                    },
                };
                let sq = workspace.s[i].scale(qdot[tree.v_offset[i]]);
                workspace.v[i] = xup.motion(workspace.v[parent]).add(sq);
                workspace.c[i] = workspace.v[i].cross_motion(sq);
            }
            JointKind::Ball { .. } => {
                let parent = link.parent.expect("ball parent");
                let off = tree.q_offset[i];
                let rotation = quaternion_matrix(
                    S::zero().sub(q[off]),
                    S::zero().sub(q[off + 1]),
                    S::zero().sub(q[off + 2]),
                    q[off + 3],
                );
                workspace.xup[i] = GXform {
                    rotation,
                    translation: GVec3::from_vec3(link.joint_offset_in_child.0)
                        .sub(rotation.mul_vec(GVec3::from_vec3(link.joint_offset_in_parent.0))),
                };
                let r = GVec3::from_vec3(link.joint_offset_in_child.0);
                workspace.s3[i] = [
                    GMotion {
                        angular: GVec3 {
                            x: S::one(),
                            y: S::zero(),
                            z: S::zero(),
                        },
                        linear: r.cross(GVec3 {
                            x: S::one(),
                            y: S::zero(),
                            z: S::zero(),
                        }),
                    },
                    GMotion {
                        angular: GVec3 {
                            x: S::zero(),
                            y: S::one(),
                            z: S::zero(),
                        },
                        linear: r.cross(GVec3 {
                            x: S::zero(),
                            y: S::one(),
                            z: S::zero(),
                        }),
                    },
                    GMotion {
                        angular: GVec3 {
                            x: S::zero(),
                            y: S::zero(),
                            z: S::one(),
                        },
                        linear: r.cross(GVec3 {
                            x: S::zero(),
                            y: S::zero(),
                            z: S::one(),
                        }),
                    },
                ];
                let voff = tree.v_offset[i];
                let sq = workspace.s3[i][0]
                    .scale(qdot[voff])
                    .add(workspace.s3[i][1].scale(qdot[voff + 1]))
                    .add(workspace.s3[i][2].scale(qdot[voff + 2]));
                workspace.v[i] = workspace.xup[i].motion(workspace.v[parent]).add(sq);
                workspace.c[i] = workspace.v[i].cross_motion(sq);
            }
        }
    }

    for i in 0..n {
        let inertia = GMat6::from_mat6(Mat6::from_spatial_inertia(tree.links[i].spatial_inertia()));
        workspace.ia[i] = inertia;
        workspace.pa[i] = workspace.v[i]
            .cross_force(inertia.times_motion(workspace.v[i]))
            .sub(external_body[i]);
    }

    for i in (1..n).rev() {
        let link = &tree.links[i];
        let parent = link.parent.expect("non-root parent");
        match link.joint {
            JointKind::Fixed => {
                workspace.ia[parent] =
                    workspace.ia[parent].add(workspace.ia[i].pull_back(workspace.xup[i]));
                workspace.pa[parent] = workspace.pa[parent].add(workspace.xup[i].transpose_force(
                    workspace.pa[i].add(workspace.ia[i].times_motion(workspace.c[i])),
                ));
            }
            JointKind::Hinge { armature, .. } | JointKind::Slide { armature, .. } => {
                let ia_s = workspace.ia[i].times_motion(workspace.s[i]);
                let d = workspace.s[i]
                    .dot_force(ia_s)
                    .add(S::from_f32(armature))
                    .add(damping_mass[i]);
                let p_stage = workspace.pa[i].add(workspace.ia[i].times_motion(workspace.c[i]));
                let u = forces.scalar[i].sub(workspace.s[i].dot_force(p_stage));
                let qdd = u.div(d);
                let pa_full = p_stage.add(ia_s.scale(qdd));
                let inv_d = S::one().div(d);
                let ia_full =
                    workspace.ia[i].sub(GMat6::outer(ia_s, force_as_motion(ia_s)).scale(inv_d));
                workspace.ia_s[i] = ia_s;
                workspace.d[i] = d;
                workspace.tau[i] = forces.scalar[i];
                workspace.ia[parent] =
                    workspace.ia[parent].add(ia_full.pull_back(workspace.xup[i]));
                workspace.pa[parent] =
                    workspace.pa[parent].add(workspace.xup[i].transpose_force(pa_full));
            }
            JointKind::Ball { armature, .. } => {
                let s3 = workspace.s3[i];
                let ia_s3 = [
                    workspace.ia[i].times_motion(s3[0]),
                    workspace.ia[i].times_motion(s3[1]),
                    workspace.ia[i].times_motion(s3[2]),
                ];
                let mut d = GMat3::zero();
                for row in 0..3 {
                    for col in 0..3 {
                        d.data[row][col] = s3[row].dot_force(ia_s3[col]);
                    }
                }
                for k in 0..3 {
                    d.data[k][k] = d.data[k][k].add(S::from_f32(armature)).add(damping_mass[i]);
                }
                let d_inv = d.inverse();
                let p_stage = workspace.pa[i].add(workspace.ia[i].times_motion(workspace.c[i]));
                let sp = GVec3 {
                    x: s3[0].dot_force(p_stage),
                    y: s3[1].dot_force(p_stage),
                    z: s3[2].dot_force(p_stage),
                };
                let qdd = d_inv.mul_vec(forces.ball[i].sub(sp));
                let pa_full = p_stage
                    .add(ia_s3[0].scale(qdd.x))
                    .add(ia_s3[1].scale(qdd.y))
                    .add(ia_s3[2].scale(qdd.z));
                let mut ia_full = workspace.ia[i];
                for col in 0..3 {
                    let mut a_col = GForce::zero();
                    for row in 0..3 {
                        a_col = a_col.add(ia_s3[row].scale(d_inv.data[row][col]));
                    }
                    ia_full = ia_full.sub(GMat6::outer(a_col, force_as_motion(ia_s3[col])));
                }
                workspace.d3_inv[i] = d_inv;
                workspace.tau3[i] = forces.ball[i];
                workspace.ia[parent] =
                    workspace.ia[parent].add(ia_full.pull_back(workspace.xup[i]));
                workspace.pa[parent] =
                    workspace.pa[parent].add(workspace.xup[i].transpose_force(pa_full));
            }
            JointKind::Free => unreachable!(),
        }
    }

    output.fill(S::zero());
    match tree.links[0].joint {
        JointKind::Free => {
            let mut rhs = forces.free.sub(workspace.pa[0]);
            let damping = tree.links[0].free_damping;
            if damping != 0.0 {
                let off = tree.v_offset[0];
                let qdot_root = GMotion {
                    angular: GVec3 {
                        x: qdot[off],
                        y: qdot[off + 1],
                        z: qdot[off + 2],
                    },
                    linear: GVec3 {
                        x: qdot[off + 3],
                        y: qdot[off + 4],
                        z: qdot[off + 5],
                    },
                };
                rhs = rhs.sub(GForce {
                    torque: qdot_root.angular.scale(S::from_f32(damping)),
                    linear: qdot_root.linear.scale(S::from_f32(damping)),
                });
                for i in 0..6 {
                    workspace.ia[0].rows[i][i] = workspace.ia[0].rows[i][i].add(damping_mass[0]);
                }
            }
            let root = workspace.ia[0].solve(rhs);
            workspace.a[0] = root;
            let off = tree.v_offset[0];
            output[off] = root.angular.x;
            output[off + 1] = root.angular.y;
            output[off + 2] = root.angular.z;
            output[off + 3] = root.linear.x;
            output[off + 4] = root.linear.y;
            output[off + 5] = root.linear.z;
        }
        JointKind::Fixed => {}
        _ => unreachable!("only Free/Fixed roots are valid"),
    }
    for i in 1..n {
        let link = &tree.links[i];
        let parent = link.parent.expect("non-root parent");
        match link.joint {
            JointKind::Fixed => {
                workspace.a[i] = workspace.xup[i].motion(workspace.a[parent]);
            }
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                let prime = workspace.xup[i]
                    .motion(workspace.a[parent])
                    .add(workspace.c[i]);
                let inner = workspace.ia[i].times_motion(prime).add(workspace.pa[i]);
                let qdd = forces.scalar[i]
                    .sub(workspace.s[i].dot_force(inner))
                    .div(workspace.d[i]);
                workspace.a[i] = prime.add(workspace.s[i].scale(qdd));
                output[tree.v_offset[i]] = qdd;
            }
            JointKind::Ball { .. } => {
                let prime = workspace.xup[i]
                    .motion(workspace.a[parent])
                    .add(workspace.c[i]);
                let inner = workspace.ia[i].times_motion(prime).add(workspace.pa[i]);
                let s3 = workspace.s3[i];
                let rhs = GVec3 {
                    x: forces.ball[i].x.sub(s3[0].dot_force(inner)),
                    y: forces.ball[i].y.sub(s3[1].dot_force(inner)),
                    z: forces.ball[i].z.sub(s3[2].dot_force(inner)),
                };
                let qdd = workspace.d3_inv[i].mul_vec(rhs);
                workspace.a[i] = prime
                    .add(s3[0].scale(qdd.x))
                    .add(s3[1].scale(qdd.y))
                    .add(s3[2].scale(qdd.z));
                let off = tree.v_offset[i];
                output[off] = qdd.x;
                output[off + 1] = qdd.y;
                output[off + 2] = qdd.z;
            }
            JointKind::Free => unreachable!(),
        }
    }
}

trait MotionDot<S: Scalar> {
    fn dot_force(self, f: GForce<S>) -> S;
}
impl<S: Scalar> MotionDot<S> for GMotion<S> {
    fn dot_force(self, f: GForce<S>) -> S {
        self.angular.dot(f.torque).add(self.linear.dot(f.linear))
    }
}

trait MatScale<S: Scalar> {
    fn scale(self, value: S) -> Self;
}
impl<S: Scalar> MatScale<S> for GMat6<S> {
    fn scale(mut self, value: S) -> Self {
        for row in 0..6 {
            for col in 0..6 {
                self.rows[row][col] = self.rows[row][col].mul(value);
            }
        }
        self
    }
}
