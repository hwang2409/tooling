//! Conservative submission culling for affine mesh instances.
//!
//! Frustum planes use the Gribb-Hartmann extraction from the rows of a
//! column-major clip matrix. See Gil Gribb and Klaus Hartmann, "Fast Extraction
//! of Viewing Frustum Planes from the World-View-Projection Matrix".

use crate::math::{Mat4, Vec3, Vec4};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
    };

    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    pub const fn min(self) -> Vec3 {
        self.min
    }
    pub const fn max(self) -> Vec3 {
        self.max
    }

    pub fn from_positions(positions: impl IntoIterator<Item = Vec3>) -> Self {
        let mut positions = positions.into_iter();
        let Some(first) = positions.next() else {
            return Self::EMPTY;
        };
        let mut min = first;
        let mut max = first;
        for position in positions {
            min.x = min.x.min(position.x);
            min.y = min.y.min(position.y);
            min.z = min.z.min(position.z);
            max.x = max.x.max(position.x);
            max.y = max.y.max(position.y);
            max.z = max.z.max(position.z);
        }
        Self::new(min, max)
    }

    pub fn corners(self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    normal: Vec3,
    distance: f32,
}

impl Plane {
    pub const fn new(normal: Vec3, distance: f32) -> Self {
        Self { normal, distance }
    }
    pub const fn normal(self) -> Vec3 {
        self.normal
    }
    pub const fn distance(self) -> f32 {
        self.distance
    }
    pub fn signed_distance(self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.distance
    }

    fn normalized(self) -> Option<Self> {
        let length = self.normal.length();
        if length.is_finite() && length > 0.0 && self.distance.is_finite() {
            Some(Self::new(self.normal / length, self.distance / length))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Extracts left, right, bottom, top, near, and far planes.
    ///
    /// `Mat4` stores columns, so each row is read as `data[column * 4 + row]`.
    /// This is the row form required by Gribb-Hartmann extraction.
    pub fn from_view_projection(matrix: Mat4) -> Self {
        let rows = [
            Vec4::new(
                matrix.data[0],
                matrix.data[4],
                matrix.data[8],
                matrix.data[12],
            ),
            Vec4::new(
                matrix.data[1],
                matrix.data[5],
                matrix.data[9],
                matrix.data[13],
            ),
            Vec4::new(
                matrix.data[2],
                matrix.data[6],
                matrix.data[10],
                matrix.data[14],
            ),
            Vec4::new(
                matrix.data[3],
                matrix.data[7],
                matrix.data[11],
                matrix.data[15],
            ),
        ];
        let raw = [
            rows[3] + rows[0],
            rows[3] - rows[0],
            rows[3] + rows[1],
            rows[3] - rows[1],
            rows[3] + rows[2],
            rows[3] - rows[2],
        ];
        Self {
            planes: raw.map(|plane| {
                Plane::new(Vec3::new(plane.x, plane.y, plane.z), plane.w)
                    .normalized()
                    .unwrap_or(Plane::new(Vec3::ZERO, 0.0))
            }),
        }
    }

    pub const fn planes(self) -> [Plane; 6] {
        self.planes
    }

    /// Returns true when the transformed AABB can contribute to this pass.
    pub fn intersects_aabb(self, bounds: Aabb, model: Mat4) -> bool {
        let corners = bounds.corners().map(|corner| {
            let point = model * Vec4::new(corner.x, corner.y, corner.z, 1.0);
            if point.w == 0.0 || !point.w.is_finite() {
                Vec3::new(f32::NAN, f32::NAN, f32::NAN)
            } else {
                Vec3::new(point.x / point.w, point.y / point.w, point.z / point.w)
            }
        });
        if corners
            .iter()
            .any(|corner| !corner.x.is_finite() || !corner.y.is_finite() || !corner.z.is_finite())
        {
            return true;
        }
        self.planes.iter().all(|plane| {
            corners
                .iter()
                .any(|&corner| plane.signed_distance(corner) >= 0.0)
        })
    }

    /// Expands planes to include known casters without changing the pass.
    pub fn from_view_projection_including_bounds(
        matrix: Mat4,
        bounds: impl IntoIterator<Item = (Aabb, Mat4)>,
    ) -> Self {
        let mut frustum = Self::from_view_projection(matrix);
        for (bounds, model) in bounds {
            for plane in &mut frustum.planes {
                let minimum = bounds
                    .corners()
                    .iter()
                    .map(|&corner| {
                        let point = model * Vec4::new(corner.x, corner.y, corner.z, 1.0);
                        if point.w == 0.0 || !point.w.is_finite() {
                            0.0
                        } else {
                            plane.signed_distance(Vec3::new(
                                point.x / point.w,
                                point.y / point.w,
                                point.z / point.w,
                            ))
                        }
                    })
                    .fold(f32::INFINITY, f32::min);
                if minimum.is_finite() && minimum < 0.0 {
                    plane.distance -= minimum;
                }
            }
        }
        frustum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_known_orthographic_planes_from_matrix_rows() {
        let matrix = Mat4::orthographic(-2.0, 4.0, -3.0, 5.0, 1.0, 9.0);
        let planes = Frustum::from_view_projection(matrix).planes();
        let expected = [
            (Vec3::new(1.0, 0.0, 0.0), 2.0),
            (Vec3::new(-1.0, 0.0, 0.0), 4.0),
            (Vec3::new(0.0, 1.0, 0.0), 3.0),
            (Vec3::new(0.0, -1.0, 0.0), 5.0),
            (Vec3::new(0.0, 0.0, -1.0), -1.0),
            (Vec3::new(0.0, 0.0, 1.0), 9.0),
        ];
        for (plane, (normal, distance)) in planes.into_iter().zip(expected) {
            assert_eq!(plane.normal(), normal);
            assert!((plane.distance() - distance).abs() < 1.0e-5);
        }
    }

    #[test]
    fn culling_keeps_straddling_and_culls_fully_outside_bounds() {
        let frustum =
            Frustum::from_view_projection(Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 5.0));
        let straddling = Aabb::new(Vec3::new(-1.2, -0.25, -2.0), Vec3::new(0.25, 0.25, -1.5));
        assert!(frustum.intersects_aabb(straddling, Mat4::IDENTITY));
        let outside = Aabb::new(Vec3::new(1.1, -0.1, -2.0), Vec3::new(1.5, 0.1, -1.5));
        assert!(!frustum.intersects_aabb(outside, Mat4::IDENTITY));
    }
}
