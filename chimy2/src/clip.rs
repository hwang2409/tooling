//! Clip-space near-plane clipping and backface culling.
//!
//! Only the near plane is clipped in M3. The condition is z + w >= 0 in
//! homogeneous clip space. Screen-space viewport clamping handles the other
//! four side planes. Culling uses signed NDC area before viewport Y inversion.
//! Positive NDC area is the front-face convention.

use crate::math::{Vec3, Vec4};
use crate::pipeline::Varyings;
use std::ops::Index;

#[derive(Clone, Debug, PartialEq)]
pub struct ClipVertex<V> {
    pub position: Vec4,
    pub varyings: V,
}

impl<V> ClipVertex<V> {
    pub const fn new(position: Vec4, varyings: V) -> Self {
        Self { position, varyings }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClippedTriangles<V> {
    triangles: [Option<[ClipVertex<V>; 3]>; 2],
    count: usize,
}

impl<V> ClippedTriangles<V> {
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &[ClipVertex<V>; 3]> {
        self.triangles[..self.count]
            .iter()
            .filter_map(Option::as_ref)
    }
}

pub struct ClippedTrianglesIntoIter<V> {
    triangles: [Option<[ClipVertex<V>; 3]>; 2],
    index: usize,
}

impl<V> Iterator for ClippedTrianglesIntoIter<V> {
    type Item = [ClipVertex<V>; 3];

    fn next(&mut self) -> Option<Self::Item> {
        let triangle = self.triangles.get_mut(self.index)?.take();
        self.index += 1;
        triangle
    }
}

impl<V> IntoIterator for ClippedTriangles<V> {
    type Item = [ClipVertex<V>; 3];
    type IntoIter = ClippedTrianglesIntoIter<V>;

    fn into_iter(self) -> Self::IntoIter {
        ClippedTrianglesIntoIter {
            triangles: self.triangles,
            index: 0,
        }
    }
}

impl<V> Index<usize> for ClippedTriangles<V> {
    type Output = [ClipVertex<V>; 3];

    fn index(&self, index: usize) -> &Self::Output {
        self.triangles[index]
            .as_ref()
            .expect("clip triangle index out of range")
    }
}

pub fn clip_triangle_near<V: Varyings + Clone>(
    triangle: [ClipVertex<V>; 3],
) -> ClippedTriangles<V> {
    let mut polygon: [Option<ClipVertex<V>>; 4] = std::array::from_fn(|_| None);
    polygon[0] = Some(triangle[0].clone());
    polygon[1] = Some(triangle[1].clone());
    polygon[2] = Some(triangle[2].clone());
    let mut polygon_len = 3;

    let mut clipped: [Option<ClipVertex<V>>; 4] = std::array::from_fn(|_| None);
    let mut clipped_len = 0;
    for index in 0..polygon_len {
        let current = polygon[index].as_ref().expect("clip polygon entry");
        let next = polygon[(index + 1) % polygon_len]
            .as_ref()
            .expect("clip polygon entry");
        let current_inside = inside_near(current.position);
        let next_inside = inside_near(next.position);
        match (current_inside, next_inside) {
            (true, true) => push_unique(&mut clipped, &mut clipped_len, next.clone()),
            (true, false) => {
                push_unique(&mut clipped, &mut clipped_len, intersection(current, next));
            }
            (false, true) => {
                push_unique(&mut clipped, &mut clipped_len, intersection(current, next));
                push_unique(&mut clipped, &mut clipped_len, next.clone());
            }
            (false, false) => {}
        }
    }

    if clipped_len > 1
        && clipped[0].as_ref().expect("clip polygon entry").position
            == clipped[clipped_len - 1]
                .as_ref()
                .expect("clip polygon entry")
                .position
    {
        clipped_len -= 1;
    }
    polygon = clipped;
    polygon_len = clipped_len;

    let mut triangles = std::array::from_fn(|_| None);
    let count = match polygon_len {
        0..=2 => 0,
        3 => {
            triangles[0] = Some([
                polygon[0].as_ref().expect("clip polygon entry").clone(),
                polygon[1].as_ref().expect("clip polygon entry").clone(),
                polygon[2].as_ref().expect("clip polygon entry").clone(),
            ]);
            1
        }
        4 => {
            let first = polygon[0].as_ref().expect("clip polygon entry");
            let second = polygon[1].as_ref().expect("clip polygon entry");
            let third = polygon[2].as_ref().expect("clip polygon entry");
            let fourth = polygon[3].as_ref().expect("clip polygon entry");
            triangles[0] = Some([first.clone(), second.clone(), third.clone()]);
            triangles[1] = Some([first.clone(), third.clone(), fourth.clone()]);
            2
        }
        _ => unreachable!("near-plane clipping produced too many vertices"),
    };

    ClippedTriangles { triangles, count }
}

fn push_unique<V>(
    polygon: &mut [Option<ClipVertex<V>>; 4],
    length: &mut usize,
    vertex: ClipVertex<V>,
) {
    if *length > 0
        && polygon[*length - 1]
            .as_ref()
            .expect("clip polygon entry")
            .position
            == vertex.position
    {
        return;
    }
    polygon[*length] = Some(vertex);
    *length += 1;
}

pub fn clip_triangle<V: Varyings + Clone>(triangle: [ClipVertex<V>; 3]) -> ClippedTriangles<V> {
    clip_triangle_near(triangle)
}

pub fn inside_near(position: Vec4) -> bool {
    position.z + position.w >= 0.0
}

fn intersection<V: Varyings + Clone>(a: &ClipVertex<V>, b: &ClipVertex<V>) -> ClipVertex<V> {
    let a_distance = a.position.z + a.position.w;
    let b_distance = b.position.z + b.position.w;
    if a_distance == 0.0 {
        return a.clone();
    }
    if b_distance == 0.0 {
        return b.clone();
    }
    let denominator = a_distance - b_distance;
    let t = if denominator == 0.0 {
        0.0
    } else {
        (a_distance / denominator).clamp(0.0, 1.0)
    };
    ClipVertex::new(
        a.position + (b.position - a.position) * t,
        V::lerp(&a.varyings, &b.varyings, t),
    )
}

pub fn ndc_signed_area<V>(triangle: &[ClipVertex<V>; 3]) -> Option<f32> {
    let mut points = [Vec3::ZERO; 3];
    for (point, vertex) in points.iter_mut().zip(triangle) {
        if vertex.position.w == 0.0 {
            return None;
        }
        *point = Vec3::new(
            vertex.position.x / vertex.position.w,
            vertex.position.y / vertex.position.w,
            vertex.position.z / vertex.position.w,
        );
        if !point.x.is_finite() || !point.y.is_finite() {
            return None;
        }
    }
    Some(
        (points[1].x - points[0].x) * (points[2].y - points[0].y)
            - (points[1].y - points[0].y) * (points[2].x - points[0].x),
    )
}

pub fn is_front_facing<V>(triangle: &[ClipVertex<V>; 3]) -> bool {
    ndc_signed_area(triangle).is_some_and(|area| area > 0.0)
}

pub fn cull_backface<V>(triangle: &[ClipVertex<V>; 3]) -> bool {
    !is_front_facing(triangle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec4;

    fn vertex(x: f32, y: f32, z: f32, value: f32) -> ClipVertex<f32> {
        ClipVertex::new(Vec4::new(x, y, z, 1.0), value)
    }

    impl Varyings for f32 {
        type Derivatives = ();

        fn lerp3(a: &Self, b: &Self, c: &Self, weights: Vec3) -> Self {
            *a * weights.x + *b * weights.y + *c * weights.z
        }
    }

    #[test]
    fn near_plane_keeps_triangle_inside() {
        let result = clip_triangle_near([
            vertex(-1.0, -1.0, 0.0, 1.0),
            vertex(1.0, -1.0, 0.0, 2.0),
            vertex(0.0, 1.0, 0.0, 3.0),
        ]);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0]
                .iter()
                .map(|vertex| vertex.varyings)
                .collect::<Vec<_>>(),
            vec![2.0, 3.0, 1.0]
        );
    }

    #[test]
    fn near_plane_turns_one_outside_vertex_into_two_triangles() {
        let result = clip_triangle_near([
            ClipVertex::new(Vec4::new(-1.0, -1.0, -2.0, 1.0), 1.0),
            vertex(1.0, -1.0, 0.0, 2.0),
            vertex(0.0, 1.0, 0.0, 3.0),
        ]);
        assert_eq!(result.len(), 2);
        assert!(
            result
                .iter()
                .flatten()
                .all(|vertex| inside_near(vertex.position))
        );
        assert!(
            result
                .iter()
                .flatten()
                .any(|vertex| { (vertex.position.z + vertex.position.w).abs() < 1e-5 })
        );
    }

    #[test]
    fn near_plane_discards_triangle_outside() {
        let result = clip_triangle_near([
            vertex(-1.0, -1.0, -2.0, 1.0),
            vertex(1.0, -1.0, -2.0, 2.0),
            vertex(0.0, 1.0, -2.0, 3.0),
        ]);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn near_plane_on_plane_vertex_emits_one_triangle() {
        let result = clip_triangle([
            vertex(-1.0, -1.0, -1.0, 1.0),
            vertex(1.0, -1.0, 0.0, 1.0),
            vertex(0.0, 1.0, 0.0, 1.0),
        ]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn near_plane_one_inside_vertex_emits_one_triangle() {
        let result = clip_triangle([
            vertex(-1.0, -1.0, 0.0, 1.0),
            vertex(1.0, -1.0, -2.0, 1.0),
            vertex(0.0, 1.0, -2.0, 1.0),
        ]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn clip_intersection_uses_homogeneous_coordinates() {
        let result = clip_triangle([
            ClipVertex::new(Vec4::new(-3.0, -1.0, -2.0, 1.0), 10.0),
            ClipVertex::new(Vec4::new(3.0, 2.0, 0.0, 2.0), 20.0),
            ClipVertex::new(Vec4::new(0.0, 4.0, 0.0, 3.0), 30.0),
        ]);
        assert_eq!(result.len(), 2);
        assert_close_vec4(
            result[0][0].position,
            Vec4::new(-1.0, 0.0, -4.0 / 3.0, 4.0 / 3.0),
        );
        assert_close_vec4(result[1][2].position, Vec4::new(-2.25, 0.25, -1.5, 1.5));
        assert!((result[0][0].varyings - 40.0 / 3.0).abs() < 1e-5);
        assert!((result[1][2].varyings - 15.0).abs() < 1e-5);
    }

    fn assert_close_vec4(actual: Vec4, expected: Vec4) {
        assert!((actual.x - expected.x).abs() < 1e-5);
        assert!((actual.y - expected.y).abs() < 1e-5);
        assert!((actual.z - expected.z).abs() < 1e-5);
        assert!((actual.w - expected.w).abs() < 1e-5);
    }

    #[test]
    fn positive_ndc_area_is_front_facing() {
        let triangle = [
            vertex(-1.0, -1.0, 0.0, 1.0),
            vertex(1.0, -1.0, 0.0, 1.0),
            vertex(0.0, 1.0, 0.0, 1.0),
        ];
        assert!(is_front_facing(&triangle));
        let reversed = [
            triangle[0].clone(),
            triangle[2].clone(),
            triangle[1].clone(),
        ];
        assert!(cull_backface(&reversed));
    }
}
