//! Dynamic AABB tree broad phase.
//!
//! The tree stores a proxy's world AABB with a 10% extent margin and a small
//! absolute epsilon. Updates inside that fat bound only update the proxy's
//! position logically; they do not remove and reinsert the leaf. This keeps
//! steady-state motion cheap while retaining conservative overlap results.

use crate::geom::{ConvexMesh, Geom, GeomPose, GeomShape, HeightField};
use crate::math::{Vec3, abs};

const FATNESS: f32 = 0.1;
const FAT_EPSILON: f32 = 1.0e-4;

/// An axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const UNBOUNDED: Self = Self {
        min: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        max: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
    };

    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_center_extents(center: Vec3, extents: Vec3) -> Self {
        Self::new(center - extents, center + extents)
    }

    fn fatten(self) -> Self {
        let extent = self.max - self.min;
        let margin = Vec3::new(
            (extent.x * FATNESS).max(FAT_EPSILON),
            (extent.y * FATNESS).max(FAT_EPSILON),
            (extent.z * FATNESS).max(FAT_EPSILON),
        );
        Self::new(self.min - margin, self.max + margin)
    }

    pub fn union(self, other: Self) -> Self {
        Self::new(
            Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        )
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    fn contains(self, other: Self) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.min.z <= other.min.z
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
            && self.max.z >= other.max.z
    }

    fn perimeter(self) -> f32 {
        let extent = self.max - self.min;
        2.0 * (extent.x + extent.y + extent.z)
    }

    fn ray_hits(self, ray: Ray) -> bool {
        let mut near: f32 = 0.0;
        let mut far: f32 = f32::INFINITY;
        for (origin, direction, min, max) in [
            (ray.origin.x, ray.direction.x, self.min.x, self.max.x),
            (ray.origin.y, ray.direction.y, self.min.y, self.max.y),
            (ray.origin.z, ray.direction.z, self.min.z, self.max.z),
        ] {
            if direction == 0.0 {
                if origin < min || origin > max {
                    return false;
                }
                continue;
            }
            let inverse = 1.0 / direction;
            let mut axis_near = (min - origin) * inverse;
            let mut axis_far = (max - origin) * inverse;
            if axis_near > axis_far {
                core::mem::swap(&mut axis_near, &mut axis_far);
            }
            near = near.max(axis_near);
            far = far.min(axis_far);
            if near > far {
                return false;
            }
        }
        true
    }
}

/// A half-line used by [`DynamicAabbTree::query_ray`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    aabb: Aabb,
    parent: Option<usize>,
    child1: Option<usize>,
    child2: Option<usize>,
    proxy: Option<usize>,
    height: i32,
}

impl Node {
    fn leaf(proxy: usize, aabb: Aabb) -> Self {
        Self {
            aabb,
            parent: None,
            child1: None,
            child2: None,
            proxy: Some(proxy),
            height: 0,
        }
    }

    fn branch(aabb: Aabb, child1: usize, child2: usize) -> Self {
        Self {
            aabb,
            parent: None,
            child1: Some(child1),
            child2: Some(child2),
            proxy: None,
            height: 1,
        }
    }

    fn empty() -> Self {
        Self::leaf(usize::MAX, Aabb::new(Vec3::ZERO, Vec3::ZERO))
    }

    fn is_leaf(self) -> bool {
        self.proxy.is_some()
    }
}

/// A dynamic, index-stable AABB tree.
#[derive(Clone, Debug)]
pub struct DynamicAabbTree {
    nodes: Vec<Node>,
    free_nodes: Vec<usize>,
    proxy_nodes: Vec<Option<usize>>,
    root: Option<usize>,
    pair_stack: Vec<(usize, usize)>,
    pairs: Vec<(usize, usize)>,
    query_stack: Vec<usize>,
}

impl Default for DynamicAabbTree {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicAabbTree {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_nodes: Vec::new(),
            proxy_nodes: Vec::new(),
            root: None,
            pair_stack: Vec::new(),
            pairs: Vec::new(),
            query_stack: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.proxy_nodes
            .iter()
            .filter(|node| node.is_some())
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn contains_proxy(&self, proxy: usize) -> bool {
        self.proxy_nodes.get(proxy).is_some_and(Option::is_some)
    }

    pub fn proxy_aabb(&self, proxy: usize) -> Option<Aabb> {
        self.proxy_nodes
            .get(proxy)
            .and_then(|node| *node)
            .map(|node| self.nodes[node].aabb)
    }

    pub fn insert(&mut self, proxy: usize, bounds: Aabb) {
        assert!(!self.contains_proxy(proxy), "proxy is already in the tree");
        let leaf = self.allocate_node(Node::leaf(proxy, bounds.fatten()));
        if self.proxy_nodes.len() <= proxy {
            self.proxy_nodes.resize(proxy + 1, None);
        }
        self.proxy_nodes[proxy] = Some(leaf);
        self.insert_leaf(leaf);
    }

    pub fn remove(&mut self, proxy: usize) -> bool {
        let Some(leaf) = self.proxy_nodes.get_mut(proxy).and_then(Option::take) else {
            return false;
        };
        self.remove_leaf(leaf);
        self.release_node(leaf);
        true
    }

    /// Update a proxy. Returns true only when the leaf was reinserted.
    pub fn update(&mut self, proxy: usize, bounds: Aabb) -> bool {
        let Some(leaf) = self.proxy_nodes.get(proxy).and_then(|node| *node) else {
            self.insert(proxy, bounds);
            return true;
        };
        if self.nodes[leaf].aabb.contains(bounds) {
            return false;
        }
        let removed = self.remove(proxy);
        debug_assert!(removed);
        self.insert(proxy, bounds);
        true
    }

    /// Return all overlapping fat-bound proxy pairs in lexicographic order.
    pub fn compute_pairs(&mut self) -> &[(usize, usize)] {
        self.pairs.clear();
        self.pair_stack.clear();
        let Some(root) = self.root else {
            return &self.pairs;
        };
        if self.nodes[root].is_leaf() {
            return &self.pairs;
        }
        self.pair_stack.push((root, root));
        while let Some((a, b)) = self.pair_stack.pop() {
            if a == b {
                if self.nodes[a].is_leaf() {
                    continue;
                }
                let node = self.nodes[a];
                let child1 = node.child1.unwrap();
                let child2 = node.child2.unwrap();
                self.pair_stack.push((child2, child2));
                self.pair_stack.push((child1, child1));
                self.pair_stack.push((child1, child2));
                continue;
            }
            if !self.nodes[a].aabb.overlaps(self.nodes[b].aabb) {
                continue;
            }
            let a_leaf = self.nodes[a].is_leaf();
            let b_leaf = self.nodes[b].is_leaf();
            match (a_leaf, b_leaf) {
                (true, true) => {
                    let proxy_a = self.nodes[a].proxy.unwrap();
                    let proxy_b = self.nodes[b].proxy.unwrap();
                    let pair = if proxy_a < proxy_b {
                        (proxy_a, proxy_b)
                    } else {
                        (proxy_b, proxy_a)
                    };
                    self.pairs.push(pair);
                }
                (true, false) => self.push_children_against(a, b),
                (false, true) => self.push_children_against(b, a),
                (false, false) => {
                    let a_node = self.nodes[a];
                    let b_node = self.nodes[b];
                    let a1 = a_node.child1.unwrap();
                    let a2 = a_node.child2.unwrap();
                    let b1 = b_node.child1.unwrap();
                    let b2 = b_node.child2.unwrap();
                    self.pair_stack.push((a2, b2));
                    self.pair_stack.push((a2, b1));
                    self.pair_stack.push((a1, b2));
                    self.pair_stack.push((a1, b1));
                }
            }
        }
        self.pairs.sort_unstable();
        self.pairs.dedup();
        &self.pairs
    }

    /// Visit AABB hits. Return false when the callback requested early exit.
    pub fn query_aabb<F>(&mut self, bounds: Aabb, mut callback: F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        self.query_stack.clear();
        if let Some(root) = self.root {
            self.query_stack.push(root);
        }
        while let Some(node) = self.query_stack.pop() {
            let current = self.nodes[node];
            if !current.aabb.overlaps(bounds) {
                continue;
            }
            if current.is_leaf() {
                if !callback(current.proxy.unwrap()) {
                    return false;
                }
            } else {
                self.query_stack.push(current.child2.unwrap());
                self.query_stack.push(current.child1.unwrap());
            }
        }
        true
    }

    /// Visit ray hits. Return false when the callback requested early exit.
    pub fn query_ray<F>(&mut self, ray: Ray, mut callback: F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        self.query_stack.clear();
        if let Some(root) = self.root {
            self.query_stack.push(root);
        }
        while let Some(node) = self.query_stack.pop() {
            let current = self.nodes[node];
            if !current.aabb.ray_hits(ray) {
                continue;
            }
            if current.is_leaf() {
                if !callback(current.proxy.unwrap()) {
                    return false;
                }
            } else {
                self.query_stack.push(current.child2.unwrap());
                self.query_stack.push(current.child1.unwrap());
            }
        }
        true
    }

    fn push_children_against(&mut self, leaf: usize, branch: usize) {
        let node = self.nodes[branch];
        let child1 = node.child1.unwrap();
        let child2 = node.child2.unwrap();
        self.pair_stack.push((leaf, child2));
        self.pair_stack.push((leaf, child1));
    }

    fn allocate_node(&mut self, node: Node) -> usize {
        if let Some(index) = self.free_nodes.pop() {
            self.nodes[index] = node;
            index
        } else {
            let index = self.nodes.len();
            self.nodes.push(node);
            index
        }
    }

    fn release_node(&mut self, index: usize) {
        self.nodes[index] = Node::empty();
        self.free_nodes.push(index);
    }

    fn insert_leaf(&mut self, leaf: usize) {
        let Some(root) = self.root else {
            self.root = Some(leaf);
            return;
        };
        let leaf_bounds = self.nodes[leaf].aabb;
        let mut sibling = root;
        while !self.nodes[sibling].is_leaf() {
            let node = self.nodes[sibling];
            let child1 = node.child1.unwrap();
            let child2 = node.child2.unwrap();
            let combined = node.aabb.union(leaf_bounds);
            let inherited = 2.0 * (combined.perimeter() - node.aabb.perimeter());
            let cost1 = self.insertion_cost(child1, leaf_bounds, inherited);
            let cost2 = self.insertion_cost(child2, leaf_bounds, inherited);
            if cost1 <= cost2 {
                sibling = child1;
            } else {
                sibling = child2;
            }
        }

        let old_parent = self.nodes[sibling].parent;
        let parent = self.allocate_node(Node::branch(
            self.nodes[sibling].aabb.union(leaf_bounds),
            sibling,
            leaf,
        ));
        self.nodes[parent].parent = old_parent;
        self.nodes[sibling].parent = Some(parent);
        self.nodes[leaf].parent = Some(parent);
        if let Some(grand) = old_parent {
            let grand_node = &mut self.nodes[grand];
            if grand_node.child1 == Some(sibling) {
                grand_node.child1 = Some(parent);
            } else {
                grand_node.child2 = Some(parent);
            }
        } else {
            self.root = Some(parent);
        }
        self.fix_upward(old_parent);
    }

    fn insertion_cost(&self, child: usize, leaf_bounds: Aabb, inherited: f32) -> f32 {
        let node = self.nodes[child];
        let combined = node.aabb.union(leaf_bounds);
        if combined.perimeter().is_infinite() {
            return f32::INFINITY;
        }
        if node.is_leaf() {
            combined.perimeter() + inherited
        } else {
            let direct = combined.perimeter() - node.aabb.perimeter();
            direct + inherited
        }
    }

    fn remove_leaf(&mut self, leaf: usize) {
        let Some(parent) = self.nodes[leaf].parent else {
            self.root = None;
            return;
        };
        let grand = self.nodes[parent].parent;
        let sibling = if self.nodes[parent].child1 == Some(leaf) {
            self.nodes[parent].child2.unwrap()
        } else {
            self.nodes[parent].child1.unwrap()
        };
        if let Some(grand) = grand {
            if self.nodes[grand].child1 == Some(parent) {
                self.nodes[grand].child1 = Some(sibling);
            } else {
                self.nodes[grand].child2 = Some(sibling);
            }
            self.nodes[sibling].parent = Some(grand);
            self.release_node(parent);
            self.nodes[leaf].parent = None;
            self.fix_upward(Some(grand));
        } else {
            self.root = Some(sibling);
            self.nodes[sibling].parent = None;
            self.release_node(parent);
            self.nodes[leaf].parent = None;
        }
    }

    fn fix_upward(&mut self, mut node: Option<usize>) {
        while let Some(index) = node {
            let current = self.nodes[index];
            if !current.is_leaf() {
                let child1 = current.child1.unwrap();
                let child2 = current.child2.unwrap();
                self.nodes[index].aabb = self.nodes[child1].aabb.union(self.nodes[child2].aabb);
                self.nodes[index].height =
                    1 + self.nodes[child1].height.max(self.nodes[child2].height);
            }
            node = current.parent;
        }
    }
}

/// Compute the conservative world AABB for a geom pose.
pub fn geom_aabb(
    geom: &Geom,
    pose: &GeomPose,
    meshes: &[ConvexMesh],
    hfields: &[HeightField],
) -> Aabb {
    match geom.shape {
        GeomShape::Plane => Aabb::UNBOUNDED,
        GeomShape::Sphere { radius } => {
            Aabb::from_center_extents(pose.position, Vec3::splat(radius))
        }
        GeomShape::Box { half_extents } => oriented_extents(pose, half_extents),
        GeomShape::Capsule {
            radius,
            half_height,
        } => {
            let axis = pose.rotate(Vec3::Z);
            Aabb::from_center_extents(
                pose.position,
                abs_vec(axis) * half_height + Vec3::splat(radius),
            )
        }
        GeomShape::Cylinder {
            radius,
            half_height,
        } => {
            let x = abs_vec(pose.rotate(Vec3::X));
            let y = abs_vec(pose.rotate(Vec3::Y));
            let z = abs_vec(pose.rotate(Vec3::Z));
            Aabb::from_center_extents(pose.position, (x + y) * radius + z * half_height)
        }
        GeomShape::Ellipsoid { semi_axes } => oriented_extents(pose, semi_axes),
        GeomShape::Mesh { mesh_id } => {
            let mesh = &meshes[mesh_id];
            transformed_vertices_aabb(pose, mesh.vertices.iter().copied())
        }
        GeomShape::Hfield { hfield_id } => {
            let hfield = &hfields[hfield_id];
            let (sx, sy, top, depth) = (
                hfield.size[0],
                hfield.size[1],
                hfield.size[2],
                hfield.size[3],
            );
            transformed_box_aabb(pose, Vec3::new(-sx, -sy, -depth), Vec3::new(sx, sy, top))
        }
    }
}

fn oriented_extents(pose: &GeomPose, local_extents: Vec3) -> Aabb {
    let x = abs_vec(pose.rotate(Vec3::X)) * local_extents.x;
    let y = abs_vec(pose.rotate(Vec3::Y)) * local_extents.y;
    let z = abs_vec(pose.rotate(Vec3::Z)) * local_extents.z;
    Aabb::from_center_extents(pose.position, x + y + z)
}

fn transformed_vertices_aabb<I>(pose: &GeomPose, vertices: I) -> Aabb
where
    I: Iterator<Item = Vec3>,
{
    let mut iter = vertices.map(|vertex| pose.point_to_world(vertex));
    let first = iter.next().unwrap_or(pose.position);
    let mut bounds = Aabb::new(first, first);
    for point in iter {
        bounds = bounds.union(Aabb::new(point, point));
    }
    bounds
}

fn transformed_box_aabb(pose: &GeomPose, min: Vec3, max: Vec3) -> Aabb {
    transformed_vertices_aabb(
        pose,
        (0..8).map(move |bits| {
            Vec3::new(
                if bits & 1 == 0 { min.x } else { max.x },
                if bits & 2 == 0 { min.y } else { max.y },
                if bits & 4 == 0 { min.z } else { max.z },
            )
        }),
    )
}

fn abs_vec(value: Vec3) -> Vec3 {
    Vec3::new(abs(value.x), abs(value.y), abs(value.z))
}
