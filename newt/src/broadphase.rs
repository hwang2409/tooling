//! Dynamic AABB tree broad phase.
//!
//! The tree stores a proxy's world AABB with a 10% extent margin and a small
//! absolute epsilon. Updates inside that fat bound only update the proxy's
//! position logically; they do not remove and reinsert the leaf. This keeps
//! steady-state motion cheap while retaining conservative overlap results.

use crate::geom::{ConvexMesh, Geom, GeomPose, GeomShape, HeightField};
use crate::math::{Vec3, abs};
use std::collections::HashSet;

const FATNESS: f32 = 0.1;
const FAT_EPSILON: f32 = 1.0e-4;

#[inline]
pub fn should_collide(a_group: u32, a_mask: u32, b_group: u32, b_mask: u32) -> bool {
    (a_group & b_mask) != 0 && (b_group & a_mask) != 0
}

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

    pub fn expanded(self, margin: f32) -> Self {
        let margin = Vec3::splat(margin);
        Self::new(self.min - margin, self.max + margin)
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

    pub(crate) fn contains(self, other: Self) -> bool {
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

#[derive(Clone, Copy, Debug)]
struct Proxy {
    node: usize,
    group: u32,
    mask: u32,
    tree_id: Option<usize>,
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
    proxies: Vec<Option<Proxy>>,
    root: Option<usize>,
    pair_stack: Vec<(usize, usize)>,
    pairs: Vec<(usize, usize)>,
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
            proxies: Vec::new(),
            root: None,
            pair_stack: Vec::new(),
            pairs: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.proxies.iter().filter(|node| node.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    #[doc(hidden)]
    pub fn max_depth(&self) -> usize {
        fn depth(nodes: &[Node], node: Option<usize>) -> usize {
            let Some(index) = node else { return 0 };
            let current = nodes[index];
            if current.is_leaf() {
                1
            } else {
                1 + depth(nodes, current.child1).max(depth(nodes, current.child2))
            }
        }
        depth(&self.nodes, self.root)
    }

    pub fn contains_proxy(&self, proxy: usize) -> bool {
        self.proxies.get(proxy).is_some_and(Option::is_some)
    }

    pub fn proxy_aabb(&self, proxy: usize) -> Option<Aabb> {
        self.proxies
            .get(proxy)
            .and_then(|node| *node)
            .map(|proxy| self.nodes[proxy.node].aabb)
    }

    pub fn insert(&mut self, proxy: usize, bounds: Aabb) {
        self.insert_with_filter(proxy, bounds, u32::MAX, u32::MAX);
    }

    pub fn insert_with_filter(&mut self, proxy: usize, bounds: Aabb, group: u32, mask: u32) {
        self.insert_with_filter_and_tree(proxy, bounds, group, mask, None);
    }

    pub fn insert_with_filter_and_tree(
        &mut self,
        proxy: usize,
        bounds: Aabb,
        group: u32,
        mask: u32,
        tree_id: Option<usize>,
    ) {
        assert!(!self.contains_proxy(proxy), "proxy is already in the tree");
        let leaf = self.allocate_node(Node::leaf(proxy, bounds.fatten()));
        if self.proxies.len() <= proxy {
            self.proxies.resize(proxy + 1, None);
        }
        self.proxies[proxy] = Some(Proxy {
            node: leaf,
            group,
            mask,
            tree_id,
        });
        self.insert_leaf(leaf);
    }

    pub fn remove(&mut self, proxy: usize) -> bool {
        let Some(proxy) = self.proxies.get_mut(proxy).and_then(Option::take) else {
            return false;
        };
        let leaf = proxy.node;
        self.remove_leaf(leaf);
        self.release_node(leaf);
        true
    }

    /// Update a proxy. Returns true only when the leaf was reinserted.
    pub fn update(&mut self, proxy: usize, bounds: Aabb) -> bool {
        let Some(proxy_state) = self.proxies.get(proxy).and_then(|proxy| *proxy) else {
            self.insert(proxy, bounds);
            return true;
        };
        let leaf = proxy_state.node;
        if self.nodes[leaf].aabb.contains(bounds) {
            return false;
        }
        let removed = self.remove(proxy);
        debug_assert!(removed);
        self.insert_with_filter_and_tree(
            proxy,
            bounds,
            proxy_state.group,
            proxy_state.mask,
            proxy_state.tree_id,
        );
        true
    }

    /// Update a proxy and its cached collision filter.
    pub fn update_with_filter(
        &mut self,
        proxy: usize,
        bounds: Aabb,
        group: u32,
        mask: u32,
    ) -> bool {
        let tree_id = self
            .proxies
            .get(proxy)
            .and_then(|proxy| proxy.and_then(|proxy| proxy.tree_id));
        self.update_with_filter_and_tree(proxy, bounds, group, mask, tree_id)
    }

    pub fn update_with_filter_and_tree(
        &mut self,
        proxy: usize,
        bounds: Aabb,
        group: u32,
        mask: u32,
        tree_id: Option<usize>,
    ) -> bool {
        let Some(proxy_state) = self.proxies.get(proxy).and_then(|proxy| *proxy) else {
            self.insert_with_filter_and_tree(proxy, bounds, group, mask, tree_id);
            return true;
        };
        let leaf = proxy_state.node;
        if self.nodes[leaf].aabb.contains(bounds) {
            self.proxies[proxy] = Some(Proxy {
                node: leaf,
                group,
                mask,
                tree_id,
            });
            return false;
        }
        let removed = self.remove(proxy);
        debug_assert!(removed);
        self.insert_with_filter_and_tree(proxy, bounds, group, mask, tree_id);
        true
    }

    /// Update a live proxy's cached collision filter without changing its AABB.
    pub fn set_proxy_filter(&mut self, proxy: usize, group: u32, mask: u32) -> bool {
        let Some(proxy_state) = self.proxies.get_mut(proxy).and_then(Option::as_mut) else {
            return false;
        };
        proxy_state.group = group;
        proxy_state.mask = mask;
        true
    }

    pub fn proxy_filter(&self, proxy: usize) -> Option<(u32, u32)> {
        self.proxies
            .get(proxy)
            .and_then(|proxy| *proxy)
            .map(|proxy| (proxy.group, proxy.mask))
    }

    pub fn proxy_tree_id(&self, proxy: usize) -> Option<usize> {
        self.proxies
            .get(proxy)
            .and_then(|proxy| *proxy)
            .and_then(|proxy| proxy.tree_id)
    }

    /// Return all overlapping fat-bound proxy pairs in lexicographic order.
    pub fn compute_pairs(&mut self) -> &[(usize, usize)] {
        self.compute_pairs_inner(None)
    }

    pub fn compute_pairs_with_disabled_self_collision(
        &mut self,
        disabled: &HashSet<usize>,
    ) -> &[(usize, usize)] {
        self.compute_pairs_inner(Some(disabled))
    }

    fn compute_pairs_inner(&mut self, disabled: Option<&HashSet<usize>>) -> &[(usize, usize)] {
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
                    let filter_a = self.proxies[pair.0].unwrap();
                    let filter_b = self.proxies[pair.1].unwrap();
                    if should_collide(filter_a.group, filter_a.mask, filter_b.group, filter_b.mask)
                        && !disabled.is_some_and(|disabled| {
                            match (filter_a.tree_id, filter_b.tree_id) {
                                (Some(tree_a), Some(tree_b)) => {
                                    tree_a == tree_b && disabled.contains(&tree_a)
                                }
                                _ => false,
                            }
                        })
                    {
                        self.pairs.push(pair);
                    }
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
    pub fn query_aabb<F>(&self, bounds: Aabb, mut callback: F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        self.query_aabb_node(self.root, bounds, &mut callback)
    }

    /// Visit ray hits. Return false when the callback requested early exit.
    pub fn query_ray<F>(&self, ray: Ray, mut callback: F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        self.query_ray_node(self.root, ray, &mut callback)
    }

    fn query_aabb_node<F>(&self, node: Option<usize>, bounds: Aabb, callback: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let Some(node) = node else { return true };
        let current = self.nodes[node];
        if !current.aabb.overlaps(bounds) {
            return true;
        }
        if current.is_leaf() {
            return callback(current.proxy.unwrap());
        }
        self.query_aabb_node(current.child1, bounds, callback)
            && self.query_aabb_node(current.child2, bounds, callback)
    }

    fn query_ray_node<F>(&self, node: Option<usize>, ray: Ray, callback: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let Some(node) = node else { return true };
        let current = self.nodes[node];
        if !current.aabb.ray_hits(ray) {
            return true;
        }
        if current.is_leaf() {
            return callback(current.proxy.unwrap());
        }
        self.query_ray_node(current.child1, ray, callback)
            && self.query_ray_node(current.child2, ray, callback)
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
            let balanced = self.balance(index);
            let child1 = self.nodes[balanced].child1;
            let child2 = self.nodes[balanced].child2;
            if let (Some(child1), Some(child2)) = (child1, child2) {
                self.nodes[balanced].aabb = self.nodes[child1].aabb.union(self.nodes[child2].aabb);
                self.nodes[balanced].height =
                    1 + self.nodes[child1].height.max(self.nodes[child2].height);
            }
            node = self.nodes[balanced].parent;
        }
    }

    // Rotations use child1 on equal heights. This left-first tie break makes
    // insertion order and all later tree shapes deterministic.
    fn balance(&mut self, index: usize) -> usize {
        let node = self.nodes[index];
        if node.is_leaf() || node.height < 2 {
            return index;
        }
        let left = node.child1.unwrap();
        let right = node.child2.unwrap();
        let balance = self.nodes[right].height - self.nodes[left].height;
        if balance > 1 {
            let right_left = self.nodes[right].child1.unwrap();
            let right_right = self.nodes[right].child2.unwrap();
            self.nodes[right].parent = node.parent;
            self.nodes[right].child1 = Some(index);
            self.nodes[index].parent = Some(right);
            self.replace_child(node.parent, index, right);
            if self.nodes[right_left].height >= self.nodes[right_right].height {
                self.nodes[right].child2 = Some(right_left);
                self.nodes[index].child2 = Some(right_right);
                self.nodes[right_left].parent = Some(right);
                self.nodes[right_right].parent = Some(index);
            } else {
                self.nodes[right].child2 = Some(right_right);
                self.nodes[index].child2 = Some(right_left);
                self.nodes[right_right].parent = Some(right);
                self.nodes[right_left].parent = Some(index);
            }
            self.recompute(index);
            self.recompute(right);
            return right;
        }
        if balance < -1 {
            let left_left = self.nodes[left].child1.unwrap();
            let left_right = self.nodes[left].child2.unwrap();
            self.nodes[left].parent = node.parent;
            self.nodes[left].child2 = Some(index);
            self.nodes[index].parent = Some(left);
            self.replace_child(node.parent, index, left);
            if self.nodes[left_left].height >= self.nodes[left_right].height {
                self.nodes[left].child1 = Some(left_left);
                self.nodes[index].child1 = Some(left_right);
                self.nodes[left_left].parent = Some(left);
                self.nodes[left_right].parent = Some(index);
            } else {
                self.nodes[left].child1 = Some(left_right);
                self.nodes[index].child1 = Some(left_left);
                self.nodes[left_right].parent = Some(left);
                self.nodes[left_left].parent = Some(index);
            }
            self.recompute(index);
            self.recompute(left);
            return left;
        }
        index
    }

    fn replace_child(&mut self, parent: Option<usize>, old: usize, new: usize) {
        if let Some(parent) = parent {
            if self.nodes[parent].child1 == Some(old) {
                self.nodes[parent].child1 = Some(new);
            } else {
                self.nodes[parent].child2 = Some(new);
            }
        } else {
            self.root = Some(new);
        }
    }

    fn recompute(&mut self, index: usize) {
        let child1 = self.nodes[index].child1.unwrap();
        let child2 = self.nodes[index].child2.unwrap();
        self.nodes[index].aabb = self.nodes[child1].aabb.union(self.nodes[child2].aabb);
        self.nodes[index].height = 1 + self.nodes[child1].height.max(self.nodes[child2].height);
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
