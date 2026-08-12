impl GltfAsset {
    /// Samples morph weights for one node primitive from mesh defaults and the
    /// same sampler used by translation, rotation, and scale channels.
    pub fn sample_morph_weights(
        &self,
        animation: Option<usize>,
        time: f32,
        node_index: usize,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<Vec<f32>, GltfError> {
        let primitive = self
            .meshes
            .get(mesh_index)
            .and_then(|mesh| mesh.primitives.get(primitive_index))
            .ok_or_else(|| GltfError::new("mesh primitive index is out of range"))?;
        let node = self
            .nodes
            .get(node_index)
            .ok_or_else(|| GltfError::new("node index is out of range"))?;
        let mut weights = node
            .weights
            .clone()
            .unwrap_or_else(|| primitive.morph_weights.clone());
        let Some(animation_index) = animation else {
            return Ok(weights);
        };
        let animation = self
            .animations
            .get(animation_index)
            .ok_or_else(|| GltfError::new("animation index is out of range"))?;
        for channel in animation
            .channels
            .iter()
            .filter(|channel| channel.node == node_index && channel.path == AnimationPath::Weights)
        {
            let sampler = animation
                .samplers
                .get(channel.sampler)
                .ok_or_else(|| GltfError::new("animation sampler index is out of range"))?;
            if sampler.output_components != weights.len() {
                return Err(GltfError::new(
                    "weights animation count does not match morph targets",
                ));
            }
            let value = sampler.sample_with_mode(time, false)?;
            weights = value[..weights.len()]
                .iter()
                .copied()
                .map(sanitize_morph_weight)
                .collect();
        }
        Ok(weights)
    }

    pub fn node_world_transforms(
        &self,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Vec<Mat4>, GltfError> {
        let locals = self.local_matrices(animation, time)?;
        let mut worlds = vec![Mat4::IDENTITY; self.nodes.len()];
        let parents = validate_nodes(&self.nodes)?;
        let mut state = vec![0u8; self.nodes.len()];
        for (root, parent) in parents.iter().enumerate() {
            if parent.is_some() {
                continue;
            }
            let mut stack = vec![(root, Mat4::IDENTITY, false)];
            while let Some((index, parent_world, exit)) = stack.pop() {
                if exit {
                    state[index] = 2;
                    continue;
                }
                if state[index] == 1 {
                    return Err(GltfError::new("node hierarchy contains a cycle"));
                }
                if state[index] == 2 {
                    return Err(GltfError::new("node hierarchy has multiple parents"));
                }
                state[index] = 1;
                worlds[index] = parent_world * locals[index];
                stack.push((index, worlds[index], true));
                for &child in self.nodes[index].children.iter().rev() {
                    stack.push((child, worlds[index], false));
                }
            }
        }
        if state.iter().any(|&value| value != 2) {
            return Err(GltfError::new("node hierarchy contains a cycle or orphan"));
        }
        Ok(worlds)
    }

    pub fn sample_animation(
        &self,
        animation: usize,
        time: f32,
    ) -> Result<Vec<NodeTransform>, GltfError> {
        self.sample_transforms(Some(animation), time)
    }

    pub fn pose_mesh(
        &self,
        mesh_index: usize,
        primitive_index: usize,
        node_index: usize,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Mesh, GltfError> {
        let morph_weights = self.sample_morph_weights(
            animation,
            time,
            node_index,
            mesh_index,
            primitive_index,
        )?;
        self.pose_mesh_with_weights(
            mesh_index,
            primitive_index,
            node_index,
            animation,
            time,
            &morph_weights,
        )
    }

    pub fn pose_mesh_with_weights(
        &self,
        mesh_index: usize,
        primitive_index: usize,
        node_index: usize,
        animation: Option<usize>,
        time: f32,
        morph_weights: &[f32],
    ) -> Result<Mesh, GltfError> {
        let primitive = self
            .meshes
            .get(mesh_index)
            .and_then(|mesh| mesh.primitives.get(primitive_index))
            .ok_or_else(|| GltfError::new("mesh primitive index is out of range"))?;
        let node = self
            .nodes
            .get(node_index)
            .ok_or_else(|| GltfError::new("node index is out of range"))?;
        let morphed = blend_morph_targets(&primitive.mesh, &primitive.morph_targets, morph_weights)?;
        let Some(skin_index) = node.skin else {
            return Ok(morphed);
        };
        let worlds = self.node_world_transforms(animation, time)?;
        let skin = self
            .skins
            .get(skin_index)
            .ok_or_else(|| GltfError::new("skin index is out of range"))?;
        if primitive.joints.len() != morphed.vertices().len()
            || primitive.weights.len() != morphed.vertices().len()
        {
            return Err(GltfError::new("skin attributes do not match mesh vertices"));
        }
        let mut joint_matrices = Vec::with_capacity(skin.joints.len());
        let mut normal_matrices = Vec::with_capacity(skin.joints.len());
        for (joint_index, &joint) in skin.joints.iter().enumerate() {
            let inverse_bind = skin
                .inverse_bind_matrices
                .get(joint_index)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let matrix = worlds
                .get(joint)
                .copied()
                .ok_or_else(|| GltfError::new("skin joint index is out of range"))?
                * inverse_bind;
            let normal = matrix
                .normal_matrix()
                .ok_or_else(|| GltfError::new("singular joint matrix cannot transform normals"))?;
            joint_matrices.push(matrix);
            normal_matrices.push(normal);
        }
        let mut vertices = Vec::with_capacity(morphed.vertices().len());
        for (index, source) in morphed.vertices().iter().enumerate() {
            let weights = normalize_weights(primitive.weights[index]);
            let mut position = Vec3::ZERO;
            let mut normal = Vec3::ZERO;
            let local_position = source.position();
            let local_normal = source.normal().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            for (influence, &weight) in weights.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                let joint = usize::from(primitive.joints[index][influence]);
                let matrix = joint_matrices
                    .get(joint)
                    .ok_or_else(|| GltfError::new("vertex joint index is out of range"))?;
                let transformed =
                    *matrix * Vec4::new(local_position.x, local_position.y, local_position.z, 1.0);
                position =
                    position + Vec3::new(transformed.x, transformed.y, transformed.z) * weight;
                normal = normal + normal_matrices[joint] * local_normal * weight;
            }
            if weights.iter().copied().sum::<f32>() == 0.0 {
                position = local_position;
                normal = local_normal;
            }
            vertices.push(MeshVertex::new(
                position,
                source.texcoord(),
                Some(normal.normalize()),
            ));
        }
        Ok(Mesh::new(vertices, morphed.indices().to_vec()))
    }

    pub fn scene_draws(
        &self,
        scene_index: usize,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Vec<GltfDraw>, GltfError> {
        let scene = self
            .scenes
            .get(scene_index)
            .ok_or_else(|| GltfError::new("scene index is out of range"))?;
        let worlds = self.node_world_transforms(animation, time)?;
        let mut draws = Vec::new();
        for &root in &scene.nodes {
            self.collect_draws(root, &worlds, animation, time, &mut draws)?;
        }
        Ok(draws)
    }

    fn collect_draws(
        &self,
        node_index: usize,
        worlds: &[Mat4],
        animation: Option<usize>,
        time: f32,
        draws: &mut Vec<GltfDraw>,
    ) -> Result<(), GltfError> {
        let mut stack = vec![node_index];
        while let Some(index) = stack.pop() {
            let node = self
                .nodes
                .get(index)
                .ok_or_else(|| GltfError::new("node index is out of range"))?;
            if let Some(mesh_index) = node.mesh {
                let mesh = self
                    .meshes
                    .get(mesh_index)
                    .ok_or_else(|| GltfError::new("node mesh index is out of range"))?;
                for primitive_index in 0..mesh.primitives.len() {
                    let morph_weights = self.sample_morph_weights(
                        animation,
                        time,
                        index,
                        mesh_index,
                        primitive_index,
                    )?;
                    let posed = self.pose_mesh_with_weights(
                        mesh_index,
                        primitive_index,
                        index,
                        animation,
                        time,
                        &morph_weights,
                    )?;
                    draws.push(GltfDraw {
                        mesh: posed,
                        model: if node.skin.is_some() {
                            Mat4::IDENTITY
                        } else {
                            worlds[index]
                        },
                        material: mesh.primitives[primitive_index].material,
                    });
                }
            }
            stack.extend(self.nodes[index].children.iter().rev().copied());
        }
        Ok(())
    }

    fn sample_transforms(
        &self,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Vec<NodeTransform>, GltfError> {
        let mut transforms: Vec<_> = self.nodes.iter().map(NodeTransform::from_node).collect();
        if let Some(animation_index) = animation {
            let animation = self
                .animations
                .get(animation_index)
                .ok_or_else(|| GltfError::new("animation index is out of range"))?;
            for channel in &animation.channels {
                let node = transforms
                    .get_mut(channel.node)
                    .ok_or_else(|| GltfError::new("animation node index is out of range"))?;
                let sampler = animation
                    .samplers
                    .get(channel.sampler)
                    .ok_or_else(|| GltfError::new("animation sampler index is out of range"))?;
                let value = sampler.sample_with_mode(
                    time,
                    channel.path == AnimationPath::Rotation,
                )?;
                match channel.path {
                    AnimationPath::Translation => {
                        node.translation = Vec3::new(value[0], value[1], value[2])
                    }
                    AnimationPath::Rotation => node.rotation = value,
                    AnimationPath::Scale => node.scale = Vec3::new(value[0], value[1], value[2]),
                    AnimationPath::Weights => {}
                }
            }
        }
        Ok(transforms)
    }

    fn local_matrices(&self, animation: Option<usize>, time: f32) -> Result<Vec<Mat4>, GltfError> {
        let transforms = self.sample_transforms(animation, time)?;
        if let Some(animation_index) = animation {
            let animation = self
                .animations
                .get(animation_index)
                .ok_or_else(|| GltfError::new("animation index is out of range"))?;
            if animation
                .channels
                .iter()
                .any(|channel| self.nodes[channel.node].matrix.is_some())
            {
                return Err(GltfError::new(
                    "animation channels targeting matrix nodes are unsupported",
                ));
            }
        }
        Ok(self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| node.matrix.unwrap_or_else(|| transforms[index].matrix()))
            .collect())
    }

}

impl MorphTarget {
    pub fn new(
        position_deltas: Vec<Vec3>,
        normal_deltas: Option<Vec<Vec3>>,
    ) -> Result<Self, GltfError> {
        if normal_deltas
            .as_ref()
            .is_some_and(|values| values.len() != position_deltas.len())
        {
            return Err(GltfError::new(
                "morph normal deltas do not match position deltas",
            ));
        }
        Ok(Self {
            position_deltas,
            normal_deltas,
        })
    }
}

/// Applies glTF morph deltas before skinning or submission.
///
/// glTF 2.0 section 3.7.3 defines the blended position as the base position
/// plus each weighted POSITION delta. NORMAL deltas are blended and then
/// normalized. The returned Mesh rebuilds its bounds, so submission culling
/// sees the deformed extent instead of stale base bounds.
pub fn blend_morph_targets(
    mesh: &Mesh,
    targets: &[MorphTarget],
    weights: &[f32],
) -> Result<Mesh, GltfError> {
    if targets.len() != weights.len() {
        return Err(GltfError::new(
            "morph weight count does not match morph targets",
        ));
    }
    if targets.is_empty() || weights.iter().all(|&weight| sanitize_morph_weight(weight) == 0.0) {
        return Ok(mesh.clone());
    }
    let vertex_count = mesh.vertices().len();
    for (index, target) in targets.iter().enumerate() {
        if target.position_deltas.len() != vertex_count {
            return Err(GltfError::new(format!(
                "morph target {index} position count does not match mesh"
            )));
        }
        if target
            .normal_deltas
            .as_ref()
            .is_some_and(|values| values.len() != vertex_count)
        {
            return Err(GltfError::new(format!(
                "morph target {index} normal count does not match mesh"
            )));
        }
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    for (vertex_index, source) in mesh.vertices().iter().enumerate() {
        let mut position = source.position();
        let mut normal = source.normal().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        for (target, &weight) in targets.iter().zip(weights) {
            let weight = sanitize_morph_weight(weight);
            position = position + target.position_deltas[vertex_index] * weight;
            if let Some(normal_deltas) = &target.normal_deltas {
                normal = normal + normal_deltas[vertex_index] * weight;
            }
        }
        vertices.push(MeshVertex::new(
            position,
            source.texcoord(),
            Some(normal.normalize()),
        ));
    }
    Ok(Mesh::new(vertices, mesh.indices().to_vec()))
}

impl GltfPrimitive {
    pub fn morph_weights(&self) -> &[f32] {
        &self.morph_weights
    }

    pub fn set_morph_weights(&mut self, weights: &[f32]) -> Result<(), GltfError> {
        if weights.len() != self.morph_targets.len() {
            return Err(GltfError::new(
                "morph weight count does not match morph targets",
            ));
        }
        self.morph_weights = weights.iter().copied().map(sanitize_morph_weight).collect();
        Ok(())
    }

    pub fn set_morph_weight(&mut self, index: usize, weight: f32) -> Result<(), GltfError> {
        let value = self
            .morph_weights
            .get_mut(index)
            .ok_or_else(|| GltfError::new("morph target index is out of range"))?;
        *value = sanitize_morph_weight(weight);
        Ok(())
    }
}
