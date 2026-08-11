impl GltfAsset {
    pub fn node_world_transforms(
        &self,
        animation: Option<usize>,
        time: f32,
    ) -> Result<Vec<Mat4>, GltfError> {
        let locals = self.local_matrices(animation, time)?;
        let mut worlds = vec![Mat4::IDENTITY; self.nodes.len()];
        validate_nodes(&self.nodes)?;
        let mut state = vec![0u8; self.nodes.len()];
        for root in 0..self.nodes.len() {
            if state[root] != 0 {
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
        let primitive = self
            .meshes
            .get(mesh_index)
            .and_then(|mesh| mesh.primitives.get(primitive_index))
            .ok_or_else(|| GltfError::new("mesh primitive index is out of range"))?;
        let node = self
            .nodes
            .get(node_index)
            .ok_or_else(|| GltfError::new("node index is out of range"))?;
        let Some(skin_index) = node.skin else {
            return Ok(primitive.mesh.clone());
        };
        let worlds = self.node_world_transforms(animation, time)?;
        let skin = self
            .skins
            .get(skin_index)
            .ok_or_else(|| GltfError::new("skin index is out of range"))?;
        if primitive.joints.len() != primitive.mesh.vertices().len()
            || primitive.weights.len() != primitive.mesh.vertices().len()
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
        let mut vertices = Vec::with_capacity(primitive.mesh.vertices().len());
        for (index, source) in primitive.mesh.vertices().iter().enumerate() {
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
        Ok(Mesh::new(vertices, primitive.mesh.indices().to_vec()))
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
                    let posed = if node.skin.is_some() {
                        self.pose_mesh(mesh_index, primitive_index, index, animation, time)?
                    } else {
                        mesh.primitives[primitive_index].mesh.clone()
                    };
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
                let value = sampler.sample(time)?;
                match channel.path {
                    AnimationPath::Translation => {
                        node.translation = Vec3::new(value[0], value[1], value[2])
                    }
                    AnimationPath::Rotation => node.rotation = value,
                    AnimationPath::Scale => node.scale = Vec3::new(value[0], value[1], value[2]),
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
