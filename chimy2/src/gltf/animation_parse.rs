fn parse_animations(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    nodes: &[GltfNode],
    meshes: &[GltfMesh],
) -> Result<Vec<GltfAnimation>, GltfError> {
    get_optional_array(object, "animations")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "animation")?;
            let mut samplers = get_array(o, "samplers")?
                .iter()
                .map(|value| {
                    let s = as_object(value, "animation sampler")?;
                    let input_accessor = get_usize(s, "input")?;
                    let input_definition = accessors.get(input_accessor).ok_or_else(|| {
                        GltfError::new(format!(
                            "animation input accessor {input_accessor} is out of range"
                        ))
                    })?;
                    if input_definition.kind != "SCALAR"
                        || input_definition.component_type != 5126
                        || input_definition.normalized
                    {
                        return Err(GltfError::new(format!(
                            "animation input accessor {input_accessor} must be FLOAT SCALAR"
                        )));
                    }
                    let input = read_accessor::<4>(
                        buffers,
                        views,
                        accessors,
                        input_accessor,
                        1,
                    )?
                    .into_iter()
                    .map(|v| v[0])
                    .collect::<Vec<_>>();
                    let output_accessor = get_usize(s, "output")?;
                    let accessor = accessors.get(output_accessor).ok_or_else(|| {
                        GltfError::new("animation output accessor is out of range")
                    })?;
                    let components = component_count(&accessor.kind);
                    if !matches!(components, 1..=4) {
                        return Err(GltfError::new(format!(
                            "animation output accessor {output_accessor} must be SCALAR, VEC2, VEC3, or VEC4"
                        )));
                    }
                    if accessor.component_type != 5126 || accessor.normalized {
                        return Err(GltfError::new(format!(
                            "animation output accessor {output_accessor} must use FLOAT"
                        )));
                    }
                    let output =
                        read_accessor::<4>(buffers, views, accessors, output_accessor, components)?;
                    let interpolation =
                        match get_optional_string(s, "interpolation")?.unwrap_or("LINEAR") {
                            "LINEAR" => Interpolation::Linear,
                            "STEP" => Interpolation::Step,
                            "CUBICSPLINE" => {
                                return Err(GltfError::new("CUBICSPLINE animation is unsupported"));
                            }
                            other => {
                                return Err(GltfError::new(format!(
                                    "unsupported animation interpolation {other}"
                                )));
                            }
                        };
                    Ok(GltfAnimationSampler {
                        input,
                        output,
                        output_components: components,
                        interpolation,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let channels = get_array(o, "channels")?
                .iter()
                .map(|value| {
                    let c = as_object(value, "animation channel")?;
                    let target = get_object(c, "target")?;
                    let node = get_usize(target, "node")?;
                    if node >= nodes.len() {
                        return Err(GltfError::new("animation target node is out of range"));
                    }
                    // glTF 2.0 section 3.6.3 permits `weights` channels for
                    // morph targets: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#animations.
                    let path = match get_string(target, "path")? {
                        "translation" => AnimationPath::Translation,
                        "rotation" => AnimationPath::Rotation,
                        "scale" => AnimationPath::Scale,
                        "weights" => AnimationPath::Weights,
                        other => {
                            return Err(GltfError::new(format!(
                                "unsupported animation target {other}"
                            )));
                        }
                    };
                    Ok(GltfAnimationChannel {
                        sampler: get_usize(c, "sampler")?,
                        node,
                        path,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            for channel in &channels {
                let valid = match channel.path {
                    AnimationPath::Translation | AnimationPath::Scale => {
                        let sampler = samplers.get(channel.sampler).ok_or_else(|| {
                            GltfError::new("animation channel sampler is out of range")
                        })?;
                        sampler.output_components == 3 && sampler.input.len() == sampler.output.len()
                    }
                    AnimationPath::Rotation => {
                        let sampler = samplers.get(channel.sampler).ok_or_else(|| {
                            GltfError::new("animation channel sampler is out of range")
                        })?;
                        sampler.output_components == 4 && sampler.input.len() == sampler.output.len()
                    }
                    AnimationPath::Weights => {
                        let target_count = nodes
                            .get(channel.node)
                            .and_then(|node| node.mesh)
                            .and_then(|mesh| meshes.get(mesh))
                            .map(|mesh| mesh.weights.len())
                            .unwrap_or(0);
                        let Some(sampler) = samplers.get_mut(channel.sampler) else {
                            return Err(GltfError::new(
                                "animation channel sampler is out of range",
                            ));
                        };
                        if sampler.output_components != 1
                            || target_count == 0
                            || target_count > 4
                            || sampler.output.len() != sampler.input.len() * target_count
                        {
                            false
                        } else {
                            let flattened = sampler
                                .output
                                .iter()
                                .map(|value| value[0])
                                .collect::<Vec<_>>();
                            sampler.output = (0..sampler.input.len())
                                .map(|frame| {
                                    let mut value = [0.0; 4];
                                    let start = frame * target_count;
                                    value[..target_count].copy_from_slice(
                                        &flattened[start..start + target_count],
                                    );
                                    value
                                })
                                .collect();
                            sampler.output_components = target_count;
                            true
                        }
                    }
                };
                if !valid {
                    return Err(GltfError::new(
                        "animation output type does not match its target path",
                    ));
                }
            }
            let duration = samplers
                .iter()
                .flat_map(|sampler| sampler.input.iter().copied())
                .fold(0.0, f32::max);
            Ok(GltfAnimation {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                samplers,
                channels,
                duration,
            })
        })
        .collect()
}

fn parse_scenes(
    object: &[(String, Value)],
    node_count: usize,
) -> Result<Vec<GltfScene>, GltfError> {
    get_optional_array(object, "scenes")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "scene")?;
            let nodes = get_usize_array(o, "nodes")?;
            if nodes.iter().any(|&node| node >= node_count) {
                return Err(GltfError::new("scene node index is out of range"));
            }
            Ok(GltfScene {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                nodes,
            })
        })
        .collect()
}
