fn parse_meshes(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
) -> Result<Vec<GltfMesh>, GltfError> {
    let Some(values) = get_optional_array(object, "meshes")? else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|value| {
            let mesh_object = as_object(value, "mesh")?;
            let primitives = get_array(mesh_object, "primitives")?
                .iter()
                .map(|value| {
                    parse_primitive(as_object(value, "primitive")?, buffers, views, accessors)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(GltfMesh {
                name: get_optional_string(mesh_object, "name")?
                    .unwrap_or_default()
                    .to_string(),
                primitives,
            })
        })
        .collect()
}

fn parse_primitive(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
) -> Result<GltfPrimitive, GltfError> {
    if get_optional_u32(object, "mode")?.unwrap_or(4) != 4 {
        return Err(GltfError::new("only TRIANGLES primitives are supported"));
    }
    let attributes = get_object(object, "attributes")?;
    let position_accessor = get_u32(attributes, "POSITION")? as usize;
    let positions = read_float_accessor(
        buffers,
        views,
        accessors,
        position_accessor,
        "POSITION",
        "VEC3",
    )?;
    let normals: Option<Vec<[f32; 4]>> = get_optional_u32(attributes, "NORMAL")?
        .map(|index| {
            read_float_accessor(buffers, views, accessors, index as usize, "NORMAL", "VEC3")
        })
        .transpose()?;
    let texcoords: Option<Vec<[f32; 4]>> = get_optional_u32(attributes, "TEXCOORD_0")?
        .map(|index| {
            read_float_accessor(
                buffers,
                views,
                accessors,
                index as usize,
                "TEXCOORD_0",
                "VEC2",
            )
        })
        .transpose()?;
    let joints = get_optional_u32(attributes, "JOINTS_0")?
        .map(|index| read_joints(buffers, views, accessors, index as usize))
        .transpose()?
        .unwrap_or_else(|| vec![[0; 4]; positions.len()]);
    let weights = get_optional_u32(attributes, "WEIGHTS_0")?
        .map(|index| {
            read_float_accessor(buffers, views, accessors, index as usize, "WEIGHTS_0", "VEC4")
        })
        .transpose()?
        .map(|values| {
            values
                .into_iter()
                .map(|value| [value[0], value[1], value[2], value[3]])
                .collect()
        })
        .unwrap_or_else(|| vec![[0.0; 4]; positions.len()]);
    if normals
        .as_ref()
        .is_some_and(|values| values.len() != positions.len())
        || texcoords
            .as_ref()
            .is_some_and(|values| values.len() != positions.len())
        || joints.len() != positions.len()
        || weights.len() != positions.len()
    {
        return Err(GltfError::new(
            "mesh attribute counts do not match POSITION",
        ));
    }
    let indices = if let Some(index) = get_optional_u32(object, "indices")? {
        read_indices(buffers, views, accessors, index as usize)?
    } else {
        (0..positions.len()).collect()
    };
    if indices.len() % 3 != 0 || indices.iter().any(|&index| index >= positions.len()) {
        return Err(GltfError::new("primitive indices are invalid"));
    }
    let vertices = positions
        .iter()
        .enumerate()
        .map(|(index, position)| {
            MeshVertex::new(
                Vec3::new(position[0], position[1], position[2]),
                texcoords
                    .as_ref()
                    .map(|values| Vec2::new(values[index][0], values[index][1])),
                normals
                    .as_ref()
                    .map(|values| Vec3::new(values[index][0], values[index][1], values[index][2])),
            )
        })
        .collect();
    let triangles = indices
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    Ok(GltfPrimitive {
        mesh: Mesh::new(vertices, triangles),
        material: get_optional_u32(object, "material")?.map(|index| index as usize),
        joints,
        weights,
    })
}

fn read_accessor<const N: usize>(
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    index: usize,
    expected_components: usize,
) -> Result<Vec<[f32; N]>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new("accessor index is out of range"))?;
    let components = component_count(&accessor.kind);
    if components != expected_components {
        return Err(GltfError::new("accessor has the wrong component count"));
    }
    let bytes_per_component = component_size(accessor.component_type)?;
    let element_size = bytes_per_component
        .checked_mul(components)
        .ok_or_else(|| GltfError::new("accessor element size overflow"))?;
    let (data, stride, view_offset) = accessor_data(buffers, views, accessor, element_size)?;
    let mut output = Vec::with_capacity(accessor.count);
    for element in 0..accessor.count {
        let start = element
            .checked_mul(stride)
            .and_then(|value| value.checked_add(view_offset))
            .and_then(|value| value.checked_add(accessor.offset))
            .ok_or_else(|| GltfError::new("accessor offset overflow"))?;
        let mut result = [0.0; N];
        for component in 0..components {
            result[component] = read_component(
                &data[start + component * bytes_per_component..],
                accessor.component_type,
                accessor.normalized,
            )?;
        }
        output.push(result);
    }
    Ok(output)
}

fn read_float_accessor(
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    index: usize,
    semantic: &str,
    expected_kind: &str,
) -> Result<Vec<[f32; 4]>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new(format!("{semantic} accessor {index} is out of range")))?;
    if accessor.kind != expected_kind || accessor.component_type != 5126 || accessor.normalized {
        return Err(GltfError::new(format!(
            "{semantic} accessor {index} must be FLOAT {expected_kind}"
        )));
    }
    read_accessor::<4>(buffers, views, accessors, index, component_count(expected_kind))
}

fn read_joints(
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    index: usize,
) -> Result<Vec<[u16; 4]>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new(format!("JOINTS_0 accessor {index} is out of range")))?;
    if component_count(&accessor.kind) != 4
        || !matches!(accessor.component_type, 5121 | 5123)
        || accessor.normalized
    {
        return Err(GltfError::new(format!(
            "JOINTS_0 accessor {index} needs UNSIGNED_BYTE or UNSIGNED_SHORT VEC4"
        )));
    }
    let values = read_accessor::<4>(buffers, views, accessors, index, 4)?;
    Ok(values
        .into_iter()
        .map(|value| {
            [
                value[0] as u16,
                value[1] as u16,
                value[2] as u16,
                value[3] as u16,
            ]
        })
        .collect())
}

fn read_indices(
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    index: usize,
) -> Result<Vec<usize>, GltfError> {
    let accessor = accessors
        .get(index)
        .ok_or_else(|| GltfError::new("index accessor is out of range"))?;
    if accessor.kind != "SCALAR"
        || !matches!(accessor.component_type, 5121 | 5123 | 5125)
        || accessor.normalized
    {
        return Err(GltfError::new(format!(
            "indices accessor {index} needs unsigned scalar components"
        )));
    }
    Ok(read_accessor::<4>(buffers, views, accessors, index, 1)?
        .into_iter()
        .map(|value| value[0] as usize)
        .collect())
}

fn accessor_data<'a>(
    buffers: &'a [Vec<u8>],
    views: &[BufferView],
    accessor: &Accessor,
    element_size: usize,
) -> Result<(&'a [u8], usize, usize), GltfError> {
    let view_index = accessor
        .view
        .ok_or_else(|| GltfError::new("accessor without bufferView is unsupported"))?;
    let view = views
        .get(view_index)
        .ok_or_else(|| GltfError::new("bufferView index is out of range"))?;
    let data = buffers
        .get(view.buffer)
        .ok_or_else(|| GltfError::new("bufferView buffer index is out of range"))?;
    let end = view
        .offset
        .checked_add(view.length)
        .ok_or_else(|| GltfError::new("bufferView range overflow"))?;
    if end > data.len() {
        return Err(GltfError::new("bufferView exceeds its buffer"));
    }
    let stride = view.stride.unwrap_or(element_size);
    if stride < element_size {
        return Err(GltfError::new(
            "byteStride is smaller than the accessor element",
        ));
    }
    let last = accessor
        .count
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|last| last.checked_add(accessor.offset))
        .and_then(|last| last.checked_add(element_size))
        .ok_or_else(|| GltfError::new("accessor range overflow"))?;
    if last > view.length {
        return Err(GltfError::new(format!(
            "accessor overruns its bufferView: end {last}, length {}",
            view.length
        )));
    }
    Ok((&data[..end], stride, view.offset))
}

fn read_component(data: &[u8], component_type: u32, normalized: bool) -> Result<f32, GltfError> {
    let raw = match component_type {
        5120 => i8::from_le_bytes([data[0]]) as f32,
        5121 => data[0] as f32,
        5122 => i16::from_le_bytes([data[0], data[1]]) as f32,
        5123 => u16::from_le_bytes([data[0], data[1]]) as f32,
        5124 => i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f32,
        5125 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f32,
        5126 => f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
        _ => return Err(GltfError::new("unsupported component type")),
    };
    if !normalized {
        return Ok(raw);
    }
    Ok(match component_type {
        5120 => (raw / 127.0).max(-1.0),
        5121 => raw / 255.0,
        5122 => (raw / 32767.0).max(-1.0),
        5123 => raw / 65535.0,
        5124 => (raw / 2147483647.0).max(-1.0),
        5125 => raw / 4294967295.0,
        5126 => raw,
        _ => raw,
    })
}

fn component_size(component_type: u32) -> Result<usize, GltfError> {
    Ok(match component_type {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5124..=5126 => 4,
        _ => return Err(GltfError::new("unsupported component type")),
    })
}
fn component_count(kind: &str) -> usize {
    match kind {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT4" => 16,
        _ => 0,
    }
}

fn parse_nodes(object: &[(String, Value)]) -> Result<Vec<GltfNode>, GltfError> {
    let values = get_optional_array(object, "nodes")?.unwrap_or(&[]);
    values
        .iter()
        .map(|value| {
            let o = as_object(value, "node")?;
            let matrix = get_optional_f32_array(o, "matrix")?
                .map(|values| {
                    if values.len() != 16 {
                        return Err(GltfError::new("node matrix needs 16 values"));
                    }
                    let mut data = [0.0; 16];
                    data.copy_from_slice(&values);
                    Ok(Mat4::new(data))
                })
                .transpose()?;
            if matrix.is_some()
                && ["translation", "rotation", "scale"]
                    .iter()
                    .any(|key| get(o, key).is_some())
            {
                return Err(GltfError::new("node cannot contain both matrix and TRS"));
            }
            let translation = vec3(
                get_optional_f32_array(o, "translation")?.unwrap_or_else(|| vec![0.0, 0.0, 0.0]),
                "translation",
            )?;
            let rotation = vec4(
                get_optional_f32_array(o, "rotation")?.unwrap_or_else(|| vec![0.0, 0.0, 0.0, 1.0]),
                "rotation",
            )?;
            let scale = vec3(
                get_optional_f32_array(o, "scale")?.unwrap_or_else(|| vec![1.0, 1.0, 1.0]),
                "scale",
            )?;
            Ok(GltfNode {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                children: get_optional_usize_array(o, "children")?.unwrap_or_default(),
                mesh: get_optional_usize(o, "mesh")?,
                skin: get_optional_usize(o, "skin")?,
                matrix,
                translation,
                rotation,
                scale,
            })
        })
        .collect()
}

fn validate_nodes(nodes: &[GltfNode]) -> Result<(), GltfError> {
    let mut parents = vec![0usize; nodes.len()];
    for node in nodes {
        for &child in &node.children {
            if child >= nodes.len() {
                return Err(GltfError::new("node child index is out of range"));
            }
            parents[child] += 1;
            if parents[child] > 1 {
                return Err(GltfError::new("node hierarchy has multiple parents"));
            }
        }
    }
    let mut state = vec![0u8; nodes.len()];
    for root in 0..nodes.len() {
        if state[root] != 0 {
            continue;
        }
        let mut stack = vec![(root, false)];
        while let Some((index, exit)) = stack.pop() {
            if exit {
                state[index] = 2;
                continue;
            }
            if state[index] == 1 {
                return Err(GltfError::new("node hierarchy contains a cycle"));
            }
            if state[index] == 2 {
                continue;
            }
            state[index] = 1;
            stack.push((index, true));
            for &child in nodes[index].children.iter().rev() {
                stack.push((child, false));
            }
        }
    }
    Ok(())
}
