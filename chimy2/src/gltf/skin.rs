fn parse_skins(
    object: &[(String, Value)],
    buffers: &[Vec<u8>],
    views: &[BufferView],
    accessors: &[Accessor],
    node_count: usize,
) -> Result<Vec<GltfSkin>, GltfError> {
    get_optional_array(object, "skins")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "skin")?;
            let joints = get_usize_array(o, "joints")?;
            if joints.iter().any(|&joint| joint >= node_count) {
                return Err(GltfError::new("skin joint index is out of range"));
            }
            let inverse = if let Some(index) = get_optional_usize(o, "inverseBindMatrices")? {
                let accessor = accessors.get(index).ok_or_else(|| {
                    GltfError::new(format!("IBM accessor {index} is out of range"))
                })?;
                if accessor.kind != "MAT4" || accessor.component_type != 5126 || accessor.normalized
                {
                    return Err(GltfError::new(format!(
                        "IBM accessor {index} must be FLOAT MAT4"
                    )));
                }
                let values = read_accessor::<16>(buffers, views, accessors, index, 16)?;
                values.into_iter().map(Mat4::new).collect()
            } else {
                vec![Mat4::IDENTITY; joints.len()]
            };
            if inverse.len() != joints.len() {
                return Err(GltfError::new(
                    "inverse bind matrix count does not match joints",
                ));
            }
            Ok(GltfSkin {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                joints,
                inverse_bind_matrices: inverse,
                skeleton: get_optional_usize(o, "skeleton")?,
            })
        })
        .collect()
}

