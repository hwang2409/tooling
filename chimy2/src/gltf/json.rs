fn safe_asset_path(root: &Path, relative: &Path) -> Result<PathBuf, GltfError> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(GltfError::new(format!(
            "asset path is absolute or traverses outside root: {}",
            relative.display()
        )));
    }
    let root =
        fs::canonicalize(root).map_err(|error| GltfError::new(format!("asset root: {error}")))?;
    let candidate = root.join(relative);
    let canonical = fs::canonicalize(&candidate)
        .map_err(|error| GltfError::new(format!("asset {}: {error}", candidate.display())))?;
    if !canonical.starts_with(&root) {
        return Err(GltfError::new("asset path escapes root"));
    }
    Ok(canonical)
}

fn as_object<'a>(value: &'a Value, what: &str) -> Result<&'a [(String, Value)], GltfError> {
    match value {
        Value::Object(value) => Ok(value),
        _ => Err(GltfError::new(format!("{what} must be an object"))),
    }
}
fn get_object<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<&'a [(String, Value)], GltfError> {
    get(object, key)
        .map(|value| as_object(value, key))
        .transpose()?
        .ok_or_else(|| GltfError::new(format!("missing object {key}")))
}
fn get_optional_object<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<Option<&'a [(String, Value)]>, GltfError> {
    get(object, key)
        .map(|value| as_object(value, key))
        .transpose()
}
fn get_array<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a [Value], GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Array(value) => Ok(value.as_slice()),
            _ => Err(GltfError::new(format!("{key} must be an array"))),
        })
        .transpose()?
        .ok_or_else(|| GltfError::new(format!("missing array {key}")))
}
fn get_optional_array<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<Option<&'a [Value]>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Array(value) => Ok(value.as_slice()),
            _ => Err(GltfError::new(format!("{key} must be an array"))),
        })
        .transpose()
}
fn get<'a>(object: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    object
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}
fn get_string<'a>(object: &'a [(String, Value)], key: &str) -> Result<&'a str, GltfError> {
    match get(object, key) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(GltfError::new(format!("{key} must be a string"))),
        None => Err(GltfError::new(format!("missing string {key}"))),
    }
}
fn get_optional_string<'a>(
    object: &'a [(String, Value)],
    key: &str,
) -> Result<Option<&'a str>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::String(value) => Ok(value.as_str()),
            _ => Err(GltfError::new(format!("{key} must be a string"))),
        })
        .transpose()
}
fn get_f64(object: &[(String, Value)], key: &str) -> Result<f64, GltfError> {
    match get(object, key) {
        Some(Value::Number(value)) => Ok(*value),
        Some(_) => Err(GltfError::new(format!("{key} must be a number"))),
        None => Err(GltfError::new(format!("missing number {key}"))),
    }
}
fn get_optional_f32(object: &[(String, Value)], key: &str) -> Result<Option<f32>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Number(value)
                if value.is_finite()
                    && *value >= f64::from(f32::MIN)
                    && *value <= f64::from(f32::MAX) =>
            {
                Ok(*value as f32)
            }
            Value::Number(_) => Err(GltfError::new(format!("{key} is not a finite f32"))),
            _ => Err(GltfError::new(format!("{key} must be a number"))),
        })
        .transpose()
}
fn get_u32(object: &[(String, Value)], key: &str) -> Result<u32, GltfError> {
    let value = get_f64(object, key)?;
    if value < 0.0 || value.fract() != 0.0 || value > f64::from(u32::MAX) {
        return Err(GltfError::new(format!("{key} must be a u32")));
    }
    Ok(value as u32)
}
fn get_optional_u32(object: &[(String, Value)], key: &str) -> Result<Option<u32>, GltfError> {
    get(object, key).map(|_| get_u32(object, key)).transpose()
}
fn get_usize(object: &[(String, Value)], key: &str) -> Result<usize, GltfError> {
    let value = get_f64(object, key)?;
    if value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(GltfError::new(format!("{key} must be a usize")));
    }
    Ok(value as usize)
}
fn get_optional_usize(object: &[(String, Value)], key: &str) -> Result<Option<usize>, GltfError> {
    get(object, key).map(|_| get_usize(object, key)).transpose()
}
fn get_optional_bool(object: &[(String, Value)], key: &str) -> Result<Option<bool>, GltfError> {
    get(object, key)
        .map(|value| match value {
            Value::Bool(value) => Ok(*value),
            _ => Err(GltfError::new(format!("{key} must be a bool"))),
        })
        .transpose()
}
fn get_usize_array(object: &[(String, Value)], key: &str) -> Result<Vec<usize>, GltfError> {
    get_array(object, key)?
        .iter()
        .map(|value| match value {
            Value::Number(value) if *value >= 0.0 && value.fract() == 0.0 => Ok(*value as usize),
            _ => Err(GltfError::new(format!("{key} needs integer values"))),
        })
        .collect()
}
fn get_optional_usize_array(
    object: &[(String, Value)],
    key: &str,
) -> Result<Option<Vec<usize>>, GltfError> {
    get_optional_array(object, key)?
        .map(|values| {
            values
                .iter()
                .map(|value| match value {
                    Value::Number(value) if *value >= 0.0 && value.fract() == 0.0 => {
                        Ok(*value as usize)
                    }
                    _ => Err(GltfError::new(format!("{key} needs integer values"))),
                })
                .collect()
        })
        .transpose()
}
fn get_optional_f32_array(
    object: &[(String, Value)],
    key: &str,
) -> Result<Option<Vec<f32>>, GltfError> {
    get_optional_array(object, key)?
        .map(|values| {
            values
                .iter()
                .map(|value| match value {
                    Value::Number(value)
                        if value.is_finite()
                            && *value >= f64::from(f32::MIN)
                            && *value <= f64::from(f32::MAX) =>
                    {
                        Ok(*value as f32)
                    }
                    _ => Err(GltfError::new(format!("{key} needs finite number values"))),
                })
                .collect()
        })
        .transpose()
}
fn vec3(values: Vec<f32>, key: &str) -> Result<Vec3, GltfError> {
    if values.len() != 3 {
        return Err(GltfError::new(format!("{key} needs 3 values")));
    }
    Ok(Vec3::new(values[0], values[1], values[2]))
}
fn vec4(values: Vec<f32>, key: &str) -> Result<[f32; 4], GltfError> {
    if values.len() != 4 {
        return Err(GltfError::new(format!("{key} needs 4 values")));
    }
    Ok([values[0], values[1], values[2], values[3]])
}

