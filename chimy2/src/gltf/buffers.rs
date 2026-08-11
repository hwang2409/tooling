#[derive(Clone, Debug)]
struct BufferView {
    buffer: usize,
    offset: usize,
    length: usize,
    stride: Option<usize>,
}
#[derive(Clone, Debug)]
struct Accessor {
    view: Option<usize>,
    offset: usize,
    count: usize,
    component_type: u32,
    kind: String,
    normalized: bool,
}
#[derive(Clone, Copy, Debug)]
struct TextureRef {
    source: usize,
}

fn load_buffers(values: &[Value], root: &Path) -> Result<Vec<Vec<u8>>, GltfError> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let object = as_object(value, "buffer")?;
            let expected = get_usize(object, "byteLength")?;
            let data = if let Some(uri) = get_optional_string(object, "uri")? {
                if let Some(encoded) = uri.strip_prefix("data:") {
                    let (_, payload) = encoded.split_once(',').ok_or_else(|| {
                        GltfError::new(format!("buffer {index} has an invalid data URI"))
                    })?;
                    decode_base64(payload)?
                } else {
                    let path = safe_asset_path(root, Path::new(uri))?;
                    fs::read(&path).map_err(|error| {
                        GltfError::new(format!("buffer {}: {error}", path.display()))
                    })?
                }
            } else {
                return Err(GltfError::new(
                    "GLB binary buffers are unsupported; buffer needs uri",
                ));
            };
            if data.len() < expected {
                return Err(GltfError::new(format!(
                    "buffer {index} is shorter than byteLength"
                )));
            }
            Ok(data)
        })
        .collect()
}

fn decode_base64(source: &str) -> Result<Vec<u8>, GltfError> {
    let bytes = source.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(GltfError::new("base64 length is not a multiple of four"));
    }
    let mut output = Vec::with_capacity(bytes.len() / 4 * 3);
    for (chunk_index, chunk) in bytes.chunks_exact(4).enumerate() {
        let a = base64_value(chunk[0]).ok_or_else(|| GltfError::new("invalid base64 character"))?;
        let b = base64_value(chunk[1]).ok_or_else(|| GltfError::new("invalid base64 character"))?;
        let c = if chunk[2] == b'=' {
            0
        } else {
            base64_value(chunk[2]).ok_or_else(|| GltfError::new("invalid base64 character"))?
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            base64_value(chunk[3]).ok_or_else(|| GltfError::new("invalid base64 character"))?
        };
        if chunk[2] == b'=' && chunk[3] != b'=' {
            return Err(GltfError::new("invalid base64 padding"));
        }
        let final_chunk = chunk_index + 1 == bytes.len() / 4;
        if !final_chunk && (chunk[2] == b'=' || chunk[3] == b'=') {
            return Err(GltfError::new("base64 padding must be at the end"));
        }
        if chunk[2] == b'=' && (b & 0x0f) != 0 {
            return Err(GltfError::new("non-zero base64 padding bits"));
        }
        if chunk[3] == b'=' && chunk[2] != b'=' && (c & 0x03) != 0 {
            return Err(GltfError::new("non-zero base64 padding bits"));
        }
        output.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            output.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            output.push((c << 6) | d);
        }
    }
    Ok(output)
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn parse_views(values: &[Value]) -> Result<Vec<BufferView>, GltfError> {
    values
        .iter()
        .map(|value| {
            let o = as_object(value, "bufferView")?;
            Ok(BufferView {
                buffer: get_usize(o, "buffer")?,
                offset: get_optional_usize(o, "byteOffset")?.unwrap_or(0),
                length: get_usize(o, "byteLength")?,
                stride: get_optional_usize(o, "byteStride")?,
            })
        })
        .collect()
}
fn parse_accessors(values: &[Value]) -> Result<Vec<Accessor>, GltfError> {
    values
        .iter()
        .map(|value| {
            let o = as_object(value, "accessor")?;
            if o.iter().any(|(key, _)| key == "sparse") {
                return Err(GltfError::new("sparse accessors are unsupported"));
            }
            let component_type = get_u32(o, "componentType")?;
            if !(5120..=5126).contains(&component_type) {
                return Err(GltfError::new("unsupported accessor component type"));
            }
            let kind = get_string(o, "type")?.to_string();
            if !matches!(kind.as_str(), "SCALAR" | "VEC2" | "VEC3" | "VEC4" | "MAT4") {
                return Err(GltfError::new("unsupported accessor type"));
            }
            Ok(Accessor {
                view: get_optional_usize(o, "bufferView")?,
                offset: get_optional_usize(o, "byteOffset")?.unwrap_or(0),
                count: get_usize(o, "count")?,
                component_type,
                kind,
                normalized: get_optional_bool(o, "normalized")?.unwrap_or(false),
            })
        })
        .collect()
}
