fn load_images(object: &[(String, Value)], root: &Path) -> Result<Vec<Option<Texture>>, GltfError> {
    get_optional_array(object, "images")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "image")?;
            if get_optional_usize(o, "bufferView")?.is_some() {
                return Err(GltfError::new("bufferView images are unsupported"));
            }
            let Some(uri) = get_optional_string(o, "uri")? else {
                return Err(GltfError::new("image needs uri"));
            };
            let (mime, bytes) = if let Some(encoded) = uri.strip_prefix("data:") {
                let (header, payload) = encoded
                    .split_once(',')
                    .ok_or_else(|| GltfError::new("invalid image data URI"))?;
                (
                    header.split(';').next().unwrap_or(""),
                    decode_base64(payload)?,
                )
            } else {
                let path = safe_asset_path(root, Path::new(uri))?;
                let mime = match path.extension().and_then(|extension| extension.to_str()) {
                    Some("qoi") => "image/qoi",
                    Some("ppm") => "image/x-portable-pixmap",
                    _ => return Err(GltfError::new("unsupported glTF image format")),
                };
                (
                    mime,
                    fs::read(path).map_err(|error| GltfError::new(error.to_string()))?,
                )
            };
            let texture = match mime {
                "image/qoi" | "image/x-qoi" => {
                    Texture::from_qoi_with_color_space(&bytes, ColorSpace::Srgb)
                }
                "image/x-portable-pixmap" | "image/ppm" => {
                    Texture::from_ppm_with_color_space(&bytes, ColorSpace::Srgb)
                }
                _ => {
                    return Err(GltfError::new(format!(
                        "unsupported glTF image MIME type {mime}"
                    )));
                }
            }
            .map_err(|error| GltfError::new(error.to_string()))?;
            Ok(Some(texture))
        })
        .collect()
}

fn parse_textures(object: &[(String, Value)]) -> Result<Vec<TextureRef>, GltfError> {
    get_optional_array(object, "textures")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "texture")?;
            Ok(TextureRef {
                source: get_usize(o, "source")?,
            })
        })
        .collect()
}

fn parse_materials(
    object: &[(String, Value)],
    images: &[Option<Texture>],
    textures: &[TextureRef],
) -> Result<Vec<GltfMaterial>, GltfError> {
    get_optional_array(object, "materials")?
        .unwrap_or(&[])
        .iter()
        .map(|value| {
            let o = as_object(value, "material")?;
            let pbr = get_optional_object(o, "pbrMetallicRoughness")?;
            let factor = vec4(
                pbr.and_then(|p| get_optional_f32_array(p, "baseColorFactor").transpose())
                    .transpose()?
                    .unwrap_or_else(|| vec![1.0, 1.0, 1.0, 1.0]),
                "baseColorFactor",
            )?;
            let base_color_factor = Vec4::new(factor[0], factor[1], factor[2], factor[3]);
            let metallic_factor = pbr
                .map(|p| get_optional_f32(p, "metallicFactor"))
                .transpose()?
                .flatten()
                .unwrap_or(1.0);
            let roughness_factor = pbr
                .map(|p| get_optional_f32(p, "roughnessFactor"))
                .transpose()?
                .flatten()
                .unwrap_or(1.0);
            let alpha_mode = match get_optional_string(o, "alphaMode")?.unwrap_or("OPAQUE") {
                "OPAQUE" => GltfAlphaMode::Opaque,
                "BLEND" => GltfAlphaMode::Blend,
                "MASK" => GltfAlphaMode::Mask,
                other => return Err(GltfError::new(format!("unsupported alphaMode {other}"))),
            };
            let alpha_cutoff = get_optional_f32(o, "alphaCutoff")?.unwrap_or(0.5);
            if alpha_mode == GltfAlphaMode::Mask {
                return Err(GltfError::new(
                    "alphaMode MASK unsupported: fragment discard is not implemented",
                ));
            }
            let albedo_texture = if let Some(pbr) = pbr {
                get_optional_object(pbr, "baseColorTexture")?
                    .map(|texture| texture_from_ref(texture, textures, images, ColorSpace::Srgb))
                    .transpose()?
            } else {
                None
            };
            let normal_map_texture = get_optional_object(o, "normalTexture")?
                .map(|texture| texture_from_ref(texture, textures, images, ColorSpace::Linear))
                .transpose()?;
            Ok(GltfMaterial {
                name: get_optional_string(o, "name")?
                    .unwrap_or_default()
                    .to_string(),
                base_color_factor,
                metallic_factor,
                roughness_factor,
                albedo_texture,
                normal_map_texture,
                alpha_mode,
                alpha_cutoff,
            })
        })
        .collect()
}

fn texture_from_ref(
    object: &[(String, Value)],
    textures: &[TextureRef],
    images: &[Option<Texture>],
    color_space: ColorSpace,
) -> Result<Texture, GltfError> {
    let index = get_usize(object, "index")?;
    let texture = textures
        .get(index)
        .ok_or_else(|| GltfError::new("texture index is out of range"))?;
    let image = images
        .get(texture.source)
        .ok_or_else(|| GltfError::new("image index is out of range"))?
        .clone()
        .ok_or_else(|| GltfError::new("image is missing"))?;
    if color_space == ColorSpace::Linear {
        let pixels = image.pixels().to_vec();
        return Texture::new_with_color_space(
            image.width(),
            image.height(),
            pixels,
            ColorSpace::Linear,
        )
        .map_err(|error| GltfError::new(error.to_string()));
    }
    Ok(image)
}

