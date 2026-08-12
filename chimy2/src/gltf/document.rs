impl GltfAsset {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, GltfError> {
        let path = path.as_ref();
        let source = fs::read(path)
            .map_err(|error| GltfError::new(format!("{}: {error}", path.display())))?;
        let source = String::from_utf8(source).map_err(|error| {
            GltfError::at(
                error.utf8_error().valid_up_to(),
                "glTF source is not valid UTF-8",
            )
        })?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        Self::from_str(&source, root)
    }

    pub fn from_str(source: &str, asset_root: impl AsRef<Path>) -> Result<Self, GltfError> {
        let root = asset_root.as_ref().to_path_buf();
        let document = json::parse(source)?;
        let object = as_object(&document, "root")?;
        let asset = get_object(object, "asset")?;
        let version = get_string(asset, "version")?;
        if version != "2.0" {
            return Err(GltfError::new(format!(
                "unsupported glTF version {version}"
            )));
        }

        let buffers = load_buffers(get_array(object, "buffers")?, &root)?;
        let views = parse_views(get_array(object, "bufferViews")?)?;
        let accessors = parse_accessors(get_array(object, "accessors")?)?;
        let images = load_images(object, &root)?;
        let textures = parse_textures(object)?;
        let materials = parse_materials(object, &images, &textures)?;
        let meshes = parse_meshes(object, &buffers, &views, &accessors)?;
        let nodes = parse_nodes(object, &meshes)?;
        validate_nodes(&nodes)?;
        let skins = parse_skins(object, &buffers, &views, &accessors, nodes.len())?;
        let animations = parse_animations(
            object,
            &buffers,
            &views,
            &accessors,
            &nodes,
            &meshes,
        )?;
        let scenes = parse_scenes(object, nodes.len())?;
        let default_scene = get_optional_usize(object, "scene")?.unwrap_or(0);
        if !scenes.is_empty() && default_scene >= scenes.len() {
            return Err(GltfError::new("default scene index is out of range"));
        }
        Ok(Self {
            meshes,
            nodes,
            scenes,
            skins,
            animations,
            materials,
            default_scene,
        })
    }


}
