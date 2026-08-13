use super::*;
pub(super) fn build_instances(config: &InstancingConfig, model: Mat4) -> Vec<Instance> {
    let mut result = Vec::with_capacity(config.count);
    match &config.source {
        InstanceSource::Transforms(transforms) => {
            for transform in transforms.iter().take(config.count) {
                result.push(Instance::new(model * transform.matrix()));
            }
        }
        InstanceSource::Grid {
            dimensions,
            spacing,
        } => {
            for z in 0..dimensions[2] {
                for y in 0..dimensions[1] {
                    for x in 0..dimensions[0] {
                        if result.len() == config.count {
                            return result;
                        }
                        let offset = Vec3::new(
                            x as f32 * spacing.x,
                            y as f32 * spacing.y,
                            z as f32 * spacing.z,
                        );
                        result.push(Instance::new(model * Mat4::translate(offset)));
                    }
                }
            }
        }
    }
    result
}

// Embedded fallback for the three meshes the browser-shipped scenes reference.
// On native builds the fs read succeeds first and this table is not consulted;
// on wasm32-unknown-unknown the fs read always fails, so the fallback serves.
// Keyed by file_name only — a scene referencing an unknown filename still
// bubbles the original fs error unchanged.
const EMBEDDED_MESHES: &[(&str, &str)] = &[
    ("cube.obj", include_str!("../../assets/cube.obj")),
    (
        "icosahedron.obj",
        include_str!("../../assets/icosahedron.obj"),
    ),
    ("icosphere.obj", include_str!("../../assets/icosphere.obj")),
];

fn embedded_mesh_source(path: &Path) -> Option<&'static str> {
    let name = path.file_name()?.to_str()?;
    EMBEDDED_MESHES
        .iter()
        .find_map(|(known, source)| (*known == name).then_some(*source))
}

pub(super) fn has_embedded_fallback(path: &Path) -> bool {
    embedded_mesh_source(path).is_some()
}

pub(super) fn load_mesh(path: &Path, error_path: &str) -> Result<Mesh, SceneError> {
    if path
        .extension()
        .is_some_and(|extension| extension == "gltf")
    {
        let asset =
            GltfAsset::load(path).map_err(|error| SceneError(format!("{error_path}: {error}")))?;
        return asset
            .meshes
            .first()
            .and_then(|mesh| mesh.primitives.first())
            .map(|primitive| primitive.mesh.clone())
            .ok_or_else(|| SceneError(format!("{error_path}: glTF contains no mesh primitives")));
    }
    match Mesh::load(path) {
        Ok(mesh) => Ok(mesh),
        Err(fs_error) => match embedded_mesh_source(path) {
            Some(source) => Mesh::parse(source)
                .map_err(|error| SceneError(format!("{error_path}: embedded fallback: {error}"))),
            None => Err(SceneError(format!("{error_path}: {fs_error}"))),
        },
    }
}

pub(super) fn load_skybox(
    environment: &EnvironmentConfig,
    root: &Path,
) -> Result<Option<CubeTexture>, SceneError> {
    let Some(config) = environment.skybox.as_ref() else {
        return Ok(None);
    };
    let faces = [
        &config.px, &config.nx, &config.py, &config.ny, &config.pz, &config.nz,
    ]
    .iter()
    .map(|path| {
        crate::image::Texture::load(root.join(path))
            .map_err(|error| SceneError(format!("environment.skybox: {error}")))
    })
    .collect::<Result<Vec<_>, _>>()?;
    let faces: [crate::image::Texture; 6] = faces
        .try_into()
        .map_err(|_| SceneError("environment.skybox: expected six faces".to_string()))?;
    Ok(Some(CubeTexture::new(faces).map_err(|error| {
        SceneError(format!("environment.skybox: {error}"))
    })?))
}

pub(super) fn build_ibl_maps(
    environment: &EnvironmentConfig,
    skybox: Option<&CubeTexture>,
) -> Option<IblMaps> {
    let config = environment.ibl?;
    let cube = skybox?;
    let defaults = IblSettings::default();
    let source = FloatCube::from_cube_texture(cube, config.intensity);
    Some(IblMaps::from_float_environment(
        &source,
        IblSettings {
            irradiance_size: config.irradiance_size,
            irradiance_samples: defaults.irradiance_samples,
            prefilter_size: config.prefilter_size,
            prefilter_levels: config.prefilter_levels,
            prefilter_samples: defaults.prefilter_samples,
            brdf_size: defaults.brdf_size,
            brdf_samples: defaults.brdf_samples,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn assets_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
    }

    #[test]
    fn embedded_fallback_matches_native_parse_when_fs_read_fails() {
        // Use a bogus root that no filesystem lookup can satisfy.
        let missing_root = PathBuf::from("/tmp/chimy2-scene-fallback-does-not-exist");
        for name in ["cube.obj", "icosahedron.obj", "icosphere.obj"] {
            let missing_path = missing_root.join(name);
            let fallback = load_mesh(&missing_path, "test.mesh")
                .expect("embedded fallback should serve when fs read fails");

            // Independent parse of the on-disk file — avoids self-comparison
            // against the same embedded source the fallback returned.
            let disk_source = std::fs::read_to_string(assets_root().join(name))
                .expect("asset file must exist under CARGO_MANIFEST_DIR/assets");
            let reference = Mesh::parse(&disk_source).expect("reference mesh must parse");

            assert_eq!(
                fallback.vertices().len(),
                reference.vertices().len(),
                "{name}: vertex count mismatch"
            );
            assert_eq!(
                fallback.indices().len(),
                reference.indices().len(),
                "{name}: triangle count mismatch"
            );
        }
    }

    #[test]
    fn unknown_filename_still_surfaces_the_original_fs_error() {
        let missing_root = PathBuf::from("/tmp/chimy2-scene-fallback-does-not-exist");
        let missing_path = missing_root.join("unknown_mesh_that_is_not_in_the_table.obj");
        let error = load_mesh(&missing_path, "test.mesh")
            .expect_err("unknown filenames must not fall back to any embedded mesh");
        // The message should include the original error_path prefix.
        assert!(
            error.0.contains("test.mesh"),
            "error message must include error_path prefix, got: {}",
            error.0
        );
    }
}
