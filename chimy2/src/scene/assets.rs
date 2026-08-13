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
    Mesh::load(path).map_err(|error| SceneError(format!("{error_path}: {error}")))
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
