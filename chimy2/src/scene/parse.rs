use super::*;
pub(super) fn parse_scene(value: &Value) -> Result<Scene, SceneError> {
    let mut object = Fields::new(
        value,
        "scene",
        &[
            "camera",
            "environment",
            "lights",
            "objects",
            "postfx",
            "particles",
            "hud",
        ],
    )?;
    let camera = parse_camera(object.required("camera")?)?;
    let environment = object
        .optional("environment")
        .map_or_else(|| Ok(default_environment()), parse_environment)?;
    let lights = parse_array(object.optional("lights"), "lights", parse_light)?;
    let objects = parse_array(object.optional("objects"), "objects", parse_object)?;
    let postfx = parse_array(object.optional("postfx"), "postfx", parse_postfx)?;
    let particles = parse_array(object.optional("particles"), "particles", parse_particle)?;
    let hud = parse_array(object.optional("hud"), "hud", parse_hud)?;
    Ok(Scene {
        camera,
        environment,
        lights,
        objects,
        postfx,
        particles,
        hud,
    })
}

fn default_environment() -> EnvironmentConfig {
    EnvironmentConfig {
        background: Vec4::new(0.015, 0.02, 0.04, 1.0),
        skybox: None,
        ibl: None,
    }
}

fn parse_camera(value: &Value) -> Result<CameraConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        "camera",
        &["position", "target", "fov", "near", "far"],
    )?;
    let position = vec3(fields.required("position")?, "camera.position")?;
    let target = vec3(fields.required("target")?, "camera.target")?;
    let fov = scalar(fields.required("fov")?, "camera.fov")?.clamp(0.01, PI - 0.01);
    let near = scalar(fields.required("near")?, "camera.near")?.max(0.0001);
    let far = scalar(fields.required("far")?, "camera.far")?.max(near + 0.0001);
    Ok(CameraConfig {
        position,
        target,
        fov,
        near,
        far,
    })
}

fn parse_environment(value: &Value) -> Result<EnvironmentConfig, SceneError> {
    let mut fields = Fields::new(value, "environment", &["background", "skybox", "ibl"])?;
    let background = vec4(
        fields.optional("background").unwrap_or(&Value::Array(vec![
            Value::Number(0.0),
            Value::Number(0.0),
            Value::Number(0.0),
            Value::Number(1.0),
        ])),
        "environment.background",
    )?;
    let skybox = fields.optional("skybox").map(parse_skybox).transpose()?;
    let ibl = fields.optional("ibl").map(parse_ibl).transpose()?;
    Ok(EnvironmentConfig {
        background,
        skybox,
        ibl,
    })
}

fn parse_skybox(value: &Value) -> Result<SkyboxConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        "environment.skybox",
        &["px", "nx", "py", "ny", "pz", "nz"],
    )?;
    Ok(SkyboxConfig {
        px: path_value(fields.required("px")?, "environment.skybox.px")?,
        nx: path_value(fields.required("nx")?, "environment.skybox.nx")?,
        py: path_value(fields.required("py")?, "environment.skybox.py")?,
        ny: path_value(fields.required("ny")?, "environment.skybox.ny")?,
        pz: path_value(fields.required("pz")?, "environment.skybox.pz")?,
        nz: path_value(fields.required("nz")?, "environment.skybox.nz")?,
    })
}

fn parse_ibl(value: &Value) -> Result<IblConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        "environment.ibl",
        &[
            "intensity",
            "irradiance_size",
            "prefilter_size",
            "prefilter_levels",
        ],
    )?;
    Ok(IblConfig {
        intensity: optional_scalar(&mut fields, "intensity", 1.0)?.max(0.0),
        irradiance_size: optional_usize(&mut fields, "irradiance_size", 16)?.max(1),
        prefilter_size: optional_usize(&mut fields, "prefilter_size", 32)?.max(1),
        prefilter_levels: optional_usize(&mut fields, "prefilter_levels", 5)?.max(1),
    })
}

fn parse_light(value: &Value, path: &str) -> Result<LightConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "direction",
            "position",
            "color",
            "constant",
            "linear",
            "quadratic",
            "shadow",
        ],
    )?;
    let kind = match string(fields.required("type")?, &format!("{path}.type"))? {
        ref value if value == "directional" => LightType::Directional,
        ref value if value == "point" => LightType::Point,
        value => {
            return Err(SceneError(format!(
                "{path}.type: unknown light type {value}"
            )));
        }
    };
    let direction = optional_vec3(
        &mut fields,
        "direction",
        Vec3::new(0.0, -1.0, 0.0),
        &format!("{path}.direction"),
    )?;
    let position = optional_vec3(
        &mut fields,
        "position",
        Vec3::ZERO,
        &format!("{path}.position"),
    )?;
    let color = optional_vec3(
        &mut fields,
        "color",
        Vec3::new(1.0, 1.0, 1.0),
        &format!("{path}.color"),
    )?;
    let shadow = fields
        .optional("shadow")
        .map(|value| parse_shadow(value, &format!("{path}.shadow")))
        .transpose()?;
    Ok(LightConfig {
        kind,
        direction,
        position,
        color,
        constant_attenuation: optional_scalar(&mut fields, "constant", 1.0)?.max(0.0),
        linear_attenuation: optional_scalar(&mut fields, "linear", 0.0)?.max(0.0),
        quadratic_attenuation: optional_scalar(&mut fields, "quadratic", 0.0)?.max(0.0),
        shadow,
    })
}

fn parse_shadow(value: &Value, path: &str) -> Result<ShadowConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "map_size",
            "cascades",
            "lambda",
            "light_size",
            "bias",
            "slope_bias",
            "near",
            "far",
        ],
    )?;
    let kind = match string(fields.required("type")?, &format!("{path}.type"))? {
        ref value if value == "basic" => ShadowType::Basic,
        ref value if value == "csm" => ShadowType::Csm,
        ref value if value == "pcss" => ShadowType::Pcss,
        ref value if value == "cube" => ShadowType::Cube,
        value => {
            return Err(SceneError(format!(
                "{path}.type: unknown shadow type {value}"
            )));
        }
    };
    let near = sanitize_near_plane(optional_scalar(&mut fields, "near", 0.1)?);
    let lambda = optional_scalar(&mut fields, "lambda", 0.5)?;
    let cascade_config =
        CascadeShadowConfig::new(optional_usize(&mut fields, "cascades", 3)?, lambda);
    Ok(ShadowConfig {
        kind,
        map_size: optional_usize(&mut fields, "map_size", 1024)?.max(1),
        cascades: cascade_config.cascade_count(),
        lambda: cascade_config.lambda(),
        light_size: sanitize_light_size(optional_scalar(&mut fields, "light_size", 0.0)?),
        bias: sanitize_bias(optional_scalar(&mut fields, "bias", 0.002)?),
        slope_bias: sanitize_bias(optional_scalar(&mut fields, "slope_bias", 0.02)?),
        near,
        far: sanitize_far_plane(optional_scalar(&mut fields, "far", 100.0)?, near),
    })
}

fn parse_object(value: &Value, path: &str) -> Result<ObjectConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &["mesh", "material", "transform", "instancing", "lod"],
    )?;
    let material = fields
        .optional("material")
        .map(|value| parse_material(value, &format!("{path}.material")))
        .transpose()?
        .unwrap_or_default();
    let transform = fields
        .optional("transform")
        .map(|value| parse_transform(value, &format!("{path}.transform")))
        .transpose()?
        .unwrap_or_default();
    let instancing = fields
        .optional("instancing")
        .map(|value| parse_instancing(value, &format!("{path}.instancing")))
        .transpose()?;
    let lod = fields
        .optional("lod")
        .map(|value| parse_lod(value, &format!("{path}.lod")))
        .transpose()?;
    if instancing.is_some() && lod.is_some() {
        return Err(SceneError(format!(
            "{path}: instancing and lod cannot be combined; use separate objects"
        )));
    }
    Ok(ObjectConfig {
        mesh: path_value(fields.required("mesh")?, &format!("{path}.mesh"))?,
        material,
        transform,
        instancing,
        lod,
    })
}

fn parse_material(value: &Value, path: &str) -> Result<MaterialConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "ambient",
            "diffuse",
            "specular",
            "shininess",
            "alpha",
            "metallic",
            "roughness",
        ],
    )?;
    let mut result = MaterialConfig::default();
    if let Some(value) = fields.optional("type") {
        result.kind = match string(value, &format!("{path}.type"))?.as_str() {
            "blinn_phong" => MaterialType::BlinnPhong,
            "ggx" => MaterialType::Ggx,
            other => {
                return Err(SceneError(format!(
                    "{path}.type: unknown material type {other}"
                )));
            }
        };
    }
    result.ambient = optional_vec3(
        &mut fields,
        "ambient",
        result.ambient,
        &format!("{path}.ambient"),
    )?;
    result.diffuse = optional_vec3(
        &mut fields,
        "diffuse",
        result.diffuse,
        &format!("{path}.diffuse"),
    )?;
    result.specular = optional_vec3(
        &mut fields,
        "specular",
        result.specular,
        &format!("{path}.specular"),
    )?;
    result.shininess = optional_scalar(&mut fields, "shininess", result.shininess)?.max(0.0);
    result.alpha = optional_scalar(&mut fields, "alpha", result.alpha)?.clamp(0.0, 1.0);
    result.metallic = optional_scalar(&mut fields, "metallic", result.metallic)?.clamp(0.0, 1.0);
    result.roughness =
        optional_scalar(&mut fields, "roughness", result.roughness)?.clamp(0.001, 1.0);
    Ok(result)
}

fn parse_transform(value: &Value, path: &str) -> Result<TransformConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["position", "rotation", "scale"])?;
    Ok(TransformConfig {
        position: optional_vec3(
            &mut fields,
            "position",
            Vec3::ZERO,
            &format!("{path}.position"),
        )?,
        rotation: optional_vec3(
            &mut fields,
            "rotation",
            Vec3::ZERO,
            &format!("{path}.rotation"),
        )?,
        scale: optional_vec3(
            &mut fields,
            "scale",
            Vec3::new(1.0, 1.0, 1.0),
            &format!("{path}.scale"),
        )?,
    })
}

fn parse_instancing(value: &Value, path: &str) -> Result<InstancingConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["count", "transforms", "grid"])?;
    let count = usize_value(fields.required("count")?, &format!("{path}.count"))?;
    let source = if let Some(value) = fields.optional("transforms") {
        InstanceSource::Transforms(parse_array(
            Some(value),
            &format!("{path}.transforms"),
            parse_transform_at,
        )?)
    } else if let Some(value) = fields.optional("grid") {
        let mut grid = Fields::new(value, &format!("{path}.grid"), &["dimensions", "spacing"])?;
        let dimensions = usize3(
            grid.required("dimensions")?,
            &format!("{path}.grid.dimensions"),
        )?;
        let spacing = vec3(grid.required("spacing")?, &format!("{path}.grid.spacing"))?;
        InstanceSource::Grid {
            dimensions,
            spacing,
        }
    } else {
        return Err(SceneError(format!("{path}: expected transforms or grid")));
    };
    Ok(InstancingConfig { count, source })
}

fn parse_lod(value: &Value, path: &str) -> Result<LodConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["ratios", "thresholds"])?;
    Ok(LodConfig {
        ratios: floats(fields.required("ratios")?, &format!("{path}.ratios"))?,
        thresholds: crate::lod::sanitize_thresholds(floats(
            fields.required("thresholds")?,
            &format!("{path}.thresholds"),
        )?),
    })
}

fn parse_postfx(value: &Value, path: &str) -> Result<PostFxConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "type",
            "radius",
            "bias",
            "strength",
            "range",
            "blur_depth_threshold",
            "focus_distance",
            "aperture",
            "max_coc_radius",
            "exposure",
        ],
    )?;
    match string(fields.required("type")?, &format!("{path}.type"))?.as_str() {
        "ssao" => Ok(PostFxConfig::Ssao {
            radius: optional_scalar(&mut fields, "radius", 0.55)?,
            bias: optional_scalar(&mut fields, "bias", 0.025)?,
            strength: optional_scalar(&mut fields, "strength", 1.0)?,
            range: optional_scalar(&mut fields, "range", 0.9)?,
            blur_depth_threshold: optional_scalar(&mut fields, "blur_depth_threshold", 0.35)?,
        }),
        "dof" => Ok(PostFxConfig::Dof {
            focus_distance: optional_scalar(&mut fields, "focus_distance", 4.0)?,
            aperture: optional_scalar(&mut fields, "aperture", 6.0)?,
            max_coc_radius: optional_scalar(&mut fields, "max_coc_radius", 8.0)?,
        }),
        "bloom" => Ok(PostFxConfig::Bloom),
        "fxaa" => Ok(PostFxConfig::Fxaa),
        "vignette" => Ok(PostFxConfig::Vignette),
        "aces" => Ok(PostFxConfig::Aces {
            exposure: optional_scalar(&mut fields, "exposure", 1.0)?,
        }),
        other => Err(SceneError(format!(
            "{path}.type: unknown postfx pass {other}"
        ))),
    }
}

fn parse_particle(value: &Value, path: &str) -> Result<ParticleConfig, SceneError> {
    let mut fields = Fields::new(
        value,
        path,
        &[
            "position",
            "emission_rate",
            "lifetime_steps",
            "initial_velocity",
            "velocity_variation",
            "gravity",
            "drag",
            "capacity",
        ],
    )?;
    Ok(ParticleConfig {
        position: optional_vec3(
            &mut fields,
            "position",
            Vec3::ZERO,
            &format!("{path}.position"),
        )?,
        emission_rate: optional_usize(&mut fields, "emission_rate", 0)?,
        lifetime_steps: optional_usize(&mut fields, "lifetime_steps", 60)?.max(1),
        initial_velocity: optional_vec3(
            &mut fields,
            "initial_velocity",
            Vec3::ZERO,
            &format!("{path}.initial_velocity"),
        )?,
        velocity_variation: optional_vec3(
            &mut fields,
            "velocity_variation",
            Vec3::ZERO,
            &format!("{path}.velocity_variation"),
        )?,
        gravity: optional_vec3(
            &mut fields,
            "gravity",
            Vec3::ZERO,
            &format!("{path}.gravity"),
        )?,
        drag: optional_scalar(&mut fields, "drag", 0.0)?.clamp(0.0, 1.0),
        capacity: optional_usize(&mut fields, "capacity", 128)?,
    })
}

fn parse_hud(value: &Value, path: &str) -> Result<HudLine, SceneError> {
    let mut fields = Fields::new(value, path, &["text", "x", "y", "scale", "color"])?;
    Ok(HudLine {
        text: string(fields.required("text")?, &format!("{path}.text"))?,
        x: integer(fields.required("x")?, &format!("{path}.x"))?,
        y: integer(fields.required("y")?, &format!("{path}.y"))?,
        scale: optional_usize(&mut fields, "scale", 1)?.max(1),
        color: vec4(fields.required("color")?, &format!("{path}.color"))?,
    })
}

fn parse_transform_at(value: &Value, path: &str) -> Result<TransformConfig, SceneError> {
    let mut fields = Fields::new(value, path, &["position", "rotation", "scale"])?;
    Ok(TransformConfig {
        position: optional_vec3(
            &mut fields,
            "position",
            Vec3::ZERO,
            &format!("{path}.position"),
        )?,
        rotation: optional_vec3(
            &mut fields,
            "rotation",
            Vec3::ZERO,
            &format!("{path}.rotation"),
        )?,
        scale: optional_vec3(
            &mut fields,
            "scale",
            Vec3::new(1.0, 1.0, 1.0),
            &format!("{path}.scale"),
        )?,
    })
}

struct Fields<'a> {
    path: String,
    values: Vec<(String, &'a Value)>,
}

impl<'a> Fields<'a> {
    fn new(value: &'a Value, path: &str, allowed: &[&str]) -> Result<Self, SceneError> {
        let Value::Object(values) = value else {
            return Err(SceneError(format!("{path}: expected object")));
        };
        for (key, _) in values {
            if !allowed.contains(&key.as_str()) {
                return Err(SceneError(format!("{path}.{key}: unknown field")));
            }
            if values.iter().filter(|(other, _)| other == key).count() > 1 {
                return Err(SceneError(format!("{path}.{key}: duplicate field")));
            }
        }
        Ok(Self {
            path: path.to_string(),
            values: values
                .iter()
                .map(|(key, value)| (key.clone(), value))
                .collect(),
        })
    }
    fn required(&mut self, key: &str) -> Result<&'a Value, SceneError> {
        self.optional(key)
            .ok_or_else(|| SceneError(format!("{}.{}: missing required field", self.path, key)))
    }
    fn optional(&mut self, key: &str) -> Option<&'a Value> {
        self.values
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| *value)
    }
}

fn parse_array<T>(
    value: Option<&Value>,
    path: &str,
    parse: impl Fn(&Value, &str) -> Result<T, SceneError>,
) -> Result<Vec<T>, SceneError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Value::Array(values) = value else {
        return Err(SceneError(format!("{path}: expected array")));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| parse(value, &format!("{path}[{index}]")))
        .collect()
}
fn scalar(value: &Value, path: &str) -> Result<f32, SceneError> {
    let Value::Number(value) = value else {
        return Err(SceneError(format!("{path}: expected number")));
    };
    Ok(*value as f32)
}
fn optional_scalar(fields: &mut Fields<'_>, key: &str, default: f32) -> Result<f32, SceneError> {
    fields.optional(key).map_or(Ok(default), |value| {
        scalar(value, &format!("{}.{}", fields.path, key))
    })
}
fn integer(value: &Value, path: &str) -> Result<i32, SceneError> {
    let value = scalar(value, path)?;
    if value.fract() != 0.0 {
        return Err(SceneError(format!("{path}: expected integer")));
    }
    Ok(value.clamp(i32::MIN as f32, i32::MAX as f32) as i32)
}
fn usize_value(value: &Value, path: &str) -> Result<usize, SceneError> {
    let value = scalar(value, path)?;
    if value.fract() != 0.0 {
        return Err(SceneError(format!("{path}: expected integer")));
    }
    Ok(if value <= 0.0 {
        0
    } else {
        (value as u64).min(usize::MAX as u64) as usize
    })
}
fn optional_usize(fields: &mut Fields<'_>, key: &str, default: usize) -> Result<usize, SceneError> {
    fields.optional(key).map_or(Ok(default), |value| {
        usize_value(value, &format!("{}.{}", fields.path, key))
    })
}
fn string(value: &Value, path: &str) -> Result<String, SceneError> {
    let Value::String(value) = value else {
        return Err(SceneError(format!("{path}: expected string")));
    };
    Ok(value.clone())
}
fn path_value(value: &Value, path: &str) -> Result<PathBuf, SceneError> {
    Ok(PathBuf::from(string(value, path)?))
}
fn vec3(value: &Value, path: &str) -> Result<Vec3, SceneError> {
    let values = array_values(value, path, 3)?;
    Ok(Vec3::new(
        scalar(&values[0], &format!("{path}[0]"))?,
        scalar(&values[1], &format!("{path}[1]"))?,
        scalar(&values[2], &format!("{path}[2]"))?,
    ))
}
fn vec4(value: &Value, path: &str) -> Result<Vec4, SceneError> {
    let values = array_values(value, path, 4)?;
    Ok(Vec4::new(
        scalar(&values[0], &format!("{path}[0]"))?,
        scalar(&values[1], &format!("{path}[1]"))?,
        scalar(&values[2], &format!("{path}[2]"))?,
        scalar(&values[3], &format!("{path}[3]"))?,
    ))
}
fn optional_vec3(
    fields: &mut Fields<'_>,
    key: &str,
    default: Vec3,
    path: &str,
) -> Result<Vec3, SceneError> {
    fields
        .optional(key)
        .map_or(Ok(default), |value| vec3(value, path))
}
fn array_values<'a>(
    value: &'a Value,
    path: &str,
    expected: usize,
) -> Result<&'a [Value], SceneError> {
    let Value::Array(values) = value else {
        return Err(SceneError(format!("{path}: expected {expected} numbers")));
    };
    if values.len() != expected {
        return Err(SceneError(format!("{path}: expected {expected} numbers")));
    }
    Ok(values)
}
fn floats(value: &Value, path: &str) -> Result<Vec<f32>, SceneError> {
    let Value::Array(values) = value else {
        return Err(SceneError(format!("{path}: expected array")));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| scalar(value, &format!("{path}[{index}]")))
        .collect()
}
fn usize3(value: &Value, path: &str) -> Result<[usize; 3], SceneError> {
    let values = array_values(value, path, 3)?;
    Ok([
        usize_value(&values[0], &format!("{path}[0]"))?,
        usize_value(&values[1], &format!("{path}[1]"))?,
        usize_value(&values[2], &format!("{path}[2]"))?,
    ])
}
