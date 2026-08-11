//! Wavefront MTL materials and safe texture resolution.
//!
//! Unknown MTL records are ignored. This keeps the loader additive: records
//! used by other renderers do not prevent a Chimy scene from loading. Records
//! that affect this renderer are validated and include their source line in
//! errors.

use crate::image::{ColorSpace, ImageError, Texture};
use crate::math::Vec3;
use crate::shaders::BlinnPhongUniforms;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    /// Authored MTL colors. The shader boundary converts them from sRGB.
    pub ambient: Vec3,
    pub diffuse: Vec3,
    pub specular: Vec3,
    pub shininess: f32,
    pub alpha: f32,
    map_kd: Option<PathBuf>,
    map_bump: Option<PathBuf>,
    map_kd_line: usize,
    map_bump_line: usize,
    albedo_texture: Option<Texture>,
    normal_map_texture: Option<Texture>,
}

impl Material {
    fn new(name: String) -> Self {
        Self {
            name,
            ambient: Vec3::ZERO,
            diffuse: Vec3::new(0.8, 0.8, 0.8),
            specular: Vec3::ZERO,
            shininess: 0.0,
            alpha: 1.0,
            map_kd: None,
            map_bump: None,
            map_kd_line: 0,
            map_bump_line: 0,
            albedo_texture: None,
            normal_map_texture: None,
        }
    }

    /// Applies the parsed material at the existing sanitized uniform boundary.
    pub fn apply_to(&self, uniforms: &mut BlinnPhongUniforms) {
        uniforms.set_ambient_color(self.ambient);
        uniforms.set_diffuse_color(self.diffuse);
        uniforms.set_specular_color(self.specular);
        uniforms.shininess = self.shininess.max(0.0);
        uniforms.set_alpha(self.alpha);
    }

    pub fn albedo_texture(&self) -> Option<&Texture> {
        self.albedo_texture.as_ref()
    }

    pub fn normal_map_texture(&self) -> Option<&Texture> {
        self.normal_map_texture.as_ref()
    }

    pub fn map_kd(&self) -> Option<&Path> {
        self.map_kd.as_deref()
    }

    pub fn map_bump(&self) -> Option<&Path> {
        self.map_bump.as_deref()
    }

    /// Atomically replaces map_Kd and its loaded texture cache.
    pub fn set_map_kd(
        &mut self,
        path: Option<PathBuf>,
        mtl_path: &Path,
        asset_root: &Path,
    ) -> Result<(), MtlError> {
        let texture = path
            .as_deref()
            .map(|path| load_map(mtl_path, asset_root, path, ColorSpace::Srgb, "map_Kd", 0))
            .transpose()?;
        self.map_kd = path;
        self.map_kd_line = 0;
        self.albedo_texture = texture;
        Ok(())
    }

    /// Atomically replaces map_bump and its loaded texture cache.
    pub fn set_map_bump(
        &mut self,
        path: Option<PathBuf>,
        mtl_path: &Path,
        asset_root: &Path,
    ) -> Result<(), MtlError> {
        let texture = path
            .as_deref()
            .map(|path| {
                load_map(
                    mtl_path,
                    asset_root,
                    path,
                    ColorSpace::Linear,
                    "map_bump",
                    0,
                )
            })
            .transpose()?;
        self.map_bump = path;
        self.map_bump_line = 0;
        self.normal_map_texture = texture;
        Ok(())
    }

    /// Resolves and loads this material's maps relative to its MTL file.
    pub fn load_maps(&mut self, mtl_path: &Path, asset_root: &Path) -> Result<(), MtlError> {
        let map_kd = self.map_kd.clone();
        let map_bump = self.map_bump.clone();
        let map_kd_line = self.map_kd_line;
        let map_bump_line = self.map_bump_line;
        let albedo_texture = map_kd
            .as_deref()
            .map(|path| {
                load_map(
                    mtl_path,
                    asset_root,
                    path,
                    ColorSpace::Srgb,
                    "map_Kd",
                    map_kd_line,
                )
            })
            .transpose()?;
        let normal_map_texture = map_bump
            .as_deref()
            .map(|path| {
                load_map(
                    mtl_path,
                    asset_root,
                    path,
                    ColorSpace::Linear,
                    "map_bump",
                    map_bump_line,
                )
            })
            .transpose()?;
        self.albedo_texture = albedo_texture;
        self.normal_map_texture = normal_map_texture;
        Ok(())
    }
}

pub type MtlMaterial = Material;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MaterialLibrary {
    materials: Vec<Material>,
}

impl MaterialLibrary {
    pub fn parse(source: &str) -> Result<Self, MtlError> {
        let mut library = Self::default();
        let mut current: Option<Material> = None;

        for (line_index, source_line) in source.lines().enumerate() {
            let line_number = line_index + 1;
            let line = source_line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let (record, rest) = split_record(line);
            match record {
                "newmtl" => {
                    if rest.is_empty() {
                        return Err(MtlError::new(line_number, "newmtl needs a name"));
                    }
                    if current
                        .as_ref()
                        .is_some_and(|material| material.name == rest)
                        || library
                            .materials
                            .iter()
                            .any(|material| material.name == rest)
                    {
                        return Err(MtlError::new(
                            line_number,
                            format!("duplicate material name {rest}"),
                        ));
                    }
                    if let Some(material) = current.take() {
                        library.materials.push(material);
                    }
                    current = Some(Material::new(rest.to_string()));
                }
                "Ka" => set_color(&mut current, rest, line_number, "Ka", |material, value| {
                    material.ambient = value
                })?,
                "Kd" => set_color(&mut current, rest, line_number, "Kd", |material, value| {
                    material.diffuse = value
                })?,
                "Ks" => set_color(&mut current, rest, line_number, "Ks", |material, value| {
                    material.specular = value
                })?,
                "Ns" => {
                    let material = current
                        .as_mut()
                        .ok_or_else(|| MtlError::new(line_number, "Ns appears before newmtl"))?;
                    material.shininess = parse_number(rest, line_number, "Ns")?;
                }
                "d" => {
                    let material = current
                        .as_mut()
                        .ok_or_else(|| MtlError::new(line_number, "d appears before newmtl"))?;
                    material.alpha = parse_number(rest, line_number, "d")?.clamp(0.0, 1.0);
                }
                "Tr" => {
                    let material = current
                        .as_mut()
                        .ok_or_else(|| MtlError::new(line_number, "Tr appears before newmtl"))?;
                    material.alpha = (1.0 - parse_number(rest, line_number, "Tr")?).clamp(0.0, 1.0);
                }
                "map_Kd" => set_map(&mut current, rest, line_number, "map_Kd", true)?,
                "map_bump" | "bump" => set_map(&mut current, rest, line_number, record, false)?,
                // Other MTL records are intentionally ignored. They do not
                // affect the current shader and are common in exported files.
                _ => {}
            }
        }

        if let Some(material) = current {
            library.materials.push(material);
        }
        if library.materials.is_empty() {
            return Err(MtlError::new(0, "MTL file contains no newmtl records"));
        }
        Ok(library)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, MtlError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(|error| MtlError {
            line: 0,
            message: format!("{}: {error}", path.display()),
        })?;
        let mut library = Self::parse(&source)?;
        let asset_root = path.parent().unwrap_or_else(|| Path::new("."));
        library.load_maps(path, asset_root)?;
        Ok(library)
    }

    pub fn load_with_root(
        path: impl AsRef<Path>,
        asset_root: impl AsRef<Path>,
    ) -> Result<Self, MtlError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(|error| MtlError {
            line: 0,
            message: format!("{}: {error}", path.display()),
        })?;
        let mut library = Self::parse(&source)?;
        library.load_maps(path, asset_root.as_ref())?;
        Ok(library)
    }

    pub fn load_maps(&mut self, mtl_path: &Path, asset_root: &Path) -> Result<(), MtlError> {
        for material in &mut self.materials {
            material.load_maps(mtl_path, asset_root)?;
        }
        Ok(())
    }

    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    pub fn get(&self, name: &str) -> Option<&Material> {
        self.materials.iter().find(|material| material.name == name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Material> {
        self.materials
            .iter_mut()
            .find(|material| material.name == name)
    }

    pub(crate) fn materials_mut_for_loader(&mut self) -> &mut Vec<Material> {
        &mut self.materials
    }
}

pub type MtlLibrary = MaterialLibrary;
pub type Mtl = MaterialLibrary;

fn split_record(line: &str) -> (&str, &str) {
    line.split_once(char::is_whitespace)
        .map_or((line, ""), |(record, rest)| (record, rest.trim()))
}

fn set_color(
    current: &mut Option<Material>,
    rest: &str,
    line: usize,
    record: &str,
    set: impl FnOnce(&mut Material, Vec3),
) -> Result<(), MtlError> {
    let material = current
        .as_mut()
        .ok_or_else(|| MtlError::new(line, format!("{record} appears before newmtl")))?;
    let values: Vec<_> = rest.split_whitespace().collect();
    if values.len() < 3 {
        return Err(MtlError::new(line, format!("{record} needs 3 values")));
    }
    let values = values
        .iter()
        .map(|value| parse_number(value, line, record))
        .collect::<Result<Vec<_>, _>>()?;
    set(material, Vec3::new(values[0], values[1], values[2]));
    Ok(())
}

fn set_map(
    current: &mut Option<Material>,
    rest: &str,
    line: usize,
    record: &str,
    albedo: bool,
) -> Result<(), MtlError> {
    let material = current
        .as_mut()
        .ok_or_else(|| MtlError::new(line, format!("{record} appears before newmtl")))?;
    let path = parse_map_path(rest, line, record)?;
    if path.is_empty() {
        return Err(MtlError::new(line, format!("{record} needs a path")));
    }
    if albedo {
        material.map_kd = Some(PathBuf::from(path));
        material.map_kd_line = line;
        material.albedo_texture = None;
    } else {
        material.map_bump = Some(PathBuf::from(path));
        material.map_bump_line = line;
        material.normal_map_texture = None;
    }
    Ok(())
}

fn parse_map_path(rest: &str, line: usize, record: &str) -> Result<String, MtlError> {
    let tokens: Vec<_> = rest.split_whitespace().collect();
    let mut index = 0;
    while index < tokens.len() && is_map_option(tokens[index]) {
        let option = tokens[index];
        index += 1;
        match option {
            "-blendu" | "-blendv" | "-cc" | "-clamp" => {
                let value = tokens.get(index).ok_or_else(|| {
                    MtlError::new(line, format!("truncated {record} option {option}"))
                })?;
                if !matches!(*value, "on" | "off") {
                    return Err(MtlError::new(
                        line,
                        format!("{record} option {option} needs on or off"),
                    ));
                }
                index += 1;
            }
            "-texres" => {
                let value = tokens.get(index).ok_or_else(|| {
                    MtlError::new(line, format!("truncated {record} option {option}"))
                })?;
                let resolution = value.parse::<usize>().map_err(|_| {
                    MtlError::new(line, format!("{record} option {option} needs an integer"))
                })?;
                if resolution == 0 {
                    return Err(MtlError::new(
                        line,
                        format!("{record} option {option} needs a positive integer"),
                    ));
                }
                index += 1;
            }
            "-mm" => {
                take_numeric_values(&tokens, &mut index, 2, 2, line, record, option)?;
            }
            "-o" | "-s" | "-t" => {
                take_numeric_values(&tokens, &mut index, 1, 3, line, record, option)?;
            }
            _ => {
                return Err(MtlError::new(
                    line,
                    format!("unknown {record} option {option}"),
                ));
            }
        }
    }
    Ok(tokens[index..].join(" "))
}

fn is_map_option(token: &str) -> bool {
    token.starts_with('-') && token.parse::<f32>().is_err()
}

fn take_numeric_values(
    tokens: &[&str],
    index: &mut usize,
    minimum: usize,
    maximum: usize,
    line: usize,
    record: &str,
    option: &str,
) -> Result<(), MtlError> {
    let start = *index;
    while *index < tokens.len() && *index - start < maximum && tokens[*index].parse::<f32>().is_ok()
    {
        *index += 1;
    }
    let count = *index - start;
    if count < minimum {
        return Err(MtlError::new(
            line,
            format!("{record} option {option} needs {minimum} to {maximum} numeric values"),
        ));
    }
    Ok(())
}

fn parse_number(value: &str, line: usize, record: &str) -> Result<f32, MtlError> {
    let value = value
        .parse::<f32>()
        .map_err(|_| MtlError::new(line, format!("invalid {record} value {value}")))?;
    if !value.is_finite() {
        return Err(MtlError::new(line, format!("{record} value is not finite")));
    }
    Ok(value)
}

fn load_map(
    mtl_path: &Path,
    asset_root: &Path,
    relative_path: &Path,
    color_space: ColorSpace,
    record: &str,
    line: usize,
) -> Result<Texture, MtlError> {
    let resolved = resolve_asset_path(mtl_path, asset_root, relative_path).map_err(|error| {
        MtlError::new(
            if error.line == 0 { line } else { error.line },
            error.message,
        )
    })?;
    Texture::load_with_color_space(&resolved, color_space).map_err(|error| {
        MtlError::new(
            line,
            format!("{record} {}: {error}", relative_path.display()),
        )
    })
}

/// Resolves a map path against the MTL directory and checks the asset root.
/// Absolute paths are rejected. Parent traversal is allowed only when
/// canonicalization keeps the result inside asset_root.
pub fn resolve_asset_path(
    mtl_path: &Path,
    asset_root: &Path,
    relative_path: &Path,
) -> Result<PathBuf, MtlError> {
    if relative_path.is_absolute() {
        return Err(MtlError::new(0, "absolute texture paths are not allowed"));
    }
    let mtl_dir = mtl_path.parent().unwrap_or_else(|| Path::new("."));
    let root = fs::canonicalize(asset_root).map_err(|error| {
        MtlError::new(0, format!("asset root {}: {error}", asset_root.display()))
    })?;
    let candidate = mtl_dir.join(relative_path);
    let canonical = fs::canonicalize(&candidate)
        .map_err(|error| MtlError::new(0, format!("texture {}: {error}", candidate.display())))?;
    if !canonical.starts_with(&root) {
        return Err(MtlError::new(
            0,
            format!(
                "texture path escapes asset root: {}",
                relative_path.display()
            ),
        ));
    }
    Ok(canonical)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtlError {
    pub line: usize,
    pub message: String,
}

impl MtlError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl Display for MtlError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if self.line == 0 {
            write!(formatter, "{}", self.message)
        } else {
            write!(formatter, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for MtlError {}

impl From<ImageError> for MtlError {
    fn from(error: ImageError) -> Self {
        Self::new(0, error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Mat4;
    use crate::shaders::{DirectionalLight, PointLight};

    #[test]
    fn parses_every_supported_field_and_skips_unknown_records() {
        let library = MaterialLibrary::parse(
            "newmtl painted\nKa .1 .2 .3\nKd .4 .5 .6\nKs .7 .8 .9\nNs 32\nd .75\nmap_Kd tex.qoi\nmap_bump normal.qoi\nillum 2\n",
        )
        .unwrap();
        let material = library.get("painted").unwrap();
        assert_eq!(material.ambient, Vec3::new(0.1, 0.2, 0.3));
        assert_eq!(material.diffuse, Vec3::new(0.4, 0.5, 0.6));
        assert_eq!(material.specular, Vec3::new(0.7, 0.8, 0.9));
        assert_eq!(material.shininess, 32.0);
        assert_eq!(material.alpha, 0.75);
        assert_eq!(material.map_kd(), Some(Path::new("tex.qoi")));
        assert_eq!(material.map_bump(), Some(Path::new("normal.qoi")));
    }

    #[test]
    fn parses_transparency_and_common_map_options() {
        let library =
            MaterialLibrary::parse("newmtl glass\nTr 0.25\nmap_Kd -s 1 1 1 textures/albedo.qoi\n")
                .unwrap();
        let material = library.get("glass").unwrap();
        assert_eq!(material.alpha, 0.75);
        assert_eq!(material.map_kd(), Some(Path::new("textures/albedo.qoi")));
    }

    #[test]
    fn malformed_records_return_line_context() {
        let error = MaterialLibrary::parse("newmtl x\nKd 1 nope\n").unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.to_string().contains("line 2"));
        assert!(MaterialLibrary::parse("garbage\n").is_err());
    }

    #[test]
    fn duplicate_material_names_are_rejected_at_the_second_definition() {
        let error = MaterialLibrary::parse("newmtl same\nKd 1 0 0\nnewmtl same\n").unwrap_err();
        assert_eq!(error.line, 3);
        assert!(error.to_string().contains("duplicate material name same"));
    }

    #[test]
    fn map_options_validate_arity_and_accept_negative_offsets() {
        let error = MaterialLibrary::parse("newmtl x\nmap_Kd -s checker.qoi\n").unwrap_err();
        assert_eq!(error.line, 2);
        let library = MaterialLibrary::parse(
            "newmtl x\nmap_Kd -o -1 -0.5 0 checker.qoi\nmap_bump -mm 0.1 2 normal.qoi\n",
        )
        .unwrap();
        let material = library.get("x").unwrap();
        assert_eq!(material.map_kd(), Some(Path::new("checker.qoi")));
        assert_eq!(material.map_bump(), Some(Path::new("normal.qoi")));
    }

    #[test]
    fn resolves_relative_maps_and_assigns_map_color_spaces() {
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        let mtl = assets.join("multi_material.mtl");
        let library = MaterialLibrary::load_with_root(&mtl, &assets).unwrap();
        let material = library.get("checker").unwrap();
        assert_eq!(
            material.albedo_texture().unwrap().color_space(),
            ColorSpace::Srgb
        );
        assert_eq!(
            material.normal_map_texture().unwrap().color_space(),
            ColorSpace::Linear
        );
        assert_eq!(
            resolve_asset_path(&mtl, &assets, Path::new("checker.qoi")).unwrap(),
            fs::canonicalize(assets.join("checker.qoi")).unwrap()
        );
        assert!(resolve_asset_path(&mtl, &assets, Path::new("../Cargo.toml")).is_err());
        assert!(resolve_asset_path(&mtl, &assets, Path::new("/tmp/nope.qoi")).is_err());
    }

    #[test]
    fn applying_a_file_material_keeps_uniform_sanitization_boundary() {
        let library = MaterialLibrary::parse("newmtl x\nKd -1 0.5 -0.25\n").unwrap();
        let material = library.get("x").unwrap();
        let mut uniforms = BlinnPhongUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::ZERO,
            8.0,
            Vec3::ZERO,
            DirectionalLight::default(),
            PointLight::default(),
        );
        material.apply_to(&mut uniforms);
        assert_eq!(uniforms.diffuse_color().x, 0.0);
        assert_eq!(uniforms.diffuse_color().z, 0.0);
        assert!(uniforms.diffuse_color().y > 0.0);
    }

    #[test]
    fn map_mutators_update_paths_and_caches_as_one_state() {
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        let mtl = assets.join("multi_material.mtl");
        let mut library = MaterialLibrary::load_with_root(&mtl, &assets).unwrap();
        let material = library.get_mut("checker").unwrap();
        assert!(material.albedo_texture().is_some());
        material.set_map_kd(None, &mtl, &assets).unwrap();
        assert_eq!(material.map_kd(), None);
        assert_eq!(material.albedo_texture(), None);
        material
            .set_map_kd(Some(PathBuf::from("checker.qoi")), &mtl, &assets)
            .unwrap();
        assert_eq!(material.map_kd(), Some(Path::new("checker.qoi")));
        assert_eq!(
            material.albedo_texture().unwrap().color_space(),
            ColorSpace::Srgb
        );
        let old_texture = material.albedo_texture().cloned();
        assert!(
            material
                .set_map_kd(Some(PathBuf::from("missing.qoi")), &mtl, &assets)
                .is_err()
        );
        assert_eq!(material.map_kd(), Some(Path::new("checker.qoi")));
        assert_eq!(material.albedo_texture(), old_texture.as_ref());
    }
}
