//! Newt's native JSON scene format: parser, validator, and loader into
//! [`World`].
//!
//! # Scope
//!
//! One entry point ([`load_str`] / [`load_from_path`]) turns a JSON document
//! into a [`Scene`] — the fully-built [`World`], indexed by name for tests
//! and demos, plus a [`Site`] table for the biped heel/toe pattern. The
//! JSON format is the v0 native model description; MJCF compatibility is
//! on the v1 roadmap.
//!
//! See `docs/model-format.md` for the schema reference.
//!
//! # Zero deps + strict-by-default
//!
//! The loader depends only on [`crate::json`] and the engine's own types.
//! Every unknown field is REJECTED at load time — a silently ignored
//! `"dampign"` typo is a physics bug in disguise (per the chimy2 lesson).
//! Every validation failure carries the JSON path to the offending value so
//! errors read as `bodies[2].inertia: expected 6 or 9 numbers, got 4`.

use std::collections::HashMap;
use std::path::Path;

use crate::actuator::PdServo;
use crate::body::Body;
use crate::equality::Equality;
use crate::geom::{Geom, GeomShape, SolRef};
use crate::joint::{JointKind, JointLimit};
use crate::json::{self, Value};
use crate::math::{Mat3, Quat, Vec3};
use crate::sensor::{Sensor, SensorAttach, SensorKind, SiteFrame};
use crate::tree::{Link, Tree, forward_kinematics};
use crate::world::World;

// ---------------------------------------------------------------------------
// public error type
// ---------------------------------------------------------------------------

/// A structured model error. `path` uses JSON-path notation like
/// `trees[0].links[1].joint.axis` so the reader can find the value without
/// re-parsing the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelError {
    pub path: String,
    pub message: String,
}

impl ModelError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

impl std::error::Error for ModelError {}

impl From<json::Error> for ModelError {
    fn from(err: json::Error) -> Self {
        Self::new(
            "<json>",
            format!("parse error at byte {}: {}", err.offset, err.message),
        )
    }
}

// ---------------------------------------------------------------------------
// scene = loaded world + name tables + sites
// ---------------------------------------------------------------------------

/// A named point on a body or tree link. World-frame pose is
/// `parent_pose ∘ (local_offset, local_orientation)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Site {
    pub name: String,
    pub attach: SiteAttach,
    pub local_offset: Vec3,
    pub local_orientation: Quat,
}

/// Where a site is anchored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteAttach {
    /// Attached to `Scene::world.bodies[i]`.
    Body(usize),
    /// Attached to `Scene::world.trees[t].links[l]`.
    Link { tree: usize, link: usize },
}

/// A fully-loaded scene: the runnable [`World`] plus name → index tables
/// so tests and demos can address things by the JSON names.
///
/// The tables are populated at load time and never mutate. Duplicates in
/// the same scope are rejected at load, so lookups always resolve to a
/// unique index.
#[derive(Clone, Debug)]
pub struct Scene {
    /// The simulation world assembled from the model.
    pub world: World,

    /// Free-body name → `world.bodies` index.
    pub bodies_by_name: HashMap<String, usize>,
    /// Tree name → `world.trees` index.
    pub trees_by_name: HashMap<String, usize>,
    /// Per-tree link name tables — `links_by_name[tree_idx][link_name]` →
    /// link index inside that tree.
    pub links_by_name: Vec<HashMap<String, usize>>,
    /// Geom name → `world.geoms` index.
    pub geoms_by_name: HashMap<String, usize>,

    /// Sites in insertion order.
    pub sites: Vec<Site>,
    /// Site name → `sites` index.
    pub sites_by_name: HashMap<String, usize>,

    /// Actuator name → `(tree_idx, actuator_idx_within_tree)`.
    pub actuators_by_name: HashMap<String, (usize, usize)>,

    /// Sensor name → index into `world.sensors.sensors`. Same lookup that
    /// [`World::sensor`] takes; the field is here purely so callers can
    /// address sensors by the JSON name.
    pub sensors_by_name: HashMap<String, usize>,
}

impl Scene {
    /// World-frame pose of a named site: `(position, orientation)`. `None`
    /// when the name is unknown.
    pub fn site_pose(&self, name: &str) -> Option<(Vec3, Quat)> {
        self.sites_by_name
            .get(name)
            .map(|&idx| self.site_pose_by_index(idx))
    }

    /// World-frame pose of a site by index.
    pub fn site_pose_by_index(&self, idx: usize) -> (Vec3, Quat) {
        let site = &self.sites[idx];
        let (parent_pos, parent_ori) = match site.attach {
            SiteAttach::Body(i) => {
                let b = &self.world.bodies[i];
                (b.position, b.orientation)
            }
            SiteAttach::Link { tree, link } => {
                let poses = forward_kinematics(&self.world.trees[tree]);
                poses[link]
            }
        };
        let world_pos = parent_pos + parent_ori.rotate(site.local_offset);
        let world_ori = parent_ori * site.local_orientation;
        (world_pos, world_ori)
    }
}

// ---------------------------------------------------------------------------
// public entry points
// ---------------------------------------------------------------------------

/// Load a scene from a JSON source string. Every error is a [`ModelError`]
/// with a JSON-path context.
pub fn load_str(source: &str) -> Result<Scene, ModelError> {
    let value = json::parse(source)?;
    build_scene(&value)
}

/// Load a scene from a JSON file on disk.
pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Scene, ModelError> {
    let src = std::fs::read_to_string(path.as_ref()).map_err(|e| {
        ModelError::new(
            "<io>",
            format!("could not read {}: {}", path.as_ref().display(), e),
        )
    })?;
    load_str(&src)
}

// ---------------------------------------------------------------------------
// value walkers with JSON-path context
// ---------------------------------------------------------------------------

fn fail<T>(path: &str, message: impl Into<String>) -> Result<T, ModelError> {
    Err(ModelError::new(path, message))
}

fn get_object<'a>(v: &'a Value, path: &str) -> Result<&'a [(String, Value)], ModelError> {
    match v {
        Value::Object(fields) => Ok(fields),
        other => fail(path, format!("expected object, got {}", other.type_name())),
    }
}

fn get_array<'a>(v: &'a Value, path: &str) -> Result<&'a [Value], ModelError> {
    match v {
        Value::Array(items) => Ok(items),
        other => fail(path, format!("expected array, got {}", other.type_name())),
    }
}

fn get_f32(v: &Value, path: &str) -> Result<f32, ModelError> {
    match v {
        Value::Number(n) => {
            let n = *n as f32;
            if !n.is_finite() {
                fail(path, "number is not finite in f32")
            } else {
                Ok(n)
            }
        }
        other => fail(path, format!("expected number, got {}", other.type_name())),
    }
}

fn get_bool(v: &Value, path: &str) -> Result<bool, ModelError> {
    match v {
        Value::Bool(b) => Ok(*b),
        other => fail(path, format!("expected bool, got {}", other.type_name())),
    }
}

fn get_str<'a>(v: &'a Value, path: &str) -> Result<&'a str, ModelError> {
    match v {
        Value::String(s) => Ok(s.as_str()),
        other => fail(path, format!("expected string, got {}", other.type_name())),
    }
}

fn required<'a>(
    fields: &'a [(String, Value)],
    name: &str,
    path: &str,
) -> Result<&'a Value, ModelError> {
    fields
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v)
        .ok_or_else(|| ModelError::new(path, format!("missing required field \"{name}\"")))
}

fn optional<'a>(fields: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn reject_unknown(
    fields: &[(String, Value)],
    allowed: &[&str],
    path: &str,
) -> Result<(), ModelError> {
    for (k, _) in fields {
        if !allowed.contains(&k.as_str()) {
            return fail(
                path,
                format!("unknown field \"{k}\"; allowed: [{}]", allowed.join(", ")),
            );
        }
    }
    // Also reject duplicate keys — the JSON parser preserves order, so a
    // duplicate would be a caller-visible ambiguity.
    for (i, (k, _)) in fields.iter().enumerate() {
        if fields.iter().skip(i + 1).any(|(kk, _)| kk == k) {
            return fail(path, format!("duplicate field \"{k}\""));
        }
    }
    Ok(())
}

fn parse_vec3(v: &Value, path: &str) -> Result<Vec3, ModelError> {
    let arr = get_array(v, path)?;
    if arr.len() != 3 {
        return fail(
            path,
            format!("expected [x, y, z] (3 numbers), got {}", arr.len()),
        );
    }
    Ok(Vec3::new(
        get_f32(&arr[0], &format!("{path}[0]"))?,
        get_f32(&arr[1], &format!("{path}[1]"))?,
        get_f32(&arr[2], &format!("{path}[2]"))?,
    ))
}

fn parse_quat(v: &Value, path: &str) -> Result<Quat, ModelError> {
    let arr = get_array(v, path)?;
    if arr.len() != 4 {
        return fail(
            path,
            format!("expected [x, y, z, w] (4 numbers), got {}", arr.len()),
        );
    }
    let q = Quat::new(
        get_f32(&arr[0], &format!("{path}[0]"))?,
        get_f32(&arr[1], &format!("{path}[1]"))?,
        get_f32(&arr[2], &format!("{path}[2]"))?,
        get_f32(&arr[3], &format!("{path}[3]"))?,
    );
    // Reject a zero quaternion (would divide by zero on renormalize). Non-
    // unit but nonzero is fine — the caller may have written slightly-off
    // components and we renormalize.
    if q.norm_squared() < 1e-12 {
        return fail(path, "quaternion has zero norm");
    }
    Ok(q.renormalize())
}

// ---------------------------------------------------------------------------
// inertia parser: diag | tensor | solid
// ---------------------------------------------------------------------------

fn parse_inertia(v: &Value, mass: f32, path: &str) -> Result<Mat3, ModelError> {
    let fields = get_object(v, path)?;
    let kind = required(fields, "kind", path)?;
    let kind_s = get_str(kind, &format!("{path}.kind"))?;
    match kind_s {
        "diag" => {
            reject_unknown(fields, &["kind", "values"], path)?;
            let values = required(fields, "values", path)?;
            let arr = get_array(values, &format!("{path}.values"))?;
            if arr.len() != 3 {
                return fail(
                    &format!("{path}.values"),
                    format!("expected 3 numbers for diagonal inertia, got {}", arr.len()),
                );
            }
            let ixx = get_f32(&arr[0], &format!("{path}.values[0]"))?;
            let iyy = get_f32(&arr[1], &format!("{path}.values[1]"))?;
            let izz = get_f32(&arr[2], &format!("{path}.values[2]"))?;
            require_positive_inertia_axis(ixx, "Ixx", path)?;
            require_positive_inertia_axis(iyy, "Iyy", path)?;
            require_positive_inertia_axis(izz, "Izz", path)?;
            Ok(Mat3::diag(ixx, iyy, izz))
        }
        "tensor" => {
            reject_unknown(fields, &["kind", "values"], path)?;
            let values = required(fields, "values", path)?;
            let arr = get_array(values, &format!("{path}.values"))?;
            let (ixx, iyy, izz, ixy, ixz, iyz) = match arr.len() {
                6 => (
                    get_f32(&arr[0], &format!("{path}.values[0]"))?,
                    get_f32(&arr[1], &format!("{path}.values[1]"))?,
                    get_f32(&arr[2], &format!("{path}.values[2]"))?,
                    get_f32(&arr[3], &format!("{path}.values[3]"))?,
                    get_f32(&arr[4], &format!("{path}.values[4]"))?,
                    get_f32(&arr[5], &format!("{path}.values[5]"))?,
                ),
                _ => {
                    return fail(
                        &format!("{path}.values"),
                        format!(
                            "expected 6 numbers [Ixx, Iyy, Izz, Ixy, Ixz, Iyz], got {}",
                            arr.len()
                        ),
                    );
                }
            };
            require_positive_inertia_axis(ixx, "Ixx", path)?;
            require_positive_inertia_axis(iyy, "Iyy", path)?;
            require_positive_inertia_axis(izz, "Izz", path)?;
            let m = Mat3::new([ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz]);
            if m.inverse().is_none() {
                return fail(path, "inertia tensor is singular");
            }
            Ok(m)
        }
        "solid" => {
            reject_unknown(fields, &["kind", "shape"], path)?;
            let shape = required(fields, "shape", path)?;
            parse_solid_inertia(shape, mass, &format!("{path}.shape"))
        }
        other => fail(
            &format!("{path}.kind"),
            format!("unknown inertia kind \"{other}\"; expected diag | tensor | solid"),
        ),
    }
}

fn require_positive_inertia_axis(v: f32, name: &str, path: &str) -> Result<(), ModelError> {
    if v <= 0.0 {
        return fail(path, format!("{name} must be > 0 (got {v})"));
    }
    Ok(())
}

fn parse_solid_inertia(v: &Value, mass: f32, path: &str) -> Result<Mat3, ModelError> {
    let fields = get_object(v, path)?;
    let kind = required(fields, "kind", path)?;
    let kind_s = get_str(kind, &format!("{path}.kind"))?;
    match kind_s {
        "box" => {
            reject_unknown(fields, &["kind", "half_extents"], path)?;
            let he = parse_vec3(
                required(fields, "half_extents", path)?,
                &format!("{path}.half_extents"),
            )?;
            if he.x <= 0.0 || he.y <= 0.0 || he.z <= 0.0 {
                return fail(path, "half_extents must be positive on every axis");
            }
            Ok(crate::geom::solid_box_inertia(mass, he))
        }
        "sphere" => {
            reject_unknown(fields, &["kind", "radius"], path)?;
            let r = get_f32(required(fields, "radius", path)?, &format!("{path}.radius"))?;
            if r <= 0.0 {
                return fail(path, "sphere radius must be positive");
            }
            Ok(crate::geom::solid_sphere_inertia(mass, r))
        }
        "capsule" => {
            reject_unknown(fields, &["kind", "radius", "half_height"], path)?;
            let r = get_f32(required(fields, "radius", path)?, &format!("{path}.radius"))?;
            let h = get_f32(
                required(fields, "half_height", path)?,
                &format!("{path}.half_height"),
            )?;
            if r <= 0.0 || h < 0.0 {
                return fail(
                    path,
                    "capsule radius must be > 0 and half_height must be ≥ 0",
                );
            }
            Ok(crate::geom::solid_capsule_inertia(mass, r, h))
        }
        "cylinder" => {
            reject_unknown(fields, &["kind", "radius", "half_height"], path)?;
            let r = get_f32(required(fields, "radius", path)?, &format!("{path}.radius"))?;
            let h = get_f32(
                required(fields, "half_height", path)?,
                &format!("{path}.half_height"),
            )?;
            if r <= 0.0 || h < 0.0 {
                return fail(
                    path,
                    "cylinder radius must be > 0 and half_height must be ≥ 0",
                );
            }
            Ok(crate::geom::solid_cylinder_inertia(mass, r, h))
        }
        "ellipsoid" => {
            reject_unknown(fields, &["kind", "semi_axes"], path)?;
            let sa = parse_vec3(
                required(fields, "semi_axes", path)?,
                &format!("{path}.semi_axes"),
            )?;
            if sa.x <= 0.0 || sa.y <= 0.0 || sa.z <= 0.0 {
                return fail(path, "ellipsoid semi_axes must be > 0 on every axis");
            }
            Ok(crate::geom::solid_ellipsoid_inertia(mass, sa))
        }
        other => fail(
            &format!("{path}.kind"),
            format!(
                "unknown solid shape \"{other}\"; expected \
                 box | sphere | capsule | cylinder | ellipsoid"
            ),
        ),
    }
}

// ---------------------------------------------------------------------------
// pose parsers
// ---------------------------------------------------------------------------

fn parse_pose(v: &Value, path: &str) -> Result<(Vec3, Quat), ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(
        fields,
        &["position", "orientation", "orientation_axis_angle"],
        path,
    )?;
    let pos = match optional(fields, "position") {
        Some(pos) => parse_vec3(pos, &format!("{path}.position"))?,
        None => Vec3::ZERO,
    };
    let quat_form = optional(fields, "orientation");
    let axis_angle_form = optional(fields, "orientation_axis_angle");
    if quat_form.is_some() && axis_angle_form.is_some() {
        return fail(
            path,
            "provide either `orientation` (quaternion) or `orientation_axis_angle` \
             (axis + angle), not both",
        );
    }
    let ori = if let Some(ori) = quat_form {
        parse_quat(ori, &format!("{path}.orientation"))?
    } else if let Some(aa) = axis_angle_form {
        parse_orientation_axis_angle(aa, &format!("{path}.orientation_axis_angle"))?
    } else {
        Quat::IDENTITY
    };
    Ok((pos, ori))
}

/// Parse the `orientation_axis_angle` form. Uses `Quat::from_axis_angle`
/// so a scene author who wants byte-identity with a programmatic
/// `Quat::from_axis_angle(axis, angle)` construction gets it — the loader
/// runs the SAME `sin`/`cos` polynomials as the engine. Preferred over
/// raw quaternion literals for orientations produced by an axis-angle
/// call.
fn parse_orientation_axis_angle(v: &Value, path: &str) -> Result<Quat, ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(fields, &["axis", "angle"], path)?;
    let axis = parse_vec3(required(fields, "axis", path)?, &format!("{path}.axis"))?;
    if axis.length_squared() == 0.0 {
        return fail(&format!("{path}.axis"), "axis must be non-zero");
    }
    let angle = get_f32(required(fields, "angle", path)?, &format!("{path}.angle"))?;
    Ok(Quat::from_axis_angle(axis, angle))
}

// ---------------------------------------------------------------------------
// joint parser
// ---------------------------------------------------------------------------

fn parse_joint(v: &Value, path: &str) -> Result<JointKind, ModelError> {
    let fields = get_object(v, path)?;
    let kind = required(fields, "kind", path)?;
    let kind_s = get_str(kind, &format!("{path}.kind"))?;
    match kind_s {
        "free" => {
            reject_unknown(fields, &["kind"], path)?;
            Ok(JointKind::Free)
        }
        "fixed" => {
            reject_unknown(fields, &["kind"], path)?;
            Ok(JointKind::Fixed)
        }
        "hinge" => {
            let SingleDofAxisJoint {
                axis,
                range,
                damping,
                armature,
                limit,
            } = parse_single_dof_axis_joint(fields, path, "hinge")?;
            Ok(JointKind::Hinge {
                axis,
                range,
                damping,
                armature,
                limit,
            })
        }
        "slide" => {
            let SingleDofAxisJoint {
                axis,
                range,
                damping,
                armature,
                limit,
            } = parse_single_dof_axis_joint(fields, path, "slide")?;
            Ok(JointKind::Slide {
                axis,
                range,
                damping,
                armature,
                limit,
            })
        }
        "ball" => {
            // Check `range` FIRST — the reject-unknown pass below would
            // otherwise fire a generic "unknown field" error. Ball limits
            // (cone / swing-twist) need the v1 constraint solver landing
            // in a follow-up ticket; give a pointer at that deferral so a
            // user who tries to add limits gets a clear message instead of
            // a generic typo error.
            if optional(fields, "range").is_some() {
                return fail(
                    &format!("{path}.range"),
                    "ball joints do not support range limits in v0/v1-tier-1; \
                     a 3-DOF cone / swing-twist limit needs the constraint \
                     solver in a follow-up ticket",
                );
            }
            reject_unknown(fields, &["kind", "damping", "armature"], path)?;
            let damping = optional(fields, "damping")
                .map(|v| get_f32(v, &format!("{path}.damping")))
                .transpose()?
                .unwrap_or(0.0);
            if damping < 0.0 {
                return fail(&format!("{path}.damping"), "damping must be ≥ 0");
            }
            let armature = optional(fields, "armature")
                .map(|v| get_f32(v, &format!("{path}.armature")))
                .transpose()?
                .unwrap_or(0.0);
            if armature < 0.0 {
                return fail(&format!("{path}.armature"), "armature must be ≥ 0");
            }
            Ok(JointKind::Ball { damping, armature })
        }
        other => fail(
            &format!("{path}.kind"),
            format!("unknown joint kind \"{other}\"; expected free | fixed | hinge | slide | ball"),
        ),
    }
}

/// Parsed fields shared by every 1-DOF axis joint (hinge, slide).
struct SingleDofAxisJoint {
    axis: Vec3,
    range: Option<(f32, f32)>,
    damping: f32,
    armature: f32,
    limit: JointLimit,
}

/// Shared parser for single-DOF axis joints (hinge / slide). `kind_label` is
/// only used in error messages (`hinge axis must be non-zero` vs `slide axis
/// must be non-zero`).
fn parse_single_dof_axis_joint(
    fields: &[(String, Value)],
    path: &str,
    kind_label: &str,
) -> Result<SingleDofAxisJoint, ModelError> {
    reject_unknown(
        fields,
        &["kind", "axis", "range", "damping", "armature", "limit"],
        path,
    )?;
    let axis = parse_vec3(required(fields, "axis", path)?, &format!("{path}.axis"))?;
    if axis.length_squared() < 1e-12 {
        return fail(
            &format!("{path}.axis"),
            format!("{kind_label} axis must be non-zero"),
        );
    }
    let axis = axis.normalize();
    let range = match optional(fields, "range") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let arr = get_array(v, &format!("{path}.range"))?;
            if arr.len() != 2 {
                return fail(
                    &format!("{path}.range"),
                    format!("expected [lo, hi] (2 numbers), got {}", arr.len()),
                );
            }
            let lo = get_f32(&arr[0], &format!("{path}.range[0]"))?;
            let hi = get_f32(&arr[1], &format!("{path}.range[1]"))?;
            if lo >= hi {
                return fail(
                    &format!("{path}.range"),
                    format!("range low ({lo}) must be < high ({hi})"),
                );
            }
            Some((lo, hi))
        }
    };
    let damping = optional(fields, "damping")
        .map(|v| get_f32(v, &format!("{path}.damping")))
        .transpose()?
        .unwrap_or(0.0);
    if damping < 0.0 {
        return fail(&format!("{path}.damping"), "damping must be ≥ 0");
    }
    let armature = optional(fields, "armature")
        .map(|v| get_f32(v, &format!("{path}.armature")))
        .transpose()?
        .unwrap_or(0.0);
    if armature < 0.0 {
        return fail(&format!("{path}.armature"), "armature must be ≥ 0");
    }
    let limit = match optional(fields, "limit") {
        None => JointLimit::DEFAULT,
        Some(v) => {
            let lfields = get_object(v, &format!("{path}.limit"))?;
            reject_unknown(
                lfields,
                &["stiffness", "damping", "solref", "solimp"],
                &format!("{path}.limit"),
            )?;
            let stiffness = get_f32(
                required(lfields, "stiffness", &format!("{path}.limit"))?,
                &format!("{path}.limit.stiffness"),
            )?;
            let damping = get_f32(
                required(lfields, "damping", &format!("{path}.limit"))?,
                &format!("{path}.limit.damping"),
            )?;
            if stiffness < 0.0 || damping < 0.0 {
                return fail(
                    &format!("{path}.limit"),
                    "limit stiffness and damping must be ≥ 0",
                );
            }
            let mut jl = JointLimit::new(stiffness, damping);
            if let Some(sv) = optional(lfields, "solref") {
                let sf = get_object(sv, &format!("{path}.limit.solref"))?;
                reject_unknown(
                    sf,
                    &["timeconst", "dampratio"],
                    &format!("{path}.limit.solref"),
                )?;
                let tc = get_f32(
                    required(sf, "timeconst", &format!("{path}.limit.solref"))?,
                    &format!("{path}.limit.solref.timeconst"),
                )?;
                let zeta = get_f32(
                    required(sf, "dampratio", &format!("{path}.limit.solref"))?,
                    &format!("{path}.limit.solref.dampratio"),
                )?;
                if tc <= 0.0 || zeta < 0.0 {
                    return fail(
                        &format!("{path}.limit.solref"),
                        "timeconst must be > 0 and dampratio must be ≥ 0",
                    );
                }
                jl.solref = Some(SolRef::new(tc, zeta));
            }
            if let Some(iv) = optional(lfields, "solimp") {
                let sf = get_object(iv, &format!("{path}.limit.solimp"))?;
                reject_unknown(
                    sf,
                    &["dmin", "dmax", "width", "midpoint", "power"],
                    &format!("{path}.limit.solimp"),
                )?;
                let dmin = get_f32(
                    required(sf, "dmin", &format!("{path}.limit.solimp"))?,
                    &format!("{path}.limit.solimp.dmin"),
                )?;
                let dmax = get_f32(
                    required(sf, "dmax", &format!("{path}.limit.solimp"))?,
                    &format!("{path}.limit.solimp.dmax"),
                )?;
                let width = get_f32(
                    required(sf, "width", &format!("{path}.limit.solimp"))?,
                    &format!("{path}.limit.solimp.width"),
                )?;
                let midpoint = get_f32(
                    required(sf, "midpoint", &format!("{path}.limit.solimp"))?,
                    &format!("{path}.limit.solimp.midpoint"),
                )?;
                let power_f = get_f32(
                    required(sf, "power", &format!("{path}.limit.solimp"))?,
                    &format!("{path}.limit.solimp.power"),
                )?;
                if power_f < 1.0 || power_f != power_f.floor() {
                    return fail(
                        &format!("{path}.limit.solimp.power"),
                        format!("power must be a positive integer, got {power_f}"),
                    );
                }
                let s = crate::solver::SolImp::new(dmin, dmax, width, midpoint, power_f as u32);
                s.validate()
                    .map_err(|m| ModelError::new(format!("{path}.limit.solimp"), m))?;
                jl.solimp = Some(s);
            }
            jl
        }
    };
    Ok(SingleDofAxisJoint {
        axis,
        range,
        damping,
        armature,
        limit,
    })
}

// ---------------------------------------------------------------------------
// scene builder
// ---------------------------------------------------------------------------

fn build_scene(root: &Value) -> Result<Scene, ModelError> {
    let path = "";
    let root_fields = get_object(root, path)?;
    reject_unknown(
        root_fields,
        &[
            "version",
            "gravity",
            "timestep",
            "solver",
            "bodies",
            "trees",
            "geoms",
            "meshes",
            "sites",
            "actuators",
            "contact_pairs",
            "equality",
            "sensors",
        ],
        path,
    )?;

    // Version — informational only in v0. Presence with a non-"1" value is
    // rejected so an incompatible file cannot silently load.
    if let Some(v) = optional(root_fields, "version") {
        let s = get_str(v, "version")?;
        if s != "1" {
            return fail(
                "version",
                format!("unsupported version \"{s}\"; expected \"1\""),
            );
        }
    }

    let mut world = World::new();
    if let Some(v) = optional(root_fields, "gravity") {
        world.gravity = parse_vec3(v, "gravity")?;
    }
    if let Some(v) = optional(root_fields, "timestep") {
        let dt = get_f32(v, "timestep")?;
        if dt <= 0.0 {
            return fail("timestep", format!("timestep must be > 0 (got {dt})"));
        }
        world.dt = dt;
    }
    if let Some(v) = optional(root_fields, "solver") {
        world.solver = parse_solver_config(v, "solver")?;
    }

    // ---- Free bodies ----
    let mut bodies_by_name: HashMap<String, usize> = HashMap::new();
    if let Some(v) = optional(root_fields, "bodies") {
        let arr = get_array(v, "bodies")?;
        for (i, body_v) in arr.iter().enumerate() {
            let p = format!("bodies[{i}]");
            let body = parse_body(body_v, &p)?;
            let name = required_name(body_v, &p, "bodies")?;
            if bodies_by_name.contains_key(&name) {
                return fail(
                    &format!("{p}.name"),
                    format!("duplicate body name \"{name}\""),
                );
            }
            let idx = world.add_body(body);
            bodies_by_name.insert(name, idx);
        }
    }

    // ---- Trees ----
    let mut trees_by_name: HashMap<String, usize> = HashMap::new();
    let mut links_by_name: Vec<HashMap<String, usize>> = Vec::new();
    // Track (tree_idx, is_self_collide, link_geom_owned: bool) — used for
    // self-collision filtering when generating contact pairs.
    let mut tree_self_collide: Vec<bool> = Vec::new();
    if let Some(v) = optional(root_fields, "trees") {
        let arr = get_array(v, "trees")?;
        for (i, tree_v) in arr.iter().enumerate() {
            let p = format!("trees[{i}]");
            let (tree, name, link_names, self_collide) = parse_tree(tree_v, &p)?;
            if trees_by_name.contains_key(&name) {
                return fail(
                    &format!("{p}.name"),
                    format!("duplicate tree name \"{name}\""),
                );
            }
            let idx = world.add_tree(tree);
            trees_by_name.insert(name, idx);
            links_by_name.push(link_names);
            tree_self_collide.push(self_collide);
        }
    }

    // ---- Meshes (asset table, referenced by geoms via `{"kind":"mesh","mesh":<name>}`)
    let mut meshes_by_name: HashMap<String, usize> = HashMap::new();
    if let Some(v) = optional(root_fields, "meshes") {
        let arr = get_array(v, "meshes")?;
        for (i, mesh_v) in arr.iter().enumerate() {
            let p = format!("meshes[{i}]");
            let (mesh, name) = parse_mesh_asset(mesh_v, &p)?;
            if meshes_by_name.contains_key(&name) {
                return fail(
                    &format!("{p}.name"),
                    format!("duplicate mesh name \"{name}\""),
                );
            }
            let idx = world.add_mesh(mesh);
            meshes_by_name.insert(name, idx);
        }
    }

    // ---- Geoms ----
    let mut geoms_by_name: HashMap<String, usize> = HashMap::new();
    if let Some(v) = optional(root_fields, "geoms") {
        let arr = get_array(v, "geoms")?;
        for (i, geom_v) in arr.iter().enumerate() {
            let p = format!("geoms[{i}]");
            let (geom, name) = parse_geom(
                geom_v,
                &p,
                &bodies_by_name,
                &trees_by_name,
                &links_by_name,
                &meshes_by_name,
            )?;
            if geoms_by_name.contains_key(&name) {
                return fail(
                    &format!("{p}.name"),
                    format!("duplicate geom name \"{name}\""),
                );
            }
            let idx = world.add_geom(geom);
            geoms_by_name.insert(name, idx);
        }
    }

    // ---- Sites ----
    let mut sites: Vec<Site> = Vec::new();
    let mut sites_by_name: HashMap<String, usize> = HashMap::new();
    if let Some(v) = optional(root_fields, "sites") {
        let arr = get_array(v, "sites")?;
        for (i, site_v) in arr.iter().enumerate() {
            let p = format!("sites[{i}]");
            let site = parse_site(site_v, &p, &bodies_by_name, &trees_by_name, &links_by_name)?;
            if sites_by_name.contains_key(&site.name) {
                return fail(
                    &format!("{p}.name"),
                    format!("duplicate site name \"{}\"", site.name),
                );
            }
            let idx = sites.len();
            sites_by_name.insert(site.name.clone(), idx);
            sites.push(site);
        }
    }

    // ---- Actuators ----
    let mut actuators_by_name: HashMap<String, (usize, usize)> = HashMap::new();
    if let Some(v) = optional(root_fields, "actuators") {
        let arr = get_array(v, "actuators")?;
        for (i, act_v) in arr.iter().enumerate() {
            let p = format!("actuators[{i}]");
            let (name, tree_idx, servo) =
                parse_actuator(act_v, &p, &trees_by_name, &links_by_name, &world)?;
            if actuators_by_name.contains_key(&name) {
                return fail(
                    &format!("{p}.name"),
                    format!("duplicate actuator name \"{name}\""),
                );
            }
            let act_idx = world.trees[tree_idx].add_actuator(servo);
            actuators_by_name.insert(name, (tree_idx, act_idx));
        }
    }

    // ---- Contact pair filtering ----
    if let Some(v) = optional(root_fields, "contact_pairs") {
        world.pair_list = Some(parse_contact_pairs(
            v,
            "contact_pairs",
            &world,
            &geoms_by_name,
            &tree_self_collide,
        )?);
    } else if tree_self_collide.iter().any(|&s| !s) {
        // Auto pair list minus geom pairs on the same non-self-colliding
        // tree. Even if every tree is self_collide=true, we can leave
        // `pair_list = None` (world uses auto pairs) — the auto list already
        // drops same-link and same-body pairs.
        world.pair_list = Some(auto_pairs_with_self_collision_filter(
            &world,
            &tree_self_collide,
        ));
    }

    // ---- Equalities ----
    if let Some(v) = optional(root_fields, "equality") {
        let arr = get_array(v, "equality")?;
        for (i, eq_v) in arr.iter().enumerate() {
            let p = format!("equality[{i}]");
            let eq = parse_equality(
                eq_v,
                &p,
                &bodies_by_name,
                &trees_by_name,
                &links_by_name,
                &world,
            )?;
            eq.validate().map_err(|m| ModelError::new(p, m))?;
            world.equalities.push(eq);
        }
    }

    // ---- Sensors ----
    // Parsed AFTER geoms/sites so name resolution has the full universe of
    // referenceable entities. World.add_sensor validates each spec, so a
    // bad reference here surfaces at the corresponding JSON path.
    let mut sensors_by_name: HashMap<String, usize> = HashMap::new();
    if let Some(v) = optional(root_fields, "sensors") {
        let arr = get_array(v, "sensors")?;
        for (i, sv) in arr.iter().enumerate() {
            let p = format!("sensors[{i}]");
            let sensor = parse_sensor(
                sv,
                &p,
                &bodies_by_name,
                &trees_by_name,
                &links_by_name,
                &geoms_by_name,
                &sites,
                &sites_by_name,
            )?;
            if sensors_by_name.contains_key(&sensor.name) {
                return fail(
                    &format!("{p}.name"),
                    format!("duplicate sensor name \"{}\"", sensor.name),
                );
            }
            let name = sensor.name.clone();
            let idx = world
                .add_sensor(sensor)
                .map_err(|e| ModelError::new(p, e.0))?;
            sensors_by_name.insert(name, idx);
        }
    }

    // ---- Loader-level pair support check ----
    // Reject any ACTIVE contact pair whose shape combination is not
    // implemented by newt's narrow phase. Loud at load time so users get a
    // JSON-path error instead of the runtime panic (which is the same
    // enforcement one level down; see `World::step`).
    if let Some(unsupported) = world.validate_supported_pairs().into_iter().next() {
        let ga = &world.geoms[unsupported.geom_a];
        let gb = &world.geoms[unsupported.geom_b];
        return fail(
            "contact_pairs",
            format!(
                "contact pair between geom {} ({:?}) and geom {} ({:?}) is not supported \
                 by newt's narrow phase — see docs/contacts.md support matrix. Restrict \
                 the pair list or defer this configuration.",
                unsupported.geom_a, ga.shape, unsupported.geom_b, gb.shape,
            ),
        );
    }

    Ok(Scene {
        world,
        bodies_by_name,
        trees_by_name,
        links_by_name,
        geoms_by_name,
        sites,
        sites_by_name,
        actuators_by_name,
        sensors_by_name,
    })
}

fn required_name(v: &Value, path: &str, scope: &str) -> Result<String, ModelError> {
    let fields = get_object(v, path)?;
    let name = required(fields, "name", path)?;
    let s = get_str(name, &format!("{path}.name"))?;
    if s.is_empty() {
        return fail(&format!("{path}.name"), format!("empty name in {scope}"));
    }
    Ok(s.to_string())
}

// ---------------------------------------------------------------------------
// body / tree / link parsers
// ---------------------------------------------------------------------------

fn parse_body(v: &Value, path: &str) -> Result<Body, ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(
        fields,
        &["name", "mass", "inertia", "pose", "velocity"],
        path,
    )?;
    let mass = get_f32(required(fields, "mass", path)?, &format!("{path}.mass"))?;
    if mass <= 0.0 {
        return fail(
            &format!("{path}.mass"),
            format!("mass must be > 0 (got {mass})"),
        );
    }
    let inertia = parse_inertia(
        required(fields, "inertia", path)?,
        mass,
        &format!("{path}.inertia"),
    )?;
    let (pos, ori) = match optional(fields, "pose") {
        Some(v) => parse_pose(v, &format!("{path}.pose"))?,
        None => (Vec3::ZERO, Quat::IDENTITY),
    };
    let mut body = Body::new(mass, inertia, pos, ori);
    if let Some(v) = optional(fields, "velocity") {
        let vf = get_object(v, &format!("{path}.velocity"))?;
        reject_unknown(vf, &["linear", "angular_body"], &format!("{path}.velocity"))?;
        if let Some(l) = optional(vf, "linear") {
            body.linear_velocity = parse_vec3(l, &format!("{path}.velocity.linear"))?;
        }
        if let Some(a) = optional(vf, "angular_body") {
            body.angular_velocity_body = parse_vec3(a, &format!("{path}.velocity.angular_body"))?;
        }
    }
    Ok(body)
}

fn parse_tree(
    v: &Value,
    path: &str,
) -> Result<(Tree, String, HashMap<String, usize>, bool), ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(fields, &["name", "self_collide", "links"], path)?;
    let name = get_str(required(fields, "name", path)?, &format!("{path}.name"))?.to_string();
    if name.is_empty() {
        return fail(&format!("{path}.name"), "tree name must not be empty");
    }
    let self_collide = match optional(fields, "self_collide") {
        None => false,
        Some(v) => get_bool(v, &format!("{path}.self_collide"))?,
    };

    let links_v = required(fields, "links", path)?;
    let links = get_array(links_v, &format!("{path}.links"))?;
    if links.is_empty() {
        return fail(
            &format!("{path}.links"),
            "tree must have at least one link (a root)",
        );
    }
    let mut tree = Tree::new();
    let mut link_names: HashMap<String, usize> = HashMap::new();

    for (i, link_v) in links.iter().enumerate() {
        let p = format!("{path}.links[{i}]");
        let (link, name) = parse_link(link_v, &p, i, &link_names)?;
        if link_names.contains_key(&name) {
            return fail(
                &format!("{p}.name"),
                format!("duplicate link name \"{name}\""),
            );
        }
        link_names.insert(name, i);
        tree.push_link(link);
    }
    Ok((tree, name, link_names, self_collide))
}

fn parse_link(
    v: &Value,
    path: &str,
    index: usize,
    prior_names: &HashMap<String, usize>,
) -> Result<(Link, String), ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(
        fields,
        &[
            "name",
            "parent",
            "joint",
            "joint_offset_in_parent",
            "joint_offset_in_child",
            "mass",
            "inertia",
        ],
        path,
    )?;
    let name = get_str(required(fields, "name", path)?, &format!("{path}.name"))?.to_string();
    if name.is_empty() {
        return fail(&format!("{path}.name"), "link name must not be empty");
    }

    let parent_v = optional(fields, "parent");
    let parent = if index == 0 {
        // Root: parent must be omitted or explicitly null.
        match parent_v {
            None | Some(Value::Null) => None,
            Some(_) => {
                return fail(
                    &format!("{path}.parent"),
                    "root link (index 0) must have parent = null (or omitted)",
                );
            }
        }
    } else {
        let pv = parent_v
            .ok_or_else(|| ModelError::new(path, "non-root link must specify \"parent\""))?;
        let parent_name = get_str(pv, &format!("{path}.parent"))?;
        match prior_names.get(parent_name) {
            Some(&idx) => Some(idx),
            None => {
                return fail(
                    &format!("{path}.parent"),
                    format!(
                        "unknown parent link \"{parent_name}\" (must reference a link defined \
                         earlier in the same tree)"
                    ),
                );
            }
        }
    };

    let joint = parse_joint(required(fields, "joint", path)?, &format!("{path}.joint"))?;
    // Consistency: only Free/Fixed are allowed at the root; the non-root
    // joints (Hinge, Slide, Ball) all need a parent to reference.
    let joint_kind_name = match &joint {
        JointKind::Free => "free",
        JointKind::Fixed => "fixed",
        JointKind::Hinge { .. } => "hinge",
        JointKind::Slide { .. } => "slide",
        JointKind::Ball { .. } => "ball",
    };
    match (index, &joint) {
        (0, JointKind::Hinge { .. } | JointKind::Slide { .. } | JointKind::Ball { .. }) => {
            return fail(
                &format!("{path}.joint"),
                format!("root joint must be \"free\" or \"fixed\", not \"{joint_kind_name}\""),
            );
        }
        (_, JointKind::Free) if index > 0 => {
            return fail(
                &format!("{path}.joint"),
                "\"free\" joint is only valid on the root link (index 0)",
            );
        }
        _ => {}
    }

    let joint_offset_in_parent = match optional(fields, "joint_offset_in_parent") {
        Some(v) => parse_pose(v, &format!("{path}.joint_offset_in_parent"))?,
        None => (Vec3::ZERO, Quat::IDENTITY),
    };
    let joint_offset_in_child = match optional(fields, "joint_offset_in_child") {
        Some(v) => parse_pose(v, &format!("{path}.joint_offset_in_child"))?,
        None => (Vec3::ZERO, Quat::IDENTITY),
    };
    // v0 constraint: joint frame orientation must be identity (see
    // Tree::push_link comment). The engine debug_assert fires only in debug;
    // enforce here so release builds still reject an ill-specified model.
    // Exemption for a root: joint_offset_in_parent doubles as the world
    // anchor pose there.
    let is_root = index == 0;
    if !is_root && !approx_identity(joint_offset_in_parent.1) {
        return fail(
            &format!("{path}.joint_offset_in_parent.orientation"),
            "v0 joint offsets must have IDENTITY orientation (only the root's parent-offset is \
             a world anchor pose and may deviate)",
        );
    }
    if !approx_identity(joint_offset_in_child.1) {
        return fail(
            &format!("{path}.joint_offset_in_child.orientation"),
            "v0 joint offsets must have IDENTITY orientation",
        );
    }

    let mass = get_f32(required(fields, "mass", path)?, &format!("{path}.mass"))?;
    if mass <= 0.0 {
        return fail(
            &format!("{path}.mass"),
            format!("link mass must be > 0 (got {mass})"),
        );
    }
    let inertia = parse_inertia(
        required(fields, "inertia", path)?,
        mass,
        &format!("{path}.inertia"),
    )?;

    let link = Link::new(
        parent,
        joint,
        joint_offset_in_parent,
        joint_offset_in_child,
        mass,
        inertia,
    );
    Ok((link, name))
}

fn approx_identity(q: Quat) -> bool {
    // Post-renormalization we're within f32 epsilon of the identity.
    (q.x.abs() < 1e-5) && (q.y.abs() < 1e-5) && (q.z.abs() < 1e-5) && ((q.w - 1.0).abs() < 1e-5)
}

// ---------------------------------------------------------------------------
// geom + site + actuator + contact_pairs parsers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn parse_geom(
    v: &Value,
    path: &str,
    bodies_by_name: &HashMap<String, usize>,
    trees_by_name: &HashMap<String, usize>,
    links_by_name: &[HashMap<String, usize>],
    meshes_by_name: &HashMap<String, usize>,
) -> Result<(Geom, String), ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(
        fields,
        &[
            "name",
            "shape",
            "attach",
            "local_offset",
            "local_orientation",
            "friction",
            "torsional_friction",
            "rolling_friction",
            "solref",
            "solimp",
            "margin",
            "gap",
            "condim",
        ],
        path,
    )?;
    let name = get_str(required(fields, "name", path)?, &format!("{path}.name"))?.to_string();
    if name.is_empty() {
        return fail(&format!("{path}.name"), "geom name must not be empty");
    }
    let shape = parse_geom_shape(
        required(fields, "shape", path)?,
        &format!("{path}.shape"),
        meshes_by_name,
    )?;
    let (attach_body, attach_link) = parse_attach(
        required(fields, "attach", path)?,
        &format!("{path}.attach"),
        bodies_by_name,
        trees_by_name,
        links_by_name,
        /*allow_static=*/ true,
    )?;
    if matches!(shape, GeomShape::Plane) && (attach_body.is_some() || attach_link.is_some()) {
        return fail(
            &format!("{path}.attach"),
            "plane geoms must have attach.kind = \"static\"",
        );
    }

    let local_offset = match optional(fields, "local_offset") {
        Some(v) => parse_vec3(v, &format!("{path}.local_offset"))?,
        None => Vec3::ZERO,
    };
    let local_orientation = match optional(fields, "local_orientation") {
        Some(v) => parse_quat(v, &format!("{path}.local_orientation"))?,
        None => Quat::IDENTITY,
    };
    let friction = match optional(fields, "friction") {
        Some(v) => {
            let f = get_f32(v, &format!("{path}.friction"))?;
            if f < 0.0 {
                return fail(&format!("{path}.friction"), "friction must be ≥ 0");
            }
            f
        }
        None => 0.5,
    };
    let solref = parse_optional_solref(fields, path)?;

    let (margin, gap) = parse_margin_gap(fields, path)?;
    let solimp = parse_optional_solimp(fields, path)?;
    let condim = parse_condim(fields, path)?;
    let torsional_friction = parse_nonneg_float(fields, "torsional_friction", path)?.unwrap_or(0.0);
    let rolling_friction = parse_nonneg_float(fields, "rolling_friction", path)?.unwrap_or(0.0);

    let geom = Geom {
        shape,
        body: attach_body,
        link: attach_link,
        local_offset,
        local_orientation,
        friction,
        solref,
        margin,
        gap,
        condim,
        torsional_friction,
        rolling_friction,
        solimp,
    };
    Ok((geom, name))
}

/// Parse an optional non-negative float field on a geom (or wherever). Returns
/// `None` if absent; `Err` if present and negative or non-finite.
fn parse_nonneg_float(
    fields: &[(String, Value)],
    key: &str,
    path: &str,
) -> Result<Option<f32>, ModelError> {
    let Some(v) = optional(fields, key) else {
        return Ok(None);
    };
    let f = get_f32(v, &format!("{path}.{key}"))?;
    if f < 0.0 {
        return fail(&format!("{path}.{key}"), format!("{key} must be ≥ 0"));
    }
    Ok(Some(f))
}

/// Parse the top-level `solver` object. Schema:
///
/// ```json
/// {
///   "mode": "penalty" | "pgs",
///   "iterations": 20,
///   "cone": "pyramidal" | "elliptic"
/// }
/// ```
///
/// All fields optional; omitted fields fall back to
/// [`crate::solver::SolverConfig::DEFAULT`]. `iterations` must be a
/// positive integer.
fn parse_solver_config(v: &Value, path: &str) -> Result<crate::solver::SolverConfig, ModelError> {
    use crate::solver::{ConeKind, SolverConfig, SolverMode};
    let fields = get_object(v, path)?;
    reject_unknown(fields, &["mode", "iterations", "cone"], path)?;
    let mut cfg = SolverConfig::DEFAULT;
    if let Some(mv) = optional(fields, "mode") {
        let s = get_str(mv, &format!("{path}.mode"))?;
        cfg.mode = match s {
            "penalty" => SolverMode::Penalty,
            "pgs" => SolverMode::Pgs,
            other => {
                return fail(
                    &format!("{path}.mode"),
                    format!("unknown solver mode \"{other}\"; expected penalty | pgs"),
                );
            }
        };
    }
    if let Some(iv) = optional(fields, "iterations") {
        let n = get_f32(iv, &format!("{path}.iterations"))?;
        if n < 1.0 || n != n.floor() {
            return fail(
                &format!("{path}.iterations"),
                format!("iterations must be a positive integer, got {n}"),
            );
        }
        cfg.iterations = n as u32;
    }
    if let Some(cv) = optional(fields, "cone") {
        let s = get_str(cv, &format!("{path}.cone"))?;
        cfg.cone = match s {
            "pyramidal" => ConeKind::Pyramidal,
            "elliptic" => ConeKind::Elliptic,
            other => {
                return fail(
                    &format!("{path}.cone"),
                    format!("unknown cone kind \"{other}\"; expected pyramidal | elliptic"),
                );
            }
        };
    }
    Ok(cfg)
}

/// Parse the optional `condim` field on a geom object. Defaults to `3`.
/// Accepts `1` (frictionless), `3` (sliding friction), `4` (adds
/// torsional friction about the normal), and `6` (adds rolling friction
/// about the two tangents). condim `≥ 4` also picks up the geom's
/// [`Geom::torsional_friction`] / [`Geom::rolling_friction`] coefficients.
fn parse_condim(fields: &[(String, Value)], path: &str) -> Result<u8, ModelError> {
    let Some(v) = optional(fields, "condim") else {
        return Ok(3);
    };
    let n = get_f32(v, &format!("{path}.condim"))?;
    if n != n.floor() {
        return fail(
            &format!("{path}.condim"),
            format!("condim must be an integer, got {n}"),
        );
    }
    let n_int = n as i32;
    match n_int {
        1 | 3 | 4 | 6 => Ok(n_int as u8),
        _ => fail(
            &format!("{path}.condim"),
            format!(
                "condim must be 1 (frictionless), 3 (sliding), 4 (torsional), \
                 or 6 (rolling); got {n_int}"
            ),
        ),
    }
}

/// Parse one entry of the top-level `meshes` array. Schema:
///
/// ```json
/// {
///   "name": "tetra",
///   "vertices": [[x, y, z], ...],   // ≥ 4 finite triples
///   "faces":    [[i, j, k], ...]    // ≥ 4 triangle index triples
/// }
/// ```
///
/// The mesh's convex hull is TRUSTED — the loader runs only the cheap
/// structural checks in [`crate::geom::ConvexMesh::validate`] (vertex
/// count, face count, index range, finite coordinates). Non-convex meshes
/// silently produce incorrect contacts.
fn parse_mesh_asset(
    v: &Value,
    path: &str,
) -> Result<(crate::geom::ConvexMesh, String), ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(fields, &["name", "vertices", "faces"], path)?;
    let name = get_str(required(fields, "name", path)?, &format!("{path}.name"))?.to_string();
    if name.is_empty() {
        return fail(&format!("{path}.name"), "mesh name must not be empty");
    }
    // Vertices — array of [x, y, z].
    let verts_v = required(fields, "vertices", path)?;
    let verts_arr = get_array(verts_v, &format!("{path}.vertices"))?;
    let mut vertices: Vec<Vec3> = Vec::with_capacity(verts_arr.len());
    for (i, ve) in verts_arr.iter().enumerate() {
        vertices.push(parse_vec3(ve, &format!("{path}.vertices[{i}]"))?);
    }
    // Faces — array of [i, j, k] indices.
    let faces_v = required(fields, "faces", path)?;
    let faces_arr = get_array(faces_v, &format!("{path}.faces"))?;
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(faces_arr.len());
    for (i, fe) in faces_arr.iter().enumerate() {
        let tri = get_array(fe, &format!("{path}.faces[{i}]"))?;
        if tri.len() != 3 {
            return fail(
                &format!("{path}.faces[{i}]"),
                format!("face must have exactly 3 vertex indices, got {}", tri.len()),
            );
        }
        let mut idx = [0u32; 3];
        for (k, e) in tri.iter().enumerate() {
            let n = get_f32(e, &format!("{path}.faces[{i}][{k}]"))?;
            if n < 0.0 || n != n.floor() {
                return fail(
                    &format!("{path}.faces[{i}][{k}]"),
                    format!("face vertex index must be a non-negative integer, got {n}"),
                );
            }
            idx[k] = n as u32;
        }
        faces.push(idx);
    }
    let mesh = crate::geom::ConvexMesh { vertices, faces };
    mesh.validate().map_err(|m| ModelError::new(path, m))?;
    Ok((mesh, name))
}

/// Parse the optional `margin` and `gap` fields on a geom object. Zero
/// defaults. Rejects negative values.
fn parse_margin_gap(fields: &[(String, Value)], path: &str) -> Result<(f32, f32), ModelError> {
    let margin = match optional(fields, "margin") {
        Some(v) => {
            let m = get_f32(v, &format!("{path}.margin"))?;
            if m < 0.0 {
                return fail(&format!("{path}.margin"), "margin must be ≥ 0");
            }
            m
        }
        None => 0.0,
    };
    let gap = match optional(fields, "gap") {
        Some(v) => {
            let g = get_f32(v, &format!("{path}.gap"))?;
            if g < 0.0 {
                return fail(&format!("{path}.gap"), "gap must be ≥ 0");
            }
            g
        }
        None => 0.0,
    };
    Ok((margin, gap))
}

fn parse_geom_shape(
    v: &Value,
    path: &str,
    meshes_by_name: &HashMap<String, usize>,
) -> Result<GeomShape, ModelError> {
    let fields = get_object(v, path)?;
    let kind = required(fields, "kind", path)?;
    let kind_s = get_str(kind, &format!("{path}.kind"))?;
    match kind_s {
        "plane" => {
            reject_unknown(fields, &["kind"], path)?;
            Ok(GeomShape::Plane)
        }
        "sphere" => {
            reject_unknown(fields, &["kind", "radius"], path)?;
            let r = get_f32(required(fields, "radius", path)?, &format!("{path}.radius"))?;
            if r <= 0.0 {
                return fail(path, "sphere radius must be > 0");
            }
            Ok(GeomShape::Sphere { radius: r })
        }
        "box" => {
            reject_unknown(fields, &["kind", "half_extents"], path)?;
            let he = parse_vec3(
                required(fields, "half_extents", path)?,
                &format!("{path}.half_extents"),
            )?;
            if he.x <= 0.0 || he.y <= 0.0 || he.z <= 0.0 {
                return fail(path, "box half_extents must be > 0 on every axis");
            }
            Ok(GeomShape::Box { half_extents: he })
        }
        "capsule" => {
            reject_unknown(fields, &["kind", "radius", "half_height"], path)?;
            let r = get_f32(required(fields, "radius", path)?, &format!("{path}.radius"))?;
            let h = get_f32(
                required(fields, "half_height", path)?,
                &format!("{path}.half_height"),
            )?;
            if r <= 0.0 || h < 0.0 {
                return fail(
                    path,
                    "capsule radius must be > 0 and half_height must be ≥ 0",
                );
            }
            Ok(GeomShape::Capsule {
                radius: r,
                half_height: h,
            })
        }
        "cylinder" => {
            reject_unknown(fields, &["kind", "radius", "half_height"], path)?;
            let r = get_f32(required(fields, "radius", path)?, &format!("{path}.radius"))?;
            let h = get_f32(
                required(fields, "half_height", path)?,
                &format!("{path}.half_height"),
            )?;
            if r <= 0.0 || h < 0.0 {
                return fail(
                    path,
                    "cylinder radius must be > 0 and half_height must be ≥ 0",
                );
            }
            Ok(GeomShape::Cylinder {
                radius: r,
                half_height: h,
            })
        }
        "ellipsoid" => {
            reject_unknown(fields, &["kind", "semi_axes"], path)?;
            let sa = parse_vec3(
                required(fields, "semi_axes", path)?,
                &format!("{path}.semi_axes"),
            )?;
            if sa.x <= 0.0 || sa.y <= 0.0 || sa.z <= 0.0 {
                return fail(path, "ellipsoid semi_axes must be > 0 on every axis");
            }
            Ok(GeomShape::Ellipsoid { semi_axes: sa })
        }
        "mesh" => {
            reject_unknown(fields, &["kind", "mesh"], path)?;
            let mn = get_str(required(fields, "mesh", path)?, &format!("{path}.mesh"))?;
            let mesh_id = meshes_by_name.get(mn).copied().ok_or_else(|| {
                ModelError::new(
                    format!("{path}.mesh"),
                    format!(
                        "unknown mesh name \"{mn}\" — declare it in the top-level \
                         \"meshes\" array before referencing"
                    ),
                )
            })?;
            Ok(GeomShape::Mesh { mesh_id })
        }
        other => fail(
            &format!("{path}.kind"),
            format!(
                "unknown geom shape \"{other}\"; expected \
                 plane | sphere | box | capsule | cylinder | ellipsoid | mesh"
            ),
        ),
    }
}

/// Parse an `attach` object. Returns `(body, link)` matching the [`Geom`]
/// convention (both `None` for static). If `allow_static` is false, a
/// static attach becomes an error (used for sites and actuators).
#[allow(clippy::type_complexity)]
fn parse_attach(
    v: &Value,
    path: &str,
    bodies_by_name: &HashMap<String, usize>,
    trees_by_name: &HashMap<String, usize>,
    links_by_name: &[HashMap<String, usize>],
    allow_static: bool,
) -> Result<(Option<usize>, Option<(usize, usize)>), ModelError> {
    let fields = get_object(v, path)?;
    let kind = required(fields, "kind", path)?;
    let kind_s = get_str(kind, &format!("{path}.kind"))?;
    match kind_s {
        "static" => {
            reject_unknown(fields, &["kind"], path)?;
            if !allow_static {
                return fail(
                    &format!("{path}.kind"),
                    "\"static\" attach is not allowed here",
                );
            }
            Ok((None, None))
        }
        "body" => {
            reject_unknown(fields, &["kind", "body"], path)?;
            let bn = get_str(required(fields, "body", path)?, &format!("{path}.body"))?;
            let idx = bodies_by_name.get(bn).copied().ok_or_else(|| {
                ModelError::new(
                    format!("{path}.body"),
                    format!("unknown body name \"{bn}\""),
                )
            })?;
            Ok((Some(idx), None))
        }
        "link" => {
            reject_unknown(fields, &["kind", "tree", "link"], path)?;
            let tn = get_str(required(fields, "tree", path)?, &format!("{path}.tree"))?;
            let tidx = trees_by_name.get(tn).copied().ok_or_else(|| {
                ModelError::new(
                    format!("{path}.tree"),
                    format!("unknown tree name \"{tn}\""),
                )
            })?;
            let ln = get_str(required(fields, "link", path)?, &format!("{path}.link"))?;
            let lidx = links_by_name[tidx].get(ln).copied().ok_or_else(|| {
                ModelError::new(
                    format!("{path}.link"),
                    format!("unknown link \"{ln}\" in tree \"{tn}\""),
                )
            })?;
            Ok((None, Some((tidx, lidx))))
        }
        other => fail(
            &format!("{path}.kind"),
            format!("unknown attach kind \"{other}\"; expected static | body | link"),
        ),
    }
}

fn parse_site(
    v: &Value,
    path: &str,
    bodies_by_name: &HashMap<String, usize>,
    trees_by_name: &HashMap<String, usize>,
    links_by_name: &[HashMap<String, usize>],
) -> Result<Site, ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(
        fields,
        &["name", "attach", "local_offset", "local_orientation"],
        path,
    )?;
    let name = get_str(required(fields, "name", path)?, &format!("{path}.name"))?.to_string();
    if name.is_empty() {
        return fail(&format!("{path}.name"), "site name must not be empty");
    }
    let (body, link) = parse_attach(
        required(fields, "attach", path)?,
        &format!("{path}.attach"),
        bodies_by_name,
        trees_by_name,
        links_by_name,
        /*allow_static=*/ false,
    )?;
    let attach = match (body, link) {
        (Some(i), None) => SiteAttach::Body(i),
        (None, Some((t, l))) => SiteAttach::Link { tree: t, link: l },
        _ => unreachable!("parse_attach with allow_static=false returns one of the two"),
    };
    let local_offset = match optional(fields, "local_offset") {
        Some(v) => parse_vec3(v, &format!("{path}.local_offset"))?,
        None => Vec3::ZERO,
    };
    let local_orientation = match optional(fields, "local_orientation") {
        Some(v) => parse_quat(v, &format!("{path}.local_orientation"))?,
        None => Quat::IDENTITY,
    };
    Ok(Site {
        name,
        attach,
        local_offset,
        local_orientation,
    })
}

fn parse_actuator(
    v: &Value,
    path: &str,
    trees_by_name: &HashMap<String, usize>,
    links_by_name: &[HashMap<String, usize>],
    world: &World,
) -> Result<(String, usize, PdServo), ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(
        fields,
        &[
            "name",
            "type",
            "tree",
            "link",
            "kp",
            "kd",
            "dampratio",
            "reflected_inertia",
            "clamp",
            "target",
        ],
        path,
    )?;
    let name = get_str(required(fields, "name", path)?, &format!("{path}.name"))?.to_string();
    let ty = get_str(required(fields, "type", path)?, &format!("{path}.type"))?;
    if ty != "position" {
        return fail(
            &format!("{path}.type"),
            format!("v0 supports only \"position\" actuators (got \"{ty}\")"),
        );
    }
    let tn = get_str(required(fields, "tree", path)?, &format!("{path}.tree"))?;
    let tidx = trees_by_name
        .get(tn)
        .copied()
        .ok_or_else(|| ModelError::new(format!("{path}.tree"), format!("unknown tree \"{tn}\"")))?;
    let ln = get_str(required(fields, "link", path)?, &format!("{path}.link"))?;
    let lidx = links_by_name[tidx].get(ln).copied().ok_or_else(|| {
        ModelError::new(
            format!("{path}.link"),
            format!("unknown link \"{ln}\" in tree \"{tn}\""),
        )
    })?;
    // Actuator must reference a hinge or slide (world's Tree::add_actuator
    // would panic otherwise; catch it here with a friendly path).
    if !matches!(
        world.trees[tidx].links[lidx].joint,
        JointKind::Hinge { .. } | JointKind::Slide { .. }
    ) {
        return fail(
            &format!("{path}.link"),
            format!(
                "actuator target link \"{ln}\" is not a hinge or slide; PD servos only actuate 1-DOF joints"
            ),
        );
    }

    let kp = get_f32(required(fields, "kp", path)?, &format!("{path}.kp"))?;
    if kp < 0.0 {
        return fail(&format!("{path}.kp"), "kp must be ≥ 0");
    }
    let clamp = optional(fields, "clamp")
        .map(|v| get_f32(v, &format!("{path}.clamp")))
        .transpose()?
        .unwrap_or(0.0);
    let target = optional(fields, "target")
        .map(|v| get_f32(v, &format!("{path}.target")))
        .transpose()?
        .unwrap_or(0.0);

    let has_kd = optional(fields, "kd").is_some();
    let has_dampratio = optional(fields, "dampratio").is_some();
    let has_ref_i = optional(fields, "reflected_inertia").is_some();

    let mut servo = if has_kd {
        if has_dampratio || has_ref_i {
            return fail(
                path,
                "specify either \"kd\" OR (\"dampratio\" + \"reflected_inertia\"), not both",
            );
        }
        let kd = get_f32(optional(fields, "kd").unwrap(), &format!("{path}.kd"))?;
        if kd < 0.0 {
            return fail(&format!("{path}.kd"), "kd must be ≥ 0");
        }
        PdServo::new(lidx, kp, kd, clamp, target)
    } else if has_dampratio {
        let zeta = get_f32(
            optional(fields, "dampratio").unwrap(),
            &format!("{path}.dampratio"),
        )?;
        if zeta < 0.0 {
            return fail(&format!("{path}.dampratio"), "dampratio must be ≥ 0");
        }
        if !has_ref_i {
            return fail(
                path,
                "\"dampratio\" requires \"reflected_inertia\" to derive kd",
            );
        }
        let ir = get_f32(
            optional(fields, "reflected_inertia").unwrap(),
            &format!("{path}.reflected_inertia"),
        )?;
        if ir <= 0.0 {
            return fail(
                &format!("{path}.reflected_inertia"),
                "reflected_inertia must be > 0",
            );
        }
        PdServo::from_dampratio(lidx, kp, zeta, ir, clamp)
    } else {
        return fail(
            path,
            "actuator must specify either \"kd\" or (\"dampratio\" + \"reflected_inertia\")",
        );
    };
    // `from_dampratio` sets target = 0; carry through the explicit target
    // for both branches for consistency.
    servo.target = target;

    Ok((name, tidx, servo))
}

/// Parse one entry in the top-level `"sensors"` array.
///
/// Schema (union over the sensor kinds — see `docs/sensors.md`):
///
/// ```json
/// // joint scalars (hinge / slide)
/// { "name": "hip_q",   "kind": "jointpos",     "tree": "t", "link": "hip" }
/// { "name": "hip_qd",  "kind": "jointvel",     "tree": "t", "link": "hip" }
/// // ball joint state
/// { "name": "sh_quat", "kind": "ballquat",     "tree": "t", "link": "sh" }
/// { "name": "sh_omg",  "kind": "ballangvel",   "tree": "t", "link": "sh" }
/// // site frame kinematics
/// { "name": "tip",     "kind": "framepos",     "site": "tip_site" }
/// { "name": "tipQ",    "kind": "framequat",    "site": "tip_site" }
/// { "name": "gyro",    "kind": "gyro",         "site": "imu_site" }
/// { "name": "accel",   "kind": "accelerometer","site": "imu_site" }
/// // contact / interaction sensors
/// { "name": "foot",    "kind": "touch",        "geom": "foot_pad" }
/// { "name": "elbowF",  "kind": "force",        "tree": "t", "link": "forearm" }
/// { "name": "elbowT",  "kind": "torque",       "tree": "t", "link": "forearm" }
/// ```
///
/// The `site` field on the site-frame kinds MUST reference a site defined
/// earlier in the `"sites"` array; the loader inlines the site's parent
/// attach and local pose into the sensor spec (sensors do not carry a
/// live reference to the `Site` table).
#[allow(clippy::too_many_arguments)]
fn parse_sensor(
    v: &Value,
    path: &str,
    bodies_by_name: &HashMap<String, usize>,
    trees_by_name: &HashMap<String, usize>,
    links_by_name: &[HashMap<String, usize>],
    geoms_by_name: &HashMap<String, usize>,
    sites: &[Site],
    sites_by_name: &HashMap<String, usize>,
) -> Result<Sensor, ModelError> {
    let fields = get_object(v, path)?;
    let name = get_str(required(fields, "name", path)?, &format!("{path}.name"))?.to_string();
    if name.is_empty() {
        return fail(&format!("{path}.name"), "sensor name must not be empty");
    }
    let kind_str = get_str(required(fields, "kind", path)?, &format!("{path}.kind"))?;
    let _ = bodies_by_name; // site kinds resolve via `sites_by_name`; joint kinds via `trees_by_name`.
    let kind = match kind_str {
        "jointpos" | "jointvel" | "ballquat" | "ballangvel" | "force" | "torque" => {
            reject_unknown(fields, &["name", "kind", "tree", "link"], path)?;
            let (t, l) = parse_tree_link_ref(fields, path, trees_by_name, links_by_name)?;
            match kind_str {
                "jointpos" => SensorKind::JointPos { tree: t, link: l },
                "jointvel" => SensorKind::JointVel { tree: t, link: l },
                "ballquat" => SensorKind::BallQuat { tree: t, link: l },
                "ballangvel" => SensorKind::BallAngVel { tree: t, link: l },
                "force" => SensorKind::Force { tree: t, link: l },
                "torque" => SensorKind::Torque { tree: t, link: l },
                _ => unreachable!(),
            }
        }
        "framepos" | "framequat" | "gyro" | "accelerometer" => {
            reject_unknown(fields, &["name", "kind", "site"], path)?;
            let sn = get_str(required(fields, "site", path)?, &format!("{path}.site"))?;
            let sidx = sites_by_name.get(sn).copied().ok_or_else(|| {
                ModelError::new(
                    format!("{path}.site"),
                    format!("unknown site \"{sn}\" — declare it in the top-level \"sites\" array"),
                )
            })?;
            let frame = site_to_frame(&sites[sidx]);
            match kind_str {
                "framepos" => SensorKind::FramePos(frame),
                "framequat" => SensorKind::FrameQuat(frame),
                "gyro" => SensorKind::Gyro(frame),
                "accelerometer" => SensorKind::Accelerometer(frame),
                _ => unreachable!(),
            }
        }
        "touch" => {
            reject_unknown(fields, &["name", "kind", "geom"], path)?;
            let gn = get_str(required(fields, "geom", path)?, &format!("{path}.geom"))?;
            let gidx = geoms_by_name.get(gn).copied().ok_or_else(|| {
                ModelError::new(format!("{path}.geom"), format!("unknown geom \"{gn}\""))
            })?;
            SensorKind::Touch { geom: gidx }
        }
        other => {
            return fail(
                &format!("{path}.kind"),
                format!(
                    "unknown sensor kind \"{other}\"; expected \
                     jointpos | jointvel | ballquat | ballangvel | framepos | framequat | \
                     gyro | accelerometer | touch | force | torque"
                ),
            );
        }
    };
    Ok(Sensor { name, kind })
}

fn parse_tree_link_ref(
    fields: &[(String, Value)],
    path: &str,
    trees_by_name: &HashMap<String, usize>,
    links_by_name: &[HashMap<String, usize>],
) -> Result<(usize, usize), ModelError> {
    let tn = get_str(required(fields, "tree", path)?, &format!("{path}.tree"))?;
    let tidx = trees_by_name
        .get(tn)
        .copied()
        .ok_or_else(|| ModelError::new(format!("{path}.tree"), format!("unknown tree \"{tn}\"")))?;
    let ln = get_str(required(fields, "link", path)?, &format!("{path}.link"))?;
    let lidx = links_by_name[tidx].get(ln).copied().ok_or_else(|| {
        ModelError::new(
            format!("{path}.link"),
            format!("unknown link \"{ln}\" in tree \"{tn}\""),
        )
    })?;
    Ok((tidx, lidx))
}

fn site_to_frame(site: &Site) -> SiteFrame {
    let attach = match site.attach {
        SiteAttach::Body(i) => SensorAttach::Body(i),
        SiteAttach::Link { tree, link } => SensorAttach::Link(tree, link),
    };
    SiteFrame {
        attach,
        local_offset: site.local_offset,
        local_orientation: site.local_orientation,
    }
}

/// Parse one entry in the top-level `"equality"` array.
///
/// ```json
/// { "kind": "connect", "body_a": "A", "body_b": "B",
///   "anchor_a": [x, y, z], "anchor_b": [x, y, z],
///   "solref": {...}, "solimp": {...} }
///
/// { "kind": "weld", ..., "relative_orientation": [x, y, z, w] }
///
/// { "kind": "joint", "tree": "t", "joint_a": "a", "joint_b": "b",
///   "polycoef": [c0, c1, c2] }
///
/// { "kind": "distance", ..., "distance": 1.5 }
/// ```
///
/// A `body_*` field of `"world"` (or absent) attaches that side to the
/// world at the raw `anchor_*` world-frame point. `solref` / `solimp`
/// share the schema of the geom fields; both are optional (defaults
/// apply).
fn parse_equality(
    v: &Value,
    path: &str,
    bodies_by_name: &HashMap<String, usize>,
    trees_by_name: &HashMap<String, usize>,
    links_by_name: &[HashMap<String, usize>],
    world: &World,
) -> Result<Equality, ModelError> {
    let fields = get_object(v, path)?;
    let kind = get_str(required(fields, "kind", path)?, &format!("{path}.kind"))?;
    match kind {
        "connect" => {
            reject_unknown(
                fields,
                &[
                    "kind", "body_a", "body_b", "anchor_a", "anchor_b", "solref", "solimp",
                ],
                path,
            )?;
            let body_a = parse_optional_body_ref(fields, "body_a", path, bodies_by_name)?;
            let body_b = parse_optional_body_ref(fields, "body_b", path, bodies_by_name)?;
            if body_a.is_none() && body_b.is_none() {
                return fail(
                    path,
                    "connect must reference at least one body (not two worlds)",
                );
            }
            let anchor_a = parse_vec3(
                required(fields, "anchor_a", path)?,
                &format!("{path}.anchor_a"),
            )?;
            let anchor_b = parse_vec3(
                required(fields, "anchor_b", path)?,
                &format!("{path}.anchor_b"),
            )?;
            let solref = parse_optional_solref(fields, path)?;
            let solimp = parse_optional_solimp(fields, path)?;
            Ok(Equality::Connect {
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                solref,
                solimp,
            })
        }
        "weld" => {
            reject_unknown(
                fields,
                &[
                    "kind",
                    "body_a",
                    "body_b",
                    "anchor_a",
                    "anchor_b",
                    "relative_orientation",
                    "solref",
                    "solimp",
                ],
                path,
            )?;
            let body_a = parse_optional_body_ref(fields, "body_a", path, bodies_by_name)?;
            let body_b = parse_optional_body_ref(fields, "body_b", path, bodies_by_name)?;
            if body_a.is_none() && body_b.is_none() {
                return fail(
                    path,
                    "weld must reference at least one body (not two worlds)",
                );
            }
            let anchor_a = parse_vec3(
                required(fields, "anchor_a", path)?,
                &format!("{path}.anchor_a"),
            )?;
            let anchor_b = parse_vec3(
                required(fields, "anchor_b", path)?,
                &format!("{path}.anchor_b"),
            )?;
            let relative_orientation = match optional(fields, "relative_orientation") {
                Some(v) => parse_quat(v, &format!("{path}.relative_orientation"))?,
                None => Quat::IDENTITY,
            };
            let solref = parse_optional_solref(fields, path)?;
            let solimp = parse_optional_solimp(fields, path)?;
            Ok(Equality::Weld {
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                relative_orientation,
                solref,
                solimp,
            })
        }
        "joint" => {
            reject_unknown(
                fields,
                &[
                    "kind", "tree", "joint_a", "joint_b", "polycoef", "solref", "solimp",
                ],
                path,
            )?;
            let tn = get_str(required(fields, "tree", path)?, &format!("{path}.tree"))?;
            let tidx = trees_by_name.get(tn).copied().ok_or_else(|| {
                ModelError::new(format!("{path}.tree"), format!("unknown tree \"{tn}\""))
            })?;
            let (link_a, la_name) =
                resolve_1dof_joint(fields, "joint_a", path, tidx, tn, links_by_name, world)?;
            let (link_b, lb_name) =
                resolve_1dof_joint(fields, "joint_b", path, tidx, tn, links_by_name, world)?;
            if link_a == link_b {
                return fail(
                    path,
                    format!("joint coupling endpoints must be distinct (both are \"{la_name}\")"),
                );
            }
            let _ = lb_name;
            let polycoef = parse_polycoef(fields, path)?;
            let solref = parse_optional_solref(fields, path)?;
            let solimp = parse_optional_solimp(fields, path)?;
            Ok(Equality::JointCoupling {
                tree: tidx,
                link_a,
                link_b,
                polycoef,
                solref,
                solimp,
            })
        }
        "distance" => {
            reject_unknown(
                fields,
                &[
                    "kind", "body_a", "body_b", "anchor_a", "anchor_b", "distance", "solref",
                    "solimp",
                ],
                path,
            )?;
            let body_a = parse_optional_body_ref(fields, "body_a", path, bodies_by_name)?;
            let body_b = parse_optional_body_ref(fields, "body_b", path, bodies_by_name)?;
            if body_a.is_none() && body_b.is_none() {
                return fail(
                    path,
                    "distance must reference at least one body (not two worlds)",
                );
            }
            let anchor_a = parse_vec3(
                required(fields, "anchor_a", path)?,
                &format!("{path}.anchor_a"),
            )?;
            let anchor_b = parse_vec3(
                required(fields, "anchor_b", path)?,
                &format!("{path}.anchor_b"),
            )?;
            let distance = get_f32(
                required(fields, "distance", path)?,
                &format!("{path}.distance"),
            )?;
            if distance < 0.0 {
                return fail(&format!("{path}.distance"), "distance must be ≥ 0");
            }
            let solref = parse_optional_solref(fields, path)?;
            let solimp = parse_optional_solimp(fields, path)?;
            Ok(Equality::Distance {
                body_a,
                body_b,
                anchor_a,
                anchor_b,
                distance,
                solref,
                solimp,
            })
        }
        other => fail(
            &format!("{path}.kind"),
            format!(
                "unknown equality kind \"{other}\"; expected connect | weld | joint | distance"
            ),
        ),
    }
}

/// Parse a `body_a` / `body_b` field. `None` (missing) or the string
/// `"world"` returns `None`; a body name resolves to that body's index.
fn parse_optional_body_ref(
    fields: &[(String, Value)],
    key: &str,
    path: &str,
    bodies_by_name: &HashMap<String, usize>,
) -> Result<Option<usize>, ModelError> {
    let Some(v) = optional(fields, key) else {
        return Ok(None);
    };
    let s = get_str(v, &format!("{path}.{key}"))?;
    if s == "world" {
        return Ok(None);
    }
    bodies_by_name
        .get(s)
        .copied()
        .map(Some)
        .ok_or_else(|| ModelError::new(format!("{path}.{key}"), format!("unknown body \"{s}\"")))
}

/// Resolve a joint reference on a coupling equality to a link index and
/// enforce that it names a 1-DOF joint (hinge or slide).
fn resolve_1dof_joint(
    fields: &[(String, Value)],
    key: &str,
    path: &str,
    tree_idx: usize,
    tree_name: &str,
    links_by_name: &[HashMap<String, usize>],
    world: &World,
) -> Result<(usize, String), ModelError> {
    let n = get_str(required(fields, key, path)?, &format!("{path}.{key}"))?;
    let lidx = links_by_name[tree_idx].get(n).copied().ok_or_else(|| {
        ModelError::new(
            format!("{path}.{key}"),
            format!("unknown link \"{n}\" in tree \"{tree_name}\""),
        )
    })?;
    if !matches!(
        world.trees[tree_idx].links[lidx].joint,
        JointKind::Hinge { .. } | JointKind::Slide { .. }
    ) {
        return fail(
            &format!("{path}.{key}"),
            format!(
                "joint \"{n}\" must be a hinge or slide (joint coupling is scalar; not supported for free/fixed/ball)"
            ),
        );
    }
    Ok((lidx, n.to_string()))
}

/// Parse the required `polycoef` array as `[c0, c1, c2]`. Accepts an
/// array of length 1, 2, or 3; missing tail entries default to 0.
fn parse_polycoef(fields: &[(String, Value)], path: &str) -> Result<[f32; 3], ModelError> {
    let v = required(fields, "polycoef", path)?;
    let arr = get_array(v, &format!("{path}.polycoef"))?;
    if arr.is_empty() || arr.len() > 3 {
        return fail(
            &format!("{path}.polycoef"),
            format!("polycoef must have 1, 2, or 3 entries (got {})", arr.len()),
        );
    }
    let mut out = [0.0f32; 3];
    for (i, entry) in arr.iter().enumerate() {
        out[i] = get_f32(entry, &format!("{path}.polycoef[{i}]"))?;
    }
    Ok(out)
}

/// Optional `solref` on an equality object. Same schema as geoms — see
/// [`parse_geom`].
fn parse_optional_solref(fields: &[(String, Value)], path: &str) -> Result<SolRef, ModelError> {
    let Some(v) = optional(fields, "solref") else {
        return Ok(SolRef::DEFAULT);
    };
    let sf = get_object(v, &format!("{path}.solref"))?;
    reject_unknown(sf, &["timeconst", "dampratio"], &format!("{path}.solref"))?;
    let tc = get_f32(
        required(sf, "timeconst", &format!("{path}.solref"))?,
        &format!("{path}.solref.timeconst"),
    )?;
    let zeta = get_f32(
        required(sf, "dampratio", &format!("{path}.solref"))?,
        &format!("{path}.solref.dampratio"),
    )?;
    if tc <= 0.0 || zeta < 0.0 {
        return fail(
            &format!("{path}.solref"),
            "timeconst must be > 0 and dampratio must be ≥ 0",
        );
    }
    Ok(SolRef::new(tc, zeta))
}

/// Optional `solimp` on a geom or equality object. Returns
/// [`SolImp::DEFAULT`] when absent. Validates ranges (see
/// [`SolImp::validate`]).
fn parse_optional_solimp(
    fields: &[(String, Value)],
    path: &str,
) -> Result<crate::solver::SolImp, ModelError> {
    let Some(v) = optional(fields, "solimp") else {
        return Ok(crate::solver::SolImp::DEFAULT);
    };
    let sf = get_object(v, &format!("{path}.solimp"))?;
    reject_unknown(
        sf,
        &["dmin", "dmax", "width", "midpoint", "power"],
        &format!("{path}.solimp"),
    )?;
    let dmin = get_f32(
        required(sf, "dmin", &format!("{path}.solimp"))?,
        &format!("{path}.solimp.dmin"),
    )?;
    let dmax = get_f32(
        required(sf, "dmax", &format!("{path}.solimp"))?,
        &format!("{path}.solimp.dmax"),
    )?;
    let width = get_f32(
        required(sf, "width", &format!("{path}.solimp"))?,
        &format!("{path}.solimp.width"),
    )?;
    let midpoint = get_f32(
        required(sf, "midpoint", &format!("{path}.solimp"))?,
        &format!("{path}.solimp.midpoint"),
    )?;
    let power_f = get_f32(
        required(sf, "power", &format!("{path}.solimp"))?,
        &format!("{path}.solimp.power"),
    )?;
    if power_f < 1.0 || power_f != power_f.floor() {
        return fail(
            &format!("{path}.solimp.power"),
            format!("power must be a positive integer, got {power_f}"),
        );
    }
    let s = crate::solver::SolImp::new(dmin, dmax, width, midpoint, power_f as u32);
    s.validate()
        .map_err(|m| ModelError::new(format!("{path}.solimp"), m))?;
    Ok(s)
}

fn parse_contact_pairs(
    v: &Value,
    path: &str,
    world: &World,
    geoms_by_name: &HashMap<String, usize>,
    tree_self_collide: &[bool],
) -> Result<Vec<(usize, usize)>, ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(fields, &["explicit", "disable"], path)?;
    let has_explicit = optional(fields, "explicit").is_some();
    let has_disable = optional(fields, "disable").is_some();
    if has_explicit && has_disable {
        return fail(
            path,
            "\"explicit\" and \"disable\" are mutually exclusive; pass one or the other",
        );
    }
    if let Some(v) = optional(fields, "explicit") {
        let arr = get_array(v, &format!("{path}.explicit"))?;
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (i, pv) in arr.iter().enumerate() {
            let p = format!("{path}.explicit[{i}]");
            let (a, b) = parse_pair(pv, &p, geoms_by_name)?;
            out.push(if a < b { (a, b) } else { (b, a) });
        }
        return Ok(out);
    }
    // No explicit list. Start with the world's auto-pairs then subtract
    // disable + apply self-collision.
    let mut auto = auto_pairs_with_self_collision_filter(world, tree_self_collide);
    if let Some(v) = optional(fields, "disable") {
        let arr = get_array(v, &format!("{path}.disable"))?;
        let mut disable: Vec<(usize, usize)> = Vec::new();
        for (i, pv) in arr.iter().enumerate() {
            let p = format!("{path}.disable[{i}]");
            let (a, b) = parse_pair(pv, &p, geoms_by_name)?;
            disable.push(if a < b { (a, b) } else { (b, a) });
        }
        auto.retain(|pair| !disable.contains(pair));
    }
    Ok(auto)
}

fn parse_pair(
    v: &Value,
    path: &str,
    geoms_by_name: &HashMap<String, usize>,
) -> Result<(usize, usize), ModelError> {
    let fields = get_object(v, path)?;
    reject_unknown(fields, &["a", "b"], path)?;
    let an = get_str(required(fields, "a", path)?, &format!("{path}.a"))?;
    let bn = get_str(required(fields, "b", path)?, &format!("{path}.b"))?;
    let a = geoms_by_name
        .get(an)
        .copied()
        .ok_or_else(|| ModelError::new(format!("{path}.a"), format!("unknown geom \"{an}\"")))?;
    let b = geoms_by_name
        .get(bn)
        .copied()
        .ok_or_else(|| ModelError::new(format!("{path}.b"), format!("unknown geom \"{bn}\"")))?;
    if a == b {
        return fail(path, "contact pair endpoints must be distinct geoms");
    }
    Ok((a, b))
}

/// Enumerate all valid contact pairs, dropping pairs where both geoms live
/// on the same tree AND that tree opts out of self-collision. Same
/// `(min, max)` sorted order as `World::auto_pairs`.
fn auto_pairs_with_self_collision_filter(
    world: &World,
    tree_self_collide: &[bool],
) -> Vec<(usize, usize)> {
    use crate::geom::GeomAttach;
    let mut out = Vec::new();
    let n = world.geoms.len();
    for a in 0..n {
        for b in (a + 1)..n {
            let att_a = world.geoms[a].attachment();
            let att_b = world.geoms[b].attachment();
            if att_a == att_b {
                continue;
            }
            // Same-tree opt-out: if both geoms attach to the same tree AND
            // that tree has self_collide=false, drop the pair.
            if let (GeomAttach::Link(ta, _), GeomAttach::Link(tb, _)) = (att_a, att_b) {
                if ta == tb && !tree_self_collide.get(ta).copied().unwrap_or(true) {
                    continue;
                }
            }
            out.push((a, b));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// tests — unit tests for the loader validation branches
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(src: &str) -> Scene {
        load_str(src).unwrap_or_else(|e| panic!("expected scene, got error: {e}"))
    }

    fn err(src: &str) -> ModelError {
        load_str(src).expect_err("expected error")
    }

    #[test]
    fn empty_scene_loads_default_world() {
        let s = scene("{}");
        assert_eq!(s.world.gravity, Vec3::new(0.0, 0.0, -9.81));
        assert!(s.world.bodies.is_empty());
        assert!(s.world.trees.is_empty());
    }

    #[test]
    fn top_level_unknown_field_rejected() {
        let e = err(r#"{"dampign":0.5}"#);
        assert!(
            e.message.contains("unknown field \"dampign\""),
            "message: {}",
            e.message
        );
    }

    #[test]
    fn duplicate_top_level_field_rejected() {
        let e = err(r#"{"gravity":[0,0,-9.81],"gravity":[0,0,0]}"#);
        assert!(e.message.contains("duplicate"), "{}", e.message);
    }

    #[test]
    fn nonpositive_mass_rejected() {
        let src = r#"{
            "bodies":[{"name":"a","mass":0.0,"inertia":{"kind":"diag","values":[1,1,1]}}]
        }"#;
        let e = err(src);
        assert!(e.path.contains("bodies[0].mass"), "path: {}", e.path);
    }

    #[test]
    fn negative_inertia_axis_rejected() {
        let src = r#"{
            "bodies":[{"name":"a","mass":1,"inertia":{"kind":"diag","values":[-1,1,1]}}]
        }"#;
        let e = err(src);
        assert!(e.path.contains("bodies[0].inertia"), "path: {}", e.path);
    }

    #[test]
    fn wrong_length_tensor_rejected() {
        let src = r#"{
            "bodies":[{"name":"a","mass":1,"inertia":{"kind":"tensor","values":[1,1,1,1]}}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("6 numbers"), "{}", e.message);
    }

    #[test]
    fn dangling_parent_rejected() {
        let src = r#"{
            "trees":[{
              "name":"t",
              "links":[
                {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
                {"name":"child","parent":"nope","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1e-6]}}
              ]
            }]
        }"#;
        let e = err(src);
        assert!(e.path.contains("parent"), "path: {}", e.path);
        assert!(e.message.contains("unknown parent"), "{}", e.message);
    }

    #[test]
    fn hinge_root_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("root joint"), "{}", e.message);
    }

    #[test]
    fn zero_slide_axis_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"s","parent":"root","joint":{"kind":"slide","axis":[0,0,0]},"mass":1,"inertia":{"kind":"diag","values":[1e-4,1e-4,1e-4]}}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("slide axis"), "{}", e.message);
        assert!(e.message.contains("non-zero"), "{}", e.message);
    }

    #[test]
    fn slide_at_root_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"slide","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("root joint"), "{}", e.message);
        assert!(e.message.contains("slide"), "{}", e.message);
    }

    #[test]
    fn ball_at_root_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"ball"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("root joint"), "{}", e.message);
        assert!(e.message.contains("ball"), "{}", e.message);
    }

    #[test]
    fn ball_with_range_rejected_with_solver_deferral_hint() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"b","parent":"root","joint":{"kind":"ball","range":[-1,1]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.path.ends_with(".range"), "path: {}", e.path);
        assert!(
            e.message.contains("do not support") && e.message.contains("solver"),
            "message should mention deferral to solver: {}",
            e.message
        );
    }

    #[test]
    fn slide_link_all_fields_round_trip() {
        // Positive-side sanity: slide with damping, armature, and range
        // loads cleanly and its DOF counts are wired through.
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"s","parent":"root",
               "joint":{"kind":"slide","axis":[0,0,1],"range":[-0.5,0.5],"damping":0.1,"armature":0.05,
                        "limit":{"stiffness":1500,"damping":50}},
               "mass":1.5,"inertia":{"kind":"diag","values":[1e-4,1e-4,1e-4]}}
            ]}]
        }"#;
        let s = scene(src);
        let tree = &s.world.trees[0];
        assert_eq!(tree.nq(), 1);
        assert_eq!(tree.nv(), 1);
        assert!(matches!(tree.links[1].joint, JointKind::Slide { .. }));
    }

    #[test]
    fn ball_link_dof_counts_wire_through() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"b","parent":"root","joint":{"kind":"ball","damping":0.2,"armature":0.03},
               "mass":1.0,"inertia":{"kind":"diag","values":[0.05,0.05,1e-6]}}
            ]}]
        }"#;
        let s = scene(src);
        let tree = &s.world.trees[0];
        assert_eq!(tree.nq(), 4);
        assert_eq!(tree.nv(), 3);
        // Default ball q = identity quaternion (renormalized to `(0,0,0,1)`).
        assert_eq!(tree.q[0], 0.0);
        assert_eq!(tree.q[1], 0.0);
        assert_eq!(tree.q[2], 0.0);
        assert_eq!(tree.q[3], 1.0);
        assert!(matches!(tree.links[1].joint, JointKind::Ball { .. }));
    }

    #[test]
    fn actuator_on_slide_accepted() {
        // PD servo on a slide joint should load cleanly (v1 tier 1 adds
        // slide-actuation coverage; previously the loader rejected non-hinge
        // actuator targets).
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"s","parent":"root","joint":{"kind":"slide","axis":[1,0,0]},
               "mass":1,"inertia":{"kind":"diag","values":[1e-4,1e-4,1e-4]}}
            ]}],
            "actuators":[{"name":"a","type":"position","tree":"t","link":"s","kp":100,"kd":10}]
        }"#;
        let s = scene(src);
        assert!(s.actuators_by_name.contains_key("a"));
    }

    #[test]
    fn actuator_on_ball_rejected() {
        // The 1-DOF PD servo cannot address a 3-DOF ball joint's rotational
        // slots. Reject at load time.
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"b","parent":"root","joint":{"kind":"ball"},
               "mass":1,"inertia":{"kind":"diag","values":[0.05,0.05,1e-6]}}
            ]}],
            "actuators":[{"name":"a","type":"position","tree":"t","link":"b","kp":100,"kd":10}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("hinge or slide"), "{}", e.message);
    }

    #[test]
    fn zero_hinge_axis_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"child","parent":"root","joint":{"kind":"hinge","axis":[0,0,0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1e-6]}}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("non-zero"), "{}", e.message);
    }

    #[test]
    fn actuator_on_non_hinge_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
            ]}],
            "actuators":[{"name":"a","type":"position","tree":"t","link":"root","kp":10,"kd":1}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("not a hinge"), "{}", e.message);
    }

    #[test]
    fn actuator_missing_kd_and_dampratio_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"h","parent":"root","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1e-6]}}
            ]}],
            "actuators":[{"name":"a","type":"position","tree":"t","link":"h","kp":10}]
        }"#;
        let e = err(src);
        assert!(
            e.message.contains("kd") && e.message.contains("dampratio"),
            "{}",
            e.message
        );
    }

    #[test]
    fn site_on_missing_body_rejected() {
        let src = r#"{
            "sites":[{"name":"s","attach":{"kind":"body","body":"nope"}}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("unknown body"), "{}", e.message);
    }

    #[test]
    fn plane_with_body_attach_rejected() {
        let src = r#"{
            "bodies":[{"name":"b","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}],
            "geoms":[{"name":"g","shape":{"kind":"plane"},"attach":{"kind":"body","body":"b"}}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("plane"), "{}", e.message);
    }

    #[test]
    fn duplicate_body_name_rejected() {
        let src = r#"{
            "bodies":[
              {"name":"b","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"b","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
            ]
        }"#;
        let e = err(src);
        assert!(e.message.contains("duplicate"), "{}", e.message);
    }

    #[test]
    fn negative_timestep_rejected() {
        let e = err(r#"{"timestep":-0.001}"#);
        assert!(e.message.contains("timestep"), "{}", e.message);
    }

    #[test]
    fn unknown_link_field_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]},"dampign":0.5}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("dampign"), "{}", e.message);
    }

    #[test]
    fn hinge_range_lo_ge_hi_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"h","parent":"root","joint":{"kind":"hinge","axis":[1,0,0],"range":[1.0, 1.0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1e-6]}}
            ]}]
        }"#;
        let e = err(src);
        assert!(e.message.contains("range"), "{}", e.message);
    }

    #[test]
    fn duplicate_actuator_name_rejected() {
        let src = r#"{
            "trees":[{"name":"t","links":[
              {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
              {"name":"h","parent":"root","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1e-6]}}
            ]}],
            "actuators":[
              {"name":"a","type":"position","tree":"t","link":"h","kp":10,"kd":1},
              {"name":"a","type":"position","tree":"t","link":"h","kp":10,"kd":1}
            ]
        }"#;
        let e = err(src);
        assert!(e.message.contains("duplicate"), "{}", e.message);
    }

    #[test]
    fn version_mismatch_rejected() {
        let e = err(r#"{"version":"2"}"#);
        assert!(e.message.contains("version"), "{}", e.message);
    }

    #[test]
    fn contact_pairs_explicit_and_disable_together_rejected() {
        let src = r#"{
            "geoms":[
              {"name":"a","shape":{"kind":"plane"},"attach":{"kind":"static"}},
              {"name":"b","shape":{"kind":"plane"},"attach":{"kind":"static"}}
            ],
            "contact_pairs":{
              "explicit":[{"a":"a","b":"b"}],
              "disable" :[{"a":"a","b":"b"}]
            }
        }"#;
        let e = err(src);
        assert!(
            e.message.contains("mutually exclusive"),
            "message: {}",
            e.message
        );
        assert_eq!(e.path, "contact_pairs");
    }

    #[test]
    fn contact_pair_unknown_geom_rejected() {
        let src = r#"{
            "geoms":[{"name":"g","shape":{"kind":"plane"},"attach":{"kind":"static"}}],
            "contact_pairs":{"explicit":[{"a":"g","b":"nope"}]}
        }"#;
        let e = err(src);
        assert!(e.message.contains("unknown geom"), "{}", e.message);
    }

    #[test]
    fn scene_with_body_and_geom_round_trips_indexing() {
        let src = r#"{
            "bodies":[{"name":"a","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.5}}}],
            "geoms":[
              {"name":"ground","shape":{"kind":"plane"},"attach":{"kind":"static"}},
              {"name":"ball","shape":{"kind":"sphere","radius":0.5},"attach":{"kind":"body","body":"a"}}
            ]
        }"#;
        let s = scene(src);
        assert_eq!(s.world.bodies.len(), 1);
        assert_eq!(s.world.geoms.len(), 2);
        assert_eq!(s.bodies_by_name.get("a"), Some(&0));
        assert_eq!(s.geoms_by_name.get("ground"), Some(&0));
        assert_eq!(s.geoms_by_name.get("ball"), Some(&1));
    }

    #[test]
    fn site_world_pose_on_rotated_link_matches_hand_computation() {
        // Fixed root at world (0.5, 0, 0), rotated 90° about z. A site on
        // the root at local_offset (1, 0, 0) sits at world (0.5 + 0, 0 + 1, 0)
        // = (0.5, 1, 0), and its world orientation should equal the parent's.
        let src = r#"{
            "trees":[{"name":"t","links":[{
              "name":"root",
              "joint":{"kind":"fixed"},
              "joint_offset_in_parent":{"position":[0.5,0,0],"orientation":[0,0,0.7071068,0.7071068]},
              "mass":1,
              "inertia":{"kind":"diag","values":[1,1,1]}
            }]}],
            "sites":[{"name":"tip","attach":{"kind":"link","tree":"t","link":"root"},"local_offset":[1,0,0]}]
        }"#;
        let s = scene(src);
        let (pos, ori) = s.site_pose("tip").expect("site should be present");
        // World +y is body +x after Rot_z(π/2).
        assert!((pos.x - 0.5).abs() < 1e-4, "x = {}", pos.x);
        assert!((pos.y - 1.0).abs() < 1e-4, "y = {}", pos.y);
        assert!(pos.z.abs() < 1e-4, "z = {}", pos.z);
        // Orientation matches the parent link's world orientation.
        assert!((ori.z - (std::f32::consts::SQRT_2 * 0.5)).abs() < 1e-3);
        assert!((ori.w - (std::f32::consts::SQRT_2 * 0.5)).abs() < 1e-3);
    }
}
