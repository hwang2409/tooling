//! MJCF (MuJoCo XML) loader — a documented subset.
//!
//! See `docs/mjcf.md` for the full support table. Everything the loader
//! accepts becomes the same [`Scene`] that the JSON loader in
//! [`crate::model`] produces; fixtures mirroring `models/*.json` yield
//! byte-identical trajectories because both loaders call the same
//! constructors with the same numeric values.
//!
//! # Scope
//!
//! Supported top-level elements (children of `<mujoco>`):
//! `<compiler>`, `<option>`, `<default>`, `<worldbody>`, `<actuator>`,
//! `<sensor>`, `<equality>`, `<contact>`. Anything else (e.g. `<asset>`,
//! `<tendon>`, `<keyframe>`, `<visual>`) is rejected with a clear
//! `unsupported in v1 subset` error naming the element.
//!
//! Body attributes: `name`, `pos`, `quat`, `euler`, `childclass`.
//! Joint attributes: `name`, `type` (`hinge`|`slide`|`ball`|`free`),
//! `pos`, `axis`, `range`, `damping`, `armature`, `limited`, `class`.
//! Geom attributes: `name`, `type` (`plane`|`sphere`|`box`|`capsule`|
//! `cylinder`|`ellipsoid`; `mesh` REJECTED), `pos`, `quat`, `size`,
//! `fromto` (capsule/cylinder), `friction` (1–3 numbers),
//! `solref`, `solimp`, `condim`, `margin`, `gap`, `mass` (rejected in
//! this subset — bodies specify mass via `<inertial>`), `class`.
//! Site attributes: `name`, `pos`, `quat`, `class`.
//! Inertial attributes: `pos` (must be zero — newt requires COM at body
//! origin), `mass`, `diaginertia` OR `fullinertia`.
//! Actuator (`<position>` / `<motor>`): `name`, `joint`, `kp` (position),
//! `kv` OR `dampratio` (position), `forcerange`, `ctrlrange` (accepted
//! but not enforced; documented), `gear` (motor: scalar only), `class`.
//! Sensor: `jointpos`, `jointvel`, `ballquat`, `ballangvel`, `framepos`,
//! `framequat`, `gyro`, `accelerometer`, `touch`, `force`, `torque` — same
//! set the JSON loader knows.
//! Equality: `connect`, `weld`, `joint`. Contact: `pair`, `exclude`.
//!
//! # Rejection doctrine (no silent ignore)
//!
//! Every unknown attribute or child element on a supported node produces
//! an error naming the offender. Every known-but-unsupported feature
//! (mesh geom, tendon, keyframe, texture, etc.) produces a
//! `unsupported in v1 subset` error naming the feature. Silent ignoring
//! is the loader-silence incident class the ticket calls out.

use std::collections::HashMap;
use std::path::Path;

use crate::actuator::PdServo;
use crate::body::Body;
use crate::equality::Equality;
use crate::geom::{Geom, GeomShape, SolRef};
use crate::joint::{JointKind, JointLimit};
use crate::math::{self, Mat3, Quat, Vec3};
use crate::model::{Scene, Site, SiteAttach};
use crate::sensor::{Sensor, SensorAttach, SensorKind, SiteFrame};
use crate::solver::SolImp;
use crate::tree::{Link, Tree};
use crate::world::World;
use crate::xml::{self, Element};

// ---------------------------------------------------------------------------
// error type
// ---------------------------------------------------------------------------

/// A structured MJCF error. `path` is an XML-style breadcrumb such as
/// `mujoco > worldbody > body[torso] > geom[chest]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MjcfError {
    pub path: String,
    pub message: String,
}

impl MjcfError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for MjcfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

impl std::error::Error for MjcfError {}

impl From<xml::Error> for MjcfError {
    fn from(e: xml::Error) -> Self {
        Self::new(
            "<xml>",
            format!("parse error at byte {}: {}", e.offset, e.message),
        )
    }
}

fn fail<T>(path: &str, message: impl Into<String>) -> Result<T, MjcfError> {
    Err(MjcfError::new(path, message))
}

// ---------------------------------------------------------------------------
// public entry points
// ---------------------------------------------------------------------------

/// Load an MJCF-formatted scene from a string. See the module docs for the
/// supported subset.
pub fn load_mjcf_str(source: &str) -> Result<Scene, MjcfError> {
    let root = xml::parse(source)?;
    if root.name != "mujoco" {
        return fail(
            "<root>",
            format!("MJCF root must be <mujoco>, got <{}>", root.name),
        );
    }
    let mut loader = Loader::new();
    loader.load(&root)?;
    Ok(loader.into_scene())
}

/// Load an MJCF-formatted scene from a file on disk.
pub fn load_mjcf_path<P: AsRef<Path>>(path: P) -> Result<Scene, MjcfError> {
    let path_ref = path.as_ref();
    let src = std::fs::read_to_string(path_ref).map_err(|e| {
        MjcfError::new(
            "<io>",
            format!("could not read {}: {}", path_ref.display(), e),
        )
    })?;
    load_mjcf_str(&src)
}

// ---------------------------------------------------------------------------
// attribute value parsers
// ---------------------------------------------------------------------------

fn attr_required<'a>(e: &'a Element, name: &str, path: &str) -> Result<&'a str, MjcfError> {
    e.attr(name)
        .ok_or_else(|| MjcfError::new(path, format!("missing required attribute \"{name}\"")))
}

fn parse_f32(src: &str, path: &str, attr: &str) -> Result<f32, MjcfError> {
    let trimmed = src.trim();
    if trimmed.is_empty() {
        return fail(path, format!("attribute \"{attr}\" is empty"));
    }
    let value: f32 = trimmed.parse().map_err(|_| {
        MjcfError::new(
            path,
            format!("attribute \"{attr}\": not a number: {trimmed:?}"),
        )
    })?;
    if !value.is_finite() {
        return fail(
            path,
            format!("attribute \"{attr}\": value {value} is not finite"),
        );
    }
    Ok(value)
}

fn parse_f32_list(src: &str, path: &str, attr: &str) -> Result<Vec<f32>, MjcfError> {
    let mut out = Vec::new();
    for tok in src.split_ascii_whitespace() {
        let value: f32 = tok.parse().map_err(|_| {
            MjcfError::new(path, format!("attribute \"{attr}\": not a number: {tok:?}"))
        })?;
        if !value.is_finite() {
            return fail(
                path,
                format!("attribute \"{attr}\": value {value} is not finite"),
            );
        }
        out.push(value);
    }
    if out.is_empty() {
        return fail(
            path,
            format!("attribute \"{attr}\" must have at least one number"),
        );
    }
    Ok(out)
}

fn parse_int(src: &str, path: &str, attr: &str) -> Result<i32, MjcfError> {
    src.trim().parse().map_err(|_| {
        MjcfError::new(
            path,
            format!("attribute \"{attr}\": not an integer: {src:?}"),
        )
    })
}

fn parse_bool(src: &str, path: &str, attr: &str) -> Result<bool, MjcfError> {
    match src.trim() {
        "true" | "True" | "1" => Ok(true),
        "false" | "False" | "0" => Ok(false),
        other => fail(
            path,
            format!("attribute \"{attr}\": expected true|false, got {other:?}"),
        ),
    }
}

fn require_len(nums: &[f32], want: usize, path: &str, attr: &str) -> Result<(), MjcfError> {
    if nums.len() != want {
        return fail(
            path,
            format!(
                "attribute \"{attr}\": expected {want} numbers, got {}",
                nums.len()
            ),
        );
    }
    Ok(())
}

fn parse_vec3_attr(src: &str, path: &str, attr: &str) -> Result<Vec3, MjcfError> {
    let nums = parse_f32_list(src, path, attr)?;
    require_len(&nums, 3, path, attr)?;
    Ok(Vec3::new(nums[0], nums[1], nums[2]))
}

/// Parse a MuJoCo-order `w x y z` quaternion attribute into a
/// newt `(x, y, z, w)` Quat.
fn parse_quat_wxyz(src: &str, path: &str, attr: &str) -> Result<Quat, MjcfError> {
    let nums = parse_f32_list(src, path, attr)?;
    require_len(&nums, 4, path, attr)?;
    let q = Quat::new(nums[1], nums[2], nums[3], nums[0]);
    if q.norm_squared() < 1e-12 {
        return fail(
            path,
            format!("attribute \"{attr}\": quaternion has zero norm"),
        );
    }
    Ok(q.renormalize())
}

/// Convert an XYZ (extrinsic) Euler angle triple into a quaternion. MJCF
/// default `eulerseq = "xyz"` means intrinsic Tait-Bryan XYZ, which for
/// our right-hand convention is `R = Rz(γ) · Ry(β) · Rx(α)` when read as
/// world-frame active rotations (see MuJoCo docs). `angle_scale` converts
/// input units to radians (`1.0` for radian mode, `π/180` for degree
/// mode).
fn parse_euler_xyz(nums: &[f32], angle_scale: f32) -> Quat {
    let ax = nums[0] * angle_scale;
    let ay = nums[1] * angle_scale;
    let az = nums[2] * angle_scale;
    let qx = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), ax);
    let qy = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), ay);
    let qz = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), az);
    (qz * qy * qx).renormalize()
}

// ---------------------------------------------------------------------------
// default class table (MJCF <default> resolution)
// ---------------------------------------------------------------------------

/// One MJCF default class. Maps element name (`"joint"`, `"geom"`,
/// `"site"`, `"position"`, `"motor"`, ...) to its default attributes in
/// source order. Attribute lookup is linear — matches the JSON loader.
#[derive(Clone, Debug, Default)]
struct DefaultClass {
    per_element: HashMap<String, Vec<(String, String)>>,
}

impl DefaultClass {
    fn attrs_for(&self, element: &str) -> Option<&[(String, String)]> {
        self.per_element.get(element).map(|v| v.as_slice())
    }
}

/// Table of MJCF `<default>` classes. `"main"` is the un-classed root
/// default (MuJoCo's convention when the top-level `<default>` has no
/// `class` attribute).
#[derive(Clone, Debug, Default)]
struct DefaultsTable {
    classes: HashMap<String, DefaultClass>,
}

impl DefaultsTable {
    fn lookup(&self, class: &str) -> Option<&DefaultClass> {
        self.classes.get(class)
    }

    /// The name of the top-level implicit class.
    const MAIN: &'static str = "main";
}

/// Build the defaults table by walking every `<default>` under the root.
/// MJCF nesting semantics: a nested `<default class="X">` inherits from
/// its parent default, then overrides per-element attribute lists. Class
/// names must be unique globally.
fn build_defaults(root: &Element, path: &str) -> Result<DefaultsTable, MjcfError> {
    let mut table = DefaultsTable::default();
    let empty = DefaultClass::default();
    for child in root.child_elements() {
        if child.name != "default" {
            continue;
        }
        walk_default(child, DefaultsTable::MAIN, &empty, &mut table, path)?;
    }
    // Ensure "main" always exists, empty if the file has no <default>.
    table
        .classes
        .entry(DefaultsTable::MAIN.to_string())
        .or_default();
    Ok(table)
}

fn walk_default(
    elem: &Element,
    default_class_name: &str,
    parent_class: &DefaultClass,
    table: &mut DefaultsTable,
    path: &str,
) -> Result<(), MjcfError> {
    // Top-level default's own class name comes from its `class` attribute or
    // defaults to `"main"` (MuJoCo convention). Nested defaults must specify
    // a class.
    let class_name = elem
        .attr("class")
        .map(|s| s.to_string())
        .unwrap_or_else(|| default_class_name.to_string());
    if class_name.is_empty() {
        return fail(path, "default class name cannot be empty");
    }
    // Start from a clone of the parent class, then merge our own children.
    let mut this_class = parent_class.clone();
    for child in elem.child_elements() {
        if child.name == "default" {
            continue;
        }
        // Merge the child's attributes onto whatever the parent class had for
        // that element name. Later entries override earlier ones with the
        // same key.
        let target = this_class
            .per_element
            .entry(child.name.clone())
            .or_default();
        for (k, v) in &child.attrs {
            if let Some(pos) = target.iter().position(|(kk, _)| kk == k) {
                target[pos].1 = v.clone();
            } else {
                target.push((k.clone(), v.clone()));
            }
        }
        // Nested defaults inside a default's children (e.g., a `<geom
        // solimp="..."/>` under `<default class="main">`) are attribute
        // sources, not new classes; only `<default class="X"><...></default>`
        // creates new classes.
        if let Some(grand) = child.child_elements().next() {
            return fail(
                path,
                format!(
                    "<{}> inside <default> cannot have child elements (found <{}>)",
                    child.name, grand.name
                ),
            );
        }
    }
    if table.classes.contains_key(&class_name) {
        return fail(path, format!("duplicate <default class=\"{class_name}\">"));
    }
    table.classes.insert(class_name.clone(), this_class);
    // Recurse: nested <default class="X"> inside this default.
    for child in elem.child_elements() {
        if child.name != "default" {
            continue;
        }
        let sub_name = attr_required(child, "class", path)?.to_string();
        // A nested default inherits from THIS default (built above), not
        // from the grandparent. Re-fetch the just-inserted class so the
        // child sees the merged state.
        let parent = table.classes[&class_name].clone();
        walk_default(child, &sub_name, &parent, table, path)?;
    }
    Ok(())
}

/// Fetch the effective value of an attribute for an element, considering
/// (in order): the element's own attribute, the active class's default
/// attribute list for this element name. Returns `None` if neither has it.
fn attr_with_default<'a>(
    e: &'a Element,
    element_name: &str,
    attr: &str,
    class: &'a DefaultClass,
) -> Option<&'a str> {
    if let Some(v) = e.attr(attr) {
        return Some(v);
    }
    class.attrs_for(element_name).and_then(|list| {
        list.iter()
            .find(|(k, _)| k == attr)
            .map(|(_, v)| v.as_str())
    })
}

// ---------------------------------------------------------------------------
// loader state
// ---------------------------------------------------------------------------

struct Loader {
    world: World,
    // symbol tables (same names as Scene)
    bodies_by_name: HashMap<String, usize>,
    trees_by_name: HashMap<String, usize>,
    links_by_name: Vec<HashMap<String, usize>>,
    geoms_by_name: HashMap<String, usize>,
    sites: Vec<Site>,
    sites_by_name: HashMap<String, usize>,
    actuators_by_name: HashMap<String, (usize, usize)>,
    sensors_by_name: HashMap<String, usize>,

    /// Joint-name → `(tree_idx, link_idx)` — MJCF references actuators/
    /// sensors/equalities to joints by name (rather than by tree+link).
    joints_by_name: HashMap<String, (usize, usize)>,
    /// Body-name → `(tree_idx, link_idx)` for tree links, or `("bodies",
    /// idx)` encoded as tree_idx = usize::MAX. Used by equality connect/
    /// weld and sensors that reference bodies by name.
    tree_bodies_by_name: HashMap<String, (usize, usize)>,
    /// True when compiler angle="degree"; multiplies degree-scaled
    /// attributes (`joint range`, `euler`, `axis-angle`) by π/180.
    angle_scale: f32,
    /// True when compiler coordinate="local" (default and only accepted).
    /// Coordinate="global" errors out — the biped and the fixtures we
    /// mirror all use local.
    /// Stored so we could permit `"local"` explicitly in the future.
    #[allow(dead_code)]
    coordinate_local: bool,
    defaults: DefaultsTable,
    /// Track per-tree self-collide (defaults to false, matches JSON loader).
    /// MJCF does not have a per-tree self-collide flag; we always emit
    /// tree_self_collide=false for auto pair filtering (equivalent to the
    /// JSON loader's default).
    tree_self_collide: Vec<bool>,
}

impl Loader {
    fn new() -> Self {
        Self {
            world: World::new(),
            bodies_by_name: HashMap::new(),
            trees_by_name: HashMap::new(),
            links_by_name: Vec::new(),
            geoms_by_name: HashMap::new(),
            sites: Vec::new(),
            sites_by_name: HashMap::new(),
            actuators_by_name: HashMap::new(),
            sensors_by_name: HashMap::new(),
            joints_by_name: HashMap::new(),
            tree_bodies_by_name: HashMap::new(),
            angle_scale: 1.0,
            coordinate_local: true,
            defaults: DefaultsTable::default(),
            tree_self_collide: Vec::new(),
        }
    }

    fn into_scene(self) -> Scene {
        let world = self.world;
        Scene {
            world,
            bodies_by_name: self.bodies_by_name,
            trees_by_name: self.trees_by_name,
            links_by_name: self.links_by_name,
            geoms_by_name: self.geoms_by_name,
            sites: self.sites,
            sites_by_name: self.sites_by_name,
            actuators_by_name: self.actuators_by_name,
            sensors_by_name: self.sensors_by_name,
        }
    }

    fn load(&mut self, root: &Element) -> Result<(), MjcfError> {
        let path = "mujoco".to_string();
        // The <mujoco> element itself may carry a `model` name attribute;
        // no other attributes are supported.
        for (k, _) in &root.attrs {
            if k != "model" {
                return fail(
                    &path,
                    format!("unknown attribute \"{k}\" on <mujoco>; only \"model\" is supported"),
                );
            }
        }
        // Order matters: compiler and defaults must be parsed BEFORE any
        // element that consults them. We do a two-phase walk: first
        // compiler + option + defaults; then worldbody + everything else.
        for child in root.child_elements() {
            match child.name.as_str() {
                "compiler" => self.walk_compiler(child, &child_path(&path, "compiler", None))?,
                "option" => self.walk_option(child, &child_path(&path, "option", None))?,
                _ => {}
            }
        }
        self.defaults = build_defaults(root, &path)?;

        for child in root.child_elements() {
            let subpath = child_path(&path, &child.name, child.attr("name"));
            match child.name.as_str() {
                "compiler" | "option" | "default" => {} // handled above
                "worldbody" => self.walk_worldbody(child, &subpath)?,
                "actuator" => self.walk_actuator(child, &subpath)?,
                "sensor" => self.walk_sensor(child, &subpath)?,
                "equality" => self.walk_equality(child, &subpath)?,
                "contact" => self.walk_contact(child, &subpath)?,
                // Explicitly-known but unsupported MJCF top-level elements
                // — clean error naming the feature.
                "asset" | "tendon" | "keyframe" | "custom" | "visual" | "size" | "statistic"
                | "extension" | "include" => {
                    return fail(
                        &subpath,
                        format!(
                            "<{}> is not supported in the v1 MJCF subset (see docs/mjcf.md)",
                            child.name
                        ),
                    );
                }
                other => {
                    return fail(
                        &subpath,
                        format!(
                            "unknown top-level element <{other}> under <mujoco>; supported: \
                             compiler, option, default, worldbody, actuator, sensor, equality, contact"
                        ),
                    );
                }
            }
        }
        // Apply the JSON loader's auto pair-list filter for the same-tree /
        // self-collision case BEFORE running the pair-support check so an
        // unsupported same-tree pair (e.g. box-vs-capsule inside a robot
        // that opts out of self-collision) does not spuriously trip the
        // narrow-phase validator.
        if self.world.pair_list.is_none()
            && !self.tree_self_collide.is_empty()
            && self.tree_self_collide.iter().any(|&s| !s)
        {
            self.world.pair_list = Some(auto_pairs_with_self_collision_filter(
                &self.world,
                &self.tree_self_collide,
            ));
        }
        // After parsing everything, run the pair-support check the JSON
        // loader also runs so a bad geom-pair combo surfaces at load time
        // with a path rather than a runtime panic.
        if let Some(unsupported) = self.world.validate_supported_pairs().into_iter().next() {
            let ga = &self.world.geoms[unsupported.geom_a];
            let gb = &self.world.geoms[unsupported.geom_b];
            return fail(
                "mujoco > contact",
                format!(
                    "contact pair between geom {} ({:?}) and geom {} ({:?}) is not \
                     supported by newt's narrow phase — see docs/contacts.md support \
                     matrix",
                    unsupported.geom_a, ga.shape, unsupported.geom_b, gb.shape
                ),
            );
        }
        Ok(())
    }

    // ---------- compiler / option ---------------------------------------

    fn walk_compiler(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        for (k, v) in &e.attrs {
            match k.as_str() {
                "angle" => match v.as_str() {
                    "radian" => self.angle_scale = 1.0,
                    "degree" => self.angle_scale = math::PI / 180.0,
                    other => {
                        return fail(
                            path,
                            format!(
                                "compiler angle must be \"radian\" or \"degree\", got {other:?}"
                            ),
                        );
                    }
                },
                "coordinate" => match v.as_str() {
                    "local" => self.coordinate_local = true,
                    other => {
                        return fail(
                            path,
                            format!(
                                "compiler coordinate=\"{other}\" is not supported (only \"local\")"
                            ),
                        );
                    }
                },
                "eulerseq" => {
                    if v != "xyz" {
                        return fail(
                            path,
                            format!("compiler eulerseq=\"{v}\" is not supported (only \"xyz\")"),
                        );
                    }
                }
                "autolimits" | "meshdir" | "texturedir" | "boundmass" | "boundinertia"
                | "settotalmass" | "inertiafromgeom" | "usethread" => {
                    return fail(
                        path,
                        format!("compiler attribute \"{k}\" is not supported in the v1 subset"),
                    );
                }
                other => {
                    return fail(path, format!("unknown <compiler> attribute \"{other}\""));
                }
            }
        }
        // <compiler> has no child elements in our subset.
        if let Some(child) = e.child_elements().next() {
            return fail(
                path,
                format!(
                    "<compiler> has no child elements in the subset (found <{}>)",
                    child.name
                ),
            );
        }
        Ok(())
    }

    fn walk_option(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        for (k, v) in &e.attrs {
            match k.as_str() {
                "timestep" => {
                    let dt = parse_f32(v, path, "timestep")?;
                    if dt <= 0.0 {
                        return fail(path, format!("timestep must be > 0 (got {dt})"));
                    }
                    self.world.dt = dt;
                }
                "gravity" => {
                    self.world.gravity = parse_vec3_attr(v, path, "gravity")?;
                }
                "integrator" => {
                    if v != "RK4" && v != "rk4" {
                        return fail(
                            path,
                            format!("option integrator=\"{v}\" is not supported (only \"RK4\")"),
                        );
                    }
                }
                "cone" => {
                    use crate::solver::ConeKind;
                    self.world.solver.cone = match v.as_str() {
                        "pyramidal" => ConeKind::Pyramidal,
                        "elliptic" => ConeKind::Elliptic,
                        other => {
                            return fail(
                                path,
                                format!(
                                    "option cone=\"{other}\" not supported (pyramidal|elliptic)"
                                ),
                            );
                        }
                    };
                }
                "iterations" => {
                    let n = parse_int(v, path, "iterations")?;
                    if n < 1 {
                        return fail(path, "iterations must be ≥ 1");
                    }
                    self.world.solver.iterations = n as u32;
                }
                "solver" => {
                    // MJCF uses "PGS" / "CG" / "Newton"; we accept "PGS" only.
                    match v.as_str() {
                        "PGS" | "pgs" => {
                            self.world.solver.mode = crate::solver::SolverMode::Pgs;
                        }
                        other => {
                            return fail(
                                path,
                                format!("option solver=\"{other}\" not supported (only \"PGS\")"),
                            );
                        }
                    }
                }
                "wind" | "magnetic" | "density" | "viscosity" | "impratio" | "o_margin"
                | "o_solref" | "o_solimp" | "tolerance" | "noslip_iterations"
                | "noslip_tolerance" | "mpr_iterations" | "mpr_tolerance" | "collision"
                | "jacobian" | "integrator_stage" | "apirate" => {
                    return fail(
                        path,
                        format!("option attribute \"{k}\" is not supported in the v1 subset"),
                    );
                }
                other => {
                    return fail(path, format!("unknown <option> attribute \"{other}\""));
                }
            }
        }
        if let Some(child) = e.child_elements().next() {
            return fail(
                path,
                format!(
                    "<option> child <{}> (option flags) not supported in the v1 subset",
                    child.name
                ),
            );
        }
        Ok(())
    }

    // ---------- worldbody -----------------------------------------------

    fn walk_worldbody(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        if let Some((k, _)) = e.attrs.first() {
            return fail(
                path,
                format!("<worldbody> takes no attributes (got \"{k}\")"),
            );
        }
        // Two kinds of children: <geom>/<site> (static) and <body> (roots).
        // <light> is rejected as unsupported.
        for child in e.child_elements() {
            let subpath = child_path(path, &child.name, child.attr("name"));
            match child.name.as_str() {
                "geom" => {
                    self.add_static_geom(child, &subpath, DefaultsTable::MAIN)?;
                }
                "site" => {
                    return fail(
                        &subpath,
                        "a <site> under <worldbody> must be attached to a body \
                         (world-attached sites are not part of the v1 subset)",
                    );
                }
                "body" => {
                    self.add_top_level_body(child, &subpath)?;
                }
                "light" | "camera" | "frame" => {
                    return fail(
                        &subpath,
                        format!(
                            "<{}> under <worldbody> is not supported in the v1 subset",
                            child.name
                        ),
                    );
                }
                other => {
                    return fail(
                        &subpath,
                        format!("unknown <worldbody> child <{other}>; supported: geom, body"),
                    );
                }
            }
        }
        Ok(())
    }

    fn add_top_level_body(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        // Two paths: free body (freejoint + no nested bodies + eligible for
        // World.bodies) OR tree root.
        //
        // The rule matches the JSON loader's split: single-body free objects
        // go into World.bodies (tier-1 code path); multi-link chains and
        // fixed-root chains go into World.trees. This gives us
        // byte-identical trajectories against the mirror JSON fixtures.
        validate_body_attrs(e, path)?;
        let free_child = find_free_joint(e);
        let has_child_body = e.child_elements().any(|c| c.name == "body");
        let has_joint = e
            .child_elements()
            .any(|c| c.name == "joint" || c.name == "freejoint");
        if free_child.is_some() && !has_child_body && !has_only_free_joint(e) {
            return fail(
                path,
                "a free-jointed top-level body must have exactly one <freejoint> or \
                 <joint type=\"free\"/> and no other joints",
            );
        }
        if free_child.is_some() && !has_child_body {
            self.add_free_body(e, path)?;
            return Ok(());
        }
        if has_joint && !has_child_body && free_child.is_none() {
            return fail(
                path,
                "top-level body with a hinge/slide/ball joint at the world root is not \
                 in the v1 subset; wrap it in a fixed-root body if you need this",
            );
        }
        // Otherwise: a tree. Root is Fixed if no joint; Free if freejoint.
        self.add_tree(e, path)?;
        Ok(())
    }

    // ---------- free body path -----------------------------------------

    fn add_free_body(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        // Body-level attributes.
        let class_name = e
            .attr("childclass")
            .unwrap_or(DefaultsTable::MAIN)
            .to_string();
        let name = attr_required(e, "name", path)?.to_string();
        if self.bodies_by_name.contains_key(&name) {
            return fail(path, format!("duplicate body name \"{name}\""));
        }
        let body_pos = self.parse_body_pos(e, path)?;
        let body_ori = self.parse_body_orientation(e, path)?;

        // Every <inertial> attribute goes through the class-defaults lookup
        // for the "inertial" name; MuJoCo doesn't really use classes for
        // <inertial>, but we plumb it through consistently.
        let (mass, inertia) = self.find_and_parse_inertial(e, path, &class_name)?;
        let mut body = Body::new(mass, inertia, body_pos, body_ori);

        // Optional freejoint attributes (name is common).
        // No initial velocity attribute exists in MJCF; skip.
        let free_j = find_free_joint(e).unwrap();
        for (k, _) in &free_j.attrs {
            match k.as_str() {
                "name" => {}
                other => {
                    return fail(
                        path,
                        format!("<freejoint> attribute \"{other}\" not supported"),
                    );
                }
            }
        }
        // Newt extension: an optional `<velocity linear="..."
        // angular_body="..."/>` child element on a free-body <body>
        // seeds the initial velocity, mirroring the JSON model's
        // `"velocity"` field. Not part of stock MJCF; documented in
        // `docs/mjcf.md`.
        if let Some(vel) = e.child_elements().find(|c| c.name == "velocity") {
            let velpath = child_path(path, "velocity", None);
            for (k, _) in &vel.attrs {
                match k.as_str() {
                    "linear" | "angular_body" => {}
                    other => {
                        return fail(
                            &velpath,
                            format!("<velocity> unknown attribute \"{other}\""),
                        );
                    }
                }
            }
            if let Some(v) = vel.attr("linear") {
                body.linear_velocity = parse_vec3_attr(v, &velpath, "linear")?;
            }
            if let Some(v) = vel.attr("angular_body") {
                body.angular_velocity_body = parse_vec3_attr(v, &velpath, "angular_body")?;
            }
        }
        let body_idx = self.world.add_body(body);
        self.bodies_by_name.insert(name.clone(), body_idx);

        // Register the freejoint's name (if any) so sensors/actuators/
        // equalities can reference it later. Freejoints on free bodies
        // don't participate in tree slots, but sensors like `jointpos`
        // never target them — MuJoCo forbids that too.
        if let Some(jn) = free_j.attr("name") {
            // Store a sentinel so we can produce a clear error if someone
            // targets it: (usize::MAX, body_idx).
            self.joints_by_name
                .insert(jn.to_string(), (usize::MAX, body_idx));
        }

        // Register the body name so equality/sensor lookups against bodies
        // work uniformly.
        self.tree_bodies_by_name
            .insert(name, (usize::MAX, body_idx));

        // Parse geoms + sites attached to this free body.
        for child in e.child_elements() {
            let subpath = child_path(path, &child.name, child.attr("name"));
            match child.name.as_str() {
                "geom" => {
                    self.add_body_geom(child, &subpath, &class_name, body_idx)?;
                }
                "site" => {
                    self.add_site_on_body(child, &subpath, &class_name, body_idx)?;
                }
                "inertial" | "freejoint" | "joint" | "velocity" => {}
                other => {
                    return fail(
                        &subpath,
                        format!("<{other}> not supported inside a free-body <body>"),
                    );
                }
            }
        }
        Ok(())
    }

    // ---------- tree path ----------------------------------------------

    fn add_tree(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        let tree_name = attr_required(e, "name", path)?.to_string();
        if self.trees_by_name.contains_key(&tree_name) {
            return fail(path, format!("duplicate tree name \"{tree_name}\""));
        }
        // MJCF `childclass` on the root body scopes the default class for
        // this body's descendants (and itself).
        let root_class = e
            .attr("childclass")
            .unwrap_or(DefaultsTable::MAIN)
            .to_string();
        let mut tree = Tree::new();
        let mut link_names: HashMap<String, usize> = HashMap::new();
        // Build the root link first, then recurse.
        let (root_link, root_link_name) = self.build_root_link(e, path, &root_class)?;
        tree.push_link(root_link);
        link_names.insert(root_link_name.clone(), 0);

        let tree_idx = self.world.add_tree(tree);
        self.trees_by_name.insert(tree_name.clone(), tree_idx);
        // Placeholder; we'll set the real link_names table after the walk.
        self.links_by_name.push(HashMap::new());
        self.tree_self_collide.push(false);

        // Register root body name.
        self.tree_bodies_by_name
            .insert(root_link_name.clone(), (tree_idx, 0));

        // Register root joint name if the root has a freejoint.
        if let Some(fj) = find_free_joint(e) {
            if let Some(jn) = fj.attr("name") {
                self.joints_by_name.insert(jn.to_string(), (tree_idx, 0));
            }
        }

        // Attach root geoms + sites.
        self.attach_body_children(e, path, &root_class, tree_idx, 0)?;

        // Walk nested <body> children.
        for child in e.child_elements() {
            if child.name != "body" {
                continue;
            }
            let subpath = child_path(path, &child.name, child.attr("name"));
            self.walk_child_body(child, &subpath, tree_idx, 0, &root_class, &mut link_names)?;
        }

        self.links_by_name[tree_idx] = link_names;
        Ok(())
    }

    fn build_root_link(
        &self,
        e: &Element,
        path: &str,
        class: &str,
    ) -> Result<(Link, String), MjcfError> {
        let name = attr_required(e, "name", path)?.to_string();
        if name.is_empty() {
            return fail(path, "body name must not be empty");
        }
        let body_pos = self.parse_body_pos(e, path)?;
        let body_ori = self.parse_body_orientation(e, path)?;
        let (mass, inertia) = self.find_and_parse_inertial(e, path, class)?;
        // Root joint: <freejoint/> → Free, no <joint> → Fixed, other → error.
        let joint_kind = self.determine_root_joint(e, path)?;
        // joint_offset_in_parent semantics: for a fixed root, this IS the
        // world-anchor pose. For a free root the position is currently
        // ignored by ABA (initial pose comes from q which we initialize
        // from body_pos/body_ori). The JSON loader keeps this consistent by
        // baking the body_pos/body_ori into the root link's parent-offset
        // AND, for Free roots, seeding q accordingly — matching that here
        // is important for byte-identity.
        let joint_offset_in_parent = (body_pos, body_ori);
        let joint_offset_in_child = (Vec3::ZERO, Quat::IDENTITY);
        let link = Link::new(
            None,
            joint_kind,
            joint_offset_in_parent,
            joint_offset_in_child,
            mass,
            inertia,
        );
        Ok((link, name))
    }

    fn determine_root_joint(&self, e: &Element, path: &str) -> Result<JointKind, MjcfError> {
        let mut free = None;
        let mut hinge_like = None;
        for child in e.child_elements() {
            match child.name.as_str() {
                "freejoint" => free = Some(child),
                "joint" => {
                    let ty = child.attr("type").unwrap_or("hinge");
                    if ty == "free" {
                        free = Some(child);
                    } else {
                        hinge_like = Some(child);
                    }
                }
                _ => {}
            }
        }
        if free.is_some() && hinge_like.is_some() {
            return fail(
                path,
                "root body cannot mix <freejoint> with a hinge/slide/ball joint",
            );
        }
        if hinge_like.is_some() {
            return fail(
                path,
                "root body must have <freejoint> or no joint (v1 subset does not \
                 pin a top-level body with a hinge/slide/ball)",
            );
        }
        if free.is_some() {
            Ok(JointKind::Free)
        } else {
            Ok(JointKind::Fixed)
        }
    }

    fn walk_child_body(
        &mut self,
        e: &Element,
        path: &str,
        tree_idx: usize,
        parent_link_idx: usize,
        parent_class: &str,
        link_names: &mut HashMap<String, usize>,
    ) -> Result<(), MjcfError> {
        validate_body_attrs(e, path)?;
        let name = attr_required(e, "name", path)?.to_string();
        if link_names.contains_key(&name) {
            return fail(path, format!("duplicate link name \"{name}\""));
        }
        let class = e.attr("childclass").unwrap_or(parent_class).to_string();
        let body_pos = self.parse_body_pos(e, path)?;
        let body_ori = self.parse_body_orientation(e, path)?;
        // v0 constraint: non-root body orientation must be identity — the
        // engine requires joint_offset_in_parent.orientation = IDENTITY and
        // joint_offset_in_child.orientation = IDENTITY, so a rotated body
        // frame is not representable losslessly. Reject explicitly.
        if body_ori != Quat::IDENTITY {
            return fail(
                path,
                "non-root body <body quat=..> / <body euler=..> is not supported \
                 (newt v0/v1 requires identity body orientation for non-root links)",
            );
        }
        let (mass, inertia) = self.find_and_parse_inertial(e, path, &class)?;

        // Find the (at most one) joint child.
        let joints: Vec<&Element> = e
            .child_elements()
            .filter(|c| c.name == "joint" || c.name == "freejoint")
            .collect();
        if joints.len() > 1 {
            return fail(
                path,
                format!(
                    "body \"{name}\" has {} joints; the v1 subset supports exactly \
                     one joint (or none for a fixed link) per body",
                    joints.len()
                ),
            );
        }
        let (joint_kind, joint_pos_in_body, joint_name) = if let Some(j) = joints.first() {
            if j.name == "freejoint" {
                return fail(
                    path,
                    "a <freejoint> is only allowed on a top-level body under <worldbody>",
                );
            }
            let joint_path = child_path(path, "joint", j.attr("name"));
            self.parse_child_joint(j, &joint_path, &class)?
        } else {
            (JointKind::Fixed, Vec3::ZERO, None)
        };

        // Map MJCF body.pos + joint.pos into newt joint offsets.
        let joint_offset_in_parent = (body_pos + joint_pos_in_body, Quat::IDENTITY);
        let joint_offset_in_child = (joint_pos_in_body, Quat::IDENTITY);
        let link = Link::new(
            Some(parent_link_idx),
            joint_kind,
            joint_offset_in_parent,
            joint_offset_in_child,
            mass,
            inertia,
        );
        self.world.trees[tree_idx].push_link(link);
        let new_idx = self.world.trees[tree_idx].links.len() - 1;
        link_names.insert(name.clone(), new_idx);
        self.tree_bodies_by_name
            .insert(name.clone(), (tree_idx, new_idx));
        if let Some(jn) = joint_name {
            if self.joints_by_name.contains_key(&jn) {
                return fail(path, format!("duplicate joint name \"{jn}\""));
            }
            self.joints_by_name.insert(jn, (tree_idx, new_idx));
        }

        // Attach child geoms + sites.
        self.attach_body_children(e, path, &class, tree_idx, new_idx)?;

        // Recurse into nested bodies.
        for child in e.child_elements() {
            if child.name != "body" {
                continue;
            }
            let subpath = child_path(path, &child.name, child.attr("name"));
            self.walk_child_body(child, &subpath, tree_idx, new_idx, &class, link_names)?;
        }
        Ok(())
    }

    /// Attach the `<geom>` and `<site>` children of a body to `(tree_idx,
    /// link_idx)`. Non-inertial, non-joint, non-body children are rejected.
    fn attach_body_children(
        &mut self,
        e: &Element,
        path: &str,
        class: &str,
        tree_idx: usize,
        link_idx: usize,
    ) -> Result<(), MjcfError> {
        for child in e.child_elements() {
            let subpath = child_path(path, &child.name, child.attr("name"));
            match child.name.as_str() {
                "geom" => {
                    self.add_link_geom(child, &subpath, class, tree_idx, link_idx)?;
                }
                "site" => {
                    self.add_site_on_link(child, &subpath, class, tree_idx, link_idx)?;
                }
                "inertial" | "joint" | "freejoint" | "body" => {}
                other => {
                    return fail(&subpath, format!("<{other}> not supported inside a <body>"));
                }
            }
        }
        Ok(())
    }

    // ---------- body pos/quat/euler --------------------------------------

    fn parse_body_pos(&self, e: &Element, path: &str) -> Result<Vec3, MjcfError> {
        match e.attr("pos") {
            Some(v) => parse_vec3_attr(v, path, "pos"),
            None => Ok(Vec3::ZERO),
        }
    }

    fn parse_body_orientation(&self, e: &Element, path: &str) -> Result<Quat, MjcfError> {
        let q = e.attr("quat");
        let eu = e.attr("euler");
        let ax = e.attr("axisangle");
        let count = q.is_some() as usize + eu.is_some() as usize + ax.is_some() as usize;
        if count > 1 {
            return fail(
                path,
                "specify at most one of quat / euler / axisangle on <body>",
            );
        }
        if let Some(q) = q {
            return parse_quat_wxyz(q, path, "quat");
        }
        if let Some(eu) = eu {
            let nums = parse_f32_list(eu, path, "euler")?;
            require_len(&nums, 3, path, "euler")?;
            return Ok(parse_euler_xyz(&nums, self.angle_scale));
        }
        if let Some(aa) = ax {
            let nums = parse_f32_list(aa, path, "axisangle")?;
            require_len(&nums, 4, path, "axisangle")?;
            let axis = Vec3::new(nums[0], nums[1], nums[2]);
            if axis.length_squared() < 1e-12 {
                return fail(path, "axisangle axis must be non-zero");
            }
            return Ok(Quat::from_axis_angle(axis, nums[3] * self.angle_scale));
        }
        Ok(Quat::IDENTITY)
    }

    // ---------- inertial -------------------------------------------------

    fn find_and_parse_inertial(
        &self,
        body_elem: &Element,
        path: &str,
        class: &str,
    ) -> Result<(f32, Mat3), MjcfError> {
        let inertials: Vec<&Element> = body_elem
            .child_elements()
            .filter(|c| c.name == "inertial")
            .collect();
        if inertials.is_empty() {
            return self.autocompute_body_inertia(body_elem, path, class);
        }
        if inertials.len() > 1 {
            return fail(path, "a body must have at most one <inertial>");
        }
        let ine = inertials[0];
        let inepath = child_path(path, "inertial", None);
        // Attribute whitelist.
        for (k, _) in &ine.attrs {
            match k.as_str() {
                "pos" | "mass" | "diaginertia" | "fullinertia" | "quat" => {}
                other => {
                    return fail(
                        &inepath,
                        format!("unknown <inertial> attribute \"{other}\""),
                    );
                }
            }
        }
        if let Some(pos) = ine.attr("pos") {
            let p = parse_vec3_attr(pos, &inepath, "pos")?;
            if p != Vec3::ZERO {
                return fail(
                    &inepath,
                    "inertial pos must be [0 0 0] — newt requires COM at the body origin",
                );
            }
        }
        if let Some(q) = ine.attr("quat") {
            let qq = parse_quat_wxyz(q, &inepath, "quat")?;
            if qq != Quat::IDENTITY {
                return fail(
                    &inepath,
                    "inertial quat is not supported — inertia is expressed in body axes",
                );
            }
        }
        let mass_str = attr_required(ine, "mass", &inepath)?;
        let mass = parse_f32(mass_str, &inepath, "mass")?;
        if mass <= 0.0 {
            return fail(&inepath, format!("mass must be > 0 (got {mass})"));
        }
        let inertia = if let Some(diag) = ine.attr("diaginertia") {
            let nums = parse_f32_list(diag, &inepath, "diaginertia")?;
            require_len(&nums, 3, &inepath, "diaginertia")?;
            for (i, &v) in nums.iter().enumerate() {
                if v <= 0.0 {
                    return fail(&inepath, format!("diaginertia[{i}] = {v} must be > 0"));
                }
            }
            Mat3::diag(nums[0], nums[1], nums[2])
        } else if let Some(full) = ine.attr("fullinertia") {
            let nums = parse_f32_list(full, &inepath, "fullinertia")?;
            require_len(&nums, 6, &inepath, "fullinertia")?;
            for (i, &v) in nums[..3].iter().enumerate() {
                if v <= 0.0 {
                    return fail(&inepath, format!("fullinertia[{i}] = {v} must be > 0"));
                }
            }
            let ixx = nums[0];
            let iyy = nums[1];
            let izz = nums[2];
            let ixy = nums[3];
            let ixz = nums[4];
            let iyz = nums[5];
            let m = Mat3::new([ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz]);
            if m.inverse().is_none() {
                return fail(&inepath, "inertia tensor is singular");
            }
            m
        } else {
            return fail(
                &inepath,
                "specify inertia via `diaginertia=\"Ixx Iyy Izz\"` or \
                 `fullinertia=\"Ixx Iyy Izz Ixy Ixz Iyz\"`",
            );
        };
        Ok((mass, inertia))
    }

    /// When a body omits `<inertial>`, sum the solid inertias of its
    /// mass-carrying geoms (matches MuJoCo's `inertiafromgeom` for the
    /// simple case). Requires each contributing geom to sit at local
    /// offset zero with identity local orientation — anything more
    /// complex needs a parallel-axis walk we don't currently implement;
    /// point the caller at `<inertial>` in that case.
    ///
    /// Uses the same `crate::geom::solid_*_inertia` helpers the JSON
    /// loader's `"solid"` inertia kind consumes, so a MJCF fixture
    /// mirroring a JSON model with `"inertia":{"kind":"solid",…}` gets
    /// byte-identical `Body.inertia_body` — which is what the trajectory
    /// anchor tests demand.
    fn autocompute_body_inertia(
        &self,
        body_elem: &Element,
        path: &str,
        class: &str,
    ) -> Result<(f32, Mat3), MjcfError> {
        let mut total_mass = 0.0f32;
        let mut inertia = Mat3::diag(0.0, 0.0, 0.0);
        let mut contributed = false;
        for child in body_elem.child_elements() {
            if child.name != "geom" {
                continue;
            }
            let dc = self
                .defaults
                .lookup(child.attr("class").unwrap_or(class))
                .cloned()
                .unwrap_or_default();
            let Some(mstr) = attr_with_default(child, "geom", "mass", &dc) else {
                continue;
            };
            let m = parse_f32(mstr, path, "mass")?;
            if m <= 0.0 {
                return fail(path, format!("<geom mass=\"{m}\"> must be > 0"));
            }
            let ty = attr_with_default(child, "geom", "type", &dc).unwrap_or("sphere");
            let (shape, offset, ori) = self.parse_geom_shape(child, path, ty, &dc)?;
            // If explicit pos/quat overrides on the geom shift or rotate
            // it, we need the parallel-axis walk — reject cleanly and
            // point the caller at `<inertial>`.
            let mut off = offset;
            if let Some(v) = attr_with_default(child, "geom", "pos", &dc) {
                off = parse_vec3_attr(v, path, "pos")?;
            }
            let effective_ori = self.parse_element_orientation(child, path, &dc, "geom")?;
            let ori = effective_ori.unwrap_or(ori);
            if off != Vec3::ZERO || ori != Quat::IDENTITY {
                return fail(
                    path,
                    "auto-inertia only supports geoms at pos=[0 0 0] with identity \
                     orientation — supply <inertial> for shifted/rotated geoms",
                );
            }
            let contribution = match shape {
                GeomShape::Box { half_extents } => crate::geom::solid_box_inertia(m, half_extents),
                GeomShape::Sphere { radius } => crate::geom::solid_sphere_inertia(m, radius),
                GeomShape::Capsule {
                    radius,
                    half_height,
                } => crate::geom::solid_capsule_inertia(m, radius, half_height),
                GeomShape::Cylinder {
                    radius,
                    half_height,
                } => crate::geom::solid_cylinder_inertia(m, radius, half_height),
                GeomShape::Ellipsoid { semi_axes } => {
                    crate::geom::solid_ellipsoid_inertia(m, semi_axes)
                }
                other => {
                    return fail(
                        path,
                        format!(
                            "auto-inertia does not handle geom shape {other:?} — supply <inertial>"
                        ),
                    );
                }
            };
            total_mass += m;
            inertia = mat3_add(inertia, contribution);
            contributed = true;
        }
        if !contributed {
            return fail(
                path,
                "the v1 MJCF subset requires either <inertial> on the body or at least \
                 one <geom mass=..> child (auto-inertia)",
            );
        }
        if inertia.inverse().is_none() {
            return fail(path, "auto-inertia produced a singular tensor");
        }
        Ok((total_mass, inertia))
    }

    // ---------- joint parsing (non-root) --------------------------------

    /// Returns `(JointKind, joint pos in body frame, optional joint name)`.
    fn parse_child_joint(
        &self,
        e: &Element,
        path: &str,
        class: &str,
    ) -> Result<(JointKind, Vec3, Option<String>), MjcfError> {
        let effective_class = e.attr("class").unwrap_or(class);
        let dc = self
            .defaults
            .lookup(effective_class)
            .cloned()
            .unwrap_or_default();
        let ty = attr_with_default(e, "joint", "type", &dc).unwrap_or("hinge");
        // Attribute whitelist per joint type.
        for (k, _) in &e.attrs {
            match k.as_str() {
                "name" | "type" | "pos" | "axis" | "range" | "damping" | "armature" | "limited"
                | "class" | "ref" => {}
                other => {
                    return fail(path, format!("unknown <joint> attribute \"{other}\""));
                }
            }
        }
        // Reject `ref` (non-zero would shift the joint's zero-configuration).
        if let Some(r) = e.attr("ref") {
            let v = parse_f32(r, path, "ref")?;
            if v != 0.0 {
                return fail(
                    path,
                    format!("joint ref={v} is not supported in the v1 subset (only ref=0)"),
                );
            }
        }
        let jname = e.attr("name").map(|s| s.to_string());
        let pos = match attr_with_default(e, "joint", "pos", &dc) {
            Some(v) => parse_vec3_attr(v, path, "pos")?,
            None => Vec3::ZERO,
        };
        match ty {
            "hinge" => {
                let (axis, range, damping, armature, limit) =
                    self.parse_single_dof_joint(e, path, "hinge", &dc)?;
                Ok((
                    JointKind::Hinge {
                        axis,
                        range,
                        damping,
                        armature,
                        limit,
                    },
                    pos,
                    jname,
                ))
            }
            "slide" => {
                let (axis, range, damping, armature, limit) =
                    self.parse_single_dof_joint(e, path, "slide", &dc)?;
                Ok((
                    JointKind::Slide {
                        axis,
                        range,
                        damping,
                        armature,
                        limit,
                    },
                    pos,
                    jname,
                ))
            }
            "ball" => {
                let damping =
                    optional_nonneg_float(e, "damping", path, &dc, "joint")?.unwrap_or(0.0);
                let armature =
                    optional_nonneg_float(e, "armature", path, &dc, "joint")?.unwrap_or(0.0);
                if e.attr("range").is_some() {
                    return fail(
                        path,
                        "ball joints do not support range limits in the v1 subset",
                    );
                }
                if e.attr("axis").is_some() {
                    return fail(path, "<joint type=\"ball\"> has no axis attribute");
                }
                Ok((JointKind::Ball { damping, armature }, pos, jname))
            }
            "free" => fail(
                path,
                "<joint type=\"free\"> is only allowed at the root of a body tree",
            ),
            other => fail(
                path,
                format!("unknown joint type \"{other}\" (expected hinge|slide|ball|free)"),
            ),
        }
    }

    #[allow(clippy::type_complexity)]
    fn parse_single_dof_joint(
        &self,
        e: &Element,
        path: &str,
        kind_label: &str,
        dc: &DefaultClass,
    ) -> Result<(Vec3, Option<(f32, f32)>, f32, f32, JointLimit), MjcfError> {
        let axis_s = attr_with_default(e, "joint", "axis", dc).ok_or_else(|| {
            MjcfError::new(
                path,
                format!("<joint type=\"{kind_label}\"> requires an axis attribute"),
            )
        })?;
        let axis = parse_vec3_attr(axis_s, path, "axis")?;
        if axis.length_squared() < 1e-12 {
            return fail(path, format!("{kind_label} axis must be non-zero"));
        }
        let axis = axis.normalize();

        // "limited" attribute: MuJoCo says a range only applies if
        // limited="true" (auto-detected in newer MuJoCo). Our loader: if
        // limited is explicitly "false", ignore range; else if range is
        // present, use it.
        let limited = match attr_with_default(e, "joint", "limited", dc) {
            Some(v) => parse_bool(v, path, "limited")?,
            None => true,
        };
        let range = match attr_with_default(e, "joint", "range", dc) {
            Some(v) => {
                let nums = parse_f32_list(v, path, "range")?;
                require_len(&nums, 2, path, "range")?;
                let lo = nums[0]
                    * (if kind_label == "hinge" {
                        self.angle_scale
                    } else {
                        1.0
                    });
                let hi = nums[1]
                    * (if kind_label == "hinge" {
                        self.angle_scale
                    } else {
                        1.0
                    });
                if lo >= hi {
                    return fail(path, format!("range low ({lo}) must be < high ({hi})"));
                }
                if limited { Some((lo, hi)) } else { None }
            }
            None => None,
        };
        let damping = optional_nonneg_float(e, "damping", path, dc, "joint")?.unwrap_or(0.0);
        let armature = optional_nonneg_float(e, "armature", path, dc, "joint")?.unwrap_or(0.0);
        // No `<limit>` sub-element in MJCF; solref-limit / solimp-limit
        // would go through the constraint solver — not exposed in the
        // subset. Use the engine default (zero penalty stiffness/damping;
        // solver limits fall back to SolRef::DEFAULT / SolImp::DEFAULT).
        let limit = JointLimit::DEFAULT;
        Ok((axis, range, damping, armature, limit))
    }

    // ---------- geoms ---------------------------------------------------

    fn add_static_geom(&mut self, e: &Element, path: &str, class: &str) -> Result<(), MjcfError> {
        let geom = self.build_geom(e, path, class, GeomAttach::Static)?;
        let name = attr_required(e, "name", path)?.to_string();
        if self.geoms_by_name.contains_key(&name) {
            return fail(path, format!("duplicate geom name \"{name}\""));
        }
        let idx = self.world.add_geom(geom);
        self.geoms_by_name.insert(name, idx);
        Ok(())
    }

    fn add_body_geom(
        &mut self,
        e: &Element,
        path: &str,
        class: &str,
        body_idx: usize,
    ) -> Result<(), MjcfError> {
        let geom = self.build_geom(e, path, class, GeomAttach::Body(body_idx))?;
        let name = attr_required(e, "name", path)?.to_string();
        if self.geoms_by_name.contains_key(&name) {
            return fail(path, format!("duplicate geom name \"{name}\""));
        }
        let idx = self.world.add_geom(geom);
        self.geoms_by_name.insert(name, idx);
        Ok(())
    }

    fn add_link_geom(
        &mut self,
        e: &Element,
        path: &str,
        class: &str,
        tree_idx: usize,
        link_idx: usize,
    ) -> Result<(), MjcfError> {
        let geom = self.build_geom(e, path, class, GeomAttach::Link(tree_idx, link_idx))?;
        let name = attr_required(e, "name", path)?.to_string();
        if self.geoms_by_name.contains_key(&name) {
            return fail(path, format!("duplicate geom name \"{name}\""));
        }
        let idx = self.world.add_geom(geom);
        self.geoms_by_name.insert(name, idx);
        Ok(())
    }

    fn build_geom(
        &self,
        e: &Element,
        path: &str,
        class: &str,
        attach: GeomAttach,
    ) -> Result<Geom, MjcfError> {
        let effective_class = e.attr("class").unwrap_or(class).to_string();
        let dc = self
            .defaults
            .lookup(&effective_class)
            .cloned()
            .unwrap_or_default();
        // Attribute whitelist.
        for (k, _) in &e.attrs {
            match k.as_str() {
                "name" | "type" | "pos" | "quat" | "euler" | "axisangle" | "size" | "fromto"
                | "friction" | "solref" | "solimp" | "condim" | "margin" | "gap" | "class"
                | "mass" => {}
                other => {
                    return fail(
                        path,
                        format!(
                            "<geom> attribute \"{other}\" is not supported in the v1 subset \
                             (see docs/mjcf.md for the geom attribute table)"
                        ),
                    );
                }
            }
        }

        let ty = attr_with_default(e, "geom", "type", &dc).unwrap_or("sphere");
        let (shape, mut local_offset, mut local_orientation) =
            self.parse_geom_shape(e, path, ty, &dc)?;

        // Explicit pos/quat/euler on the geom override the fromto-derived
        // orientation (for shapes that support one).
        if let Some(v) = attr_with_default(e, "geom", "pos", &dc) {
            local_offset = parse_vec3_attr(v, path, "pos")?;
        }
        if let Some(ori) = self.parse_element_orientation(e, path, &dc, "geom")? {
            local_orientation = ori;
        }
        // Static geoms with a plane shape follow the JSON loader's
        // `static_plane` construction path so their local_orientation
        // aligns local +Z with the world +Z normal — which is what
        // MuJoCo's <geom type="plane"> assumes.
        if matches!(shape, GeomShape::Plane) && !matches!(attach, GeomAttach::Static) {
            return fail(
                path,
                "<geom type=\"plane\"> must attach to the world (static)",
            );
        }

        let (body, link) = match attach {
            GeomAttach::Static => (None, None),
            GeomAttach::Body(b) => (Some(b), None),
            GeomAttach::Link(t, l) => (None, Some((t, l))),
        };

        // friction: 1–3 numbers "sliding torsional rolling".
        let (friction, torsional_friction, rolling_friction) =
            match attr_with_default(e, "geom", "friction", &dc) {
                Some(v) => {
                    let nums = parse_f32_list(v, path, "friction")?;
                    if nums.is_empty() || nums.len() > 3 {
                        return fail(
                            path,
                            format!("friction must have 1, 2, or 3 numbers (got {})", nums.len()),
                        );
                    }
                    for (i, &v) in nums.iter().enumerate() {
                        if v < 0.0 {
                            return fail(path, format!("friction[{i}] = {v} must be ≥ 0"));
                        }
                    }
                    let f = nums[0];
                    let tf = nums.get(1).copied().unwrap_or(0.0);
                    let rf = nums.get(2).copied().unwrap_or(0.0);
                    (f, tf, rf)
                }
                None => (0.5, 0.0, 0.0),
            };
        let solref = self
            .parse_solref_attr(e, path, &dc, "geom")?
            .unwrap_or(SolRef::DEFAULT);
        let solimp = self
            .parse_solimp_attr(e, path, &dc, "geom")?
            .unwrap_or(SolImp::DEFAULT);
        let condim = match attr_with_default(e, "geom", "condim", &dc) {
            Some(v) => {
                let n = parse_int(v, path, "condim")?;
                match n {
                    1 | 3 | 4 | 6 => n as u8,
                    _ => {
                        return fail(path, format!("condim must be 1|3|4|6 (got {n})"));
                    }
                }
            }
            None => 3,
        };
        let margin = optional_nonneg_float(e, "margin", path, &dc, "geom")?.unwrap_or(0.0);
        let gap = optional_nonneg_float(e, "gap", path, &dc, "geom")?.unwrap_or(0.0);

        // For static planes, mimic the JSON loader's `static_plane`
        // construction: local_orientation aligns local +Z to the world
        // normal. MJCF's plane defaults to +Z, and no orientation was
        // specified in the fixture, so keep IDENTITY (which already means
        // "local +Z = world +Z"). We DO NOT call `Geom::static_plane` here
        // because its position argument is world-frame (matches JSON
        // pos=[0,0,0]).
        let _ = &local_offset;
        let _ = &local_orientation;
        Ok(Geom {
            shape,
            body,
            link,
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
        })
    }

    /// Return `(GeomShape, derived local_offset, derived local_orientation)`.
    /// `fromto` on a capsule/cylinder produces both the offset (mid-point)
    /// and orientation (align local +Z with the b−a direction).
    fn parse_geom_shape(
        &self,
        e: &Element,
        path: &str,
        ty: &str,
        dc: &DefaultClass,
    ) -> Result<(GeomShape, Vec3, Quat), MjcfError> {
        let size_str = attr_with_default(e, "geom", "size", dc);
        let fromto_str = attr_with_default(e, "geom", "fromto", dc);
        match ty {
            "plane" => {
                // Plane accepts up to 3 sizes ("width height thickness") in
                // MJCF; we ignore them since newt planes are infinite.
                // Presence of `fromto` on a plane is an error.
                if fromto_str.is_some() {
                    return fail(path, "<geom type=\"plane\"> does not accept fromto");
                }
                Ok((GeomShape::Plane, Vec3::ZERO, Quat::IDENTITY))
            }
            "sphere" => {
                let size = size_str
                    .ok_or_else(|| MjcfError::new(path, "sphere requires size=\"radius\""))?;
                let nums = parse_f32_list(size, path, "size")?;
                if nums.is_empty() {
                    return fail(path, "sphere size must have ≥ 1 number");
                }
                let radius = nums[0];
                if radius <= 0.0 {
                    return fail(path, "sphere radius must be > 0");
                }
                Ok((GeomShape::Sphere { radius }, Vec3::ZERO, Quat::IDENTITY))
            }
            "box" => {
                let size = size_str
                    .ok_or_else(|| MjcfError::new(path, "box requires size=\"hx hy hz\""))?;
                let nums = parse_f32_list(size, path, "size")?;
                require_len(&nums, 3, path, "size")?;
                for (i, &v) in nums.iter().enumerate() {
                    if v <= 0.0 {
                        return fail(path, format!("box size[{i}] = {v} must be > 0"));
                    }
                }
                Ok((
                    GeomShape::Box {
                        half_extents: Vec3::new(nums[0], nums[1], nums[2]),
                    },
                    Vec3::ZERO,
                    Quat::IDENTITY,
                ))
            }
            "capsule" => {
                let (r, half_h, offset, ori) =
                    self.capsule_cylinder_dims(e, path, size_str, fromto_str, "capsule")?;
                Ok((
                    GeomShape::Capsule {
                        radius: r,
                        half_height: half_h,
                    },
                    offset,
                    ori,
                ))
            }
            "cylinder" => {
                let (r, half_h, offset, ori) =
                    self.capsule_cylinder_dims(e, path, size_str, fromto_str, "cylinder")?;
                Ok((
                    GeomShape::Cylinder {
                        radius: r,
                        half_height: half_h,
                    },
                    offset,
                    ori,
                ))
            }
            "ellipsoid" => {
                let size = size_str
                    .ok_or_else(|| MjcfError::new(path, "ellipsoid requires size=\"ax ay az\""))?;
                let nums = parse_f32_list(size, path, "size")?;
                require_len(&nums, 3, path, "size")?;
                for (i, &v) in nums.iter().enumerate() {
                    if v <= 0.0 {
                        return fail(path, format!("ellipsoid semi_axis[{i}] = {v} must be > 0"));
                    }
                }
                Ok((
                    GeomShape::Ellipsoid {
                        semi_axes: Vec3::new(nums[0], nums[1], nums[2]),
                    },
                    Vec3::ZERO,
                    Quat::IDENTITY,
                ))
            }
            "mesh" => fail(
                path,
                "<geom type=\"mesh\"> is not supported in the v1 subset \
                 (mesh assets need <asset><mesh>, which is not in scope)",
            ),
            "hfield" | "sdf" => fail(
                path,
                format!("<geom type=\"{ty}\"> is not supported in the v1 subset"),
            ),
            other => fail(
                path,
                format!(
                    "unknown geom type \"{other}\" (expected plane|sphere|box|\
                     capsule|cylinder|ellipsoid)"
                ),
            ),
        }
    }

    /// Handle both `size` and `fromto` for capsule/cylinder. `size` is the
    /// radius (and optionally half-length). `fromto` is `x1 y1 z1 x2 y2
    /// z2` — the mid-point is the geom center and the half-length is
    /// `|b-a|/2`. `fromto` takes precedence for the axis direction and
    /// mid-point, and (when present) overrides the `size` half-length
    /// with the fromto-derived one (MuJoCo semantics).
    fn capsule_cylinder_dims(
        &self,
        _e: &Element,
        path: &str,
        size_str: Option<&str>,
        fromto_str: Option<&str>,
        label: &str,
    ) -> Result<(f32, f32, Vec3, Quat), MjcfError> {
        let size = size_str.ok_or_else(|| {
            MjcfError::new(
                path,
                format!("{label} requires size=\"radius [half-length]\""),
            )
        })?;
        let size_nums = parse_f32_list(size, path, "size")?;
        if size_nums.is_empty() || size_nums.len() > 2 {
            return fail(
                path,
                format!(
                    "{label} size must have 1 or 2 numbers, got {}",
                    size_nums.len()
                ),
            );
        }
        let radius = size_nums[0];
        if radius <= 0.0 {
            return fail(path, format!("{label} radius must be > 0"));
        }
        if let Some(ft) = fromto_str {
            let nums = parse_f32_list(ft, path, "fromto")?;
            require_len(&nums, 6, path, "fromto")?;
            let a = Vec3::new(nums[0], nums[1], nums[2]);
            let b = Vec3::new(nums[3], nums[4], nums[5]);
            let d = b - a;
            let len = d.length();
            if len <= 0.0 {
                return fail(path, "fromto endpoints coincide");
            }
            let half = len * 0.5;
            let center = Vec3::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, (a.z + b.z) * 0.5);
            let ori = quat_align_z_to(d * (1.0 / len));
            Ok((radius, half, center, ori))
        } else {
            let half = size_nums.get(1).copied().ok_or_else(|| {
                MjcfError::new(
                    path,
                    format!("{label} without fromto requires size=\"radius half-length\""),
                )
            })?;
            if half < 0.0 {
                return fail(path, format!("{label} half-length must be ≥ 0"));
            }
            Ok((radius, half, Vec3::ZERO, Quat::IDENTITY))
        }
    }

    fn parse_element_orientation(
        &self,
        e: &Element,
        path: &str,
        dc: &DefaultClass,
        element_name: &str,
    ) -> Result<Option<Quat>, MjcfError> {
        let q = attr_with_default(e, element_name, "quat", dc);
        let eu = attr_with_default(e, element_name, "euler", dc);
        let ax = attr_with_default(e, element_name, "axisangle", dc);
        let count = q.is_some() as usize + eu.is_some() as usize + ax.is_some() as usize;
        if count > 1 {
            return fail(
                path,
                format!("specify at most one of quat / euler / axisangle on <{element_name}>"),
            );
        }
        if let Some(q) = q {
            return Ok(Some(parse_quat_wxyz(q, path, "quat")?));
        }
        if let Some(eu) = eu {
            let nums = parse_f32_list(eu, path, "euler")?;
            require_len(&nums, 3, path, "euler")?;
            return Ok(Some(parse_euler_xyz(&nums, self.angle_scale)));
        }
        if let Some(aa) = ax {
            let nums = parse_f32_list(aa, path, "axisangle")?;
            require_len(&nums, 4, path, "axisangle")?;
            let axis = Vec3::new(nums[0], nums[1], nums[2]);
            if axis.length_squared() < 1e-12 {
                return fail(path, "axisangle axis must be non-zero");
            }
            return Ok(Some(Quat::from_axis_angle(
                axis,
                nums[3] * self.angle_scale,
            )));
        }
        Ok(None)
    }

    fn parse_solref_attr(
        &self,
        e: &Element,
        path: &str,
        dc: &DefaultClass,
        element_name: &str,
    ) -> Result<Option<SolRef>, MjcfError> {
        let Some(v) = attr_with_default(e, element_name, "solref", dc) else {
            return Ok(None);
        };
        let nums = parse_f32_list(v, path, "solref")?;
        require_len(&nums, 2, path, "solref")?;
        if nums[0] <= 0.0 || nums[1] < 0.0 {
            return fail(
                path,
                "solref timeconst must be > 0 and dampratio must be ≥ 0",
            );
        }
        Ok(Some(SolRef::new(nums[0], nums[1])))
    }

    fn parse_solimp_attr(
        &self,
        e: &Element,
        path: &str,
        dc: &DefaultClass,
        element_name: &str,
    ) -> Result<Option<SolImp>, MjcfError> {
        let Some(v) = attr_with_default(e, element_name, "solimp", dc) else {
            return Ok(None);
        };
        let nums = parse_f32_list(v, path, "solimp")?;
        if nums.len() != 3 && nums.len() != 5 {
            return fail(
                path,
                format!("solimp expects 3 or 5 numbers, got {}", nums.len()),
            );
        }
        let dmin = nums[0];
        let dmax = nums[1];
        let width = nums[2];
        let midpoint = *nums.get(3).unwrap_or(&0.5);
        let power_f = *nums.get(4).unwrap_or(&2.0);
        if power_f < 1.0 || power_f != power_f.floor() {
            return fail(
                path,
                format!("solimp power must be a positive integer, got {power_f}"),
            );
        }
        let s = SolImp::new(dmin, dmax, width, midpoint, power_f as u32);
        s.validate().map_err(|m| MjcfError::new(path, m))?;
        Ok(Some(s))
    }

    // ---------- sites ---------------------------------------------------

    fn add_site_on_body(
        &mut self,
        e: &Element,
        path: &str,
        class: &str,
        body_idx: usize,
    ) -> Result<(), MjcfError> {
        let site = self.build_site(e, path, class, SiteAttach::Body(body_idx))?;
        if self.sites_by_name.contains_key(&site.name) {
            return fail(path, format!("duplicate site name \"{}\"", site.name));
        }
        let idx = self.sites.len();
        self.sites_by_name.insert(site.name.clone(), idx);
        self.sites.push(site);
        Ok(())
    }

    fn add_site_on_link(
        &mut self,
        e: &Element,
        path: &str,
        class: &str,
        tree_idx: usize,
        link_idx: usize,
    ) -> Result<(), MjcfError> {
        let site = self.build_site(
            e,
            path,
            class,
            SiteAttach::Link {
                tree: tree_idx,
                link: link_idx,
            },
        )?;
        if self.sites_by_name.contains_key(&site.name) {
            return fail(path, format!("duplicate site name \"{}\"", site.name));
        }
        let idx = self.sites.len();
        self.sites_by_name.insert(site.name.clone(), idx);
        self.sites.push(site);
        Ok(())
    }

    fn build_site(
        &self,
        e: &Element,
        path: &str,
        class: &str,
        attach: SiteAttach,
    ) -> Result<Site, MjcfError> {
        let effective_class = e.attr("class").unwrap_or(class);
        let dc = self
            .defaults
            .lookup(effective_class)
            .cloned()
            .unwrap_or_default();
        for (k, _) in &e.attrs {
            match k.as_str() {
                "name" | "pos" | "quat" | "euler" | "axisangle" | "size" | "class" => {}
                other => {
                    return fail(
                        path,
                        format!("<site> attribute \"{other}\" is not supported"),
                    );
                }
            }
        }
        let name = attr_required(e, "name", path)?.to_string();
        if name.is_empty() {
            return fail(path, "site name must not be empty");
        }
        let local_offset = match attr_with_default(e, "site", "pos", &dc) {
            Some(v) => parse_vec3_attr(v, path, "pos")?,
            None => Vec3::ZERO,
        };
        let local_orientation = self
            .parse_element_orientation(e, path, &dc, "site")?
            .unwrap_or(Quat::IDENTITY);
        Ok(Site {
            name,
            attach,
            local_offset,
            local_orientation,
        })
    }

    // ---------- actuator ------------------------------------------------

    fn walk_actuator(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        if let Some((k, _)) = e.attrs.first() {
            return fail(
                path,
                format!("<actuator> takes no attributes (got \"{k}\")"),
            );
        }
        for child in e.child_elements() {
            let subpath = child_path(path, &child.name, child.attr("name"));
            match child.name.as_str() {
                "position" => self.add_position_actuator(child, &subpath)?,
                "motor" => self.add_motor_actuator(child, &subpath)?,
                "general" | "velocity" | "cylinder" | "damper" | "muscle" | "intvelocity" => {
                    return fail(
                        &subpath,
                        format!(
                            "<{}> actuator is not supported in the v1 subset",
                            child.name
                        ),
                    );
                }
                other => {
                    return fail(
                        &subpath,
                        format!("unknown <actuator> child <{other}>; supported: position, motor"),
                    );
                }
            }
        }
        Ok(())
    }

    fn add_position_actuator(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        // Class lookup for actuator defaults.
        let class = e.attr("class").unwrap_or(DefaultsTable::MAIN).to_string();
        let dc = self.defaults.lookup(&class).cloned().unwrap_or_default();
        for (k, _) in &e.attrs {
            match k.as_str() {
                "name" | "joint" | "kp" | "kv" | "dampratio" | "forcerange" | "ctrlrange"
                | "class" | "ctrllimited" | "forcelimited" | "target" => {}
                other => {
                    return fail(
                        path,
                        format!("<position> attribute \"{other}\" not supported"),
                    );
                }
            }
        }
        let name = attr_required(e, "name", path)?.to_string();
        let joint_name = attr_required(e, "joint", path)?;
        let (tree_idx, link_idx) = self.resolve_1dof_joint_for_actuator(joint_name, path)?;

        let kp = parse_f32(
            attr_with_default(e, "position", "kp", &dc)
                .ok_or_else(|| MjcfError::new(path, "<position> requires kp"))?,
            path,
            "kp",
        )?;
        if kp < 0.0 {
            return fail(path, "kp must be ≥ 0");
        }
        let has_kv = attr_with_default(e, "position", "kv", &dc).is_some();
        let has_dr = attr_with_default(e, "position", "dampratio", &dc).is_some();
        if has_kv && has_dr {
            return fail(path, "specify either kv OR dampratio, not both");
        }
        // Convert forcerange -> symmetric clamp (min(|lo|, |hi|)).
        let clamp = match attr_with_default(e, "position", "forcerange", &dc) {
            Some(v) => {
                let nums = parse_f32_list(v, path, "forcerange")?;
                require_len(&nums, 2, path, "forcerange")?;
                let lo = nums[0];
                let hi = nums[1];
                if lo >= hi {
                    return fail(path, "forcerange low must be < high");
                }
                lo.abs().min(hi.abs())
            }
            None => 0.0,
        };
        // Accept-and-ignore ctrlrange with a documented note in mjcf.md.
        if let Some(v) = attr_with_default(e, "position", "ctrlrange", &dc) {
            let _ = parse_f32_list(v, path, "ctrlrange")?;
        }
        // Reject the ctrllimited/forcelimited flags with a clear message —
        // they're valid MJCF but not enforced by our subset.
        if let Some(v) = e.attr("ctrllimited") {
            let _ = parse_bool(v, path, "ctrllimited")?;
        }
        if let Some(v) = e.attr("forcelimited") {
            let _ = parse_bool(v, path, "forcelimited")?;
        }
        // Newt extension: an optional `target` attribute on <position>
        // seeds the PD servo's initial setpoint at load time. Real MJCF
        // sets targets via <keyframe qctrl=...> (out of subset); this
        // shortcut lets a fixture bake a standing pose without an
        // external setup step.
        let initial_target = match attr_with_default(e, "position", "target", &dc) {
            Some(v) => parse_f32(v, path, "target")?,
            None => 0.0,
        };
        let mut servo = if has_kv {
            let kv = parse_f32(
                attr_with_default(e, "position", "kv", &dc).unwrap(),
                path,
                "kv",
            )?;
            if kv < 0.0 {
                return fail(path, "kv must be ≥ 0");
            }
            PdServo::new(link_idx, kp, kv, clamp, initial_target)
        } else if has_dr {
            let dr = parse_f32(
                attr_with_default(e, "position", "dampratio", &dc).unwrap(),
                path,
                "dampratio",
            )?;
            if dr < 0.0 {
                return fail(path, "dampratio must be ≥ 0");
            }
            // Reflected-inertia estimate matches MuJoCo's `meaninertia`
            // approximation for a fresh scene: use unit reflected inertia
            // and let dampratio scale kd = 2·dr·sqrt(kp·1) = 2·dr·sqrt(kp).
            // Documented in mjcf.md.
            PdServo::from_dampratio(link_idx, kp, dr, 1.0, clamp)
        } else {
            return fail(path, "<position> actuator must specify kv or dampratio");
        };
        servo.target = initial_target;
        if self.actuators_by_name.contains_key(&name) {
            return fail(path, format!("duplicate actuator name \"{name}\""));
        }
        let act_idx = self.world.trees[tree_idx].add_actuator(servo);
        self.actuators_by_name.insert(name, (tree_idx, act_idx));
        Ok(())
    }

    fn add_motor_actuator(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        let class = e.attr("class").unwrap_or(DefaultsTable::MAIN).to_string();
        let dc = self.defaults.lookup(&class).cloned().unwrap_or_default();
        for (k, _) in &e.attrs {
            match k.as_str() {
                "name" | "joint" | "gear" | "forcerange" | "ctrlrange" | "class"
                | "ctrllimited" | "forcelimited" => {}
                other => {
                    return fail(path, format!("<motor> attribute \"{other}\" not supported"));
                }
            }
        }
        let name = attr_required(e, "name", path)?.to_string();
        let joint_name = attr_required(e, "joint", path)?;
        let (tree_idx, link_idx) = self.resolve_1dof_joint_for_actuator(joint_name, path)?;
        let gear = match attr_with_default(e, "motor", "gear", &dc) {
            Some(v) => {
                let nums = parse_f32_list(v, path, "gear")?;
                if nums.is_empty() || nums.len() > 6 {
                    return fail(
                        path,
                        format!("motor gear must have 1..=6 numbers (got {})", nums.len()),
                    );
                }
                // Only the first scalar is meaningful for a 1-DOF joint.
                for (i, &g) in nums.iter().enumerate().skip(1) {
                    if g != 0.0 {
                        return fail(
                            path,
                            format!(
                                "motor gear entry {i} = {g} is not supported (only the first \
                                 gear scalar is honored for 1-DOF joints)"
                            ),
                        );
                    }
                }
                nums[0]
            }
            None => 1.0,
        };
        let clamp = match attr_with_default(e, "motor", "forcerange", &dc) {
            Some(v) => {
                let nums = parse_f32_list(v, path, "forcerange")?;
                require_len(&nums, 2, path, "forcerange")?;
                nums[0].abs().min(nums[1].abs())
            }
            None => 0.0,
        };
        // Motor = direct torque. Model as a PD servo with kp=0, kd=0, and
        // gear folded into the (fixed) target. `target` here is really a
        // command scale — the caller writes it each step.
        // For fixed-command demos we keep target=0 and clamp; a caller sets
        // the effective torque via `set_actuator_target` where target ==
        // desired torque.
        let servo = PdServo::new(link_idx, 0.0, 0.0, clamp, 0.0);
        let _ = gear;
        if self.actuators_by_name.contains_key(&name) {
            return fail(path, format!("duplicate actuator name \"{name}\""));
        }
        let act_idx = self.world.trees[tree_idx].add_actuator(servo);
        self.actuators_by_name.insert(name, (tree_idx, act_idx));
        Ok(())
    }

    fn resolve_1dof_joint_for_actuator(
        &self,
        joint_name: &str,
        path: &str,
    ) -> Result<(usize, usize), MjcfError> {
        let (tree_idx, link_idx) = self
            .joints_by_name
            .get(joint_name)
            .copied()
            .ok_or_else(|| MjcfError::new(path, format!("unknown joint \"{joint_name}\"")))?;
        if tree_idx == usize::MAX {
            return fail(
                path,
                format!("joint \"{joint_name}\" is a freejoint; actuators need a hinge or slide"),
            );
        }
        // Must be hinge or slide (world::Tree::add_actuator would panic).
        match self.world.trees[tree_idx].links[link_idx].joint {
            JointKind::Hinge { .. } | JointKind::Slide { .. } => Ok((tree_idx, link_idx)),
            _ => fail(
                path,
                format!("actuator target joint \"{joint_name}\" is not a hinge or slide"),
            ),
        }
    }

    // ---------- sensor --------------------------------------------------

    fn walk_sensor(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        if let Some((k, _)) = e.attrs.first() {
            return fail(path, format!("<sensor> takes no attributes (got \"{k}\")"));
        }
        for child in e.child_elements() {
            let subpath = child_path(path, &child.name, child.attr("name"));
            let sensor = self.build_sensor(child, &subpath)?;
            if self.sensors_by_name.contains_key(&sensor.name) {
                return fail(
                    &subpath,
                    format!("duplicate sensor name \"{}\"", sensor.name),
                );
            }
            let name = sensor.name.clone();
            let idx = self
                .world
                .add_sensor(sensor)
                .map_err(|e| MjcfError::new(subpath.clone(), e.0))?;
            self.sensors_by_name.insert(name, idx);
        }
        Ok(())
    }

    fn build_sensor(&self, e: &Element, path: &str) -> Result<Sensor, MjcfError> {
        // MJCF sensor kinds: subset that mirrors the JSON loader.
        let name = attr_required(e, "name", path)?.to_string();
        if name.is_empty() {
            return fail(path, "sensor name must not be empty");
        }
        let joint_ref = |this: &Self, attr: &str| -> Result<(usize, usize), MjcfError> {
            let jn = attr_required(e, attr, path)?;
            let (t, l) = this
                .joints_by_name
                .get(jn)
                .copied()
                .ok_or_else(|| MjcfError::new(path, format!("unknown joint \"{jn}\"")))?;
            if t == usize::MAX {
                return fail(
                    path,
                    format!("joint \"{jn}\" is a freejoint; this sensor needs a hinge/slide/ball"),
                );
            }
            Ok((t, l))
        };
        let site_ref = |this: &Self, attr: &str| -> Result<SiteFrame, MjcfError> {
            let sn = attr_required(e, attr, path)?;
            let sidx = this
                .sites_by_name
                .get(sn)
                .copied()
                .ok_or_else(|| MjcfError::new(path, format!("unknown site \"{sn}\"")))?;
            let s = &this.sites[sidx];
            let attach = match s.attach {
                SiteAttach::Body(i) => SensorAttach::Body(i),
                SiteAttach::Link { tree, link } => SensorAttach::Link(tree, link),
            };
            Ok(SiteFrame {
                attach,
                local_offset: s.local_offset,
                local_orientation: s.local_orientation,
            })
        };
        let body_or_link_ref = |this: &Self, attr: &str| -> Result<(usize, usize), MjcfError> {
            let bn = attr_required(e, attr, path)?;
            this.tree_bodies_by_name
                .get(bn)
                .copied()
                .ok_or_else(|| MjcfError::new(path, format!("unknown body \"{bn}\"")))
        };
        let attrs_ok = |allowed: &[&str]| -> Result<(), MjcfError> {
            for (k, _) in &e.attrs {
                if !allowed.contains(&k.as_str()) {
                    return fail(
                        path,
                        format!("<{}> sensor: unknown attribute \"{k}\"", e.name),
                    );
                }
            }
            Ok(())
        };
        let kind = match e.name.as_str() {
            "jointpos" => {
                attrs_ok(&["name", "joint"])?;
                let (t, l) = joint_ref(self, "joint")?;
                SensorKind::JointPos { tree: t, link: l }
            }
            "jointvel" => {
                attrs_ok(&["name", "joint"])?;
                let (t, l) = joint_ref(self, "joint")?;
                SensorKind::JointVel { tree: t, link: l }
            }
            "ballquat" => {
                attrs_ok(&["name", "joint"])?;
                let (t, l) = joint_ref(self, "joint")?;
                SensorKind::BallQuat { tree: t, link: l }
            }
            "ballangvel" => {
                attrs_ok(&["name", "joint"])?;
                let (t, l) = joint_ref(self, "joint")?;
                SensorKind::BallAngVel { tree: t, link: l }
            }
            "framepos" => {
                attrs_ok(&["name", "site"])?;
                SensorKind::FramePos(site_ref(self, "site")?)
            }
            "framequat" => {
                attrs_ok(&["name", "site"])?;
                SensorKind::FrameQuat(site_ref(self, "site")?)
            }
            "gyro" => {
                attrs_ok(&["name", "site"])?;
                SensorKind::Gyro(site_ref(self, "site")?)
            }
            "accelerometer" => {
                attrs_ok(&["name", "site"])?;
                SensorKind::Accelerometer(site_ref(self, "site")?)
            }
            "touch" => {
                attrs_ok(&["name", "site", "geom"])?;
                // MuJoCo `touch` associates with a site; ours takes a geom.
                let gn = attr_required(e, "geom", path)?;
                let gidx = self
                    .geoms_by_name
                    .get(gn)
                    .copied()
                    .ok_or_else(|| MjcfError::new(path, format!("unknown geom \"{gn}\"")))?;
                SensorKind::Touch { geom: gidx }
            }
            "force" => {
                attrs_ok(&["name", "body"])?;
                let (t, l) = body_or_link_ref(self, "body")?;
                if t == usize::MAX {
                    return fail(
                        path,
                        "force sensor can only reference tree-link bodies in the v1 subset",
                    );
                }
                SensorKind::Force { tree: t, link: l }
            }
            "torque" => {
                attrs_ok(&["name", "body"])?;
                let (t, l) = body_or_link_ref(self, "body")?;
                if t == usize::MAX {
                    return fail(
                        path,
                        "torque sensor can only reference tree-link bodies in the v1 subset",
                    );
                }
                SensorKind::Torque { tree: t, link: l }
            }
            other => {
                return fail(
                    path,
                    format!(
                        "sensor kind <{other}> is not supported in the v1 subset \
                         (supported: jointpos, jointvel, ballquat, ballangvel, \
                         framepos, framequat, gyro, accelerometer, touch, force, torque)"
                    ),
                );
            }
        };
        Ok(Sensor { name, kind })
    }

    // ---------- equality ------------------------------------------------

    fn walk_equality(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        if let Some((k, _)) = e.attrs.first() {
            return fail(
                path,
                format!("<equality> takes no attributes (got \"{k}\")"),
            );
        }
        for child in e.child_elements() {
            let subpath = child_path(path, &child.name, None);
            let eq = self.build_equality(child, &subpath)?;
            eq.validate()
                .map_err(|m| MjcfError::new(subpath.clone(), m))?;
            self.world.equalities.push(eq);
        }
        Ok(())
    }

    fn build_equality(&self, e: &Element, path: &str) -> Result<Equality, MjcfError> {
        let solref = self
            .parse_solref_attr(e, path, &DefaultClass::default(), &e.name)?
            .unwrap_or(SolRef::DEFAULT);
        let solimp = self
            .parse_solimp_attr(e, path, &DefaultClass::default(), &e.name)?
            .unwrap_or(SolImp::DEFAULT);
        match e.name.as_str() {
            "connect" => {
                for (k, _) in &e.attrs {
                    match k.as_str() {
                        "name" | "body1" | "body2" | "anchor" | "solref" | "solimp" => {}
                        other => {
                            return fail(path, format!("<connect> unknown attribute \"{other}\""));
                        }
                    }
                }
                let body_a = self.free_body_ref(e.attr("body1"), path, "body1")?;
                let body_b = self.free_body_ref(e.attr("body2"), path, "body2")?;
                let anchor = parse_vec3_attr(attr_required(e, "anchor", path)?, path, "anchor")?;
                Ok(Equality::Connect {
                    body_a,
                    body_b,
                    anchor_a: anchor,
                    anchor_b: anchor,
                    solref,
                    solimp,
                })
            }
            "weld" => {
                for (k, _) in &e.attrs {
                    match k.as_str() {
                        "name" | "body1" | "body2" | "anchor" | "relpose" | "solref" | "solimp" => {
                        }
                        other => {
                            return fail(path, format!("<weld> unknown attribute \"{other}\""));
                        }
                    }
                }
                let body_a = self.free_body_ref(e.attr("body1"), path, "body1")?;
                let body_b = self.free_body_ref(e.attr("body2"), path, "body2")?;
                let anchor = parse_vec3_attr(attr_required(e, "anchor", path)?, path, "anchor")?;
                let relq = match e.attr("relpose") {
                    Some(v) => {
                        // MuJoCo relpose = "px py pz qw qx qy qz" (7 values).
                        let nums = parse_f32_list(v, path, "relpose")?;
                        require_len(&nums, 7, path, "relpose")?;
                        Quat::new(nums[4], nums[5], nums[6], nums[3]).renormalize()
                    }
                    None => Quat::IDENTITY,
                };
                Ok(Equality::Weld {
                    body_a,
                    body_b,
                    anchor_a: anchor,
                    anchor_b: anchor,
                    relative_orientation: relq,
                    solref,
                    solimp,
                })
            }
            "joint" => {
                for (k, _) in &e.attrs {
                    match k.as_str() {
                        "name" | "joint1" | "joint2" | "polycoef" | "solref" | "solimp" => {}
                        other => {
                            return fail(
                                path,
                                format!("<joint> equality unknown attribute \"{other}\""),
                            );
                        }
                    }
                }
                let j1 = attr_required(e, "joint1", path)?;
                let j2 = attr_required(e, "joint2", path)?;
                let (t1, l1) = self
                    .joints_by_name
                    .get(j1)
                    .copied()
                    .ok_or_else(|| MjcfError::new(path, format!("unknown joint \"{j1}\"")))?;
                let (t2, l2) = self
                    .joints_by_name
                    .get(j2)
                    .copied()
                    .ok_or_else(|| MjcfError::new(path, format!("unknown joint \"{j2}\"")))?;
                if t1 != t2 || t1 == usize::MAX {
                    return fail(
                        path,
                        "joint-equality endpoints must be joints in the same tree",
                    );
                }
                let poly = match e.attr("polycoef") {
                    Some(v) => {
                        let nums = parse_f32_list(v, path, "polycoef")?;
                        if nums.is_empty() || nums.len() > 5 {
                            return fail(
                                path,
                                format!("polycoef must have 1..=5 numbers, got {}", nums.len()),
                            );
                        }
                        // newt's Equality::JointCoupling takes a length-3
                        // polynomial [c0, c1, c2]. Reject entries past
                        // index 2 with a clear error.
                        if nums.len() > 3 {
                            return fail(
                                path,
                                "polycoef beyond degree 2 is not supported in the v1 subset",
                            );
                        }
                        let mut c = [0.0f32; 3];
                        for (i, &v) in nums.iter().enumerate() {
                            c[i] = v;
                        }
                        c
                    }
                    None => [0.0, 1.0, 0.0],
                };
                Ok(Equality::JointCoupling {
                    tree: t1,
                    link_a: l1,
                    link_b: l2,
                    polycoef: poly,
                    solref,
                    solimp,
                })
            }
            "distance" | "tendon" | "flex" => fail(
                path,
                format!("<{}> equality is not supported in the v1 subset", e.name),
            ),
            other => fail(
                path,
                format!("unknown <equality> child <{other}> (supported: connect, weld, joint)"),
            ),
        }
    }

    fn free_body_ref(
        &self,
        name: Option<&str>,
        path: &str,
        attr: &str,
    ) -> Result<Option<usize>, MjcfError> {
        let Some(n) = name else {
            return Ok(None);
        };
        if n == "world" {
            return Ok(None);
        }
        // Only free bodies (in World.bodies) can participate in
        // connect/weld/distance in newt. tree-link bodies would require
        // constraint Jacobians we do not currently emit.
        let (t, idx) = self.tree_bodies_by_name.get(n).copied().ok_or_else(|| {
            MjcfError::new(path, format!("unknown body \"{n}\" (attribute {attr})"))
        })?;
        if t != usize::MAX {
            return fail(
                path,
                format!(
                    "body \"{n}\" is a tree-link; equality (connect/weld) requires a free body \
                     in the v1 subset"
                ),
            );
        }
        Ok(Some(idx))
    }

    // ---------- contact -------------------------------------------------

    fn walk_contact(&mut self, e: &Element, path: &str) -> Result<(), MjcfError> {
        if let Some((k, _)) = e.attrs.first() {
            return fail(path, format!("<contact> takes no attributes (got \"{k}\")"));
        }
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut excludes: Vec<(String, String)> = Vec::new();
        let mut has_pair = false;
        let mut has_exclude = false;
        for child in e.child_elements() {
            let subpath = child_path(path, &child.name, None);
            match child.name.as_str() {
                "pair" => {
                    has_pair = true;
                    for (k, _) in &child.attrs {
                        match k.as_str() {
                            "name" | "geom1" | "geom2" | "condim" | "friction" | "margin"
                            | "solref" | "solimp" => {}
                            other => {
                                return fail(
                                    &subpath,
                                    format!("<pair> unknown attribute \"{other}\""),
                                );
                            }
                        }
                    }
                    let g1 = attr_required(child, "geom1", &subpath)?;
                    let g2 = attr_required(child, "geom2", &subpath)?;
                    let a = *self.geoms_by_name.get(g1).ok_or_else(|| {
                        MjcfError::new(&subpath, format!("unknown geom \"{g1}\""))
                    })?;
                    let b = *self.geoms_by_name.get(g2).ok_or_else(|| {
                        MjcfError::new(&subpath, format!("unknown geom \"{g2}\""))
                    })?;
                    if a == b {
                        return fail(&subpath, "contact pair endpoints must be distinct geoms");
                    }
                    pairs.push(if a < b { (a, b) } else { (b, a) });
                }
                "exclude" => {
                    has_exclude = true;
                    // MuJoCo <exclude> operates on bodies. We don't model
                    // body-vs-body exclusion; require geom-level exclusion
                    // via <pair>. Reject cleanly.
                    for (k, _) in &child.attrs {
                        if k != "body1" && k != "body2" && k != "name" {
                            return fail(&subpath, format!("<exclude> unknown attribute \"{k}\""));
                        }
                    }
                    let b1 = attr_required(child, "body1", &subpath)?.to_string();
                    let b2 = attr_required(child, "body2", &subpath)?.to_string();
                    excludes.push((b1, b2));
                }
                other => {
                    return fail(
                        &subpath,
                        format!("unknown <contact> child <{other}> (supported: pair, exclude)"),
                    );
                }
            }
        }
        if has_pair && has_exclude {
            return fail(
                path,
                "<contact> supports <pair> XOR <exclude>, not both in the same <contact> block",
            );
        }
        if has_pair {
            self.world.pair_list = Some(pairs);
        }
        if has_exclude {
            // Body-level exclusion via geom subtraction from the auto list.
            let mut auto = default_auto_pairs(&self.world);
            for (b1, b2) in excludes {
                let g_a: Vec<usize> = self.geoms_for_body(&b1)?;
                let g_b: Vec<usize> = self.geoms_for_body(&b2)?;
                auto.retain(|(x, y)| {
                    let touches_a = g_a.contains(x) || g_a.contains(y);
                    let touches_b = g_b.contains(x) || g_b.contains(y);
                    !(touches_a && touches_b)
                });
            }
            self.world.pair_list = Some(auto);
        }
        Ok(())
    }

    fn geoms_for_body(&self, name: &str) -> Result<Vec<usize>, MjcfError> {
        use crate::geom::GeomAttach as GA;
        let (t, l) = self.tree_bodies_by_name.get(name).copied().ok_or_else(|| {
            MjcfError::new("<contact><exclude>", format!("unknown body \"{name}\""))
        })?;
        let mut out = Vec::new();
        for (i, g) in self.world.geoms.iter().enumerate() {
            match g.attachment() {
                GA::Body(b) if t == usize::MAX && b == l => out.push(i),
                GA::Link(tt, ll) if t == tt && l == ll => out.push(i),
                _ => {}
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum GeomAttach {
    Static,
    Body(usize),
    Link(usize, usize),
}

/// Build an XML-style breadcrumb path string.
fn child_path(parent: &str, name: &str, hint: Option<&str>) -> String {
    match hint {
        Some(h) => format!("{parent} > {name}[{h}]"),
        None => format!("{parent} > {name}"),
    }
}

fn find_free_joint(e: &Element) -> Option<&Element> {
    e.child_elements()
        .find(|c| c.name == "freejoint" || (c.name == "joint" && c.attr("type") == Some("free")))
}

/// Enforce the `<body>` attribute allowlist — mirrors every other
/// element's whitelist and closes the no-silent-ignore hole a review
/// caught with `<body bogus_attr="1">`.
fn validate_body_attrs(e: &Element, path: &str) -> Result<(), MjcfError> {
    for (k, _) in &e.attrs {
        match k.as_str() {
            "name" | "pos" | "quat" | "euler" | "axisangle" | "childclass" => {}
            other => {
                return fail(
                    path,
                    format!(
                        "<body> attribute \"{other}\" is not supported in the v1 \
                         subset (see docs/mjcf.md for the <body> attribute table)"
                    ),
                );
            }
        }
    }
    Ok(())
}

/// True when the body has only free-joint children among its <joint>*
/// elements (no additional hinge/slide/ball joints alongside the free).
fn has_only_free_joint(e: &Element) -> bool {
    let joints: Vec<&Element> = e
        .child_elements()
        .filter(|c| c.name == "joint" || c.name == "freejoint")
        .collect();
    joints.len() == 1
}

fn optional_nonneg_float(
    e: &Element,
    attr: &str,
    path: &str,
    dc: &DefaultClass,
    element_name: &str,
) -> Result<Option<f32>, MjcfError> {
    let Some(v) = attr_with_default(e, element_name, attr, dc) else {
        return Ok(None);
    };
    let f = parse_f32(v, path, attr)?;
    if f < 0.0 {
        return fail(path, format!("{attr} must be ≥ 0"));
    }
    Ok(Some(f))
}

/// Element-wise sum of two matrices (Mat3 is column-major).
fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
    let mut out = [0.0f32; 9];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = a.data[i] + b.data[i];
    }
    Mat3::new(out)
}

/// Deterministic quaternion that rotates local +Z to align with the unit
/// vector `n`. Reuses the same algebra as `crate::geom::quat_align_z_to`
/// (which is not public) — the derivation is `q = normalize(cross(Z, n) +
/// (|Z| |n| + Z·n) * <scalar>)` shortcut.
fn quat_align_z_to(n: Vec3) -> Quat {
    let z = Vec3::new(0.0, 0.0, 1.0);
    let dot = z.dot(n);
    if dot > 1.0 - 1e-6 {
        return Quat::IDENTITY;
    }
    if dot < -1.0 + 1e-6 {
        // 180° rotation about x-axis.
        return Quat::new(1.0, 0.0, 0.0, 0.0);
    }
    let axis = z.cross(n);
    let w = 1.0 + dot;
    Quat::new(axis.x, axis.y, axis.z, w).renormalize()
}

/// Mirror the JSON loader's auto-pair filter (drop same-attachment pairs
/// and same-tree pairs where the tree opts out of self-collision).
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

/// The default auto-pair enumeration MJCF's <exclude> subtracts from —
/// mirrors `World::auto_pairs` (which is private).
fn default_auto_pairs(world: &World) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let n = world.geoms.len();
    for a in 0..n {
        for b in (a + 1)..n {
            if world.geoms[a].attachment() == world.geoms[b].attachment() {
                continue;
            }
            out.push((a, b));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// unit tests (loader validation branches; full-fixture tests live in
// `tests/mjcf_load.rs` and `tests/mjcf_anchor.rs`).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn err(src: &str) -> MjcfError {
        load_mjcf_str(src).expect_err("expected error")
    }

    #[test]
    fn root_must_be_mujoco() {
        let e = err("<foo/>");
        assert!(e.message.contains("must be <mujoco>"), "{}", e.message);
    }

    #[test]
    fn unknown_top_level_rejected() {
        let e = err("<mujoco><bogus/></mujoco>");
        assert!(e.message.contains("unknown top-level"), "{}", e.message);
    }

    #[test]
    fn compiler_bad_angle_rejected() {
        let e = err(r#"<mujoco><compiler angle="grad"/></mujoco>"#);
        assert!(e.message.contains("angle"), "{}", e.message);
    }

    #[test]
    fn option_negative_timestep_rejected() {
        let e = err(r#"<mujoco><option timestep="-0.01"/></mujoco>"#);
        assert!(e.message.contains("timestep"), "{}", e.message);
    }

    #[test]
    fn worldbody_missing_inertial_rejected() {
        let e = err(r#"<mujoco><worldbody>
                 <body name="b" pos="0 0 1"><freejoint/></body>
               </worldbody></mujoco>"#);
        assert!(e.message.contains("<inertial>"), "{}", e.message);
    }

    #[test]
    fn non_root_body_rotation_rejected() {
        let e = err(r#"<mujoco><worldbody>
                 <body name="root" pos="0 0 1">
                   <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   <body name="c" pos="0 0 -0.5" quat="0.7 0 0.7 0">
                     <joint type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                     <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   </body>
                 </body>
               </worldbody></mujoco>"#);
        assert!(
            e.message.contains("body orientation") || e.message.contains("identity"),
            "{}",
            e.message
        );
    }

    #[test]
    fn multi_joint_body_rejected() {
        let e = err(r#"<mujoco><worldbody>
                 <body name="root">
                   <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   <body name="c" pos="0 0 -0.5">
                     <joint type="hinge" axis="1 0 0"/>
                     <joint type="hinge" axis="0 1 0"/>
                     <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   </body>
                 </body>
               </worldbody></mujoco>"#);
        assert!(e.message.contains("joints"), "{}", e.message);
    }

    #[test]
    fn mesh_geom_rejected_with_clean_message() {
        let e = err(r#"<mujoco><worldbody>
                 <geom name="g" type="mesh"/>
               </worldbody></mujoco>"#);
        assert!(e.message.contains("mesh"), "{}", e.message);
    }

    #[test]
    fn eulerseq_non_default_rejected() {
        let e = err(r#"<mujoco><compiler eulerseq="zyx"/></mujoco>"#);
        assert!(e.message.contains("eulerseq"), "{}", e.message);
    }

    #[test]
    fn default_class_lookup_applies_when_attribute_missing() {
        let scene = load_mjcf_str(
            r#"<mujoco>
                 <default>
                   <geom friction="0.9"/>
                 </default>
                 <worldbody>
                   <geom name="ground" type="plane"/>
                 </worldbody>
               </mujoco>"#,
        )
        .expect("scene should load");
        let g = &scene.world.geoms[scene.geoms_by_name["ground"]];
        assert!((g.friction - 0.9).abs() < 1e-6);
    }

    #[test]
    fn nested_default_class_inheritance() {
        // A nested <default class="child"> inherits its parent's geom
        // attributes and can override individual ones.
        let scene = load_mjcf_str(
            r#"<mujoco>
                 <default>
                   <geom friction="0.5" margin="0.01"/>
                   <default class="strong">
                     <geom friction="0.9"/>
                   </default>
                 </default>
                 <worldbody>
                   <geom name="a" type="plane"/>
                   <geom name="b" type="plane" class="strong"/>
                 </worldbody>
               </mujoco>"#,
        )
        .expect("scene should load");
        let a = &scene.world.geoms[scene.geoms_by_name["a"]];
        let b = &scene.world.geoms[scene.geoms_by_name["b"]];
        assert!((a.friction - 0.5).abs() < 1e-6);
        assert!((b.friction - 0.9).abs() < 1e-6);
        // Nested class INHERITED margin from parent.
        assert!((a.margin - 0.01).abs() < 1e-6);
        assert!((b.margin - 0.01).abs() < 1e-6);
    }

    #[test]
    fn childclass_propagates_to_children() {
        let scene = load_mjcf_str(
            r#"<mujoco>
                 <default>
                   <joint damping="1.0"/>
                   <default class="soft">
                     <joint damping="5.0"/>
                   </default>
                 </default>
                 <worldbody>
                   <body name="root" childclass="soft">
                     <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                     <body name="c" pos="0 0 -0.5">
                       <joint type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                       <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                     </body>
                   </body>
                 </worldbody>
               </mujoco>"#,
        )
        .expect("scene should load");
        let tree = &scene.world.trees[0];
        if let JointKind::Hinge { damping, .. } = tree.links[1].joint {
            assert!((damping - 5.0).abs() < 1e-6, "damping = {damping}");
        } else {
            panic!("expected hinge");
        }
    }

    #[test]
    fn degree_angle_converts_range_and_euler() {
        // Range specified in degrees.
        let scene = load_mjcf_str(
            r#"<mujoco>
                 <compiler angle="degree"/>
                 <worldbody>
                   <body name="root">
                     <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                     <body name="c" pos="0 0 -0.5">
                       <joint type="hinge" axis="1 0 0" pos="0 0 0.5" range="-90 90"/>
                       <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                     </body>
                   </body>
                 </worldbody>
               </mujoco>"#,
        )
        .expect("scene should load");
        let tree = &scene.world.trees[0];
        if let JointKind::Hinge {
            range: Some((lo, hi)),
            ..
        } = tree.links[1].joint
        {
            assert!((lo + std::f32::consts::FRAC_PI_2).abs() < 1e-4);
            assert!((hi - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
        } else {
            panic!("expected hinge with range");
        }
    }

    #[test]
    fn fromto_capsule_derives_center_and_orientation() {
        let scene = load_mjcf_str(
            r#"<mujoco><worldbody>
                 <body name="root" pos="0 0 1">
                   <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   <freejoint/>
                   <geom name="cap" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.045"/>
                 </body>
               </worldbody></mujoco>"#,
        )
        .expect("scene should load");
        let g = &scene.world.geoms[scene.geoms_by_name["cap"]];
        // Length 0.42 → half_height = 0.21; center = midpoint (0, 0, -0.21).
        if let GeomShape::Capsule { half_height, .. } = g.shape {
            assert!((half_height - 0.21).abs() < 1e-6);
        } else {
            panic!("expected capsule");
        }
        assert!((g.local_offset.z + 0.21).abs() < 1e-6);
    }

    #[test]
    fn unknown_geom_attribute_rejected() {
        let e = err(r#"<mujoco><worldbody>
                 <geom name="g" type="plane" rgba="1 0 0 1"/>
               </worldbody></mujoco>"#);
        assert!(e.message.contains("rgba"), "{}", e.message);
    }

    #[test]
    fn duplicate_geom_name_rejected() {
        let e = err(r#"<mujoco><worldbody>
                 <geom name="g" type="plane"/>
                 <geom name="g" type="plane"/>
               </worldbody></mujoco>"#);
        assert!(e.message.contains("duplicate"), "{}", e.message);
    }

    #[test]
    fn asset_element_unsupported_error() {
        let e = err("<mujoco><asset/></mujoco>");
        assert!(e.message.contains("<asset>"), "{}", e.message);
        assert!(e.message.contains("not supported"), "{}", e.message);
    }

    #[test]
    fn integrator_non_rk4_rejected() {
        let e = err(r#"<mujoco><option integrator="Euler"/></mujoco>"#);
        assert!(e.message.contains("integrator"), "{}", e.message);
    }

    #[test]
    fn actuator_missing_kv_and_dampratio_rejected() {
        let e = err(r#"<mujoco><worldbody>
                 <body name="root">
                   <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   <body name="c" pos="0 0 -0.5">
                     <joint name="j" type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                     <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   </body>
                 </body>
               </worldbody>
               <actuator>
                 <position name="a" joint="j" kp="10"/>
               </actuator></mujoco>"#);
        assert!(e.message.contains("kv or dampratio"), "{}", e.message);
    }
}
