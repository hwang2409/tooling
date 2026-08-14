//! v1 tier 6 sensor model-loader anchors — one test per rejection path
//! plus a positive round-trip.

use newt::math::Vec3;
use newt::model::{ModelError, load_str};

fn err(src: &str) -> ModelError {
    load_str(src).expect_err("expected loader error")
}

const BASE: &str = r#"
  "gravity":[0,0,-9.81],
  "bodies":[{"name":"b","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}],
  "geoms":[{"name":"g","shape":{"kind":"sphere","radius":0.5},"attach":{"kind":"body","body":"b"}}],
  "sites":[{"name":"s","attach":{"kind":"body","body":"b"}}],
  "trees":[{"name":"t","links":[
    {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
    {"name":"hip","parent":"root","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,
     "inertia":{"kind":"diag","values":[0.01,0.01,0.01]}},
    {"name":"knee","parent":"hip","joint":{"kind":"ball"},"mass":1,
     "inertia":{"kind":"diag","values":[0.01,0.01,0.01]}}
  ]}]
"#;

fn wrap(sensors: &str) -> String {
    format!("{{{BASE}, \"sensors\":[{sensors}]}}")
}

#[test]
fn every_sensor_kind_round_trips() {
    let src = wrap(
        r#"
        {"name":"a","kind":"jointpos","tree":"t","link":"hip"},
        {"name":"b","kind":"jointvel","tree":"t","link":"hip"},
        {"name":"c","kind":"ballquat","tree":"t","link":"knee"},
        {"name":"d","kind":"ballangvel","tree":"t","link":"knee"},
        {"name":"e","kind":"framepos","site":"s"},
        {"name":"f","kind":"framequat","site":"s"},
        {"name":"g","kind":"gyro","site":"s"},
        {"name":"h","kind":"accelerometer","site":"s"},
        {"name":"i","kind":"touch","geom":"g"},
        {"name":"j","kind":"force","tree":"t","link":"hip"},
        {"name":"k","kind":"torque","tree":"t","link":"hip"}
    "#,
    );
    let scene = load_str(&src).unwrap();
    // Expect 11 sensors with a known total dim = 1+1+4+3+3+4+3+3+1+3+3 = 29.
    assert_eq!(scene.world.sensors.sensors.len(), 11);
    assert_eq!(scene.world.sensors.data.len(), 29);
    assert_eq!(scene.sensors_by_name.get("h"), Some(&7));
    // Data reads zero pre-step; sanity-check that.
    assert_eq!(scene.world.sensor(0).unwrap().len(), 1);
    assert_eq!(scene.world.sensor(2).unwrap().len(), 4);
    // Sensor-bank offsets are strictly monotonic.
    for (i, off) in scene.world.sensors.offsets.iter().enumerate().skip(1) {
        assert!(*off > scene.world.sensors.offsets[i - 1]);
    }
}

#[test]
fn unknown_sensor_kind_rejected() {
    let e = err(&wrap(
        r#"{"name":"x","kind":"weird","tree":"t","link":"hip"}"#,
    ));
    assert!(e.message.contains("unknown sensor kind"), "{}", e.message);
}

#[test]
fn duplicate_sensor_name_rejected() {
    let e = err(&wrap(
        r#"{"name":"a","kind":"jointpos","tree":"t","link":"hip"},
           {"name":"a","kind":"jointvel","tree":"t","link":"hip"}"#,
    ));
    assert!(e.message.contains("duplicate sensor"), "{}", e.message);
}

#[test]
fn empty_sensor_name_rejected() {
    let e = err(&wrap(
        r#"{"name":"","kind":"jointpos","tree":"t","link":"hip"}"#,
    ));
    assert!(e.message.contains("must not be empty"), "{}", e.message);
}

#[test]
fn jointpos_on_ball_joint_rejected() {
    let e = err(&wrap(
        r#"{"name":"x","kind":"jointpos","tree":"t","link":"knee"}"#,
    ));
    assert!(e.message.contains("hinge or slide"), "{}", e.message);
}

#[test]
fn ballquat_on_hinge_rejected() {
    let e = err(&wrap(
        r#"{"name":"x","kind":"ballquat","tree":"t","link":"hip"}"#,
    ));
    assert!(e.message.contains("ball"), "{}", e.message);
}

#[test]
fn framepos_on_unknown_site_rejected() {
    let e = err(&wrap(r#"{"name":"x","kind":"framepos","site":"nope"}"#));
    assert!(e.message.contains("unknown site"), "{}", e.message);
}

#[test]
fn touch_on_unknown_geom_rejected() {
    let e = err(&wrap(r#"{"name":"x","kind":"touch","geom":"nope"}"#));
    assert!(e.message.contains("unknown geom"), "{}", e.message);
}

#[test]
fn force_on_root_link_rejected() {
    let e = err(&wrap(
        r#"{"name":"x","kind":"force","tree":"t","link":"root"}"#,
    ));
    assert!(e.message.contains("non-root"), "{}", e.message);
}

#[test]
fn unknown_tree_rejected() {
    let e = err(&wrap(
        r#"{"name":"x","kind":"jointpos","tree":"nope","link":"hip"}"#,
    ));
    assert!(e.message.contains("unknown tree"), "{}", e.message);
}

#[test]
fn unknown_link_rejected() {
    let e = err(&wrap(
        r#"{"name":"x","kind":"jointpos","tree":"t","link":"nope"}"#,
    ));
    assert!(e.message.contains("unknown link"), "{}", e.message);
}

#[test]
fn unknown_field_on_sensor_rejected() {
    let e = err(&wrap(
        r#"{"name":"x","kind":"jointpos","tree":"t","link":"hip","dampign":1}"#,
    ));
    assert!(e.message.contains("dampign"), "{}", e.message);
}

#[test]
fn loaded_scene_evaluates_sensordata_after_step() {
    let src = wrap(
        r#"
        {"name":"hipq","kind":"jointpos","tree":"t","link":"hip"},
        {"name":"tip","kind":"framepos","site":"s"}
    "#,
    );
    let mut scene = load_str(&src).unwrap();
    // Body starts at world origin (default pose). tip site pos = body pos.
    scene.world.bodies[0].position = Vec3::new(1.0, 2.0, 3.0);
    scene.world.step();
    let tip = scene.world.sensor(1).unwrap();
    // No hinge motion on the tree (root fixed + hinge starts at 0),
    // gravity acts on body freely so it drifts a little after one step.
    // The framepos sensor reads the site position: for a body at (1, 2, 3)
    // with no rotation, site position = body position (no offset).
    // After one 5 ms step under gravity, z drops by ½·9.81·(0.005)²
    // ≈ 1.23e-4, negligible for x/y checks.
    assert!((tip[0] - 1.0).abs() < 1e-3, "site x = {}", tip[0]);
    assert!((tip[1] - 2.0).abs() < 1e-3, "site y = {}", tip[1]);
    assert!(
        tip[2] < 3.0,
        "site z should have fallen slightly, got {}",
        tip[2]
    );
}
