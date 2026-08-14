//! Sanity: every differential MJCF loads under newt too.
//!
//! This lives outside the main `differential.rs` suite so the load
//! surface has a fast failure signal even before fixture reading.

use std::fs;

#[test]
fn every_differential_scenario_loads() {
    let names = [
        "ballistic",
        "tumble",
        "double_pendulum",
        "servo_arm",
        "sphere_drop",
        "box_stack",
        "joint_limit_swing",
    ];
    for name in names {
        let path = format!("tests/references/{name}.xml");
        let src =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("missing scenario {name}: {e}"));
        let scene = newt::mjcf::load_mjcf_str(&src)
            .unwrap_or_else(|e| panic!("newt cannot load {name}: {e}"));
        let n_free: usize = scene.world.bodies.len();
        let n_trees: usize = scene.world.trees.len();
        let tree_nq: usize = scene.world.trees.iter().map(|t| t.nq()).sum();
        let tree_nv: usize = scene.world.trees.iter().map(|t| t.nv()).sum();
        println!(
            "load {name}: free_bodies={n_free} trees={n_trees} tree_nq={tree_nq} tree_nv={tree_nv} geoms={} actuators={}",
            scene.world.geoms.len(),
            scene.actuators_by_name.len()
        );
        assert!(
            !scene.world.bodies.is_empty() || !scene.world.trees.is_empty(),
            "{name} has no bodies or trees"
        );
    }
}
