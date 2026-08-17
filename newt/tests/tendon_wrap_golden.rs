//! Byte golden for a symmetry-broken cylinder-wrap tendon actuator.

use std::fs;

const GOLDEN_PATH: &str = "tests/goldens/tendon_wrap_cylinder.bin";

fn trajectory() -> Vec<u8> {
    let src = fs::read_to_string("tests/references/tendon_cylinder_lift.xml").unwrap();
    let mut scene = newt::mjcf::load_mjcf_str(&src).unwrap();
    let (tree_idx, actuator_idx) = scene.actuators_by_name["lift_motor"];
    scene.world.trees[tree_idx].set_actuator_target(actuator_idx, 2.5);
    let mut bytes = Vec::new();
    for step in 0..=240 {
        if step % 60 == 0 {
            let tree = &scene.world.trees[0];
            for value in &tree.q {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in &tree.qdot {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let poses = newt::tree::forward_kinematics(tree);
            let length = newt::tendon::tendon_kinematics(&tree.tendons[0], tree, &poses).length;
            bytes.extend_from_slice(&length.to_le_bytes());
        }
        scene.world.step();
    }
    bytes
}

#[test]
fn cylinder_wrap_golden_is_byte_identical() {
    let expected = fs::read(GOLDEN_PATH).unwrap_or_else(|error| {
        panic!("golden file missing: {error}; run ignored regenerate_tendon_wrap_golden")
    });
    let actual = trajectory();
    assert_eq!(actual, expected, "cylinder wrap golden changed");
}

#[test]
#[ignore]
fn regenerate_tendon_wrap_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!("golden regeneration is guarded to macOS aarch64");
    }
    fs::write(GOLDEN_PATH, trajectory()).unwrap();
}
