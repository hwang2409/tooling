//! MJCF loader integration tests: rejection paths, MJCF-specific
//! attributes (defaults, classes, degrees, fromto), and the biped-simple
//! load-and-stability smoke (v1 tier 7 finale).
//!
//! Byte-identical trajectory anchors against the JSON mirror fixtures
//! live in `tests/mjcf_anchor.rs`.

use newt::mjcf::{load_mjcf_path, load_mjcf_str};
use newt::tree::forward_kinematics;

// ---------------------------------------------------------------------------
// rejection paths — every "no silent ignore" error surface
// ---------------------------------------------------------------------------

fn expect_err(src: &str, needle: &str) {
    let e = load_mjcf_str(src).expect_err("expected error");
    assert!(
        e.message.contains(needle),
        "expected error containing {needle:?}, got: {} ({})",
        e.message,
        e.path
    );
}

#[test]
fn asset_top_level_rejected() {
    expect_err(
        r#"<mujoco><asset><mesh name="m" file="f.stl"/></asset></mujoco>"#,
        "<asset>",
    );
}

#[test]
fn tendon_top_level_rejected() {
    expect_err(r#"<mujoco><tendon/></mujoco>"#, "<tendon>");
}

#[test]
fn keyframe_top_level_rejected() {
    expect_err(r#"<mujoco><keyframe/></mujoco>"#, "<keyframe>");
}

#[test]
fn compiler_coordinate_global_rejected() {
    expect_err(
        r#"<mujoco><compiler coordinate="global"/></mujoco>"#,
        "coordinate",
    );
}

#[test]
fn compiler_eulerseq_zyx_rejected() {
    expect_err(r#"<mujoco><compiler eulerseq="zyx"/></mujoco>"#, "eulerseq");
}

#[test]
fn geom_type_mesh_rejected_cleanly() {
    expect_err(
        r#"<mujoco><worldbody><geom name="g" type="mesh"/></worldbody></mujoco>"#,
        "mesh",
    );
}

#[test]
fn unknown_body_attribute_via_material_rejected() {
    // MJCF has a `material` attribute on geoms (rendering); we reject it.
    expect_err(
        r#"<mujoco><worldbody>
             <geom name="g" type="plane" material="whatever"/>
           </worldbody></mujoco>"#,
        "material",
    );
}

#[test]
fn actuator_missing_forcerange_ok_but_missing_joint_errors() {
    expect_err(
        r#"<mujoco>
             <worldbody>
               <body name="root">
                 <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                 <body name="c" pos="0 0 -0.5">
                   <joint name="j" type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                   <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                 </body>
               </body>
             </worldbody>
             <actuator>
               <position name="a" joint="nope" kp="10" kv="1"/>
             </actuator>
           </mujoco>"#,
        "unknown joint",
    );
}

#[test]
fn duplicate_joint_name_rejected() {
    expect_err(
        r#"<mujoco><worldbody>
             <body name="root">
               <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
               <body name="a" pos="0 0 -0.5">
                 <joint name="j" type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                 <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                 <body name="b" pos="0 0 -0.5">
                   <joint name="j" type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                   <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                 </body>
               </body>
             </body>
           </worldbody></mujoco>"#,
        "duplicate joint",
    );
}

#[test]
fn inertial_pos_nonzero_rejected() {
    expect_err(
        r#"<mujoco><worldbody>
             <body name="root">
               <inertial pos="0.1 0 0" mass="1" diaginertia="1 1 1"/>
             </body>
           </worldbody></mujoco>"#,
        "COM",
    );
}

#[test]
fn contact_pair_and_exclude_together_rejected() {
    expect_err(
        r#"<mujoco>
             <worldbody>
               <geom name="a" type="plane"/>
               <geom name="b" type="plane"/>
             </worldbody>
             <contact>
               <pair geom1="a" geom2="b"/>
               <exclude body1="a" body2="b"/>
             </contact>
           </mujoco>"#,
        "pair",
    );
}

#[test]
fn joint_ref_nonzero_rejected() {
    expect_err(
        r#"<mujoco><worldbody>
             <body name="root">
               <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
               <body name="c" pos="0 0 -0.5">
                 <joint type="hinge" axis="1 0 0" pos="0 0 0.5" ref="0.3"/>
                 <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
               </body>
             </body>
           </worldbody></mujoco>"#,
        "ref",
    );
}

// ---------------------------------------------------------------------------
// hand-derived expectations for defaults + degrees + fromto
// ---------------------------------------------------------------------------

#[test]
fn nested_default_and_childclass_resolution() {
    // Class layers:
    //   main (top-level): joint damping=1.0, geom friction=0.2
    //     strong: joint damping=5.0  (inherits nothing else because we
    //             only override the joint block; friction stays 0.2 for
    //             geoms under the strong class)
    // Body B has childclass="strong" so its own joint picks up damping=5.0.
    // Body C explicitly uses class="main" on its joint, so it stays at 1.0.
    let scene = load_mjcf_str(
        r#"<mujoco>
             <default>
               <joint damping="1.0"/>
               <geom friction="0.2"/>
               <default class="strong">
                 <joint damping="5.0"/>
               </default>
             </default>
             <worldbody>
               <body name="root">
                 <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                 <body name="b" pos="0 0 -0.5" childclass="strong">
                   <joint name="bj" type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                   <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   <body name="c" pos="0 0 -0.5">
                     <joint name="cj" type="hinge" axis="1 0 0" pos="0 0 0.5" class="main"/>
                     <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                   </body>
                 </body>
               </body>
             </worldbody>
           </mujoco>"#,
    )
    .expect("scene should load");
    use newt::joint::JointKind;
    let tree = &scene.world.trees[0];
    let idx_b = scene.links_by_name[0]["b"];
    let idx_c = scene.links_by_name[0]["c"];
    if let JointKind::Hinge { damping, .. } = tree.links[idx_b].joint {
        assert!((damping - 5.0).abs() < 1e-6, "b damping = {damping}");
    } else {
        panic!("expected hinge on b");
    }
    if let JointKind::Hinge { damping, .. } = tree.links[idx_c].joint {
        assert!((damping - 1.0).abs() < 1e-6, "c damping = {damping}");
    } else {
        panic!("expected hinge on c");
    }
}

#[test]
fn degrees_convert_range_and_euler_at_root() {
    // Root body specifies a 45° euler about z; range specified in degrees.
    let scene = load_mjcf_str(
        r#"<mujoco>
             <compiler angle="degree"/>
             <worldbody>
               <body name="anchor" euler="0 0 45">
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
    // Anchor's world orientation: euler xyz 0,0,45° → quaternion (0, 0, sin(22.5°), cos(22.5°)).
    let anchor = scene.trees_by_name["anchor"];
    let (_, q) = forward_kinematics(&scene.world.trees[anchor])[0];
    let sin22p5 = (std::f32::consts::FRAC_PI_4 * 0.5).sin();
    let cos22p5 = (std::f32::consts::FRAC_PI_4 * 0.5).cos();
    assert!(q.x.abs() < 1e-5);
    assert!(q.y.abs() < 1e-5);
    assert!((q.z - sin22p5).abs() < 1e-4, "q.z = {}", q.z);
    assert!((q.w - cos22p5).abs() < 1e-4, "q.w = {}", q.w);

    use newt::joint::JointKind;
    let tree = &scene.world.trees[anchor];
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
fn fromto_capsule_hand_computed() {
    // fromto="0 0 0.5 0 0 -0.5" for a capsule of size 0.05 (radius) →
    // half-height = |b-a|/2 = 0.5, center = midpoint = (0,0,0), axis
    // along local +Z (down-then-up doesn't matter; the orientation
    // aligns +Z with (b-a)/|b-a| = (0,0,-1) which is a 180° flip →
    // orientation quat (1, 0, 0, 0) per quat_align_z_to's antipode case).
    let scene = load_mjcf_str(
        r#"<mujoco><worldbody>
             <body name="root" pos="0 0 1">
               <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
               <freejoint/>
               <geom name="cap" type="capsule" fromto="0 0 0.5 0 0 -0.5" size="0.05"/>
             </body>
           </worldbody></mujoco>"#,
    )
    .expect("scene should load");
    use newt::geom::GeomShape;
    let g = &scene.world.geoms[scene.geoms_by_name["cap"]];
    if let GeomShape::Capsule {
        radius,
        half_height,
    } = g.shape
    {
        assert!((radius - 0.05).abs() < 1e-6);
        assert!((half_height - 0.5).abs() < 1e-6);
    } else {
        panic!("expected capsule");
    }
    assert_eq!(g.local_offset, newt::math::Vec3::ZERO);
    // 180° flip about x-axis: (x=1, y=0, z=0, w=0).
    assert!((g.local_orientation.x - 1.0).abs() < 1e-5);
    assert!(g.local_orientation.w.abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// biped-simple: load, step 2000, root height stays bounded (v1 finale)
// ---------------------------------------------------------------------------

#[test]
fn biped_simple_loads_and_stands_for_2000_steps() {
    let base = std::env::current_dir().unwrap();
    let scene = load_mjcf_path(base.join("models/biped-simple.xml")).unwrap();
    // Structural sanity — one tree, 11 links (torso + 10 leg segments),
    // 10 hinge actuators.
    assert_eq!(scene.world.trees.len(), 1);
    assert_eq!(scene.world.trees[0].links.len(), 11);
    assert_eq!(scene.actuators_by_name.len(), 10);

    let mut world = scene.world.clone();
    let initial_root_z = forward_kinematics(&world.trees[0])[0].0.z;
    assert!(
        (initial_root_z - 1.30).abs() < 1e-6,
        "initial root z = {initial_root_z}"
    );

    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for _ in 0..2000 {
        world.step();
        let root = forward_kinematics(&world.trees[0])[0].0;
        assert!(
            root.x.is_finite() && root.y.is_finite() && root.z.is_finite(),
            "root pose went non-finite: {:?}",
            root
        );
        if root.z < min_z {
            min_z = root.z;
        }
        if root.z > max_z {
            max_z = root.z;
        }
    }
    // Root height stays bounded — the biped squats under load but must
    // not sink through the ground or fly up. Lateral drift is not
    // constrained: this is a load-and-stability smoke, not a walking
    // controller (the ticket calls out that walking is v2).
    assert!(min_z > 0.0, "root sank through the ground: min_z = {min_z}");
    assert!(max_z < 3.0, "root shot up: max_z = {max_z}");
}
