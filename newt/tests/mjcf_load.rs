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
fn tendon_unknown_child_rejected() {
    // <tendon> is supported (v2 tier 3); its unknown children are not.
    expect_err(
        r#"<mujoco><tendon><flex name="x"/></tendon></mujoco>"#,
        "<flex>",
    );
}

#[test]
fn keyframe_top_level_accepted() {
    let scene = newt::mjcf::load_mjcf_str(r#"<mujoco><keyframe/></mujoco>"#).unwrap();
    assert!(scene.world.keyframes.is_empty());
}

#[test]
fn keyframe_and_mocap_load_from_mjcf() {
    let scene = newt::mjcf::load_mjcf_str(
        r#"<mujoco><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="hinge"><joint name="h" axis="0 0 1"/>
              <inertial mass="1" diaginertia="1 1 1"/>
            </body>
          </body>
        </worldbody><keyframe><key name="ready" qpos="0.2" qvel="-0.1"/></keyframe></mujoco>"#,
    )
    .unwrap();
    assert_eq!(scene.world.keyframes[0].q, vec![0.2]);
    assert_eq!(scene.world.keyframes[0].qdot, vec![-0.1]);
}

#[test]
fn mocap_movable_descendant_rejected_from_mjcf() {
    expect_err(
        r#"<mujoco><worldbody>
          <body name="root" mocap="true"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="hinge"><joint name="h" axis="0 0 1"/>
              <inertial mass="1" diaginertia="1 1 1"/>
            </body>
          </body>
        </worldbody></mujoco>"#,
        "mocap root cannot have movable descendants",
    );
}

#[test]
fn keyframe_remaps_free_root_qpos_and_qvel() {
    let scene = load_mjcf_str(
        r#"<mujoco><worldbody>
          <body name="root"><freejoint/><inertial mass="1" diaginertia="1 1 1"/>
            <body name="fixed"><inertial mass="1" diaginertia="1 1 1"/></body>
          </body>
        </worldbody><keyframe>
          <key name="pose" qpos="0 0 0 1 0 0 0" qvel="1 2 3 4 5 6"/>
        </keyframe></mujoco>"#,
    )
    .unwrap();
    assert_eq!(
        scene.world.keyframes[0].q,
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        scene.world.keyframes[0].qdot,
        vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn keyframe_remaps_ball_quaternion_order() {
    let scene = load_mjcf_str(
        r#"<mujoco><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="ball"><joint name="b" type="ball"/>
              <inertial mass="1" diaginertia="1 1 1"/>
            </body>
          </body>
        </worldbody><keyframe>
          <key name="pose" qpos="1 2 3 4" qvel="0.1 0.2 0.3"/>
        </keyframe></mujoco>"#,
    )
    .unwrap();
    let norm = (30.0_f32).sqrt();
    let q = &scene.world.keyframes[0].q;
    assert_eq!(q.len(), 4);
    assert!((q[0] - 2.0 / norm).abs() < 1e-6);
    assert!((q[1] - 3.0 / norm).abs() < 1e-6);
    assert!((q[2] - 4.0 / norm).abs() < 1e-6);
    assert!((q[3] - 1.0 / norm).abs() < 1e-6);
    assert_eq!(scene.world.keyframes[0].qdot, vec![0.1, 0.2, 0.3]);
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
fn body_bogus_attribute_rejected() {
    // Every other element already whitelists its attributes — <body> now
    // does too. Reviewer probe: an unknown attribute on <body> must
    // error with the attribute name, not silently load.
    expect_err(
        r#"<mujoco><worldbody>
             <body name="root" bogus_attr="1">
               <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
             </body>
           </worldbody></mujoco>"#,
        "bogus_attr",
    );
    // The same enforcement applies to nested bodies.
    expect_err(
        r#"<mujoco><worldbody>
             <body name="root">
               <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
               <body name="c" pos="0 0 -0.5" bogus_attr="1">
                 <joint type="hinge" axis="1 0 0" pos="0 0 0.5"/>
                 <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
               </body>
             </body>
           </worldbody></mujoco>"#,
        "bogus_attr",
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
// biped-simple: v1 finale. TWO SEPARATE TESTS — one records the pure-PD
// behavior (the biped falls, matching the source biped's behavior when
// balance assist is off — this is the honest smoke), and one asserts
// standing WITH the source biped's balance assist explicitly applied.
// ---------------------------------------------------------------------------

/// Faithful mirror of the source biped's torso balance controller —
/// `~/me/fun/biped/biped/mujoco_biped.py::_apply_balance_controller`
/// lines 2570-2593, at `assist_scale=1.0` (the setting the source
/// `stand` scenario ships with). All six wrench components, all
/// source constants, all source clamps, expressed as the source
/// intends (world-frame linear force + world-frame torque on the
/// torso body via `Tree::applied_wrenches[0]`, which newt sums into
/// the ABA external-wrench path exactly like `data.xfrc_applied`).
///
/// **Frame conversion.** The source reads `data.qvel[0..6]` — MuJoCo
/// freejoint qvel is world-frame `(vx, vy, vz, ωx, ωy, ωz)`. newt's
/// free-root `Tree::qdot` layout is `(ω_body, v_body)`, so the
/// components need both a slot re-order AND a body→world rotation
/// via `torso_ori.rotate(...)` before feeding the PDs.
///
/// This is explicit external stabilization on the torso — NOT joint
/// PD. Callers that want a pure-PD run must not invoke it.
///
/// Source formulas (target_x = 0, target_speed = 0 for stand):
/// ```text
///   force_x  = clamp(42 * -x + 82 * -vx,  -75,  75)
///   force_y  = clamp(90 * -y  - 35 *  vy, -35,  35)
///   force_z  = clamp(240 * (target_z - z) - 70 * vz, -90, 260)
///   torque_x = clamp( 135 * up_y - 24 * ωx,          -95,  95)
///   torque_y = clamp(-135 * up_x - 24 * ωy,          -95,  95)
///   torque_z = clamp(-12 * ωz,                       -28,  28)
/// ```
fn apply_source_balance_wrench(world: &mut newt::world::World, target_z: f32) {
    let (torso_pos, torso_ori) = forward_kinematics(&world.trees[0])[0];
    let up_world = torso_ori.rotate(newt::math::Vec3::new(0.0, 0.0, 1.0));
    // newt free-root qdot: [ωx_body, ωy_body, ωz_body, vx_body, vy_body, vz_body].
    let omega_body = newt::math::Vec3::new(
        world.trees[0].qdot[0],
        world.trees[0].qdot[1],
        world.trees[0].qdot[2],
    );
    let v_body = newt::math::Vec3::new(
        world.trees[0].qdot[3],
        world.trees[0].qdot[4],
        world.trees[0].qdot[5],
    );
    let omega_world = torso_ori.rotate(omega_body);
    let v_world = torso_ori.rotate(v_body);
    let force_x = (42.0 * -torso_pos.x + 82.0 * -v_world.x).clamp(-75.0, 75.0);
    let force_y = (90.0 * -torso_pos.y - 35.0 * v_world.y).clamp(-35.0, 35.0);
    let force_z = (240.0 * (target_z - torso_pos.z) - 70.0 * v_world.z).clamp(-90.0, 260.0);
    let torque_x = (135.0 * up_world.y - 24.0 * omega_world.x).clamp(-95.0, 95.0);
    let torque_y = (-135.0 * up_world.x - 24.0 * omega_world.y).clamp(-95.0, 95.0);
    let torque_z = (-12.0 * omega_world.z).clamp(-28.0, 28.0);
    world.trees[0].applied_wrenches[0] = (
        newt::math::Vec3::new(force_x, force_y, force_z),
        newt::math::Vec3::new(torque_x, torque_y, torque_z),
    );
}

/// Load-and-stability smoke under PURE joint PD — NO external assist.
/// The biped is expected to fall: joint PD alone with source-range
/// gains (`kp ∈ [45, 80]`) cannot stabilize the inverted-pendulum
/// dynamics of a ~20 kg body above ~0.9 m ankles (tipping moment
/// ≈ 177 θ Nm/rad, total ankle stiffness ≤ 2·80 = 160 Nm/rad). The
/// source biped's own `stand` scenario ships with `balance_mode:
/// "controller"` + `assist_scale: 1.0` for exactly this reason. This
/// test asserts only that the simulation stays finite and stays in a
/// generous world-sized box — a clean load-and-step smoke — and
/// documents the falling behavior as the current pure-PD baseline
/// pending the NEWT-13 differential-vs-MuJoCo work.
#[test]
fn biped_simple_pure_pd_smoke() {
    let base = std::env::current_dir().unwrap();
    let scene = load_mjcf_path(base.join("models/biped-simple.xml")).unwrap();
    assert_eq!(scene.world.trees.len(), 1);
    assert_eq!(scene.world.trees[0].links.len(), 11);
    assert_eq!(scene.actuators_by_name.len(), 10);

    let mut world = scene.world.clone();
    let initial_root_z = forward_kinematics(&world.trees[0])[0].0.z;
    assert!(
        (initial_root_z - 1.235).abs() < 1e-4,
        "initial root z = {initial_root_z}"
    );

    let mut min_ratio = 1.0f32;
    let mut fell_step: Option<usize> = None;
    for step in 0..2000 {
        // NO balance wrench — pure joint PD only.
        world.step();
        let root = forward_kinematics(&world.trees[0])[0].0;
        assert!(
            root.x.is_finite() && root.y.is_finite() && root.z.is_finite(),
            "root pose went non-finite at step {step}: {:?}",
            root
        );
        // Bounded box — the simulation must not explode. Root does drift
        // during the fall (see `docs/mjcf.md#biped-simple-standing-note`
        // for the recorded trajectory) but stays inside a reasonable
        // world-scale envelope.
        assert!(
            root.x.abs() < 10.0 && root.y.abs() < 10.0 && root.z > -1.0 && root.z < 5.0,
            "root escaped the world box at step {step}: {:?}",
            root
        );
        let ratio = root.z / initial_root_z;
        if ratio < min_ratio {
            min_ratio = ratio;
        }
        if fell_step.is_none() && ratio < 0.5 {
            fell_step = Some(step + 1);
        }
    }
    // Documented current behavior: under pure PD the biped falls
    // between steps ~850 and ~1200 and lies flat afterwards. The
    // window reflects NEWT-22 tree-contact routing plus NEWT-23's
    // exact MuJoCo reference acceleration. If any
    // future change lifts pure-PD standing above 50% of initial
    // height, this assertion catches it so we can promote the
    // fixture / test accordingly.
    let fell_step = fell_step.expect(
        "biped stayed above 50% of initial height under pure PD — \
         update the smoke and docs; a pure-PD standing biped is a \
         significant behavior change worth surfacing.",
    );
    assert!(
        (850..=1200).contains(&fell_step),
        "biped fell at step {fell_step} — outside the recorded [850, 1200] \
         window (see docs/mjcf.md#biped-simple-standing-note)",
    );
    assert!(
        min_ratio < 0.20,
        "biped only dropped to {:.1}% of initial — recorded floor is ~11%",
        100.0 * min_ratio,
    );
}

/// The reference "standing" test — biped holds a quiet upright pose
/// WHEN the source biped's balance controller is applied each step.
/// This is NOT joint PD alone: `apply_source_balance_wrench` writes
/// an external torso wrench mirroring the same
/// `_apply_balance_controller` the source biped's `stand` scenario
/// runs at `assist_scale=1.0` (see
/// `~/me/fun/biped/biped/mujoco_biped.py::SCENARIO_DEFINITIONS["stand"]`
/// and `_apply_balance_controller`). Standing here reproduces the
/// source `stand` scenario's own behavior — joint PD + balance
/// controller together, which is the source's own architecture for a
/// held stand.
#[test]
fn biped_simple_stands_with_source_balance_assist() {
    let base = std::env::current_dir().unwrap();
    let scene = load_mjcf_path(base.join("models/biped-simple.xml")).unwrap();
    let mut world = scene.world.clone();
    let initial_root_z = forward_kinematics(&world.trees[0])[0].0.z;

    let mut min_ratio = 1.0f32;
    let mut max_tilt: f32 = 0.0;
    for step in 0..2000 {
        // DISCLOSED external assist — see fn docstring.
        apply_source_balance_wrench(&mut world, initial_root_z);
        world.step();
        let (root, ori) = forward_kinematics(&world.trees[0])[0];
        assert!(
            root.x.is_finite() && root.y.is_finite() && root.z.is_finite(),
            "root pose went non-finite at step {step}: {:?}",
            root
        );
        let ratio = root.z / initial_root_z;
        if ratio < min_ratio {
            min_ratio = ratio;
        }
        let up_world = ori.rotate(newt::math::Vec3::new(0.0, 0.0, 1.0));
        let tilt = (1.0 - up_world.z * up_world.z).max(0.0).sqrt();
        if tilt > max_tilt {
            max_tilt = tilt;
        }
    }
    // With the source balance controller running the biped stays
    // above 80% of initial standing height and torso tilt stays
    // under ~15° for every one of the 2000 steps (10 s at dt=5 ms).
    assert!(
        min_ratio > 0.80,
        "root height dropped to {:.1}% of initial (assist ON — \
         should stay > 80%)",
        100.0 * min_ratio,
    );
    assert!(
        max_tilt < 0.26,
        "torso tilted to {:.3} rad (~{:.1} deg) — assist ON should \
         hold tilt < 15°",
        max_tilt,
        max_tilt.asin().to_degrees(),
    );
}
