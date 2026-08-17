//! Sanity-check that the differential-scenario MJCF fixtures also load in
//! newt (they must — both engines see the SAME file at capture time and
//! comparison time).

#[test]
fn tendon_coupled_mjcf_loads() {
    let src = std::fs::read_to_string("tests/references/tendon_coupled.xml").unwrap();
    let scene = newt::mjcf::load_mjcf_str(&src).expect("newt loads tendon_coupled");
    assert!(scene.tendons_by_name.contains_key("coup"));
}

#[test]
fn tendon_wrap_mjcf_loads() {
    let src = std::fs::read_to_string("tests/references/tendon_wrap.xml").unwrap();
    let scene = newt::mjcf::load_mjcf_str(&src).expect("newt loads tendon_wrap");
    assert!(scene.tendons_by_name.contains_key("cable"));
}

#[test]
fn mjcf_accepts_moving_wrap_for_envelope_jacobian() {
    let src = r#"
        <mujoco model="moving_wrap">
          <worldbody>
            <body name="root">
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <site name="a" pos="-1 0 0"/>
              <body name="moving">
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom name="wrap" type="sphere" size="0.1"/>
                <site name="b" pos="1 0 0"/>
              </body>
            </body>
          </worldbody>
          <tendon>
            <spatial name="cable">
              <site site="a"/>
              <geom geom="wrap"/>
              <site site="b"/>
            </spatial>
          </tendon>
        </mujoco>
    "#;
    let scene = newt::mjcf::load_mjcf_str(src).expect("moving wrap is supported");
    assert_eq!(scene.world.trees[0].tendons.len(), 1);
}

#[test]
fn mjcf_rejects_leading_wrap_geom() {
    let src = r#"
        <mujoco>
          <worldbody>
            <body name="root">
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <geom name="wrap" type="cylinder" size="0.2 1"/>
              <site name="a" pos="-1 0 0"/>
              <site name="b" pos="1 0 0"/>
            </body>
          </worldbody>
          <tendon><spatial name="c"><geom geom="wrap"/><site site="a"/><site site="b"/></spatial></tendon>
        </mujoco>
    "#;
    let error = newt::mjcf::load_mjcf_str(src).unwrap_err();
    assert!(error.message.contains("must follow a site"), "{error}");
}

#[test]
fn mjcf_rejects_invalid_pulley_and_wrap_configs() {
    let zero_divisor = r#"
        <mujoco><worldbody><body name="root"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><site name="a"/><site name="b"/></body></worldbody>
        <tendon><spatial name="c"><site site="a"/><site site="b"/><pulley divisor="0"/><site site="a"/><site site="b"/></spatial></tendon></mujoco>
    "#;
    let error = newt::mjcf::load_mjcf_str(zero_divisor).unwrap_err();
    assert!(error.message.contains("divisor must be > 0"), "{error}");

    let bad_wrap = r#"
        <mujoco><worldbody><body name="root"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><site name="a"/><site name="b"/><geom name="box" type="box" size="1 1 1"/></body></worldbody>
        <tendon><spatial name="c"><site site="a"/><geom geom="box"/><site site="b"/></spatial></tendon></mujoco>
    "#;
    let error = newt::mjcf::load_mjcf_str(bad_wrap).unwrap_err();
    assert!(
        error.message.contains("not a sphere or cylinder"),
        "{error}"
    );
}

#[test]
fn mjcf_loads_asymmetric_pulley_branches_and_divisors() {
    let src = std::fs::read_to_string("tests/references/tendon_pulley_2to1.xml").unwrap();
    let scene = newt::mjcf::load_mjcf_str(&src).expect("pulley fixture loads");
    let tendon = &scene.world.trees[0].tendons[0];
    match &tendon.kind {
        newt::tendon::TendonKind::Spatial { branches } => {
            assert_eq!(branches.len(), 2);
            assert_eq!(branches[0].divisor, 1.0);
            assert_eq!(branches[1].divisor, 2.0);
            assert_ne!(
                branches[0].sites[0].position_local,
                branches[1].sites[0].position_local
            );
        }
        other => panic!("expected spatial tendon, got {other:?}"),
    }
}

#[test]
fn mjcf_accepts_fixed_cross_tree_sidesite() {
    let src = r#"
        <mujoco>
          <worldbody>
            <body name="tendon_root">
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <site name="a" pos="-2 0 0"/>
              <site name="b" pos="2 0 0"/>
              <geom name="cylinder" type="cylinder" size="0.5 1"/>
            </body>
            <body name="side_root" pos="0 1 0">
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <site name="side" pos="0 0 0"/>
            </body>
          </worldbody>
          <tendon><spatial name="c"><site site="a"/><geom geom="cylinder" sidesite="side"/><site site="b"/></spatial></tendon>
        </mujoco>
    "#;
    let scene = newt::mjcf::load_mjcf_str(src).expect("fixed cross-tree sidesite loads");
    assert_eq!(scene.world.trees.len(), 2);
}
