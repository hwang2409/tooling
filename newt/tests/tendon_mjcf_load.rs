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
