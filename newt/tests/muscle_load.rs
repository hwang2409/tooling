use newt::actuator::{ActuatorFlavor, DynType};
use newt::mjcf::load_mjcf_str;
use newt::model::load_str;

const MJCF: &str = r#"
<mujoco>
  <worldbody>
    <body name="root">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body name="hinge_body">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="hinge" type="hinge" axis="0 1 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <muscle name="joint_muscle" joint="hinge" lengthrange="0 1"/>
    <general name="general_muscle" joint="hinge" gaintype="muscle"
             biastype="muscle" dyntype="muscle" lengthrange="0 1"
             gainprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2"
             biasprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2"
             dynprm="0.02 0.05 0.1"/>
  </actuator>
</mujoco>
"#;

#[test]
fn mjcf_muscle_shortcut_and_general_round_trip() {
    let scene = load_mjcf_str(MJCF).expect("muscle MJCF should load");
    let tree = &scene.world.trees[0];
    assert_eq!(tree.actuators.len(), 2);
    assert!(matches!(
        tree.actuators[0].flavor,
        ActuatorFlavor::Muscle { .. }
    ));
    assert!(matches!(
        tree.actuators[1].flavor,
        ActuatorFlavor::Muscle { .. }
    ));
    assert_eq!(tree.actuators[1].muscle_dyn_prm, [0.02, 0.05, 0.1]);
    assert_eq!(tree.actuators[0].ctrl_range, Some((0.0, 1.0)));
    assert_eq!(tree.actuators[0].dyn_type, DynType::Muscle);
}

#[test]
fn json_muscle_shortcut_and_general_load() {
    let source = r#"
    {
      "trees": [{
        "name": "tree",
        "links": [
          {"name":"root", "joint":{"kind":"fixed"},
           "mass":1, "inertia":{"kind":"diag","values":[1,1,1]}},
          {"name":"hinge", "parent":"root",
           "joint":{"kind":"hinge","axis":[0,1,0]},
           "mass":1, "inertia":{"kind":"diag","values":[1,1,1]}}
        ]
      }],
      "actuators": [
        {"name":"m", "type":"muscle", "tree":"tree", "link":"hinge",
         "lengthrange":[0,1]},
        {"name":"g", "type":"general", "tree":"tree", "link":"hinge",
         "gaintype":"muscle", "biastype":"muscle", "dyntype":"muscle",
         "lengthrange":[0,1],
         "gainprm":[0.75,1.05,-1,200,0.5,1.6,1.5,1.3,1.2],
         "biasprm":[0.75,1.05,-1,200,0.5,1.6,1.5,1.3,1.2],
         "dynprm":[0.02,0.05,0.1]}
      ]
    }
    "#;
    let scene = load_str(source).expect("muscle JSON should load");
    let tree = &scene.world.trees[0];
    assert_eq!(tree.actuators.len(), 2);
    assert_eq!(tree.actuators[1].muscle_dyn_prm, [0.02, 0.05, 0.1]);
}

#[test]
fn muscle_lengthrange_is_required_loudly() {
    let source = MJCF.replace(" lengthrange=\"0 1\"", "");
    let error = load_mjcf_str(&source).expect_err("missing lengthrange must reject");
    assert!(error.message.contains("lengthrange"), "{error}");
}

#[test]
fn muscle_general_requires_gain_and_bias_parameters() {
    let missing_bias = MJCF.replace("biasprm=\"0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2\"", "");
    let error = load_mjcf_str(&missing_bias).expect_err("missing biasprm must reject");
    assert!(error.message.contains("biasprm"), "{error}");

    let missing_gain = r#"{
      "trees":[{"name":"tree","links":[
        {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
        {"name":"hinge","parent":"root","joint":{"kind":"hinge","axis":[0,1,0]},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
      ]}],
      "actuators":[{"name":"g","type":"general","tree":"tree","link":"hinge",
        "gaintype":"muscle","biastype":"muscle","dyntype":"muscle",
        "biasprm":[0.75,1.05,-1,200,0.5,1.6,1.5,1.3,1.2],"lengthrange":[0,1]}]
    }"#;
    let error = load_str(missing_gain).expect_err("missing gainprm must reject");
    assert!(error.to_string().contains("gainprm"), "{error}");
}

#[test]
fn muscle_general_dynprm_omission_matches_compiled_default() {
    let source = MJCF.replace("dynprm=\"0.02 0.05 0.1\"", "");
    let scene = load_mjcf_str(&source).expect("omitted dynprm should load");
    assert_eq!(
        scene.world.trees[0].actuators[1].muscle_dyn_prm,
        [1.0, 0.0, 0.0]
    );
}

#[test]
fn muscle_curve_domain_matches_mujoco() {
    let accepted = MJCF.replace(
        "<muscle name=\"joint_muscle\" joint=\"hinge\" lengthrange=\"0 1\"/>",
        "<muscle name=\"joint_muscle\" joint=\"hinge\" lengthrange=\"0 1\" lmin=\"0.5\" lmax=\"1.6\" fvmax=\"0.9\"/>",
    );
    load_mjcf_str(&accepted).expect("lmin<1<lmax and fvmax<1 should load");

    for (curve, message) in [
        ("lmin=\"1\" lmax=\"1.6\"", "lmin=1 must reject"),
        ("lmin=\"0.5\" lmax=\"1\"", "lmax=1 must reject"),
    ] {
        let source = MJCF.replace(
            "<muscle name=\"joint_muscle\" joint=\"hinge\" lengthrange=\"0 1\"/>",
            &format!("<muscle name=\"joint_muscle\" joint=\"hinge\" lengthrange=\"0 1\" {curve}/>"),
        );
        load_mjcf_str(&source).expect_err(message);
    }

    let json_source = |lmin: f32, lmax: f32, fvmax: f32| {
        format!(
            r#"{{"trees":[{{"name":"tree","links":[
              {{"name":"root","joint":{{"kind":"fixed"}},"mass":1,"inertia":{{"kind":"diag","values":[1,1,1]}}}},
              {{"name":"hinge","parent":"root","joint":{{"kind":"hinge","axis":[0,1,0]}},"mass":1,"inertia":{{"kind":"diag","values":[1,1,1]}}}}
            ]}}],"actuators":[{{"name":"m","type":"muscle","tree":"tree","link":"hinge",
              "lengthrange":[0,1],"lmin":{lmin},"lmax":{lmax},"fvmax":{fvmax}}}]}}"#
        )
    };
    load_str(&json_source(0.5, 1.6, 0.9)).expect("JSON fvmax<1 should load");
    load_str(&json_source(1.0, 1.6, 0.9)).expect_err("JSON lmin=1 must reject");
    load_str(&json_source(0.5, 1.0, 0.9)).expect_err("JSON lmax=1 must reject");
}

#[test]
fn gear_loaded_and_programmatic_muscles_match() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/references");
    let params = [0.75, 1.05, 1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2];
    for (file, length_range, expected_gear) in [
        ("muscle_pendulum_gear.xml", [-1.0, 1.0], 1.7),
        ("muscle_wrapped_tendon_gear.xml", [0.0, 2.0], 1.6),
    ] {
        let scene = newt::mjcf::load_mjcf_path(root.join(file)).expect("gear scene loads");
        let loaded = &scene.world.trees[0].actuators[0];
        let loaded_gear = match loaded.flavor {
            ActuatorFlavor::Muscle { gear, .. } => gear,
            other => panic!("expected muscle, got {other:?}"),
        };
        assert_eq!(loaded_gear, expected_gear);
        let mut programmatic = newt::actuator::Actuator::muscle(
            1,
            params,
            params,
            length_range,
            1.0,
            expected_gear,
            [0.01, 0.04, 0.0],
            None,
            None,
        );
        programmatic.act = 0.7;
        let mut loaded_copy = *loaded;
        loaded_copy.act = 0.7;
        assert!((loaded_copy.torque(0.4, -0.2) - programmatic.torque(0.4, -0.2)).abs() < 1e-5);
    }
}

#[test]
fn muscle_limit_flags_control_activation_and_force_clamps() {
    let source = MJCF.replace(
        "<muscle name=\"joint_muscle\" joint=\"hinge\" lengthrange=\"0 1\"/>",
        "<muscle name=\"joint_muscle\" joint=\"hinge\" lengthrange=\"0 1\" ctrlrange=\"0 0.5\" ctrllimited=\"true\" forcerange=\"-0.1 0.1\" forcelimited=\"false\"/>",
    );
    let mut scene = load_mjcf_str(&source).expect("limited muscle should load");
    let actuator = &mut scene.world.trees[0].actuators[0];
    assert!(actuator.ctrl_limited);
    assert!(!actuator.force_limited);
    actuator.ctrl = 1.0;
    assert_eq!(actuator.clamped_ctrl(), 0.5);
    actuator.integrate_activation(0.01);
    assert!((actuator.act - 1.0).abs() < 1e-6, "act={}", actuator.act);

    let raw = actuator.torque(0.5, 0.0);
    actuator.force_limited = true;
    let limited = actuator.torque(0.5, 0.0);
    assert!(raw.abs() > 0.1);
    assert!((limited.abs() - 0.1).abs() < 1e-6);
}
