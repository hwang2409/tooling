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
         "lengthrange":[0,1], "dynprm":[0.02,0.05,0.1]}
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
