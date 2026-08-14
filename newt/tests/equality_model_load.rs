//! Loader-level validation for the v1-tier-5 additions: `equality`
//! section, `condim` 4/6, and the torsional/rolling friction fields on
//! geoms.

use newt::model::{ModelError, load_str};

fn scene(src: &str) -> newt::model::Scene {
    load_str(src).unwrap_or_else(|e| panic!("expected scene, got: {e}"))
}

fn err(src: &str) -> ModelError {
    load_str(src).expect_err("expected error")
}

#[test]
fn condim_4_and_6_now_accepted() {
    for condim in [1, 3, 4, 6] {
        let src = format!(
            r#"{{
                "bodies":[{{"name":"b","mass":1,"inertia":{{"kind":"solid","shape":{{"kind":"sphere","radius":0.1}}}}}}],
                "geoms":[
                  {{"name":"g","shape":{{"kind":"sphere","radius":0.1}},"attach":{{"kind":"body","body":"b"}},"condim":{condim}}}
                ]
            }}"#
        );
        let _ = scene(&src);
    }
}

#[test]
fn condim_out_of_set_rejected() {
    let src = r#"{
        "bodies":[{"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}],
        "geoms":[
          {"name":"g","shape":{"kind":"sphere","radius":0.1},"attach":{"kind":"body","body":"b"},"condim":5}
        ]
    }"#;
    let e = err(src);
    assert!(
        e.message.contains("condim") && e.message.contains("5"),
        "message: {}",
        e.message
    );
}

#[test]
fn torsional_and_rolling_friction_load() {
    let src = r#"{
        "bodies":[{"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}],
        "geoms":[
          {"name":"g","shape":{"kind":"sphere","radius":0.1},"attach":{"kind":"body","body":"b"},
           "condim":6,"torsional_friction":0.5,"rolling_friction":0.1}
        ]
    }"#;
    let s = scene(src);
    assert_eq!(s.world.geoms[0].torsional_friction, 0.5);
    assert_eq!(s.world.geoms[0].rolling_friction, 0.1);
}

#[test]
fn negative_torsional_friction_rejected() {
    let src = r#"{
        "bodies":[{"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}],
        "geoms":[
          {"name":"g","shape":{"kind":"sphere","radius":0.1},"attach":{"kind":"body","body":"b"},"torsional_friction":-1}
        ]
    }"#;
    let e = err(src);
    assert!(e.message.contains("torsional_friction"), "{}", e.message);
}

#[test]
fn equality_connect_round_trips() {
    let src = r#"{
        "bodies":[
          {"name":"a","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}},
          {"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}
        ],
        "equality":[
          {"kind":"connect","body_a":"a","body_b":"b","anchor_a":[0,0,0.1],"anchor_b":[0,0,-0.1]}
        ]
    }"#;
    let s = scene(src);
    assert_eq!(s.world.equalities.len(), 1);
    match &s.world.equalities[0] {
        newt::equality::Equality::Connect { body_a, body_b, .. } => {
            assert_eq!(*body_a, Some(0));
            assert_eq!(*body_b, Some(1));
        }
        _ => panic!("expected Connect"),
    }
}

#[test]
fn equality_weld_default_relative_orientation_is_identity() {
    let src = r#"{
        "bodies":[
          {"name":"a","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}},
          {"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}
        ],
        "equality":[
          {"kind":"weld","body_a":"a","body_b":"b","anchor_a":[0,0,0],"anchor_b":[0,0,0]}
        ]
    }"#;
    let s = scene(src);
    match &s.world.equalities[0] {
        newt::equality::Equality::Weld {
            relative_orientation,
            ..
        } => {
            assert_eq!(*relative_orientation, newt::math::Quat::IDENTITY);
        }
        _ => panic!("expected Weld"),
    }
}

#[test]
fn equality_distance_negative_rejected() {
    let src = r#"{
        "bodies":[
          {"name":"a","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}},
          {"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}
        ],
        "equality":[
          {"kind":"distance","body_a":"a","body_b":"b","anchor_a":[0,0,0],"anchor_b":[0,0,0],"distance":-0.1}
        ]
    }"#;
    let e = err(src);
    assert!(e.message.contains("distance"), "{}", e.message);
}

#[test]
fn equality_joint_polycoef_too_long_rejected() {
    let src = r#"{
        "trees":[{"name":"t","links":[
          {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
          {"name":"a","parent":"root","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[0.01,0.01,0.01]}},
          {"name":"b","parent":"root","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[0.01,0.01,0.01]}}
        ]}],
        "equality":[
          {"kind":"joint","tree":"t","joint_a":"a","joint_b":"b","polycoef":[0,1,0,0]}
        ]
    }"#;
    let e = err(src);
    assert!(e.message.contains("polycoef"), "{}", e.message);
}

#[test]
fn equality_joint_on_ball_rejected() {
    let src = r#"{
        "trees":[{"name":"t","links":[
          {"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
          {"name":"a","parent":"root","joint":{"kind":"hinge","axis":[1,0,0]},"mass":1,"inertia":{"kind":"diag","values":[0.01,0.01,0.01]}},
          {"name":"b","parent":"root","joint":{"kind":"ball"},"mass":1,"inertia":{"kind":"diag","values":[0.01,0.01,0.01]}}
        ]}],
        "equality":[
          {"kind":"joint","tree":"t","joint_a":"a","joint_b":"b","polycoef":[0,1,0]}
        ]
    }"#;
    let e = err(src);
    assert!(
        e.message.contains("hinge") || e.message.contains("slide"),
        "{}",
        e.message
    );
}

#[test]
fn equality_connect_both_world_rejected() {
    let src = r#"{
        "equality":[
          {"kind":"connect","anchor_a":[0,0,0],"anchor_b":[0,0,0]}
        ]
    }"#;
    let e = err(src);
    assert!(e.message.contains("world"), "{}", e.message);
}

#[test]
fn equality_unknown_kind_rejected() {
    let src = r#"{
        "equality":[
          {"kind":"cursed","body_a":"a"}
        ]
    }"#;
    let e = err(src);
    assert!(e.message.contains("cursed"), "{}", e.message);
}

#[test]
fn equality_unknown_field_rejected() {
    let src = r#"{
        "bodies":[
          {"name":"a","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}},
          {"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}
        ],
        "equality":[
          {"kind":"connect","body_a":"a","body_b":"b","anchor_a":[0,0,0],"anchor_b":[0,0,0],"dampign":0.5}
        ]
    }"#;
    let e = err(src);
    assert!(e.message.contains("dampign"), "{}", e.message);
}

#[test]
fn equality_world_anchor_string_accepted() {
    let src = r#"{
        "bodies":[
          {"name":"b","mass":1,"inertia":{"kind":"solid","shape":{"kind":"sphere","radius":0.1}}}
        ],
        "equality":[
          {"kind":"connect","body_a":"world","body_b":"b","anchor_a":[0,0,1],"anchor_b":[0,0,0.1]}
        ]
    }"#;
    let s = scene(src);
    match &s.world.equalities[0] {
        newt::equality::Equality::Connect { body_a, body_b, .. } => {
            assert_eq!(*body_a, None);
            assert_eq!(*body_b, Some(0));
        }
        _ => panic!("expected Connect"),
    }
}
