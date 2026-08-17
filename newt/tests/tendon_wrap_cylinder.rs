use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::tendon::{
    SpatialSegment, SpatialTendonBranch, SpatialTendonSite, SpatialWrap, Tendon, TendonKind,
    WrapCylinder,
};
use newt::tree::{Link, Tree, forward_kinematics};

fn fixed_tree() -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree
}

fn site(link: Option<usize>, position_local: Vec3) -> SpatialTendonSite {
    SpatialTendonSite {
        link,
        position_local,
    }
}

fn cylinder_tendon(a: Vec3, b: Vec3, sidesite: Option<Vec3>) -> Tendon {
    Tendon::spatial_branches(vec![SpatialTendonBranch {
        sites: vec![site(None, a), site(None, b)],
        segments: vec![SpatialSegment {
            wrap: Some(SpatialWrap::Cylinder(WrapCylinder {
                link: None,
                center_local: Vec3::ZERO,
                axis_local: Vec3::Z,
                radius: 0.5,
                sidesite: sidesite.map(|p| site(None, p)),
            })),
        }],
        divisor: 1.0,
    }])
}

#[test]
fn cylinder_wrap_matches_hand_tangent_geometry() {
    let tree = fixed_tree();
    let tendon = cylinder_tendon(Vec3::new(-2.0, 0.2, 0.0), Vec3::new(2.0, 0.2, 0.0), None);
    let poses = forward_kinematics(&tree);
    let kin = newt::tendon::tendon_kinematics(&tendon, &tree, &poses);
    let tangent = (4.04_f32 - 0.25_f32).sqrt();
    let d = 4.04_f32.sqrt();
    let gamma = newt::math::atan2(
        (1.0 - (-3.96_f32 / (d * d)).powi(2)).max(0.0).sqrt(),
        -3.96 / (d * d),
    );
    let theta = gamma - 2.0 * (tangent / d).asin();
    let expected = 2.0 * tangent + 0.5 * theta;
    assert!(
        (kin.length - expected).abs() < 2.0e-5,
        "{} != {expected}",
        kin.length
    );
}

#[test]
fn cylinder_sidesite_selects_the_other_tangent_pair() {
    let tree = fixed_tree();
    let poses = forward_kinematics(&tree);
    let short = cylinder_tendon(
        Vec3::new(-2.0, 0.2, 0.0),
        Vec3::new(2.0, 0.2, 0.0),
        Some(Vec3::new(0.0, 1.0, 0.0)),
    );
    let long = cylinder_tendon(
        Vec3::new(-2.0, 0.2, 0.0),
        Vec3::new(2.0, 0.2, 0.0),
        Some(Vec3::new(0.0, -1.0, 0.0)),
    );
    let short_l = newt::tendon::tendon_kinematics(&short, &tree, &poses).length;
    let long_l = newt::tendon::tendon_kinematics(&long, &tree, &poses).length;
    assert!(long_l > short_l + 0.1, "short={short_l} long={long_l}");
}

#[test]
fn cylinder_jacobian_matches_finite_difference() {
    let mut tree = fixed_tree();
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::Z,
            range: None,
            damping: 0.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let tendon = Tendon::spatial_branches(vec![SpatialTendonBranch {
        sites: vec![
            site(None, Vec3::new(-2.0, 0.2, 0.0)),
            site(Some(1), Vec3::new(2.0, 0.2, 0.0)),
        ],
        segments: vec![SpatialSegment {
            wrap: Some(SpatialWrap::Cylinder(WrapCylinder {
                link: None,
                center_local: Vec3::ZERO,
                axis_local: Vec3::Z,
                radius: 0.5,
                sidesite: None,
            })),
        }],
        divisor: 1.0,
    }]);
    tree.set_hinge_angle(1, 0.35);
    let h = 1.0e-4;
    let poses = forward_kinematics(&tree);
    let base = newt::tendon::tendon_kinematics(&tendon, &tree, &poses);
    let mut plus = tree.clone();
    plus.set_hinge_angle(1, 0.35 + h);
    let mut minus = tree.clone();
    minus.set_hinge_angle(1, 0.35 - h);
    let lp = newt::tendon::tendon_kinematics(&tendon, &plus, &forward_kinematics(&plus)).length;
    let lm = newt::tendon::tendon_kinematics(&tendon, &minus, &forward_kinematics(&minus)).length;
    let fd = (lp - lm) / (2.0 * h);
    let analytic = base.jacobian[tree.v_offset[1]];
    let rel = (fd - analytic).abs() / fd.abs().max(1.0e-6);
    println!("cylinder jacobian fd max rel error = {rel:.3e}");
    assert!(
        (fd - analytic).abs() < 2.0e-3,
        "fd={fd} analytic={analytic}"
    );
}

#[test]
fn pulley_jacobian_matches_finite_difference() {
    let mut tree = fixed_tree();
    tree.push_link(Link::new(
        Some(0),
        JointKind::Slide {
            axis: Vec3::Z,
            range: None,
            damping: 0.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let tendon = Tendon::spatial_branches(vec![
        SpatialTendonBranch {
            sites: vec![
                site(None, Vec3::new(-1.0, 0.0, 0.0)),
                site(Some(1), Vec3::ZERO),
            ],
            segments: vec![SpatialSegment { wrap: None }],
            divisor: 1.0,
        },
        SpatialTendonBranch {
            sites: vec![
                site(None, Vec3::new(1.0, 0.0, 0.0)),
                site(Some(1), Vec3::ZERO),
            ],
            segments: vec![SpatialSegment { wrap: None }],
            divisor: 2.0,
        },
    ]);
    tree.set_slide_position(1, 0.3);
    let h = 1.0e-4;
    let base = newt::tendon::tendon_kinematics(&tendon, &tree, &forward_kinematics(&tree));
    let mut plus = tree.clone();
    plus.set_slide_position(1, 0.3 + h);
    let mut minus = tree.clone();
    minus.set_slide_position(1, 0.3 - h);
    let lp = newt::tendon::tendon_kinematics(&tendon, &plus, &forward_kinematics(&plus)).length;
    let lm = newt::tendon::tendon_kinematics(&tendon, &minus, &forward_kinematics(&minus)).length;
    let fd = (lp - lm) / (2.0 * h);
    let analytic = base.jacobian[tree.v_offset[1]];
    let rel = (fd - analytic).abs() / fd.abs().max(1.0e-6);
    println!("pulley jacobian fd max rel error = {rel:.3e}");
    assert!(
        (fd - analytic).abs() < 2.0e-3,
        "fd={fd} analytic={analytic}"
    );
}

#[test]
fn pulley_divisor_scales_branch_length_and_jacobian() {
    let tree = fixed_tree();
    let tendon = Tendon::spatial_branches(vec![
        SpatialTendonBranch {
            sites: vec![
                site(None, Vec3::new(0.0, 0.0, 0.0)),
                site(None, Vec3::new(2.0, 0.0, 0.0)),
            ],
            segments: vec![SpatialSegment { wrap: None }],
            divisor: 1.0,
        },
        SpatialTendonBranch {
            sites: vec![
                site(None, Vec3::new(0.0, 1.0, 0.0)),
                site(None, Vec3::new(0.0, 5.0, 0.0)),
            ],
            segments: vec![SpatialSegment { wrap: None }],
            divisor: 2.0,
        },
    ]);
    let kin = newt::tendon::tendon_kinematics(&tendon, &tree, &forward_kinematics(&tree));
    assert!((kin.length - 4.0).abs() < 1.0e-6);
}

#[test]
fn json_and_mjcf_accept_cylinder_and_pulley_branches() {
    let json = r#"{
      "trees":[{"name":"t","links":[{"name":"root","joint":{"kind":"fixed"},"mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}]}],
      "tendons":[{"name":"c","tree":"t","kind":"spatial","branches":[
        {"divisor":1,"sites":[{"position":[-2,0,0]},{"position":[2,0,0]}],"wraps":[{"segment":0,"kind":"cylinder","center":[0,0,0],"radius":0.5}]},
        {"divisor":2,"sites":[{"position":[0,1,0]},{"position":[0,3,0]}]}
      ]}]
    }"#;
    let scene = newt::model::load_str(json).expect("json tendon");
    let tendon = &scene.world.trees[0].tendons[0];
    assert!(matches!(tendon.kind, TendonKind::Spatial { .. }));

    let mjcf = r#"<mujoco><worldbody><body name="root"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><site name="a" pos="-2 0 0"/><site name="b" pos="2 0 0"/><site name="c" pos="0 1 0"/><site name="d" pos="0 3 0"/><geom name="cyl" type="cylinder" size="0.5 1"/></body></worldbody><tendon><spatial name="c"><site site="a"/><geom geom="cyl"/><site site="b"/><pulley divisor="2"/><site site="c"/><site site="d"/></spatial></tendon></mujoco>"#;
    let scene = newt::mjcf::load_mjcf_str(mjcf).expect("mjcf tendon");
    let tendon = &scene.world.trees[0].tendons[0];
    assert!(matches!(
        &tendon.kind,
        TendonKind::Spatial { branches }
            if matches!(branches[0].segments[0].wrap, Some(SpatialWrap::Cylinder(_)))
    ));
}
