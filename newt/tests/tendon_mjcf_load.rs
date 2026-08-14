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
