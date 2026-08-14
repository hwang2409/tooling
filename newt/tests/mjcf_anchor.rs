//! MJCF ↔ JSON byte-identical trajectory anchor.
//!
//! The MJCF loader (`newt::mjcf::load_mjcf_path`) and the JSON loader
//! (`newt::model::load_from_path`) must produce Scenes that step to
//! bit-identical trajectories. This test loads each mirror-pair, steps
//! both worlds by a nontrivial number of steps, and compares every
//! floating-point state slot using [`f32::to_bits`] — the strongest
//! equality this project ever asks for (matches the golden-trajectory
//! doctrine in `docs/superpowers/specs/2026-08-13-newt-physics-design.md`).
//!
//! If a fixture drifts, either the two files disagreed at load time
//! (chase the drift back to whichever field diverged — likely a
//! rounded-out MJCF inertia number) or the MJCF loader took a
//! different construction path (chase into `crate::mjcf`).

use newt::mjcf::load_mjcf_path;
use newt::model::load_from_path;
use newt::world::World;

/// Compare two worlds' full float state bit-for-bit. Returns a mismatched
/// field name (with an index if applicable) on the first divergence, or
/// `Ok(())` if the two states are bit-identical.
fn assert_worlds_bit_identical(a: &World, b: &World) {
    assert_eq!(a.bodies.len(), b.bodies.len(), "body count");
    assert_eq!(a.trees.len(), b.trees.len(), "tree count");
    assert_eq!(a.geoms.len(), b.geoms.len(), "geom count");

    for (i, (ba, bb)) in a.bodies.iter().zip(b.bodies.iter()).enumerate() {
        assert_eq!(
            (
                ba.position.x.to_bits(),
                ba.position.y.to_bits(),
                ba.position.z.to_bits()
            ),
            (
                bb.position.x.to_bits(),
                bb.position.y.to_bits(),
                bb.position.z.to_bits()
            ),
            "body[{i}].position"
        );
        assert_eq!(
            (
                ba.linear_velocity.x.to_bits(),
                ba.linear_velocity.y.to_bits(),
                ba.linear_velocity.z.to_bits(),
            ),
            (
                bb.linear_velocity.x.to_bits(),
                bb.linear_velocity.y.to_bits(),
                bb.linear_velocity.z.to_bits(),
            ),
            "body[{i}].linear_velocity"
        );
        assert_eq!(
            (
                ba.orientation.x.to_bits(),
                ba.orientation.y.to_bits(),
                ba.orientation.z.to_bits(),
                ba.orientation.w.to_bits(),
            ),
            (
                bb.orientation.x.to_bits(),
                bb.orientation.y.to_bits(),
                bb.orientation.z.to_bits(),
                bb.orientation.w.to_bits(),
            ),
            "body[{i}].orientation"
        );
        assert_eq!(
            (
                ba.angular_velocity_body.x.to_bits(),
                ba.angular_velocity_body.y.to_bits(),
                ba.angular_velocity_body.z.to_bits(),
            ),
            (
                bb.angular_velocity_body.x.to_bits(),
                bb.angular_velocity_body.y.to_bits(),
                bb.angular_velocity_body.z.to_bits(),
            ),
            "body[{i}].angular_velocity_body"
        );
    }

    for (ti, (ta, tb)) in a.trees.iter().zip(b.trees.iter()).enumerate() {
        assert_eq!(ta.q.len(), tb.q.len(), "tree[{ti}] q length");
        assert_eq!(ta.qdot.len(), tb.qdot.len(), "tree[{ti}] qdot length");
        for (i, (qa, qb)) in ta.q.iter().zip(tb.q.iter()).enumerate() {
            assert_eq!(qa.to_bits(), qb.to_bits(), "tree[{ti}].q[{i}]");
        }
        for (i, (qa, qb)) in ta.qdot.iter().zip(tb.qdot.iter()).enumerate() {
            assert_eq!(qa.to_bits(), qb.to_bits(), "tree[{ti}].qdot[{i}]");
        }
    }
}

fn step_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

#[test]
fn pendulum_xml_matches_json_bit_for_bit_after_600_steps() {
    let base = std::env::current_dir().unwrap();
    let json = load_from_path(base.join("models/pendulum.json")).unwrap();
    let xml = load_mjcf_path(base.join("models/pendulum.xml")).unwrap();
    // Sanity: same tree shape.
    assert_eq!(
        json.world.trees[0].links.len(),
        xml.world.trees[0].links.len()
    );
    let mut a = json.world.clone();
    let mut b = xml.world.clone();
    step_n(&mut a, 600);
    step_n(&mut b, 600);
    assert_worlds_bit_identical(&a, &b);
}

#[test]
fn stack_xml_matches_json_bit_for_bit_after_600_steps() {
    let base = std::env::current_dir().unwrap();
    let json = load_from_path(base.join("models/stack.json")).unwrap();
    let xml = load_mjcf_path(base.join("models/stack.xml")).unwrap();
    assert_eq!(json.world.bodies.len(), xml.world.bodies.len());
    let mut a = json.world.clone();
    let mut b = xml.world.clone();
    step_n(&mut a, 600);
    step_n(&mut b, 600);
    assert_worlds_bit_identical(&a, &b);
}

#[test]
fn arm_xml_matches_json_bit_for_bit_after_600_steps() {
    let base = std::env::current_dir().unwrap();
    let json = load_from_path(base.join("models/arm.json")).unwrap();
    let xml = load_mjcf_path(base.join("models/arm.xml")).unwrap();
    let mut a = json.world.clone();
    let mut b = xml.world.clone();
    step_n(&mut a, 600);
    step_n(&mut b, 600);
    assert_worlds_bit_identical(&a, &b);
}
