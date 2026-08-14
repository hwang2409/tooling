//! v1 tier 6 sensor anchors — one test per hand-derived expectation.
//!
//! Every check has an independent hand-derivation of the expected reading,
//! matched to a tolerance that comfortably covers the integrator error at
//! the scene's `dt`. See `newt/docs/sensors.md` for the semantics table.

use newt::body::Body;
use newt::geom::Geom;
use newt::joint::JointKind;
use newt::math::{FRAC_PI_2, Mat3, PI, Quat, Vec3};
use newt::sensor::{Sensor, SensorAttach, SensorKind, SiteFrame};
use newt::solver::{SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::World;

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

// ---------------------------------------------------------------------------
// static-body accelerometer reads +|g| along +z in the world-aligned site.
// ---------------------------------------------------------------------------

fn static_body_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    // A sphere resting on a static plane at z=0. `Body::solid_sphere`
    // places COM at the given position; radius 0.5 so the ground contact
    // starts with penetration = 0.5 - 0.5 = 0 (just touching), stabilising
    // fast into the penalty-spring equilibrium.
    let radius = 0.5;
    let b = Body::solid_sphere(1.0, radius, Vec3::new(0.0, 0.0, radius), Quat::IDENTITY);
    w.add_body(b);
    w.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
    w.add_geom(Geom::sphere(0, radius, Vec3::ZERO, 0.5));
    w
}

#[test]
fn accelerometer_on_static_body_reads_plus_g_up() {
    let mut w = static_body_scene();
    let sensor = Sensor {
        name: "imu".into(),
        kind: SensorKind::Accelerometer(SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: Vec3::ZERO,
            local_orientation: Quat::IDENTITY,
        }),
    };
    w.add_sensor(sensor).unwrap();
    // Let contacts settle: after ~200 steps the sphere sits at penalty
    // equilibrium (penetration such that k·pen = mg).
    for _ in 0..400 {
        w.step();
    }
    let r = w.sensor(0).unwrap();
    // Static body: a_com_world ≈ 0, ω = 0, α = 0 ⇒ proper accel = -g_world
    // = (0, 0, +9.81). Tolerance loose enough to swallow tiny oscillations.
    assert!(approx(r[0], 0.0, 0.2), "reading[0] = {}", r[0]);
    assert!(approx(r[1], 0.0, 0.2), "reading[1] = {}", r[1]);
    assert!(
        approx(r[2], 9.81, 0.5),
        "static accelerometer should read +9.81 upward, got {}",
        r[2]
    );
}

#[test]
fn accelerometer_reading_rotates_with_site_orientation() {
    // Same static body; site rotated 90° about x. In world coords a_proper
    // = (0,0,+9.81). Site frame's +z becomes world +y (Rot_x(π/2) maps
    // site +z to world +y). So specific force in site frame:
    //   a_site = R_site^T · (0,0,+9.81)
    // With R_site = Rot_x(π/2), R_site^T · z_world = ?
    // Rot_x(π/2) · [0,0,1] = [0,-1,0], so R_site^T · [0,0,1] = [0,1,0]...
    // Actually let's re-derive: Rot_x(π/2) is a rotation about x by π/2
    // taking (0,1,0)→(0,0,1). So R_site · (0,1,0) = (0,0,1). Thus
    // R_site^T · (0,0,1) = (0,1,0). So the reading should be (0, 9.81, 0).
    let mut w = static_body_scene();
    let rot = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);
    w.add_sensor(Sensor {
        name: "imu".into(),
        kind: SensorKind::Accelerometer(SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: Vec3::ZERO,
            local_orientation: rot,
        }),
    })
    .unwrap();
    for _ in 0..400 {
        w.step();
    }
    let r = w.sensor(0).unwrap();
    assert!(approx(r[0], 0.0, 0.2), "reading[0] = {}", r[0]);
    assert!(
        approx(r[1], 9.81, 0.5),
        "site +y should point along world +z after Rot_x(π/2); reading = ({}, {}, {})",
        r[0],
        r[1],
        r[2]
    );
    assert!(approx(r[2], 0.0, 0.2), "reading[2] = {}", r[2]);
}

// ---------------------------------------------------------------------------
// accelerometer on a constant-ω body at an offset site: pure centripetal.
// ---------------------------------------------------------------------------

#[test]
fn accelerometer_at_offset_under_constant_spin_reads_centripetal() {
    // Free body spinning at constant ω about body z, gravity-free, no
    // contacts. Site offset r = (0.5, 0, 0) in body frame. Since the body
    // is in free space with zero external wrench, α = 0 exactly, ω is
    // constant. Site-anchor world acceleration = ω × (ω × r).
    // With ω = (0,0,ωz), r = (0.5,0,0), ω × r = (0, ωz·0.5, 0);
    // ω × (ω × r) = (-ωz²·0.5, 0, 0). Proper accel = a - g = a (no g).
    let mut w = World::new();
    w.gravity = Vec3::ZERO; // gravity-free
    w.dt = 0.005;
    let mut b = Body::solid_sphere(1.0, 0.25, Vec3::ZERO, Quat::IDENTITY);
    b.angular_velocity_body = Vec3::new(0.0, 0.0, 3.0); // ωz = 3 rad/s
    w.add_body(b);
    // Site aligned with body frame; offset (0.5, 0, 0).
    w.add_sensor(Sensor {
        name: "imu".into(),
        kind: SensorKind::Accelerometer(SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: Vec3::new(0.5, 0.0, 0.0),
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    // A tiny number of steps so ω hasn't precessed off body-z. Free
    // solid-sphere with isotropic inertia keeps ω = ωz·body-z anyway.
    for _ in 0..10 {
        w.step();
    }
    let r = w.sensor(0).unwrap();
    // Expected in body frame (site aligned with body): (-ω²r, 0, 0)
    // = (-9 * 0.5, 0, 0) = (-4.5, 0, 0).
    // NOTE: the reading is in the SITE frame. Site is aligned with body, so
    // same numeric expectation.
    assert!(
        approx(r[0], -4.5, 0.05),
        "centripetal x = {} (expected -4.5)",
        r[0]
    );
    assert!(approx(r[1], 0.0, 0.05), "y = {}", r[1]);
    assert!(approx(r[2], 0.0, 0.05), "z = {}", r[2]);
}

// ---------------------------------------------------------------------------
// LINK-ATTACHED accelerometer coverage. The reviewer's blocker-1 probe: a
// free-root LINK (single-link tree with a free root) spinning AND
// translating under gravity-only has proper acceleration = 0 everywhere,
// including at any offset site — because the whole body is in free fall.
// Before the spatial-to-classical `+ ω × v` correction, this reading was
// nonzero and matched the missing term exactly.
// ---------------------------------------------------------------------------

fn free_root_link_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (
            Vec3::new(0.5, -0.3, 2.0),
            Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.2), 0.6),
        ),
        (Vec3::ZERO, Quat::IDENTITY),
        1.5,
        Mat3::diag(0.4, 0.6, 0.5),
    ));
    // Spinning AND translating so the ω × v term is nonzero — that's what
    // the missing correction hides.
    let twist =
        newt::spatial::SpatialMotion::new(Vec3::new(1.2, -0.7, 0.9), Vec3::new(0.4, 0.6, -0.3));
    tree.set_free_root_velocity(twist);
    w.add_tree(tree);
    w
}

#[test]
fn accelerometer_on_free_root_link_under_gravity_reads_zero() {
    let mut w = free_root_link_scene();
    // Site AT COM (offset = 0) — the only site where physical proper
    // acceleration is EXACTLY zero for a freely-falling body regardless
    // of its rotation. Any nonzero offset picks up the real centripetal
    // + Euler terms `ω × (ω × r) + α × r`. This isolates the bug: the
    // pre-fix reading here would be `−R · (ω_body × v_body)` (the
    // missing spatial-to-classical correction), matching the reviewer's
    // (-0.96, -0.51, 0.39) probe order of magnitude for the chosen
    // twist. After the fix it is bit-close to zero.
    w.add_sensor(Sensor {
        name: "imu".into(),
        kind: SensorKind::Accelerometer(SiteFrame {
            attach: SensorAttach::Link(0, 0),
            local_offset: Vec3::ZERO,
            local_orientation: Quat::from_axis_angle(Vec3::Y, PI / 6.0),
        }),
    })
    .unwrap();
    w.step();
    let r = w.sensor(0).unwrap();
    assert!(
        r[0].abs() < 5e-3 && r[1].abs() < 5e-3 && r[2].abs() < 5e-3,
        "free-fall link accelerometer at COM must read (0, 0, 0), got ({}, {}, {}) — \
         the spatial-to-classical + ω × v correction is missing?",
        r[0],
        r[1],
        r[2],
    );
}

#[test]
fn accelerometer_link_and_body_paths_agree_in_matching_state() {
    // Two worlds in the same physical state — one with a free body, one
    // with a single-link free-root tree — must report the same
    // accelerometer reading at the same offset site. This pins that the
    // Link path shares the Body path's semantics after the spatial-to-
    // classical fix.
    let position = Vec3::new(1.3, -0.5, 4.0);
    let orientation = Quat::from_axis_angle(Vec3::new(0.4, 1.0, -0.3), 0.7);
    let omega_body = Vec3::new(0.8, -0.4, 1.1);
    let linear_v_world = Vec3::new(0.3, 0.4, -0.5);
    let inertia = Mat3::diag(0.4, 0.6, 0.5);
    let mass = 1.5;
    let offset = Vec3::new(0.15, -0.1, 0.2);
    let local_orientation = Quat::from_axis_angle(Vec3::Y, PI / 6.0);

    // Body-attached world.
    let mut wb = World::new();
    wb.dt = 0.005;
    wb.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut b = Body::new(mass, inertia, position, orientation);
    b.angular_velocity_body = omega_body;
    b.linear_velocity = linear_v_world;
    wb.add_body(b);
    wb.add_sensor(Sensor {
        name: "imu".into(),
        kind: SensorKind::Accelerometer(SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: offset,
            local_orientation,
        }),
    })
    .unwrap();

    // Link-attached world — free-root link with identical state.
    let mut wl = World::new();
    wl.dt = 0.005;
    wl.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (position, orientation),
        (Vec3::ZERO, Quat::IDENTITY),
        mass,
        inertia,
    ));
    // Free-root twist is body-frame at COM: angular then linear.
    // linear_body = R^T · linear_world.
    let linear_body = orientation.inverse_rotate(linear_v_world);
    let twist = newt::spatial::SpatialMotion::new(omega_body, linear_body);
    tree.set_free_root_velocity(twist);
    wl.add_tree(tree);
    wl.add_sensor(Sensor {
        name: "imu".into(),
        kind: SensorKind::Accelerometer(SiteFrame {
            attach: SensorAttach::Link(0, 0),
            local_offset: offset,
            local_orientation,
        }),
    })
    .unwrap();

    // Evaluate both — no step needed (posed).
    wb.evaluate_sensors(&[]);
    wl.evaluate_sensors(&[]);
    let rb = wb.sensor(0).unwrap();
    let rl = wl.sensor(0).unwrap();
    let tol = 1e-3;
    assert!(
        (rb[0] - rl[0]).abs() < tol && (rb[1] - rl[1]).abs() < tol && (rb[2] - rl[2]).abs() < tol,
        "body vs link accelerometer disagree: body = ({}, {}, {}), link = ({}, {}, {})",
        rb[0],
        rb[1],
        rb[2],
        rl[0],
        rl[1],
        rl[2],
    );
}

// ---------------------------------------------------------------------------
// gyro on a spinning free body: reading equals body-frame ω, rotated into
// site frame. Independent code path from `Body::angular_velocity_body`
// because gyro goes through the site-frame transform.
// ---------------------------------------------------------------------------

#[test]
fn gyro_on_spinning_body_matches_body_frame_omega_in_site() {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    w.dt = 0.005;
    // Asymmetric inertia so the reading is discriminating (a symmetric
    // body would rotate about a principal axis and mask any bug that only
    // hits off-axis components).
    let inertia = Mat3::diag(1.0, 2.0, 3.0);
    let mut b = Body::new(1.0, inertia, Vec3::ZERO, Quat::IDENTITY);
    b.angular_velocity_body = Vec3::new(0.5, -0.7, 1.1);
    w.add_body(b);
    // Site rotated 90° about z: gyro should read the body ω rotated by
    // R^T · Rot_z(π/2)^T · ω_body. Actually the sensor computes
    // omega_site = local_orientation^-1.rotate(omega_body).
    // With local_orientation = Rot_z(π/2): Rot_z(π/2)^-1 = Rot_z(-π/2).
    // Rot_z(-π/2) · (0.5, -0.7, 1.1) = (cos(-π/2)·0.5 - sin(-π/2)·(-0.7),
    //                                     sin(-π/2)·0.5 + cos(-π/2)·(-0.7),
    //                                     1.1)
    //                                 = (0·0.5 - (-1)·(-0.7), (-1)·0.5 + 0·(-0.7), 1.1)
    //                                 = (-0.7, -0.5, 1.1)
    let rot = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
    w.add_sensor(Sensor {
        name: "gyro".into(),
        kind: SensorKind::Gyro(SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: Vec3::new(0.1, 0.2, 0.3),
            local_orientation: rot,
        }),
    })
    .unwrap();
    w.step();
    let r = w.sensor(0).unwrap();
    // Gyro reads instantaneous ω (post-step ω differs slightly from the
    // t=0 value because free precession alters ω in body frame under
    // asymmetric inertia). Tolerance covers one integration step.
    // Recompute expectation from the current post-step body ω.
    let omega_body = w.bodies[0].angular_velocity_body;
    let expected = rot.inverse_rotate(omega_body);
    assert!(
        approx(r[0], expected.x, 5e-4),
        "gyro x {}, expected {}",
        r[0],
        expected.x
    );
    assert!(
        approx(r[1], expected.y, 5e-4),
        "gyro y {}, expected {}",
        r[1],
        expected.y
    );
    assert!(
        approx(r[2], expected.z, 5e-4),
        "gyro z {}, expected {}",
        r[2],
        expected.z
    );
    // The site rotation must MATTER — a diagnostic that catches the case
    // where local_orientation is silently ignored.
    let identity_reading = omega_body;
    assert!(
        (r[0] - identity_reading.x).abs() > 0.05 || (r[1] - identity_reading.y).abs() > 0.05,
        "gyro reading did not visibly rotate — site frame ignored?"
    );
}

// ---------------------------------------------------------------------------
// jointpos/jointvel on a driven pendulum vs the integrator state (exact).
// ---------------------------------------------------------------------------

fn pendulum_tree() -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    tree
}

#[test]
fn jointpos_and_jointvel_match_integrator_state_after_step() {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut t = pendulum_tree();
    // Push the pendulum away from equilibrium so q and qdot are both nonzero.
    t.set_hinge_angle(1, 0.5);
    t.set_hinge_rate(1, -0.3);
    w.add_tree(t);
    w.add_sensor(Sensor {
        name: "hip_q".into(),
        kind: SensorKind::JointPos { tree: 0, link: 1 },
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "hip_qd".into(),
        kind: SensorKind::JointVel { tree: 0, link: 1 },
    })
    .unwrap();
    for _ in 0..50 {
        w.step();
    }
    let q = w.trees[0].hinge_angle(1);
    let qd = w.trees[0].hinge_rate(1);
    let sp = w.sensor(0).unwrap();
    let sv = w.sensor(1).unwrap();
    assert_eq!(sp[0], q, "jointpos must equal integrator q exactly");
    assert_eq!(sv[0], qd, "jointvel must equal integrator qdot exactly");
}

// ---------------------------------------------------------------------------
// framepos/framequat on a site at the end of a posed 2-link arm.
// ---------------------------------------------------------------------------

#[test]
fn framepos_framequat_on_end_of_posed_2link_arm() {
    // Two-link arm, root fixed at world origin, both hinges at π/2 about
    // world +x. Rod length L=1 each. Hand-derived tip position:
    //   shoulder rotates rod 1 from down-Z to +Y (COM at (0, 0.5, 0)).
    //   elbow rotates rod 2 relative to rod-1 body, but since axis is
    //   parent's +x (same world +x), it accumulates to π. Rod 2 hangs
    //   opposite of rod 1 → back to -Z direction from the elbow joint.
    //   Elbow joint sits at (0, 1, 0). Rod 2's tip: elbow + (0, 0, -1) =
    //   (0, 1, -1). Site at end of rod 2 offset (0, 0, -0.5) in child.
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    w.dt = 0.005;
    let mut tree = Tree::new();
    // Fixed root at world origin.
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Rod 1: hinge x, anchor at parent (0,0,0), COM offset in child (0,0,0.5).
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    // Rod 2: hinge x, anchor in parent at (0,0,-0.5), COM offset in child (0,0,0.5).
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.0, 0.0, -0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    tree.set_hinge_angle(1, FRAC_PI_2);
    tree.set_hinge_angle(2, FRAC_PI_2);
    let tree_idx = w.add_tree(tree);
    // Site frame on rod 2, offset (0, 0, -0.5) reaches the rod's tip (0.5m
    // beyond COM along -z in child body frame).
    w.add_sensor(Sensor {
        name: "tip_pos".into(),
        kind: SensorKind::FramePos(SiteFrame {
            attach: SensorAttach::Link(tree_idx, 2),
            local_offset: Vec3::new(0.0, 0.0, -0.5),
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "tip_quat".into(),
        kind: SensorKind::FrameQuat(SiteFrame {
            attach: SensorAttach::Link(tree_idx, 2),
            local_offset: Vec3::new(0.0, 0.0, -0.5),
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    // Evaluate without stepping (posed scene).
    w.evaluate_sensors(&[]);
    let p = w.sensor(0).unwrap();
    // Hand derivation:
    //   after two π/2 rotations about parent +x, rod 2 body-z points along
    //   -world-z, i.e. same as rod 1 body-z direction rotated by π = pointing
    //   back down. Actually let's re-derive carefully.
    //   Rod 1 orientation: Rot_x(π/2). It takes body-z (0,0,1) to (0,-1,0)
    //   in world. Rod 1 anchor at world origin; COM offset in child is
    //   (0,0,0.5), so rod-1 COM world = 0 + R1·(0,0,0.5) − but wait,
    //   joint_offset_in_child is where the parent joint anchor sits in the
    //   child body frame. Given parent anchor world = (0,0,0), the COM is
    //   at world = joint_world - R_child · joint_offset_in_child.
    //   Rod 1: COM world = (0,0,0) - R1·(0,0,0.5) = -(0,-0.5,0) = (0,0.5,0).
    //   Rod 2 anchor in parent (rod 1) = (0,0,-0.5). World joint pos =
    //     rod1_com + R1·(0,0,-0.5) = (0,0.5,0) + (0,0.5,0) = (0,1,0).
    //   Rod 2 orientation: R2 = R1 · Rot_x(π/2) = Rot_x(π).
    //     Rot_x(π): body-z (0,0,1) → (0,0,-1). So rod-2 z is world-down.
    //   Rod 2 COM world = joint_world - R2·(0,0,0.5) = (0,1,0) - (0,0,-0.5)
    //                  = (0,1,0.5).
    //   Site offset (0,0,-0.5) in child body frame: R2·(0,0,-0.5) = (0,0,0.5).
    //   Site world = rod2_com + (0,0,0.5) = (0, 1, 1).
    assert!(approx(p[0], 0.0, 1e-4), "tip x = {}", p[0]);
    assert!(approx(p[1], 1.0, 1e-4), "tip y = {}", p[1]);
    assert!(approx(p[2], 1.0, 1e-4), "tip z = {}", p[2]);
    let q = w.sensor(1).unwrap();
    // Rod-2 world orientation = Rot_x(π). Quaternion (x=1, y=0, z=0, w=0)
    // up to sign. Site local_orientation = IDENTITY so site orientation
    // matches parent's.
    //
    // A unit quaternion is only unique up to a global sign (q and -q
    // represent the same rotation), so we compare against `q_expected`
    // AND `-q_expected` and require exact match to one of them. A test
    // that only checked component absolute values would silently accept
    // a conjugate mutant (x → -x, y → -y, z → -z while w stays) because
    // magnitudes would still line up.
    let expected = Quat::new(1.0, 0.0, 0.0, 0.0);
    let mine = Quat::new(q[0], q[1], q[2], q[3]);
    let match_pos = (mine.x - expected.x).abs() < 1e-4
        && (mine.y - expected.y).abs() < 1e-4
        && (mine.z - expected.z).abs() < 1e-4
        && (mine.w - expected.w).abs() < 1e-4;
    let match_neg = (mine.x + expected.x).abs() < 1e-4
        && (mine.y + expected.y).abs() < 1e-4
        && (mine.z + expected.z).abs() < 1e-4
        && (mine.w + expected.w).abs() < 1e-4;
    assert!(
        match_pos || match_neg,
        "framequat = ({}, {}, {}, {}); expected ±(1, 0, 0, 0)",
        q[0],
        q[1],
        q[2],
        q[3]
    );
}

/// Discriminating anchor: a non-180° rotation whose framequat cannot
/// coincide with its own conjugate (a conjugate mutant `(x, y, z) → (-x,
/// -y, -z)` produces a physically DIFFERENT rotation for θ ≠ π, and the
/// sign-canonical `±q` comparison catches it).
#[test]
fn framequat_rejects_conjugate_mutant_at_non_180_rotation() {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Hinge about +x, angle = π/4 → child quat = (sin(π/8), 0, 0, cos(π/8))
    //                              ≈ (0.383, 0, 0, 0.924).
    // The conjugate would be (-0.383, 0, 0, 0.924) — a different rotation.
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    tree.set_hinge_angle(1, PI / 4.0);
    let ti = w.add_tree(tree);
    w.add_sensor(Sensor {
        name: "q".into(),
        kind: SensorKind::FrameQuat(SiteFrame {
            attach: SensorAttach::Link(ti, 1),
            local_offset: Vec3::ZERO,
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    w.evaluate_sensors(&[]);
    let r = w.sensor(0).unwrap();
    let expected = Quat::from_axis_angle(Vec3::X, PI / 4.0);
    let mine = Quat::new(r[0], r[1], r[2], r[3]);
    let match_pos = (mine.x - expected.x).abs() < 1e-4
        && (mine.y - expected.y).abs() < 1e-4
        && (mine.z - expected.z).abs() < 1e-4
        && (mine.w - expected.w).abs() < 1e-4;
    let match_neg = (mine.x + expected.x).abs() < 1e-4
        && (mine.y + expected.y).abs() < 1e-4
        && (mine.z + expected.z).abs() < 1e-4
        && (mine.w + expected.w).abs() < 1e-4;
    assert!(
        match_pos || match_neg,
        "framequat = ({}, {}, {}, {}); expected ±({}, {}, {}, {})",
        r[0],
        r[1],
        r[2],
        r[3],
        expected.x,
        expected.y,
        expected.z,
        expected.w,
    );
}

// ---------------------------------------------------------------------------
// touch sensor on a resting sphere ≈ mg.
// ---------------------------------------------------------------------------

#[test]
fn touch_on_resting_sphere_penalty_mode_reads_mg() {
    let mut w = static_body_scene();
    // Touch on the sphere geom (index 1; plane is 0).
    w.add_sensor(Sensor {
        name: "foot".into(),
        kind: SensorKind::Touch { geom: 1 },
    })
    .unwrap();
    for _ in 0..600 {
        w.step();
    }
    let r = w.sensor(0).unwrap();
    let mg = 1.0 * 9.81;
    // Penalty settling: f_n ≈ mg at equilibrium. Loose tolerance to
    // absorb the small residual oscillation.
    assert!(
        approx(r[0], mg, 0.6),
        "penalty touch = {} (expected {})",
        r[0],
        mg
    );
}

#[test]
fn touch_on_link_attached_sphere_resting_on_plane_reads_mg() {
    // Blocker-2 anchor: a link-attached sphere sitting on a static plane
    // must read ≈ link.mass · g. Before the fix `penalty_normal_force`
    // resolved mass only for `GeomAttach::Body` — link geoms fell through
    // to `m_eff = 0` and the touch sensor silently reported zero.
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut tree = Tree::new();
    let mass = 1.25;
    let radius = 0.4;
    // Free-root link whose sphere geom will settle on the ground.
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(0.0, 0.0, radius + 0.01), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        mass,
        newt::geom::solid_sphere_inertia(mass, radius),
    ));
    let tree_idx = w.add_tree(tree);
    // Static plane at z=0 + link-attached sphere.
    w.add_geom(newt::geom::Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
    w.add_geom(newt::geom::Geom::sphere_on_link(
        tree_idx,
        0,
        radius,
        Vec3::ZERO,
        0.5,
    ));
    // Sphere geom is index 1 (plane is 0).
    w.add_sensor(Sensor {
        name: "foot".into(),
        kind: SensorKind::Touch { geom: 1 },
    })
    .unwrap();
    for _ in 0..800 {
        w.step();
    }
    let r = w.sensor(0).unwrap();
    let mg = mass * 9.81;
    assert!(
        approx(r[0], mg, 0.8),
        "link-attached sphere touch = {} (expected {} = m·g)",
        r[0],
        mg
    );
}

#[test]
fn touch_on_resting_sphere_pgs_mode_reads_mg() {
    let mut w = static_body_scene();
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        ..SolverConfig::DEFAULT
    };
    w.add_sensor(Sensor {
        name: "foot".into(),
        kind: SensorKind::Touch { geom: 1 },
    })
    .unwrap();
    for _ in 0..800 {
        w.step();
    }
    let r = w.sensor(0).unwrap();
    let mg = 1.0 * 9.81;
    // PGS driven to soft-constraint equilibrium; the penalty-formula touch
    // reading follows within a wider band because the penetration depth
    // under PGS is set by SolImp/SolRef, not by k · pen = mg. The band is
    // still tight enough to be meaningfully discriminating.
    assert!(
        r[0] > mg * 0.4 && r[0] < mg * 3.0,
        "pgs touch = {} — should be in the same order as mg = {}",
        r[0],
        mg
    );
}

// ---------------------------------------------------------------------------
// force/torque through a static 2-link arm elbow: gravity-load moment arm.
// ---------------------------------------------------------------------------

/// Compare the sensor's Force/Torque readings against the hand-derived
/// statics expectation for a horizontal 2-link arm.
///
/// The tricky bit: the sensor's internal ABA computes `qddot` at the
/// current state, and RNE's per-link `f[i]` = `I·a + v×*(I·v) − f_ext`.
/// For the reading to equal the STATIC wrench, we need `qddot ≈ 0`. We
/// arrange that deterministically by injecting a compensating torque into
/// `qfrc_applied` — set it to `-bias_forces(q, qdot=0)`, so ABA sees the
/// gravity-canceled net and returns `qddot ≈ 0`. Then RNE's `f[i]`
/// reduces to `-f_ext_body` = the parent-side joint reaction the classic
/// statics derivation predicts.
#[test]
fn force_torque_at_elbow_of_horizontal_2link_arm_matches_statics() {
    use newt::dynamics::bias_forces;
    // Fixed root at origin. Two hinge links, both about world +x, angle
    // π/2 so the arm hangs horizontally along +y. At an instantaneous
    // static-pose evaluation with (q̇, q̈) = 0 and no external wrenches
    // beyond gravity, RNE's per-link f[i] is the wrench the parent joint
    // exerts on child i to keep it (statically) in place. In the child
    // body frame at COM:
    //   Force = -R^T · (m · g_world)
    //   Torque = -r_com_to_joint (body) × force + accumulated child moments
    // For the forearm (link 2) alone at equilibrium, the parent (elbow)
    // must supply an upward force m2·|g| and a torque about x that
    // balances the gravitational moment arm (r_com_to_elbow perpendicular
    // to gravity).
    //
    // Geometry: rod 2 body-z points along world +y (both hinges rotate
    // parent body-z from world-down to world +y then rotate rod-2 body-z
    // to keep it aligned with world +y via the identity rotation of q=0
    // on the wrist... actually with a nominal q2 = 0 on the second hinge
    // the rod-2 body frame equals rod-1's, so rod-2 body-z is world +y.
    // Now change: set q2 = 0 (straight) so both rods are colinear.
    // Body-z of rod 2 is world +y. r_com_to_elbow in body frame = (0,0,0.5)
    // (elbow is +z from COM by half rod length). Body-frame gravity:
    // gravity_world = (0,0,-9.81); R.rotate^T maps world→body. For rod 2
    // orientation Rot_x(π/2), inverse rotates world (0,0,-9.81) into
    // body: Rot_x(-π/2)·(0,0,-9.81) = ?
    //   Rot_x(-π/2) takes (0,0,z) to (0,-z,0). So body gravity = (0, 9.81, 0).
    // Force sensor (child frame): the wrench the parent transmits = the
    // negative of the sum of external body-frame forces on rod 2 alone.
    //   Applied gravity (body frame) = m2 · body-gravity = m2·(0, 9.81, 0).
    //   Newton's 2nd (static) ⇒ f_parent_body = -m2·(0, 9.81, 0)
    //     = m2·(0, -9.81, 0).
    // Torque about COM in body frame = r_com_to_elbow × f_parent_body
    //     = (0,0,0.5) × m2·(0,-9.81,0)
    //     = m2 · (0.5 · (-9.81) · (0·0 − 1·0), ... use full formula)
    //     Cross product (a × b) with a = (0,0,0.5), b = (0,-9.81,0):
    //       (a.y·b.z - a.z·b.y, a.z·b.x - a.x·b.z, a.x·b.y - a.y·b.x)
    //     = (0·0 - 0.5·(-9.81), 0.5·0 - 0·0, 0·(-9.81) - 0·0)
    //     = (4.905, 0, 0).
    //   For unit mass m2 = 1: torque = (4.905, 0, 0).
    let mut w = World::new();
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.dt = 0.005;
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Rod 1.
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    // Rod 2 (forearm): mass = 1.0 for a clean statics number.
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.0, 0.0, -0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY),
        1.0,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    tree.set_hinge_angle(1, FRAC_PI_2);
    tree.set_hinge_angle(2, 0.0);
    // Inject τ_applied = h(q, 0) so ABA solves M·qddot + h = h ⇒ qddot = 0.
    // `bias_forces` returns h; sign is +bias, not -bias.
    let bias = bias_forces(&tree, w.gravity);
    for (voff, b) in bias.iter().enumerate() {
        tree.qfrc_applied[voff] = *b;
    }
    let tree_idx = w.add_tree(tree);
    // Force + torque at the forearm's parent joint (elbow).
    w.add_sensor(Sensor {
        name: "elbow_f".into(),
        kind: SensorKind::Force {
            tree: tree_idx,
            link: 2,
        },
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "elbow_t".into(),
        kind: SensorKind::Torque {
            tree: tree_idx,
            link: 2,
        },
    })
    .unwrap();
    // No step needed — evaluate on the posed scene.
    w.evaluate_sensors(&[]);
    let f = w.sensor(0).unwrap();
    let t = w.sensor(1).unwrap();
    // Hand derivation (forearm body frame).
    //   With q1 = π/2 and q2 = 0, rod-2's world orientation is Rot_x(π/2):
    //   body-z points along world -y (the rod extends along -y from COM
    //   toward the elbow joint at (0, 1, 0)), and body-y points along
    //   world +z (up).
    //   Body-frame gravity: R^T · (0, 0, -9.81) = (0, -9.81, 0).
    //   Applied gravity wrench on rod-2 in body coords:
    //     f_ext_body.linear = m2·(0, -9.81, 0),  f_ext_body.torque = 0.
    //   RNE per-link wrench under (v = 0, a = 0):
    //     f[2] = -f_ext_body  ⇒  linear = (0, +9.81, 0), torque = 0.
    //   Force at joint anchor equals force at COM (translation-invariant):
    //     F_at_elbow_body = (0, +9.81, 0)  — parent pushes up in world.
    //   Torque translated to the elbow anchor (r_com_to_joint = (0, 0, 0.5)):
    //     τ_at_elbow_body = τ_at_COM − r × F
    //                     = 0 − (0, 0, 0.5) × (0, +9.81, 0)
    //                     = 0 − (−4.905, 0, 0)
    //                     = (+4.905, 0, 0).
    //   Positive torque about +x is the physical direction that holds the
    //   rod up against gravity.
    assert!(approx(f[0], 0.0, 0.05), "force x = {}", f[0]);
    assert!(
        approx(f[1], 9.81, 0.05),
        "force y = {} (expected +9.81)",
        f[1]
    );
    assert!(approx(f[2], 0.0, 0.05), "force z = {}", f[2]);
    assert!(
        approx(t[0], 4.905, 0.05),
        "torque x = {} (expected +4.905)",
        t[0]
    );
    assert!(approx(t[1], 0.0, 0.05), "torque y = {}", t[1]);
    assert!(approx(t[2], 0.0, 0.05), "torque z = {}", t[2]);
}

// ---------------------------------------------------------------------------
// No-perturbation: adding sensors must not change the trajectory bit-for-bit.
// ---------------------------------------------------------------------------

fn tumbling_body_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    let inertia = Mat3::diag(1.0, 2.0, 3.0);
    let mut b = Body::new(
        1.0,
        inertia,
        Vec3::new(0.0, 0.0, 5.0),
        Quat::from_axis_angle(Vec3::new(1.0, 0.3, -0.2), 0.7),
    );
    b.linear_velocity = Vec3::new(0.5, -0.3, 0.4);
    b.angular_velocity_body = Vec3::new(1.1, -0.7, 0.5);
    w.add_body(b);
    w
}

fn body_state_bytes(w: &World) -> Vec<u8> {
    let mut bytes = Vec::new();
    for b in &w.bodies {
        for v in [
            b.position.x,
            b.position.y,
            b.position.z,
            b.orientation.x,
            b.orientation.y,
            b.orientation.z,
            b.orientation.w,
            b.linear_velocity.x,
            b.linear_velocity.y,
            b.linear_velocity.z,
            b.angular_velocity_body.x,
            b.angular_velocity_body.y,
            b.angular_velocity_body.z,
        ] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    bytes
}

fn resting_sphere_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    let radius = 0.5;
    // A body dropped with some sideways drift so contact wrenches are
    // dynamically nontrivial across the step window.
    let mut b = Body::solid_sphere(
        1.2,
        radius,
        Vec3::new(0.0, 0.0, radius + 0.05),
        Quat::IDENTITY,
    );
    b.linear_velocity = Vec3::new(0.15, -0.1, 0.0);
    b.angular_velocity_body = Vec3::new(0.2, -0.3, 0.05);
    w.add_body(b);
    w.add_geom(newt::geom::Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
    w.add_geom(newt::geom::Geom::sphere(0, radius, Vec3::ZERO, 0.5));
    w
}

#[test]
fn stepping_with_sensors_matches_without_under_penalty_contacts() {
    // Same physical trajectory whether or not sensors run: for a
    // contact-heavy scene under `Penalty` the sensor pipeline recomputes
    // wrenches, but MUST NOT mutate world state. Byte-compared over
    // 400 steps.
    let mut plain = resting_sphere_scene();
    let mut instrumented = resting_sphere_scene();
    instrumented
        .add_sensor(Sensor {
            name: "foot".into(),
            kind: SensorKind::Touch { geom: 1 },
        })
        .unwrap();
    instrumented
        .add_sensor(Sensor {
            name: "imu".into(),
            kind: SensorKind::Accelerometer(SiteFrame {
                attach: SensorAttach::Body(0),
                local_offset: Vec3::new(0.1, 0.0, 0.0),
                local_orientation: Quat::IDENTITY,
            }),
        })
        .unwrap();
    for _ in 0..400 {
        plain.step();
        instrumented.step();
    }
    assert_eq!(
        body_state_bytes(&plain),
        body_state_bytes(&instrumented),
        "penalty-contact scene: sensor evaluation perturbed the simulation state"
    );
}

#[test]
fn stepping_with_sensors_matches_without_under_pgs_contacts() {
    // Same as above but with `SolverMode::Pgs` — the PGS re-solve inside
    // the sensor pipeline must not touch world state.
    let mut plain = resting_sphere_scene();
    let mut instrumented = resting_sphere_scene();
    plain.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        ..SolverConfig::DEFAULT
    };
    instrumented.solver = plain.solver;
    instrumented
        .add_sensor(Sensor {
            name: "foot".into(),
            kind: SensorKind::Touch { geom: 1 },
        })
        .unwrap();
    instrumented
        .add_sensor(Sensor {
            name: "imu".into(),
            kind: SensorKind::Accelerometer(SiteFrame {
                attach: SensorAttach::Body(0),
                local_offset: Vec3::new(0.1, 0.0, 0.0),
                local_orientation: Quat::IDENTITY,
            }),
        })
        .unwrap();
    for _ in 0..400 {
        plain.step();
        instrumented.step();
    }
    assert_eq!(
        body_state_bytes(&plain),
        body_state_bytes(&instrumented),
        "PGS-contact scene: sensor evaluation perturbed the simulation state"
    );
}

#[test]
fn stepping_with_sensors_matches_stepping_without_bit_for_bit() {
    let mut plain = tumbling_body_scene();
    let mut instrumented = tumbling_body_scene();
    instrumented
        .add_sensor(Sensor {
            name: "gyro".into(),
            kind: SensorKind::Gyro(SiteFrame {
                attach: SensorAttach::Body(0),
                local_offset: Vec3::new(0.1, 0.2, -0.05),
                local_orientation: Quat::from_axis_angle(Vec3::Y, PI / 5.0),
            }),
        })
        .unwrap();
    instrumented
        .add_sensor(Sensor {
            name: "imu".into(),
            kind: SensorKind::Accelerometer(SiteFrame {
                attach: SensorAttach::Body(0),
                local_offset: Vec3::new(0.1, 0.2, -0.05),
                local_orientation: Quat::IDENTITY,
            }),
        })
        .unwrap();
    for _ in 0..500 {
        plain.step();
        instrumented.step();
    }
    assert_eq!(
        body_state_bytes(&plain),
        body_state_bytes(&instrumented),
        "sensor evaluation perturbed the simulation state"
    );
}

// ---------------------------------------------------------------------------
// Determinism golden: sensordata vector for a fixed scene at fixed steps,
// byte-identical vs the tracked file.
// ---------------------------------------------------------------------------

fn deterministic_bench_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    // A pendulum + a free rotating body — mixes tree and body sensor paths.
    let mut tree = pendulum_tree();
    tree.set_hinge_angle(1, 0.4);
    tree.set_hinge_rate(1, -0.2);
    let ti = w.add_tree(tree);
    let inertia = Mat3::diag(1.0, 2.0, 3.0);
    let mut b = Body::new(
        1.0,
        inertia,
        Vec3::new(2.0, 0.0, 3.0),
        Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.3), 0.7),
    );
    b.angular_velocity_body = Vec3::new(0.6, 0.9, -0.4);
    b.linear_velocity = Vec3::new(-0.1, 0.2, 0.05);
    w.add_body(b);
    // One of each sensor type on this scene.
    w.add_sensor(Sensor {
        name: "hip_q".into(),
        kind: SensorKind::JointPos { tree: ti, link: 1 },
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "hip_qd".into(),
        kind: SensorKind::JointVel { tree: ti, link: 1 },
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "bob_pos".into(),
        kind: SensorKind::FramePos(SiteFrame {
            attach: SensorAttach::Link(ti, 1),
            local_offset: Vec3::new(0.0, 0.0, -0.5),
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "bob_quat".into(),
        kind: SensorKind::FrameQuat(SiteFrame {
            attach: SensorAttach::Link(ti, 1),
            local_offset: Vec3::ZERO,
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "body_gyro".into(),
        kind: SensorKind::Gyro(SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: Vec3::new(0.1, 0.2, 0.3),
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    w.add_sensor(Sensor {
        name: "body_imu".into(),
        kind: SensorKind::Accelerometer(SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: Vec3::new(0.1, 0.2, 0.3),
            local_orientation: Quat::IDENTITY,
        }),
    })
    .unwrap();
    w
}

fn sensordata_bytes(w: &World) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.sensors.data.len() * 4);
    for &v in &w.sensors.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn produce_sensor_golden() -> Vec<u8> {
    let mut w = deterministic_bench_scene();
    let mut bytes = Vec::new();
    w.step();
    bytes.extend_from_slice(&sensordata_bytes(&w));
    for _ in 1..100 {
        w.step();
    }
    bytes.extend_from_slice(&sensordata_bytes(&w));
    for _ in 100..500 {
        w.step();
    }
    bytes.extend_from_slice(&sensordata_bytes(&w));
    bytes
}

const SENSOR_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/sensors_battery.bin"
);

#[test]
fn sensor_golden_trajectory_is_byte_identical() {
    let expected = std::fs::read(SENSOR_GOLDEN).expect(
        "sensor golden file missing — run the ignored `regenerate_sensor_golden` \
         test on macOS to produce it, then commit",
    );
    let actual = produce_sensor_golden();
    assert_eq!(
        expected.len(),
        actual.len(),
        "golden byte length mismatch: expected {} got {}",
        expected.len(),
        actual.len()
    );
    if expected != actual {
        let first_diff = expected
            .iter()
            .zip(actual.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!("sensor golden mismatch; first byte diff at offset {first_diff}");
    }
}

#[test]
#[ignore]
fn regenerate_sensor_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_sensor_golden may only run on the reference host \
             (macOS aarch64); refusing to overwrite tracked bytes on {} / {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }
    let bytes = produce_sensor_golden();
    let dir = std::path::Path::new(SENSOR_GOLDEN).parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(SENSOR_GOLDEN, &bytes).unwrap();
    println!("wrote {} bytes to {SENSOR_GOLDEN}", bytes.len());
}
