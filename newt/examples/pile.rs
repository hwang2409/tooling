//! v1-tier-2 demo: a mixed-geom "pile" — cylinder, ellipsoid, mesh
//! tetrahedron, and a yawed box (stacked as two yawed boxes to showcase
//! the box-box edge-edge SAT completion). Each object settles onto its
//! own patch of the ground plane — cross-object pairs among
//! cylinder/ellipsoid/mesh are DEFERRED (see `docs/contacts.md`), so the
//! scene is arranged to avoid those unsupported pairs during settle.
//!
//! Run:
//! ```text
//! cargo run --release --example pile -- --frames 800 --out /tmp/pile.ppm
//! ```
//!
//! Wireframes:
//!   - cylinder: two 16-segment rings at the caps plus 8 axial ribs.
//!   - ellipsoid: three 24-segment great circles (XY, YZ, XZ).
//!   - mesh: face edges (the tetrahedron has 6 edges).
//!   - box: 12 edges.

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::body::Body;
use newt::geom::{ConvexMesh, Geom};
use newt::math::{FRAC_PI_4, Mat3, Quat, TAU, Vec3, cos, sin};
use newt::world::World;

use std::path::PathBuf;

fn parse_args() -> (usize, PathBuf, (usize, usize)) {
    let mut frames = 800usize;
    let mut out = PathBuf::from("newt-pile.ppm");
    let mut size = (800usize, 480usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => frames = args.next().unwrap().parse().unwrap(),
            "--out" => out = PathBuf::from(args.next().unwrap()),
            "--size" => {
                let s = args.next().unwrap();
                let (w, h) = s.split_once('x').expect("--size WxH");
                size = (w.parse().unwrap(), h.parse().unwrap());
            }
            _ => panic!("unknown arg: {a}"),
        }
    }
    (frames, out, size)
}

// ---------------------------------------------------------------------------
// scene
// ---------------------------------------------------------------------------

const CYL_RADIUS: f32 = 0.30;
const CYL_HALF_H: f32 = 0.35;
const ELL_AXES: Vec3 = Vec3::new(0.35, 0.25, 0.40);
const BOX_HALF: Vec3 = Vec3::new(0.25, 0.25, 0.25);
const TETRA_SCALE: f32 = 0.5;

fn build_world() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.7));

    // Mesh: scaled unit tetrahedron with a face at z=0 so it can rest on
    // that face. Vertices v0/v1/v2 lie at z=0, v3 at z=1 (before scaling).
    let mesh = ConvexMesh {
        vertices: vec![
            Vec3::new(0.0, 0.0, 0.0) * TETRA_SCALE,
            Vec3::new(1.0, 0.0, 0.0) * TETRA_SCALE,
            Vec3::new(0.0, 1.0, 0.0) * TETRA_SCALE,
            Vec3::new(0.0, 0.0, 1.0) * TETRA_SCALE,
        ],
        faces: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    };
    let mesh_id = world.add_mesh(mesh);

    // Cylinder (position in the -X/-Y quadrant).
    let m = 1.0f32;
    let cyl_body = Body::new(
        m,
        newt::geom::solid_cylinder_inertia(m, CYL_RADIUS, CYL_HALF_H),
        Vec3::new(-0.9, -0.9, 1.4),
        Quat::from_axis_angle(Vec3::X, 0.15),
    );
    let ic = world.add_body(cyl_body);
    world.add_geom(Geom::cylinder(
        ic,
        CYL_RADIUS,
        CYL_HALF_H,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.7,
    ));

    // Ellipsoid (+X/-Y).
    let ell_body = Body::new(
        m,
        newt::geom::solid_ellipsoid_inertia(m, ELL_AXES),
        Vec3::new(0.9, -0.9, 1.5),
        Quat::from_axis_angle(Vec3::Y, 0.35),
    );
    let ie = world.add_body(ell_body);
    world.add_geom(Geom::ellipsoid(
        ie,
        ELL_AXES,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.7,
    ));

    // Mesh tetrahedron (-X/+Y).
    let mesh_body = Body::new(
        m,
        Mat3::diag(0.02, 0.02, 0.02),
        Vec3::new(-0.9, 0.9, 1.5),
        Quat::IDENTITY,
    );
    let im = world.add_body(mesh_body);
    world.add_geom(Geom::mesh(im, mesh_id, Vec3::ZERO, Quat::IDENTITY, 0.7));

    // Two yawed boxes stacked (+X/+Y): the lower is Rot_z(π/4), the upper
    // is identity — 45° relative yaw. This is the NEWT-5-incident-closure
    // configuration; before edge-edge SAT the upper collapsed through.
    let lower_box = Body::solid_box(
        m,
        BOX_HALF,
        Vec3::new(0.9, 0.9, 0.3),
        Quat::from_axis_angle(Vec3::Z, FRAC_PI_4),
    );
    let ibl = world.add_body(lower_box);
    world.add_geom(Geom::r#box(ibl, BOX_HALF, Vec3::ZERO, Quat::IDENTITY, 0.7));
    let upper_box = Body::solid_box(m, BOX_HALF, Vec3::new(0.92, 0.91, 1.0), Quat::IDENTITY);
    let ibu = world.add_body(upper_box);
    world.add_geom(Geom::r#box(ibu, BOX_HALF, Vec3::ZERO, Quat::IDENTITY, 0.7));

    // Add EXPLICIT pair list restricted to supported same-object-vs-plane
    // and box-vs-box pairs — auto_pairs would enumerate every
    // cylinder-vs-ellipsoid etc combination, which are deferred and would
    // silently no-op. This makes the demo honest about what's implemented.
    let plane = 0;
    let cyl_g = 1;
    let ell_g = 2;
    let mesh_g = 3;
    let lower_box_g = 4;
    let upper_box_g = 5;
    world.pair_list = Some(vec![
        (plane, cyl_g),
        (plane, ell_g),
        (plane, mesh_g),
        (plane, lower_box_g),
        (plane, upper_box_g),
        // The yawed stack: box-vs-box (supported with edge-edge fallback).
        (lower_box_g, upper_box_g),
    ]);

    world
}

// ---------------------------------------------------------------------------
// wireframe rendering (chimy2)
// ---------------------------------------------------------------------------

fn draw_line(fb: &mut Framebuffer, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: u32) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let w = fb.width as i32;
    let h = fb.height as i32;
    loop {
        if x0 >= 0 && y0 >= 0 && x0 < w && y0 < h {
            fb.put_pixel(x0 as usize, y0 as usize, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn project(camera: Mat4, world_pt: Vec3, width: usize, height: usize) -> Option<(i32, i32)> {
    let clip = camera * Vec4::new(world_pt.x, world_pt.y, world_pt.z, 1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let ndc_z = clip.z / clip.w;
    if !(-1.0..=1.0).contains(&ndc_z) {
        return None;
    }
    let sx = (ndc_x * 0.5 + 0.5) * (width as f32);
    let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * (height as f32);
    Some((sx as i32, sy as i32))
}

fn draw_ground_grid(fb: &mut Framebuffer, camera: Mat4, width: usize, height: usize, color: u32) {
    let span = 2.5f32;
    let step = 0.5f32;
    let n = (2.0 * span / step) as i32;
    for i in 0..=n {
        let t = -span + (i as f32) * step;
        let a = Vec3::new(-span, t, 0.0);
        let b = Vec3::new(span, t, 0.0);
        if let (Some(p0), Some(p1)) = (
            project(camera, a, width, height),
            project(camera, b, width, height),
        ) {
            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
        }
        let a = Vec3::new(t, -span, 0.0);
        let b = Vec3::new(t, span, 0.0);
        if let (Some(p0), Some(p1)) = (
            project(camera, a, width, height),
            project(camera, b, width, height),
        ) {
            draw_line(fb, p0.0, p0.1, p1.0, p1.1, color);
        }
    }
}

fn body_local_to_world(body: &Body, p_local: Vec3) -> Vec3 {
    body.position + body.orientation.rotate(p_local)
}

/// Cylinder wireframe: cap rings + axial ribs.
#[allow(clippy::too_many_arguments)]
fn draw_cylinder(
    fb: &mut Framebuffer,
    camera: Mat4,
    w: usize,
    h: usize,
    body: &Body,
    r: f32,
    hh: f32,
    color: u32,
) {
    const SEG: usize = 16;
    let step = TAU / SEG as f32;
    let ring_pts = |z: f32| -> Vec<Vec3> {
        (0..SEG)
            .map(|i| {
                let t = i as f32 * step;
                body_local_to_world(body, Vec3::new(r * cos(t), r * sin(t), z))
            })
            .collect()
    };
    let top = ring_pts(hh);
    let bot = ring_pts(-hh);
    for i in 0..SEG {
        let j = (i + 1) % SEG;
        if let (Some(a), Some(b)) = (project(camera, top[i], w, h), project(camera, top[j], w, h)) {
            draw_line(fb, a.0, a.1, b.0, b.1, color);
        }
        if let (Some(a), Some(b)) = (project(camera, bot[i], w, h), project(camera, bot[j], w, h)) {
            draw_line(fb, a.0, a.1, b.0, b.1, color);
        }
    }
    // 8 axial ribs.
    for k in 0..8 {
        let i = (k * SEG) / 8;
        if let (Some(a), Some(b)) = (project(camera, top[i], w, h), project(camera, bot[i], w, h)) {
            draw_line(fb, a.0, a.1, b.0, b.1, color);
        }
    }
}

/// Ellipsoid wireframe: 3 great circles in body-frame planes XY, YZ, XZ.
fn draw_ellipsoid(
    fb: &mut Framebuffer,
    camera: Mat4,
    w: usize,
    h: usize,
    body: &Body,
    sa: Vec3,
    color: u32,
) {
    const SEG: usize = 24;
    let step = TAU / SEG as f32;
    let circle = |plane: u8| -> Vec<Vec3> {
        (0..SEG)
            .map(|i| {
                let t = i as f32 * step;
                let c = cos(t);
                let s = sin(t);
                let p_local = match plane {
                    0 => Vec3::new(sa.x * c, sa.y * s, 0.0),
                    1 => Vec3::new(0.0, sa.y * c, sa.z * s),
                    _ => Vec3::new(sa.x * c, 0.0, sa.z * s),
                };
                body_local_to_world(body, p_local)
            })
            .collect()
    };
    for plane in 0u8..3 {
        let pts = circle(plane);
        for i in 0..SEG {
            let j = (i + 1) % SEG;
            if let (Some(a), Some(b)) =
                (project(camera, pts[i], w, h), project(camera, pts[j], w, h))
            {
                draw_line(fb, a.0, a.1, b.0, b.1, color);
            }
        }
    }
}

/// Mesh wireframe: unique edges of the triangular faces (no dedup — the
/// unit tetrahedron has 6 unique edges but iterating faces re-draws each
/// twice; harmless for wireframe).
fn draw_mesh(
    fb: &mut Framebuffer,
    camera: Mat4,
    w: usize,
    h: usize,
    body: &Body,
    mesh: &ConvexMesh,
    color: u32,
) {
    for face in &mesh.faces {
        let vs = [
            body_local_to_world(body, mesh.vertices[face[0] as usize]),
            body_local_to_world(body, mesh.vertices[face[1] as usize]),
            body_local_to_world(body, mesh.vertices[face[2] as usize]),
        ];
        for k in 0..3 {
            let a = vs[k];
            let b = vs[(k + 1) % 3];
            if let (Some(pa), Some(pb)) = (project(camera, a, w, h), project(camera, b, w, h)) {
                draw_line(fb, pa.0, pa.1, pb.0, pb.1, color);
            }
        }
    }
}

/// Box wireframe: 12 edges of an OBB with given half-extents.
fn draw_box(
    fb: &mut Framebuffer,
    camera: Mat4,
    w: usize,
    h: usize,
    body: &Body,
    half: Vec3,
    color: u32,
) {
    const CORNERS: [(f32, f32, f32); 8] = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ];
    const EDGES: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let corners: [Vec3; 8] = std::array::from_fn(|i| {
        let (sx, sy, sz) = CORNERS[i];
        body_local_to_world(body, Vec3::new(sx * half.x, sy * half.y, sz * half.z))
    });
    for &(a, b) in EDGES.iter() {
        if let (Some(pa), Some(pb)) = (
            project(camera, corners[a], w, h),
            project(camera, corners[b], w, h),
        ) {
            draw_line(fb, pa.0, pa.1, pb.0, pb.1, color);
        }
    }
}

fn render(world: &World, width: usize, height: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.1,
        200.0,
    ) * Mat4::look_at(
        CVec3::new(3.2, -3.5, 2.4),
        CVec3::new(0.0, 0.0, 0.4),
        CVec3::new(0.0, 0.0, 1.0),
    );
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));

    // Body 0: cylinder (goldenrod). 1: ellipsoid (cyan). 2: mesh (magenta).
    // 3: lower box (yaw, orange). 4: upper box (green).
    let cyl_color = argb8888(0xff, 240, 200, 90);
    let ell_color = argb8888(0xff, 90, 220, 240);
    let mesh_color = argb8888(0xff, 240, 100, 200);
    let box_lo_color = argb8888(0xff, 240, 140, 80);
    let box_up_color = argb8888(0xff, 130, 220, 140);

    draw_cylinder(
        &mut fb,
        camera,
        width,
        height,
        &world.bodies[0],
        CYL_RADIUS,
        CYL_HALF_H,
        cyl_color,
    );
    draw_ellipsoid(
        &mut fb,
        camera,
        width,
        height,
        &world.bodies[1],
        ELL_AXES,
        ell_color,
    );
    // Mesh: read the mesh from world.meshes (the only registered mesh is idx 0).
    let mesh = &world.meshes[0];
    draw_mesh(
        &mut fb,
        camera,
        width,
        height,
        &world.bodies[2],
        mesh,
        mesh_color,
    );
    draw_box(
        &mut fb,
        camera,
        width,
        height,
        &world.bodies[3],
        BOX_HALF,
        box_lo_color,
    );
    draw_box(
        &mut fb,
        camera,
        width,
        height,
        &world.bodies[4],
        BOX_HALF,
        box_up_color,
    );

    fb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height)) = parse_args();
    let mut world = build_world();
    // Sanity: no pair should be flagged as unsupported (we constrained the
    // pair list to supported combinations). Panic if a change to the scene
    // introduces an unsupported pair.
    let unsupported = world.validate_supported_pairs();
    if !unsupported.is_empty() {
        panic!(
            "pile scene contains unsupported pairs: {:?}. Update the scene \
             or the deferred-pair list.",
            unsupported
        );
    }
    for _ in 0..frames {
        world.step();
    }
    let fb = render(&world, width, height);
    write_ppm(&out, &fb)?;
    // Report final positions so the human running the demo can eyeball
    // whether everything settled (no NaN, no negative z, boxes stacked).
    println!(
        "wrote {} ({}x{}) — final positions:",
        out.display(),
        width,
        height
    );
    let labels = [
        "cylinder",
        "ellipsoid",
        "mesh_tetra",
        "yaw_box_lower",
        "box_upper",
    ];
    for (i, body) in world.bodies.iter().enumerate() {
        let name = labels.get(i).unwrap_or(&"?");
        let p = body.position;
        // Also compute a tilt indicator so the human sees whether the object
        // is upright: 1 − |q.w|. Small = upright.
        let tilt = 1.0 - body.orientation.w.abs();
        println!(
            "  {name}: pos = ({:.3}, {:.3}, {:.3}), tilt = {tilt:.3}, |v| = {:.3}",
            p.x,
            p.y,
            p.z,
            body.linear_velocity.length()
        );
    }
    // Golden path check: upper box must not have collapsed through lower.
    let z_lower = world.bodies[3].position.z;
    let z_upper = world.bodies[4].position.z;
    if z_upper < z_lower + 0.3 {
        eprintln!(
            "WARNING: upper box collapsed through lower ({z_upper} < {z_lower} + 0.3) — \
             box-box edge-edge SAT regressed?"
        );
    } else {
        println!("verdict: yawed stack holds ({z_upper:.3} > {z_lower:.3} + 0.3)");
    }
    Ok(())
}
