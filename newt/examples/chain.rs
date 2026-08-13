//! Tier 3 demo: 5-link chain hanging from a fixed root, swinging under
//! gravity, and settling onto a static plane below. Exercises the
//! joints+contacts coupling: each link carries a capsule collider, and the
//! plane at z=0 stops the chain from falling through.
//!
//! Run:
//! ```text
//! cargo run --release --example chain -- --frames 1200 --out /tmp/chain.ppm
//! ```

use chimy2::demo::write_ppm;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3 as CVec3, Vec4};

use newt::geom::Geom;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, forward_kinematics};
use newt::world::World;

use std::path::PathBuf;

fn parse_args() -> (usize, PathBuf, (usize, usize)) {
    let mut frames = 1200usize;
    let mut out = PathBuf::from("newt-chain.ppm");
    let mut size = (640usize, 360usize);
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

const N_LINKS: usize = 5;
// Link geometry: a short capsule aligned along local z.
const LINK_LEN: f32 = 0.35; // COM offset from either endpoint = LINK_LEN
const LINK_RADIUS: f32 = 0.05;
const LINK_MASS: f32 = 0.3;

fn build_world() -> World {
    let mut world = World::new();
    // dt = 2 ms keeps the penalty-contact spring stable when the whole
    // chain lands on the plane (default SolRef timeconst 0.02 s → contact
    // period 0.125 s; we want ≳20 samples per period during the impact).
    world.dt = 0.002;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    // Static plane at z = 0.
    let plane_geom = world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.6));

    // Chain: root fixed at (0, 0, 2.5). Each following link joins with a
    // hinge about +x, joint anchor at the previous link's bottom.
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        // Chain length ≈ 5 * 0.7 m = 3.5 m; anchor at 3 m so a straight
        // hang would push the bottom half a meter below the plane, so the
        // contact spring engages and the last few links pile onto the
        // ground.
        (Vec3::new(0.0, 0.0, 3.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let i_perp = (1.0 / 12.0) * LINK_MASS * (2.0 * LINK_LEN) * (2.0 * LINK_LEN);
    for i in 0..N_LINKS {
        // Joint 1 sits at the root; subsequent joints at the bottom of the
        // previous link (which lies at (0, 0, -LINK_LEN) in that link's
        // body frame — COM is at the middle).
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, 0.0, -LINK_LEN)
        };
        // Alternate hinge axes to break symmetry (chain buckles out of the
        // vertical plane if we ever kick it sideways). Uses only +x for
        // in-plane demo; can add +y for out-of-plane in a variant.
        tree.push_link(Link::new(
            Some(i),
            JointKind::Hinge {
                axis: Vec3::X,
                range: None,
                damping: 0.08,
                armature: 0.0,
                limit: newt::joint::HingeLimit::DEFAULT,
            },
            (parent_anchor, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, LINK_LEN), Quat::IDENTITY),
            LINK_MASS,
            Mat3::diag(i_perp, i_perp, 1e-6),
        ));
    }
    // Small initial angle on the first joint to start the swing.
    tree.set_hinge_angle(1, 1.2);
    let tree_idx = world.add_tree(tree);

    // Attach a capsule collider to each dynamic link. Record their geom
    // indices so we can build an explicit pair_list that excludes chain
    // self-collision (adjacent capsules would interpenetrate at every
    // joint, which is a runaway force in the v0 penalty model).
    let mut capsule_geoms = Vec::new();
    for l in 1..=N_LINKS {
        capsule_geoms.push(world.add_geom(Geom::capsule_on_link(
            tree_idx,
            l,
            LINK_RADIUS,
            LINK_LEN - LINK_RADIUS,
            Vec3::ZERO,
            Quat::IDENTITY,
            0.7,
        )));
    }
    // Only capsule-vs-plane contacts fire. `(min, max)` per pair, sorted.
    let mut pairs: Vec<(usize, usize)> = capsule_geoms
        .iter()
        .map(|&c| {
            if c < plane_geom {
                (c, plane_geom)
            } else {
                (plane_geom, c)
            }
        })
        .collect();
    pairs.sort();
    world.pair_list = Some(pairs);
    world
}

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
    let span = 2.5;
    let step = 0.5;
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

fn render(world: &World, width: usize, height: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    fb.clear(argb8888(0xff, 12, 14, 22));
    let camera = Mat4::perspective(
        std::f32::consts::FRAC_PI_4,
        (width as f32) / (height as f32).max(1.0),
        0.05,
        100.0,
    ) * Mat4::look_at(
        CVec3::new(4.0, -0.5, 1.5),
        CVec3::new(0.0, 0.0, 0.8),
        CVec3::new(0.0, 0.0, 1.0),
    );
    draw_ground_grid(&mut fb, camera, width, height, argb8888(0xff, 40, 45, 55));
    let poses = forward_kinematics(&world.trees[0]);
    let colors = [
        argb8888(0xff, 240, 200, 90),
        argb8888(0xff, 220, 170, 100),
        argb8888(0xff, 200, 140, 130),
        argb8888(0xff, 170, 120, 170),
        argb8888(0xff, 130, 130, 220),
    ];
    for l in 1..=N_LINKS {
        let (com, ori) = poses[l];
        let top = com + ori.rotate(Vec3::new(0.0, 0.0, LINK_LEN));
        let bot = com + ori.rotate(Vec3::new(0.0, 0.0, -LINK_LEN));
        if let (Some(p0), Some(p1)) = (
            project(camera, top, width, height),
            project(camera, bot, width, height),
        ) {
            draw_line(
                &mut fb,
                p0.0,
                p0.1,
                p1.0,
                p1.1,
                colors[(l - 1) % colors.len()],
            );
        }
    }
    fb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (frames, out, (width, height)) = parse_args();
    let mut world = build_world();
    for _ in 0..frames {
        world.step();
    }
    let fb = render(&world, width, height);
    write_ppm(&out, &fb)?;
    let final_link_pos = world.tree_link_pose(0, N_LINKS).0;
    println!(
        "wrote {} ({}x{}) — bottom link COM = {final_link_pos:?}",
        out.display(),
        width,
        height,
    );
    Ok(())
}
