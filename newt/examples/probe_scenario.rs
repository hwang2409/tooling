//! Diagnostic probe: run a differential scenario under newt and dump
//! per-sample body state + contact set to TSV. Mirrors the columns
//! `tools/probe_mujoco.py` produces so a plain diff surfaces where the
//! two engines disagree (contact ordering, contact-point placement,
//! solver normal-force magnitude, or trajectory).
//!
//! Usage:
//! ```text
//! cargo run --release --example probe_scenario -- box_stack /tmp/newt-probe/nt_box_stack.tsv
//! cargo run --release --example probe_scenario -- sphere_drop /tmp/nt.tsv --iterations 200
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use newt::contact::Contact;
use newt::geom::{GeomAttach, SolRef};
use newt::model::Scene;
use newt::solver::SolImp;
use newt::world::World;

fn refs_dir() -> PathBuf {
    // Reach from newt/target/... back to newt/tests/references.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("references")
}

struct Args {
    scenario: String,
    out: PathBuf,
    iterations: Option<u32>,
    solref: Option<(f32, f32)>,
    solimp: Option<(f32, f32, f32)>,
    n_steps: Option<u32>,
    stride: Option<u32>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let scenario = it.next().expect("scenario name required");
    let out = PathBuf::from(it.next().expect("output path required"));
    let mut a = Args {
        scenario,
        out,
        iterations: None,
        solref: None,
        solimp: None,
        n_steps: None,
        stride: None,
    };
    while let Some(k) = it.next() {
        match k.as_str() {
            "--iterations" => a.iterations = Some(it.next().unwrap().parse().unwrap()),
            "--solref" => {
                let tc = it.next().unwrap().parse().unwrap();
                let dr = it.next().unwrap().parse().unwrap();
                a.solref = Some((tc, dr));
            }
            "--solimp" => {
                let dmin = it.next().unwrap().parse().unwrap();
                let dmax = it.next().unwrap().parse().unwrap();
                let width = it.next().unwrap().parse().unwrap();
                a.solimp = Some((dmin, dmax, width));
            }
            "--n-steps" => a.n_steps = Some(it.next().unwrap().parse().unwrap()),
            "--stride" => a.stride = Some(it.next().unwrap().parse().unwrap()),
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

#[derive(Debug, Clone, Copy)]
struct ScenarioSpec {
    n_steps: u32,
    stride: u32,
    mjcf: &'static str,
}

fn scenario_spec(name: &str) -> ScenarioSpec {
    match name {
        "box_stack" => ScenarioSpec {
            n_steps: 2000,
            stride: 100,
            mjcf: "box_stack.xml",
        },
        "sphere_drop" => ScenarioSpec {
            n_steps: 1500,
            stride: 50,
            mjcf: "sphere_drop.xml",
        },
        other => panic!("no probe spec for {other}"),
    }
}

fn main() {
    let args = parse_args();
    let spec = scenario_spec(&args.scenario);
    let mjcf_path = refs_dir().join(spec.mjcf);
    let src = fs::read_to_string(&mjcf_path).expect("mjcf");
    let mut scene: Scene = newt::mjcf::load_mjcf_str(&src).expect("mjcf parse");
    if let Some(iter) = args.iterations {
        scene.world.solver.iterations = iter;
    }
    if let Some((tc, dr)) = args.solref {
        for g in scene.world.geoms.iter_mut() {
            g.solref = SolRef::new(tc, dr);
        }
    }
    if let Some((dmin, dmax, width)) = args.solimp {
        for g in scene.world.geoms.iter_mut() {
            g.solimp = SolImp::new(dmin, dmax, width, g.solimp.midpoint, g.solimp.power);
        }
    }
    let n_steps = args.n_steps.unwrap_or(spec.n_steps);
    let stride = args.stride.unwrap_or(spec.stride);

    let mut lines: Vec<String> = Vec::new();
    let mut header = String::from("step\tn_active_contacts");
    for b in 0..scene.world.bodies.len() {
        for suffix in [
            "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz",
        ] {
            header.push('\t');
            header.push_str(&format!("body{}_{}", b, suffix));
        }
    }
    lines.push(header);

    snapshot(&scene.world, 0, &mut lines);
    for step in 1..=n_steps {
        scene.world.step();
        if step % stride == 0 {
            snapshot(&scene.world, step, &mut lines);
        }
    }
    fs::write(&args.out, lines.join("\n") + "\n").expect("write");
    println!("wrote {} ({} samples)", args.out.display(), lines.len() - 1);
}

fn snapshot(world: &World, step: u32, lines: &mut Vec<String>) {
    let contacts = world.detect_contacts();
    // Report every contact with penetration > gap (matches solver's active
    // criterion in solve_free_bodies).
    let active: Vec<&Contact> = contacts
        .iter()
        .filter(|c| c.penetration - c.gap > 0.0)
        .collect();
    let mut cols: Vec<String> = Vec::new();
    cols.push(step.to_string());
    cols.push(active.len().to_string());
    for b in &world.bodies {
        cols.push(fmt(b.position.x));
        cols.push(fmt(b.position.y));
        cols.push(fmt(b.position.z));
        // MJ layout for compare: (qw, qx, qy, qz)
        cols.push(fmt(b.orientation.w));
        cols.push(fmt(b.orientation.x));
        cols.push(fmt(b.orientation.y));
        cols.push(fmt(b.orientation.z));
        cols.push(fmt(b.linear_velocity.x));
        cols.push(fmt(b.linear_velocity.y));
        cols.push(fmt(b.linear_velocity.z));
        // World-frame angular velocity for compare against MuJoCo cvel.
        let w_world = b.orientation.rotate(b.angular_velocity_body);
        cols.push(fmt(w_world.x));
        cols.push(fmt(w_world.y));
        cols.push(fmt(w_world.z));
    }
    // Contact block.
    for c in &active {
        cols.push("CONTACT".to_string());
        cols.push(c.geom_a.to_string());
        cols.push(c.geom_b.to_string());
        cols.push(fmt(c.position_world.x));
        cols.push(fmt(c.position_world.y));
        cols.push(fmt(c.position_world.z));
        cols.push(fmt(c.normal_world.x));
        cols.push(fmt(c.normal_world.y));
        cols.push(fmt(c.normal_world.z));
        // MuJoCo prints dist (negative when penetrating); newt exposes
        // penetration (positive). Emit as negative so columns align.
        cols.push(fmt(-(c.penetration - c.gap)));
        // Normal force placeholder (compute below).
        cols.push("NA".to_string());
        cols.push("NA".to_string());
        cols.push("NA".to_string());
    }
    lines.push(cols.join("\t"));
    // Also emit a same-step block with the PGS-computed normal force per
    // active contact for the free-body pool (mirrors MJ's mj_contactForce).
    if world.solver.mode == newt::solver::SolverMode::Pgs {
        emit_solver_contact_forces(world, step, lines);
    }
}

fn emit_solver_contact_forces(world: &World, step: u32, lines: &mut Vec<String>) {
    // Compute per-contact normal force via solve_free_bodies_diag on the
    // free-body pool.
    let contacts_full = world.detect_contacts();
    // Filter to free-body-only contacts (drop link-attached).
    let mut free_body_contacts: Vec<Contact> = Vec::new();
    for c in &contacts_full {
        let att_a = world.geoms[c.geom_a].attachment();
        let att_b = world.geoms[c.geom_b].attachment();
        if matches!(att_a, GeomAttach::Link(_, _)) || matches!(att_b, GeomAttach::Link(_, _)) {
            continue;
        }
        if c.penetration - c.gap <= 0.0 {
            continue;
        }
        free_body_contacts.push(*c);
    }
    let (_wrenches, normal_forces) = newt::solver::solve_free_bodies_diag(
        &world.bodies,
        &world.geoms,
        &free_body_contacts,
        &world.equalities,
        world.gravity,
        world.dt,
        world.solver.cone,
        world.solver.iterations,
    );
    let mut extra = format!("SOLVER_FORCES\t{}\t{}", step, free_body_contacts.len());
    for (c, fn_val) in free_body_contacts.iter().zip(normal_forces.iter()) {
        extra.push_str(&format!("\t{}:{}:{:.9e}", c.geom_a, c.geom_b, fn_val));
    }
    lines.push(extra);
}

fn fmt(x: f32) -> String {
    // Match probe_mujoco.py's %.9g formatting closely enough for diff.
    format!("{:.9e}", x as f64)
}
