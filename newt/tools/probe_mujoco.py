#!/usr/bin/env python3
"""Instrumentation probe for a differential scenario, MuJoCo side.

Loads a scenario's MJCF under real MuJoCo, steps it, and writes a TSV
with per-sample:
- step index
- per-body world position + world quat (bottom, middle, top)
- per-body world linear + world angular velocity
- number of active contacts (dist < 0)
- per-contact fields: geom1, geom2, pos (world), frame-normal (world), dist,
  and normal-force magnitude (mj_contactForce)

Runs against the biped venv (re-execs like capture_mujoco.py does).

USAGE
    python probe_mujoco.py box_stack /tmp/newt-probe/mj_box_stack.tsv
    python probe_mujoco.py sphere_drop /tmp/newt-probe/mj_sphere_drop.tsv --stride 1

Optional flags:
    --iterations N     override PGS iterations (default: MJCF value)
    --solref TC DR     override every geom's solref (default: MJCF values)
    --solimp DMIN DMAX WIDTH   override every geom's solimp
    --n-steps N        override the total step count
    --stride K         record every K'th step (default: MJCF-scenario stride)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

DEFAULT_VENV_PYTHON = Path.home() / "me/fun/biped/.venv/bin/python"


def _reexec_under_venv(venv_python: Path, argv: list[str]) -> None:
    if not venv_python.is_file():
        return
    if str(Path(sys.executable)) == str(venv_python):
        return
    os.execv(str(venv_python), [str(venv_python), *argv])


def _load_scenarios(refs_dir: Path) -> list[dict]:
    with open(refs_dir / "scenarios.json", "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["scenarios"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("out")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--solref", type=float, nargs=2, default=None,
                        metavar=("TIMECONST", "DAMPRATIO"))
    parser.add_argument("--solimp", type=float, nargs=3, default=None,
                        metavar=("DMIN", "DMAX", "WIDTH"))
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--venv", default=str(DEFAULT_VENV_PYTHON))
    args = parser.parse_args(argv[1:])

    if args.venv:
        _reexec_under_venv(Path(args.venv), argv)

    import mujoco  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    refs_dir = Path(__file__).resolve().parent.parent / "tests" / "references"
    scenarios = _load_scenarios(refs_dir)
    spec = next((s for s in scenarios if s["name"] == args.scenario), None)
    if spec is None:
        print(f"unknown scenario {args.scenario}", file=sys.stderr)
        return 1

    mjcf_path = refs_dir / spec["mjcf"]
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    if args.iterations is not None:
        model.opt.iterations = args.iterations
    if args.solref is not None:
        # Override every geom's solref (positive-timeconst form).
        tc, dr = args.solref
        model.geom_solref[:, 0] = tc
        model.geom_solref[:, 1] = dr
    if args.solimp is not None:
        dmin, dmax, width = args.solimp
        model.geom_solimp[:, 0] = dmin
        model.geom_solimp[:, 1] = dmax
        model.geom_solimp[:, 2] = width

    if "init_qpos" in spec:
        init = np.asarray(spec["init_qpos"], dtype="float64")
        data.qpos[:] = init
    if "init_qvel" in spec:
        init = np.asarray(spec["init_qvel"], dtype="float64")
        data.qvel[:] = init
    if "actuator_targets" in spec:
        arr = np.zeros(model.nu, dtype="float64")
        for name, ctrl in spec["actuator_targets"].items():
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            arr[act_id] = float(ctrl)
        data.ctrl[:] = arr

    n_steps = args.n_steps if args.n_steps is not None else int(spec["n_steps"])
    stride = args.stride if args.stride is not None else int(spec["stride"])

    n_free_bodies = model.nq // 7 if model.nq % 7 == 0 else 0

    mujoco.mj_forward(model, data)

    lines = []
    header_cols = [
        "step",
        "n_active_contacts",
    ]
    for b in range(model.nbody - 1):  # skip world body
        header_cols += [
            f"body{b}_px", f"body{b}_py", f"body{b}_pz",
            f"body{b}_qw", f"body{b}_qx", f"body{b}_qy", f"body{b}_qz",
            f"body{b}_vx", f"body{b}_vy", f"body{b}_vz",
            f"body{b}_wx", f"body{b}_wy", f"body{b}_wz",
        ]
    lines.append("\t".join(header_cols))

    def snapshot(step_idx):
        # Force fresh derived quantities incl. contact forces.
        mujoco.mj_forward(model, data)
        # Body-frame data (skip world body 0). MuJoCo body layout:
        # data.xpos[b], data.xquat[b] (w,x,y,z), data.cvel[b] = 6-vec
        # (angular then linear in world coords).
        cols = [str(step_idx)]
        active = 0
        for c in range(data.ncon):
            if data.contact.dist[c] < 0:
                active += 1
        cols.append(str(active))
        for b in range(1, model.nbody):
            px, py, pz = data.xpos[b]
            qw, qx, qy, qz = data.xquat[b]
            cvel = data.cvel[b]  # (wx wy wz vx vy vz) in world coords
            wx, wy, wz = cvel[0], cvel[1], cvel[2]
            vx, vy, vz = cvel[3], cvel[4], cvel[5]
            cols += [f"{px:.9g}", f"{py:.9g}", f"{pz:.9g}",
                     f"{qw:.9g}", f"{qx:.9g}", f"{qy:.9g}", f"{qz:.9g}",
                     f"{vx:.9g}", f"{vy:.9g}", f"{vz:.9g}",
                     f"{wx:.9g}", f"{wy:.9g}", f"{wz:.9g}"]
        # Per-contact block (append; may be empty).
        for c in range(data.ncon):
            if data.contact.dist[c] >= 0:
                continue
            g1 = int(data.contact.geom[c][0])
            g2 = int(data.contact.geom[c][1])
            pos = data.contact.pos[c]
            frame = data.contact.frame[c]  # 9-vec, first row is normal
            nx, ny, nz = frame[0], frame[1], frame[2]
            dist = float(data.contact.dist[c])
            f6 = np.zeros(6, dtype="float64")
            mujoco.mj_contactForce(model, data, c, f6)
            fn = float(f6[0])  # normal component in contact frame
            ft1 = float(f6[1])
            ft2 = float(f6[2])
            cols += [
                "CONTACT",
                str(g1), str(g2),
                f"{pos[0]:.9g}", f"{pos[1]:.9g}", f"{pos[2]:.9g}",
                f"{nx:.9g}", f"{ny:.9g}", f"{nz:.9g}",
                f"{dist:.9g}",
                f"{fn:.9g}", f"{ft1:.9g}", f"{ft2:.9g}",
            ]
        lines.append("\t".join(cols))

    snapshot(0)
    for step in range(1, n_steps + 1):
        mujoco.mj_step(model, data)
        if step % stride == 0:
            snapshot(step)

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)-1} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
