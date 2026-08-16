#!/usr/bin/env python3
"""Capture reference MuJoCo trajectories for the newt differential harness.

This script is TOOLING, not part of the engine. It lives outside newt/src so
the engine's zero-dependency, libm-free rule is unaffected. The Rust
comparison suite reads the fixtures this script writes; regenerating them
is a manual step done on a machine that has the biped venv (real MuJoCo)
installed.

USAGE
-----

  python capture_mujoco.py                    # capture ALL scenarios
  python capture_mujoco.py sphere_drop        # capture one scenario
  python capture_mujoco.py --force            # ignore mujoco-version mismatch
  python capture_mujoco.py --list             # print scenarios and exit
  python capture_mujoco.py --venv PATH        # bootstrap check against a venv
  python capture_mujoco.py --row-diagnostics PATH
  python capture_mujoco.py --solref-sweep PATH

The default venv is ~/me/fun/biped/.venv/bin/python; if you invoke this
script under a different Python (e.g. the venv's own), the venv check is
skipped (`sys.executable` already IS the venv).

FIXTURE FORMAT
--------------

Binary little-endian:

    magic         8 bytes  "NEWTDIF1"
    prov_len      u32      length of provenance line
    provenance    utf8     one line: name|mujoco=X.Y.Z|dt=..|integ=..|
                           solver=..|cone=..|iters=..|nq=..|nv=..|stride=..
                           |n_steps=..|date=YYYY-MM-DD
    nq            u32
    nv            u32
    stride        u32      (samples come every `stride` steps)
    n_samples     u32      (== 1 + n_steps // stride; sample 0 = initial)
    n_steps       u32      (total steps run)
    -- per sample --
    step          u32
    qpos          f64 * nq
    qvel          f64 * nv

Ints are unsigned LE; floats are IEEE 754 f64 LE. Values come from MuJoCo
as f64 and are stored as f64. The Rust reader widens newt's f32 output to
f64 for the comparison.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# venv guard
# ---------------------------------------------------------------------------

DEFAULT_VENV_PYTHON = Path.home() / "me/fun/biped/.venv/bin/python"


def _reexec_under_venv(venv_python: Path, argv: list[str]) -> None:
    """If the current interpreter is not the biped venv, re-exec under it.

    Only triggers when the venv actually exists. Comparison is by
    (unresolved) path string because a venv's `python` is typically a
    symlink into the base install — resolving both sides would falsely
    mark them equal even when the venv's site-packages is what we want.
    """
    if not venv_python.is_file():
        return
    if str(Path(sys.executable)) == str(venv_python):
        return
    # Re-execute with the venv's Python, keeping the same argv.
    os.execv(str(venv_python), [str(venv_python), *argv])


# ---------------------------------------------------------------------------
# fixture writer
# ---------------------------------------------------------------------------

MAGIC = b"NEWTDIF1"


def _write_fixture(
    path: Path,
    provenance: str,
    stride: int,
    n_steps: int,
    qpos_samples: list,  # list[np.ndarray]  (avoiding numpy import at top)
    qvel_samples: list,
) -> None:
    assert len(qpos_samples) == len(qvel_samples)
    assert len(qpos_samples) >= 1
    nq = int(qpos_samples[0].shape[0])
    nv = int(qvel_samples[0].shape[0])
    prov_bytes = provenance.encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(prov_bytes)))
        f.write(prov_bytes)
        f.write(struct.pack("<IIIII", nq, nv, stride, len(qpos_samples), n_steps))
        for i, (qp, qv) in enumerate(zip(qpos_samples, qvel_samples)):
            step = i * stride
            f.write(struct.pack("<I", step))
            f.write(struct.pack(f"<{nq}d", *qp.astype("float64").tolist()))
            f.write(struct.pack(f"<{nv}d", *qv.astype("float64").tolist()))


def _read_provenance(path: Path) -> str | None:
    """Read the provenance line from an existing fixture, or None on any
    mismatch / IO error. Used by the version guard."""
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                return None
            (n,) = struct.unpack("<I", f.read(4))
            return f.read(n).decode("utf-8")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# scenario loader
# ---------------------------------------------------------------------------


def _load_scenarios(refs_dir: Path) -> list[dict]:
    with open(refs_dir / "scenarios.json", "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["scenarios"]


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def _capture_scenario(
    mujoco,
    np,
    scenario: dict,
    refs_dir: Path,
    integrator_override: str | None = None,
    solver_override: str | None = None,
    suffix: str = "",
) -> tuple[Path, str]:
    """Run one scenario and write its fixture. Returns (path, provenance).

    If `scenario["check_kind"] == "energy"`, an additional sidecar
    `<name>_energy.bin` file is written next to the main fixture with
    per-sample MuJoCo kinetic and potential energies (f64 pairs). The
    Rust harness reads this to compare long-horizon energy drift.
    If `scenario["compare_sensors"]` is true, a `<name>_sensors.bin`
    sidecar stores MuJoCo's `sensordata` at each sampled step.
    """
    name = scenario["name"]
    mjcf_path = refs_dir / scenario["mjcf"]
    n_steps = int(scenario["n_steps"])
    stride = int(scenario["stride"])
    check_kind = scenario.get("check_kind", "state")
    want_energy = check_kind == "energy"
    want_sensors = bool(scenario.get("compare_sensors", False))
    if stride <= 0 or n_steps <= 0:
        raise ValueError(f"{name}: stride and n_steps must be > 0")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if integrator_override is not None:
        integrators = {
            "Euler": mujoco.mjtIntegrator.mjINT_EULER,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }
        model.opt.integrator = integrators[integrator_override]
    if solver_override is not None:
        solvers = {
            "PGS": mujoco.mjtSolver.mjSOL_PGS,
            "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }
        model.opt.solver = solvers[solver_override]
    data = mujoco.MjData(model)

    # Apply overrides. Both are applied in MuJoCo layout — the scenarios.json
    # file is authored to that layout by design.
    if "init_qpos" in scenario:
        init = np.asarray(scenario["init_qpos"], dtype="float64")
        if init.shape[0] != model.nq:
            raise ValueError(
                f"{name}: init_qpos length {init.shape[0]} != model.nq {model.nq}"
            )
        data.qpos[:] = init
    if "init_qvel" in scenario:
        init = np.asarray(scenario["init_qvel"], dtype="float64")
        if init.shape[0] != model.nv:
            raise ValueError(
                f"{name}: init_qvel length {init.shape[0]} != model.nv {model.nv}"
            )
        data.qvel[:] = init

    # Actuator targets: if present (name -> ctrl value), apply on every step.
    targets: "np.ndarray | None" = None
    if "actuator_targets" in scenario:
        by_name = scenario["actuator_targets"]
        arr = np.zeros(model.nu, dtype="float64")
        for act_name, ctrl in by_name.items():
            act_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
            )
            if act_id < 0:
                raise ValueError(
                    f"{name}: actuator_targets references unknown actuator {act_name!r}"
                )
            arr[act_id] = float(ctrl)
        targets = arr

    # Forward once so any derived quantities (qacc etc) are consistent
    # with the initial (qpos, qvel) before sampling step 0.
    mujoco.mj_forward(model, data)

    def snap_energy():
        # MuJoCo layout is `data.energy = [potential, kinetic]` — verified
        # empirically (zero qvel yields energy[1] == 0). mj_energyPos and
        # mj_energyVel fill these fields respectively.
        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)
        pot = float(data.energy[0])
        kin = float(data.energy[1])
        return kin, pot

    qpos_samples = [data.qpos.copy()]
    qvel_samples = [data.qvel.copy()]
    energy_samples: list[tuple[float, float]] = []
    sensor_samples = [data.sensordata.copy()] if want_sensors else []
    if want_energy:
        energy_samples.append(snap_energy())

    for step in range(1, n_steps + 1):
        if targets is not None:
            data.ctrl[:] = targets
        mujoco.mj_step(model, data)
        if step % stride == 0:
            qpos_samples.append(data.qpos.copy())
            qvel_samples.append(data.qvel.copy())
            if want_energy:
                energy_samples.append(snap_energy())
            if want_sensors:
                sensor_samples.append(data.sensordata.copy())

    provenance = (
        f"name={name}|mujoco={mujoco.__version__}"
        f"|dt={model.opt.timestep:.9g}"
        f"|integ={int(model.opt.integrator)}"
        f"|solver={int(model.opt.solver)}"
        f"|cone={int(model.opt.cone)}"
        f"|iters={model.opt.iterations}"
        f"|nq={model.nq}|nv={model.nv}"
        f"|stride={stride}|n_steps={n_steps}"
        f"|date={_dt.date.today().isoformat()}"
    )
    path = refs_dir / f"{name}{suffix}.bin"
    _write_fixture(path, provenance, stride, n_steps, qpos_samples, qvel_samples)
    if want_energy:
        energy_path = refs_dir / f"{name}_energy.bin"
        with open(energy_path, "wb") as f:
            for kin, pot in energy_samples:
                f.write(struct.pack("<dd", kin, pot))
    if want_sensors:
        sensor_path = refs_dir / f"{name}_sensors.bin"
        sensor_dim = int(sensor_samples[0].shape[0])
        with open(sensor_path, "wb") as f:
            f.write(struct.pack("<II", len(sensor_samples), sensor_dim))
            for sample in sensor_samples:
                if int(sample.shape[0]) != sensor_dim:
                    raise ValueError(f"{name}: sensordata dimension changed during capture")
                f.write(struct.pack(f"<{sensor_dim}d", *sample.astype("float64").tolist()))
    return path, provenance


def _capture_row_diagnostics(mujoco, np, refs_dir: Path, path: Path) -> None:
    """Dump MuJoCo's first-contact constraint factors for sphere_drop.

    The snapshot is taken after the first step that creates a contact, then
    forwarded once so all efc_* arrays describe the same qpos/qvel state.
    This is a diagnostic artifact, not a trajectory fixture.
    """
    mjcf_path = refs_dir / "sphere_drop.xml"
    triples = [
        (0.005, 0.5),
        (0.010, 1.0),
        (0.020, 1.0),
        (0.050, 1.0),
        (0.100, 1.0),
        (0.200, 2.0),
    ]
    records = []
    for tc, dr in triples:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        model.geom_solref[:, 0] = tc
        model.geom_solref[:, 1] = dr
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for step in range(1, 5000):
            mujoco.mj_step(model, data)
            if data.ncon == 0:
                continue
            mujoco.mj_forward(model, data)
            nefc = int(data.nefc)
            record = {
                "tc": tc,
                "dampratio": dr,
                "step": step,
                "dt": float(model.opt.timestep),
                "nq": int(model.nq),
                "nv": int(model.nv),
                "qpos": data.qpos.astype("float64").tolist(),
                "qvel": data.qvel.astype("float64").tolist(),
                "qacc": data.qacc.astype("float64").tolist(),
                "qfrc_constraint": data.qfrc_constraint.astype("float64").tolist(),
                "nefc": nefc,
                "efc_pos": data.efc_pos[:nefc].astype("float64").tolist(),
                "efc_vel": data.efc_vel[:nefc].astype("float64").tolist(),
                "efc_aref": data.efc_aref[:nefc].astype("float64").tolist(),
                "efc_margin": data.efc_margin[:nefc].astype("float64").tolist(),
                "efc_R": data.efc_R[:nefc].astype("float64").tolist(),
                "efc_D": data.efc_D[:nefc].astype("float64").tolist(),
                "efc_KBIP": data.efc_KBIP[:nefc].reshape(nefc, 4).astype("float64").tolist(),
                "efc_J": data.efc_J.reshape(nefc, model.nv).astype("float64").tolist(),
                "efc_force": data.efc_force[:nefc].astype("float64").tolist(),
                "contacts": [
                    {
                        "geom": [int(g) for g in data.contact[i].geom],
                        "dist": float(data.contact[i].dist),
                        "pos": data.contact[i].pos.astype("float64").tolist(),
                        "frame": data.contact[i].frame.astype("float64").tolist(),
                    }
                    for i in range(data.ncon)
                ],
            }
            records.append(record)
            break
        else:
            raise RuntimeError(f"no contact found for tc={tc}, dampratio={dr}")
    path.write_text(json.dumps({"mujoco": mujoco.__version__, "records": records}, indent=2) + "\n")
    print(f"wrote {path}")


def _capture_solref_sweep(mujoco, np, refs_dir: Path, path: Path) -> None:
    """Capture matched-Euler final states for the sphere-drop solref sweep."""
    mjcf_path = refs_dir / "sphere_drop.xml"
    triples = [
        (0.005, 0.5),
        (0.010, 1.0),
        (0.020, 1.0),
        (0.050, 1.0),
        (0.070, 1.0),
        (0.100, 1.0),
        (0.200, 2.0),
    ]
    records = []
    for tc, dr in triples:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        model.geom_solref[:, 0] = tc
        model.geom_solref[:, 1] = dr
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(1500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        records.append(
            {
                "tc": tc,
                "dampratio": dr,
                "steps": 1500,
                "integrator": "Euler",
                "qpos": data.qpos.astype("float64").tolist(),
                "qvel": data.qvel.astype("float64").tolist(),
            }
        )
    path.write_text(json.dumps({"mujoco": mujoco.__version__, "records": records}, indent=2) + "\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# version guard
# ---------------------------------------------------------------------------


def _extract_mujoco_version(prov: str) -> str | None:
    for tok in prov.split("|"):
        if tok.startswith("mujoco="):
            return tok[len("mujoco=") :]
    return None


def _check_version_guard(
    refs_dir: Path, scenarios: list[dict], live_version: str, force: bool
) -> None:
    """Compare the MuJoCo version in every existing fixture against the
    running interpreter's MuJoCo version. Refuses to overwrite fixtures
    captured under a different version unless `--force` is set."""
    mismatches = []
    for scenario in scenarios:
        path = refs_dir / f"{scenario['name']}.bin"
        if not path.is_file():
            continue
        prov = _read_provenance(path)
        if prov is None:
            continue
        old_version = _extract_mujoco_version(prov)
        if old_version is None:
            continue
        if old_version != live_version:
            mismatches.append((scenario["name"], old_version))
    if mismatches and not force:
        print("REFUSING TO CAPTURE — mujoco version drift:", file=sys.stderr)
        for name, old in mismatches:
            print(f"  {name}.bin was captured with mujoco {old}", file=sys.stderr)
        print(
            f"  live interpreter has mujoco {live_version}",
            file=sys.stderr,
        )
        print(
            "  pass --force to overwrite (and update the scorecard's known-cause "
            "notes if divergences move)",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Capture reference MuJoCo trajectories for newt's differential harness."
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="Scenario names to capture. Empty = capture ALL scenarios in scenarios.json.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List scenario names and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite fixtures even if they were captured with a different mujoco version.",
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV_PYTHON),
        help="Path to the venv Python to re-exec under (default: biped venv). "
        "Pass --venv '' to disable re-exec.",
    )
    parser.add_argument(
        "--integrator",
        choices=["Euler", "implicitfast"],
        help="Override the XML integrator for a matched-integrator capture.",
    )
    parser.add_argument(
        "--solver",
        choices=["PGS", "Newton"],
        help="Override the XML solver for a matched-solver capture.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix inserted before .bin, for example _euler.",
    )
    parser.add_argument(
        "--row-diagnostics",
        type=Path,
        metavar="PATH",
        help="Write first-contact MuJoCo efc_* factors for the sphere-drop sweep as JSON.",
    )
    parser.add_argument(
        "--solref-sweep",
        type=Path,
        metavar="PATH",
        help="Write the matched-Euler sphere-drop solref sweep final states as JSON.",
    )
    args = parser.parse_args(argv[1:])

    # Re-exec under the venv if we're not already in one that has mujoco.
    if args.venv:
        _reexec_under_venv(Path(args.venv), argv)

    # Import after any re-exec.
    try:
        import mujoco  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as e:
        print(
            "ERROR: mujoco/numpy not importable. Run under the biped venv:",
            file=sys.stderr,
        )
        print(f"  {DEFAULT_VENV_PYTHON} {' '.join(argv)}", file=sys.stderr)
        print(f"  underlying: {e}", file=sys.stderr)
        return 1

    refs_dir = Path(__file__).resolve().parent.parent / "tests" / "references"
    scenarios = _load_scenarios(refs_dir)

    if args.row_diagnostics is not None:
        _capture_row_diagnostics(mujoco, np, refs_dir, args.row_diagnostics)
        return 0
    if args.solref_sweep is not None:
        _capture_solref_sweep(mujoco, np, refs_dir, args.solref_sweep)
        return 0

    if args.list:
        for s in scenarios:
            print(s["name"])
        return 0

    if args.scenarios:
        wanted = set(args.scenarios)
        known = {s["name"] for s in scenarios}
        unknown = wanted - known
        if unknown:
            print(f"unknown scenarios: {sorted(unknown)}", file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s["name"] in wanted]

    _check_version_guard(refs_dir, scenarios, mujoco.__version__, args.force)

    for scenario in scenarios:
        path, prov = _capture_scenario(
            mujoco,
            np,
            scenario,
            refs_dir,
            integrator_override=args.integrator,
            solver_override=args.solver,
            suffix=args.suffix,
        )
        print(f"wrote {path.name} :: {prov}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
