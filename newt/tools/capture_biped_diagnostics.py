#!/usr/bin/env python3
"""Capture early MuJoCo contact and solver diagnostics for biped parity."""

from __future__ import annotations

import argparse
import datetime as date
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "me/fun/biped"))
from biped import mujoco_biped as biped  # noqa: E402
import mujoco  # noqa: E402


def capture(assist_scale: float, steps: int, output: Path) -> None:
    scenario = "joint_walk" if assist_scale == 0.0 else "stable_joint_walk"
    config = biped.BipedSimConfig.for_scenario(
        scenario,
        steps=5000,
        frame_stride=1,
        assist_scale=assist_scale,
    )
    controller = biped.CONTROLLERS[config.controller_name]
    mujoco_module, model, data = biped._load_model(config)
    model.opt.integrator = mujoco_module.mjtIntegrator.mjINT_EULER
    model.opt.solver = mujoco_module.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco_module.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 20
    biped._set_initial_pose(mujoco_module, model, data, config, controller)
    records = []
    for step in range(1, steps + 1):
        biped._apply_balance_controller(mujoco_module, model, data, config, data.time)
        controller.before_step(mujoco_module, model, data, step, config)
        targets = controller.targets(config, data.time, data=data)
        biped._apply_controls(mujoco_module, model, data, targets)
        mujoco_module.mj_step(model, data)
        points = biped._foot_contact_points_3d(mujoco_module, model, data)
        contact_mask = (1 if biped._foot_in_contact(points, "left") else 0) | (
            2 if biped._foot_in_contact(points, "right") else 0
        )
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            contacts.append(
                {
                    "geom1": mujoco_module.mj_id2name(
                        model, mujoco_module.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    ),
                    "geom2": mujoco_module.mj_id2name(
                        model, mujoco_module.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    ),
                    "dist": float(contact.dist),
                }
            )
        records.append(
            {
                "step": step,
                "contact_mask": contact_mask,
                "contacts": contacts,
                "qfrc_constraint": data.qfrc_constraint.tolist(),
                "efc_pos": data.efc_pos[: data.nefc].tolist(),
                "efc_vel": data.efc_vel[: data.nefc].tolist(),
                "efc_aref": data.efc_aref[: data.nefc].tolist(),
                "efc_R": data.efc_R[: data.nefc].tolist(),
                "efc_force": data.efc_force[: data.nefc].tolist(),
            }
        )
    payload = {
        "mujoco": mujoco.__version__,
        "captured": date.date.today().isoformat(),
        "assist_scale": assist_scale,
        "controller": config.controller_name,
        "dt": config.dt,
        "integrator": "Euler",
        "solver": "Newton",
        "cone": "pyramidal",
        "iterations": 20,
        "steps": steps,
        "model": biped.MODEL_ID,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} :: assist={assist_scale:.1f} records={len(records)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assist-scale", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.assist_scale, args.steps, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
