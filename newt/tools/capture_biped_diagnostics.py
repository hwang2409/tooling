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

    def row_mapping():
        mapping = []
        for contact_index in range(data.ncon):
            address = int(data.contact[contact_index].efc_address)
            if address < 0:
                continue
            next_addresses = [
                int(data.contact[index].efc_address)
                for index in range(contact_index + 1, data.ncon)
                if int(data.contact[index].efc_address) >= 0
            ]
            end = next_addresses[0] if next_addresses else data.nefc
            mapping.extend(
                {"row": row, "contact_index": contact_index}
                for row in range(address, end)
            )
        return mapping

    def record(step: int):
        points = biped._foot_contact_points_3d(mujoco_module, model, data)
        contact_mask = (1 if biped._foot_in_contact(points, "left") else 0) | (
            2 if biped._foot_in_contact(points, "right") else 0
        )
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            address = int(contact.efc_address)
            next_addresses = [
                int(data.contact[next_index].efc_address)
                for next_index in range(index + 1, data.ncon)
                if int(data.contact[next_index].efc_address) >= 0
            ]
            row_end = next_addresses[0] if next_addresses else data.nefc
            contacts.append(
                {
                    "geom1": mujoco_module.mj_id2name(
                        model, mujoco_module.mjtObj.mjOBJ_GEOM, geom1
                    ),
                    "geom2": mujoco_module.mj_id2name(
                        model, mujoco_module.mjtObj.mjOBJ_GEOM, geom2
                    ),
                    "position": [float(value) for value in contact.pos],
                    "normal": [float(value) for value in contact.frame[:3]],
                    "condim": min(int(model.geom_condim[geom1]), int(model.geom_condim[geom2])),
                    "geom_condim": [int(model.geom_condim[geom1]), int(model.geom_condim[geom2])],
                    "frame": [float(value) for value in contact.frame],
                    "dist": float(contact.dist),
                    "efc_address": address,
                    "row_indices": list(range(address, row_end)) if address >= 0 else [],
                }
            )
        return {
            "step": step,
            "contact_mask": contact_mask,
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "contacts": contacts,
            "row_to_contact": row_mapping(),
            "qfrc_constraint": data.qfrc_constraint.tolist(),
            "efc_pos": data.efc_pos[: data.nefc].tolist(),
            "efc_vel": data.efc_vel[: data.nefc].tolist(),
            "efc_aref": data.efc_aref[: data.nefc].tolist(),
            "efc_R": data.efc_R[: data.nefc].tolist(),
            "efc_force": data.efc_force[: data.nefc].tolist(),
        }

    records.append(record(0))
    for step in range(1, steps + 1):
        biped._apply_balance_controller(mujoco_module, model, data, config, data.time)
        controller.before_step(mujoco_module, model, data, step, config)
        targets = controller.targets(config, data.time, data=data)
        biped._apply_controls(mujoco_module, model, data, targets)
        mujoco_module.mj_step(model, data)
        records.append(record(step))
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
