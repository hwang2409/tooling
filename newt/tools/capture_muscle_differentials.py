#!/usr/bin/env python3
"""Capture matched MuJoCo muscle trajectories for committed newt fixtures."""

import json
import pathlib
import sys

import mujoco


CASES = {
    "muscle_pendulum": {"ctrl": lambda step: 1.0 if step < 250 else 0.35, "steps": 500},
    "muscle_wrapped_tendon": {"ctrl": lambda step: 1.0, "steps": 400},
    "muscle_isometric_twitch": {
        "ctrl": lambda step: 1.0 if step < 50 else 0.0,
        "steps": 300,
    },
}


def capture(xml_path: pathlib.Path, case: str) -> dict:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    rows = []
    for step in range(CASES[case]["steps"]):
        data.ctrl[:] = CASES[case]["ctrl"](step)
        mujoco.mj_step(model, data)
        rows.append(
            {
                "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist(),
                "act": data.act.tolist(),
            }
        )
    return {
        "oracle": "MuJoCo 3.11.0",
        "scene": xml_path.name,
        "timestep": float(model.opt.timestep),
        "steps": len(rows),
        "rows": rows,
    }


def main() -> None:
    output_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
        "newt/tests/references"
    )
    for case in CASES:
        xml_path = output_dir / f"{case}.xml"
        fixture_path = output_dir / f"{case}.json"
        fixture_path.write_text(json.dumps(capture(xml_path, case), indent=2) + "\n")
        print(f"wrote {fixture_path}")


if __name__ == "__main__":
    main()
