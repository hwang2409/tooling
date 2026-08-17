#!/usr/bin/env python3
"""Capture MuJoCo early/full-window anchors for enabled convex routes."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np


CASES = (
    (
        "mesh-mesh-tumble",
        "contact_dynamic_mesh_mesh.xml",
        "mesh_body",
        "mesh_tumble",
        (1.0, 0.7, -0.4),
    ),
    (
        "mesh-mesh-rotated-drop",
        "contact_dynamic_mesh_mesh_rotated.xml",
        "mesh_rotated_body",
        "mesh_rotated",
        (-0.6, 0.9, 0.5),
    ),
)
WINDOWS = (("early", 20), ("full", 100))


def capture_case(mujoco, references: Path, case: tuple) -> dict:
    case_id, source_xml, body_name, geom_name, angular_velocity = case
    model = mujoco.MjModel.from_xml_path(str(references / source_xml))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if body_id < 0 or geom_id < 0:
        raise ValueError(f"{source_xml}: missing body or geom")
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.geom_contype[:] = 1
    model.geom_conaffinity[:] = 1
    data = mujoco.MjData(model)
    if angular_velocity:
        data.qvel[3:6] = np.array(angular_velocity)
    mujoco.mj_forward(model, data)
    samples = []
    for step in range(max(window for _, window in WINDOWS) + 1):
        if step in {0, 20, 100}:
            samples.append(
                {
                    "step": step,
                    "position": data.xpos[body_id].astype("float64").tolist(),
                    "orientation_wxyz": data.xquat[body_id].astype("float64").tolist(),
                    "contacts": sum(
                        1
                        for index in range(data.ncon)
                        if geom_id in {int(data.contact[index].geom[0]), int(data.contact[index].geom[1])}
                    ),
                }
            )
        if step < max(window for _, window in WINDOWS):
            mujoco.mj_step(model, data)
    return {"id": case_id, "source_xml": source_xml, "route": "mjc_Convex", "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, default=Path(__file__).resolve().parent.parent / "tests" / "references")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    import mujoco  # type: ignore[import-not-found]

    output = args.output or args.references / "contact_dynamic_anchors.json"
    document = {
        "mujoco": mujoco.__version__,
        "capture_provenance": {
            "script": "tools/capture_convex_dynamic_anchors.py",
            "date": datetime.date.today().isoformat(),
            "method": "mj_step from each source_xml; snapshots at steps 0, 20, and 100",
        },
        "windows": {name: step for name, step in WINDOWS},
        "cases": [capture_case(mujoco, args.references, case) for case in CASES],
    }
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
