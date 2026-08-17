#!/usr/bin/env python3
"""Capture MuJoCo default route probes used by docs/contacts.md."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np


PROBES = (
    {
        "id": "P1",
        "pair": "sphere-ellipsoid",
        "route": "mjc_Convex",
        "source_xml": "contact_route_sphere_ellipsoid.xml",
        "bounds": {"count_delta": 0, "position": 0.11, "normal": 0.0001, "penetration": 0.0001},
        "cases": (
            ("no_contact", "sphere_no_contact", "ellipsoid_no_contact"),
            ("shallow", "sphere_shallow", "ellipsoid_shallow"),
            ("deep", "sphere_deep", "ellipsoid_deep"),
            ("off_axis_rotated", "sphere_off_axis", "ellipsoid_off_axis"),
        ),
    },
    {
        "id": "P2",
        "pair": "sphere-mesh",
        "route": "mjc_Convex",
        "source_xml": "contact_route_sphere_mesh.xml",
        "bounds": {"count_delta": 0, "position": 0.11, "normal": 0.0001, "penetration": 0.0001},
        "cases": (
            ("no_contact", "sphere_no_contact", "mesh_no_contact"),
            ("shallow", "sphere_shallow", "mesh_shallow"),
            ("deep", "sphere_deep", "mesh_deep"),
            ("off_axis_rotated", "sphere_off_axis", "mesh_off_axis"),
        ),
    },
    {
        "id": "P3",
        "pair": "plane-ellipsoid",
        "route": "mjc_PlaneConvex",
        "source_xml": "contact_route_plane_ellipsoid.xml",
        "bounds": {"count_delta": 0, "position": 0.12, "normal": 0.0001, "penetration": 0.0001},
        "cases": (
            ("no_contact", "plane", "ellipsoid_no_contact"),
            ("shallow", "plane", "ellipsoid_shallow"),
            ("deep", "plane", "ellipsoid_deep"),
            ("off_axis_rotated", "plane", "ellipsoid_off_axis"),
        ),
    },
    {
        "id": "P4",
        "pair": "plane-mesh",
        "route": "mjc_PlaneConvex",
        "source_xml": "contact_route_plane_mesh.xml",
        "bounds": {"count_delta": 0, "position": 0.18, "normal": 0.0001, "penetration": 0.0001},
        "cases": (
            ("no_contact", "plane", "mesh_no_contact"),
            ("shallow", "plane", "mesh_shallow"),
            ("deep", "plane", "mesh_deep"),
            ("off_axis_rotated", "plane", "mesh_off_axis"),
        ),
    },
    {
        "id": "P5",
        "pair": "box-mesh",
        "route": "mjc_Convex",
        "source_xml": "contact_route_box_mesh.xml",
        "bounds": {"count_delta": 0, "position": 1.0e-6, "normal": 1.01, "penetration": 1.0e-6},
        "cases": (
            ("no_contact", "box_no_contact", "mesh_no_contact"),
            ("shallow", "box_shallow", "mesh_shallow"),
            ("deep", "box_deep", "mesh_deep"),
            ("off_axis_rotated", "box_off_axis", "mesh_off_axis"),
        ),
    },
    {
        "id": "P6",
        "pair": "mesh-mesh",
        "route": "mjc_Convex",
        "source_xml": "contact_route_mesh_mesh.xml",
        "bounds": {"count_delta": 0, "position": 1.0e-6, "normal": 1.0e-5, "penetration": 1.0e-6},
        "cases": (
            ("no_contact", "mesh_a_no_contact", "mesh_b_no_contact"),
            ("shallow", "mesh_a_shallow", "mesh_b_shallow"),
            ("deep", "mesh_a_deep", "mesh_b_deep"),
            ("off_axis_rotated", "mesh_a_off_axis", "mesh_b_off_axis"),
        ),
    },
)


def capture(mujoco, references: Path) -> dict:
    probes = []
    for probe in PROBES:
        source_xml = probe["source_xml"]
        model = mujoco.MjModel.from_xml_path(str(references / source_xml))
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        poses = []
        for pose_id, geom_a_name, geom_b_name in probe["cases"]:
            geom_a = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_a_name
            )
            geom_b = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_b_name
            )
            if geom_a < 0 or geom_b < 0:
                raise ValueError(f"{source_xml}: missing {geom_a_name} or {geom_b_name}")
            model.geom_contype[geom_a] = 1
            model.geom_conaffinity[geom_a] = 1
            model.geom_contype[geom_b] = 1
            model.geom_conaffinity[geom_b] = 1
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            quat_a = np.zeros(4, dtype=np.float64)
            quat_b = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat_a, data.geom_xmat[geom_a])
            mujoco.mju_mat2Quat(quat_b, data.geom_xmat[geom_b])
            contacts = []
            for index in range(data.ncon):
                contact = data.contact[index]
                if set(int(value) for value in contact.geom) != {geom_a, geom_b}:
                    continue
                contacts.append(
                    {
                        "geom": [int(value) for value in contact.geom],
                        "position": [float(value) for value in contact.pos],
                        "frame_normal": [float(value) for value in contact.frame[:3]],
                        "penetration": float(-contact.dist),
                    }
                )
            poses.append(
                {
                    "id": pose_id,
                    "geom_a": geom_a_name,
                    "geom_b": geom_b_name,
                    "pose_a": {
                        "position": data.geom_xpos[geom_a].astype("float64").tolist(),
                        "orientation_wxyz": quat_a.tolist(),
                    },
                    "pose_b": {
                        "position": data.geom_xpos[geom_b].astype("float64").tolist(),
                        "orientation_wxyz": quat_b.tolist(),
                    },
                    "contacts": contacts,
                }
            )
            model.geom_contype[geom_a] = 0
            model.geom_conaffinity[geom_a] = 0
            model.geom_contype[geom_b] = 0
            model.geom_conaffinity[geom_b] = 0
        probes.append(
            {
                "id": probe["id"],
                "pair": probe["pair"],
                "mujoco_route": probe["route"],
                "source_xml": source_xml,
                "bounds": probe["bounds"],
                "poses": poses,
                "mesh": (
                    {
                        "vertices": model.mesh_vert[: model.mesh_vertnum[0]]
                        .astype("float64")
                        .tolist(),
                        "faces": model.mesh_face[: model.mesh_facenum[0]]
                        .astype("int64")
                        .tolist(),
                    }
                    if "mesh" in probe["pair"]
                    else None
                ),
            }
        )
    return {
        "mujoco": mujoco.__version__,
        "capture_provenance": {
            "script": "tools/capture_contact_route_probes.py",
            "date": datetime.date.today().isoformat(),
            "method": "mj_forward default contacts from each source_xml",
        },
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "references",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    import mujoco  # type: ignore[import-not-found]

    output = args.output or args.references / "contact_route_probes.json"
    output.write_text(
        json.dumps(capture(mujoco, args.references), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
