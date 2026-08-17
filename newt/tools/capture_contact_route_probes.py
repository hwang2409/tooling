#!/usr/bin/env python3
"""Capture MuJoCo default route probes used by docs/contacts.md."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


PROBES = (
    ("P1", "sphere-ellipsoid", "mjc_Convex", "contact_route_sphere_ellipsoid.xml"),
    ("P2", "sphere-mesh", "mjc_Convex", "contact_route_sphere_mesh.xml"),
    ("P3", "plane-ellipsoid", "mjc_PlaneConvex", "contact_route_plane_ellipsoid.xml"),
    ("P4", "plane-mesh", "mjc_PlaneConvex", "contact_route_plane_mesh.xml"),
)


def capture(mujoco, references: Path) -> dict:
    probes = []
    for probe_id, pair, route, source_xml in PROBES:
        model = mujoco.MjModel.from_xml_path(str(references / source_xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            contacts.append(
                {
                    "geom": [int(value) for value in contact.geom],
                    "position": [float(value) for value in contact.pos],
                    "frame_normal": [float(value) for value in contact.frame[:3]],
                    "penetration": float(-contact.dist),
                }
            )
        probes.append(
            {
                "id": probe_id,
                "pair": pair,
                "mujoco_route": route,
                "source_xml": source_xml,
                "contacts": contacts,
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
