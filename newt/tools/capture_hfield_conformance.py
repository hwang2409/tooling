#!/usr/bin/env python3
"""Recapture static hfield contacts from the executable provenance XMLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def capture_case(mujoco, source: Path, case: dict) -> tuple[list[dict], int, int]:
    model = mujoco.MjModel.from_xml_path(str(source))
    if model.nhfield < 1 or model.ngeom < 1:
        raise ValueError(f"{source.name}: expected a complete hfield model")

    name = case["name"]
    field_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"field_geom_{name}"
    )
    shape_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"shape_{name}"
    )
    if field_id < 0 or shape_id < 0:
        raise ValueError(f"{source.name}: missing geom binding for {name}")

    # The source model contains all poses for this mode. Capture one declared
    # pair at a time so unrelated fixture poses cannot contact each other.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.geom_contype[field_id] = 1
    model.geom_conaffinity[field_id] = 1
    model.geom_contype[shape_id] = 1
    model.geom_conaffinity[shape_id] = 1

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist >= 0:
            continue
        if set(int(value) for value in contact.geom) != {field_id, shape_id}:
            continue
        position = [float(value) for value in contact.pos]
        normal = [float(value) for value in contact.frame[:3]]
        contacts.append(
            {
                "position": position,
                "normal": normal,
                "penetration": float(-contact.dist),
            }
        )
    if not contacts:
        raise ValueError(f"{source.name}: no contact captured for {name}")
    return contacts, model.ngeom, model.nhfield


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "references",
    )
    args = parser.parse_args()

    import mujoco  # type: ignore[import-not-found]

    fixture_path = args.references / "hfield_conformance.json"
    document = json.loads(fixture_path.read_text(encoding="utf-8"))
    for case in document["cases"]:
        source = args.references / case["provenance"]["source_xml"]
        contacts, ngeom, nhfield = capture_case(mujoco, source, case)
        case["contacts"] = contacts
        print(
            f"{case['name']}: {source.name}, "
            f"ngeom={ngeom}, nhfield={nhfield}, "
            f"contacts={len(case['contacts'])}"
        )
    fixture_path.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {fixture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
