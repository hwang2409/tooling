#!/usr/bin/env python3
"""Re-run the pinned MuJoCo captures and verify checked-in fixture rows."""

from __future__ import annotations

import argparse
import datetime
import json
import tempfile
from pathlib import Path


def without_date(document: dict) -> dict:
    result = json.loads(json.dumps(document))
    result["capture_provenance"].pop("date", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "references",
    )
    args = parser.parse_args()
    import mujoco  # type: ignore[import-not-found]

    from capture_contact_route_probes import capture as capture_routes
    from capture_convex_dynamic_anchors import CASES, WINDOWS, capture_case

    expected_routes = json.loads(
        (args.references / "contact_route_probes.json").read_text(encoding="utf-8")
    )
    expected_dynamic = json.loads(
        (args.references / "contact_dynamic_anchors.json").read_text(encoding="utf-8")
    )
    assert len(expected_dynamic["cases"]) >= 2, "dynamic evidence needs two independent anchors"
    with tempfile.TemporaryDirectory(prefix="newt-convex-fixtures-") as temp:
        temp_references = Path(temp)
        # MuJoCo loads the checked-in XML paths. The temporary directory only
        # holds canonical comparison data, so this also tests XML provenance.
        generated_routes = capture_routes(mujoco, args.references)
        generated_dynamic = {
            "mujoco": mujoco.__version__,
            "capture_provenance": {
                "script": "tools/capture_convex_dynamic_anchors.py",
                "date": datetime.date.today().isoformat(),
                "method": "mj_step from each source_xml; snapshots at steps 0, 20, and 100",
            },
            "windows": {name: step for name, step in WINDOWS},
            "cases": [capture_case(mujoco, args.references, case) for case in CASES],
        }
        (temp_references / "contact_route_probes.json").write_text(
            json.dumps(generated_routes, indent=2) + "\n", encoding="utf-8"
        )
        (temp_references / "contact_dynamic_anchors.json").write_text(
            json.dumps(generated_dynamic, indent=2) + "\n", encoding="utf-8"
        )
        assert without_date(generated_routes) == without_date(expected_routes)
        assert without_date(generated_dynamic) == without_date(expected_dynamic)
    bounds = json.loads(
        (args.references / "contact_dynamic_anchor_bounds.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(bounds["cases"]) >= 2, "dynamic bounds need two independent anchors"
    for case in bounds["cases"]:
        for window in ("early", "full"):
            observed = case[window]["observed_max"]
            bound = case[window]
            for field in ("position", "orientation", "contact_count"):
                assert observed[field] <= bound[field], (case["id"], window, field)
    print("verified MuJoCo route samples, dynamic samples, and reviewed bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
