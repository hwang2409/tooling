#!/usr/bin/env python3
"""Re-run the pinned MuJoCo captures and verify checked-in fixture rows."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path


def without_date(document: dict) -> dict:
    result = json.loads(json.dumps(document))
    result["capture_provenance"].pop("date", None)
    return result


def orientation_error(newt: list[float], mujoco: list[float]) -> float:
    squares = [
        float32(float32(actual) - float32(expected)) ** 2
        for actual, expected in zip(newt, mujoco)
    ]
    total = float32(float32(squares[0]) + float32(squares[1]))
    total = float32(total + float32(squares[2]))
    total = float32(total + float32(squares[3]))
    return float32(math.sqrt(total))


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def dynamic_bounds(
    dynamic: dict,
    replay: dict,
    tolerance: dict,
    window_limits: tuple[tuple[str, int], ...],
) -> dict:
    replay_cases = {case["id"]: case for case in replay["cases"]}
    bounds_cases = []
    for expected_case in dynamic["cases"]:
        replay_case = replay_cases.get(expected_case["id"])
        assert replay_case is not None, expected_case["id"]
        expected_samples = {sample["step"]: sample for sample in expected_case["samples"]}
        replay_samples = {sample["step"]: sample for sample in replay_case["samples"]}
        assert expected_samples.keys() == replay_samples.keys(), expected_case["id"]
        errors = {}
        for step, expected in expected_samples.items():
            actual = replay_samples[step]
            position_squares = [
                float32(
                    float32(
                        float32(actual_value) - float32(expected_value)
                    )
                    ** 2
                )
                for actual_value, expected_value in zip(
                    actual["position"], expected["position"]
                )
            ]
            position_total = float32(
                float32(position_squares[0]) + float32(position_squares[1])
            )
            position_total = float32(position_total + float32(position_squares[2]))
            position = float32(math.sqrt(position_total))
            orientation = orientation_error(
                actual["orientation_wxyz"], expected["orientation_wxyz"]
            )
            contacts = abs(actual["contacts"] - expected["contacts"])
            errors[step] = {
                "position": position,
                "orientation": orientation,
                "contact_count": contacts,
            }
        generated_windows = {}
        for name, end_step in window_limits:
            window_errors = [error for step, error in errors.items() if step <= end_step]
            observed = {
                "position": float32(max(error["position"] for error in window_errors)),
                "orientation": float32(
                    max(error["orientation"] for error in window_errors)
                ),
                "contact_count": max(error["contact_count"] for error in window_errors),
            }
            generated_windows[name] = {
                "observed_max": observed,
                "position": float32(
                    observed["position"] + float32(tolerance["position"])
                ),
                "orientation": float32(
                    observed["orientation"] + float32(tolerance["orientation"])
                ),
                "contact_count": observed["contact_count"],
            }
        bounds_cases.append({"id": expected_case["id"], **generated_windows})
    return {
        "method": "Newt replay maxima plus reviewed tolerance over every captured step",
        "review_tolerance": tolerance,
        "cases": bounds_cases,
    }


def run_newt_replay(references: Path, output: Path, update_bounds: bool) -> dict:
    environment = os.environ.copy()
    environment["NEWT_DYNAMIC_REPLAY_OUTPUT"] = str(output)
    if update_bounds:
        environment["NEWT_DYNAMIC_REPLAY_ONLY"] = "1"
    result = subprocess.run(
        [
            "cargo",
            "test",
            "--test",
            "contacts_geoms_v1",
            "dynamic_enabled_convex_anchors_are_fixture_backed",
            "--quiet",
            "--",
            "--nocapture",
        ],
        cwd=references.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(output.read_text(encoding="utf-8"))


def assert_bounds_cover(measured: dict, reviewed: dict) -> None:
    reviewed_cases = {case["id"]: case for case in reviewed["cases"]}
    for measured_case in measured["cases"]:
        reviewed_case = reviewed_cases.get(measured_case["id"])
        assert reviewed_case is not None, measured_case["id"]
        for window in ("early", "full"):
            observed = measured_case[window]["observed_max"]
            bound = reviewed_case[window]
            for field in ("position", "orientation", "contact_count"):
                assert observed[field] <= bound[field], (
                    measured_case["id"],
                    window,
                    field,
                    observed[field],
                    bound[field],
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--references",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "references",
    )
    parser.add_argument(
        "--update-bounds",
        action="store_true",
        help="write bounds generated from the full Newt replay",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="use fresh MuJoCo states and tolerance bounds without byte identity",
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
                "method": "mj_step from each source_xml; snapshots at every step 0 through 100",
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
        if not args.ci:
            assert without_date(generated_routes) == without_date(expected_routes)
            assert without_date(generated_dynamic) == without_date(expected_dynamic)
        else:
            assert [case["id"] for case in generated_dynamic["cases"]] == [
                case["id"] for case in expected_dynamic["cases"]
            ]
    bounds_path = args.references / "contact_dynamic_anchor_bounds.json"
    bounds = json.loads(
        bounds_path.read_text(encoding="utf-8")
    )
    assert len(bounds["cases"]) >= 2, "dynamic bounds need two independent anchors"
    with tempfile.TemporaryDirectory(prefix="newt-convex-replay-") as temp:
        replay = run_newt_replay(
            args.references,
            Path(temp) / "contact_dynamic_replay.json",
            args.update_bounds,
        )
    measured_dynamic = dynamic_bounds(
        generated_dynamic if args.ci else expected_dynamic,
        replay,
        bounds["review_tolerance"],
        WINDOWS,
    )
    if args.update_bounds:
        bounds_path.write_text(json.dumps(measured_dynamic, indent=2) + "\n", encoding="utf-8")
    elif args.ci:
        assert_bounds_cover(measured_dynamic, bounds)
    else:
        assert bounds == measured_dynamic, "dynamic bounds are not generated from full replay"
    print("verified MuJoCo route samples, dynamic samples, and reviewed bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
