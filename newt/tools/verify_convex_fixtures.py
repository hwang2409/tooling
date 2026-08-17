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


# This is separate from the Newt parity tolerances stored in the bounds fixture.
# It covers only cross-platform MuJoCo capture drift in the reviewed sample data.
CI_ORACLE_TOLERANCE = {
    "position": 1.0e-6,
    "orientation_wxyz": 1.0e-6,
    "frame_normal": 1.0e-6,
    "penetration": 1.0e-6,
}


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


def compare_vector(
    committed: list[float],
    fresh: list[float],
    field: str,
    path: str,
    maxima: dict[str, float],
) -> None:
    assert len(committed) == len(fresh), path
    tolerance = CI_ORACLE_TOLERANCE[field]
    for index, (committed_value, fresh_value) in enumerate(zip(committed, fresh)):
        difference = abs(float(committed_value) - float(fresh_value))
        maxima[field] = max(maxima[field], difference)
        assert difference <= tolerance, (
            f"{path}[{index}] differs from fresh MuJoCo capture",
            committed_value,
            fresh_value,
            difference,
            tolerance,
        )


def inverse_rotate(quaternion: list[float], vector: list[float]) -> list[float]:
    """Apply the transpose of a WXYZ unit-quaternion rotation."""

    w, x, y, z = quaternion
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    vx, vy, vz = vector
    return [
        (1.0 - 2.0 * (yy + zz)) * vx
        + 2.0 * (xy - wz) * vy
        + 2.0 * (xz + wy) * vz,
        2.0 * (xy + wz) * vx
        + (1.0 - 2.0 * (xx + zz)) * vy
        + 2.0 * (yz - wx) * vz,
        2.0 * (xz - wy) * vx
        + 2.0 * (yz + wx) * vy
        + (1.0 - 2.0 * (xx + yy)) * vz,
    ]


def quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def quaternion_conjugate(quaternion: list[float]) -> list[float]:
    return [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]


def dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def scale(vector: list[float], factor: float) -> list[float]:
    return [component * factor for component in vector]


def cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(dot(vector, vector))
    assert length > 0.0
    return scale(vector, 1.0 / length)


def mesh_basis(mesh: dict) -> tuple[list[float], list[float], list[float]]:
    vertices = mesh["vertices"]
    origin = vertices[0]
    first = normalize(subtract(vertices[1], origin))
    second_unscaled = subtract(vertices[2], origin)
    second = normalize(subtract(second_unscaled, scale(first, dot(second_unscaled, first))))
    third = normalize(cross(first, second))
    return first, second, third


def basis_coordinates(basis: tuple[list[float], list[float], list[float]], vector: list[float]) -> list[float]:
    return [dot(axis, vector) for axis in basis]


def body_frame_contact(body_pose: dict, geom_pose: dict, mesh: dict, contact: dict) -> dict:
    body_position = body_pose["position"]
    body_orientation = body_pose["orientation_wxyz"]
    relative_position = inverse_rotate(
        body_orientation,
        subtract(contact["position"], body_position),
    )
    geom_position = inverse_rotate(
        body_orientation,
        subtract(geom_pose["position"], body_position),
    )
    relative_position = subtract(relative_position, geom_position)
    relative_orientation = quaternion_multiply(
        quaternion_conjugate(body_orientation), geom_pose["orientation_wxyz"]
    )
    mesh_position = inverse_rotate(relative_orientation, relative_position)
    mesh_normal = inverse_rotate(
        relative_orientation,
        inverse_rotate(body_orientation, contact["frame_normal"]),
    )
    basis = mesh_basis(mesh)
    return {
        "geom": tuple(contact["geom"]),
        "position": basis_coordinates(basis, mesh_position),
        "frame_normal": basis_coordinates(basis, mesh_normal),
        "penetration": float(contact["penetration"]),
    }


def sorted_body_frame_contacts(
    body_pose: dict,
    geom_pose: dict,
    mesh: dict,
    contacts: list[dict],
) -> list[dict]:
    transformed = [
        body_frame_contact(body_pose, geom_pose, mesh, contact)
        for contact in contacts
    ]
    return sorted(
        transformed,
        key=lambda contact: (
            contact["geom"],
            -contact["penetration"],
            tuple(contact["position"]),
            tuple(contact["frame_normal"]),
        ),
    )


def compare_dynamic_oracle(
    committed: dict,
    fresh: dict,
    maxima: dict[str, float],
) -> None:
    assert committed["mujoco"] == fresh["mujoco"]
    assert without_date(committed)["capture_provenance"] == without_date(fresh)[
        "capture_provenance"
    ]
    assert committed["windows"] == fresh["windows"]
    assert len(committed["cases"]) == len(fresh["cases"])
    for committed_case, fresh_case in zip(committed["cases"], fresh["cases"]):
        for field in ("id", "source_xml", "route"):
            assert committed_case[field] == fresh_case[field], field
        assert len(committed_case["samples"]) == len(fresh_case["samples"])
        for committed_sample, fresh_sample in zip(
            committed_case["samples"], fresh_case["samples"]
        ):
            path = f"dynamic.{committed_case['id']}.step{committed_sample['step']}"
            assert committed_sample["step"] == fresh_sample["step"], path
            compare_vector(
                committed_sample["position"],
                fresh_sample["position"],
                "position",
                f"{path}.position",
                maxima,
            )
            compare_vector(
                committed_sample["orientation_wxyz"],
                fresh_sample["orientation_wxyz"],
                "orientation_wxyz",
                f"{path}.orientation_wxyz",
                maxima,
            )
            assert committed_sample["contacts"] == fresh_sample["contacts"], path


def compare_route_oracle(
    committed: dict,
    fresh: dict,
    maxima: dict[str, float],
) -> None:
    assert committed["mujoco"] == fresh["mujoco"]
    assert without_date(committed)["capture_provenance"] == without_date(fresh)[
        "capture_provenance"
    ]
    assert len(committed["probes"]) == len(fresh["probes"])
    for committed_probe, fresh_probe in zip(committed["probes"], fresh["probes"]):
        for field in ("id", "pair", "mujoco_route", "source_xml"):
            assert committed_probe[field] == fresh_probe[field], field
        assert len(committed_probe["poses"]) == len(fresh_probe["poses"])
        for committed_pose, fresh_pose in zip(
            committed_probe["poses"], fresh_probe["poses"]
        ):
            path = f"route.{committed_probe['id']}.{committed_pose['id']}"
            for field in ("id", "geom_a", "geom_b"):
                assert committed_pose[field] == fresh_pose[field], f"{path}.{field}"
            for geom in ("pose_a", "pose_b"):
                compare_vector(
                    committed_pose[geom]["position"],
                    fresh_pose[geom]["position"],
                    "position",
                    f"{path}.{geom}.position",
                    maxima,
                )
                # MuJoCo's mesh compiler may choose different principal-axis
                # frames across platforms. The fixture keeps that frame for
                # the Rust route replay, so CI compares contact samples in the
                # owning body frame, but not this mesh-local orientation.
                mesh_frame = "mesh" in committed_probe["pair"]
                if not mesh_frame:
                    compare_vector(
                        committed_pose[geom]["orientation_wxyz"],
                        fresh_pose[geom]["orientation_wxyz"],
                        "orientation_wxyz",
                        f"{path}.{geom}.orientation_wxyz",
                        maxima,
                    )
            for body in ("body_pose_a", "body_pose_b"):
                compare_vector(
                    committed_pose[body]["position"],
                    fresh_pose[body]["position"],
                    "position",
                    f"{path}.{body}.position",
                    maxima,
                )
                compare_vector(
                    committed_pose[body]["orientation_wxyz"],
                    fresh_pose[body]["orientation_wxyz"],
                    "orientation_wxyz",
                    f"{path}.{body}.orientation_wxyz",
                    maxima,
                )
            assert len(committed_pose["contacts"]) == len(fresh_pose["contacts"])
            mesh_pair = "mesh" in committed_probe["pair"]
            if mesh_pair:
                committed_contacts = sorted_body_frame_contacts(
                    committed_pose["body_pose_b"],
                    committed_pose["pose_b"],
                    committed_probe["mesh"],
                    committed_pose["contacts"],
                )
                fresh_contacts = sorted_body_frame_contacts(
                    fresh_pose["body_pose_b"],
                    fresh_pose["pose_b"],
                    fresh_probe["mesh"],
                    fresh_pose["contacts"],
                )
            else:
                committed_contacts = committed_pose["contacts"]
                fresh_contacts = fresh_pose["contacts"]
            for index, (committed_contact, fresh_contact) in enumerate(
                zip(committed_contacts, fresh_contacts)
            ):
                contact_path = f"{path}.contacts[{index}]"
                assert committed_contact["geom"] == fresh_contact["geom"], contact_path
                compare_vector(
                    committed_contact["position"],
                    fresh_contact["position"],
                    "position",
                    f"{contact_path}.position",
                    maxima,
                )
                compare_vector(
                    committed_contact["frame_normal"],
                    fresh_contact["frame_normal"],
                    "frame_normal",
                    f"{contact_path}.frame_normal",
                    maxima,
                )
                difference = abs(
                    float(committed_contact["penetration"])
                    - float(fresh_contact["penetration"])
                )
                maxima["penetration"] = max(maxima["penetration"], difference)
                assert difference <= CI_ORACLE_TOLERANCE["penetration"], (
                    f"{contact_path}.penetration differs from fresh MuJoCo capture",
                    committed_contact["penetration"],
                    fresh_contact["penetration"],
                    difference,
                    CI_ORACLE_TOLERANCE["penetration"],
                )


def run_ci_regression_self_test() -> None:
    fresh = {
        "mujoco": "3.11.0",
        "capture_provenance": {"script": "test", "method": "test"},
        "windows": {"early": 2, "full": 2},
        "cases": [
            {
                "id": "non-max-sample",
                "source_xml": "test.xml",
                "route": "mjc_Convex",
                "samples": [
                    {
                        "step": 0,
                        "position": [0.0, 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "contacts": 1,
                    },
                    {
                        "step": 1,
                        "position": [0.1, 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "contacts": 1,
                    },
                    {
                        "step": 2,
                        "position": [1.0, 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "contacts": 1,
                    },
                ],
            }
        ],
    }
    committed = json.loads(json.dumps(fresh))
    committed["cases"][0]["samples"][1]["position"][0] = 0.2
    try:
        compare_dynamic_oracle(committed, fresh, {field: 0.0 for field in CI_ORACLE_TOLERANCE})
    except AssertionError:
        pass
    else:
        raise AssertionError("CI oracle comparison accepted a non-max sample mutation")

    fresh_route = {
        "mujoco": "3.11.0",
        "capture_provenance": {"script": "test", "method": "test"},
        "probes": [
            {
                "id": "P6",
                "pair": "mesh-mesh",
                "mujoco_route": "mjc_Convex",
                "source_xml": "test.xml",
                "mesh": {
                    "vertices": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "faces": [],
                },
                "poses": [
                    {
                        "id": "near_touch",
                        "geom_a": "mesh_a",
                        "geom_b": "mesh_b",
                        "pose_a": {
                            "position": [0.0, 0.0, 0.0],
                            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        "pose_b": {
                            "position": [1.0, 0.0, 0.0],
                            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        "body_pose_a": {
                            "position": [0.0, 0.0, 0.0],
                            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        "body_pose_b": {
                            "position": [0.0, 0.0, 0.0],
                            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        "contacts": [
                            {
                                "geom": [8, 9],
                                "position": [
                                    0.9991666682,
                                    0.0001666682,
                                    0.0001666682,
                                ],
                                "frame_normal": [
                                    0.5773502692,
                                    0.5773502692,
                                    0.5773502692,
                                ],
                                "penetration": 0.0005773399,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    committed_route = json.loads(json.dumps(fresh_route))
    # Match the review mutation: rotate one mesh contact position and normal
    # around x while preserving both vectors' norms.
    contact = committed_route["probes"][0]["poses"][0]["contacts"][0]
    contact["position"][1] *= -1.0
    contact["frame_normal"][1] *= -1.0
    try:
        compare_route_oracle(
            committed_route,
            fresh_route,
            {field: 0.0 for field in CI_ORACLE_TOLERANCE},
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("CI oracle comparison accepted a mesh direction mutation")


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
    if args.ci:
        run_ci_regression_self_test()
        print("CI oracle mutation self-test passed")
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
            maxima = {field: 0.0 for field in CI_ORACLE_TOLERANCE}
            compare_route_oracle(expected_routes, generated_routes, maxima)
            compare_dynamic_oracle(expected_dynamic, generated_dynamic, maxima)
            print(f"CI MuJoCo capture drift maxima: {maxima}")
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
