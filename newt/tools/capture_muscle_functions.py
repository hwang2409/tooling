#!/usr/bin/env python3
"""Capture MuJoCo 3.11 muscle helper values for pointwise conformance tests."""

import json
import pathlib
import sys

import mujoco


PARAMS = [0.75, 1.05, -1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2]
LENGTH_RANGE = [0.0, 1.0]
ACC0 = 2.0
LENGTHS = [0.0, 0.25, 0.5, 0.8333333, 1.0]
VELOCITIES = [-5.0, -2.5, 0.0, 0.25, 2.5]
ACTIVATIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
CONTROLS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    output = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
        "newt/tests/references/muscle_function_grid.json"
    )
    gain = []
    bias = []
    for length in LENGTHS:
        for velocity in VELOCITIES:
            gain.append(
                {
                    "length": length,
                    "velocity": velocity,
                    "value": mujoco.mju_muscleGain(
                        length, velocity, LENGTH_RANGE, ACC0, PARAMS
                    ),
                }
            )
        bias.append(
            {
                "length": length,
                "value": mujoco.mju_muscleBias(length, LENGTH_RANGE, ACC0, PARAMS),
            }
        )
    dynamics = []
    dynprm = [0.1, 0.2, 0.4]
    for activation in ACTIVATIONS:
        for control in CONTROLS:
            dynamics.append(
                {
                    "activation": activation,
                    "control": control,
                    "value": mujoco.mju_muscleDynamics(control, activation, dynprm),
                }
            )
    document = {
        "oracle": "MuJoCo 3.11.0",
        "params": PARAMS,
        "lengthrange": LENGTH_RANGE,
        "acc0": ACC0,
        "dynprm": dynprm,
        "gain": gain,
        "bias": bias,
        "dynamics": dynamics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {output} using MuJoCo {mujoco.__version__}")


if __name__ == "__main__":
    main()
