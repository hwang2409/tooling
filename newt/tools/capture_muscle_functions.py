#!/usr/bin/env python3
"""Capture MuJoCo 3.11 muscle helper values for pointwise conformance tests."""

import json
import pathlib
import sys

import mujoco


PARAMS = [0.75, 1.05, -1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2]
LENGTH_RANGE = [0.0, 1.0]
ACC0 = 2.0
# 319 points. Lengths are placed on both sides of every FL knot. The
# normalized values become physical lengths through the compiled range.
NORMALIZED_LENGTHS = [
    0.49, 0.5, 0.500001, 0.749999, 0.75, 0.750001,
    0.999999, 1.0, 1.000001, 1.299999, 1.3, 1.300001,
    1.599999, 1.6, 1.600001, 1.61,
]
LENGTHS = [LENGTH_RANGE[0] + value * (LENGTH_RANGE[1] - LENGTH_RANGE[0])
           for value in NORMALIZED_LENGTHS]
VELOCITIES = [-1.000001, -1.0, -0.999999, -0.000001, 0.0,
              0.000001, 0.199999, 0.2, 0.200001, 1.0]
ACTIVATIONS = [-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 0.75,
               0.9, 1.0, 1.1, 1.5, 2.0, 3.0]
CONTROLS = [-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 0.75,
            0.9, 1.0, 1.1, 2.0]


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
    assert len(gain) + len(bias) + len(dynamics) == 319
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
