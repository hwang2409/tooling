#!/usr/bin/env python3
"""Capture fixed MuJoCo oracle rows for the v3 biped acceptance sweep."""

from __future__ import annotations

import argparse
import datetime as date
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "me/fun/biped"))
from biped import mujoco_biped as biped  # noqa: E402
import mujoco  # noqa: E402


MAGIC = b"NEWTBIP3"
JOINT_NAMES = biped.JOINT_NAMES
QPOS_COUNT = 7 + len(JOINT_NAMES)
QVEL_COUNT = 6 + len(JOINT_NAMES)


def _matched_load_model(config, original_load_model):
    mujoco_module, model, data = original_load_model(config)
    model.opt.integrator = mujoco_module.mjtIntegrator.mjINT_EULER
    model.opt.solver = mujoco_module.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco_module.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 20
    return mujoco_module, model, data


def _state_vectors(sample) -> tuple[list[float], list[float]]:
    qpos = list(sample.root_position) + list(sample.root_orientation)
    qpos.extend(sample.joint_positions[name] for name in JOINT_NAMES)
    qvel = list(sample.root_velocity)
    qvel.extend(sample.joint_velocities[name] for name in JOINT_NAMES)
    return qpos, qvel


def capture(assist_scale: float, output: Path, steps: int, stride: int) -> None:
    scenario = "joint_walk" if assist_scale == 0.0 else "stable_joint_walk"
    config = biped.BipedSimConfig.for_scenario(
        scenario,
        steps=steps,
        frame_stride=stride,
        assist_scale=assist_scale,
    )
    original_load_model = biped._load_model
    biped._load_model = lambda current_config: _matched_load_model(
        current_config, original_load_model
    )
    try:
        result = biped.run_biped_simulation(config)
    finally:
        biped._load_model = original_load_model
    summary = result.summary
    outcome = 1 if summary["final_status"] == "fallen" else 0
    fall_step = int(summary["steps_simulated"]) if outcome else 0
    provenance = (
        f"name=biped_walk_oracle_v3|mujoco={mujoco.__version__}"
        f"|source_model={biped.MODEL_ID}|controller={config.controller_name}"
        f"|dt={config.dt:.9g}|integrator=Euler|solver=Newton|cone=pyramidal|iters=20"
        f"|steps={steps}|stride={stride}"
        f"|assist_scale={assist_scale:.1f}|nq={QPOS_COUNT}|nv={QVEL_COUNT}"
        f"|date={date.date.today().isoformat()}"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(provenance)))
        stream.write(provenance)
        stream.write(struct.pack("<dIII", assist_scale, steps, int(summary["steps_simulated"]), fall_step))
        stream.write(struct.pack("<B", outcome))
        metrics = (
            float(summary["forward_distance"]),
            float(summary["cadence"]),
            float(summary["mean_step_length"]),
            float(summary["mean_stride_length"]),
            float(summary["max_foot_clearance"]),
            float(summary["final_root_height"]),
            float(summary["final_forward_speed"]),
            float(summary["max_self_contact_force"]),
        )
        stream.write(struct.pack("<8dII", *metrics, int(summary["self_contact_force_steps"]), int(summary["ground_contact_force_steps"])))
        stream.write(struct.pack("<I", len(result.state_samples)))
        for sample in result.state_samples:
            qpos, qvel = _state_vectors(sample)
            assert len(qpos) == QPOS_COUNT
            assert len(qvel) == QVEL_COUNT
            stream.write(struct.pack("<I", sample.step))
            contact_mask = (1 if any("left" in name for name in sample.contacts) else 0) | (
                2 if any("right" in name for name in sample.contacts) else 0
            )
            stream.write(struct.pack("<B", contact_mask))
            stream.write(struct.pack(f"<{QPOS_COUNT}d", *qpos))
            stream.write(struct.pack(f"<{QVEL_COUNT}d", *qvel))
    print(
        f"wrote {output} :: assist={assist_scale:.1f} status={summary['final_status']} "
        f"steps={summary['steps_simulated']} distance={summary['forward_distance']:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assist-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.assist_scale not in {0.0, 0.2, 0.4, 0.8}:
        parser.error("assist-scale must be one of 0.0, 0.2, 0.4, or 0.8")
    if args.steps <= 0 or args.stride <= 0:
        parser.error("steps and stride must be positive")
    capture(args.assist_scale, args.output, args.steps, args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
