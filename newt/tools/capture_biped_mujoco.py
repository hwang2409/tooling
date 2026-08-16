#!/usr/bin/env python3
"""Capture the source-faithful assisted biped under matched MuJoCo settings.

This uses the biped repository's controller on newt's ported MJCF. The Rust
acceptance harness uses the same model and controller port, so the fixture
isolates engine contact and integration differences.
"""

from __future__ import annotations

import argparse
import datetime as date
import re
import struct
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path.home() / "me/fun/biped"))
from biped import mujoco_biped as biped  # noqa: E402


MAGIC = b"NEWTDIF1"
MODEL = Path(__file__).resolve().parents[1] / "models/biped-walk.xml"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tests/references/biped_assisted_walk_newton_euler.bin"


def _mujoco_source() -> str:
    source = MODEL.read_text(encoding="utf-8")
    # Newt's geom touch sensors are an extension. The controller only needs
    # contact geometry and foot sites, so omit that non-MuJoCo sensor block.
    source = re.sub(r"<sensor>.*?</sensor>", "", source, flags=re.S)

    def add_inertial_pos(match: re.Match[str]) -> str:
        tag = match.group(0)
        if " pos=" in tag:
            return tag
        if tag.endswith("/>"):
            return tag[:-2] + ' pos="0 0 0"/>'
        return tag[:-1] + ' pos="0 0 0">'

    return re.sub(r"<inertial[^>]*>", add_inertial_pos, source)


def _write_fixture(path: Path, qpos_samples: list, qvel_samples: list, steps: int, stride: int) -> None:
    provenance = (
        f"name=biped_assisted_walk|mujoco={mujoco.__version__}|dt=0.005"
        f"|integ=0|solver=2|cone=0|iters=20|nq={qpos_samples[0].size}"
        f"|nv={qvel_samples[0].size}|stride={stride}|n_steps={steps}"
        f"|date={date.date.today().isoformat()}"
    ).encode()
    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<I", len(provenance)))
        output.write(provenance)
        output.write(
            struct.pack(
                "<IIIII",
                qpos_samples[0].size,
                qvel_samples[0].size,
                stride,
                len(qpos_samples),
                steps,
            )
        )
        for sample, (qpos, qvel) in enumerate(zip(qpos_samples, qvel_samples)):
            output.write(struct.pack("<I", sample * stride))
            output.write(struct.pack(f"<{qpos.size}d", *qpos.astype("float64")))
            output.write(struct.pack(f"<{qvel.size}d", *qvel.astype("float64")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.steps <= 0 or args.stride <= 0:
        parser.error("steps and stride must be positive")

    config = biped.BipedSimConfig.for_scenario("stable_joint_walk", steps=args.steps)
    model = mujoco.MjModel.from_xml_string(_mujoco_source())
    model.opt.timestep = config.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 20
    data = mujoco.MjData(model)
    controller = biped.CONTROLLERS[config.controller_name]
    biped._set_initial_pose(mujoco, model, data, config, controller)

    qpos_samples = [data.qpos.copy()]
    qvel_samples = [data.qvel.copy()]
    for step in range(1, args.steps + 1):
        biped._apply_balance_controller(mujoco, model, data, config, data.time)
        controller.before_step(mujoco, model, data, step, config)
        targets = controller.targets(config, data.time, data=data)
        biped._apply_controls(mujoco, model, data, targets)
        mujoco.mj_step(model, data)
        if step % args.stride == 0:
            qpos_samples.append(data.qpos.copy())
            qvel_samples.append(data.qvel.copy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_fixture(args.output, qpos_samples, qvel_samples, args.steps, args.stride)
    print(f"wrote {args.output} :: nq={model.nq} nv={model.nv} steps={args.steps} stride={args.stride}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
