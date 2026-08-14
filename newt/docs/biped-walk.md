# biped walking milestone

this document records the newt port of the source biped model and its
`joint_walk` controller. the source oracle is
`~/me/fun/biped/biped/mujoco_biped.py` and
`~/me/fun/biped/biped/models/biped.xml`.

## run the demo

from the repository root:

```text
cargo run --release --manifest-path newt/Cargo.toml --example biped_walk
```

the default run uses the source `stable_joint_walk` preset. it runs 5000
steps with `assist_scale=0.8`, prints gait metrics, and writes eight fixed
phase frames to `/tmp/newt-biped-walk-frames/frame-01.ppm` through
`frame-08.ppm`. each frame line includes its window distance, sampled maximum
clearance, root state, speed, contact state, and assist scale.

use `--no-assist` to run the source `joint_walk` preset. use `--steps N` and
`--out-dir PATH` to change the rollout and frame output.

## model port

`newt/models/biped-walk.xml` keeps the source model's physical values:

| part | port details |
| --- | --- |
| timestep and gravity | `0.005 s`, `0 0 -9.81` |
| integrator | rk4 |
| body mass | torso 9.7 kg; thigh 2.2 kg; shin 1.6 kg; foot 0.8 kg |
| joint layout | one free root plus ten source hinge joints |
| joint damping and armature | source values, including roll and ankle values |
| actuators | source position gains, control ranges, and force ranges |
| geometry | source chest, pelvis, head, capsule legs, and box feet |
| foot sites | source heel and toe points, transformed to each com link frame |
| contact | source friction and solver parameters |

newt stores a link origin at its inertial com. the port moves the source
geometry, joint anchors, and child link offsets into that frame. the root com
offset is `0.023298969 m` above the source torso body origin.

the source sets robot geoms to `contype=2` and `conaffinity=1`. newt's mjcf
loader disables same-tree self collision for this tree. this gives the same
robot-versus-robot collision filter while preserving robot-versus-ground
contact. the walker also declares touch sensors for every robot geom and the
ground. self force is half the robot touch sum after subtracting ground touch.

## controller port

the source scenario values come from `mujoco_biped.py:126-165`:

| value | source and newt |
| --- | ---: |
| target speed | `0.0870267973 m/s` |
| gait amplitude | `0.2124573361` |
| gait frequency | `0.9792516799 hz` |
| knee target | `0.1623653310 rad` |
| ankle target | `0.0916965369 rad` |
| source root height | `1.2431770031 m` |

the rust support module ports the tuned joint-walk path. the main pieces map
to these source sections:

- phase split and foot contact height: `mujoco_biped.py:516-533`
- tuned joint-walk constants: `mujoco_biped.py:622-710`
- contact timing, hold, and stance extension: `mujoco_biped.py:901-1050`
- reach, capture, speed regulation, swing, and stance targets:
  `mujoco_biped.py:1271-1545`
- physical balance assist: `mujoco_biped.py:2556-2593`
- gait event, stride, duty, and clearance metrics:
  `mujoco_biped.py:2780-2844`

newt applies balance as a root-link wrench. `assist_scale=0.8` scales the
source force and torque values. `assist_scale=0.0` applies no external
wrench.

the rust implementation lives in
`newt/examples/biped_walk_support.rs`. the engine crate is unchanged.

## cross-engine metrics

these are recorded from 0.005 s fixed-step runs. `root z` uses the native
coordinate of each engine. the source reports its torso body origin. newt
reports its root com.

| scenario | engine | steps | assist | distance m | cadence bpm | step m | stride m | duty l/r | clearance m | self force steps | root z m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `joint_walk` | source mujoco | 1000 | 0.0 | `0.050958` | `84.00` | `0.478817` | `0.196834` | `0.895/0.885` | `0.137258` | 0 | `1.090112` |
| `joint_walk` | newt | 1000 | 0.0 | `-0.638890` | `132.00` | `0.356900` | `0.190100` | `0.333/0.864` | `0.369000` | 0 | `0.240000` |
| `stable_joint_walk` | source mujoco | 5000 | 0.8 | `2.260855` | `117.60` | `0.499528` | `0.109163` | `0.668/0.764` | `0.212095` | 0 | `0.988434` |
| `stable_joint_walk` | newt | 5000 | 0.8 | `2.108600` | `163.20` | `0.101700` | `0.073500` | `0.592/0.582` | `0.137300` | 0 | `1.124800` |

the source and newt definitions use the same metric formulas. contact events
use a false-to-true transition with a `0.16 s` debounce. duty factor is
stance time divided by stance plus swing time. newt also reports the actual
touch-force maximum. it is `0.0 N` for both acceptance runs.

## acceptance ladder

### tier 1: assisted walk

the short ci test runs 2000 steps and requires at least `0.8 m`. it passes.

the full acceptance run uses 5000 steps and `assist_scale=0.8`:

| metric | result | requirement |
| --- | ---: | ---: |
| distance | `2.1086 m` | `>= 2.0 m` |
| cadence | `163.20 bpm` | `> 80 bpm` |
| mean step length | `0.1017 m` | `> 0.05 m` |
| max foot clearance | `0.1373 m` | `> 0.06 m` |
| self-contact force steps | `0` | `0` |

### tier 2: no-assist target

the 1000-step no-assist run does not reach the `0.4 m` target. it records
`-0.6389 m` and falls to root com height `0.2400 m`. it has zero self-contact
force and zero active self contacts.

a bounded 27-run sweep checked target speed at `0.8x`, `1.0x`, and `1.2x`,
gait amplitude at `0.9x`, `1.0x`, and `1.1x`, and gait frequency at `0.9x`,
`1.0x`, and `1.1x`. each run used 1000 steps and no assist. the best result
was `-0.284973 m` at speed `1.2x`, amplitude `0.9x`, and frequency `1.0x`.
no candidate reached `0.4 m`.

the no-assist outcome is therefore an honest diagnosis fallback, not a
passing target claim. the source no-assist oracle also records only
`0.050958 m` over 1000 steps.

the first clear actuator symptom is at step 23. source mujoco has a maximum
actuator force of `3.534 N` then, with no actuator at its force limit. newt
hits the `45 N` ankle actuator limit at the same step. the root state is still
close after accounting for the root com offset. newt later loses vertical
support near step 300 and remains fallen. this points to a remaining actuator
semantics or calibration gap between the engines. it does not justify changing
newt's engine during this model port.

## tests

`newt/tests/biped_walk.rs` covers:

- short tier-1 assisted acceptance;
- the ignored full 5000-step tier-1 acceptance;
- the conditional tier-2 target or diagnosis fallback;
- hand-built contact metric formulas;
- byte-identical deterministic rollouts;
- active self-contact and touch-force zero checks on acceptance runs.

run the focused test with:

```text
cargo test --manifest-path newt/Cargo.toml --test biped_walk -- --nocapture
```
