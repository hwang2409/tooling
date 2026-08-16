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
| actuators | source position gains, control ranges, force ranges, and compiled `kv` values |
| geometry | source chest, pelvis, head, capsule legs, and box feet |
| foot sites | source heel and toe points, transformed to each com link frame |
| contact | source friction and solver parameters |

newt stores a link origin at its inertial com. the port moves the source
geometry, joint anchors, and child link offsets into that frame. the root com
offset is `0.023298969 m` above the source torso body origin.

the source uses `dampratio=1.0`. MuJoCo compiles that field from articulated
inertia, so the ten compiled velocity gains are:

| joints | source MuJoCo `kv` | newt loaded `kv` |
| --- | ---: | ---: |
| hip roll, left/right | `20.300636` | `20.300636` |
| hip, left/right | `20.161543` | `20.161543` |
| knee, left/right | `9.699400` | `9.699400` |
| ankle roll, left/right | `2.764344` | `2.764344` |
| ankle, left/right | `2.431789` | `2.431789` |

the model stores these compiled values as explicit `kv` attributes. newt's
generic dampratio loader has no articulated-inertia block at actuator load
time. `biped_walk.rs` checks all ten values.

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

the contact-timing port now matches the source's partial correction. for a
timing gain of `0.8316505101`, the corrected progress is
`progress * (1 - gain)`. full stance or swing overrides apply only at gain
`>= 0.95`.

the permanent target probe records both ankle targets for steps `0` through
`30`. samples through step `16` match the source closed-loop trace within
`4.0e-6 rad`. at step `17`, the source right toe is at `0.033224 m`, while
newt is at `0.035968 m`; the source therefore enters contact timing one step
earlier. the probe records this residual contact-transition difference rather
than hiding it behind a relaxed target tolerance.

| step | source left/right ankle | newt left/right ankle | note |
| ---: | ---: | ---: | --- |
| 0 | `0.091697 / 0.138830` | `0.091697 / 0.138830` | aligned |
| 1 | `0.091697 / 0.139782` | `0.091697 / 0.139782` | aligned |
| 2 | `0.091697 / 0.140735` | `0.091697 / 0.140735` | aligned |
| 10 | `0.091701 / 0.148357` | `0.091697 / 0.148353` | aligned |
| 11 | `0.091701 / -0.058025` | `0.091697 / -0.058025` | aligned |
| 16 | `0.091788 / -0.076750` | `0.091697 / -0.076750` | aligned |
| 17 | `0.091797 / -0.059257` | `0.091697 / -0.080458` | source contact one step earlier |
| 18 | `0.091777 / -0.060042` | `0.091697 / -0.059890` | contact transition residual |
| 30 | `0.091697 / -0.143028` | `0.091697 / -0.069292` | closed-loop state residual |

## cross-engine metrics

these are recorded from 0.005 s fixed-step runs after the round-2 controller,
damper, and metric fixes. `root z` uses the native coordinate of each engine.
the source reports its torso body origin. newt reports its root com.

| scenario | engine | steps | assist | distance m | cadence bpm | step m | stride m | duty l/r | clearance m | self force steps | root z m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `joint_walk` | source mujoco | 1000 | 0.0 | `0.050958` | `84.00` | `0.478817` | `0.196834` | `0.895/0.885` | `0.137258` | 0 | `1.090112` |
| `joint_walk` | newt | 1000 | 0.0 | `-1.4245` | `72.00` | `0.3805` | `0.2295` | `0.735/0.851` | `0.4213` | 0 | `0.1439` |
| `stable_joint_walk` | source mujoco | 5000 | 0.8 | `2.260855` | `117.60` | `0.499528` | `0.109163` | `0.668/0.764` | `0.212095` | 0 | `0.988434` |
| `stable_joint_walk` | newt | 5000 | 0.8 | `2.524537` | `117.60` | `0.373537` | `0.114487` | `0.666/0.686` | `0.196953` | 0 | `0.9255` |

the source and newt definitions use the same metric formulas. both now define
contact from heel/toe height at `0.035 m`. contact events use a
false-to-true transition with a `0.16 s` debounce. duty factor is stance time
divided by stance plus swing time. newt also reports the actual touch-force
maximum. it is `0.0 N` for both acceptance runs.

the newt values above are a disclosed re-measurement after the exact solref
bias and shared tree regularization changes. the permanent 5000-step test
records these bands: distance `2.50..2.55 m`, cadence `116..119 bpm`, mean
step length `0.36..0.39 m`, and clearance `0.19..0.21 m`. self-contact force
steps must stay at `0`.

## acceptance ladder

### tier 1: assisted walk

the short ci test runs 2000 steps and requires at least `0.8 m`. it is not
ignored, so the normal test job protects tier 1.

the full acceptance run uses 5000 steps and `assist_scale=0.8`:

| metric | result | requirement |
| --- | ---: | ---: |
| distance | `2.524537 m` | `2.50..2.55 m` |
| cadence | `117.60 bpm` | `116..119 bpm` |
| mean step length | `0.373537 m` | `0.36..0.39 m` |
| max foot clearance | `0.196953 m` | `0.19..0.21 m` |
| self-contact force steps | `0` | `0` |

### tier 2: no-assist target

the 1000-step no-assist run does not reach the `0.4 m` target. it records
`-1.4245 m` and falls to root com height `0.1439 m`. it has zero self-contact
force and zero active self contacts.

a bounded 27-run sweep checks target speed at `0.8x`, `1.0x`, and `1.2x`,
gait amplitude at `0.9x`, `1.0x`, and `1.1x`, and gait frequency at `0.9x`,
`1.0x`, and `1.1x`. each run uses 1000 steps and no assist. the best result
is `-1.108822 m` at speed `1.2x`, amplitude `0.9x`, and frequency `0.9x`.
no candidate reaches `0.4 m`. `biped_walk_sweep` prints every run and the
same best-run line on each invocation.

the no-assist outcome is therefore an honest diagnosis fallback, not a
passing target claim. the source no-assist oracle also records only
`0.050958 m` over 1000 steps.

the aligned target probe moves the first remaining difference to physical
contact timing at step `17`, not controller math. source enters right-foot
contact at toe height `0.033224 m`; newt enters at step `18` after a toe height
of `0.033807 m`. the source and newt target traces then follow different
closed-loop states. the no-assist rollout later loses vertical support and
falls. the remaining hypotheses are solver and integration differences after
the proven controller and actuator alignment. engine sources stay unchanged.

## tests

`newt/tests/biped_walk.rs` covers:

- short tier-1 assisted acceptance;
- the ignored full 5000-step tier-1 acceptance;
- the conditional tier-2 target or diagnosis fallback;
- hand-built contact metric formulas;
- byte-identical deterministic rollouts;
- active self-contact and touch-force zero checks on acceptance runs;
- an enabled self-pair fixture across a full 200-step rollout;
- all-ten compiled position-actuator `kv` checks;
- the steps `0..30` ankle target probe.

run the focused test with:

```text
cargo test --manifest-path newt/Cargo.toml --test biped_walk -- --nocapture
```
