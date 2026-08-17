# newt performance

this document records the permanent newt benchmark contract and its measured
results. timing changes are not a physics acceptance criterion. all goldens
and differential fixtures remain byte-identical.

## method

the runner is `scripts/bench-newt.sh`. it invokes the std-only harness in
`benches/newt_perf.rs`. the harness uses `std::time::Instant` and reports
nearest-rank p10, median, and p90 values over sorted samples. each run uses
three warmup samples and 15 measured samples. each sample runs 1,000 steps.

the suite uses these fixed scenes:

| scene | fixture | coverage |
| --- | --- | --- |
| sphere_drop | `tests/references/sphere_drop.xml` | one free body and contact |
| box_stack | `tests/references/box_stack.xml` | three free bodies and contacts |
| biped_assisted_walk | `models/biped-walk.xml` | assisted controller, tree, contacts |
| tendon_arm | `tests/references/tendon_mixed_wrap.xml` | sphere and cylinder wraps |
| hfield_terrain_roll | `tests/references/hfield_sphere_ramp.xml` | heightfield contact |
| pile | `models/pile.json` | mixed free-body contact shapes |
| muscle_pendulum | `tests/references/muscle_pendulum.xml` | muscle actuator and tree |

every scene runs `penalty-rk4`, `pgs-euler`, and `newton-euler`. the biped
case uses `stable_joint_walk(1000)`, with assist scale `0.8`. its timed helper
includes scene setup and controller setup because the controller is an
example-only module. all other cases load and configure the scene outside the
timed loop, then clone the configured initial world before each sample.

the json-lines report goes to stdout. the human table goes to stderr. the
checksum is a run-use guard and is not a trajectory comparison. use the
existing golden and differential tests for byte identity.

## baseline

baseline commit: `ef9a53f` (`Add newt benchmark harness`)

machine: Apple M5 Pro, 15 cores, macOS 26.5.1. rustc 1.95.0.
the machine was otherwise idle. values are wall-clock medians from one run.

| scene | config | median ns/step | p10 ns/step | p90 ns/step | steps/sec |
| --- | --- | ---: | ---: | ---: | ---: |
| sphere_drop | penalty-rk4 | 803.5 | 800.9 | 885.5 | 1,244,620 |
| sphere_drop | pgs-euler | 3,021.1 | 2,956.9 | 3,550.0 | 331,003 |
| sphere_drop | newton-euler | 1,823.0 | 1,735.3 | 2,232.2 | 548,559 |
| box_stack | penalty-rk4 | 5,479.8 | 5,109.7 | 7,999.7 | 182,487 |
| box_stack | pgs-euler | 44,994.7 | 38,012.2 | 52,817.5 | 22,225 |
| box_stack | newton-euler | 110,390.5 | 98,353.5 | 127,208.3 | 9,059 |
| biped_assisted_walk | penalty-rk4 | 36,907.2 | 35,064.6 | 45,334.8 | 27,095 |
| biped_assisted_walk | pgs-euler | 76,057.2 | 68,124.8 | 84,727.2 | 13,148 |
| biped_assisted_walk | newton-euler | 88,912.5 | 78,608.5 | 105,283.7 | 11,247 |
| tendon_arm | penalty-rk4 | 6,455.2 | 5,682.0 | 16,907.8 | 154,913 |
| tendon_arm | pgs-euler | 1,416.0 | 1,385.8 | 3,530.0 | 706,236 |
| tendon_arm | newton-euler | 1,874.5 | 1,399.3 | 2,079.2 | 533,488 |
| hfield_terrain_roll | penalty-rk4 | 4,257.8 | 4,099.5 | 5,932.0 | 234,864 |
| hfield_terrain_roll | pgs-euler | 3,288.9 | 3,071.9 | 4,137.8 | 304,052 |
| hfield_terrain_roll | newton-euler | 2,262.8 | 2,252.6 | 2,454.7 | 441,924 |
| pile | penalty-rk4 | 5,444.0 | 5,335.7 | 5,882.2 | 183,688 |
| pile | pgs-euler | 29,813.4 | 27,326.8 | 33,811.8 | 33,542 |
| pile | newton-euler | 70,461.0 | 61,132.3 | 81,762.8 | 14,192 |
| muscle_pendulum | penalty-rk4 | 3,401.2 | 3,315.7 | 3,523.6 | 294,013 |
| muscle_pendulum | pgs-euler | 829.2 | 816.8 | 847.9 | 1,205,970 |
| muscle_pendulum | newton-euler | 828.9 | 819.8 | 861.0 | 1,206,393 |

## pre-optimization profile

the feature-gated manual timers were enabled with
`--features instrumentation`. they add no fields or timer calls to a
default build. percentages below are sums across the 15 measured samples.
the `integration` bucket includes the free-body solver for Euler cases.

| scene/config | collision | solver phase | integration | sensors |
| --- | ---: | ---: | ---: | ---: |
| box_stack / pgs-euler | 3.4% | 0.1% | 96.2% | 0.1% |
| box_stack / newton-euler | 1.4% | 0.1% | 98.5% | 0.0% |
| tendon_arm / penalty-rk4 | 0.3% | 0.3% | 98.0% | 0.6% |
| muscle_pendulum / penalty-rk4 | 0.5% | 0.4% | 96.6% | 1.0% |
| hfield_terrain_roll / pgs-euler | 30.4% | 1.0% | 65.8% | 1.1% |
| hfield_terrain_roll / newton-euler | 39.6% | 1.3% | 55.4% | 1.4% |

the measured hotspot groups are:

1. articulated-tree integration dominates tendon and muscle cases at
   96.6% to 98.0% of timed steps.
2. the free-body constraint and integration path dominates box stack solver
   cases at 96.2% to 98.5%.
3. heightfield contact candidate and narrow-phase work takes 30.4% for PGS
   and 39.6% for Newton.
4. the remaining solver and sensor phases stay below 5.5% in these samples.

the first optimization targets the empty free-body solver dispatch inside the
integration bucket. later results will record the exact before and after
numbers, the measured change, and any scene that does not improve.

## optimizations

the optimization commits are independent and reversible:

| commit | change | measured result |
| --- | --- | --- |
| `5969385` | reuse the ABA workspace for Euler steps | muscle PGS: 829.2 → 573.9 ns/step, 30.8% faster; muscle Newton: 828.9 → 562.1 ns/step, 32.2% faster |
| `e2f6ee6` | reuse the ABA workspace across RK4 stages | tendon penalty: 6,455.2 → 4,797.5 ns/step, 25.7% faster; muscle penalty: 3,401.2 → 2,542.1 ns/step, 25.3% faster |

both changes reuse scratch storage only. they do not change arithmetic order.
the full differential suite and all byte-identity golden suites passed after
each commit.

## final table

the final run used the same machine, warmup, iteration count, and step count
as the baseline. `change` is the median wall-time change. negative values are
faster. The final checksums matched the baseline checksums in all 21 rows.

| scene | config | baseline ns/step | final ns/step | change |
| --- | --- | ---: | ---: | ---: |
| sphere_drop | penalty-rk4 | 803.5 | 740.9 | -7.8% |
| sphere_drop | pgs-euler | 3,021.1 | 2,662.7 | -11.9% |
| sphere_drop | newton-euler | 1,823.0 | 1,634.6 | -10.3% |
| box_stack | penalty-rk4 | 5,479.8 | 4,738.1 | -13.5% |
| box_stack | pgs-euler | 44,994.7 | 33,806.7 | -24.9% |
| box_stack | newton-euler | 110,390.5 | 94,735.9 | -14.2% |
| biped_assisted_walk | penalty-rk4 | 36,907.2 | 31,675.2 | -14.2% |
| biped_assisted_walk | pgs-euler | 76,057.2 | 65,050.1 | -14.5% |
| biped_assisted_walk | newton-euler | 88,912.5 | 75,685.4 | -14.9% |
| tendon_arm | penalty-rk4 | 6,455.2 | 4,381.2 | -32.1% |
| tendon_arm | pgs-euler | 1,416.0 | 1,042.1 | -26.4% |
| tendon_arm | newton-euler | 1,874.5 | 1,046.3 | -44.2% |
| hfield_terrain_roll | penalty-rk4 | 4,257.8 | 3,819.6 | -10.3% |
| hfield_terrain_roll | pgs-euler | 3,288.9 | 2,862.2 | -13.0% |
| hfield_terrain_roll | newton-euler | 2,262.8 | 2,238.6 | -1.1% |
| pile | penalty-rk4 | 5,444.0 | 4,980.7 | -8.5% |
| pile | pgs-euler | 29,813.4 | 26,301.5 | -11.8% |
| pile | newton-euler | 70,461.0 | 59,096.8 | -16.1% |
| muscle_pendulum | penalty-rk4 | 3,401.2 | 2,459.3 | -27.7% |
| muscle_pendulum | pgs-euler | 829.2 | 560.0 | -32.5% |
| muscle_pendulum | newton-euler | 828.9 | 557.5 | -32.8% |

the hfield implementation did not change in these optimization commits. its
final run was faster, but this suite does not attribute that change to either
optimization. no final row regressed in this run.

## instrumentation check

the default build removes the `StepTimings` field, timer calls, and accessor
with `cfg(feature = "instrumentation")`. A repeated feature-off
`sphere_drop/penalty-rk4` run measured 740.9 and 759.9 ns/step. The spread was
2.5%, within this wall-clock run's noise. The feature-on run measured 961.9
ns/step because it records five `Instant` values per step. This is expected
instrumentation cost and is absent from the default binary.

the optimized workspace does not make a zero-allocation claim for the full
public `World::step` path. contact buffers, poses, and acceleration results
still allocate in existing paths. The guard is byte identity, not an
allocation count.
