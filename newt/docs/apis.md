# newt public dynamics apis

## jacobians

`Tree::link_jacobian`, `Tree::point_jacobian`, and
`World::tree_point_jacobian` return a dense `Jacobian`. Each column is a
world-frame vector. `translational[i]` maps a generalized rate to point
linear velocity. `rotational[i]` maps it to angular velocity.

| columns | slots |
|---|---|
| free root | angular body x/y/z, then linear body x/y/z |
| hinge | one scalar at `Tree::v_offset[link]` |
| slide | one scalar at `Tree::v_offset[link]` |
| ball | three scalar slots at `Tree::v_offset[link]..+3` |

`Jacobian::velocity(qdot)` returns `(linear_world, angular_world)`.

## inverse dynamics

`Tree::inverse_dynamics_at(q, qdot, qddot, gravity, external_wrenches)` and
`World::inverse_dynamics_at` expose RNE for an explicit state.

RNE includes rigid-body inertia, Coriolis terms, and gravity. It excludes
damping, armature, limits, actuators, `qfrc_applied`, and persistent applied
wrenches. Pass external wrenches explicitly. The world method uses its
`gravity` field.

## keyframes

JSON stores snapshots under `keyframes`:

```json
{"name":"ready","q":[0.2],"qdot":[0],"act":[0],"ctrl":[0.4]}
```

Vectors flatten trees and actuators in declaration order. Dimensions are
checked at load time. `World::reset_to_keyframe` copies all four vectors.
MJCF uses `<keyframe><key name="ready" qpos="..." qvel="..." act="..." ctrl="..."/></keyframe>`.

## mocap bodies

Set the root link's `mocap` flag in JSON or MJCF. Use
`World::set_mocap_pose` each step. Use `Tree::set_mocap_velocity` for the
contact-relative velocity. Mocap roots are not integrated and have infinite
effective contact mass. Contacts still apply forces to dynamic bodies.
