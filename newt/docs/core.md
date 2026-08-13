# newt core (tier 1)

math + spatial algebra + free bodies + RK4 + gravity. no joints, no contacts,
no actuators (later tiers). the tumbling boxes demo is what this tier ships.

design of record: [superpowers/specs/2026-08-13-newt-physics-design.md](superpowers/specs/2026-08-13-newt-physics-design.md).

## state layout

per body:

- `mass: f32`
- `inertia_body: Mat3` — inertia tensor about the COM in body axes; passed in
  by the caller (`Body::solid_box` fills a uniform-density brick for you)
- `inertia_body_inverse: Mat3` — precomputed at construction time (`Body::new`
  panics on a singular inertia)
- `position: Vec3` — world-frame COM
- `linear_velocity: Vec3` — world-frame
- `orientation: Quat` — body → world; unit quaternion, x/y/z/w with w scalar
- `angular_velocity_body: Vec3` — body-frame ω. we keep it in the body frame
  so the inertia tensor stays constant and euler's equations take their
  simple form; convert to world with `body.angular_velocity_world()` when
  needed

per world:

- `dt: f32` (default 0.005 = 5 ms; matches biped)
- `gravity: Vec3` (default (0, 0, -9.81))
- `bodies: Vec<Body>` — insertion-ordered; index-stable

## integration order (per rk4 step)

for each body, `world.step()` runs one full rk4 step:

1. evaluate k1 at the current state
2. evaluate k2 at state + k1 * (dt/2)
3. evaluate k3 at state + k2 * (dt/2)
4. evaluate k4 at state + k3 * dt
5. combine: `y_new = y + (k1 + 2 k2 + 2 k3 + k4) * (dt/6)`
6. quaternion `renormalize()` once, at the very end of the step

what each `evaluate` does:

- `dp/dt = linear_velocity`
- `dv/dt = gravity` (tier 1 has no other forces)
- `dω_body/dt = I⁻¹ * (- ω × (I ω))` — euler's equation, torque-free
- `dq/dt = 0.5 * q * (ω_body, 0)` — pure-quaternion multiply

intermediate stages do NOT renormalize the quaternion — that would make
`evaluate` nonlinear across a step and break rk4's formal order. renormalize
once at step end.

## scalar policy (determinism)

only `+ - * /` and `f32::sqrt` are allowed in the engine (they are
ieee-exact and so bit-identical across x86_64, aarch64, wasm32). every
transcendental is hand-written in `src/math.rs`:

- `newt::math::sin`, `newt::math::cos`, `newt::math::tan` — cody-waite π/2
  range reduction feeding degree-7/8 minimax polynomials on `[-π/4, π/4]`
- `newt::math::abs` — bit fiddling

no platform libm (`.sin() .cos() .tan() .exp() .ln() .powf()`) anywhere in
the engine. grep it if you doubt me:

```sh
rg -tp rust '\.(sin|cos|tan|exp|ln|powf)\(' src/
```

should be empty.

## anchor tests

`cargo test` runs all of these. see `tests/*.rs`.

| file | what it pins down |
|------|-------------------|
| `tests/projectile.rs` | free-fall COM matches closed-form parabola after 1000 steps |
| `tests/energy_and_momentum.rs` | torque-free tumbling: energy drift < 1e-3 over 10k steps, angular momentum in world frame < 5e-3 |
| `tests/energy_and_momentum.rs::dzhanibekov_intermediate_axis_flip_occurs` | intermediate-axis instability flips a body spun about I₂ |
| `tests/energy_and_momentum.rs::major_axis_spin_stays_bounded`, `::minor_axis_spin_stays_bounded` | I₁ and I₃ spins are stable |
| `tests/golden.rs` | serialize `(q, qdot)` for a fixed 3-body scene at steps 0/100/1000; byte-compare against `tests/goldens/tumbling_3_body.bin` |

the golden was generated on macOS aarch64. CI on ubuntu-latest re-runs the
golden test; if the bytes differ, the scalar policy is being violated
somewhere. regenerate ONLY on the reference machine:

```sh
cargo test --test golden regenerate_golden -- --ignored --nocapture
```

## running the demo

three tumbling boxes under gravity, rendered as wireframes into a ppm via
chimy2's `Framebuffer` and projection helpers:

```sh
cargo run --release --example tumble -- --frames 200 --out /tmp/tumble.ppm --size 640x360
```

then convert to png (macOS):

```sh
sips -s format png /tmp/tumble.ppm --out /tmp/tumble.png
```

or open the ppm directly in most image viewers. the demo is deliberately
minimal — no lighting, no shaders — its job is to prove that chimy2 can
consume newt state, not to exercise the full renderer. tier 6 wires the
proper renderer in.

## verification

fmt, clippy, test:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

CI runs the same three jobs from `.github/workflows/newt.yml`.
