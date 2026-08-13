# newt

A from-scratch rigid-body physics engine in rust. sibling to
[chimy2](../chimy2/) (the software rasterizer that visualizes it). the endgame
is an interactive robot simulation in the browser via wasm, culminating in a
port of the biped walker.

design and scope: [docs/superpowers/specs/2026-08-13-newt-physics-design.md](docs/superpowers/specs/2026-08-13-newt-physics-design.md).
current tier reference: [docs/core.md](docs/core.md).

zero runtime dependencies. demos may path-depend on chimy2 (dev-dependency).
no platform libm anywhere in the engine (sin/cos/tan/exp/ln are hand-written
range-reduced polynomials). fixed timestep, deterministic scalar policy,
byte-identical goldens across macos and linux.
