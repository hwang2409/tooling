# depth of field

chimy2 uses a thin-lens circle of confusion in view units:

`CoC = aperture * |focal_length * (focus_distance - depth)| / (depth * (focus_distance - focal_length))`

the implementation fixes `focal_length` at one view unit. `aperture` is a
texel-scaled strength. the result is clamped to `max_coc_radius` texels.

the pass uses a deterministic 24-tap disc gather with interior samples. it clamps taps at the frame
edge and rejects a more-defocused background tap behind a less-defocused
center. this reduces sharp-foreground halos. it does not implement true
near-field scatter. that needs a separate foreground layer and is out of scope.

DoF runs after SSAO and before bloom and tonemapping in linear color space.
the gather is expensive: each defocused pixel can load up to 24 color samples
and 24 reconstructed depth samples. compare it with and without `--dof` at
the demo resolution with:

```text
CHIMY_THREADS=1 cargo run --manifest-path chimy2/Cargo.toml --bin postfx_demo -- --dof --frames 1 --size 960x640 --screenshot /tmp/chimy2-dof.ppm
CHIMY_THREADS=1 cargo run --manifest-path chimy2/Cargo.toml --bin postfx_demo -- --frames 1 --size 960x640 --screenshot /tmp/chimy2-no-dof.ppm
```

measure each command with `/usr/bin/time -p` on the target machine. the
single-thread command makes the comparison repeatable. the renderer remains
deterministic with its normal worker count.
