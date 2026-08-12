# image-based lighting

`IblMaps::from_environment` bakes all maps once after the source
`CubeTexture` loads. The source cube uses its documented face table and color
space decoder. All bake and shader arithmetic uses linear RGB.

The default knobs are:

| map | size | fixed samples | method |
| --- | --- | --- | --- |
| irradiance | 16x16 per face | 64 per texel | cosine-weighted hemisphere, normalized by PI |
| prefiltered environment | 32x32 per face, 5 levels | 64 per texel per level | GGX importance sampling, roughness 0 to 1 |
| environment BRDF | 64x64 | 64 per texel | Karis split-sum Smith-Schlick visibility |

The sequence is Hammersley: the first coordinate is `(i + 0.5) / n`, and the
second is the base-2 radical inverse of `i`. It has no runtime random state.
The same source and settings therefore produce byte-identical maps.

The irradiance map stores the normalized integral
`integral(environment * N dot L * d_omega) / PI`. A constant environment is
therefore unchanged. The shader multiplies it by albedo and
`kd = (1 - F) * (1 - metallic)`.

The specular term samples the roughness-selected prefiltered cube and the BRDF
LUT. It evaluates as `prefiltered * (F0 * lut.x + lut.y)`. At `N dot V = 1`
and roughness zero, the LUT anchor is `(1, 0)` within the fixed-sample
calculation.

`IblCookTorranceShader` keeps direct lights unchanged. `pbr_demo --ibl` uses
this variant. Add `--hdr` to retain bright floating-point environment values
through ACES tonemapping. If the environment changes, call
`IblMaps::from_environment_with_settings` again; the maps are owned bake
results and have no stale cache path. Applications with decoded HDR float
faces can use `IblMaps::from_float_environment`; values above 1.0 stay linear.
