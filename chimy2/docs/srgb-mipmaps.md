# sRGB and mipmap pipeline

Texture files store RGB as sRGB u8 values. `Texture::new` decodes those
values with the 256-entry sRGB LUT. Alpha stays a linear normalized value.

Mip levels use linear RGB and a box filter. Each non-power-of-two dimension is
ceil-halved. An edge box averages only source texels that exist; it does not
duplicate the last texel. Bilinear filtering runs within one linear mip.
Trilinear filtering blends the two bilinear mip results at the analytic LOD.

The raster core computes affine barycentric gradients, then passes them with
the inverse clip-w values to `Varyings::interpolate3`. The textured varying
implementation applies the quotient rule to its perspective-correct UVs and
returns `TextureDerivatives`. This is the sampler-facing seam: the core does
not know texture or shader details, and general shaders receive default empty
derivatives.

Blinn-Phong material and light colors are authored as sRGB `Vec3` values and
converted in their constructors. Textures, material colors, and light values
then stay linear through lighting and blending. `argb8888_linear` is the
framebuffer boundary: it encodes linear RGB to sRGB u8 values. Clear colors and
flat u32 colors are already framebuffer sRGB values.
