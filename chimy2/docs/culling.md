# frustum culling

mesh bounds use one local-space axis-aligned bounding box per mesh. the cache
is private and rebuilds during every public mesh mutation. each submission
transforms all eight corners through its local-to-clip matrix.

the test rejects a submission only when all eight transformed corners are
outside one plane. a straddling bound stays submitted. this is conservative
for affine model transforms, including non-uniform scale and rotation.

planes use the Gribb-Hartmann method: Gil Gribb and Klaus Hartmann, "Fast
Extraction of Viewing Frustum Planes from the World-View-Projection Matrix."
`Mat4` stores columns, so extraction reads rows with `data[column * 4 + row]`.

color mesh draws get the camera clip transform from their shader uniforms.
depth draws get the active light clip transform from `ShadowDepthUniforms`.
the pipeline defaults to culling enabled; `set_culling_enabled(false)` keeps
the byte-identical culling-off path for debugging and anchor tests.

shadow passes install a pass-local frustum. they expand its planes to include
all caster bounds for that pass. this keeps casters outside the camera view,
or outside a tight cascade xy box, when their shadows reach visible receivers.
directional cascades, point-light cube faces, and the PCSS pass use their own
light matrices. no shadow pass uses the camera frustum.
