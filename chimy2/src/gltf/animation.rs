#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeTransform {
    pub translation: Vec3,
    pub rotation: [f32; 4],
    pub scale: Vec3,
}

impl NodeTransform {
    fn from_node(node: &GltfNode) -> Self {
        Self {
            translation: node.translation,
            rotation: node.rotation,
            scale: node.scale,
        }
    }
    fn matrix(self) -> Mat4 {
        if self.rotation == [0.0, 0.0, 0.0, 1.0] {
            Mat4::translate(self.translation) * Mat4::scale(self.scale)
        } else {
            Mat4::translate(self.translation) * quat_matrix(self.rotation) * Mat4::scale(self.scale)
        }
    }
}

impl GltfAnimationSampler {
    pub fn sample(&self, time: f32) -> Result<[f32; 4], GltfError> {
        self.sample_with_mode(time, self.output_components == 4)
    }

    /// Samples one shared animation interpolation path. Rotation channels use
    /// quaternion slerp; weights and vector channels use component lerp.
    fn sample_with_mode(&self, time: f32, rotation: bool) -> Result<[f32; 4], GltfError> {
        if self.input.is_empty() || self.output.is_empty() {
            return Err(GltfError::new("animation sampler has no keyframes"));
        }
        let last = self.input.len() - 1;
        if time <= self.input[0] {
            return Ok(self.output[0]);
        }
        if time >= self.input[last] {
            return Ok(self.output[last]);
        }
        let index = self
            .input
            .windows(2)
            .position(|window| time < window[1])
            .unwrap_or(last - 1);
        let factor = ((time - self.input[index]) / (self.input[index + 1] - self.input[index]))
            .clamp(0.0, 1.0);
        if self.interpolation == Interpolation::Step {
            return Ok(self.output[index]);
        }
        if rotation {
            Ok(slerp(self.output[index], self.output[index + 1], factor))
        } else {
            Ok(lerp4(self.output[index], self.output[index + 1], factor))
        }
    }
}

pub fn sanitize_morph_weight(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub fn normalize_weights(weights: [f32; 4]) -> [f32; 4] {
    let sum = weights.into_iter().map(|value| value.max(0.0)).sum::<f32>();
    if sum < WEIGHT_TOLERANCE {
        [0.0; 4]
    } else {
        weights.map(|value| value.max(0.0) / sum)
    }
}

pub fn slerp(a: [f32; 4], b: [f32; 4], factor: f32) -> [f32; 4] {
    let mut b = b;
    let mut dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    if dot < 0.0 {
        b = b.map(|value| -value);
        dot = -dot;
    }
    if dot > 0.9995 {
        return normalize_quat(lerp4(a, b, factor));
    }
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();
    let first = ((1.0 - factor) * theta).sin() / sin_theta;
    let second = (factor * theta).sin() / sin_theta;
    normalize_quat([
        a[0] * first + b[0] * second,
        a[1] * first + b[1] * second,
        a[2] * first + b[2] * second,
        a[3] * first + b[3] * second,
    ])
}

fn lerp4(a: [f32; 4], b: [f32; 4], factor: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * factor,
        a[1] + (b[1] - a[1]) * factor,
        a[2] + (b[2] - a[2]) * factor,
        a[3] + (b[3] - a[3]) * factor,
    ]
}
fn normalize_quat(value: [f32; 4]) -> [f32; 4] {
    let length = value.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length == 0.0 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        value.map(|v| v / length)
    }
}
fn quat_matrix(value: [f32; 4]) -> Mat4 {
    let q = normalize_quat(value);
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    Mat4::new([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        0.0,
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        0.0,
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}
