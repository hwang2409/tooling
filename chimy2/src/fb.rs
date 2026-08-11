#[derive(Clone, Debug, PartialEq)]
pub struct Framebuffer {
    pub color: Vec<u32>,
    pub depth: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            color: vec![0; width.saturating_mul(height)],
            depth: vec![1.0; width.saturating_mul(height)],
            width,
            height,
        }
    }

    pub fn clear(&mut self, color: u32) {
        self.color.fill(color);
        self.depth.fill(1.0);
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        let length = width.saturating_mul(height);
        self.width = width;
        self.height = height;
        self.color = vec![0; length];
        self.depth = vec![1.0; length];
    }

    /// Writes a pixel. Coordinates outside the framebuffer are ignored.
    pub fn put_pixel(&mut self, x: usize, y: usize, color: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let Some(index) = y.checked_mul(self.width).and_then(|row| row.checked_add(x)) else {
            return;
        };
        if let Some(pixel) = self.color.get_mut(index) {
            *pixel = color;
        }
    }

    /// Downsamples an integer supersampled framebuffer with a linear-light
    /// box filter. The destination dimensions must divide the source.
    pub fn downsample_linear_into(&self, destination: &mut Self) {
        if destination.width == 0
            || destination.height == 0
            || self.width == 0
            || self.height == 0
            || self.width % destination.width != 0
            || self.height % destination.height != 0
        {
            return;
        }
        let scale_x = self.width / destination.width;
        let scale_y = self.height / destination.height;
        if scale_x == 0 || scale_x != scale_y {
            return;
        }
        let sample_count = (scale_x * scale_y) as f32;
        for y in 0..destination.height {
            for x in 0..destination.width {
                let mut rgb = [0.0; 3];
                let mut alpha = 0.0;
                for sample_y in 0..scale_y {
                    for sample_x in 0..scale_x {
                        let source_index =
                            (y * scale_y + sample_y) * self.width + x * scale_x + sample_x;
                        let [source_alpha, red, green, blue] =
                            self.color[source_index].to_be_bytes();
                        rgb[0] += crate::image::srgb_to_linear_u8(red);
                        rgb[1] += crate::image::srgb_to_linear_u8(green);
                        rgb[2] += crate::image::srgb_to_linear_u8(blue);
                        alpha += f32::from(source_alpha) / 255.0;
                    }
                }
                destination.color[y * destination.width + x] = argb8888_linear(
                    alpha / sample_count,
                    [
                        rgb[0] / sample_count,
                        rgb[1] / sample_count,
                        rgb[2] / sample_count,
                    ],
                );
                destination.depth[y * destination.width + x] = 1.0;
            }
        }
    }
}

pub const fn argb8888(alpha: u8, red: u8, green: u8, blue: u8) -> u32 {
    u32::from_be_bytes([alpha, red, green, blue])
}

/// Encodes a linear-light shader result at the framebuffer boundary.
/// Alpha is linear; RGB is encoded to the sRGB presentation format.
pub fn argb8888_linear(alpha: f32, rgb: [f32; 3]) -> u32 {
    argb8888(
        (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
        crate::image::linear_to_srgb(rgb[0]),
        crate::image::linear_to_srgb(rgb[1]),
        crate::image::linear_to_srgb(rgb[2]),
    )
}

pub fn blend_argb8888_linear(destination: u32, source: u32) -> u32 {
    let [source_alpha, source_red, source_green, source_blue] = source.to_be_bytes();
    let [
        destination_alpha,
        destination_red,
        destination_green,
        destination_blue,
    ] = destination.to_be_bytes();
    let source_alpha = f32::from(source_alpha) / 255.0;
    let destination_alpha = f32::from(destination_alpha) / 255.0;
    let inverse_source_alpha = 1.0 - source_alpha;
    let output_alpha = source_alpha + destination_alpha * inverse_source_alpha;
    let source_rgb = [
        crate::image::srgb_to_linear_u8(source_red),
        crate::image::srgb_to_linear_u8(source_green),
        crate::image::srgb_to_linear_u8(source_blue),
    ];
    let destination_rgb = [
        crate::image::srgb_to_linear_u8(destination_red),
        crate::image::srgb_to_linear_u8(destination_green),
        crate::image::srgb_to_linear_u8(destination_blue),
    ];
    argb8888_linear(
        output_alpha,
        std::array::from_fn(|channel| {
            let premultiplied = source_rgb[channel] * source_alpha
                + destination_rgb[channel] * destination_alpha * inverse_source_alpha;
            if output_alpha > 0.0 {
                premultiplied / output_alpha
            } else {
                0.0
            }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_sets_color_and_depth() {
        let mut framebuffer = Framebuffer::new(2, 2);
        framebuffer.depth[1] = 0.25;
        framebuffer.clear(argb8888(255, 10, 20, 30));
        assert_eq!(framebuffer.color, vec![0xff0a141e; 4]);
        assert_eq!(framebuffer.depth, vec![1.0; 4]);
    }

    #[test]
    fn resize_clears_modified_buffers() {
        let mut framebuffer = Framebuffer::new(2, 2);
        framebuffer.color.fill(0xdead_beef);
        framebuffer.depth.fill(0.25);
        framebuffer.resize(3, 1);
        assert_eq!((framebuffer.width, framebuffer.height), (3, 1));
        assert_eq!(framebuffer.color, vec![0; 3]);
        assert_eq!(framebuffer.depth, vec![1.0; 3]);
    }

    #[test]
    fn resize_clears_equal_area_shape_change() {
        let mut framebuffer = Framebuffer::new(4, 2);
        framebuffer.color.fill(0xdead_beef);
        framebuffer.depth.fill(0.25);
        framebuffer.resize(2, 4);
        assert_eq!((framebuffer.width, framebuffer.height), (2, 4));
        assert_eq!(framebuffer.color, vec![0; 8]);
        assert_eq!(framebuffer.depth, vec![1.0; 8]);
    }

    #[test]
    fn put_pixel_and_argb_packing() {
        let mut framebuffer = Framebuffer::new(2, 2);
        let color = argb8888(0x80, 0x10, 0x20, 0x40);
        framebuffer.put_pixel(1, 0, color);
        framebuffer.put_pixel(2, 0, 0xffffffff);
        assert_eq!(color, 0x80102040);
        assert_eq!(framebuffer.color, vec![0, color, 0, 0]);
    }

    #[test]
    fn put_pixel_ignores_inconsistent_buffer() {
        let mut framebuffer = Framebuffer::new(2, 2);
        framebuffer.color.clear();
        framebuffer.put_pixel(1, 1, 0xffffffff);
    }

    #[test]
    fn alpha_blending_decodes_and_reencodes_rgb() {
        let result =
            blend_argb8888_linear(argb8888(255, 128, 128, 128), argb8888(128, 255, 255, 255));
        assert_eq!(result, argb8888(255, 205, 205, 205));
        assert_ne!(result, argb8888(255, 192, 192, 192));
    }

    #[test]
    fn downsample_uses_linear_box_filter() {
        let mut source = Framebuffer::new(2, 2);
        source.color = vec![
            argb8888(255, 0, 0, 0),
            argb8888(255, 255, 255, 255),
            argb8888(255, 0, 0, 0),
            argb8888(255, 255, 255, 255),
        ];
        let mut destination = Framebuffer::new(1, 1);
        source.downsample_linear_into(&mut destination);
        assert_eq!(destination.color[0], argb8888(255, 188, 188, 188));
        assert_ne!(destination.color[0], argb8888(255, 128, 128, 128));
    }
}
