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
}
