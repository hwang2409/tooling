use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::postfx::SsaoPass;
use std::fs;
use std::path::PathBuf;

const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const FOV_Y: f32 = 1.0;
const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;
const NEAR: f32 = 0.1;
const FAR: f32 = 100.0;

fn projection() -> Mat4 {
    Mat4::perspective(FOV_Y, ASPECT, NEAR, FAR)
}

fn depth_for_view_position(projection: Mat4, position: Vec3) -> f32 {
    let clip = projection * Vec4::new(position.x, position.y, position.z, 1.0);
    clip.z / clip.w
}

fn depth_for_view_z(projection: Mat4, z: f32) -> f32 {
    depth_for_view_position(projection, Vec3::new(0.0, 0.0, z))
}

fn plane_depth(z: f32) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.color.fill(argb8888(255, 180, 180, 180));
    framebuffer.depth.fill(depth_for_view_z(projection(), z));
    framebuffer
}

fn ssao_frame(pass: &SsaoPass, framebuffer: &mut Framebuffer) {
    pass.apply_to_framebuffer(framebuffer);
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join(format!("{name}.ppm"))
}

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

fn assert_golden(name: &str, framebuffer: &Framebuffer) {
    let path = golden_path(name);
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).expect("write SSAO golden");
    }
    let expected = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing golden {}: {error}; run GOLDEN_REGEN=1 cargo test --test ssao",
            path.display()
        )
    });
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

#[test]
fn depth_reconstruction_round_trip_uses_projection_mapping() {
    // Mutation gate: replacing inverse-projection reconstruction with linear
    // view-z mapping fails the x/y round trip assertion.
    let projection = projection();
    let pass = SsaoPass::new(projection);
    let x = 23;
    let y = 17;
    let z = -3.5;
    let inverse_projection = projection.inverse().expect("projection is invertible");
    let ndc_x = ((x as f32 + 0.5) / WIDTH as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((y as f32 + 0.5) / HEIGHT as f32) * 2.0;
    let ray = inverse_projection * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let ray = Vec3::new(ray.x / ray.w, ray.y / ray.w, ray.z / ray.w);
    let point = ray * (z / ray.z);
    let depth = depth_for_view_position(projection, point);
    let reconstructed = pass
        .reconstruct_view_position(x, y, depth, WIDTH, HEIGHT)
        .expect("valid depth reconstructs");
    assert!((reconstructed.x - point.x).abs() < 1e-5);
    assert!((reconstructed.y - point.y).abs() < 1e-5);
    assert!((reconstructed.z - point.z).abs() < 1e-5);
}

#[test]
fn flat_plane_has_no_interior_self_occlusion() {
    // Mutation gate: removing the bias makes the precision-jitter samples
    // self-occlude and fails the interior assertion.
    let mut framebuffer = plane_depth(-3.0);
    let mut pass = SsaoPass::new(projection());
    let near_depth = depth_for_view_z(projection(), -2.96);
    let base_depth = depth_for_view_z(projection(), -3.0);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            framebuffer.depth[y * WIDTH + x] = if (x + y) % 2 == 0 {
                near_depth
            } else {
                base_depth
            };
        }
    }
    pass.set_radius(0.2);
    pass.set_bias(0.2);
    let occlusion = pass.occlusion_buffer(&framebuffer);
    assert!(occlusion[20 * WIDTH + 20] < 1e-6);
    assert!(occlusion[24 * WIDTH + 40] < 1e-6);
    ssao_frame(&pass, &mut framebuffer);
    assert_eq!(
        framebuffer.color[20 * WIDTH + 20],
        argb8888(255, 180, 180, 180)
    );
}

#[test]
fn crease_has_more_occlusion_than_open_plane() {
    // Mutation gate: sampling a full sphere instead of the oriented
    // hemisphere reverses this contrast and fails the assertion.
    let projection = projection();
    let inverse_projection = projection.inverse().expect("projection is invertible");
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.color.fill(argb8888(255, 220, 220, 220));
    let floor_y = -0.65;
    let wall_x = 0.65;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let ndc_x = ((x as f32 + 0.5) / WIDTH as f32) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((y as f32 + 0.5) / HEIGHT as f32) * 2.0;
            let near = inverse_projection * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
            let far = inverse_projection * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
            let near = Vec3::new(near.x / near.w, near.y / near.w, near.z / near.w);
            let far = Vec3::new(far.x / far.w, far.y / far.w, far.z / far.w);
            let ray = (far - near).normalize();
            let mut hit = None;
            if ray.y < 0.0 {
                let t = floor_y / ray.y;
                let point = ray * t;
                if t > 0.0 && point.x < wall_x {
                    hit = Some(point);
                }
            }
            if ray.x > 0.0 {
                let t = wall_x / ray.x;
                let point = ray * t;
                if t > 0.0 && point.y > floor_y && hit.is_none_or(|floor| point.z > floor.z) {
                    hit = Some(point);
                }
            }
            if let Some(point) = hit {
                framebuffer.depth[y * WIDTH + x] = depth_for_view_position(projection, point);
            }
        }
    }
    let pass = SsaoPass::new(projection);
    let occlusion = pass.occlusion_buffer(&framebuffer);
    let open_floor = occlusion[36 * WIDTH + 20];
    let crease = occlusion[24 * WIDTH + 32];
    assert!(
        crease > open_floor + 0.02,
        "crease={crease}, floor={open_floor}"
    );
}

#[test]
fn range_check_rejects_far_background_halo() {
    // Mutation gate: removing the range test makes the near quad halo the
    // adjacent far-background pixels.
    let projection = projection();
    let mut framebuffer = plane_depth(-8.0);
    let near_depth = depth_for_view_z(projection, -2.0);
    for y in 14..34 {
        for x in 24..40 {
            framebuffer.depth[y * WIDTH + x] = near_depth;
        }
    }
    let mut pass = SsaoPass::new(projection);
    pass.set_range(0.5);
    let occlusion = pass.occlusion_buffer(&framebuffer);
    assert!(occlusion[24 * WIDTH + 23] < 1e-6);
    assert!(occlusion[24 * WIDTH + 40] < 1e-6);
}

#[test]
fn occlusion_is_deterministic_and_strength_zero_is_off() {
    let framebuffer = plane_depth(-3.0);
    let pass = SsaoPass::new(projection());
    assert_eq!(
        pass.occlusion_buffer(&framebuffer),
        pass.occlusion_buffer(&framebuffer)
    );

    let mut off = framebuffer.clone();
    let mut off_pass = SsaoPass::new(projection());
    off_pass.set_strength(0.0);
    ssao_frame(&off_pass, &mut off);
    assert_eq!(off.color, framebuffer.color);
}

#[test]
fn setters_sanitize_values_and_rebuild_projection_cache() {
    let projection = projection();
    let mut pass = SsaoPass::new(Mat4::IDENTITY);
    pass.set_projection(projection);
    assert_eq!(pass.projection(), projection);
    pass.set_radius(f32::NAN);
    assert_eq!(pass.radius(), chimy2::postfx::SSAO_DEFAULT_RADIUS);
    pass.set_bias(f32::INFINITY);
    assert_eq!(pass.bias(), 1000.0);
    pass.set_strength(-1.0);
    assert_eq!(pass.strength(), 0.0);
    pass.set_range(f32::NEG_INFINITY);
    assert_eq!(pass.range(), chimy2::postfx::SSAO_DEFAULT_RANGE);
    pass.set_blur_depth_threshold(0.25);
    assert_eq!(pass.blur_depth_threshold(), 0.25);

    let x = 20;
    let y = 15;
    let inverse = projection.inverse().expect("projection is invertible");
    let ndc_x = ((x as f32 + 0.5) / WIDTH as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((y as f32 + 0.5) / HEIGHT as f32) * 2.0;
    let ray = inverse * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let ray = Vec3::new(ray.x / ray.w, ray.y / ray.w, ray.z / ray.w);
    let point = ray * (-2.0 / ray.z);
    let depth = depth_for_view_position(projection, point);
    let reconstructed = pass
        .reconstruct_view_position(x, y, depth, WIDTH, HEIGHT)
        .expect("valid depth reconstructs after setter");
    assert!((reconstructed.z - point.z).abs() < 1e-5);
}

#[test]
fn ssao_scene_golden() {
    let projection = projection();
    let mut framebuffer = plane_depth(-3.0);
    let near_depth = depth_for_view_z(projection, -2.0);
    for y in 16..32 {
        for x in 25..39 {
            framebuffer.depth[y * WIDTH + x] = near_depth;
            framebuffer.color[y * WIDTH + x] = argb8888(255, 230, 120, 60);
        }
    }
    let pass = SsaoPass::new(projection);
    ssao_frame(&pass, &mut framebuffer);
    assert_golden("m19-postfx-ssao", &framebuffer);
}
