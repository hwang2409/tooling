use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::postfx::{
    DOF_DEFAULT_APERTURE, DOF_DEFAULT_FOCUS_DISTANCE, DOF_FOCAL_LENGTH, DofPass, PostChain,
    reconstruct_view_position,
};
use std::fs;
use std::path::PathBuf;

const WIDTH: usize = 32;
const HEIGHT: usize = 24;
const FOV_Y: f32 = 1.0;
const ASPECT: f32 = WIDTH as f32 / HEIGHT as f32;
const NEAR: f32 = 0.1;
const FAR: f32 = 100.0;

fn projection() -> Mat4 {
    Mat4::perspective(FOV_Y, ASPECT, NEAR, FAR)
}

fn depth_for_view_z(z: f32) -> f32 {
    let projection = projection();
    let clip = projection * Vec4::new(0.0, 0.0, z, 1.0);
    clip.z / clip.w
}

fn plane(depth: f32) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.depth.fill(depth_for_view_z(-depth));
    for (index, pixel) in framebuffer.color.iter_mut().enumerate() {
        *pixel = argb8888(
            255,
            (index * 17 % 251) as u8,
            (index * 31 % 251) as u8,
            (index * 47 % 251) as u8,
        );
    }
    framebuffer
}

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join(format!("{name}.ppm"))
}

fn assert_golden(name: &str, framebuffer: &Framebuffer) {
    let path = golden_path(name);
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).expect("write DoF golden");
        panic!(
            "regenerated golden {}; rerun without GOLDEN_REGEN",
            path.display()
        );
    }
    let expected = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing golden {}: {error}; run GOLDEN_REGEN=1 cargo test --test dof",
            path.display()
        )
    });
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

#[test]
fn coc_formula_matches_hand_computed_near_focus_and_far_samples() {
    let mut pass = DofPass::new(projection());
    pass.set_aperture(DOF_DEFAULT_APERTURE);
    pass.set_focus_distance(DOF_DEFAULT_FOCUS_DISTANCE);
    pass.set_max_coc_radius(64.0);
    let expected = |depth: f32| {
        DOF_DEFAULT_APERTURE * (DOF_FOCAL_LENGTH * (DOF_DEFAULT_FOCUS_DISTANCE - depth)).abs()
            / (depth * (DOF_DEFAULT_FOCUS_DISTANCE - DOF_FOCAL_LENGTH))
    };
    assert!((pass.circle_of_confusion(2.0) - expected(2.0)).abs() < 1e-6);
    assert_eq!(pass.circle_of_confusion(4.0), 0.0);
    assert!((pass.circle_of_confusion(8.0) - expected(8.0)).abs() < 1e-6);
}

#[test]
fn focal_plane_is_byte_identical_through_the_post_chain() {
    let source = plane(4.0);
    let mut enabled = source.clone();
    let pass = DofPass::new(projection());
    PostChain::new().with_pass(pass).apply(&mut enabled);
    assert_eq!(enabled.color, source.color);
}

#[test]
fn strength_zero_is_byte_identical_through_the_post_chain() {
    let source = plane(2.0);
    let mut enabled = source.clone();
    let mut pass = DofPass::new(projection());
    pass.set_aperture(0.0);
    PostChain::new().with_pass(pass).apply(&mut enabled);
    assert_eq!(enabled, source);
}

#[test]
fn invalid_projection_is_byte_identical_through_the_post_chain() {
    let mut source = plane(2.0);
    source.color[0] = 0x8012_3456;
    source.depth.fill(-0.37);
    let before = source.clone();
    PostChain::new()
        .with_pass(DofPass::new(Mat4::new([0.0; 16])))
        .apply(&mut source);
    assert_eq!(source, before);
}

#[test]
fn setters_sanitize_immediately_and_rebuild_projection_cache() {
    let projection = projection();
    let mut pass = DofPass::new(Mat4::IDENTITY);
    pass.set_projection(projection);
    assert_eq!(pass.projection(), projection);
    pass.set_focus_distance(f32::NAN);
    assert_eq!(pass.focus_distance(), DOF_DEFAULT_FOCUS_DISTANCE);
    pass.set_focus_distance(-1.0);
    assert!(pass.focus_distance() > DOF_FOCAL_LENGTH);
    pass.set_focus_distance(0.0);
    assert!(pass.focus_distance() > DOF_FOCAL_LENGTH);
    pass.set_aperture(f32::NEG_INFINITY);
    assert_eq!(pass.aperture(), DOF_DEFAULT_APERTURE);
    pass.set_max_coc_radius(f32::INFINITY);
    assert_eq!(pass.max_coc_radius(), 64.0);

    let x = 12;
    let y = 9;
    let inverse = projection.inverse().expect("projection is invertible");
    let ndc_x = ((x as f32 + 0.5) / WIDTH as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((y as f32 + 0.5) / HEIGHT as f32) * 2.0;
    let ray = inverse * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let ray = Vec3::new(ray.x / ray.w, ray.y / ray.w, ray.z / ray.w);
    let point = ray * (-2.0 / ray.z);
    let clip = projection * Vec4::new(point.x, point.y, point.z, 1.0);
    let depth = clip.z / clip.w;
    let reconstructed = reconstruct_view_position(inverse, x, y, depth, WIDTH, HEIGHT)
        .expect("valid depth reconstructs after setter");
    assert!((reconstructed.z - point.z).abs() < 1e-5);
}

#[test]
fn depth_weighting_keeps_less_defocused_foreground_edge_sharp() {
    let mut framebuffer = plane(8.0);
    let foreground_depth = depth_for_view_z(-3.5);
    let center_x = WIDTH / 2;
    let center_y = HEIGHT / 2;
    for y in 0..HEIGHT {
        framebuffer.depth[y * WIDTH + center_x] = foreground_depth;
        framebuffer.color[y * WIDTH + center_x] = argb8888(255, 240, 20, 20);
    }
    framebuffer.color[center_y * WIDTH + center_x - 1] = argb8888(255, 240, 20, 20);
    let expected = framebuffer.color[center_y * WIDTH + center_x];
    let mut pass = DofPass::new(projection());
    pass.set_aperture(24.0);
    pass.set_focus_distance(4.0);
    pass.set_max_coc_radius(8.0);
    pass.apply_to_framebuffer(&mut framebuffer);
    assert_eq!(framebuffer.color[center_y * WIDTH + center_x], expected);
}

#[test]
fn edge_clamping_does_not_wrap_corner_taps() {
    let mut framebuffer = plane(2.0);
    framebuffer.color.fill(argb8888(255, 0, 0, 0));
    framebuffer.color[0] = argb8888(255, 255, 0, 0);
    framebuffer.color[WIDTH - 8] = argb8888(255, 0, 0, 255);
    let mut pass = DofPass::new(projection());
    pass.set_focus_distance(4.0);
    pass.set_aperture(24.0);
    pass.set_max_coc_radius(8.0);
    pass.apply_to_framebuffer(&mut framebuffer);
    let [_, red, _green, blue] = framebuffer.color[0].to_be_bytes();
    assert!(red > blue + 20, "corner wrapped blue: {red}, {blue}");
}

#[test]
fn dof_render_is_deterministic() {
    let source = plane(2.0);
    let pass = DofPass::new(projection());
    let mut first = source.clone();
    let mut second = source;
    pass.apply_to_framebuffer(&mut first);
    pass.apply_to_framebuffer(&mut second);
    assert_eq!(first, second);
}

#[test]
fn dof_scene_golden() {
    let mut framebuffer = plane(8.0);
    let foreground = depth_for_view_z(-2.0);
    let middle = depth_for_view_z(-4.0);
    for y in 7..17 {
        for x in 5..11 {
            let index = y * WIDTH + x;
            framebuffer.depth[index] = foreground;
            framebuffer.color[index] = argb8888(255, 240, 65, 25);
        }
        for x in 13..19 {
            let index = y * WIDTH + x;
            framebuffer.depth[index] = middle;
            framebuffer.color[index] = argb8888(255, 25, 220, 120);
        }
        for x in 21..27 {
            let index = y * WIDTH + x;
            framebuffer.color[index] = argb8888(255, 35, 90, 235);
        }
    }
    let mut pass = DofPass::new(projection());
    pass.set_focus_distance(4.0);
    pass.set_aperture(9.0);
    pass.set_max_coc_radius(5.0);
    pass.apply_to_framebuffer(&mut framebuffer);
    assert_golden("m33-postfx-dof", &framebuffer);
}
