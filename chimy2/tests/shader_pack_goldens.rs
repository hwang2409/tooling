use chimy2::camera::OrbitController;
use chimy2::demo::uv_sphere;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{
    DitherShader, DitherUniforms, FogShader, FogUniforms, NormalsShader, NormalsUniforms,
    PsxShader, PsxUniforms, ToonShader, ToonUniforms, WireframeShader, WireframeUniforms,
    expand_mesh_with_barycentrics,
};
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join(format!("{name}.ppm"))
}

fn assert_golden(name: &str, framebuffer: &Framebuffer) {
    let path = golden_path(name);
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).expect("write golden");
        panic!(
            "regenerated {}, rerun without GOLDEN_REGEN to compare",
            path.display()
        );
    }
    let expected = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing golden {}: {error}; run GOLDEN_REGEN=1 cargo test --test shader_pack_goldens",
            path.display()
        )
    });
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

fn scene() -> (
    Vec<chimy2::shaders::ShaderPackVertex>,
    Vec<[usize; 3]>,
    Mat4,
    Mat4,
    Mat4,
) {
    let mesh = uv_sphere(1.35, 12, 24);
    let (vertices, triangles) = expand_mesh_with_barycentrics(&mesh);
    let orbit = OrbitController::new(Vec3::ZERO, 4.4, 0.42, 0.24);
    let camera = orbit.camera(1.05, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);
    (
        vertices,
        triangles,
        Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.35),
        camera.view_matrix(),
        camera.projection_matrix(),
    )
}

fn framebuffer() -> Framebuffer {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 8, 10, 16));
    framebuffer
}

#[test]
fn toon_shader_scene_golden() {
    let (vertices, triangles, model, view, projection) = scene();
    let uniforms = ToonUniforms::new(
        model,
        view,
        projection,
        Vec3::new(0.84, 0.30, 0.10),
        Vec3::new(-0.4, 0.8, 1.0),
    );
    let mut framebuffer = framebuffer();
    let mut pipeline = Pipeline::new(ToonShader, ToonShader);
    pipeline.set_thread_count(1);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw(target, &vertices, &triangles, &uniforms);
    });
    assert_golden("m7-shader-toon", &framebuffer);
}

#[test]
fn psx_shader_scene_golden() {
    let (vertices, triangles, model, view, projection) = scene();
    let uniforms = PsxUniforms::new(
        model,
        view,
        projection,
        Vec3::new(0.18, 0.66, 0.92),
        (WIDTH as u32, HEIGHT as u32),
    );
    let mut framebuffer = framebuffer();
    let mut pipeline = Pipeline::new(PsxShader, PsxShader);
    pipeline.set_thread_count(1);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw(target, &vertices, &triangles, &uniforms);
    });
    assert_golden("m7-shader-psx", &framebuffer);
}

#[test]
fn dither_shader_scene_golden() {
    let (vertices, triangles, model, view, projection) = scene();
    let uniforms = DitherUniforms::new(
        model,
        view,
        projection,
        Vec3::new(0.95, 0.62, 0.16),
        (WIDTH as u32, HEIGHT as u32),
    );
    let mut framebuffer = framebuffer();
    let mut pipeline = Pipeline::new(DitherShader, DitherShader);
    pipeline.set_thread_count(1);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw(target, &vertices, &triangles, &uniforms);
    });
    assert_golden("m7-shader-dither", &framebuffer);
}

#[test]
fn fog_shader_scene_golden() {
    let (vertices, triangles, model, view, projection) = scene();
    let uniforms = FogUniforms::new(
        model,
        view,
        projection,
        Vec3::new(0.18, 0.72, 0.34),
        Vec3::new(0.12, 0.16, 0.28),
        3.0,
        6.0,
    );
    let mut framebuffer = framebuffer();
    let mut pipeline = Pipeline::new(FogShader, FogShader);
    pipeline.set_thread_count(1);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw(target, &vertices, &triangles, &uniforms);
    });
    assert_golden("m7-shader-fog", &framebuffer);
}

#[test]
fn normals_shader_scene_golden() {
    let (vertices, triangles, model, view, projection) = scene();
    let uniforms = NormalsUniforms::new(model, view, projection);
    let mut framebuffer = framebuffer();
    let mut pipeline = Pipeline::new(NormalsShader, NormalsShader);
    pipeline.set_thread_count(1);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw(target, &vertices, &triangles, &uniforms);
    });
    assert_golden("m7-shader-normals", &framebuffer);
}

#[test]
fn wireframe_shader_scene_golden() {
    let (vertices, triangles, model, view, projection) = scene();
    let uniforms = WireframeUniforms::new(
        model,
        view,
        projection,
        Vec3::new(0.08, 0.10, 0.16),
        Vec3::new(0.95, 0.72, 0.12),
    );
    let mut framebuffer = framebuffer();
    let mut pipeline = Pipeline::new(WireframeShader, WireframeShader);
    pipeline.set_thread_count(1);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw(target, &vertices, &triangles, &uniforms);
    });
    assert_golden("m7-shader-wireframe", &framebuffer);
}
