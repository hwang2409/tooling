use chimy2::camera::Camera;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::math::{Mat4, Quat, Vec3, Vec4};
use chimy2::mesh::Mesh;
use chimy2::pipeline::{ColorVarying, Pipeline, VertexOutput, vertex_stage};
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
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
            "missing golden {}: {error}; run GOLDEN_REGEN=1 cargo test --test m4_goldens",
            path.display()
        )
    });
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 6.0),
        Quat::IDENTITY,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    )
}

fn lit_uniforms(
    camera: Camera,
    model: Mat4,
    directional_light: DirectionalLight,
    point_light: PointLight,
) -> BlinnPhongUniforms {
    BlinnPhongUniforms::new(
        model,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.02, 0.02, 0.02),
        Vec3::new(0.75, 0.42, 0.12),
        Vec3::new(0.8, 0.8, 0.8),
        24.0,
        camera.position,
        directional_light,
        point_light,
    )
}

fn draw_mesh(framebuffer: &mut Framebuffer, mesh: &Mesh, uniforms: BlinnPhongUniforms) {
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.draw_mesh(framebuffer, mesh, &uniforms);
}

#[derive(Clone, Copy)]
struct PerspectiveVertex {
    clip_position: Vec4,
    color: Vec4,
}

#[test]
fn perspective_correctness_scene_golden() {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let mut pipeline = Pipeline::new(
        vertex_stage(|vertex: &PerspectiveVertex, _: &()| {
            VertexOutput::new(vertex.clip_position, ColorVarying::new(vertex.color))
        }),
        chimy2::pipeline::fragment_stage(|varyings: &ColorVarying, _: &()| {
            let color = varyings.color;
            argb8888(
                255,
                (color.x.clamp(0.0, 1.0) * 255.0).round() as u8,
                (color.y.clamp(0.0, 1.0) * 255.0).round() as u8,
                (color.z.clamp(0.0, 1.0) * 255.0).round() as u8,
            )
        }),
    );
    pipeline.draw(
        &mut framebuffer,
        &[
            PerspectiveVertex {
                clip_position: Vec4::new(-0.9, -0.8, 0.0, 1.0),
                color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            },
            PerspectiveVertex {
                clip_position: Vec4::new(2.4, -2.4, 0.0, 3.0),
                color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            },
            PerspectiveVertex {
                clip_position: Vec4::new(-0.9, 0.9, 0.0, 1.0),
                color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            },
        ],
        &[[0, 1, 2]],
        &(),
    );
    assert_golden("m4-perspective-correctness", &framebuffer);
}

#[test]
fn icosahedron_directional_light_golden() {
    let mesh =
        Mesh::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icosahedron.obj")).unwrap();
    let camera = camera();
    let directional = DirectionalLight::new(
        Vec3::new(0.4, 0.7, 1.0).normalize(),
        Vec3::new(1.0, 0.95, 0.85),
    );
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    draw_mesh(
        &mut framebuffer,
        &mesh,
        lit_uniforms(camera, Mat4::IDENTITY, directional, point),
    );
    assert_golden("m4-icosahedron-directional", &framebuffer);
}

#[test]
fn point_light_falloff_scene_golden() {
    let mesh = Mesh::parse("v -0.8 -0.8 0\nv 0.8 -0.8 0\nv 0 0.8 0\nf 1 2 3\n").unwrap();
    let camera = camera();
    let directional = DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO);
    let point = PointLight::new(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::new(1.0, 0.75, 0.5),
        1.0,
        0.1,
        0.04,
    );
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    draw_mesh(
        &mut framebuffer,
        &mesh,
        lit_uniforms(
            camera,
            Mat4::translate(Vec3::new(-1.2, 0.0, 0.0)),
            directional,
            point,
        ),
    );
    draw_mesh(
        &mut framebuffer,
        &mesh,
        lit_uniforms(
            camera,
            Mat4::translate(Vec3::new(1.2, 0.0, -2.0)),
            directional,
            point,
        ),
    );
    assert_golden("m4-point-light-falloff", &framebuffer);
}

#[test]
fn non_uniform_scale_normal_matrix_scene_golden() {
    let mesh = Mesh::parse("v -1 -1 0\nv 1 -1 1\nv 0 1 0\nf 1 2 3\n").unwrap();
    let camera = camera();
    let directional = DirectionalLight::new(
        Vec3::new(0.0, -0.8, 1.0).normalize(),
        Vec3::new(1.0, 1.0, 1.0),
    );
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    draw_mesh(
        &mut framebuffer,
        &mesh,
        lit_uniforms(
            camera,
            Mat4::translate(Vec3::new(0.0, 0.0, -1.0)) * Mat4::scale(Vec3::new(2.0, 1.0, 0.35)),
            directional,
            point,
        ),
    );
    assert_golden("m4-non-uniform-normal-matrix", &framebuffer);
}
