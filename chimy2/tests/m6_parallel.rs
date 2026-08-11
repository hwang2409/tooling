use chimy2::camera::Camera;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::image::{Texture, WrapMode};
use chimy2::math::{Mat4, Quat, Vec3, Vec4};
use chimy2::mesh::Mesh;
use chimy2::pipeline::{ColorVarying, Pipeline, VertexOutput, vertex_stage};
use chimy2::raster::set_simd_for_tests;
use chimy2::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, FlatColorShader, FlatColorUniforms,
    MeshShader, MeshUniforms, PointLight, TextureFilter, TexturedBlinnPhongShader,
    TexturedBlinnPhongUniforms, TexturedShader, TexturedUniforms,
};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

const FLAT_WIDTH: usize = 8;
const FLAT_HEIGHT: usize = 8;
const MESH_WIDTH: usize = 64;
const MESH_HEIGHT: usize = 48;
type Scene = (&'static str, fn(usize) -> Framebuffer);

fn assert_identical(name: &str, serial: Framebuffer, parallel: Framebuffer) {
    assert_eq!(serial, parallel, "serial and parallel differ for {name}");
}

fn render_with_simd_mode(
    render: fn(usize) -> Framebuffer,
    threads: usize,
    enabled: bool,
) -> Framebuffer {
    set_simd_for_tests(Some(enabled));
    let framebuffer = render(threads);
    set_simd_for_tests(None);
    framebuffer
}

fn simd_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn draw_flat(
    thread_count: usize,
    framebuffer: &mut Framebuffer,
    vertices: &[Vec4],
    triangles: &[[usize; 3]],
    color: u32,
    z: f32,
) {
    let transformed: Vec<_> = vertices
        .iter()
        .map(|&vertex| Vec4::new(vertex.x, vertex.y, z, vertex.w))
        .collect();
    let mut pipeline = Pipeline::new(FlatColorShader, FlatColorShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw(
        framebuffer,
        &transformed,
        triangles,
        &FlatColorUniforms::new(Mat4::IDENTITY, color),
    );
}

fn flat_single(thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(FLAT_WIDTH, FLAT_HEIGHT);
    draw_flat(
        thread_count,
        &mut framebuffer,
        &[
            Vec4::new(-0.75, -0.7, 0.0, 1.0),
            Vec4::new(0.75, -0.7, 0.0, 1.0),
            Vec4::new(0.0, 0.75, 0.0, 1.0),
        ],
        &[[0, 1, 2]],
        argb8888(255, 225, 60, 60),
        0.0,
    );
    framebuffer
}

fn flat_overlapping(thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(FLAT_WIDTH, FLAT_HEIGHT);
    let vertices = [
        Vec4::new(-0.85, -0.8, 0.0, 1.0),
        Vec4::new(0.85, -0.8, 0.0, 1.0),
        Vec4::new(0.0, 0.85, 0.0, 1.0),
    ];
    draw_flat(
        thread_count,
        &mut framebuffer,
        &vertices,
        &[[0, 1, 2]],
        argb8888(255, 235, 70, 70),
        -0.5,
    );
    draw_flat(
        thread_count,
        &mut framebuffer,
        &vertices,
        &[[0, 1, 2]],
        argb8888(255, 70, 90, 235),
        0.5,
    );
    framebuffer
}

fn flat_shared_edge(thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(FLAT_WIDTH, FLAT_HEIGHT);
    let vertices = [
        Vec4::new(-1.0, 1.0, 0.0, 1.0),
        Vec4::new(1.0, 1.0, 0.0, 1.0),
        Vec4::new(-1.0, -1.0, 0.0, 1.0),
        Vec4::new(1.0, 1.0, -0.1, 1.0),
        Vec4::new(1.0, -1.0, -0.1, 1.0),
        Vec4::new(-1.0, -1.0, -0.1, 1.0),
    ];
    draw_flat(
        thread_count,
        &mut framebuffer,
        &vertices,
        &[[0, 1, 2]],
        argb8888(255, 235, 70, 70),
        0.0,
    );
    draw_flat(
        thread_count,
        &mut framebuffer,
        &vertices,
        &[[3, 4, 5]],
        argb8888(255, 70, 90, 235),
        -0.1,
    );
    framebuffer
}

fn mesh_framebuffer() -> Framebuffer {
    let mut framebuffer = Framebuffer::new(MESH_WIDTH, MESH_HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    framebuffer
}

fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.0, 6.0),
        Quat::IDENTITY,
        1.0,
        MESH_WIDTH as f32 / MESH_HEIGHT as f32,
        0.1,
        100.0,
    )
}

fn m3_cube(thread_count: usize) -> Framebuffer {
    let mesh = Mesh::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cube.obj")).unwrap();
    let camera = camera();
    let mut framebuffer = mesh_framebuffer();
    let uniforms = MeshUniforms::new(
        Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.35),
        camera.view_matrix(),
        camera.projection_matrix(),
        argb8888(255, 225, 160, 70),
    );
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh(&mut framebuffer, &mesh, &uniforms);
    framebuffer
}

fn m3_near_plane(thread_count: usize) -> Framebuffer {
    let mesh = Mesh::parse("v -0.8 -0.7 -0.25\nv 0.8 -0.7 -0.7\nv 0 0.8 -0.7\nf 1 2 3\n").unwrap();
    let camera = Camera::new(
        Vec3::ZERO,
        Quat::IDENTITY,
        1.0,
        MESH_WIDTH as f32 / MESH_HEIGHT as f32,
        0.5,
        10.0,
    );
    let mut framebuffer = mesh_framebuffer();
    let uniforms = MeshUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        argb8888(255, 80, 200, 120),
    );
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh(&mut framebuffer, &mesh, &uniforms);
    framebuffer
}

fn m3_backface(thread_count: usize) -> Framebuffer {
    let front = Mesh::parse("v -0.8 -0.7 0\nv 0.8 -0.7 0\nv 0 0.8 0\nf 1 2 3\n").unwrap();
    let back = Mesh::parse("v -0.8 -0.7 -0.1\nv 0 0.8 -0.1\nv 0.8 -0.7 -0.1\nf 1 2 3\n").unwrap();
    let mut framebuffer = mesh_framebuffer();
    let mut pipeline = Pipeline::new(MeshShader, MeshShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh(
        &mut framebuffer,
        &front,
        &MeshUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            argb8888(255, 225, 60, 60),
        ),
    );
    pipeline.draw_mesh(
        &mut framebuffer,
        &back,
        &MeshUniforms::new(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            argb8888(255, 60, 80, 225),
        ),
    );
    framebuffer
}

fn lighting_uniforms(
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

fn draw_blinn(
    thread_count: usize,
    framebuffer: &mut Framebuffer,
    mesh: &Mesh,
    uniforms: BlinnPhongUniforms,
) {
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh(framebuffer, mesh, &uniforms);
}

fn m4_icosahedron(thread_count: usize) -> Framebuffer {
    let mesh =
        Mesh::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icosahedron.obj")).unwrap();
    let camera = camera();
    let directional = DirectionalLight::new(
        Vec3::new(0.4, 0.7, 1.0).normalize(),
        Vec3::new(1.0, 0.95, 0.85),
    );
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let mut framebuffer = mesh_framebuffer();
    draw_blinn(
        thread_count,
        &mut framebuffer,
        &mesh,
        lighting_uniforms(camera, Mat4::IDENTITY, directional, point),
    );
    framebuffer
}

fn m4_point_light(thread_count: usize) -> Framebuffer {
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
    let mut framebuffer = mesh_framebuffer();
    draw_blinn(
        thread_count,
        &mut framebuffer,
        &mesh,
        lighting_uniforms(
            camera,
            Mat4::translate(Vec3::new(-1.2, 0.0, 0.0)),
            directional,
            point,
        ),
    );
    draw_blinn(
        thread_count,
        &mut framebuffer,
        &mesh,
        lighting_uniforms(
            camera,
            Mat4::translate(Vec3::new(1.2, 0.0, -2.0)),
            directional,
            point,
        ),
    );
    framebuffer
}

fn m4_non_uniform(thread_count: usize) -> Framebuffer {
    let mesh = Mesh::parse("v -1 -1 0\nv 1 -1 1\nv 0 1 0\nf 1 2 3\n").unwrap();
    let camera = camera();
    let directional = DirectionalLight::new(
        Vec3::new(0.0, -0.8, 1.0).normalize(),
        Vec3::new(1.0, 1.0, 1.0),
    );
    let point = PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0);
    let mut framebuffer = mesh_framebuffer();
    draw_blinn(
        thread_count,
        &mut framebuffer,
        &mesh,
        lighting_uniforms(
            camera,
            Mat4::translate(Vec3::new(0.0, 0.0, -1.0)) * Mat4::scale(Vec3::new(2.0, 1.0, 0.35)),
            directional,
            point,
        ),
    );
    framebuffer
}

#[derive(Clone, Copy)]
struct PerspectiveVertex {
    clip_position: Vec4,
    color: Vec4,
}

fn m4_perspective(thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(MESH_WIDTH, MESH_HEIGHT);
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
    pipeline.set_thread_count(thread_count);
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
    framebuffer
}

fn textured_background() -> Framebuffer {
    let mut framebuffer = Framebuffer::new(MESH_WIDTH, MESH_HEIGHT);
    framebuffer.clear(argb8888(255, 12, 16, 24));
    framebuffer
}

fn textured_quad() -> Mesh {
    Mesh::parse(
        "v -1.5 -1 0\nv 1.5 -1 -2\nv 1.5 1 -2\nv -1.5 1 0\n\
         vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
         f 1/1 2/2 3/3 4/4\n",
    )
    .unwrap()
}

fn m5_perspective(thread_count: usize) -> Framebuffer {
    let texture = Texture::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/checker.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        1.0,
        MESH_WIDTH as f32 / MESH_HEIGHT as f32,
        0.1,
        100.0,
    );
    let mut framebuffer = textured_background();
    let mut pipeline = Pipeline::new(TexturedShader, TexturedShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh_with_sampling(
        &mut framebuffer,
        &textured_quad(),
        &TexturedUniforms::new(camera.view_projection(), &texture, TextureFilter::Nearest),
    );
    framebuffer
}

fn m5_gradient(thread_count: usize, filter: TextureFilter) -> Framebuffer {
    let texture = Texture::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/gradient.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::ClampToEdge);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        1.0,
        MESH_WIDTH as f32 / MESH_HEIGHT as f32,
        0.1,
        100.0,
    );
    let mut framebuffer = textured_background();
    let mut pipeline = Pipeline::new(TexturedShader, TexturedShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh_with_sampling(
        &mut framebuffer,
        &textured_quad(),
        &TexturedUniforms::new(camera.view_projection(), &texture, filter),
    );
    framebuffer
}

fn m5_nearest_gradient(thread_count: usize) -> Framebuffer {
    m5_gradient(thread_count, TextureFilter::Nearest)
}

fn m5_bilinear_gradient(thread_count: usize) -> Framebuffer {
    m5_gradient(thread_count, TextureFilter::Bilinear)
}

fn m5_lit(thread_count: usize) -> Framebuffer {
    let texture = Texture::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/checker.qoi"))
        .unwrap()
        .with_wrap_mode(WrapMode::Repeat);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        1.0,
        MESH_WIDTH as f32 / MESH_HEIGHT as f32,
        0.1,
        100.0,
    );
    let lighting = BlinnPhongUniforms::new(
        Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), 0.25),
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.04, 0.04, 0.04),
        Vec3::new(0.9, 0.9, 0.9),
        Vec3::new(0.7, 0.7, 0.7),
        24.0,
        camera.position,
        DirectionalLight::new(
            Vec3::new(-0.4, 0.7, 1.0).normalize(),
            Vec3::new(1.0, 0.95, 0.9),
        ),
        PointLight::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 0.5, 0.3),
            1.0,
            0.08,
            0.02,
        ),
    );
    let mut framebuffer = textured_background();
    let mut pipeline = Pipeline::new(TexturedBlinnPhongShader, TexturedBlinnPhongShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh_with_sampling(
        &mut framebuffer,
        &textured_quad(),
        &TexturedBlinnPhongUniforms::new(lighting, &texture, TextureFilter::Bilinear),
    );
    framebuffer
}

#[derive(Clone, Copy)]
struct TileVertex {
    clip_position: Vec4,
    color: Vec4,
}

fn tile_boundary_scene(thread_count: usize) -> Framebuffer {
    let width = 130;
    let height = 130;
    let mut framebuffer = Framebuffer::new(width, height);
    let mut pipeline = Pipeline::new(
        vertex_stage(|vertex: &TileVertex, _: &()| {
            VertexOutput::new(vertex.clip_position, ColorVarying::new(vertex.color))
        }),
        chimy2::pipeline::fragment_stage(|varyings: &ColorVarying, _: &()| {
            let color = varyings.color;
            argb8888(
                255,
                (color.x * 255.0).round() as u8,
                (color.y * 255.0).round() as u8,
                (color.z * 255.0).round() as u8,
            )
        }),
    );
    pipeline.set_thread_count(thread_count);
    pipeline.draw(
        &mut framebuffer,
        &[
            TileVertex {
                clip_position: Vec4::new(-1.0, -1.0, 0.0, 1.0),
                color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(1.0, -1.0, 0.0, 1.0),
                color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(-1.0, 1.0, 0.0, 1.0),
                color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(-1.0, -1.0, 0.0, 1.0),
                color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(1.0, -1.0, 0.0, 1.0),
                color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(-1.0, 1.0, 0.0, 1.0),
                color: Vec4::new(0.0, 0.0, 1.0, 1.0),
            },
        ],
        &[[0, 1, 2], [3, 4, 5]],
        &(),
    );
    framebuffer
}

fn nonlinear_fragment_scene(thread_count: usize) -> Framebuffer {
    let mut framebuffer = Framebuffer::new(97, 73);
    let mut pipeline = Pipeline::new(
        vertex_stage(|vertex: &TileVertex, _: &()| {
            VertexOutput::new(vertex.clip_position, ColorVarying::new(vertex.color))
        }),
        chimy2::pipeline::fragment_stage(|varyings: &ColorVarying, _: &()| {
            let color = varyings.color;
            let red = ((color.x * 17.0 + color.y * color.y * 31.0 + color.z.sin())
                .abs()
                .fract()
                * 255.0)
                .round() as u8;
            let green = ((color.y * 13.0 + color.z * color.z * 29.0 + color.x.cos())
                .abs()
                .fract()
                * 255.0)
                .round() as u8;
            let blue = ((color.z * 11.0 + color.x * color.x * 23.0 + color.y.tan())
                .abs()
                .fract()
                * 255.0)
                .round() as u8;
            argb8888(255, red, green, blue)
        }),
    );
    pipeline.set_thread_count(thread_count);
    pipeline.draw(
        &mut framebuffer,
        &[
            TileVertex {
                clip_position: Vec4::new(-1.0, -1.0, 0.0, 1.0),
                color: Vec4::new(0.13, 0.71, 0.29, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(1.0, -1.0, 0.0, 1.0),
                color: Vec4::new(0.83, 0.17, 0.61, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(-1.0, 1.0, 0.0, 1.0),
                color: Vec4::new(0.47, 0.43, 0.97, 1.0),
            },
            TileVertex {
                clip_position: Vec4::new(1.0, 1.0, -0.1, 1.0),
                color: Vec4::new(0.31, 0.89, 0.07, 1.0),
            },
        ],
        &[[0, 1, 2], [1, 3, 2]],
        &(),
    );
    framebuffer
}

#[test]
fn every_golden_scene_is_byte_identical_with_multiple_thread_counts() {
    let _lock = simd_test_lock();
    let scenes: &[Scene] = &[
        ("single-triangle", flat_single),
        ("shared-edge", flat_shared_edge),
        ("overlapping-triangles-depth", flat_overlapping),
        ("m3-cube-camera", m3_cube),
        ("m3-near-plane-clipping", m3_near_plane),
        ("m3-backface-culling", m3_backface),
        ("m4-perspective-correctness", m4_perspective),
        ("m4-icosahedron-directional", m4_icosahedron),
        ("m4-point-light-falloff", m4_point_light),
        ("m4-non-uniform-normal-matrix", m4_non_uniform),
        ("m5-perspective-textured-quad", m5_perspective),
        ("m5-nearest-textured-quad", m5_nearest_gradient),
        ("m5-bilinear-textured-quad", m5_bilinear_gradient),
        ("m5-textured-lit-mesh", m5_lit),
    ];
    let thread_counts = [1, 2, 3, 15];
    for &(name, render) in scenes {
        let scalar = render_with_simd_mode(render, 1, false);
        let simd_serial = render_with_simd_mode(render, 1, true);
        assert_identical(name, scalar.clone(), simd_serial);
        for &thread_count in &thread_counts {
            assert_identical(
                name,
                scalar.clone(),
                render_with_simd_mode(render, thread_count, true),
            );
        }
    }
    let serial = render_with_simd_mode(tile_boundary_scene, 1, false);
    assert_identical(
        "tile-boundary-overlap",
        serial.clone(),
        render_with_simd_mode(tile_boundary_scene, 1, true),
    );
    for &thread_count in &thread_counts {
        assert_identical(
            "tile-boundary-overlap",
            serial.clone(),
            render_with_simd_mode(tile_boundary_scene, thread_count, true),
        );
    }
}

#[test]
fn nonlinear_fragment_is_byte_identical_with_multiple_thread_counts() {
    let _lock = simd_test_lock();
    let serial = render_with_simd_mode(nonlinear_fragment_scene, 1, false);
    for thread_count in [1, 2, 3, 15] {
        assert_identical(
            "nonlinear-fragment",
            serial.clone(),
            render_with_simd_mode(nonlinear_fragment_scene, thread_count, true),
        );
    }
}
