use chimy2::camera::Camera;
use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
use chimy2::raster::set_simd_for_tests;
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::path::Path;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

fn uniforms() -> BlinnPhongUniforms {
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        1.0,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        100.0,
    );
    BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.02, 0.02, 0.02),
        Vec3::new(0.75, 0.42, 0.12),
        Vec3::new(0.8, 0.8, 0.8),
        24.0,
        camera.position,
        DirectionalLight::new(
            Vec3::new(0.4, 0.7, 1.0).normalize(),
            Vec3::new(1.0, 0.95, 0.85),
        ),
        PointLight::new(
            Vec3::new(2.0, 2.0, 3.0),
            Vec3::new(1.0, 0.45, 0.2),
            1.0,
            0.08,
            0.02,
        ),
    )
}

fn render(mesh: &Mesh, thread_count: usize, simd: bool) {
    set_simd_for_tests(Some(simd));
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.set_thread_count(thread_count);
    pipeline.draw_mesh(&mut framebuffer, mesh, &uniforms());
    black_box(framebuffer.color);
    set_simd_for_tests(None);
}

fn subdivided_icosahedron(levels: usize) -> Mesh {
    let base = Mesh::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icosahedron.obj"))
        .expect("load base icosahedron");
    let mut triangles: Vec<[Vec3; 3]> = base
        .triangles
        .iter()
        .map(|&[a, b, c]| {
            [
                base.vertices[a].position,
                base.vertices[b].position,
                base.vertices[c].position,
            ]
        })
        .collect();

    for _ in 0..levels {
        let mut next = Vec::with_capacity(triangles.len() * 4);
        for [a, b, c] in triangles {
            let ab = (a + b).normalize();
            let bc = (b + c).normalize();
            let ca = (c + a).normalize();
            next.extend_from_slice(&[[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
        }
        triangles = next;
    }

    let mut mesh = Mesh::default();
    mesh.vertices.reserve(triangles.len() * 3);
    mesh.triangles.reserve(triangles.len());
    for triangle in triangles {
        let first = mesh.vertices.len();
        for position in triangle {
            mesh.vertices.push(MeshVertex {
                position,
                texcoord: None,
                normal: Some(position.normalize()),
            });
        }
        mesh.triangles.push([first, first + 1, first + 2]);
    }
    mesh
}

fn exact_triangle_scene(triangle_count: usize) -> Mesh {
    let mut mesh = subdivided_icosahedron(6);
    let source = mesh.triangles.clone();
    mesh.triangles.extend(
        source
            .into_iter()
            .cycle()
            .take(triangle_count - mesh.triangles.len()),
    );
    mesh
}

fn bench_raster(c: &mut Criterion) {
    let asset = Mesh::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icosahedron.obj"))
        .expect("load benchmark asset");
    let subdivided = subdivided_icosahedron(6);
    let exact_100k = exact_triangle_scene(100_000);
    let parallel_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);

    for (name, mesh) in [
        ("asset icosahedron", &asset),
        ("81920 triangle icosahedron", &subdivided),
        ("100000 triangle scene", &exact_100k),
    ] {
        for (mode, simd) in [("scalar", false), ("simd", true)] {
            c.bench_function(&format!("{name} {mode} serial"), |b| {
                b.iter(|| render(black_box(mesh), 1, simd))
            });
            c.bench_function(&format!("{name} {mode} parallel"), |b| {
                b.iter(|| render(black_box(mesh), parallel_threads, simd))
            });
        }
    }
}

criterion_group!(benches, bench_raster);
criterion_main!(benches);
