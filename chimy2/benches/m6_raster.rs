use chimy2::camera::Camera;
use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
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

fn render(mesh: &Mesh, thread_count: usize) {
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.set_thread_count(thread_count);
    let draw_uniforms = uniforms();
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, mesh, &draw_uniforms);
    });
    black_box(framebuffer.color);
}

fn subdivided_icosahedron(levels: usize) -> Mesh {
    let base = Mesh::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icosahedron.obj"))
        .expect("load base icosahedron");
    let mut triangles: Vec<[Vec3; 3]> = base
        .indices()
        .iter()
        .map(|&[a, b, c]| {
            [
                base.vertex(a).unwrap().position(),
                base.vertex(b).unwrap().position(),
                base.vertex(c).unwrap().position(),
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

    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    let mut indices = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let first = vertices.len();
        for position in triangle {
            vertices.push(MeshVertex::new(position, None, Some(position.normalize())));
        }
        indices.push([first, first + 1, first + 2]);
    }
    Mesh::new(vertices, indices)
}

fn exact_triangle_scene(triangle_count: usize) -> Mesh {
    let mut mesh = subdivided_icosahedron(6);
    let source = mesh.indices().to_vec();
    let mut indices = source.clone();
    indices.extend(
        source
            .into_iter()
            .cycle()
            .take(triangle_count - indices.len()),
    );
    mesh.set_indices(indices);
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

    c.bench_function("asset icosahedron serial", |b| {
        b.iter(|| render(black_box(&asset), 1))
    });
    c.bench_function("asset icosahedron parallel", |b| {
        b.iter(|| render(black_box(&asset), parallel_threads))
    });
    c.bench_function("81920 triangle icosahedron serial", |b| {
        b.iter(|| render(black_box(&subdivided), 1))
    });
    c.bench_function("81920 triangle icosahedron parallel", |b| {
        b.iter(|| render(black_box(&subdivided), parallel_threads))
    });
    c.bench_function("100000 triangle scene serial", |b| {
        b.iter(|| render(black_box(&exact_100k), 1))
    });
    c.bench_function("100000 triangle scene parallel", |b| {
        b.iter(|| render(black_box(&exact_100k), parallel_threads))
    });
}

criterion_group!(benches, bench_raster);
criterion_main!(benches);
