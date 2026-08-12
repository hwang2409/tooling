use chimy2::csm::fit_cascade_light_projection;
use chimy2::culling::Frustum;
use chimy2::demo::{build_culling_instancing_scene, render_instancing_scene_with_culling};
use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::{Mesh, MeshVertex};
use chimy2::pipeline::Pipeline;
use chimy2::shaders::{MeshShader, MeshUniforms};
use chimy2::shadow::{ShadowDepthShader, ShadowDepthUniforms, cube_face_view_projections};
use std::fs;
use std::path::Path;

fn bytes(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut output = Vec::with_capacity(framebuffer.color.len() * 4);
    for pixel in &framebuffer.color {
        output.extend_from_slice(&pixel.to_ne_bytes());
    }
    output.extend(
        framebuffer
            .depth
            .iter()
            .flat_map(|depth| depth.to_ne_bytes()),
    );
    output
}

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut output =
        format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        output.extend_from_slice(&[red, green, blue]);
    }
    output
}

#[test]
fn instanced_demo_culling_on_and_off_are_byte_identical() {
    let scene = build_culling_instancing_scene(96.0 / 64.0);
    let mut culled = Framebuffer::new(96, 64);
    let mut unculled = Framebuffer::new(96, 64);
    render_instancing_scene_with_culling(&mut culled, &scene, true);
    render_instancing_scene_with_culling(&mut unculled, &scene, false);
    assert_eq!(bytes(&culled), bytes(&unculled));
}

#[test]
fn culling_demo_golden() {
    let mut framebuffer = Framebuffer::new(96, 64);
    let scene = build_culling_instancing_scene(96.0 / 64.0);
    render_instancing_scene_with_culling(&mut framebuffer, &scene, true);
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join("culling-instancing-field.ppm");
    let actual = ppm(&framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).expect("write golden");
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    let expected = fs::read(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    assert_eq!(actual, expected, "golden mismatch: {}", path.display());
}

#[test]
fn production_culling_keeps_a_mesh_that_straddles_a_camera_plane() {
    let mesh = Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(-1.2, -0.5, -2.0), None, None),
            MeshVertex::new(Vec3::new(0.2, -0.5, -2.0), None, None),
            MeshVertex::new(Vec3::new(-1.2, 0.5, -2.0), None, None),
        ],
        vec![[0, 1, 2]],
    );
    let uniforms = MeshUniforms::new(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 5.0),
        0xFFFF8040,
    );
    let render = |culling_enabled| {
        let mut framebuffer = Framebuffer::new(32, 32);
        let mut pipeline = Pipeline::new(MeshShader, MeshShader);
        pipeline.set_culling_enabled(culling_enabled);
        pipeline.set_thread_count(1);
        pipeline.render(&mut framebuffer, |frame, target| {
            frame.draw_mesh(target, &mesh, &uniforms);
        });
        bytes(&framebuffer)
    };
    assert_eq!(render(true), render(false));
}

#[test]
fn depth_culling_uses_each_light_pass_frustum() {
    let caster = Mesh::new(
        vec![
            MeshVertex::new(Vec3::new(-0.5, -0.5, -0.5), None, None),
            MeshVertex::new(Vec3::new(0.5, -0.5, -0.5), None, None),
            MeshVertex::new(Vec3::new(-0.5, 0.5, -0.5), None, None),
        ],
        vec![[0, 1, 2]],
    );
    let model = Mat4::translate(Vec3::new(8.0, 0.0, -2.0));
    let camera = Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 5.0);
    let light = Mat4::orthographic(-10.0, 10.0, -10.0, 10.0, 1.0, 20.0)
        * Mat4::look_at(
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        );
    let cascade = fit_cascade_light_projection(
        chimy2::camera::Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            chimy2::math::Quat::IDENTITY,
            1.0,
            1.0,
            0.1,
            20.0,
        ),
        Vec3::new(0.0, 1.0, 0.0),
        0.1,
        10.0,
        32,
    );
    let cube = cube_face_view_projections(Vec3::ZERO, 0.1, 20.0)[0];
    for matrix in [light, cascade, cube] {
        let expanded =
            Frustum::from_view_projection_including_bounds(matrix, [(caster.bounds(), model)]);
        let render = |culling_enabled| {
            let mut framebuffer = Framebuffer::new(32, 32);
            let mut pipeline = Pipeline::new(ShadowDepthShader, ShadowDepthShader);
            pipeline.set_culling_enabled(culling_enabled);
            pipeline.set_culling_frustum(Some(expanded));
            pipeline.draw_mesh_depth_with_varyings(
                &mut framebuffer,
                &caster,
                &ShadowDepthUniforms::new(model, matrix),
            );
            bytes(&framebuffer)
        };
        assert_eq!(render(true), render(false));
        assert!(!Frustum::from_view_projection(camera).intersects_aabb(caster.bounds(), model,));
    }
}
