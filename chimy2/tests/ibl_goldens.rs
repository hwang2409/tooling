use chimy2::camera::Camera;
use chimy2::demo::uv_sphere;
use chimy2::fb::{Framebuffer, argb8888};
use chimy2::ibl::{FloatCube, IblMaps, IblSettings};
use chimy2::image::{ColorSpace, Texture};
use chimy2::math::{Mat4, Quat, Vec3};
use chimy2::pipeline::Pipeline;
use chimy2::postfx::{AcesTonemapPass, PostChain};
use chimy2::shaders::{
    BlinnPhongUniforms, CookTorranceShader, CookTorranceUniforms, DirectionalLight,
    IblCookTorranceShader, IblCookTorranceUniforms, PointLight,
};
use chimy2::skybox::CubeTexture;
use std::fs;
use std::path::{Path, PathBuf};

const WIDTH: usize = 96;
const HEIGHT: usize = 64;

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
        .join("tests/goldens")
        .join(format!("{name}.ppm"))
}

fn assert_golden(name: &str, framebuffer: &Framebuffer) {
    let path = golden_path(name);
    let actual = ppm(framebuffer);
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, actual).unwrap();
        return;
    }
    assert_eq!(fs::read(path).unwrap(), actual, "{name}");
}

fn camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 0.15, 6.5),
        Quat::IDENTITY,
        0.85,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        30.0,
    )
}

fn environment() -> CubeTexture {
    let colors = [
        [210, 80, 45, 255],
        [45, 100, 210, 255],
        [80, 190, 105, 255],
        [35, 65, 110, 255],
        [255, 235, 155, 255],
        [25, 35, 60, 255],
    ];
    CubeTexture::new(std::array::from_fn(|face| {
        let pixels = (0..8 * 8)
            .map(|index| {
                if face == 4 && (index % 8 >= 5) && (index / 8 <= 2) {
                    [255, 255, 240, 255]
                } else {
                    colors[face]
                }
            })
            .collect();
        Texture::new_with_color_space(8, 8, pixels, ColorSpace::Linear).unwrap()
    }))
    .unwrap()
}

fn settings() -> IblSettings {
    IblSettings {
        irradiance_size: 8,
        irradiance_samples: 64,
        prefilter_size: 16,
        prefilter_levels: 5,
        prefilter_samples: 64,
        brdf_size: 32,
        brdf_samples: 64,
    }
}

fn lighting(model: Mat4, camera: &Camera, ambient: Vec3) -> BlinnPhongUniforms {
    BlinnPhongUniforms::new_with_linear_colors(
        model,
        camera.view_matrix(),
        camera.projection_matrix(),
        ambient,
        Vec3::ZERO,
        Vec3::ZERO,
        0.0,
        camera.position,
        DirectionalLight::new(Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO),
        PointLight::default(),
    )
}

fn ibl_uniforms<'a>(
    model: Mat4,
    camera: &Camera,
    maps: &'a IblMaps,
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
) -> IblCookTorranceUniforms<'a> {
    IblCookTorranceUniforms::new_with_linear_base_color(
        lighting(model, camera, Vec3::ZERO),
        maps,
        base_color,
        metallic,
        roughness,
    )
}

fn render_metal_roughness_sweep() -> Framebuffer {
    let source = environment();
    let maps = IblMaps::from_environment_with_settings(&source, settings());
    let camera = camera();
    let sphere = uv_sphere(0.78, 24, 48);
    let roughness = [0.05, 0.25, 0.5, 0.85];
    let uniforms: Vec<_> = roughness
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let x = (index as f32 - 1.5) * 1.55;
            ibl_uniforms(
                Mat4::translate(Vec3::new(x, 0.0, 0.0)),
                &camera,
                &maps,
                Vec3::new(0.92, 0.68, 0.22),
                1.0,
                value,
            )
        })
        .collect();
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 5, 7, 12));
    let mut pipeline = Pipeline::new(IblCookTorranceShader, IblCookTorranceShader);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &source, camera);
        for uniform in &uniforms {
            frame.draw_mesh(target, &sphere, uniform);
        }
    });
    framebuffer
}

fn render_dielectric_pair() -> Framebuffer {
    let source = environment();
    let maps = IblMaps::from_environment_with_settings(&source, settings());
    let camera = camera();
    let sphere = uv_sphere(0.92, 24, 48);
    let ibl = ibl_uniforms(
        Mat4::translate(Vec3::new(-1.25, 0.0, 0.0)),
        &camera,
        &maps,
        Vec3::new(0.72, 0.18, 0.06),
        0.0,
        0.42,
    );
    let constant = CookTorranceUniforms::new_with_linear_base_color(
        lighting(
            Mat4::translate(Vec3::new(1.25, 0.0, 0.0)),
            &camera,
            Vec3::new(0.18, 0.04, 0.02),
        ),
        Vec3::new(0.72, 0.18, 0.06),
        0.0,
        0.42,
    );
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 5, 7, 12));
    let mut ibl_pipeline = Pipeline::new(IblCookTorranceShader, IblCookTorranceShader);
    ibl_pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &source, camera);
        frame.draw_mesh(target, &sphere, &ibl);
    });
    let mut constant_pipeline = Pipeline::new(CookTorranceShader, CookTorranceShader);
    constant_pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_mesh(target, &sphere, &constant);
    });
    framebuffer
}

fn render_hdr_aces() -> Framebuffer {
    let source = environment();
    let hdr_source = FloatCube::from_cube_texture(&source, 4.0);
    let maps = IblMaps::from_float_environment(&hdr_source, settings());
    let camera = camera();
    let sphere = uv_sphere(0.95, 24, 48);
    let uniforms = [
        ibl_uniforms(
            Mat4::translate(Vec3::new(-1.15, 0.0, 0.0)),
            &camera,
            &maps,
            Vec3::new(0.85, 0.2, 0.04),
            1.0,
            0.12,
        ),
        ibl_uniforms(
            Mat4::translate(Vec3::new(1.15, 0.0, 0.0)),
            &camera,
            &maps,
            Vec3::new(0.2, 0.55, 0.9),
            0.0,
            0.35,
        ),
    ];
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);
    framebuffer.clear(argb8888(255, 5, 7, 12));
    let mut pipeline = Pipeline::new(IblCookTorranceShader, IblCookTorranceShader);
    pipeline.set_hdr(true);
    pipeline.set_post_chain(PostChain::new().with_pass(AcesTonemapPass::new(1.0)));
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &source, camera);
        for uniform in &uniforms {
            frame.draw_mesh(target, &sphere, uniform);
        }
    });
    framebuffer
}

#[test]
fn metal_roughness_sweep_reflects_prefiltered_levels() {
    assert_golden("ibl-metal-roughness-sweep", &render_metal_roughness_sweep());
}

#[test]
fn dielectric_ibl_differs_from_constant_ambient_sibling() {
    let framebuffer = render_dielectric_pair();
    assert_golden("ibl-dielectric-ambient-pair", &framebuffer);
}

#[test]
fn ibl_hdr_aces_full_stack() {
    assert_golden("ibl-hdr-aces", &render_hdr_aces());
}
