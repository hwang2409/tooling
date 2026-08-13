use chimy2::csm::CascadeShadowConfig;
use chimy2::fb::Framebuffer;
use chimy2::math::{Mat4, Vec3};
use chimy2::mesh::Mesh;
use chimy2::pipeline::Pipeline;
use chimy2::postfx::{AcesTonemapPass, BloomPass, PostChain};
use chimy2::scene::{LightType, MaterialType, PostFxConfig, Scene};
use chimy2::shaders::{BlinnPhongShader, BlinnPhongUniforms, DirectionalLight, PointLight};
use std::fs;
use std::path::{Path, PathBuf};

fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn ppm(framebuffer: &Framebuffer) -> Vec<u8> {
    let mut bytes = format!("P6\n{} {}\n255\n", framebuffer.width, framebuffer.height).into_bytes();
    for &pixel in &framebuffer.color {
        let [_, red, green, blue] = pixel.to_be_bytes();
        bytes.extend_from_slice(&[red, green, blue]);
    }
    bytes
}

fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/scene-showcase.ppm")
}

const ANCHOR: &str = r#"{
  "camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":100},
  "environment":{"background":[0.02,0.03,0.05,1]},
  "lights":[{"type":"directional","direction":[0,0,1],"color":[1,1,1]}],
  "objects":[{"mesh":"cube.obj","material":{"ambient":[0.03,0.03,0.03],"diffuse":[0.8,0.4,0.1],"specular":[0.2,0.2,0.2],"shininess":24,"alpha":1},"transform":{"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}}],
  "postfx":[],
  "particles":[],
  "hud":[]
}"#;

const ORDER_ANCHOR: &str = r#"{
  "camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":100},
  "environment":{"background":[0.02,0.03,0.05,1]},
  "lights":[{"type":"directional","direction":[0,0,1],"color":[1,1,1]}],
  "objects":[
    {"mesh":"cube.obj","material":{"ambient":[0.03,0.03,0.03],"diffuse":[1,0,0],"specular":[0.2,0.2,0.2],"alpha":0.5},"transform":{"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}},
    {"mesh":"cube.obj","material":{"ambient":[0.03,0.03,0.03],"diffuse":[0,0,1],"specular":[0.2,0.2,0.2],"alpha":0.5},"transform":{"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}}
  ]
}"#;

const POSTFX_ANCHOR: &str = r#"{
  "camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":100},
  "environment":{"background":[0.02,0.03,0.05,1]},
  "lights":[{"type":"directional","direction":[0,0,1],"color":[1,1,1]}],
  "objects":[{"mesh":"cube.obj","material":{"diffuse":[10,4,1],"shininess":24},"transform":{"position":[0,0,0],"rotation":[0,0,0],"scale":[1,1,1]}}],
  "postfx":[{"type":"bloom"},{"type":"aces","exposure":1.0}]
}"#;

#[test]
fn anchor_scene_matches_hand_coded_twin_byte_for_byte() {
    let scene = Scene::from_str(ANCHOR).unwrap();
    let mut from_scene = Framebuffer::new(64, 48);
    scene.render(&mut from_scene, assets()).unwrap();

    let camera = scene.camera.camera(64.0 / 48.0);
    let mesh = Mesh::load(assets().join("cube.obj")).unwrap();
    let uniforms = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.03, 0.03, 0.03),
        Vec3::new(0.8, 0.4, 0.1),
        Vec3::new(0.2, 0.2, 0.2),
        24.0,
        camera.position,
        DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    );
    let mut hand = Framebuffer::new(64, 48);
    hand.clear(chimy2::fb::argb8888(255, 5, 8, 13));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(&mut hand, |frame, target| {
        frame.draw_mesh(target, &mesh, &uniforms);
    });
    assert_eq!(from_scene.color, hand.color);
    assert_eq!(from_scene.depth, hand.depth);
}

#[test]
fn field_coverage_probe_reads_every_schema_section() {
    let scene =
        Scene::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/showcase.scene.json"))
            .unwrap();
    assert_eq!(scene.camera.far, 100.0);
    assert!(scene.environment.skybox.is_some());
    assert!(scene.environment.ibl.is_some());
    assert_eq!(scene.lights.len(), 2);
    assert_eq!(scene.lights[0].kind, LightType::Directional);
    assert_eq!(scene.lights[0].shadow.unwrap().cascades, 3);
    assert_eq!(scene.lights[1].kind, LightType::Point);
    assert_eq!(
        scene.lights[1].shadow.unwrap().kind,
        chimy2::scene::ShadowType::Cube
    );
    assert_eq!(scene.objects[0].material.kind, MaterialType::Ggx);
    assert_eq!(scene.objects[0].transform.scale.x, 1.1);
    assert_eq!(scene.objects[0].instancing.as_ref().unwrap().count, 3);
    assert!(matches!(scene.postfx[0], PostFxConfig::Ssao { .. }));
    assert!(matches!(scene.postfx[1], PostFxConfig::Bloom));
    assert!(matches!(scene.postfx[2], PostFxConfig::Aces { .. }));
    assert_eq!(scene.particles[0].capacity, 128);
    assert_eq!(scene.hud[0].text, "CHIMY2 SCENE FORMAT");
}

#[test]
fn malformed_inputs_are_errors_with_paths_and_no_panics() {
    assert!(Scene::from_str("{").is_err());
    let wrong_type = Scene::from_str(
        r#"{"camera":{"position":0,"target":[0,0,0],"fov":1,"near":0.1,"far":10}}"#,
    )
    .unwrap_err();
    assert!(wrong_type.0.contains("camera.position"));
    let unknown = Scene::from_str(
        r#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":10},"objects":[{"mesh":"cube.obj","transform":{"scale":[1,1,1],"typo":1}}]}"#,
    )
    .unwrap_err();
    assert!(unknown.0.contains("objects[0].transform.typo"));
    let missing = Scene::from_str(r#"{"camera":{"position":[0,0,5]}}"#).unwrap_err();
    assert!(missing.0.contains("camera.target"));
    let asset = Scene::from_str(
        r#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":10},"objects":[{"mesh":"missing.obj"}]}"#,
    )
    .unwrap();
    let asset_error = asset.validate_assets(assets()).unwrap_err();
    assert!(asset_error.0.contains("objects[0].mesh"));
    let nested = format!("{}0{}", "[".repeat(140), "]".repeat(140));
    assert!(Scene::from_str(&nested).is_err());
}

#[test]
fn sanitation_boundary_matches_production_setters() {
    let scene = Scene::from_str(
        r#"{
          "camera":{"position":[0,0,5],"target":[0,0,0],"fov":0,"near":-1,"far":-2},
          "lights":[{"type":"directional","direction":[0,1,0],"shadow":{"type":"csm","map_size":0,"cascades":1,"lambda":-2,"near":-1,"far":-2}}],
          "objects":[{"mesh":"cube.obj","material":{"alpha":4,"metallic":-1,"roughness":0},"transform":{"scale":[-2,3,4]}}]
        }"#,
    )
    .unwrap();
    assert_eq!(scene.camera.fov, 0.01);
    assert_eq!(scene.camera.near, 0.0001);
    assert_eq!(scene.camera.far, 0.0002);
    let shadow = scene.lights[0].shadow.unwrap();
    let direct = CascadeShadowConfig::new(1, -2.0);
    assert_eq!(shadow.map_size, 1);
    assert_eq!(shadow.cascades, direct.cascade_count());
    assert_eq!(shadow.lambda, direct.lambda());
    assert_eq!(scene.objects[0].material.alpha, 1.0);
    assert_eq!(scene.objects[0].material.metallic, 0.0);
    assert_eq!(scene.objects[0].material.roughness, 0.001);
}

#[test]
fn scene_render_is_deterministic() {
    let scene = Scene::from_str(ANCHOR).unwrap();
    let mut left = Framebuffer::new(64, 48);
    let mut right = Framebuffer::new(64, 48);
    scene.render(&mut left, assets()).unwrap();
    scene.render(&mut right, assets()).unwrap();
    assert_eq!(ppm(&left), ppm(&right));
}

#[test]
fn object_file_order_is_the_render_order() {
    let scene = Scene::from_str(ORDER_ANCHOR).unwrap();
    let mut from_scene = Framebuffer::new(64, 48);
    scene.render(&mut from_scene, assets()).unwrap();
    let camera = scene.camera.camera(64.0 / 48.0);
    let mesh = Mesh::load(assets().join("cube.obj")).unwrap();
    let mut red = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.03, 0.03, 0.03),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.2, 0.2, 0.2),
        24.0,
        camera.position,
        DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    );
    red.set_alpha(0.5);
    let mut blue = red.clone();
    blue.set_diffuse_color(Vec3::new(0.0, 0.0, 1.0));
    let mut hand = Framebuffer::new(64, 48);
    hand.clear(chimy2::fb::argb8888(255, 5, 8, 13));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.render(&mut hand, |frame, target| {
        frame.draw_mesh(target, &mesh, &red);
        frame.draw_mesh(target, &mesh, &blue);
    });
    assert_eq!(from_scene.color, hand.color);
}

#[test]
fn postfx_list_is_wired_in_file_order() {
    let scene = Scene::from_str(POSTFX_ANCHOR).unwrap();
    let camera = scene.camera.camera(64.0 / 48.0);
    assert_eq!(scene.post_chain(camera.projection_matrix()).len(), 2);
    let mut from_scene = Framebuffer::new(64, 48);
    scene.render(&mut from_scene, assets()).unwrap();
    let mesh = Mesh::load(assets().join("cube.obj")).unwrap();
    let uniforms = BlinnPhongUniforms::new(
        Mat4::IDENTITY,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.03, 0.03, 0.03),
        Vec3::new(10.0, 4.0, 1.0),
        Vec3::new(0.2, 0.2, 0.2),
        24.0,
        camera.position,
        DirectionalLight::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0)),
        PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
    );
    let mut hand = Framebuffer::new(64, 48);
    hand.clear(chimy2::fb::argb8888(255, 5, 8, 13));
    let mut chain = PostChain::new();
    chain.push(BloomPass);
    chain.push(AcesTonemapPass::new(1.0));
    let mut pipeline = Pipeline::new(BlinnPhongShader, BlinnPhongShader);
    pipeline.set_hdr(true);
    pipeline.set_post_chain(chain);
    pipeline.render(&mut hand, |frame, target| {
        frame.draw_mesh(target, &mesh, &uniforms);
    });
    assert_eq!(from_scene.color, hand.color);
}

#[test]
fn shipped_scene_golden_uses_the_demo_scene_file() {
    let scene =
        Scene::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/showcase.scene.json"))
            .unwrap();
    let mut framebuffer = Framebuffer::new(96, 64);
    scene
        .render(
            &mut framebuffer,
            Path::new(env!("CARGO_MANIFEST_DIR")).join("examples"),
        )
        .unwrap();
    let actual = ppm(&framebuffer);
    let path = golden_path();
    if std::env::var_os("GOLDEN_REGEN").is_some() {
        fs::write(&path, &actual).unwrap();
        panic!("regenerated {}, rerun without GOLDEN_REGEN", path.display());
    }
    assert_eq!(fs::read(path).unwrap(), actual);
}
