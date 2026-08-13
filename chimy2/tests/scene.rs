use chimy2::csm::{CascadeShadowConfig, render_cascade_shadow_maps_with_config};
use chimy2::fb::Framebuffer;
use chimy2::ibl::{FloatCube, IblMaps, IblSettings};
use chimy2::math::{Mat4, Vec3, Vec4};
use chimy2::mesh::Mesh;
use chimy2::mesh::MeshVertex;
use chimy2::pipeline::{
    FragmentStage, Instance, InstanceUniforms, Pipeline, VertexOutput, VertexStage,
};
use chimy2::postfx::{AcesTonemapPass, BloomPass, PostChain, SsaoPass};
use chimy2::scene::{CameraConfig, LightType, MaterialType, PostFxConfig, Scene, TransformConfig};
use chimy2::shaders::{
    BlinnPhongShader, BlinnPhongUniforms, BlinnPhongVaryings, DirectionalLight,
    IblCookTorranceShader, IblCookTorranceUniforms, PointLight,
};
use chimy2::shadow::{CubeShadowState, render_cube_shadow_map};
use chimy2::shadow::{ShadowMap, ShadowState};
use chimy2::skybox::CubeTexture;
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
  "camera":{"position":[4,2.5,6],"target":[0,0,0],"fov":0.9,"near":0.1,"far":30},
  "environment":{
    "background":[0.01,0.02,0.04,1],
    "skybox":{"px":"skybox_px.qoi","nx":"skybox_nx.qoi","py":"skybox_py.qoi","ny":"skybox_ny.qoi","pz":"skybox_pz.qoi","nz":"skybox_nz.qoi"},
    "ibl":{"intensity":0.8,"irradiance_size":2,"prefilter_size":2,"prefilter_levels":2}
  },
  "lights":[
    {"type":"directional","direction":[-0.4,1,0.5],"color":[1,0.9,0.8],"shadow":{"type":"csm","map_size":64,"cascades":2,"lambda":0.5,"bias":0.002,"slope_bias":0.02,"near":0.1,"far":30}},
    {"type":"point","position":[2,2,3],"color":[1,0.3,0.1],"constant":1,"linear":0.04,"quadratic":0.01,"shadow":{"type":"cube","map_size":64,"near":0.1,"far":20,"bias":0.002,"slope_bias":0.02}}
  ],
  "objects":[
    {"mesh":"icosahedron.obj","material":{"type":"ggx","ambient":[0.03,0.04,0.08],"diffuse":[0.16,0.38,0.95],"specular":[1,0.85,0.55],"metallic":0.8,"roughness":0.24},"transform":{"position":[-1.1,0,0],"rotation":[0,0.4,0],"scale":[1,1,1]},"instancing":{"count":2,"transforms":[{"position":[0,0,0]},{"position":[2,0,0]}]}},
    {"mesh":"cube.obj","material":{"type":"blinn_phong","ambient":[0.08,0.025,0.012],"diffuse":[0.95,0.24,0.06],"specular":[1,0.55,0.22],"shininess":32},"transform":{"position":[1,-0.3,0],"rotation":[0.2,-0.5,0.15],"scale":[0.8,0.8,0.8]}}
  ],
  "postfx":[
    {"type":"ssao","radius":0.55,"bias":0.025,"strength":1,"range":0.9,"blur_depth_threshold":0.35},
    {"type":"bloom"},
    {"type":"aces","exposure":1}
  ],
  "particles":[],
  "hud":[{"text":"ANCHOR","x":4,"y":4,"scale":1,"color":[0.85,0.92,1,1]}]
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

#[derive(Clone, Debug)]
enum TwinUniforms<'a> {
    Blinn(BlinnPhongUniforms),
    IblGgx(IblCookTorranceUniforms<'a>),
}

#[derive(Clone, Copy, Debug, Default)]
struct TwinShader;

impl<'a> VertexStage<MeshVertex, TwinUniforms<'a>> for TwinShader {
    type Varyings = BlinnPhongVaryings;

    fn run(
        &self,
        vertex: &MeshVertex,
        uniforms: &TwinUniforms<'a>,
    ) -> VertexOutput<Self::Varyings> {
        match uniforms {
            TwinUniforms::Blinn(value) => VertexStage::run(&BlinnPhongShader, vertex, value),
            TwinUniforms::IblGgx(value) => VertexStage::run(&IblCookTorranceShader, vertex, value),
        }
    }
}

impl<'a> FragmentStage<BlinnPhongVaryings, TwinUniforms<'a>> for TwinShader {
    fn run(&self, varyings: &BlinnPhongVaryings, uniforms: &TwinUniforms<'a>) -> u32 {
        match uniforms {
            TwinUniforms::Blinn(value) => FragmentStage::run(&BlinnPhongShader, varyings, value),
            TwinUniforms::IblGgx(value) => {
                FragmentStage::run(&IblCookTorranceShader, varyings, value)
            }
        }
    }

    fn run_linear(&self, varyings: &BlinnPhongVaryings, uniforms: &TwinUniforms<'a>) -> [f32; 4] {
        match uniforms {
            TwinUniforms::Blinn(value) => {
                FragmentStage::run_linear(&BlinnPhongShader, varyings, value)
            }
            TwinUniforms::IblGgx(value) => {
                FragmentStage::run_linear(&IblCookTorranceShader, varyings, value)
            }
        }
    }

    fn is_opaque(&self, uniforms: &TwinUniforms<'a>) -> bool {
        match uniforms {
            TwinUniforms::Blinn(value) => FragmentStage::is_opaque(&BlinnPhongShader, value),
            TwinUniforms::IblGgx(value) => FragmentStage::is_opaque(&IblCookTorranceShader, value),
        }
    }

    fn model_view(&self, uniforms: &TwinUniforms<'a>) -> Option<Mat4> {
        match uniforms {
            TwinUniforms::Blinn(value) => FragmentStage::model_view(&BlinnPhongShader, value),
            TwinUniforms::IblGgx(value) => FragmentStage::model_view(&IblCookTorranceShader, value),
        }
    }

    fn culling_transform(&self, uniforms: &TwinUniforms<'a>) -> Option<Mat4> {
        match uniforms {
            TwinUniforms::Blinn(value) => {
                FragmentStage::culling_transform(&BlinnPhongShader, value)
            }
            TwinUniforms::IblGgx(value) => {
                FragmentStage::culling_transform(&IblCookTorranceShader, value)
            }
        }
    }
}

impl<'a> InstanceUniforms for TwinUniforms<'a> {
    fn set_instance_model(&mut self, model: Mat4) {
        match self {
            TwinUniforms::Blinn(value) => value.set_model(model),
            TwinUniforms::IblGgx(value) => value.lighting.lighting.set_model(model),
        }
    }

    fn apply_instance_tint(&mut self, tint: Vec4) {
        match self {
            TwinUniforms::Blinn(value) => {
                value.set_diffuse_color(value.diffuse_color() * Vec3::new(tint.x, tint.y, tint.z));
                value.set_alpha(value.alpha * tint.w);
            }
            TwinUniforms::IblGgx(value) => {
                value
                    .lighting
                    .set_base_color_linear(value.base_color() * Vec3::new(tint.x, tint.y, tint.z));
                value
                    .lighting
                    .set_alpha(value.lighting.lighting.alpha * tint.w);
            }
        }
    }

    fn for_instance(&self, instance: &Instance) -> Self {
        match self {
            TwinUniforms::Blinn(value) => TwinUniforms::Blinn(value.for_instance(instance)),
            TwinUniforms::IblGgx(value) => {
                let mut result = value.clone();
                result.lighting.lighting.set_model(instance.model());
                if let Some(tint) = instance.tint() {
                    result.lighting.set_base_color_linear(
                        result.base_color() * Vec3::new(tint.x, tint.y, tint.z),
                    );
                    result
                        .lighting
                        .set_alpha(result.lighting.lighting.alpha * tint.w);
                }
                TwinUniforms::IblGgx(result)
            }
        }
    }
}

fn hand_coded_anchor() -> Framebuffer {
    let root = assets();
    let camera_config = CameraConfig {
        position: Vec3::new(4.0, 2.5, 6.0),
        target: Vec3::ZERO,
        fov: 0.9,
        near: 0.1,
        far: 30.0,
    };
    let camera = camera_config.camera(64.0 / 48.0);
    let directional = DirectionalLight::new(Vec3::new(-0.4, 1.0, 0.5), Vec3::new(1.0, 0.9, 0.8));
    let point = PointLight::new(
        Vec3::new(2.0, 2.0, 3.0),
        Vec3::new(1.0, 0.3, 0.1),
        1.0,
        0.04,
        0.01,
    );
    let ico = Mesh::load(root.join("icosahedron.obj")).unwrap();
    let cube = Mesh::load(root.join("cube.obj")).unwrap();
    let ico_model = TransformConfig {
        position: Vec3::new(-1.1, 0.0, 0.0),
        rotation: Vec3::new(0.0, 0.4, 0.0),
        scale: Vec3::new(1.0, 1.0, 1.0),
    }
    .matrix();
    let cube_model = TransformConfig {
        position: Vec3::new(1.0, -0.3, 0.0),
        rotation: Vec3::new(0.2, -0.5, 0.15),
        scale: Vec3::new(0.8, 0.8, 0.8),
    }
    .matrix();
    let instances = vec![
        Instance::new(ico_model),
        Instance::new(
            ico_model
                * TransformConfig {
                    position: Vec3::new(2.0, 0.0, 0.0),
                    ..TransformConfig::default()
                }
                .matrix(),
        ),
    ];
    let shadow_meshes = vec![
        (&ico, instances[0].model()),
        (&ico, instances[1].model()),
        (&cube, cube_model),
    ];
    let mut cascades = render_cascade_shadow_maps_with_config(
        camera,
        Vec3::new(-0.4, 1.0, 0.5),
        CascadeShadowConfig::new(2, 0.5),
        64,
        &shadow_meshes,
    )
    .unwrap();
    cascades.set_bias(0.002, 0.02);
    let cube_map = render_cube_shadow_map(point.position, 0.1, 20.0, 64, &shadow_meshes).unwrap();
    let mut cube_shadow = CubeShadowState::new(point.position, cube_map);
    cube_shadow.set_bias(0.002, 0.02);

    let skybox = {
        let faces = [
            "skybox_px.qoi",
            "skybox_nx.qoi",
            "skybox_py.qoi",
            "skybox_ny.qoi",
            "skybox_pz.qoi",
            "skybox_nz.qoi",
        ]
        .map(|name| chimy2::image::Texture::load(root.join(name)).unwrap());
        CubeTexture::new(faces).unwrap()
    };
    let source = FloatCube::from_cube_texture(&skybox, 0.8);
    let defaults = IblSettings::default();
    let ibl = IblMaps::from_float_environment(
        &source,
        IblSettings {
            irradiance_size: 2,
            irradiance_samples: defaults.irradiance_samples,
            prefilter_size: 2,
            prefilter_levels: 2,
            prefilter_samples: defaults.prefilter_samples,
            brdf_size: defaults.brdf_size,
            brdf_samples: defaults.brdf_samples,
        },
    );

    let mut ggx_lighting = BlinnPhongUniforms::new(
        ico_model,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.03, 0.04, 0.08),
        Vec3::new(0.16, 0.38, 0.95),
        Vec3::new(1.0, 0.85, 0.55),
        24.0,
        camera.position,
        directional,
        point,
    );
    ggx_lighting.set_alpha(1.0);
    ggx_lighting.clear_directional_lights();
    ggx_lighting.clear_point_lights();
    ggx_lighting.add_directional_light(directional).unwrap();
    ggx_lighting.add_point_light(point).unwrap();
    ggx_lighting.set_directional_cascaded_shadow(directional, cascades.clone());
    ggx_lighting
        .set_point_light_shadow(0, Some(cube_shadow.clone()))
        .unwrap();
    let ggx = TwinUniforms::IblGgx(IblCookTorranceUniforms::new(
        ggx_lighting,
        &ibl,
        Vec3::new(0.16, 0.38, 0.95),
        0.8,
        0.24,
    ));

    let mut blinn = BlinnPhongUniforms::new(
        cube_model,
        camera.view_matrix(),
        camera.projection_matrix(),
        Vec3::new(0.08, 0.025, 0.012),
        Vec3::new(0.95, 0.24, 0.06),
        Vec3::new(1.0, 0.55, 0.22),
        32.0,
        camera.position,
        directional,
        point,
    );
    blinn.set_alpha(1.0);
    blinn.clear_directional_lights();
    blinn.clear_point_lights();
    blinn.add_directional_light(directional).unwrap();
    blinn.add_point_light(point).unwrap();
    blinn.set_directional_cascaded_shadow(directional, cascades);
    blinn.set_point_light_shadow(0, Some(cube_shadow)).unwrap();
    let blinn = TwinUniforms::Blinn(blinn);

    let mut framebuffer = Framebuffer::new(64, 48);
    framebuffer.clear(chimy2::fb::argb8888(255, 3, 5, 10));
    let mut pipeline = Pipeline::new(TwinShader, TwinShader);
    pipeline.set_hdr(true);
    let mut ssao = SsaoPass::new(camera.projection_matrix());
    ssao.set_radius(0.55);
    ssao.set_bias(0.025);
    ssao.set_strength(1.0);
    ssao.set_range(0.9);
    ssao.set_blur_depth_threshold(0.35);
    let mut post_chain = PostChain::new();
    post_chain.push(ssao);
    post_chain.push(BloomPass);
    post_chain.push(AcesTonemapPass::new(1.0));
    pipeline.set_post_chain(post_chain);
    pipeline.render(&mut framebuffer, |frame, target| {
        frame.draw_skybox(target, &skybox, camera);
        frame.draw_mesh_instanced(target, &ico, &ggx, &instances);
        frame.draw_mesh(target, &cube, &blinn);
    });
    framebuffer.draw_text(4, 4, "ANCHOR", 1, chimy2::fb::argb8888(255, 217, 235, 255));
    framebuffer
}

#[test]
fn anchor_scene_matches_hand_coded_twin_byte_for_byte() {
    let scene = Scene::from_str(ANCHOR).unwrap();
    let mut from_scene = Framebuffer::new(64, 48);
    scene.render(&mut from_scene, assets()).unwrap();
    let hand = hand_coded_anchor();
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
    assert!(scene.objects.iter().any(|object| object.lod.is_some()));
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
    let incompatible = Scene::from_str(
        r#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":10},"objects":[{"mesh":"cube.obj","instancing":{"count":1,"grid":{"dimensions":[1,1,1],"spacing":[0,0,0]}},"lod":{"ratios":[0.5],"thresholds":[100]}}]}"#,
    )
    .unwrap_err();
    assert!(incompatible.0.contains("objects[0]: instancing and lod"));
    let unsupported_shadow = Scene::from_str(
        r#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":10},"lights":[{"type":"point","shadow":{"type":"csm"}}]}"#,
    )
    .unwrap_err();
    assert!(
        unsupported_shadow
            .0
            .contains("lights[0].shadow.type: unsupported shadow type")
    );
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
          "lights":[{"type":"directional","direction":[0,1,0],"shadow":{"type":"csm","map_size":0,"cascades":-1,"lambda":-2,"near":-1,"far":-2,"light_size":1e39}}],
          "objects":[{"mesh":"cube.obj","material":{"alpha":4,"metallic":-1,"roughness":0},"transform":{"scale":[-2,3,4]},"lod":{"ratios":[0.5,0.25],"thresholds":[100,300,50]}}]
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
    let mut direct_pcss = ShadowState::new(
        Mat4::IDENTITY,
        ShadowMap::from_depth(1, 1, vec![0.0]).unwrap(),
    );
    direct_pcss.set_light_size(f32::INFINITY);
    assert_eq!(shadow.light_size, direct_pcss.light_size());
    assert_eq!(scene.objects[0].material.alpha, 1.0);
    assert_eq!(scene.objects[0].material.metallic, 0.0);
    assert_eq!(scene.objects[0].material.roughness, 0.001);
    assert_eq!(
        scene.objects[0].lod.as_ref().unwrap().thresholds,
        vec![100.0, 100.0, 50.0]
    );
}

fn shadow_probe(shadow: &str, instanced: bool) -> Framebuffer {
    let caster = if instanced {
        r#"{"mesh":"cube.obj","material":{"diffuse":[0.8,0.2,0.1]},"transform":{"position":[0,0,0],"scale":[1,1,1]},"instancing":{"count":2,"transforms":[{"position":[0,0,0]},{"position":[2,0,0]}]}}"#
    } else {
        r#"{"mesh":"cube.obj","material":{"diffuse":[0.8,0.2,0.1]},"transform":{"position":[0,0,0],"scale":[1,1,1]}}"#
    };
    let source = format!(
        r#"{{
          "camera":{{"position":[5,4,7],"target":[0,0,0],"fov":0.9,"near":0.1,"far":30}},
          "lights":[{{"type":"directional","direction":[0.4,1,0.2],"color":[1,1,1]{shadow}}}],
          "objects":[{{"mesh":"cube.obj","material":{{"diffuse":[0.35,0.35,0.35]}},"transform":{{"position":[0,-1.2,0],"scale":[5,0.2,5]}}}},{caster}],
          "postfx":[],"hud":[],"particles":[]}}
        "#,
        shadow = shadow,
        caster = caster,
    );
    let scene = Scene::from_str(&source).unwrap();
    let mut framebuffer = Framebuffer::new(96, 64);
    scene.render(&mut framebuffer, assets()).unwrap();
    framebuffer
}

#[test]
fn every_directional_shadow_mode_changes_rendering() {
    let no_shadow = shadow_probe("", false);
    for shadow in [
        ",\"shadow\":{\"type\":\"basic\",\"map_size\":64}",
        ",\"shadow\":{\"type\":\"pcss\",\"map_size\":64,\"light_size\":2}",
        ",\"shadow\":{\"type\":\"csm\",\"map_size\":64,\"cascades\":2,\"lambda\":0.5}",
    ] {
        assert_ne!(
            shadow_probe(shadow, false).color,
            no_shadow.color,
            "{shadow}"
        );
    }
}

#[test]
fn shadow_bias_and_instanced_casters_reach_rendering() {
    let no_shadow = shadow_probe("", true);
    let instanced = shadow_probe(
        ",\"shadow\":{\"type\":\"csm\",\"map_size\":64,\"cascades\":2,\"bias\":0,\"slope_bias\":0}",
        true,
    );
    assert_ne!(instanced.color, no_shadow.color);
    let low_bias = shadow_probe(
        ",\"shadow\":{\"type\":\"basic\",\"map_size\":64,\"bias\":0,\"slope_bias\":0}",
        false,
    );
    let high_bias = shadow_probe(
        ",\"shadow\":{\"type\":\"basic\",\"map_size\":64,\"bias\":100,\"slope_bias\":100}",
        false,
    );
    assert_ne!(low_bias.color, high_bias.color);
}

fn point_shadow_probe(first_shadow: bool, second_shadow: bool) -> Framebuffer {
    let shadow = |enabled| {
        if enabled {
            ",\"shadow\":{\"type\":\"cube\",\"map_size\":64,\"near\":0.1,\"far\":20,\"bias\":0,\"slope_bias\":0}"
        } else {
            ""
        }
    };
    let source = format!(
        r#"{{
          "camera":{{"position":[0,5,8],"target":[0,0,0],"fov":0.9,"near":0.1,"far":30}},
          "lights":[
            {{"type":"point","position":[-3,3,2],"color":[1,0.2,0.1],"constant":1,"linear":0.03,"quadratic":0.01{first}}},
            {{"type":"point","position":[3,3,2],"color":[0.1,0.3,1],"constant":1,"linear":0.03,"quadratic":0.01{second}}}
          ],
          "objects":[
            {{"mesh":"cube.obj","material":{{"diffuse":[0.35,0.35,0.35]}},"transform":{{"position":[0,-1.2,0],"scale":[5,0.2,5]}}}},
            {{"mesh":"cube.obj","material":{{"diffuse":[0.8,0.2,0.1]}},"transform":{{"position":[-2,0,0],"scale":[0.8,1.2,0.8]}}}},
            {{"mesh":"cube.obj","material":{{"diffuse":[0.1,0.2,0.8]}},"transform":{{"position":[2,0,0],"scale":[0.8,1.2,0.8]}}}}
          ]
        }}"#,
        first = shadow(first_shadow),
        second = shadow(second_shadow),
    );
    let scene = Scene::from_str(&source).unwrap();
    let mut framebuffer = Framebuffer::new(96, 64);
    scene.render(&mut framebuffer, assets()).unwrap();
    framebuffer
}

#[test]
fn every_shadowed_point_light_reaches_its_shader_slot() {
    let control = point_shadow_probe(false, false);
    let first = point_shadow_probe(true, false);
    let second = point_shadow_probe(false, true);
    assert_ne!(
        first.color, control.color,
        "first point shadow had no effect"
    );
    assert_ne!(
        second.color, control.color,
        "second point shadow had no effect"
    );
    assert_ne!(first.color, second.color);
}

#[test]
fn lod_scene_golden_changes_when_selection_threshold_changes() {
    let source = |threshold: &str| {
        format!(
            r#"{{
              "camera":{{"position":[0,0,6],"target":[0,0,0],"fov":0.9,"near":0.1,"far":30}},
              "lights":[{{"type":"directional","direction":[0,1,1],"color":[1,1,1]}}],
              "objects":[{{"mesh":"icosahedron.obj","material":{{"diffuse":[0.2,0.7,0.3]}},"lod":{{"ratios":[0.05],"thresholds":[{threshold}]}},"transform":{{"position":[0,0,0],"scale":[1,1,1]}}}}]
            }}"#,
            threshold = threshold,
        )
    };
    let render = |source: String| {
        let scene = Scene::from_str(&source).unwrap();
        let mut framebuffer = Framebuffer::new(96, 64);
        scene.render(&mut framebuffer, assets()).unwrap();
        framebuffer
    };
    assert_ne!(render(source("100.0")).color, render(source("0.0")).color);
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

#[test]
fn shipped_scene_golden_rejects_extreme_lod_thresholds() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/showcase.scene.json");
    let source = fs::read_to_string(&path).unwrap();
    let golden = fs::read(golden_path()).unwrap();
    for thresholds in ["[0.0, 0.0]", "[10000.0, 10000.0]"] {
        let mutated = source.replace(
            "\"thresholds\": [1000.0, 0.1]",
            &format!("\"thresholds\": {thresholds}"),
        );
        assert_ne!(mutated, source);
        let scene = Scene::from_str(&mutated).unwrap();
        let mut framebuffer = Framebuffer::new(96, 64);
        scene
            .render(&mut framebuffer, path.parent().unwrap())
            .unwrap();
        assert_ne!(ppm(&framebuffer), golden, "LOD mutation {thresholds}");
    }
}
