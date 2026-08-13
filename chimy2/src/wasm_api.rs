#![cfg(any(target_arch = "wasm32", test))]
#![allow(dead_code)]

//! The hand-written browser boundary for the showcase renderer.
//!
//! The renderer stores ARGB8888 pixels because that is the core framebuffer
//! format. The exported byte buffer is RGBA8 for direct `ImageData` upload.
//! The boundary only reorders bytes. It does not apply an sRGB conversion.

use crate::camera::{Camera, OrbitController};
use crate::fb::{Framebuffer, argb8888};
use crate::gltf::{GltfAsset, submit_gltf_draws};
use crate::image::Texture;
use crate::material::MaterialLibrary;
use crate::math::{Mat4, Vec3};
use crate::mesh::Mesh;
use crate::pipeline::{FragmentStage, Pipeline, VertexStage};
use crate::scene::Scene;
use crate::shaders::{
    BlinnPhongUniforms, DirectionalLight, DitherShader, DitherUniforms, FogShader, FogUniforms,
    NormalMappedBlinnPhongShader, NormalMappedBlinnPhongUniforms, NormalsShader, NormalsUniforms,
    PsxShader, PsxUniforms, ShaderPackVaryings, ShaderPackVertex, TextureFilter, ToonShader,
    ToonUniforms, WireframeShader, WireframeUniforms, expand_mesh_with_barycentrics,
};
use crate::skybox::CubeTexture;
use std::path::Path;

#[cfg(target_arch = "wasm32")]
use std::cell::UnsafeCell;

const MAX_DIMENSION: u32 = 2048;
const CAMERA_DISTANCE: f32 = 4.2;

const SHOWCASE_OBJ: &str = include_str!("../assets/multi_material.obj");
const SHOWCASE_MTL: &str = include_str!("../assets/multi_material.mtl");
const ARM_GLTF: &str = include_str!("../assets/arm.gltf");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShowcaseErrorCode {
    InvalidDimensions = -1,
    NotInitialized = -2,
    Render = -3,
    Scene = -4,
}

fn load_scene_bytes_state(state: &mut Option<Scene>, bytes: &[u8]) -> i32 {
    let Ok(source) = std::str::from_utf8(bytes) else {
        return api_error(ShowcaseErrorCode::Scene);
    };
    match Scene::from_str(source) {
        Ok(scene) => {
            *state = Some(scene);
            0
        }
        Err(_) => api_error(ShowcaseErrorCode::Scene),
    }
}

#[derive(Debug)]
struct Showcase {
    framebuffer: Framebuffer,
    rgba: Vec<u8>,
    mesh: Mesh,
    shader_vertices: Vec<ShaderPackVertex>,
    shader_triangles: Vec<[usize; 3]>,
    checker: Texture,
    normal_bump: Texture,
    skybox: CubeTexture,
    gltf: GltfAsset,
    materials: MaterialLibrary,
}

impl Showcase {
    fn new(width: u32, height: u32) -> Result<Self, String> {
        if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err("framebuffer dimensions are outside the supported range".to_string());
        }

        let mut mesh = Mesh::parse(SHOWCASE_OBJ).map_err(|error| error.to_string())?;
        mesh.generate_tangents();
        let (shader_vertices, shader_triangles) = expand_mesh_with_barycentrics(&mesh);
        let checker = Texture::from_qoi(include_bytes!("../assets/checker.qoi"))
            .map_err(|error| error.to_string())?;
        let normal_bump = Texture::from_qoi(include_bytes!("../assets/normal_bump.qoi"))
            .map_err(|error| error.to_string())?;
        let skybox = CubeTexture::new([
            Texture::from_qoi(include_bytes!("../assets/skybox_px.qoi"))
                .map_err(|error| error.to_string())?,
            Texture::from_qoi(include_bytes!("../assets/skybox_nx.qoi"))
                .map_err(|error| error.to_string())?,
            Texture::from_qoi(include_bytes!("../assets/skybox_py.qoi"))
                .map_err(|error| error.to_string())?,
            Texture::from_qoi(include_bytes!("../assets/skybox_ny.qoi"))
                .map_err(|error| error.to_string())?,
            Texture::from_qoi(include_bytes!("../assets/skybox_pz.qoi"))
                .map_err(|error| error.to_string())?,
            Texture::from_qoi(include_bytes!("../assets/skybox_nz.qoi"))
                .map_err(|error| error.to_string())?,
        ])
        .map_err(|error| error.to_string())?;
        let materials = MaterialLibrary::parse(SHOWCASE_MTL).map_err(|error| error.to_string())?;
        let gltf =
            GltfAsset::from_str(ARM_GLTF, Path::new(".")).map_err(|error| error.to_string())?;
        let pixel_count = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "framebuffer dimensions overflow".to_string())?;

        Ok(Self {
            framebuffer: Framebuffer::new(width as usize, height as usize),
            rgba: vec![0; pixel_count * 4],
            mesh,
            shader_vertices,
            shader_triangles,
            checker,
            normal_bump,
            skybox,
            gltf,
            materials,
        })
    }

    fn render(&mut self, time_ms: f64, yaw: f32, pitch: f32, mode: u32) -> Result<(), String> {
        let time_seconds = if time_ms.is_finite() {
            (time_ms / 1000.0).clamp(-1_000_000.0, 1_000_000.0) as f32
        } else {
            0.0
        };
        let yaw = finite_or(yaw, 0.0);
        let pitch = finite_or(pitch, 0.0).clamp(-1.55, 1.55);
        let mode = clamp_mode(mode);
        let aspect = self.framebuffer.width as f32 / self.framebuffer.height as f32;
        let orbit = OrbitController::new(Vec3::ZERO, CAMERA_DISTANCE, yaw, pitch);
        let camera = orbit.camera(1.0, aspect, 0.1, 100.0);
        let view = camera.view_matrix();
        let projection = camera.projection_matrix();
        let model = Mat4::rotate(Vec3::new(0.0, 1.0, 0.0), time_seconds * 0.35);

        self.framebuffer.clear(argb8888(255, 5, 8, 18));
        match mode {
            0 => self.render_blinn_phong(camera, model, view, projection)?,
            1 => self.render_toon(camera, model, view, projection),
            2 => self.render_psx(camera, model, view, projection),
            3 => self.render_dither(camera, model, view, projection),
            4 => self.render_fog(camera, model, view, projection),
            5 => self.render_normals(camera, model, view, projection),
            6 => self.render_wireframe(camera, model, view, projection),
            7 => self.render_gltf(time_seconds, camera, view, projection)?,
            _ => {}
        }
        self.framebuffer.draw_text(
            12,
            12,
            "CHIMY2 WASM / DRAG TO ORBIT",
            1,
            argb8888(255, 240, 244, 255),
        );
        self.framebuffer.draw_text(
            12,
            22,
            "BITMAP FONT OVERLAY",
            1,
            argb8888(255, 255, 196, 96),
        );
        self.copy_rgba();
        Ok(())
    }

    fn render_blinn_phong(
        &mut self,
        camera: Camera,
        model: Mat4,
        view: Mat4,
        projection: Mat4,
    ) -> Result<(), String> {
        let material = self
            .materials
            .get("checker")
            .ok_or_else(|| "showcase MTL has no checker material".to_string())?;
        let directional = DirectionalLight::new(
            Vec3::new(-0.3, -0.8, 1.0).normalize(),
            Vec3::new(1.0, 0.92, 0.82),
        );
        let point = crate::shaders::PointLight::new(
            Vec3::new(1.5, 1.5, 2.8),
            Vec3::new(0.7, 0.8, 1.0),
            1.0,
            0.05,
            0.04,
        );
        let mut lighting = BlinnPhongUniforms::new(
            model,
            view,
            projection,
            Vec3::new(0.04, 0.04, 0.04),
            Vec3::new(0.8, 0.8, 0.8),
            Vec3::new(0.2, 0.2, 0.2),
            32.0,
            camera.position,
            directional,
            point,
        );
        material.apply_to(&mut lighting);
        let uniforms = NormalMappedBlinnPhongUniforms::new(
            lighting,
            &self.checker,
            &self.normal_bump,
            TextureFilter::Bilinear,
        )
        .map_err(str::to_string)?;
        let mut pipeline =
            Pipeline::new(NormalMappedBlinnPhongShader, NormalMappedBlinnPhongShader);
        pipeline.set_thread_count(1);
        let skybox = &self.skybox;
        let mesh = &self.mesh;
        pipeline.render(&mut self.framebuffer, |frame, target| {
            frame.draw_skybox(target, skybox, camera);
            frame.draw_mesh_with_sampling(target, mesh, &uniforms);
        });
        Ok(())
    }

    fn render_toon(&mut self, camera: Camera, model: Mat4, view: Mat4, projection: Mat4) {
        self.render_shader_pack(
            ToonShader,
            ToonShader,
            ToonUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.85, 0.16, 0.08),
                Vec3::new(-0.3, -0.8, 1.0),
            ),
            camera,
        );
    }

    fn render_psx(&mut self, camera: Camera, model: Mat4, view: Mat4, projection: Mat4) {
        self.render_shader_pack(
            PsxShader,
            PsxShader,
            PsxUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.86, 0.25, 0.08),
                (
                    self.framebuffer.width as u32,
                    self.framebuffer.height as u32,
                ),
            ),
            camera,
        );
    }

    fn render_dither(&mut self, camera: Camera, model: Mat4, view: Mat4, projection: Mat4) {
        self.render_shader_pack(
            DitherShader,
            DitherShader,
            DitherUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.95, 0.36, 0.12),
                (
                    self.framebuffer.width as u32,
                    self.framebuffer.height as u32,
                ),
            ),
            camera,
        );
    }

    fn render_fog(&mut self, camera: Camera, model: Mat4, view: Mat4, projection: Mat4) {
        self.render_shader_pack(
            FogShader,
            FogShader,
            FogUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.8, 0.18, 0.08),
                Vec3::new(0.08, 0.13, 0.25),
                2.0,
                5.0,
            ),
            camera,
        );
    }

    fn render_normals(&mut self, camera: Camera, model: Mat4, view: Mat4, projection: Mat4) {
        self.render_shader_pack(
            NormalsShader,
            NormalsShader,
            NormalsUniforms::new(model, view, projection),
            camera,
        );
    }

    fn render_wireframe(&mut self, camera: Camera, model: Mat4, view: Mat4, projection: Mat4) {
        self.render_shader_pack(
            WireframeShader,
            WireframeShader,
            WireframeUniforms::new(
                model,
                view,
                projection,
                Vec3::new(0.08, 0.12, 0.22),
                Vec3::new(0.9, 0.72, 0.22),
            ),
            camera,
        );
    }

    fn render_shader_pack<VS, FS, Uniforms>(
        &mut self,
        vertex: VS,
        fragment: FS,
        uniforms: Uniforms,
        camera: Camera,
    ) where
        VS: VertexStage<ShaderPackVertex, Uniforms, Varyings = ShaderPackVaryings> + Sync,
        FS: FragmentStage<ShaderPackVaryings, Uniforms> + Sync,
        Uniforms: Sync,
    {
        let skybox = &self.skybox;
        let vertices = &self.shader_vertices;
        let triangles = &self.shader_triangles;
        let mut pipeline = Pipeline::new(vertex, fragment);
        pipeline.set_thread_count(1);
        pipeline.render(&mut self.framebuffer, |frame, target| {
            frame.draw_skybox(target, skybox, camera);
            frame.draw(target, vertices, triangles, &uniforms);
        });
    }

    fn render_gltf(
        &mut self,
        time_seconds: f32,
        camera: Camera,
        view: Mat4,
        projection: Mat4,
    ) -> Result<(), String> {
        let mut draws = self
            .gltf
            .scene_draws(0, Some(0), time_seconds)
            .map_err(|error| error.to_string())?;
        let placement = Mat4::translate(Vec3::new(-1.0, -0.9, 0.0));
        for draw in &mut draws {
            draw.model = placement * draw.model;
        }
        crate::skybox::render_skybox(&mut self.framebuffer, camera, &self.skybox);
        submit_gltf_draws(
            &mut self.framebuffer,
            &self.gltf,
            &draws,
            view,
            projection,
            camera.position,
        )
        .map_err(|error| error.to_string())
    }

    fn copy_rgba(&mut self) {
        for (pixel, bytes) in self
            .framebuffer
            .color
            .iter()
            .zip(self.rgba.chunks_exact_mut(4))
        {
            bytes.copy_from_slice(&argb_to_rgba(*pixel));
        }
    }
}

fn argb_to_rgba(pixel: u32) -> [u8; 4] {
    let [alpha, red, green, blue] = pixel.to_be_bytes();
    [red, green, blue, alpha]
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn clamp_mode(mode: u32) -> u32 {
    if mode <= 7 { mode } else { 0 }
}

#[cfg(target_arch = "wasm32")]
struct WasmState(UnsafeCell<Option<Showcase>>);

#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WasmState {}

#[cfg(target_arch = "wasm32")]
static STATE: WasmState = WasmState(UnsafeCell::new(None));

#[cfg(target_arch = "wasm32")]
static SCENE_STATE: WasmSceneState = WasmSceneState(UnsafeCell::new(None));

#[cfg(target_arch = "wasm32")]
struct WasmSceneState(UnsafeCell<Option<Scene>>);

#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WasmSceneState {}

#[cfg(target_arch = "wasm32")]
fn state() -> &'static mut Option<Showcase> {
    // JavaScript calls this API on one thread. The browser contract forbids
    // reentrant calls while a frame is being rendered.
    unsafe { &mut *STATE.0.get() }
}

#[cfg(target_arch = "wasm32")]
fn scene_state() -> &'static mut Option<Scene> {
    unsafe { &mut *SCENE_STATE.0.get() }
}

fn api_error(code: ShowcaseErrorCode) -> i32 {
    code as i32
}

fn init_state(state: &mut Option<Showcase>, width: u32, height: u32) -> i32 {
    match Showcase::new(width, height) {
        Ok(showcase) => {
            *state = Some(showcase);
            0
        }
        Err(_) => api_error(ShowcaseErrorCode::InvalidDimensions),
    }
}

fn render_state(
    state: &mut Option<Showcase>,
    time_ms: f64,
    yaw: f32,
    pitch: f32,
    mode: u32,
) -> i32 {
    let Some(showcase) = state.as_mut() else {
        return api_error(ShowcaseErrorCode::NotInitialized);
    };
    match showcase.render(time_ms, yaw, pitch, mode) {
        Ok(()) => 0,
        Err(_) => api_error(ShowcaseErrorCode::Render),
    }
}

fn framebuffer_ptr_state(state: &Option<Showcase>) -> *const u8 {
    state
        .as_ref()
        .map_or(std::ptr::null(), |showcase| showcase.rgba.as_ptr())
}

fn framebuffer_len_state(state: &Option<Showcase>) -> usize {
    state.as_ref().map_or(0, |showcase| showcase.rgba.len())
}

fn framebuffer_width_state(state: &Option<Showcase>) -> u32 {
    state
        .as_ref()
        .map_or(0, |showcase| showcase.framebuffer.width as u32)
}

fn framebuffer_height_state(state: &Option<Showcase>) -> u32 {
    state
        .as_ref()
        .map_or(0, |showcase| showcase.framebuffer.height as u32)
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn init(width: u32, height: u32) -> i32 {
    init_state(state(), width, height)
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn render_frame(time_ms: f64, yaw: f32, pitch: f32, mode: u32) -> i32 {
    render_state(state(), time_ms, yaw, pitch, mode)
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_ptr() -> *const u8 {
    framebuffer_ptr_state(state())
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_len() -> usize {
    framebuffer_len_state(state())
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_width() -> u32 {
    framebuffer_width_state(state())
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_height() -> u32 {
    framebuffer_height_state(state())
}

/// Loads a scene JSON string from JavaScript memory.
///
/// The caller keeps the bytes alive for the duration of this call.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn load_scene_json(bytes: *const u8, length: usize) -> i32 {
    if bytes.is_null() {
        return api_error(ShowcaseErrorCode::Scene);
    }
    let bytes = unsafe { std::slice::from_raw_parts(bytes, length) };
    load_scene_bytes_state(scene_state(), bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_boundary_reorders_argb_bytes() {
        assert_eq!(argb_to_rgba(0x12_34_56_78), [0x34, 0x56, 0x78, 0x12]);
    }

    #[test]
    fn unknown_mode_renders_like_blinn_phong() {
        let mut unknown = Showcase::new(64, 64).expect("embedded showcase assets are valid");
        let mut default = Showcase::new(64, 64).expect("embedded showcase assets are valid");
        unknown
            .render(350.0, 0.25, -0.1, 42)
            .expect("render succeeds");
        default
            .render(350.0, 0.25, -0.1, 0)
            .expect("render succeeds");
        assert_eq!(unknown.rgba, default.rgba);
    }

    #[test]
    fn state_guards_render_before_init_and_reject_invalid_dimensions() {
        let mut state = None;
        assert_eq!(render_state(&mut state, 0.0, 0.0, 0.0, 0), -2);
        assert_eq!(framebuffer_len_state(&state), 0);
        assert_eq!(init_state(&mut state, 0, 64), -1);
        assert!(state.is_none());
        assert_eq!(init_state(&mut state, 2049, 64), -1);
        assert!(state.is_none());

        assert_eq!(init_state(&mut state, 64, 64), 0);
        assert_eq!(render_state(&mut state, 0.0, 0.0, 0.0, 0), 0);
        assert_ne!(framebuffer_ptr_state(&state), std::ptr::null());

        let state_snapshot = |state: &Option<Showcase>| {
            state.as_ref().map(|showcase| {
                (
                    framebuffer_ptr_state(state),
                    framebuffer_len_state(state),
                    framebuffer_width_state(state),
                    framebuffer_height_state(state),
                    showcase.rgba.clone(),
                )
            })
        };
        let snapshot = state_snapshot(&state);
        assert_eq!(init_state(&mut state, 0, 64), -1);
        assert_eq!(state_snapshot(&state), snapshot);
        assert_eq!(init_state(&mut state, 2049, 64), -1);
        assert_eq!(state_snapshot(&state), snapshot);

        assert_eq!(init_state(&mut state, 32, 24), 0);
        assert_eq!(framebuffer_width_state(&state), 32);
        assert_eq!(framebuffer_height_state(&state), 24);
    }

    #[test]
    fn scene_json_boundary_uses_the_strict_loader() {
        let mut scene = None;
        let valid =
            br#"{"camera":{"position":[0,0,5],"target":[0,0,0],"fov":1,"near":0.1,"far":10}}"#;
        assert_eq!(load_scene_bytes_state(&mut scene, valid), 0);
        assert!(scene.is_some());
        assert_eq!(
            load_scene_bytes_state(&mut scene, br#"{"camera":{"position":[0,0,5]}}"#),
            -4
        );
        assert!(scene.is_some());
    }

    #[test]
    fn headless_render_is_deterministic() {
        let mut left = Showcase::new(64, 64).expect("embedded showcase assets are valid");
        let mut right = Showcase::new(64, 64).expect("embedded showcase assets are valid");
        left.render(350.0, 0.25, -0.1, 0).expect("render succeeds");
        right.render(350.0, 0.25, -0.1, 0).expect("render succeeds");
        assert_eq!(left.rgba, right.rgba);
    }
}
