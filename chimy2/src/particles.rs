//! Deterministic CPU particles and camera-facing billboard submission.
//!
//! A particle step uses a fixed one-sixtieth second timestep. Emission
//! variation comes from an integer hash of the spawn index. No random state or
//! unordered collection is part of the simulation.

use crate::camera::Camera;
use crate::fb::Framebuffer;
use crate::math::{Mat4, Vec2, Vec3, Vec4};
use crate::mesh::{Mesh, MeshVertex};
use crate::pipeline::{Instance, RenderFrame};
use crate::shaders::{TexturedShader, TexturedUniforms};

pub const PARTICLE_DT: f32 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleEmitter {
    position: Vec3,
    emission_rate: usize,
    lifetime_steps: usize,
    initial_velocity: Vec3,
    velocity_variation: Vec3,
    gravity: Vec3,
    drag: f32,
}

impl ParticleEmitter {
    pub fn new(
        position: Vec3,
        emission_rate: usize,
        lifetime_steps: usize,
        initial_velocity: Vec3,
    ) -> Self {
        let mut emitter = Self {
            position: Vec3::ZERO,
            emission_rate: 0,
            lifetime_steps: 1,
            initial_velocity: Vec3::ZERO,
            velocity_variation: Vec3::ZERO,
            gravity: Vec3::ZERO,
            drag: 0.0,
        };
        emitter.set_position(position);
        emitter.set_emission_rate(emission_rate);
        emitter.set_lifetime_steps(lifetime_steps);
        emitter.set_initial_velocity(initial_velocity);
        emitter
    }

    pub const fn position(&self) -> Vec3 {
        self.position
    }

    pub const fn emission_rate(&self) -> usize {
        self.emission_rate
    }

    pub const fn lifetime_steps(&self) -> usize {
        self.lifetime_steps
    }

    pub const fn initial_velocity(&self) -> Vec3 {
        self.initial_velocity
    }

    pub const fn velocity_variation(&self) -> Vec3 {
        self.velocity_variation
    }

    pub const fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub const fn drag(&self) -> f32 {
        self.drag
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = finite_vec3_or(position, Vec3::ZERO);
    }

    pub fn set_emission_rate(&mut self, emission_rate: usize) {
        self.emission_rate = emission_rate;
    }

    pub fn set_lifetime_steps(&mut self, lifetime_steps: usize) {
        self.lifetime_steps = lifetime_steps.max(1);
    }

    pub fn set_initial_velocity(&mut self, velocity: Vec3) {
        self.initial_velocity = finite_vec3_or(velocity, Vec3::ZERO);
    }

    pub fn set_velocity_variation(&mut self, variation: Vec3) {
        self.velocity_variation = finite_vec3_or(variation, Vec3::ZERO);
    }

    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = finite_vec3_or(gravity, Vec3::ZERO);
    }

    /// Sets drag as the fraction of velocity removed after gravity.
    pub fn set_drag(&mut self, drag: f32) {
        self.drag = if drag.is_finite() {
            drag.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self::new(Vec3::ZERO, 0, 60, Vec3::ZERO)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub age_steps: usize,
    pub lifetime_steps: usize,
    pub spawn_index: u64,
}

impl Particle {
    pub const fn alpha(&self) -> f32 {
        particle_alpha(self.age_steps, self.lifetime_steps)
    }
}

/// Returns the documented linear fade: `1 - age / lifetime`.
pub const fn particle_alpha(age_steps: usize, lifetime_steps: usize) -> f32 {
    if lifetime_steps == 0 {
        0.0
    } else {
        (1.0 - age_steps as f32 / lifetime_steps as f32).max(0.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSystem {
    emitter: ParticleEmitter,
    slots: Vec<Option<Particle>>,
    step_index: u64,
    next_spawn_index: u64,
}

impl ParticleSystem {
    pub fn new(emitter: ParticleEmitter, capacity: usize) -> Self {
        Self {
            emitter,
            slots: vec![None; capacity],
            step_index: 0,
            next_spawn_index: 0,
        }
    }

    pub fn from_emitter(emitter: ParticleEmitter, capacity: usize) -> Self {
        Self::new(emitter, capacity)
    }

    pub const fn emitter(&self) -> ParticleEmitter {
        self.emitter
    }

    pub fn set_emitter(&mut self, emitter: ParticleEmitter) {
        self.emitter = emitter;
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub const fn step_index(&self) -> u64 {
        self.step_index
    }

    pub fn live_count(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn slots(&self) -> &[Option<Particle>] {
        &self.slots
    }

    pub fn particle(&self, slot: usize) -> Option<Particle> {
        self.slots.get(slot).copied().flatten()
    }

    pub fn live_particles(&self) -> impl Iterator<Item = (usize, &Particle)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, particle)| particle.as_ref().map(|particle| (slot, particle)))
    }

    pub fn reset(&mut self) {
        self.slots.fill(None);
        self.step_index = 0;
        self.next_spawn_index = 0;
    }

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Advances exactly one fixed timestep.
    pub fn step(&mut self) {
        let gravity = self.emitter.gravity();
        let drag = 1.0 - self.emitter.drag();
        let lifetime = self.emitter.lifetime_steps();

        for slot in &mut self.slots {
            let Some(particle) = slot.as_mut() else {
                continue;
            };

            // Semi-implicit Euler order is intentional and is part of the
            // deterministic physics contract: move, then apply acceleration.
            particle.position = particle.position + particle.velocity * PARTICLE_DT;
            particle.velocity = (particle.velocity + gravity * PARTICLE_DT) * drag;
            particle.age_steps += 1;
            if particle.age_steps >= lifetime {
                *slot = None;
            }
        }

        for _ in 0..self.emitter.emission_rate() {
            let Some(slot) = self.slots.iter().position(Option::is_none) else {
                break;
            };
            let spawn_index = self.next_spawn_index;
            self.next_spawn_index = self.next_spawn_index.wrapping_add(1);
            self.slots[slot] = Some(Particle {
                position: self.emitter.position(),
                velocity: self.spawn_velocity(spawn_index),
                age_steps: 0,
                lifetime_steps: lifetime,
                spawn_index,
            });
        }
        self.step_index = self.step_index.wrapping_add(1);
    }

    /// Encodes every slot and counter with explicit little-endian fields.
    pub fn state_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32 + self.slots.len() * 40);
        push_u64(&mut bytes, self.step_index);
        push_u64(&mut bytes, self.next_spawn_index);
        push_u64(&mut bytes, self.slots.len() as u64);
        for slot in &self.slots {
            match slot {
                None => bytes.push(0),
                Some(particle) => {
                    bytes.push(1);
                    for value in [
                        particle.position.x,
                        particle.position.y,
                        particle.position.z,
                        particle.velocity.x,
                        particle.velocity.y,
                        particle.velocity.z,
                    ] {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    push_u64(&mut bytes, particle.age_steps as u64);
                    push_u64(&mut bytes, particle.lifetime_steps as u64);
                    push_u64(&mut bytes, particle.spawn_index);
                }
            }
        }
        bytes
    }

    pub fn billboard_instances(&self, camera: Camera, size: f32, tint: Vec4) -> Vec<Instance> {
        let size = if size.is_finite() { size.max(0.0) } else { 0.0 };
        let tint = sanitize_tint(tint);
        let view = camera.view_matrix();
        self.live_particles()
            .map(|(_, particle)| {
                let mut instance_tint = tint;
                instance_tint.w *= particle.alpha();
                Instance::with_tint(
                    billboard_model(view, particle.position, size),
                    instance_tint,
                )
            })
            .collect()
    }

    /// Submits all live particles through the existing instanced sampling path.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_instanced<'a>(
        &self,
        frame: &mut RenderFrame<'a, TexturedShader, TexturedShader>,
        target: &Framebuffer,
        quad: &Mesh,
        uniforms: &'a mut TexturedUniforms<'a>,
        camera: Camera,
        size: f32,
        tint: Vec4,
    ) {
        uniforms.set_model_view(camera.view_matrix());
        let instances = self.billboard_instances(camera, size, tint);
        frame.draw_mesh_instanced_with_sampling(target, quad, uniforms, &instances);
    }

    fn spawn_velocity(&self, spawn_index: u64) -> Vec3 {
        let base = self.emitter.initial_velocity();
        let variation = self.emitter.velocity_variation();
        Vec3::new(
            base.x + variation.x * signed_hash(spawn_index, 0),
            base.y + variation.y * signed_hash(spawn_index, 1),
            base.z + variation.z * signed_hash(spawn_index, 2),
        )
    }
}

/// Builds the shared unit quad used by particle submissions.
pub fn billboard_quad() -> Mesh {
    let vertices = vec![
        MeshVertex::new(Vec3::new(-0.5, -0.5, 0.0), Some(Vec2::new(0.0, 1.0)), None),
        MeshVertex::new(Vec3::new(0.5, -0.5, 0.0), Some(Vec2::new(1.0, 1.0)), None),
        MeshVertex::new(Vec3::new(0.5, 0.5, 0.0), Some(Vec2::new(1.0, 0.0)), None),
        MeshVertex::new(Vec3::new(-0.5, 0.5, 0.0), Some(Vec2::new(0.0, 0.0)), None),
    ];
    Mesh::new(vertices, vec![[0, 1, 2], [0, 2, 3]])
}

/// Returns the camera-facing model matrix used by particle instances.
pub fn billboard_model(view: Mat4, position: Vec3, size: f32) -> Mat4 {
    let right = Vec3::new(view.data[0], view.data[4], view.data[8]);
    let up = Vec3::new(view.data[1], view.data[5], view.data[9]);
    let forward = Vec3::new(-view.data[2], -view.data[6], -view.data[10]);
    let size = if size.is_finite() { size.max(0.0) } else { 0.0 };
    Mat4::new([
        right.x * size,
        right.y * size,
        right.z * size,
        0.0,
        up.x * size,
        up.y * size,
        up.z * size,
        0.0,
        forward.x * size,
        forward.y * size,
        forward.z * size,
        0.0,
        position.x,
        position.y,
        position.z,
        1.0,
    ])
}

fn finite_vec3_or(value: Vec3, fallback: Vec3) -> Vec3 {
    if value.x.is_finite() && value.y.is_finite() && value.z.is_finite() {
        value
    } else {
        fallback
    }
}

fn sanitize_tint(tint: Vec4) -> Vec4 {
    Vec4::new(
        tint_component(tint.x),
        tint_component(tint.y),
        tint_component(tint.z),
        tint_component(tint.w),
    )
}

fn tint_component(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn signed_hash(index: u64, lane: u64) -> f32 {
    let mut value = (index as u32).wrapping_add((lane as u32).wrapping_mul(0x9e37_79b9));
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    let unit = (value >> 8) as f32 / 16_777_215.0;
    unit * 2.0 - 1.0
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fb::Framebuffer;
    use crate::math::Quat;
    use crate::pipeline::{InstanceUniforms, Pipeline};
    use crate::shaders::{TextureFilter, TexturedShader, TexturedUniforms};

    fn emitter() -> ParticleEmitter {
        let mut emitter = ParticleEmitter::new(Vec3::ZERO, 1, 8, Vec3::new(1.0, 2.0, 0.0));
        emitter.set_gravity(Vec3::new(0.0, -3.0, 0.0));
        emitter.set_velocity_variation(Vec3::new(0.2, 0.1, 0.2));
        emitter
    }

    #[test]
    fn simulation_is_byte_deterministic() {
        let mut left = ParticleSystem::new(emitter(), 32);
        let mut right = ParticleSystem::new(emitter(), 32);
        left.step_n(100);
        right.step_n(100);
        assert_eq!(left.state_bytes(), right.state_bytes());
    }

    #[test]
    fn gravity_uses_move_then_accelerate_order() {
        let mut system = ParticleSystem::new(
            ParticleEmitter::new(Vec3::ZERO, 1, 10, Vec3::new(2.0, 3.0, 0.0)),
            1,
        );
        system.emitter.set_gravity(Vec3::new(0.0, -6.0, 0.0));
        system.emitter.set_emission_rate(0);
        system.slots[0] = Some(Particle {
            position: Vec3::ZERO,
            velocity: Vec3::new(2.0, 3.0, 0.0),
            age_steps: 0,
            lifetime_steps: 10,
            spawn_index: 0,
        });
        system.step();
        let particle = system.particle(0).unwrap();
        assert!((particle.position.x - 2.0 * PARTICLE_DT).abs() < 1e-6);
        assert!((particle.position.y - 3.0 * PARTICLE_DT).abs() < 1e-6);
        assert!((particle.velocity.y - (3.0 - 6.0 * PARTICLE_DT)).abs() < 1e-6);
    }

    #[test]
    fn lifetime_recycles_the_lowest_slot() {
        let emitter = ParticleEmitter::new(Vec3::ZERO, 1, 2, Vec3::ZERO);
        let mut system = ParticleSystem::new(emitter, 1);
        system.step();
        assert_eq!(system.live_count(), 1);
        assert_eq!(system.particle(0).unwrap().age_steps, 0);
        system.step();
        assert_eq!(system.live_count(), 1);
        assert_eq!(system.particle(0).unwrap().age_steps, 1);
        system.step();
        assert_eq!(system.live_count(), 1);
        assert_eq!(system.particle(0).unwrap().spawn_index, 1);
    }

    #[test]
    fn billboard_axes_come_from_the_view_matrix() {
        let view = Mat4::look_at(
            Vec3::new(2.0, 1.0, 4.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        );
        let model = billboard_model(view, Vec3::new(1.0, 2.0, 3.0), 2.0);
        let right = Vec3::new(model.data[0], model.data[1], model.data[2]);
        let up = Vec3::new(model.data[4], model.data[5], model.data[6]);
        assert_eq!(
            right,
            Vec3::new(view.data[0], view.data[4], view.data[8]) * 2.0
        );
        assert_eq!(
            up,
            Vec3::new(view.data[1], view.data[5], view.data[9]) * 2.0
        );
        assert_eq!(
            model * Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 2.0, 3.0, 1.0)
        );
    }

    #[test]
    fn alpha_fade_has_one_application() {
        assert_eq!(particle_alpha(0, 10), 1.0);
        assert_eq!(particle_alpha(5, 10), 0.5);
        assert!((particle_alpha(9, 10) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn malformed_emitter_values_are_sanitized_immediately() {
        let mut emitter = ParticleEmitter::default();
        emitter.set_position(Vec3::new(f32::NAN, 1.0, 2.0));
        emitter.set_gravity(Vec3::new(1.0, f32::INFINITY, 2.0));
        emitter.set_drag(f32::NAN);
        emitter.set_lifetime_steps(0);
        assert_eq!(emitter.position(), Vec3::ZERO);
        assert_eq!(emitter.gravity(), Vec3::ZERO);
        assert_eq!(emitter.drag(), 0.0);
        assert_eq!(emitter.lifetime_steps(), 1);
    }

    #[test]
    fn billboard_helper_uses_camera_orientation() {
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 2.0),
            Quat::IDENTITY,
            1.0,
            1.0,
            0.1,
            10.0,
        );
        let model = billboard_model(camera.view_matrix(), Vec3::ZERO, 1.0);
        assert_eq!(model.data[8], 0.0);
        assert_eq!(model.data[10], -1.0);
    }

    #[test]
    fn transparent_particles_match_equivalent_individual_submission() {
        let mut emitter = ParticleEmitter::new(Vec3::ZERO, 3, 20, Vec3::new(0.0, 0.0, -2.0));
        emitter.set_velocity_variation(Vec3::new(0.4, 0.4, 0.0));
        let mut system = ParticleSystem::new(emitter, 8);
        system.step();
        let camera = Camera::new(Vec3::ZERO, Quat::IDENTITY, 1.0, 1.0, 0.1, 10.0);
        let quad = billboard_quad();
        let texture = crate::image::Texture::new(1, 1, vec![[255, 255, 255, 255]]).unwrap();
        let instances = system.billboard_instances(camera, 0.3, Vec4::new(1.0, 1.0, 1.0, 0.6));
        let view = camera.view_matrix();
        let projection = Mat4::IDENTITY;
        let base = TexturedUniforms::new(projection * view, &texture, TextureFilter::Nearest);

        let mut instanced = Framebuffer::new(32, 32);
        let mut instanced_uniforms = base;
        let mut instanced_pipeline = Pipeline::new(TexturedShader, TexturedShader);
        instanced_pipeline.set_culling_enabled(false);
        instanced_pipeline.render(&mut instanced, |frame, target| {
            system.draw_instanced(
                frame,
                target,
                &quad,
                &mut instanced_uniforms,
                camera,
                0.3,
                Vec4::new(1.0, 1.0, 1.0, 0.6),
            );
        });

        let individual_uniforms = instances
            .iter()
            .map(|instance| base.for_instance(instance))
            .collect::<Vec<_>>();
        let mut individual = Framebuffer::new(32, 32);
        let mut individual_pipeline = Pipeline::new(TexturedShader, TexturedShader);
        individual_pipeline.set_culling_enabled(false);
        individual_pipeline.render(&mut individual, |frame, target| {
            for uniforms in &individual_uniforms {
                frame.draw_mesh_with_sampling(target, &quad, uniforms);
            }
        });
        assert_eq!(instanced.color, individual.color);
        assert_eq!(instanced.depth, individual.depth);
    }
}
