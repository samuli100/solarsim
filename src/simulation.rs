use glam::Vec3;
use rand::Rng;
use crate::types::*;

/// High-level simulation state
pub struct Simulation {
    pub params: SimParams,
    pub bodies: Vec<GpuCelestialBody>,
    pub particles: Vec<GpuParticle>,
    pub num_alive_particles: u32,
    pub time: f32,
    pub paused: bool,
    pub time_scale: f32,

    // Interaction
    pub target_pos: Option<Vec3>,
    pub spawn_mode: SpawnMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpawnMode {
    Swarm,      // Spawn swarm particles with boids behavior
    Free,       // Spawn free particles (just gravity)
    Burst,      // Spawn a burst of particles
}

impl Simulation {
    pub fn new(bodies: Vec<GpuCelestialBody>) -> Self {
        let num_bodies = bodies.len();
        let mut params = SimParams::default();
        params.num_bodies = num_bodies as u32;
        params.num_particles = MAX_PARTICLES as u32;

        Self {
            params,
            bodies,
            particles: vec![GpuParticle::dead(); MAX_PARTICLES],
            num_alive_particles: 0,
            time: 0.0,
            paused: false,
            time_scale: 1.0,
            target_pos: None,
            spawn_mode: SpawnMode::Swarm,
        }
    }

    /// Find the next dead particle slot
    fn find_dead_slot(&self) -> Option<usize> {
        self.particles.iter().position(|p| p.data[3] < 0.5)
    }

    /// Spawn a single particle
    pub fn spawn_particle(&mut self, pos: Vec3, vel: Vec3) {
        if let Some(idx) = self.find_dead_slot() {
            let particle = match self.spawn_mode {
                SpawnMode::Swarm => GpuParticle::new_swarm(pos, vel, 0.1),
                SpawnMode::Free => GpuParticle::new_free(pos, vel, 0.1),
                SpawnMode::Burst => GpuParticle::new_free(pos, vel, 0.05),
            };
            self.particles[idx] = particle;
            self.num_alive_particles += 1;
        }
    }

    /// Spawn a burst of particles around a position
    pub fn spawn_burst(&mut self, center: Vec3, count: usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let offset = Vec3::new(
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.1..0.1),
            );
            let vel = Vec3::new(
                rng.gen_range(-0.5..0.5),
                rng.gen_range(-0.5..0.5),
                rng.gen_range(-0.5..0.5),
            );
            self.spawn_particle(center + offset, vel);
        }
    }

    /// Spawn a swarm formation around a position
    pub fn spawn_swarm(&mut self, center: Vec3, count: usize) {
        let mut rng = rand::thread_rng();
        let old_mode = self.spawn_mode;
        self.spawn_mode = SpawnMode::Swarm;

        for _ in 0..count {
            let offset = Vec3::new(
                rng.gen_range(-0.15..0.15),
                rng.gen_range(-0.05..0.05),
                rng.gen_range(-0.15..0.15),
            );
            let vel = Vec3::new(
                rng.gen_range(-0.3..0.3),
                rng.gen_range(-0.15..0.15),
                rng.gen_range(-0.3..0.3),
            );
            self.spawn_particle(center + offset, vel);
        }

        self.spawn_mode = old_mode;
    }

    /// Set the swarm target position
    pub fn set_target(&mut self, pos: Vec3) {
        self.target_pos = Some(pos);
        self.params.target_x = pos.x;
        self.params.target_y = pos.y;
        self.params.target_z = pos.z;
        self.params.target_active = 1.0;
    }

    /// Clear the swarm target
    pub fn clear_target(&mut self) {
        self.target_pos = None;
        self.params.target_active = 0.0;
    }

    /// Update simulation parameters for this frame
    pub fn update_params(&mut self, dt: f32) {
        if self.paused {
            self.params.dt = 0.0;
        } else {
            self.params.dt = dt * self.time_scale;
        }
        self.time += self.params.dt;
        self.params.time = self.time;

        // Count alive particles
        self.num_alive_particles = self.particles.iter().filter(|p| p.data[3] > 0.5).count() as u32;
    }

    /// Kill all particles
    pub fn clear_particles(&mut self) {
        for p in &mut self.particles {
            p.data[3] = 0.0;
        }
        self.num_alive_particles = 0;
    }

    /// Spawn particles in an orbit around a body
    pub fn spawn_orbital_swarm(&mut self, body_index: usize, count: usize, orbit_radius: f32) {
        if body_index >= self.bodies.len() {
            return;
        }

        let body = &self.bodies[body_index];
        let body_pos = Vec3::new(body.position[0], body.position[1], body.position[2]);
        let body_mass = body.position[3];
        let g = self.params.gravitational_constant;

        let mut rng = rand::thread_rng();
        let old_mode = self.spawn_mode;
        self.spawn_mode = SpawnMode::Swarm;

        for i in 0..count {
            let angle = (i as f32 / count as f32) * std::f32::consts::TAU
                + rng.gen_range(-0.1..0.1);
            let r = orbit_radius + rng.gen_range(-0.02..0.02);

            let pos = body_pos + Vec3::new(r * angle.cos(), rng.gen_range(-0.01..0.01), r * angle.sin());

            // Orbital velocity
            let orbital_speed = (g * body_mass / r).sqrt() * 0.3; // slower than full orbit
            let vel = Vec3::new(-angle.sin(), 0.0, angle.cos()) * orbital_speed;

            self.spawn_particle(pos, vel);
        }

        self.spawn_mode = old_mode;
    }
}
