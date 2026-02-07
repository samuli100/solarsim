use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// GPU particle data - must match WGSL struct layout exactly
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 4], // xyz = position, w = mass
    pub velocity: [f32; 4], // xyz = velocity, w = particle_type (0=free, 1=swarm)
    pub color: [f32; 4],    // rgba
    pub data: [f32; 4],     // x = radius, y = is_planet, z = trail_timer, w = alive
}

impl GpuParticle {
    pub fn new_swarm(pos: Vec3, vel: Vec3, mass: f32) -> Self {
        Self {
            position: [pos.x, pos.y, pos.z, mass],
            velocity: [vel.x, vel.y, vel.z, 1.0], // 1.0 = swarm type
            color: [0.4, 0.7, 1.0, 0.9],
            data: [0.008, 0.0, 0.0, 1.0], // radius, not_planet, trail, alive
        }
    }

    pub fn new_free(pos: Vec3, vel: Vec3, mass: f32) -> Self {
        Self {
            position: [pos.x, pos.y, pos.z, mass],
            velocity: [vel.x, vel.y, vel.z, 0.0], // 0.0 = free type
            color: [1.0, 0.8, 0.3, 0.8],
            data: [0.006, 0.0, 0.0, 1.0],
        }
    }

    pub fn dead() -> Self {
        Self {
            position: [0.0; 4],
            velocity: [0.0; 4],
            color: [0.0; 4],
            data: [0.0, 0.0, 0.0, 0.0], // w = 0 means dead
        }
    }
}

/// GPU celestial body data - must match WGSL struct layout
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCelestialBody {
    pub position: [f32; 4], // xyz = position, w = mass
    pub velocity: [f32; 4], // xyz = velocity, w = radius
    pub color: [f32; 4],    // rgba
    pub data: [f32; 4],     // x = is_star, y = orbital_speed
}

/// Simulation parameters uniform - must match WGSL
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParams {
    pub dt: f32,
    pub gravitational_constant: f32,
    pub num_particles: u32,
    pub num_bodies: u32,
    pub separation_radius: f32,
    pub alignment_radius: f32,
    pub cohesion_radius: f32,
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
    pub max_speed: f32,
    pub max_force: f32,
    pub target_x: f32,
    pub target_y: f32,
    pub target_z: f32,
    pub target_active: f32,
    pub softening: f32,
    pub damping: f32,
    pub swarm_gravity_weight: f32,
    pub time: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            dt: 0.016,
            gravitational_constant: 39.4784176, // 4 * pi^2 in AU^3/(M_sun * yr^2)
            num_particles: 0,
            num_bodies: 0,
            separation_radius: 0.1,
            alignment_radius: 0.3,
            cohesion_radius: 0.5,
            separation_weight: 1.8,
            alignment_weight: 1.0,
            cohesion_weight: 1.2,
            max_speed: 10.0,
            max_force: 5.0,
            target_x: 0.0,
            target_y: 0.0,
            target_z: 0.0,
            target_active: 0.0,
            softening: 0.01,
            damping: 1.0, // Changed from 0.999 to 1.0 to prevent energy loss
            swarm_gravity_weight: 0.3,
            time: 0.0,
        }
    }
}

/// Camera uniform data
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub eye_pos: [f32; 4],
    pub screen_size: [f32; 4],
}

/// Grid vertex for orbit lines
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridVertex {
    pub position: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

/// Maximum particles we can simulate
pub const MAX_PARTICLES: usize = 65536;
/// Maximum celestial bodies
pub const MAX_BODIES: usize = 32;