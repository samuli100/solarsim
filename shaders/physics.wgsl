// ============================================================================
// GPU Compute Shader: Gravity + Swarm Behavior
// Runs N-body gravitational simulation and boids-like swarm AI on the GPU
// ============================================================================

struct Particle {
    position: vec4<f32>,   // xyz = position, w = mass
    velocity: vec4<f32>,   // xyz = velocity, w = particle_type (0=free, 1=swarm)
    color: vec4<f32>,      // rgba
    data: vec4<f32>,       // x = radius, y = is_planet (1.0), z = trail_timer, w = alive
};

struct CelestialBody {
    position: vec4<f32>,   // xyz = position, w = mass
    velocity: vec4<f32>,   // xyz = velocity, w = radius
    color: vec4<f32>,      // rgba
    data: vec4<f32>,       // x = is_star, y = orbital_speed, z = unused, w = unused
};

// Matches GridVertex in Rust
struct TrailVertex {
    position_pad: vec4<f32>, // xyz = pos, w = pad
    color: vec4<f32>,
};

struct SimParams {
    dt: f32,
    gravitational_constant: f32,
    num_particles: u32,
    num_bodies: u32,
    // Swarm parameters
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    separation_weight: f32,
    alignment_weight: f32,
    cohesion_weight: f32,
    max_speed: f32,
    max_force: f32,
    // Target/control
    target_x: f32,
    target_y: f32,
    target_z: f32,
    target_active: f32,
    // Softening factor to prevent singularities
    softening: f32,
    damping: f32,
    swarm_gravity_weight: f32,
    time: f32,
};

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<storage, read> bodies: array<CelestialBody>;
@group(0) @binding(3) var<uniform> params: SimParams;

// Compute gravitational acceleration from all celestial bodies
fn compute_gravity(pos: vec3<f32>) -> vec3<f32> {
    var accel = vec3<f32>(0.0, 0.0, 0.0);
    let num_bodies = params.num_bodies;

    for (var i = 0u; i < num_bodies; i = i + 1u) {
        let body_pos = bodies[i].position.xyz;
        let body_mass = bodies[i].position.w;

        let diff = body_pos - pos;
        let dist_sq = dot(diff, diff) + params.softening * params.softening;
        let dist = sqrt(dist_sq);
        let inv_dist3 = 1.0 / (dist * dist_sq);

        accel += diff * (params.gravitational_constant * body_mass * inv_dist3);
    }

    return accel;
}

// Boids-like swarm behavior
fn compute_swarm(index: u32, pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    var separation = vec3<f32>(0.0);
    var alignment = vec3<f32>(0.0);
    var cohesion = vec3<f32>(0.0);

    var sep_count = 0u;
    var align_count = 0u;
    var coh_count = 0u;

    let num = params.num_particles;

    for (var i = 0u; i < num; i = i + 1u) {
        if (i == index) { continue; }

        let other = particles_in[i];
        if (other.data.w < 0.5) { continue; } // skip dead particles
        if (other.velocity.w < 0.5) { continue; } // skip non-swarm particles

        let other_pos = other.position.xyz;
        let other_vel = other.velocity.xyz;
        let diff = pos - other_pos;
        let dist = length(diff);

        // Separation: steer away from nearby particles
        if (dist < params.separation_radius && dist > 0.001) {
            separation += normalize(diff) / dist;
            sep_count += 1u;
        }

        // Alignment: match velocity of nearby particles
        if (dist < params.alignment_radius) {
            alignment += other_vel;
            align_count += 1u;
        }

        // Cohesion: steer towards center of nearby particles
        if (dist < params.cohesion_radius) {
            cohesion += other_pos;
            coh_count += 1u;
        }
    }

    var force = vec3<f32>(0.0);

    if (sep_count > 0u) {
        separation /= f32(sep_count);
        if (length(separation) > 0.0) {
            separation = normalize(separation) * params.max_speed - vel;
            separation = clamp_length(separation, params.max_force);
        }
        force += separation * params.separation_weight;
    }

    if (align_count > 0u) {
        alignment /= f32(align_count);
        if (length(alignment) > 0.0) {
            alignment = normalize(alignment) * params.max_speed - vel;
            alignment = clamp_length(alignment, params.max_force);
        }
        force += alignment * params.alignment_weight;
    }

    if (coh_count > 0u) {
        cohesion /= f32(coh_count);
        cohesion = cohesion - pos; // steer towards center
        if (length(cohesion) > 0.0) {
            cohesion = normalize(cohesion) * params.max_speed - vel;
            cohesion = clamp_length(cohesion, params.max_force);
        }
        force += cohesion * params.cohesion_weight;
    }

    // Steer towards target if active
    if (params.target_active > 0.5) {
        let goal_pos = vec3<f32>(params.target_x, params.target_y, params.target_z);
        var seek = goal_pos - pos;
        if (length(seek) > 0.0) {
            seek = normalize(seek) * params.max_speed - vel;
            seek = clamp_length(seek, params.max_force * 2.0);
        }
        force += seek * 1.5;
    }

    return force;
}

fn clamp_length(v: vec3<f32>, max_len: f32) -> vec3<f32> {
    let len = length(v);
    if (len > max_len && len > 0.0) {
        return v * (max_len / len);
    }
    return v;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_particles) { return; }

    var particle = particles_in[index];

    // Skip dead particles
    if (particle.data.w < 0.5) {
        particles_out[index] = particle;
        return;
    }

    let pos = particle.position.xyz;
    let vel = particle.velocity.xyz;
    let mass = particle.position.w;
    let is_swarm = particle.velocity.w > 0.5;
    let dt = params.dt;

    // Compute gravitational acceleration
    var accel = compute_gravity(pos);

    // Add swarm forces for swarm particles
    if (is_swarm) {
        let swarm_force = compute_swarm(index, pos, vel);
        accel += swarm_force / max(mass, 0.01);

        // Reduce gravity influence for swarm particles (they have thrusters!)
        accel = compute_gravity(pos) * params.swarm_gravity_weight + swarm_force / max(mass, 0.01);
    }

    // Symplectic Euler integration (better energy conservation)
    var new_vel = vel + accel * dt;

    // Apply speed limit for swarm particles
    if (is_swarm) {
        new_vel = clamp_length(new_vel, params.max_speed);
    }

    // Apply damping
    new_vel *= params.damping;

    let new_pos = pos + new_vel * dt;

    // Check collision with celestial bodies
    var alive = particle.data.w;
    for (var i = 0u; i < params.num_bodies; i = i + 1u) {
        let body_pos = bodies[i].position.xyz;
        let body_radius = bodies[i].velocity.w;
        let dist = length(new_pos - body_pos);
        if (dist < body_radius * 1.1) {
            alive = 0.0; // destroyed on collision
        }
    }

    // Despawn if too far from origin
    if (length(new_pos) > 100.0) {
        alive = 0.0;
    }

    // Write output
    particle.position = vec4<f32>(new_pos, mass);
    particle.velocity = vec4<f32>(new_vel, particle.velocity.w);
    particle.data.w = alive;
    particle.data.z += dt; // trail timer

    // Color based on speed for swarm particles
    if (is_swarm) {
        let speed = length(new_vel);
        let speed_ratio = clamp(speed / params.max_speed, 0.0, 1.0);
        // Blue (slow) -> Cyan -> White (fast)
        particle.color = vec4<f32>(
            0.3 + speed_ratio * 0.7,
            0.6 + speed_ratio * 0.4,
            1.0,
            0.8 + speed_ratio * 0.2
        );
    }

    particles_out[index] = particle;
}

// ============================================================================
// Celestial body orbital compute shader
// ============================================================================

@group(0) @binding(0) var<storage, read> bodies_in: array<CelestialBody>;
@group(0) @binding(1) var<storage, read_write> bodies_out_buf: array<CelestialBody>;
@group(0) @binding(2) var<uniform> orbit_params: SimParams;
@group(0) @binding(3) var<storage, read_write> trails: array<TrailVertex>;

@compute @workgroup_size(32)
fn cs_orbit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= orbit_params.num_bodies) { return; }

    var body = bodies_in[index];
    let TRAIL_LENGTH = 512u;

    // 1. UPDATE PHYSICS

    // Stars don't move
    if (body.data.x > 0.5) {
        bodies_out_buf[index] = body;
        return;
    }

    let pos = body.position.xyz;
    let vel = body.velocity.xyz;
    let mass = body.position.w;
    let dt = orbit_params.dt;

    // HARDCODED G: Locks planetary orbits so user tuning of swarm gravity
    // doesn't cause planets to fly off into deep space.
    let G = 39.4784176;

    // Compute gravity from other bodies
    var accel = vec3<f32>(0.0);
    for (var i = 0u; i < orbit_params.num_bodies; i = i + 1u) {
        if (i == index) { continue; }
        let other_pos = bodies_in[i].position.xyz;
        let other_mass = bodies_in[i].position.w;
        let diff = other_pos - pos;
        let dist_sq = dot(diff, diff) + orbit_params.softening * orbit_params.softening;
        let dist = sqrt(dist_sq);
        let inv_dist3 = 1.0 / (dist * dist_sq);
        accel += diff * (G * other_mass * inv_dist3);
    }

    // Integration
    let new_vel = vel + accel * dt;
    let new_pos = pos + new_vel * dt;

    body.position = vec4<f32>(new_pos, mass);
    body.velocity = vec4<f32>(new_vel, body.velocity.w);

    bodies_out_buf[index] = body;

    // 2. UPDATE TRAILS
    // Shift all points for this body by one index to create a "moving history"
    let start_idx = index * TRAIL_LENGTH;

    for (var i = 0u; i < TRAIL_LENGTH - 1u; i = i + 1u) {
        trails[start_idx + i] = trails[start_idx + i + 1u];
    }

    // Write new position at the end of the trail
    trails[start_idx + TRAIL_LENGTH - 1u].position_pad = vec4<f32>(new_pos, 0.0);
    trails[start_idx + TRAIL_LENGTH - 1u].color = body.color * 0.5; // Dimmer trail
}