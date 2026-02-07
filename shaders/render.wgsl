// ============================================================================
// GPU Render Shader: Instanced billboard rendering for particles + sphere 
// rendering for celestial bodies
// ============================================================================

struct Camera {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    screen_size: vec4<f32>,  // xy = screen size, z = time, w = unused
};

@group(0) @binding(0) var<uniform> camera: Camera;

// ============================================================================
// Billboard particle rendering (for spacecraft/satellites)
// ============================================================================

struct ParticleInstance {
    @location(0) position: vec4<f32>,  // xyz = pos, w = mass
    @location(1) velocity: vec4<f32>,  // xyz = vel, w = type
    @location(2) color: vec4<f32>,
    @location(3) data: vec4<f32>,      // x = radius, y = is_planet, z = trail, w = alive
};

struct ParticleVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_particle(
    @builtin(vertex_index) vertex_index: u32,
    instance: ParticleInstance,
) -> ParticleVsOut {
    var out: ParticleVsOut;
    
    // Skip dead particles
    if (instance.data.w < 0.5) {
        out.position = vec4<f32>(0.0, 0.0, -999.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.uv = vec2<f32>(0.0);
        return out;
    }
    
    // CORRECTED: Define as a local var inside the function to allow dynamic indexing
    var quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let quad_pos = quad_positions[vertex_index % 6u];

    // Extract camera right and up from view matrix
    let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    let size = instance.data.x * 2.0;
    let center = instance.position.xyz;
    let world_pos = center + (cam_right * quad_pos.x + cam_up * quad_pos.y) * size;

    out.position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = instance.color;
    out.uv = quad_pos;
    out.world_pos = world_pos;

    return out;
}

@fragment
fn fs_particle(in: ParticleVsOut) -> @location(0) vec4<f32> {
    // Circular particle with glow
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }

    // Soft glow effect
    let core = smoothstep(0.5, 0.0, dist);
    let glow = smoothstep(1.0, 0.2, dist);

    let color = in.color.rgb * (core * 2.0 + glow * 0.5);
    let alpha = glow * in.color.a;

    return vec4<f32>(color, alpha);
}

// ============================================================================
// Celestial body rendering (sphere impostor via billboards)
// ============================================================================

struct BodyInstance {
    @location(0) position: vec4<f32>,  // xyz = pos, w = mass
    @location(1) velocity: vec4<f32>,  // xyz = vel, w = radius
    @location(2) color: vec4<f32>,
    @location(3) data: vec4<f32>,      // x = is_star
};

struct BodyVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_center: vec3<f32>,
    @location(3) radius: f32,
    @location(4) is_star: f32,
};

@vertex
fn vs_body(
    @builtin(vertex_index) vertex_index: u32,
    instance: BodyInstance,
) -> BodyVsOut {
    var out: BodyVsOut;

    // CORRECTED: Define as a local var inside the function to allow dynamic indexing
    var quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let quad_pos = quad_positions[vertex_index % 6u];
    let cam_right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let cam_up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    let radius = instance.velocity.w;
    let center = instance.position.xyz;
    // Make billboard slightly larger to accommodate glow
    let billboard_size = radius * 2.5;
    let world_pos = center + (cam_right * quad_pos.x + cam_up * quad_pos.y) * billboard_size;

    out.position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = instance.color;
    out.uv = quad_pos * 2.5; // scale UV to match enlarged billboard
    out.world_center = center;
    out.radius = radius;
    out.is_star = instance.data.x;

    return out;
}

@fragment
fn fs_body(in: BodyVsOut) -> @location(0) vec4<f32> {
    let dist = length(in.uv);

    if (in.is_star > 0.5) {
        // Star rendering: bright core with corona
        let core = smoothstep(1.2, 0.0, dist);
        let corona = smoothstep(2.5, 0.5, dist) * 0.4;
        let flicker = sin(camera.screen_size.z * 3.0 + dist * 5.0) * 0.05 + 1.0;

        let brightness = (core * 3.0 + corona) * flicker;
        let color = in.color.rgb * brightness;
        let alpha = max(core, corona);

        if (alpha < 0.01) { discard; }
        return vec4<f32>(color, min(alpha, 1.0));
    } else {
        // Planet rendering: sphere impostor with basic shading
        if (dist > 1.3) { discard; }

        // Sphere impostor: compute normal from UV
        if (dist > 1.0) {
            // Atmosphere glow
            let atmo = smoothstep(1.3, 1.0, dist) * 0.3;
            return vec4<f32>(in.color.rgb * 0.5, atmo);
        }

        // Ensure UV is clamped for sqrt to avoid NaNs
        let clamped_uv = clamp(in.uv, vec2<f32>(-1.0), vec2<f32>(1.0));
        let z = sqrt(max(0.0, 1.0 - clamped_uv.x * clamped_uv.x - clamped_uv.y * clamped_uv.y));
        let normal = normalize(vec3<f32>(clamped_uv.x, clamped_uv.y, z));

        // Simple directional lighting from star (assumed at origin)
        let to_star = normalize(-in.world_center);
        let ndotl = max(dot(normal, to_star), 0.0);
        let ambient = 0.08;
        let diffuse = ndotl * 0.9;

        // Specular
        let view_dir = normalize(camera.eye_pos.xyz - in.world_center);
        let half_dir = normalize(to_star + view_dir);
        let spec = pow(max(dot(normal, half_dir), 0.0), 32.0) * 0.3;

        let lighting = ambient + diffuse + spec;
        let color = in.color.rgb * lighting;

        return vec4<f32>(color, 1.0);
    }
}

// ============================================================================
// Grid / orbit path rendering
// ============================================================================

struct GridVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct GridVertex {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_grid(input: GridVertex) -> GridVsOut {
    var out: GridVsOut;
    out.position = camera.view_proj * vec4<f32>(input.position, 1.0);
    out.world_pos = input.position;
    out.color = input.color;
    return out;
}

@fragment
fn fs_grid(in: GridVsOut) -> @location(0) vec4<f32> {
    // Fade with distance
    let dist = length(in.world_pos - camera.eye_pos.xyz);
    let fade = smoothstep(80.0, 5.0, dist);
    return vec4<f32>(in.color.rgb, in.color.a * fade);
}

// ============================================================================
// Fullscreen post-process (bloom approximation)
// ============================================================================

struct FullscreenVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> FullscreenVsOut {
    var out: FullscreenVsOut;
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}