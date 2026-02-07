use glam::Vec3;
use crate::types::GpuCelestialBody;

/// Create the real solar system with accurate physical properties
/// Units: AU (distance), solar masses (mass), years (time)
/// G = 4*pi^2 AU^3/(M_sun*yr^2), orbital velocity v = sqrt(G*M_sun/r)
pub fn create_solar_system() -> Vec<GpuCelestialBody> {
    let mut bodies = Vec::new();

    // Sun (at origin) - 1 solar mass
    bodies.push(GpuCelestialBody {
        position: [0.0, 0.0, 0.0, 1.0],    // mass = 1.0 solar mass
        velocity: [0.0, 0.0, 0.0, 0.15],    // visual radius (capped for display)
        color: [1.0, 0.95, 0.7, 1.0],       // warm yellow-white
        data: [1.0, 0.0, 0.0, 0.0],         // is_star = true
    });

    // Real solar system planets: (distance_AU, mass_solar_masses, visual_radius, color)
    // Visual radii use sqrt-compressed scaling: 0.03 * sqrt(real_radius / earth_radius)
    let planets: Vec<(f32, f32, f32, [f32; 4])> = vec![
        // Mercury: 0.387 AU, 3.301e23 kg
        (0.387,  1.660e-7, 0.019, [0.7, 0.6, 0.5, 1.0]),
        // Venus: 0.723 AU, 4.867e24 kg
        (0.723,  2.448e-6, 0.029, [0.9, 0.7, 0.3, 1.0]),
        // Earth: 1.000 AU, 5.972e24 kg
        (1.000,  3.003e-6, 0.030, [0.2, 0.5, 0.9, 1.0]),
        // Mars: 1.524 AU, 6.417e23 kg
        (1.524,  3.227e-7, 0.022, [0.8, 0.3, 0.2, 1.0]),
        // Jupiter: 5.203 AU, 1.898e27 kg
        (5.203,  9.543e-4, 0.099, [0.8, 0.6, 0.4, 1.0]),
        // Saturn: 9.537 AU, 5.683e26 kg
        (9.537,  2.858e-4, 0.091, [0.9, 0.8, 0.5, 1.0]),
        // Uranus: 19.191 AU, 8.681e25 kg
        (19.191, 4.366e-5, 0.060, [0.6, 0.8, 0.9, 1.0]),
        // Neptune: 30.069 AU, 1.024e26 kg
        (30.069, 5.150e-5, 0.059, [0.3, 0.4, 0.9, 1.0]),
    ];

    let star_mass = 1.0_f32;
    let g = 39.4784176_f32; // 4*pi^2, must match SimParams gravitational_constant

    for (dist, mass, radius, color) in &planets {
        // Compute circular orbital velocity: v = sqrt(G * M / r)
        let orbital_speed = (g * star_mass / dist).sqrt();

        // Start on x-axis, velocity in z direction for circular orbit
        let pos = Vec3::new(*dist, 0.0, 0.0);
        let vel = Vec3::new(0.0, 0.0, orbital_speed);

        bodies.push(GpuCelestialBody {
            position: [pos.x, pos.y, pos.z, *mass],
            velocity: [vel.x, vel.y, vel.z, *radius],
            color: *color,
            data: [0.0, orbital_speed, 0.0, 0.0], // not a star
        });
    }

    bodies
}

/// Generate orbit visualization lines
pub fn generate_orbit_lines(bodies: &[GpuCelestialBody]) -> Vec<([f32; 3], [f32; 4])> {
    let mut vertices = Vec::new();
    let segments = 128;

    for body in bodies.iter().skip(1) {
        // skip star
        let dist = (body.position[0] * body.position[0]
            + body.position[1] * body.position[1]
            + body.position[2] * body.position[2])
        .sqrt();

        let color = [
            body.color[0] * 0.3,
            body.color[1] * 0.3,
            body.color[2] * 0.3,
            0.25,
        ];

        for i in 0..=segments {
            let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let x = dist * angle.cos();
            let z = dist * angle.sin();
            vertices.push(([x, 0.0, z], color));
        }
    }

    vertices
}
