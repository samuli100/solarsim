mod camera;
mod gpu;
mod simulation;
mod solar_system;
mod types;

use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

use camera::{Camera, MouseButton as CamButton};
use gpu::GpuState;
use simulation::{Simulation, SpawnMode};
use solar_system::create_solar_system;
use types::*;

fn upload_particles(gpu: &GpuState, sim: &Simulation) {
    let idx = gpu.frame_index % 2;
    gpu.queue.write_buffer(
        &gpu.particle_buffers[idx],
        0,
        bytemuck::cast_slice(&sim.particles),
    );
    gpu.queue.write_buffer(
        &gpu.particle_buffers[(idx + 1) % 2],
        0,
        bytemuck::cast_slice(&sim.particles),
    );
}

fn print_controls() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              ⭐  STAR SYSTEM SIMULATOR  ⭐                  ║");
    println!("║                GPU-Accelerated Physics                      ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  CAMERA                                                     ║");
    println!("║    Left Mouse + Drag    Orbit camera                        ║");
    println!("║    Scroll Wheel         Zoom in/out                         ║");
    println!("║                                                             ║");
    println!("║  INTERACTION                                                ║");
    println!("║    Right Click          Set swarm target (waypoint)          ║");
    println!("║    Middle Click         Spawn 100 particles at cursor       ║");
    println!("║    S                    Spawn 200 swarm particles           ║");
    println!("║    O                    Spawn orbital swarm around Earth    ║");
    println!("║    T                    Clear swarm target                   ║");
    println!("║    C                    Clear all particles                  ║");
    println!("║                                                             ║");
    println!("║  MODES                                                      ║");
    println!("║    1                    Swarm mode (boids + gravity)        ║");
    println!("║    2                    Free mode  (gravity only)           ║");
    println!("║    3                    Burst mode (scatter)                ║");
    println!("║                                                             ║");
    println!("║  TUNING                                                     ║");
    println!("║    Q / W / E            Increase sep / align / cohesion     ║");
    println!("║    G                    Increase gravity influence on swarm ║");
    println!("║    +/-                  Speed up / slow down time           ║");
    println!("║    Space                Pause / Resume                      ║");
    println!("║                                                             ║");
    println!("║  H = Toggle help  |  Esc = Quit                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("⭐ Star System Simulator — GPU Accelerated")
            .with_inner_size(PhysicalSize::new(1600u32, 900u32))
            .build(&event_loop)
            .unwrap(),
    );

    let mut gpu = pollster::block_on(GpuState::new(window.clone()));
    let mut camera = Camera::new();
    let size = window.inner_size();
    camera.resize(size.width, size.height);

    // Initialize solar system
    let bodies = create_solar_system();
    let mut sim = Simulation::new(bodies.clone());

    // Initialize Dynamic Orbit Trails
    // Instead of pre-calculating the orbit lines, we fill the buffer with
    // the planets' starting positions repeated TRAIL_LENGTH times.
    let trail_length = 512;
    let mut grid_vertices = Vec::with_capacity(bodies.len() * trail_length);

    for body in &bodies {
        let pos = [body.position[0], body.position[1], body.position[2]];
        let color = body.color;

        // Fill the entire trail with the starting position
        for _ in 0..trail_length {
            grid_vertices.push(GridVertex {
                position: pos,
                _pad: 0.0,
                color,
            });
        }
    }

    gpu.queue.write_buffer(
        &gpu.orbit_vertex_buffer,
        0,
        bytemuck::cast_slice(&grid_vertices),
    );
    gpu.orbit_vertex_count = grid_vertices.len() as u32;

    // Upload initial celestial bodies to both ping-pong buffers
    gpu.queue.write_buffer(&gpu.body_buffers[0], 0, bytemuck::cast_slice(&sim.bodies));
    gpu.queue.write_buffer(&gpu.body_buffers[1], 0, bytemuck::cast_slice(&sim.bodies));

    // Spawn initial swarm near the Earth-like planet
    sim.spawn_swarm(Vec3::new(1.0, 0.0, 0.2), 500);
    upload_particles(&gpu, &sim);

    let mut last_frame = Instant::now();
    let mut mouse_pos: (f32, f32) = (0.0, 0.0);
    let mut frame_count: u64 = 0;
    let mut fps_timer = Instant::now();

    print_controls();

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(size) => {
                        gpu.resize(size.width, size.height);
                        camera.resize(size.width, size.height);
                    }

                    WindowEvent::MouseInput {
                        state: btn_state,
                        button,
                        ..
                    } => {
                        let pressed = btn_state == ElementState::Pressed;
                        match button {
                            MouseButton::Left => {
                                camera.handle_mouse_button(CamButton::Left, pressed);
                            }
                            MouseButton::Right => {
                                if pressed {
                                    let (origin, dir) = camera.screen_to_world_ray(
                                        mouse_pos.0,
                                        mouse_pos.1,
                                        gpu.config.width as f32,
                                        gpu.config.height as f32,
                                    );
                                    if let Some(world_pos) =
                                        camera.ray_plane_intersection(origin, dir, 0.0)
                                    {
                                        sim.set_target(world_pos);
                                        log::info!(
                                            "Target set at ({:.1}, {:.1}, {:.1})",
                                            world_pos.x,
                                            world_pos.y,
                                            world_pos.z
                                        );
                                    }
                                }
                                camera.handle_mouse_button(CamButton::Right, pressed);
                            }
                            MouseButton::Middle => {
                                if pressed {
                                    let (origin, dir) = camera.screen_to_world_ray(
                                        mouse_pos.0,
                                        mouse_pos.1,
                                        gpu.config.width as f32,
                                        gpu.config.height as f32,
                                    );
                                    if let Some(world_pos) =
                                        camera.ray_plane_intersection(origin, dir, 0.0)
                                    {
                                        sim.spawn_burst(world_pos, 100);
                                        upload_particles(&gpu, &sim);
                                        log::info!("Spawned 100 particles");
                                    }
                                }
                                camera.handle_mouse_button(CamButton::Middle, pressed);
                            }
                            _ => {}
                        }
                    }

                    WindowEvent::CursorMoved { position, .. } => {
                        mouse_pos = (position.x as f32, position.y as f32);
                        camera.handle_mouse_move(position.x as f32, position.y as f32);
                    }

                    WindowEvent::MouseWheel { delta, .. } => {
                        let scroll = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                        };
                        camera.handle_scroll(scroll);
                    }

                    WindowEvent::KeyboardInput {
                        event:
                        KeyEvent {
                            logical_key,
                            state: key_state,
                            ..
                        },
                        ..
                    } => {
                        if key_state == ElementState::Pressed {
                            match logical_key.as_ref() {
                                // Spawn mode selection
                                Key::Character("1") => {
                                    sim.spawn_mode = SpawnMode::Swarm;
                                    log::info!("Mode: Swarm");
                                }
                                Key::Character("2") => {
                                    sim.spawn_mode = SpawnMode::Free;
                                    log::info!("Mode: Free");
                                }
                                Key::Character("3") => {
                                    sim.spawn_mode = SpawnMode::Burst;
                                    log::info!("Mode: Burst");
                                }

                                // Spawn swarm at cursor
                                Key::Character("s") => {
                                    let (origin, dir) = camera.screen_to_world_ray(
                                        mouse_pos.0,
                                        mouse_pos.1,
                                        gpu.config.width as f32,
                                        gpu.config.height as f32,
                                    );
                                    if let Some(pos) =
                                        camera.ray_plane_intersection(origin, dir, 0.0)
                                    {
                                        sim.spawn_swarm(pos, 200);
                                        upload_particles(&gpu, &sim);
                                        log::info!("Spawned 200 swarm particles");
                                    }
                                }

                                // Spawn orbital swarm around Earth
                                Key::Character("o") => {
                                    sim.spawn_orbital_swarm(3, 300, 0.1);
                                    upload_particles(&gpu, &sim);
                                    log::info!("Spawned orbital swarm around Earth");
                                }

                                // Clear target / particles
                                Key::Character("t") => {
                                    sim.clear_target();
                                    log::info!("Target cleared");
                                }
                                Key::Character("c") => {
                                    sim.clear_particles();
                                    upload_particles(&gpu, &sim);
                                    log::info!("Particles cleared");
                                }

                                // Pause
                                Key::Named(NamedKey::Space) => {
                                    sim.paused = !sim.paused;
                                    log::info!(
                                        "{}",
                                        if sim.paused { "⏸ Paused" } else { "▶ Resumed" }
                                    );
                                }

                                // Time scale
                                Key::Character("+") | Key::Character("=") => {
                                    sim.time_scale = (sim.time_scale * 1.5).min(10.0);
                                    log::info!("Time scale: {:.2}x", sim.time_scale);
                                }
                                Key::Character("-") => {
                                    sim.time_scale = (sim.time_scale / 1.5).max(0.1);
                                    log::info!("Time scale: {:.2}x", sim.time_scale);
                                }

                                // Help
                                Key::Character("h") => print_controls(),

                                // Swarm tuning
                                Key::Character("q") => {
                                    sim.params.separation_weight += 0.2;
                                    log::info!("Separation: {:.1}", sim.params.separation_weight);
                                }
                                Key::Character("w") => {
                                    sim.params.alignment_weight += 0.2;
                                    log::info!("Alignment: {:.1}", sim.params.alignment_weight);
                                }
                                Key::Character("e") => {
                                    sim.params.cohesion_weight += 0.2;
                                    log::info!("Cohesion: {:.1}", sim.params.cohesion_weight);
                                }
                                Key::Character("g") => {
                                    sim.params.swarm_gravity_weight =
                                        (sim.params.swarm_gravity_weight + 0.1).min(2.0);
                                    log::info!(
                                        "Gravity weight: {:.1}",
                                        sim.params.swarm_gravity_weight
                                    );
                                }

                                Key::Named(NamedKey::Escape) => elwt.exit(),
                                _ => {}
                            }
                        }
                    }

                    // ========================================================
                    // RENDER FRAME
                    // ========================================================
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = now.duration_since(last_frame).as_secs_f32().min(0.05);
                        last_frame = now;

                        // FPS counter
                        frame_count += 1;
                        let fps_elapsed = now.duration_since(fps_timer).as_secs_f32();
                        if fps_elapsed >= 1.0 {
                            let fps = frame_count as f32 / fps_elapsed;
                            frame_count = 0;
                            fps_timer = now;
                            window.set_title(&format!(
                                "⭐ Star System Sim | {:.0} FPS | {} particles | {:?}",
                                fps, sim.num_alive_particles, sim.spawn_mode
                            ));
                        }

                        // Update simulation state
                        sim.update_params(dt);
                        camera.update(dt);

                        // Upload simulation parameters
                        gpu.queue.write_buffer(
                            &gpu.sim_params_buffer,
                            0,
                            bytemuck::bytes_of(&sim.params),
                        );

                        // Upload camera uniform
                        let eye = camera.eye_position();
                        let cam_uniform = CameraUniform {
                            view_proj: camera.view_proj_matrix().to_cols_array_2d(),
                            view: camera.view_matrix().to_cols_array_2d(),
                            proj: camera.proj_matrix().to_cols_array_2d(),
                            eye_pos: [eye.x, eye.y, eye.z, 1.0],
                            screen_size: [
                                gpu.config.width as f32,
                                gpu.config.height as f32,
                                sim.time,
                                0.0,
                            ],
                        };
                        gpu.queue.write_buffer(
                            &gpu.camera_buffer,
                            0,
                            bytemuck::bytes_of(&cam_uniform),
                        );

                        // Get surface texture
                        let frame_idx = gpu.frame_index;
                        let output = match gpu.surface.get_current_texture() {
                            Ok(t) => t,
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let size = window.inner_size();
                                gpu.resize(size.width, size.height);
                                return;
                            }
                            Err(e) => {
                                log::error!("Surface error: {:?}", e);
                                return;
                            }
                        };

                        let view = output
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        let mut encoder =
                            gpu.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Frame"),
                                });

                        // === COMPUTE PASS 1: Update celestial body orbits ===
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Orbit Compute"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(&gpu.orbit_compute_pipeline);
                            pass.set_bind_group(
                                0,
                                &gpu.orbit_compute_bind_groups[frame_idx % 2],
                                &[],
                            );
                            pass.dispatch_workgroups(1, 1, 1);
                        }

                        // Sync: copy updated bodies so particle compute can read them
                        encoder.copy_buffer_to_buffer(
                            &gpu.body_buffers[(frame_idx + 1) % 2],
                            0,
                            &gpu.body_buffers[frame_idx % 2],
                            0,
                            (MAX_BODIES * std::mem::size_of::<GpuCelestialBody>()) as u64,
                        );

                        // === COMPUTE PASS 2: Update particles (gravity + swarm) ===
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Particle Compute"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(&gpu.particle_compute_pipeline);
                            pass.set_bind_group(
                                0,
                                &gpu.particle_compute_bind_groups[frame_idx % 2],
                                &[],
                            );
                            let workgroups = (MAX_PARTICLES as u32 + 255) / 256;
                            pass.dispatch_workgroups(workgroups, 1, 1);
                        }

                        // === RENDER PASS ===
                        {
                            let mut rp =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Main Render"),
                                    color_attachments: &[Some(
                                        wgpu::RenderPassColorAttachment {
                                            view: &view,
                                            resolve_target: None,
                                            ops: wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                                    r: 0.005,
                                                    g: 0.005,
                                                    b: 0.02,
                                                    a: 1.0,
                                                }),
                                                store: wgpu::StoreOp::Store,
                                            },
                                        },
                                    )],
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: &gpu.depth_texture,
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0),
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None,
                                        },
                                    ),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });

                            // 1. Draw orbit path lines (Dynamic Trails)
                            if gpu.orbit_vertex_count > 0 {
                                rp.set_pipeline(&gpu.orbit_render_pipeline);
                                rp.set_bind_group(0, &gpu.render_bind_group, &[]);
                                rp.set_vertex_buffer(0, gpu.orbit_vertex_buffer.slice(..));

                                let trail_len = 512u32;
                                let num_bodies = gpu.orbit_vertex_count / trail_len;
                                for i in 0..num_bodies {
                                    rp.draw(i * trail_len .. (i + 1) * trail_len, 0..1);
                                }
                            }

                            // 2. Draw celestial bodies (from updated buffer)
                            let body_buf_idx = (frame_idx + 1) % 2;
                            rp.set_pipeline(&gpu.body_render_pipeline);
                            rp.set_bind_group(0, &gpu.render_bind_group, &[]);
                            rp.set_vertex_buffer(0, gpu.body_buffers[body_buf_idx].slice(..));
                            rp.draw(0..6, 0..sim.bodies.len() as u32);

                            // 3. Draw all particles (from updated buffer)
                            let particle_buf_idx = (frame_idx + 1) % 2;
                            rp.set_pipeline(&gpu.particle_render_pipeline);
                            rp.set_bind_group(0, &gpu.render_bind_group, &[]);
                            rp.set_vertex_buffer(
                                0,
                                gpu.particle_buffers[particle_buf_idx].slice(..),
                            );
                            rp.draw(0..6, 0..MAX_PARTICLES as u32);
                        }

                        gpu.queue.submit(std::iter::once(encoder.finish()));
                        output.present();
                        gpu.frame_index += 1;
                    }

                    _ => {}
                },

                Event::AboutToWait => {
                    window.request_redraw();
                }

                _ => {}
            }
        })
        .unwrap();
}