# ⭐ Star System Simulator

A high-performance, GPU-accelerated star system simulation written in **Rust** with **wgpu** (Vulkan/Metal/DX12/OpenGL backend). Features real-time gravitational N-body physics, interactive swarm behavior (boids algorithm), and instanced 3D rendering — all running on the GPU via compute shaders.

![Architecture](https://img.shields.io/badge/Rust-wgpu-orange) ![GPU](https://img.shields.io/badge/GPU-Compute%20Shaders-blue) ![Physics](https://img.shields.io/badge/Physics-N--Body%20Gravity-green)

## Features

- **GPU-Accelerated Physics**: All gravitational calculations and swarm behavior run as WGSL compute shaders on the GPU
- **N-Body Gravity**: Star and planets exert realistic gravitational forces on all particles
- **Swarm Behavior**: Boids-like flocking (separation, alignment, cohesion) with configurable weights
- **65,536 Particles**: Handles tens of thousands of particles at interactive frame rates
- **Interactive Control**: Set waypoints for swarm navigation, spawn particles with mouse clicks
- **Orbital Mechanics**: Planets orbit the star with physically correct circular velocities
- **Sphere Impostor Rendering**: Celestial bodies rendered as billboard sphere impostors with lighting
- **Cross-Platform**: Runs on Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows), or OpenGL fallback

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CPU (Rust)                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │  Camera   │  │ Simulation│  │  Input / Events   │  │
│  │ Controller│  │   State   │  │    (winit)        │  │
│  └────┬─────┘  └─────┬────┘  └────────┬──────────┘  │
│       │              │                │              │
│       ▼              ▼                ▼              │
│  ┌─────────────────────────────────────────────────┐ │
│  │              GPU Command Encoder                │ │
│  └──────────────────┬──────────────────────────────┘ │
└─────────────────────┼───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│                   GPU (wgpu)                         │
│                                                      │
│  ┌─ COMPUTE PASS 1 ──────────────────────────────┐  │
│  │  Orbit Shader: Update planet positions         │  │
│  │  (N-body gravity between celestial bodies)     │  │
│  └────────────────────────────────────────────────┘  │
│                      │                               │
│  ┌─ COMPUTE PASS 2 ──────────────────────────────┐  │
│  │  Particle Shader: For each of 65K particles:   │  │
│  │  ├─ Compute gravity from all celestial bodies  │  │
│  │  ├─ Compute swarm forces (boids neighbors)     │  │
│  │  ├─ Integrate velocity + position              │  │
│  │  └─ Collision detection + lifecycle            │  │
│  └────────────────────────────────────────────────┘  │
│                      │                               │
│  ┌─ RENDER PASS ─────────────────────────────────┐  │
│  │  1. Orbit lines (LineStrip)                    │  │
│  │  2. Celestial bodies (Instanced billboards)    │  │
│  │  3. Particles (Instanced billboards + glow)    │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  Buffers: Ping-pong double buffering for particles   │
│  and bodies to avoid read/write hazards              │
└──────────────────────────────────────────────────────┘
```

## Requirements

- **Rust** 1.75+ (install via [rustup](https://rustup.rs))
- A GPU with Vulkan 1.0, Metal, DX12, or OpenGL 4.3+ support
- Linux: `libwayland-dev` or `libx11-dev` packages

### Linux Dependencies (Ubuntu/Debian)

```bash
sudo apt install -y libwayland-dev libxkbcommon-dev libx11-dev libxi-dev libxrandr-dev
```

### macOS / Windows

No additional dependencies needed — Metal/DX12 are built-in.

## Build & Run

```bash
# Debug build (faster compilation, slower runtime)
cargo run

# Release build (optimized, recommended for large particle counts)
cargo run --release
```

## Controls

| Input | Action |
|-------|--------|
| **Left Mouse + Drag** | Orbit camera |
| **Scroll Wheel** | Zoom in/out |
| **Right Click** | Set swarm waypoint target |
| **Middle Click** | Spawn 100 particles at cursor |
| **S** | Spawn 200 swarm particles at cursor |
| **O** | Spawn orbital swarm around Earth |
| **T** | Clear swarm target |
| **C** | Clear all particles |
| **1 / 2 / 3** | Switch mode: Swarm / Free / Burst |
| **Q / W / E** | Tune: Separation / Alignment / Cohesion |
| **G** | Increase gravity influence on swarm |
| **+/-** | Speed up / slow down simulation |
| **Space** | Pause / Resume |
| **H** | Toggle help display |
| **Esc** | Quit |

## Solar System

The simulation includes a star and 7 planets with these characteristics:

| Body | Distance | Color | Notes |
|------|----------|-------|-------|
| Star | Center | Yellow-white | Mass: 1000, emissive rendering with corona |
| Mercury | 15 AU | Gray-brown | Small, fast orbit |
| Venus | 25 AU | Orange-yellow | Hot, thick atmosphere glow |
| Earth | 38 AU | Blue | Starting location for swarm |
| Mars | 52 AU | Red | Smaller, slower |
| Jupiter | 80 AU | Orange-tan | Gas giant, strong gravity |
| Saturn | 110 AU | Gold | Ringed gas giant |
| Neptune | 145 AU | Deep blue | Ice giant |

## Technical Details

### Physics

- **Gravitational constant**: Tuned for visual appeal (G=40)
- **Softening parameter**: ε=0.5 prevents singularities at close range
- **Integration**: Symplectic Euler (good energy conservation)
- **Swarm model**: Reynolds boids with 3 forces — separation, alignment, cohesion
- **Collision**: Particles destroyed on contact with celestial bodies

### Performance

- Compute shader workgroup size: 256 threads
- Particle buffer: Ping-pong double-buffered to avoid GPU hazards
- Billboard rendering: No mesh geometry — 6 vertices per quad via `vertex_index`
- Instanced drawing: Single draw call for all particles, single draw call for all bodies
- VSync enabled by default (switch to `PresentMode::Immediate` for uncapped FPS)

### Extending

To add more celestial bodies, edit `src/solar_system.rs`. To change swarm behavior, modify the compute shader in `shaders/physics.wgsl`. The boids parameters can also be tuned at runtime with Q/W/E/G keys.

## File Structure

```
starsystem-sim/
├── Cargo.toml                  # Dependencies and build config
├── README.md
├── shaders/
│   ├── physics.wgsl            # GPU compute: gravity + swarm behavior
│   └── render.wgsl             # GPU render: billboards, sphere impostors, grid
└── src/
    ├── main.rs                 # Entry point, event loop, input handling
    ├── types.rs                # GPU-compatible data structures (Pod/Zeroable)
    ├── camera.rs               # Orbital camera with smooth interpolation
    ├── gpu.rs                  # wgpu device, pipelines, buffer management
    ├── simulation.rs           # High-level sim state, spawning, interaction
    └── solar_system.rs         # Planet definitions and orbit visualization
```

## License

MIT
