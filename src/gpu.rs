use std::mem;
use wgpu::util::DeviceExt;
use crate::types::*;

/// Holds all GPU resources and pipelines
pub struct GpuState {
    // Core
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,

    // Buffers
    pub particle_buffers: [wgpu::Buffer; 2], // ping-pong
    pub body_buffers: [wgpu::Buffer; 2],     // ping-pong
    pub sim_params_buffer: wgpu::Buffer,
    pub camera_buffer: wgpu::Buffer,
    pub orbit_vertex_buffer: wgpu::Buffer,
    pub orbit_vertex_count: u32,

    // Compute pipelines
    pub particle_compute_pipeline: wgpu::ComputePipeline,
    pub orbit_compute_pipeline: wgpu::ComputePipeline,
    pub particle_compute_bind_groups: [wgpu::BindGroup; 2], // ping-pong
    pub orbit_compute_bind_groups: [wgpu::BindGroup; 2],

    // Render pipelines
    pub particle_render_pipeline: wgpu::RenderPipeline,
    pub body_render_pipeline: wgpu::RenderPipeline,
    pub orbit_render_pipeline: wgpu::RenderPipeline,
    pub render_bind_group: wgpu::BindGroup,

    // Depth buffer
    pub depth_texture: wgpu::TextureView,

    // State
    pub frame_index: usize,
}

impl GpuState {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        log::info!("GPU: {}", adapter.get_info().name);
        log::info!("Backend: {:?}", adapter.get_info().backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Main Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),

                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo, // VSync
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create shader modules
        let physics_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/physics.wgsl").into()),
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });

        // ====================================================================
        // Create buffers
        // ====================================================================

        let particle_size = mem::size_of::<GpuParticle>();
        let _particle_buf_size = (MAX_PARTICLES * particle_size) as u64;

        // Initialize with dead particles
        let initial_particles: Vec<GpuParticle> = vec![GpuParticle::dead(); MAX_PARTICLES];
        let particle_data = bytemuck::cast_slice(&initial_particles);

        let particle_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Particles A"),
                contents: particle_data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC, // Added COPY_SRC
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Particles B"),
                contents: particle_data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC, // Added COPY_SRC
            }),
        ];

        let body_size = mem::size_of::<GpuCelestialBody>();
        let _body_buf_size = (MAX_BODIES * body_size) as u64;
        let initial_bodies: Vec<GpuCelestialBody> =
            vec![bytemuck::Zeroable::zeroed(); MAX_BODIES];
        let body_data = bytemuck::cast_slice(&initial_bodies);

        let body_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bodies A"),
                contents: body_data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC, // Added COPY_SRC
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bodies B"),
                contents: body_data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC, // Added COPY_SRC
            }),
        ];

        let sim_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimParams"),
            size: mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera"),
            size: mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Orbit lines / Trails
        // 32 bodies * 512 points per trail
        let trail_length = 512;
        let orbit_buffer_size = (32 * trail_length * mem::size_of::<GridVertex>()) as u64;

        let orbit_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Orbit Trails"),
            size: orbit_buffer_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // ====================================================================
        // Compute pipeline: particles
        // ====================================================================

        let particle_compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Particle Compute BGL"),
                entries: &[
                    // particles_in
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // particles_out
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // bodies (read-only in particle shader)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let particle_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Compute PL"),
                bind_group_layouts: &[&particle_compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let particle_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Particle Compute"),
                layout: Some(&particle_compute_pipeline_layout),
                module: &physics_shader,
                entry_point: "cs_main",
                compilation_options: Default::default(),
            });

        // Ping-pong bind groups for particles
        let particle_compute_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Particle Compute BG 0"),
                layout: &particle_compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: particle_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: particle_buffers[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: body_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: sim_params_buffer.as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Particle Compute BG 1"),
                layout: &particle_compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: particle_buffers[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: particle_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: body_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: sim_params_buffer.as_entire_binding() },
                ],
            }),
        ];

        // ====================================================================
        // Compute pipeline: orbits (celestial body updates)
        // ====================================================================

        let orbit_compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Orbit Compute BGL"),
                entries: &[
                    // Bodies In
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Bodies Out
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Trails Out
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let orbit_compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Orbit Compute PL"),
                bind_group_layouts: &[&orbit_compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let orbit_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Orbit Compute"),
                layout: Some(&orbit_compute_pipeline_layout),
                module: &physics_shader,
                entry_point: "cs_orbit",
                compilation_options: Default::default(),
            });

        let orbit_compute_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Orbit Compute BG 0"),
                layout: &orbit_compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: body_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: body_buffers[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: sim_params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: orbit_vertex_buffer.as_entire_binding() },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Orbit Compute BG 1"),
                layout: &orbit_compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: body_buffers[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: body_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: sim_params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: orbit_vertex_buffer.as_entire_binding() },
                ],
            }),
        ];

        // ====================================================================
        // Render pipelines
        // ====================================================================

        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render BG"),
            layout: &render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render PL"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Instance buffer layout for particles
        let particle_instance_layout = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<GpuParticle>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 3 },
            ],
        };

        let blend_additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let blend_alpha = wgpu::BlendState::ALPHA_BLENDING;

        let depth_stencil_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false, // no depth write for transparent
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // Particle render pipeline
        let particle_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Particle Render"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: "vs_particle",
                    compilation_options: Default::default(),
                    buffers: &[particle_instance_layout.clone()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: "fs_particle",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(blend_additive),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(depth_stencil_state.clone()),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        // Body instance layout
        let body_instance_layout = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<GpuCelestialBody>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 3 },
            ],
        };

        // Body render pipeline
        let body_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Body Render"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: "vs_body",
                    compilation_options: Default::default(),
                    buffers: &[body_instance_layout],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: "fs_body",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(blend_alpha),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        // Orbit line vertex layout
        let grid_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<GridVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 1 },
            ],
        };

        let orbit_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Orbit Render"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: "vs_grid",
                    compilation_options: Default::default(),
                    buffers: &[grid_vertex_layout],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: "fs_grid",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(blend_alpha),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineStrip,
                    strip_index_format: None,
                    ..Default::default()
                },
                depth_stencil: Some(depth_stencil_state),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        // Depth texture
        let depth_texture = Self::create_depth_texture(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            particle_buffers,
            body_buffers,
            sim_params_buffer,
            camera_buffer,
            orbit_vertex_buffer,
            orbit_vertex_count: 0,
            particle_compute_pipeline,
            orbit_compute_pipeline,
            particle_compute_bind_groups,
            orbit_compute_bind_groups,
            particle_render_pipeline,
            body_render_pipeline,
            orbit_render_pipeline,
            render_bind_group,
            depth_texture,
            frame_index: 0,
        }
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
        }
    }
}