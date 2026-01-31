use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

/// ========== App (owns window + engine timing + input) ==========

#[derive(Copy, Clone, Debug, Default)]
struct Aabb {
    // center + half extents, in world units
    c: [f32; 2],
    h: [f32; 2],
}

impl Aabb {
    fn min(&self) -> [f32; 2] {
        [self.c[0] - self.h[0], self.c[1] - self.h[1]]
    }
    fn max(&self) -> [f32; 2] {
        [self.c[0] + self.h[0], self.c[1] + self.h[1]]
    }
}

#[derive(Copy, Clone, Debug)]
struct Camera2D {
    // visible height in world units; width depends on aspect ratio
    view_height: f32,
    aspect: f32,
}

impl Camera2D {
    fn new(view_height: f32, width_px: u32, height_px: u32) -> Self {
        let aspect = if height_px == 0 {
            1.0
        } else {
            width_px as f32 / height_px as f32
        };
        Self {
            view_height,
            aspect,
        }
    }

    fn view_width(&self) -> f32 {
        self.view_height * self.aspect
    }

    // Convert world position (x,y) to NDC (-1..1). Camera centered at origin.
    fn world_to_ndc(&self, p: [f32; 2]) -> [f32; 2] {
        let hw = self.view_width() * 0.5;
        let hh = self.view_height * 0.5;
        [p[0] / hw, p[1] / hh]
    }

    // Convert world half extents to NDC scale.
    fn half_to_ndc_scale(&self, half: [f32; 2]) -> [f32; 2] {
        let hw = self.view_width() * 0.5;
        let hh = self.view_height * 0.5;
        [half[0] / hw, half[1] / hh]
    }
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,

    // Timing
    next_frame: Instant,
    frame_dt: Duration,

    // Input
    input: InputState,
}

#[derive(Default, Copy, Clone)]
struct InputState {
    left: bool,
    right: bool,
    jump_pressed: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
            next_frame: Instant::now(),
            frame_dt: Duration::from_secs_f64(1.0 / 120.0),
            input: InputState::default(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_title("Cube Platformer — Group 2"))
            .expect("create window");

        let window = Arc::new(window);
        let state = pollster::block_on(State::new(window.clone()));

        self.window = Some(window);
        self.state = Some(state);

        self.next_frame = Instant::now();
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let Some(window) = self.window.as_ref() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    state.resize(size.width, size.height);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                // Track key state
                let pressed = event.state == ElementState::Pressed;

                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::ArrowLeft | KeyCode::KeyA => self.input.left = pressed,
                        KeyCode::ArrowRight | KeyCode::KeyD => self.input.right = pressed,
                        // Jump is edge-triggered: set true only on pressed event.
                        KeyCode::Space | KeyCode::ArrowUp | KeyCode::KeyW => {
                            if pressed {
                                self.input.jump_pressed = true;
                            }
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // Update simulation at 120 Hz pacing.
                state.update(self.frame_dt.as_secs_f32(), self.input);

                // Render one frame
                match state.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = window.inner_size();
                        if size.width > 0 && size.height > 0 {
                            state.resize(size.width, size.height);
                        }
                    }
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of memory");
                        event_loop.exit();
                    }
                    Err(wgpu::SurfaceError::Other) => {
                        log::warn!("Surface error: Other");
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
            self.input.jump_pressed = false;
        }

        let now = Instant::now();
        if now >= self.next_frame {
            self.next_frame = now + self.frame_dt;
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("run app");
}

/// ========== GPU State (wgpu) ==========

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadUniform {
    // NDC offset (x,y): move the cube around
    offset: [f32; 2],
    // NDC scale (x,y): size of the cube
    scale: [f32; 2],
}

struct State {
    // Keep the window alive by owning an Arc
    _window: Arc<Window>,

    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,

    uniform: QuadUniform,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    camera: Camera2D,

    // Physics
    cube_pos: [f32; 2], // world units
    cube_vel: [f32; 2], // world units / second
    cube_half: [f32; 2],
    on_ground: bool,

    // Tunables
    move_speed: f32,
    gravity: f32,
    jump_speed: f32,

    ground: Aabb,
}

impl State {
    fn sync_uniform(&mut self) {
        // World -> NDC
        let ndc_pos = self.camera.world_to_ndc(self.cube_pos);
        let ndc_half = self.camera.half_to_ndc_scale(self.cube_half);

        // Our shader uses: pos * scale + offset, where pos is in [-0.5..0.5]
        // So scale should be full size in NDC (half -> full):
        self.uniform.offset = ndc_pos;
        self.uniform.scale = [ndc_half[0] * 2.0, ndc_half[1] * 2.0];

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));
    }

    async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();

        // Key trick: pass Arc<Window> to create_surface so Surface can be 'static.
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .expect("request adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),

                // wgpu 28+ fields:
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request device");

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ---------- Geometry: a quad made of 2 triangles ----------
        // Positions are centered at origin; shader applies scale + offset.
        #[rustfmt::skip]
        let vertices: [f32; 12] = [
            -0.5, -0.5,
             0.5, -0.5,
             0.5,  0.5,

            -0.5, -0.5,
             0.5,  0.5,
            -0.5,  0.5,
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_count = 6;

        // ---------- Uniform (offset + scale) ----------
        let uniform = QuadUniform {
            offset: [0.0, -0.2],
            scale: [0.15, 0.15],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad uniform buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // ---------- Shaders + pipeline ----------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("quad pipeline"),
            layout: Some(&pipeline_layout),
            multiview_mask: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },

            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),

            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },

            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
        });

        let camera = Camera2D::new(12.0, config.width, config.height);

        // Cube is 1x1 world units
        let cube_half = [0.5, 0.5];

        // Put the ground at y = -4.0, thickness 1.0, wide enough to cover view
        let ground = Aabb {
            c: [0.0, -4.5],
            h: [50.0, 0.5],
        };

        let cube_pos = [0.0, -3.0];
        let cube_vel = [0.0, 0.0];

        let mut s = Self {
            _window: window,
            surface,
            device,
            queue,
            config,

            render_pipeline,
            vertex_buffer,
            vertex_count,

            uniform,
            uniform_buffer,
            uniform_bind_group,

            camera,

            cube_pos,
            cube_vel,
            cube_half,
            on_ground: false,

            move_speed: 6.0,
            gravity: -30.0,
            jump_speed: 12.0,

            ground,
        };

        // Write initial uniform using camera mapping:
        s.sync_uniform();
        s
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);

        self.camera = Camera2D::new(
            self.camera.view_height,
            self.config.width,
            self.config.height,
        );
        self.sync_uniform();
    }

    fn update(&mut self, dt: f32, input: InputState) {
        // --- Horizontal control (arcade) ---
        let mut dir = 0.0;
        if input.left {
            dir -= 1.0;
        }
        if input.right {
            dir += 1.0;
        }

        self.cube_vel[0] = dir * self.move_speed;

        // --- Jump (only if grounded) ---
        if input.jump_pressed && self.on_ground {
            self.cube_vel[1] = self.jump_speed;
            self.on_ground = false;
        }

        // --- Gravity ---
        self.cube_vel[1] += self.gravity * dt;

        // --- Integrate position ---
        self.cube_pos[0] += self.cube_vel[0] * dt;
        self.cube_pos[1] += self.cube_vel[1] * dt;

        // --- Collision with ground AABB (only resolve from above) ---
        let cube = Aabb {
            c: self.cube_pos,
            h: self.cube_half,
        };

        // Check overlap cube vs ground
        let cmin = cube.min();
        let cmax = cube.max();
        let gmin = self.ground.min();
        let gmax = self.ground.max();

        let overlap_x = cmin[0] < gmax[0] && cmax[0] > gmin[0];
        let overlap_y = cmin[1] < gmax[1] && cmax[1] > gmin[1];

        if overlap_x && overlap_y {
            // If we are falling (or stationary) and we intersect, snap cube to top of ground
            // Ground top:
            let ground_top = gmax[1];

            // Cube bottom:
            let cube_bottom = cmin[1];

            // Only treat it as landing if cube bottom is below ground top and velocity is downward
            if self.cube_vel[1] <= 0.0 && cube_bottom < ground_top {
                self.cube_pos[1] = ground_top + self.cube_half[1];
                self.cube_vel[1] = 0.0;
                self.on_ground = true;
            }
        } else {
            // If not overlapping, we are not grounded (simple rule).
            self.on_ground = false;
        }

        // --- Keep cube in a reasonable horizontal range for now ---
        self.cube_pos[0] = self.cube_pos[0].clamp(-10.0, 10.0);

        // Update GPU uniform after physics
        self.sync_uniform();
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None, // wgpu 28+
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.06,
                            g: 0.06,
                            b: 0.09,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None, // wgpu 28+
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.uniform_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..self.vertex_count, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}
