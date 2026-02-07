use glam::{Mat4, Vec3};

pub struct Camera {
    /// Spherical coordinates
    pub distance: f32,
    pub theta: f32,    // horizontal angle (radians)
    pub phi: f32,      // vertical angle (radians)
    pub target: Vec3,  // look-at target

    /// Smooth interpolation targets
    target_distance: f32,
    target_theta: f32,
    target_phi: f32,
    target_target: Vec3,

    /// Configuration
    pub min_distance: f32,
    pub max_distance: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,

    /// Interaction state
    pub is_panning: bool,
    pub is_orbiting: bool,
    last_mouse: Option<(f32, f32)>,
}

impl Camera {
    pub fn new() -> Self {
        let distance = 5.0;
        let theta = std::f32::consts::FRAC_PI_4;
        let phi = 0.5;

        Self {
            distance,
            theta,
            phi,
            target: Vec3::ZERO,
            target_distance: distance,
            target_theta: theta,
            target_phi: phi,
            target_target: Vec3::ZERO,
            min_distance: 0.1,
            max_distance: 100.0,
            fov: 60.0_f32.to_radians(),
            near: 0.001,
            far: 300.0,
            aspect: 16.0 / 9.0,
            is_panning: false,
            is_orbiting: false,
            last_mouse: None,
        }
    }

    pub fn eye_position(&self) -> Vec3 {
        let x = self.distance * self.phi.cos() * self.theta.sin();
        let y = self.distance * self.phi.sin();
        let z = self.distance * self.phi.cos() * self.theta.cos();
        self.target + Vec3::new(x, y, z)
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    pub fn handle_scroll(&mut self, delta: f32) {
        self.target_distance = (self.target_distance - delta * self.target_distance * 0.1)
            .clamp(self.min_distance, self.max_distance);
    }

    pub fn handle_mouse_move(&mut self, x: f32, y: f32) {
        if let Some((lx, ly)) = self.last_mouse {
            let dx = x - lx;
            let dy = y - ly;

            if self.is_orbiting {
                self.target_theta -= dx * 0.005;
                self.target_phi = (self.target_phi + dy * 0.005).clamp(
                    -std::f32::consts::FRAC_PI_2 + 0.1,
                    std::f32::consts::FRAC_PI_2 - 0.1,
                );
            }

            if self.is_panning {
                let view = self.view_matrix();
                let right = Vec3::new(view.col(0).x, view.col(1).x, view.col(2).x);
                let up = Vec3::new(view.col(0).y, view.col(1).y, view.col(2).y);
                let pan_speed = self.distance * 0.002;
                self.target_target -= right * dx * pan_speed;
                self.target_target += up * dy * pan_speed;
            }
        }
        self.last_mouse = Some((x, y));
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.is_orbiting = pressed,
            MouseButton::Right => self.is_panning = pressed,
            MouseButton::Middle => self.is_panning = pressed,
        }
        if !pressed {
            self.last_mouse = None;
        }
    }

    /// Smooth interpolation update
    pub fn update(&mut self, dt: f32) {
        let lerp_speed = 8.0 * dt;
        self.distance += (self.target_distance - self.distance) * lerp_speed;
        self.theta += (self.target_theta - self.theta) * lerp_speed;
        self.phi += (self.target_phi - self.phi) * lerp_speed;
        self.target += (self.target_target - self.target) * lerp_speed;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }

    /// Get a world-space ray from screen coordinates for picking
    pub fn screen_to_world_ray(&self, screen_x: f32, screen_y: f32, width: f32, height: f32) -> (Vec3, Vec3) {
        let ndc_x = (2.0 * screen_x / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / height);

        let inv_proj = self.proj_matrix().inverse();
        let inv_view = self.view_matrix().inverse();

        let ray_clip = glam::Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let ray_eye = inv_proj * ray_clip;
        let ray_eye = glam::Vec4::new(ray_eye.x, ray_eye.y, -1.0, 0.0);
        let ray_world = inv_view * ray_eye;
        let ray_dir = Vec3::new(ray_world.x, ray_world.y, ray_world.z).normalize();

        (self.eye_position(), ray_dir)
    }

    /// Intersect ray with a plane at given Y to get world position
    pub fn ray_plane_intersection(&self, ray_origin: Vec3, ray_dir: Vec3, plane_y: f32) -> Option<Vec3> {
        if ray_dir.y.abs() < 1e-6 {
            return None;
        }
        let t = (plane_y - ray_origin.y) / ray_dir.y;
        if t < 0.0 {
            return None;
        }
        Some(ray_origin + ray_dir * t)
    }
}

pub enum MouseButton {
    Left,
    Right,
    Middle,
}
