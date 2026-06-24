pub trait Scheduler: Send {
    fn step(&mut self, latent: &[f32], noise: &[f32], timestep: u32) -> Vec<f32>;
    fn timesteps(&self) -> &[u32];
    fn guidance_scale(&self) -> f32;
}

pub struct DdpmScheduler {}

impl Scheduler for DdpmScheduler {
    fn step(&mut self, _latent: &[f32], _noise: &[f32], _timestep: u32) -> Vec<f32> {
        vec![]
    }
    fn timesteps(&self) -> &[u32] {
        &[]
    }
    fn guidance_scale(&self) -> f32 {
        7.5
    }
}

pub struct FlowMatchingScheduler {}

impl Scheduler for FlowMatchingScheduler {
    fn step(&mut self, _latent: &[f32], _noise: &[f32], _timestep: u32) -> Vec<f32> {
        vec![]
    }
    fn timesteps(&self) -> &[u32] {
        &[]
    }
    fn guidance_scale(&self) -> f32 {
        4.0
    }
}

pub struct RectifiedFlowScheduler {}

impl Scheduler for RectifiedFlowScheduler {
    fn step(&mut self, _latent: &[f32], _noise: &[f32], _timestep: u32) -> Vec<f32> {
        vec![]
    }
    fn timesteps(&self) -> &[u32] {
        &[]
    }
    fn guidance_scale(&self) -> f32 {
        5.0
    }
}
