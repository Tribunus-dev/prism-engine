//! Prism ECS composition root.
//!
//! The daemon owns the authoritative ECS world and binds it to the runtime
//! kernel and validated execution schedule. Higher layers may install compile
//! systems, but the ownership boundary remains here.

use prism_ecs_core::World;
use prism_ecs_runtime::{KernelHealth, RuntimeError, RuntimeKernel, RuntimeSchedule};
use std::sync::Arc;

#[derive(Clone)]
pub struct PrismDaemon {
    world: Arc<std::sync::RwLock<World>>,
    kernel: Arc<RuntimeKernel>,
    instance_id: Arc<str>,
}

impl PrismDaemon {
    pub fn new(instance_id: impl Into<String>) -> Self {
        let world = Arc::new(std::sync::RwLock::new(World::new()));
        let kernel = RuntimeKernel::with_existing_world(world.clone());
        Self {
            world,
            kernel: Arc::new(kernel),
            instance_id: Arc::from(instance_id.into()),
        }
    }

    pub fn world(&self) -> Arc<std::sync::RwLock<World>> {
        self.world.clone()
    }
    pub fn kernel(&self) -> Arc<RuntimeKernel> {
        self.kernel.clone()
    }
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    pub fn install_schedule(&self, mut schedule: RuntimeSchedule) {
        schedule.bind(&self.kernel.handle());
        self.kernel.set_schedule(schedule);
    }

    pub fn health(&self) -> KernelHealth {
        self.kernel.health()
    }
    pub fn tick(&self) -> Result<(), RuntimeError> {
        self.kernel.run_kernel_tick(&self.instance_id)
    }
}

impl Default for PrismDaemon {
    fn default() -> Self {
        Self::new("prism-daemon")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_root_owns_world_and_kernel() {
        let daemon = PrismDaemon::default();
        assert_eq!(daemon.instance_id(), "prism-daemon");
        assert_eq!(daemon.health().entity_count, 0);
    }

    #[test]
    fn schedule_is_bound_through_kernel() {
        let daemon = PrismDaemon::new("test");
        daemon.install_schedule(RuntimeSchedule::new());
        assert!(daemon.tick().is_ok());
    }
}
