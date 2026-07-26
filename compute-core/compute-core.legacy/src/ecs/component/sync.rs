use crate::ecs::{CompEntity, Component};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimelineSemaphore {
    pub initial_value: u64,
    pub current_value: u64,
}
impl Component for TimelineSemaphore {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FenceEdge {
    pub wait_semaphores: Vec<(CompEntity, u64)>,
    pub signal_semaphores: Vec<(CompEntity, u64)>,
}
impl Component for FenceEdge {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QueueAffinity(pub u64);
impl Component for QueueAffinity {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncBarrier {
    pub memory_scope: MemoryScope,
    pub access_mask: u32,
}
impl Component for SyncBarrier {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryScope {
    Device,
    Host,
    CrossDevice,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompletionSignal(pub bool);
impl Component for CompletionSignal {}
