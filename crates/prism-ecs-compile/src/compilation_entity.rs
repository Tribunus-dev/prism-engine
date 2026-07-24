use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompilationStatus {
    Created,
    Running,
    Complete,
    Failed,
}
#[derive(Debug)]
pub struct CompilationEntity {
    pub status: CompilationStatus,
}
impl prism_ecs_core::Component for CompilationEntity {}
impl CompilationEntity {
    pub fn new<T>(_: T) -> Self {
        Self {
            status: CompilationStatus::Created,
        }
    }
}
