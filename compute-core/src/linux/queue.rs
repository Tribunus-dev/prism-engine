#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Hash)]
pub enum QueueClass {
    ForegroundCompute,
    BackgroundAnalysis,
    Transfer,
    Conformance,
}

use crate::linux::backend::RuntimeResourceId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueueHandle {
    pub id: RuntimeResourceId,
    pub class: QueueClass,
}
