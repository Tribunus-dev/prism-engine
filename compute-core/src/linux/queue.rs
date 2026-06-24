#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueClass {
    ForegroundCompute,
    BackgroundAnalysis,
    Transfer,
    Conformance,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueueHandle {
    pub opaque_id: u64,
}
