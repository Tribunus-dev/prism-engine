use serde::{Deserialize, Serialize};

/// Opaque trace identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceId(pub String);

/// Backend identifier for trace targeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendId {
    Metal,
    Accelerate,
    Ane,
    Mlx,
    Cpu,
    Xdna,
}

/// Binding from a tensor name to a buffer role in a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBinding {
    pub tensor_name: String,
    pub role: BufferRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferRole {
    Input,
    Output,
    Scratch { size_bytes: usize },
}

/// A recorded execution trace for zero-overhead replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub trace_id: TraceId,
    pub target_backend: BackendId,
    pub resource_bindings: Vec<ResourceBinding>,
    pub record_count: u32,
}

/// Evidence for one replayed heterogeneous AOT step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecutionEvidence {
    pub step_id: usize,
    pub backend: BackendId,
    pub started_ns: u64,
    pub elapsed_ns: u64,
    pub input_region: String,
    pub output_region: String,
    pub zero_copy: bool,
    pub residency_window: usize,
    /// Strategy annotation that produced this replayed step.
    #[serde(default)]
    pub fusion_strategy: Option<crate::fused_ops::FusionStrategy>,
}

/// Complete evidence receipt for one AOT schedule replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousExecutionReceipt {
    pub plan_id: String,
    pub steps: Vec<StepExecutionEvidence>,
    pub model_residency_windows: usize,
    pub total_elapsed_ns: u64,
}

impl ExecutionTrace {
    pub fn new(backend: BackendId) -> Self {
        Self {
            trace_id: TraceId(format!(
                "trace_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            )),
            target_backend: backend,
            resource_bindings: Vec::new(),
            record_count: 0,
        }
    }
}
