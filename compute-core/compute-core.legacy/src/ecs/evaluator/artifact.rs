//! Concrete compiled artifacts for each backend.

use serde::{Deserialize, Serialize};

/// Concrete compiled artifact for a specific backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendArtifact {
    /// Metal compiled library (.metallib bytes).
    Metal(Vec<u8>),
    /// ANE compiled model (.mlmodelc or MIL program).
    Ane(Vec<u8>),
    /// Accelerate/CPU precompiled plan.
    Accelerate(Vec<u8>),
    /// Future NPU artifact (opaque bytes).
    FutureNpu(Vec<u8>),
}
