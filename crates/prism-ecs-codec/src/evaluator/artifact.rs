//! BackendArtifact — concrete compiled artifacts for each backend.
//!
//! This module owns the canonical authority for the backend-specific
//! bytes a [`GeneratedExecutable`](super::GeneratedExecutable) compiles
//! into. The variant discriminates the backend; the bytes are
//! opaque to the evaluator. Bytes are content-addressed by the
//! parent executable's `artifact_digest`.

use serde::{Deserialize, Serialize};

/// Concrete compiled artifact for a specific backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

impl BackendArtifact {
    /// Returns the backend name of this artifact.
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Metal(_) => "metal",
            Self::Ane(_) => "ane",
            Self::Accelerate(_) => "accelerate",
            Self::FutureNpu(_) => "future_npu",
        }
    }

    /// Returns the byte length of the artifact.
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Metal(b) | Self::Ane(b) | Self::Accelerate(b) | Self::FutureNpu(b) => b.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_carry_their_bytes() {
        let metal = BackendArtifact::Metal(vec![1, 2, 3]);
        let ane = BackendArtifact::Ane(vec![4, 5, 6]);
        let accel = BackendArtifact::Accelerate(vec![7, 8, 9]);
        let npu = BackendArtifact::FutureNpu(vec![10, 11, 12]);

        assert_eq!(metal.backend_name(), "metal");
        assert_eq!(ane.backend_name(), "ane");
        assert_eq!(accel.backend_name(), "accelerate");
        assert_eq!(npu.backend_name(), "future_npu");

        assert_eq!(metal.byte_len(), 3);
        assert_eq!(ane.byte_len(), 3);
        assert_eq!(accel.byte_len(), 3);
        assert_eq!(npu.byte_len(), 3);
    }

    #[test]
    fn artifacts_serialize() {
        let a = BackendArtifact::Metal(vec![0xDE, 0xAD]);
        let json = serde_json::to_string(&a).expect("serialize");
        let restored: BackendArtifact = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, a);
    }
}
