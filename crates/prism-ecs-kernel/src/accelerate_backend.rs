use crate::{BackendKind, CpuBackend, KernelBackend, KernelDescriptor};

/// Accelerate-compatible reference backend. On non-Apple hosts it preserves
/// the contract while delegating execution to the portable CPU implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct AccelerateBackend;

pub struct TernaryParityReport {
    pub max_abs_error: f32,
    pub passed: bool,
}

impl KernelBackend for AccelerateBackend {
    fn validate(&self, descriptor: &KernelDescriptor) -> Result<(), crate::KernelError> {
        if descriptor.backend != BackendKind::CPU {
            return Err(crate::KernelError::ValidationFailed(
                "Accelerate reference requires a CPU descriptor".into(),
            ));
        }
        CpuBackend.validate(descriptor)
    }
    fn compile(
        &self,
        request: &crate::KernelCompileRequest,
    ) -> Result<crate::KernelArtifact, crate::KernelError> {
        CpuBackend.compile(request)
    }
    fn dispatch(
        &self,
        request: &crate::KernelDispatchRequest,
    ) -> Result<crate::KernelOutput, crate::KernelError> {
        CpuBackend.dispatch(request)
    }
    fn measure(
        &self,
        request: &crate::KernelMeasurementRequest,
    ) -> Result<crate::KernelMeasurement, crate::KernelError> {
        CpuBackend.measure(request)
    }
    fn name(&self) -> &str {
        "accelerate-reference"
    }
}
