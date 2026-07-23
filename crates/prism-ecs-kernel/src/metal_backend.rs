use sha2::{Digest, Sha256};

use crate::{
    BackendKind, KernelArtifact, KernelBackend, KernelCompileRequest, KernelDescriptor,
    KernelDispatchRequest, KernelError, KernelManifest, KernelMeasurement,
    KernelMeasurementRequest, KernelOutput, KernelPayload,
};

/// Portable Metal-facing backend. Compilation is deterministic and platform
/// independent; dispatch uses the native implementation only on macOS.
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackend;

pub const FP16_GEMV_MSL: &str = "kernel void fp16_gemv() {}";
pub const FP16_MATMUL_MSL: &str = "kernel void fp16_matmul() {}";
pub const INT8_GEMV_MSL: &str = "kernel void int8_gemv() {}";
pub const NF4_TILE640_GEMV_MSL: &str = "kernel void nf4_tile640_gemv() {}";
pub const TERNARY_TILE640_GEMV_MSL: &str = "kernel void ternary_tile640_gemv() {}";
pub const BUILTIN_KERNELS: &[&str] = &[
    FP16_GEMV_MSL,
    FP16_MATMUL_MSL,
    INT8_GEMV_MSL,
    NF4_TILE640_GEMV_MSL,
    TERNARY_TILE640_GEMV_MSL,
];

impl MetalBackend {
    pub fn new() -> Self {
        Self
    }
}

impl KernelBackend for MetalBackend {
    fn validate(&self, descriptor: &KernelDescriptor) -> Result<(), KernelError> {
        if descriptor.backend != BackendKind::Metal {
            return Err(KernelError::ValidationFailed(
                "descriptor is not Metal-targeted".into(),
            ));
        }
        Ok(())
    }

    fn compile(&self, request: &KernelCompileRequest) -> Result<KernelArtifact, KernelError> {
        self.validate(&request.descriptor)?;
        let source_digest = hex::encode(Sha256::digest(&request.source));
        let binary = request.source.clone();
        let binary_digest = hex::encode(Sha256::digest(&binary));
        let mut descriptor = request.descriptor.clone();
        descriptor.source_digest = source_digest;
        descriptor.binary_digest = binary_digest;
        let manifest = KernelManifest {
            kernels: vec![descriptor.clone()],
            fusion_plan: None,
            manifest_digest: String::new(),
        };
        Ok(KernelArtifact {
            payloads: vec![KernelPayload { binary, descriptor }],
            manifest,
            artifact_digest: hex::encode(Sha256::digest(&request.source)),
        })
    }

    fn dispatch(&self, request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        {
            return crate::metal_dispatch::dispatch_artifact(request);
        }
        #[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
        {
            let _ = request;
            Err(KernelError::UnsupportedBackend(
                "Metal dispatch requires macOS with the metal-dispatch feature".into(),
            ))
        }
    }

    fn measure(
        &self,
        request: &KernelMeasurementRequest,
    ) -> Result<KernelMeasurement, KernelError> {
        if request.iterations == 0 {
            return Err(KernelError::MeasurementFailed(
                "iterations must be nonzero".into(),
            ));
        }
        let mut samples = Vec::with_capacity(request.iterations as usize);
        for _ in 0..request.iterations {
            let start = std::time::Instant::now();
            self.dispatch(&KernelDispatchRequest {
                artifact: request.artifact.clone(),
                inputs: request.inputs.clone(),
                bindings: vec![],
            })?;
            samples.push(start.elapsed().as_nanos() as f64);
        }
        let avg = samples.iter().sum::<f64>() / samples.len() as f64;
        Ok(KernelMeasurement {
            avg_time_ns: avg,
            min_time_ns: samples.iter().copied().fold(f64::INFINITY, f64::min),
            max_time_ns: samples.iter().copied().fold(0.0, f64::max),
            bandwidth_gbps: 0.0,
        })
    }

    fn name(&self) -> &str {
        "metal"
    }
}
