use sha2::{Digest, Sha256};

use crate::{
    BackendKind, KernelArtifact, KernelBackend, KernelCompileRequest, KernelDescriptor,
    KernelDispatchRequest, KernelError, KernelManifest, KernelMeasurement,
    KernelMeasurementRequest, KernelOutput, KernelPayload, ResidentKernelDispatchRequest,
};

/// Portable Metal-facing backend. Compilation is deterministic and platform
/// independent; dispatch uses the native implementation only on macOS.
#[derive(Debug, Default, Clone, Copy)]
pub struct MetalBackend;

pub const FP16_GEMV_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void fp16_gemv(device const half* weights [[buffer(0)]],
                      device const float* input [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& m [[buffer(3)]],
                      constant uint& n [[buffer(4)]],
                      uint row [[thread_position_in_grid]]) {
    if (row >= m) return;
    float acc = 0.0f;
    for (uint col = 0; col < n; ++col) acc += float(weights[row * n + col]) * input[col];
    output[row] = acc;
}
"#;
pub const FP16_MATMUL_MSL: &str = FP16_GEMV_MSL;
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
        let binary = compile_msl_payload(&request.source)?;
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

    fn dispatch_resident<'a>(
        &self,
        request: ResidentKernelDispatchRequest<'a>,
    ) -> Result<KernelOutput, KernelError> {
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        {
            return crate::metal_dispatch::dispatch_artifact_resident(
                request.artifact,
                request.inputs,
                request.bindings,
            );
        }
        #[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
        {
            let _ = request;
            Err(KernelError::UnsupportedBackend(
                "resident Metal dispatch requires macOS with the metal-dispatch feature".into(),
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
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        let resident_inputs: Vec<&[u8]> = request.inputs.iter().map(Vec::as_slice).collect();
        for _ in 0..request.iterations {
            let start = std::time::Instant::now();
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            {
                crate::metal_dispatch::dispatch_artifact_resident(
                    &request.artifact,
                    &resident_inputs,
                    &[],
                )?;
            }
            #[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
            {
                self.dispatch(&KernelDispatchRequest {
                    artifact: request.artifact.clone(),
                    inputs: request.inputs.clone(),
                    bindings: vec![],
                })?;
            }
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

fn compile_msl_payload(source: &[u8]) -> Result<Vec<u8>, KernelError> {
    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    {
        use std::{fs, process::Command};
        let nonce = format!(
            "prism-metal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(nonce);
        fs::create_dir_all(&root).map_err(|e| KernelError::CompilationFailed(e.to_string()))?;
        let msl = root.join("kernel.metal");
        let air = root.join("kernel.air");
        let lib = root.join("kernel.metallib");
        fs::write(&msl, source).map_err(|e| KernelError::CompilationFailed(e.to_string()))?;
        let metal = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&msl)
            .arg("-o")
            .arg(&air)
            .output()
            .map_err(|e| KernelError::CompilationFailed(format!("launch xcrun metal: {e}")))?;
        if !metal.status.success() {
            let _ = fs::remove_dir_all(&root);
            return Err(KernelError::CompilationFailed(
                String::from_utf8_lossy(&metal.stderr).into_owned(),
            ));
        }
        let metallib = Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(&air)
            .arg("-o")
            .arg(&lib)
            .output()
            .map_err(|e| KernelError::CompilationFailed(format!("launch xcrun metallib: {e}")))?;
        if !metallib.status.success() {
            let _ = fs::remove_dir_all(&root);
            return Err(KernelError::CompilationFailed(
                String::from_utf8_lossy(&metallib.stderr).into_owned(),
            ));
        }
        let bytes = fs::read(&lib).map_err(|e| KernelError::CompilationFailed(e.to_string()))?;
        let _ = fs::remove_dir_all(&root);
        return Ok(bytes);
    }
    #[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
    {
        Ok(source.to_vec())
    }
}
