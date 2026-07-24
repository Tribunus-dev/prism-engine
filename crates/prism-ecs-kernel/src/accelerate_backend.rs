use crate::{
    BackendKind, CpuBackend, KernelBackend, KernelDescriptor, ResidentKernelDispatchRequest,
};

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

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
        #[cfg(target_os = "macos")]
        if let Some(output) = dispatch_sgemv(request)? {
            return Ok(output);
        }
        CpuBackend.dispatch(request)
    }
    fn dispatch_resident<'a>(
        &self,
        request: ResidentKernelDispatchRequest<'a>,
    ) -> Result<crate::KernelOutput, crate::KernelError> {
        #[cfg(target_os = "macos")]
        if let Some(output) = dispatch_sgemv_resident(request.artifact, request.inputs)? {
            return Ok(output);
        }
        // This is deliberately explicit: non-Accelerate variants retain the
        // compatibility path until they have a native borrowed implementation.
        let owned = crate::KernelDispatchRequest {
            artifact: request.artifact.clone(),
            inputs: request.inputs.iter().map(|input| input.to_vec()).collect(),
            bindings: request.bindings.to_vec(),
        };
        self.dispatch(&owned)
    }
    fn measure(
        &self,
        request: &crate::KernelMeasurementRequest,
    ) -> Result<crate::KernelMeasurement, crate::KernelError> {
        if request.iterations == 0 {
            return Err(crate::KernelError::MeasurementFailed(
                "iterations must be nonzero".into(),
            ));
        }
        #[cfg(target_os = "macos")]
        {
            let inputs: Vec<&[u8]> = request.inputs.iter().map(Vec::as_slice).collect();
            if matches!(
                request
                    .artifact
                    .payloads
                    .first()
                    .map(|p| &p.descriptor.variant),
                Some(crate::KernelVariant::FP16GEMV)
            ) {
                let mut samples = Vec::with_capacity(request.iterations as usize);
                for _ in 0..request.iterations {
                    let start = std::time::Instant::now();
                    dispatch_sgemv_resident(&request.artifact, &inputs)?.ok_or_else(|| {
                        crate::KernelError::UnsupportedBackend(
                            "Accelerate artifact variant is not resident-dispatchable".into(),
                        )
                    })?;
                    samples.push(start.elapsed().as_nanos() as f64);
                }
                return Ok(measurement(&samples));
            }
        }
        CpuBackend.measure(request)
    }
    fn name(&self) -> &str {
        "accelerate-reference"
    }
}

fn measurement(samples: &[f64]) -> crate::KernelMeasurement {
    let avg = samples.iter().sum::<f64>() / samples.len() as f64;
    crate::KernelMeasurement {
        avg_time_ns: avg,
        min_time_ns: samples.iter().copied().fold(f64::INFINITY, f64::min),
        max_time_ns: samples.iter().copied().fold(0.0, f64::max),
        bandwidth_gbps: 0.0,
    }
}

#[cfg(target_os = "macos")]
fn dispatch_sgemv(
    request: &crate::KernelDispatchRequest,
) -> Result<Option<crate::KernelOutput>, crate::KernelError> {
    let inputs: Vec<&[u8]> = request.inputs.iter().map(Vec::as_slice).collect();
    dispatch_sgemv_resident(&request.artifact, &inputs)
}

#[cfg(target_os = "macos")]
fn dispatch_sgemv_resident(
    artifact: &crate::KernelArtifact,
    inputs: &[&[u8]],
) -> Result<Option<crate::KernelOutput>, crate::KernelError> {
    use half::f16;
    let Some(payload) = artifact.payloads.first() else {
        return Ok(None);
    };
    if !matches!(payload.descriptor.variant, crate::KernelVariant::FP16GEMV) || inputs.len() < 2 {
        return Ok(None);
    }
    let weights = inputs[0];
    let input = inputs[1];
    if weights.len() % 2 != 0 || input.len() % 4 != 0 {
        return Err(crate::KernelError::BindingMismatch(
            "Accelerate FP16 GEMV buffers are unaligned".into(),
        ));
    }
    let n = input.len() / 4;
    if n == 0 || weights.len() % (n * 2) != 0 {
        return Err(crate::KernelError::BindingMismatch(
            "Accelerate GEMV shape mismatch".into(),
        ));
    }
    let m = weights.len() / (n * 2);
    let weights_f32: Vec<f32> = weights
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    let input_f32: Vec<f32> = input
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mut output = vec![0.0f32; m];
    let started = std::time::Instant::now();
    unsafe {
        cblas_sgemv(
            101,
            111,
            m as i32,
            n as i32,
            1.0,
            weights_f32.as_ptr(),
            n as i32,
            input_f32.as_ptr(),
            1,
            0.0,
            output.as_mut_ptr(),
            1,
        );
    }
    Ok(Some(crate::KernelOutput {
        outputs: vec![output.iter().flat_map(|v| v.to_ne_bytes()).collect()],
        dispatch_time_ns: started.elapsed().as_nanos() as u64,
    }))
}
