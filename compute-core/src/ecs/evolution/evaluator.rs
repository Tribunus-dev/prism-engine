//! CandidateEvaluator trait and supporting types.
//!
//! Defines the evaluation pipeline for evolutionary search candidates:
//! static validation → compilation → numerical validation → performance measurement.
//! Each stage produces a typed receipt.

use crate::ecs::canonical::identity::*;
use crate::ecs::evolution::foundation::{
    EvolutionCandidate, NumericalReceipt, PerformanceReceipt, StaticValidationReceipt,
};
use crate::ecs::nf4tile640::{dequant_matmul_reference, pack_nf4_weights};

// ── Supporting types ─────────────────────────────────────────────────────────

/// Performance workload description.
#[derive(Debug, Clone)]
pub struct Workload {
    pub tensor_id: LogicalTensorId,
    pub shape: Vec<usize>,
    pub repetitions: usize,
}

/// A compiled candidate ready for numerical validation and measurement.
#[derive(Debug, Clone)]
pub struct CompiledCandidate {
    pub candidate_id: CandidateId,
    pub compiled_bytes: Vec<u8>,
    pub compile_duration_ms: u64,
}

#[cfg(feature = "metal-dispatch")]
#[repr(C, align(16))]
struct Nf4Tile640DispatchParams {
    abi_version: u32,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    reserved: [u32; 3],
}

#[cfg(feature = "metal-dispatch")]
struct Nf4Fixture {
    input: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    reference: Vec<f32>,
    m: usize,
    k: usize,
    n: usize,
}

#[cfg(feature = "metal-dispatch")]
fn nf4_fixture() -> Result<Nf4Fixture, String> {
    let m = 2;
    let k = 4;
    let n = 640;
    let input = vec![0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25];
    let weights: Vec<f32> = (0..k * n).map(|i| ((i / n) as f32 + 1.0) * 0.01).collect();
    let (codes, scales, biases, _, _) = pack_nf4_weights(&weights, k, n);
    let mut reference = vec![0.0; m * n];
    dequant_matmul_reference(&input, &codes, &scales, &biases, m, k, n, &mut reference)?;
    Ok(Nf4Fixture {
        input,
        codes,
        scales,
        biases,
        reference,
        m,
        k,
        n,
    })
}

// ── Trait ────────────────────────────────────────────────────────────────────

/// Evaluator for search candidates — compiles, validates, measures.
#[allow(unused_variables)]
pub trait CandidateEvaluator {
    /// Return the set of codec formats this evaluator supports.
    fn supported_formats(&self) -> Vec<CodecFamily> {
        Vec::new()
    }

    /// Validate a candidate against static constraints (ABI, device limits).
    fn validate_static(
        &self,
        candidate: &mut EvolutionCandidate,
    ) -> Result<StaticValidationReceipt, String>;

    /// Compile a validated candidate into runnable form.
    fn compile(&self, candidate: &EvolutionCandidate) -> Result<CompiledCandidate, String>;

    /// Validate numerical correctness against a CPU reference.
    fn validate_numerical(
        &self,
        candidate: &mut EvolutionCandidate,
        compiled: &CompiledCandidate,
    ) -> Result<NumericalReceipt, String>;

    /// Measure performance on a target workload.
    fn measure(
        &self,
        candidate: &mut EvolutionCandidate,
        compiled: &CompiledCandidate,
        workload: &Workload,
    ) -> Result<PerformanceReceipt, String>;
}

// ── MetalCandidateEvaluator ────────────────────────────────────────────────────

use crate::ecs::canonical::kernel_abi::KernelSemanticId;
use crate::ecs::metal_backend::catalogue_source_for;
use crate::ecs::plan::CodecFamily;

/// Metal candidate evaluator — compiles through MetalBackendCompiler,
/// dispatches on GPU, validates numerically against CPU reference,
/// and measures performance.
///
/// Plan Section 9: "MetalCandidateEvaluator compiles through
/// MetalBackendCompiler, dispatches through a controlled runner, uses
/// Metal validation, compares against CPU reference output."
pub struct MetalCandidateEvaluator {
    #[cfg(feature = "metal-dispatch")]
    device: Option<metal::Device>,
    #[cfg(not(feature = "metal-dispatch"))]
    _private: (),
}

impl MetalCandidateEvaluator {
    #[cfg(feature = "metal-dispatch")]
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        let device = metal::Device::system_default();
        #[cfg(not(target_os = "macos"))]
        let device = None;
        Self { device }
    }

    #[cfg(not(feature = "metal-dispatch"))]
    pub fn new() -> Self {
        Self { _private: () }
    }

    fn validate_static_impl(&self, candidate: &mut EvolutionCandidate) -> StaticValidationReceipt {
        let mut violations = Vec::new();

        // Validate tile dimensions against Metal limits
        if candidate.genome.metal_geometry.threadgroup_width > 1024 {
            violations.push("threadgroup_width exceeds Metal limit of 1024".into());
        }
        if candidate.genome.metal_geometry.threadgroup_height > 1024 {
            violations.push("threadgroup_height exceeds Metal limit of 1024".into());
        }
        if candidate.genome.metal_geometry.threadgroup_depth > 1024 {
            violations.push("threadgroup_depth exceeds Metal limit of 1024".into());
        }
        let geometry = &candidate.genome.metal_geometry;
        let total_threads = (geometry.threadgroup_width as u64)
            .saturating_mul(geometry.threadgroup_height as u64)
            .saturating_mul(geometry.threadgroup_depth as u64);
        if total_threads > 1024 {
            violations.push(format!(
                "threadgroup size {} exceeds Metal maxTotalThreadsPerThreadgroup 1024",
                total_threads
            ));
        }
        if geometry.simd_width == 0 || geometry.threadgroup_width % geometry.simd_width != 0 {
            violations.push("threadgroup_width must be a positive multiple of simd_width".into());
        }
        if geometry.grid_width == 0 || geometry.grid_height == 0 {
            violations.push("grid dimensions must be non-zero".into());
        }
        // Grid dimensions
        if geometry.grid_width > 65536 || geometry.grid_height > 65536 {
            violations.push("grid dimensions exceed Metal maxGridSize of 65536".into());
        }
        // SIMD width must be 32 for Apple GPU
        if geometry.simd_width != 32 {
            violations.push("simd_width must be 32 for Apple GPU".into());
        }
        // Threadgroup staging memory
        let max_threadgroup_memory = 32768u64;
        if candidate.genome.memory_config.threadgroup_staging > max_threadgroup_memory {
            violations.push(format!(
                "threadgroup_staging {} exceeds maxThreadgroupMemory {}",
                candidate.genome.memory_config.threadgroup_staging, max_threadgroup_memory,
            ));
        }
        // CodecFamily-specific constraints
        match candidate.genome.representation {
            CodecFamily::Ternary => {
                let group_size = candidate.genome.packing.group_size;
                if group_size == 0 || group_size % 4 != 0 {
                    violations.push(format!(
                        "Ternary group_size {} must be non-zero and multiple of 4 for SIMD packing",
                        group_size,
                    ));
                }
            }
            CodecFamily::Nf4 => {
                let group_size = candidate.genome.packing.group_size;
                if group_size == 0 || group_size % 32 != 0 {
                    violations.push(format!(
                        "NF4 group_size {} must be non-zero and multiple of 32",
                        group_size,
                    ));
                }
            }
            _ => {}
        }

        StaticValidationReceipt {
            candidate_id: candidate.candidate_id.clone(),
            passed: violations.is_empty(),
            violations,
            validated_at: format!(
                "{:020}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ),
        }
    }

    #[cfg(feature = "metal-dispatch")]
    fn compile_pipeline(&self, source: &[u8]) -> Result<metal::ComputePipelineState, String> {
        let device = self.device.as_ref().ok_or("no Metal device available")?;
        let source =
            std::str::from_utf8(source).map_err(|e| format!("invalid Metal source: {e}"))?;
        let library = device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal library compile failed: {e}"))?;
        let function = library
            .get_function(
                "dequant_mul_nf4tile640",
                None::<metal::FunctionConstantValues>,
            )
            .map_err(|e| format!("Metal kernel lookup failed: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Metal pipeline creation failed: {e}"))
    }

    #[cfg(feature = "metal-dispatch")]
    fn dispatch_fixture(
        &self,
        source: &[u8],
        fixture: &Nf4Fixture,
        repetitions: usize,
    ) -> Result<Vec<f32>, String> {
        let pipeline = self.compile_pipeline(source)?;
        self.dispatch_pipeline(&pipeline, fixture, repetitions)
    }

    #[cfg(feature = "metal-dispatch")]
    fn dispatch_pipeline(
        &self,
        pipeline: &metal::ComputePipelineState,
        fixture: &Nf4Fixture,
        repetitions: usize,
    ) -> Result<Vec<f32>, String> {
        let device = self.device.as_ref().ok_or("no Metal device available")?;

        let buffer = |bytes: &[u8]| {
            device.new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };
        let input_bytes = unsafe {
            std::slice::from_raw_parts(
                fixture.input.as_ptr() as *const u8,
                fixture.input.len() * std::mem::size_of::<f32>(),
            )
        };
        let scale_bytes = unsafe {
            std::slice::from_raw_parts(
                fixture.scales.as_ptr() as *const u8,
                fixture.scales.len() * std::mem::size_of::<f32>(),
            )
        };
        let bias_bytes = unsafe {
            std::slice::from_raw_parts(
                fixture.biases.as_ptr() as *const u8,
                fixture.biases.len() * std::mem::size_of::<f32>(),
            )
        };
        let codes_buf = buffer(&fixture.codes);
        let scales_buf = buffer(scale_bytes);
        let biases_buf = buffer(bias_bytes);
        let input_buf = buffer(input_bytes);
        let output_buf = device.new_buffer(
            (fixture.reference.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let params = Nf4Tile640DispatchParams {
            abi_version: 1,
            m: fixture.m as u32,
            k: fixture.k as u32,
            n: fixture.n as u32,
            group_size: 128,
            reserved: [0; 3],
        };
        let params_buf = buffer(unsafe {
            std::slice::from_raw_parts(
                &params as *const _ as *const u8,
                std::mem::size_of::<Nf4Tile640DispatchParams>(),
            )
        });

        let queue = device.new_command_queue();
        for _ in 0..repetitions.max(1) {
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&codes_buf), 0);
            encoder.set_buffer(1, Some(&scales_buf), 0);
            encoder.set_buffer(2, Some(&biases_buf), 0);
            encoder.set_buffer(3, Some(&input_buf), 0);
            encoder.set_buffer(4, Some(&output_buf), 0);
            encoder.set_buffer(5, Some(&params_buf), 0);
            encoder.dispatch_threads(
                metal::MTLSize::new(fixture.n as u64, fixture.m as u64, 1),
                metal::MTLSize::new(16, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
        }
        let ptr = output_buf.contents() as *const f32;
        Ok((0..fixture.reference.len())
            .map(|i| unsafe { *ptr.add(i) })
            .collect())
    }
}

impl CandidateEvaluator for MetalCandidateEvaluator {
    fn supported_formats(&self) -> Vec<CodecFamily> {
        vec![CodecFamily::Nf4, CodecFamily::Int8, CodecFamily::Ternary]
    }

    fn validate_static(
        &self,
        candidate: &mut EvolutionCandidate,
    ) -> Result<StaticValidationReceipt, String> {
        let receipt = self.validate_static_impl(candidate);
        candidate.correctness_receipt = Some(receipt.clone());
        Ok(receipt)
    }

    fn compile(&self, candidate: &EvolutionCandidate) -> Result<CompiledCandidate, String> {
        let (semantic_id, entry_point) = match candidate.genome.kernel_variant.as_str() {
            "prism.linear.nf4.v1" | "tile640_gemv" | "gemv_nf4_tile640" => (
                KernelSemanticId("prism.nf4tile640.dequant_mul.v1".into()),
                "dequant_mul_nf4tile640",
            ),
            other => (KernelSemanticId(other.to_string()), other),
        };

        #[cfg(not(feature = "metal-dispatch"))]
        {
            let _ = catalogue_source_for(&semantic_id);
            return Err("Metal compilation requires metal-dispatch feature".into());
        }

        #[cfg(feature = "metal-dispatch")]
        {
            let source = catalogue_source_for(&semantic_id)
                .ok_or_else(|| format!("no catalogue entry for {:?}", semantic_id))?;
            let device = self.device.as_ref().ok_or("no Metal device available")?;
            let start = std::time::Instant::now();
            let library = device
                .new_library_with_source(&source, &metal::CompileOptions::new())
                .map_err(|e| format!("Metal compile failed: {e}"))?;
            let _function = library
                .get_function(entry_point, None)
                .map_err(|e| format!("kernel not found: {e}"))?;
            let dur = start.elapsed().as_millis() as u64;
            Ok(CompiledCandidate {
                candidate_id: candidate.candidate_id.clone(),
                compiled_bytes: source.into_bytes(),
                compile_duration_ms: dur,
            })
        }
    }

    fn validate_numerical(
        &self,
        candidate: &mut EvolutionCandidate,
        compiled: &CompiledCandidate,
    ) -> Result<NumericalReceipt, String> {
        #[cfg(not(feature = "metal-dispatch"))]
        {
            let _ = (candidate, compiled);
            return Err("numerical validation requires metal-dispatch".into());
        }
        #[cfg(feature = "metal-dispatch")]
        {
            let fixture = nf4_fixture()?;
            let output = self.dispatch_fixture(&compiled.compiled_bytes, &fixture, 1)?;
            let mut max_abs: f64 = 0.0;
            let mut max_rel: f64 = 0.0;
            for (actual, expected) in output.iter().zip(&fixture.reference) {
                let abs = (*actual as f64 - *expected as f64).abs();
                max_abs = max_abs.max(abs);
                max_rel = max_rel.max(abs / (*expected as f64).abs().max(1e-8));
            }
            let threshold = 0.05;
            let receipt = NumericalReceipt {
                candidate_id: compiled.candidate_id.clone(),
                passed: max_abs <= threshold,
                max_absolute_error: max_abs,
                max_relative_error: max_rel,
                threshold,
            };
            candidate.quality_receipt = Some(receipt.clone());
            Ok(receipt)
        }
    }

    fn measure(
        &self,
        candidate: &mut EvolutionCandidate,
        compiled: &CompiledCandidate,
        workload: &Workload,
    ) -> Result<PerformanceReceipt, String> {
        #[cfg(not(feature = "metal-dispatch"))]
        {
            let _ = (candidate, compiled, workload);
            return Err("performance measurement requires metal-dispatch".into());
        }
        #[cfg(feature = "metal-dispatch")]
        {
            let fixture = nf4_fixture()?;
            let repetitions = workload.repetitions.max(1);
            // Compile exactly once. Warm-up and measured repetitions reuse the
            // same pipeline so the receipt reflects execution rather than
            // repeatedly paying Metal library and PSO creation costs.
            let pipeline = self.compile_pipeline(&compiled.compiled_bytes)?;
            self.dispatch_pipeline(&pipeline, &fixture, 1)?;
            let start = std::time::Instant::now();
            self.dispatch_pipeline(&pipeline, &fixture, repetitions)?;
            let total_ns = start.elapsed().as_nanos() as u64;
            let per_dispatch = (total_ns / repetitions as u64).max(1);
            let receipt = PerformanceReceipt {
                candidate_id: compiled.candidate_id.clone(),
                latency_p50_ns: per_dispatch,
                latency_p95_ns: per_dispatch,
                encode_time_ns: 0,
                sync_time_ns: total_ns,
                memory_traffic_bytes: (fixture.codes.len()
                    + (fixture.scales.len() + fixture.biases.len()) * 4
                    + fixture.input.len() * 4
                    + fixture.reference.len() * 4) as u64,
                energy_uj: None,
                repetitions,
            };
            candidate.performance_receipt = Some(receipt.clone());
            Ok(receipt)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::cimage::PhysicalTileLayout;
    use crate::ecs::evolution::foundation::{
        CandidateGenome, CandidateStatus, DecompositionStrategy, MemoryConfig, MetalGeometry,
    };
    use crate::ecs::plan::CodecFamily;

    fn make_test_candidate() -> EvolutionCandidate {
        EvolutionCandidate {
            candidate_id: crate::ecs::canonical::identity::CandidateId("test-candidate".into()),
            parent_ids: vec![],
            generation: 0,
            genome: CandidateGenome {
                representation: CodecFamily::Nf4,
                packing: PhysicalTileLayout {
                    tile_m: 64,
                    tile_n: 64,
                    tiles_per_row: 0,
                    total_tiles: 0,
                    padded_cols: 0,
                    group_size: 32,
                    groups_per_tile: 0,
                    packed_bytes_per_tile: 0,
                    metadata_f32_per_tile: 0,
                },
                metal_geometry: MetalGeometry {
                    grid_width: 1,
                    grid_height: 1,
                    simd_width: 32,
                    threadgroup_width: 256,
                    threadgroup_height: 1,
                    threadgroup_depth: 1,
                },
                decomposition: DecompositionStrategy::Sequential,
                memory_config: MemoryConfig {
                    vector_width: 32,
                    cache_policy: "writeback".into(),
                    threadgroup_staging: 32768,
                },
                fusion_strategy: None,
                engram_config: None,
                kernel_variant: "prism.linear.nf4.v1".into(),
            },
            compiled_artifacts: vec![],
            correctness_receipt: None,
            quality_receipt: None,
            performance_receipt: None,
            fitness: None,
            status: CandidateStatus::Created,
        }
    }

    #[test]
    fn test_metal_validator_rejects_bad_geometry() {
        let evaluator = MetalCandidateEvaluator::new();
        let mut candidate = make_test_candidate();
        candidate.genome.metal_geometry.threadgroup_width = 99999;
        let receipt = evaluator.validate_static(&mut candidate).unwrap();
        assert!(!receipt.passed);
        assert!(!receipt.violations.is_empty());
    }

    #[test]
    fn test_metal_compile_requires_feature() {
        let _evaluator = MetalCandidateEvaluator::new();
        let _candidate = make_test_candidate();
        #[cfg(not(feature = "metal-dispatch"))]
        assert!(_evaluator.compile(&_candidate).is_err());
    }

    #[test]
    fn test_metal_validates_all_axes() {
        let evaluator = MetalCandidateEvaluator::new();
        let mut candidate = make_test_candidate();

        let receipt = evaluator.validate_static(&mut candidate).unwrap();
        assert!(receipt.passed);
        assert!(receipt.violations.is_empty());

        candidate.genome.metal_geometry.threadgroup_height = 2048;
        let receipt = evaluator.validate_static(&mut candidate).unwrap();
        assert!(!receipt.passed);
        assert!(receipt
            .violations
            .iter()
            .any(|v| v.contains("threadgroup_height")));

        candidate.genome.metal_geometry.threadgroup_height = 1;
        candidate.genome.metal_geometry.threadgroup_depth = 2048;
        let receipt = evaluator.validate_static(&mut candidate).unwrap();
        assert!(!receipt.passed);
        assert!(receipt
            .violations
            .iter()
            .any(|v| v.contains("threadgroup_depth")));
    }

    #[test]
    fn test_metal_validate_static_returns_timestamp() {
        let evaluator = MetalCandidateEvaluator::new();
        let mut candidate = make_test_candidate();
        let receipt = evaluator.validate_static(&mut candidate).unwrap();
        assert!(!receipt.validated_at.is_empty());
        assert!(receipt.validated_at.parse::<u64>().is_ok());
    }

    #[test]
    fn test_metal_validate_numerical_tolerances() {
        let evaluator = MetalCandidateEvaluator::new();
        #[cfg(feature = "metal-dispatch")]
        {
            let mut ec = make_test_candidate();
            let compiled = evaluator.compile(&ec).unwrap();
            let receipt = evaluator.validate_numerical(&mut ec, &compiled).unwrap();
            assert!(receipt.passed);
            assert!(receipt.max_absolute_error <= receipt.threshold);
            assert!(ec.quality_receipt.is_some());
        }
        #[cfg(not(feature = "metal-dispatch"))]
        {
            let mut ec = make_test_candidate();
            let compiled = CompiledCandidate {
                candidate_id: crate::ecs::canonical::identity::CandidateId("test".into()),
                compiled_bytes: vec![],
                compile_duration_ms: 0,
            };
            assert!(evaluator.validate_numerical(&mut ec, &compiled).is_err());
        }
    }

    #[test]
    fn test_metal_measure_returns_reasonable_latency() {
        let evaluator = MetalCandidateEvaluator::new();
        let workload = Workload {
            tensor_id: crate::ecs::canonical::identity::LogicalTensorId("test".into()),
            shape: vec![64, 64],
            repetitions: 10,
        };
        #[cfg(feature = "metal-dispatch")]
        {
            let mut ec = make_test_candidate();
            let compiled = evaluator.compile(&ec).unwrap();
            let receipt = evaluator.measure(&mut ec, &compiled, &workload).unwrap();
            assert!(receipt.latency_p50_ns > 0);
            assert_eq!(receipt.repetitions, 10);
            assert!(ec.performance_receipt.is_some());
        }
        #[cfg(not(feature = "metal-dispatch"))]
        {
            let mut ec = make_test_candidate();
            let compiled = CompiledCandidate {
                candidate_id: crate::ecs::canonical::identity::CandidateId("test".into()),
                compiled_bytes: vec![],
                compile_duration_ms: 0,
            };
            assert!(evaluator.measure(&mut ec, &compiled, &workload).is_err());
        }
    }

    #[test]
    fn test_metal_candidate_evaluator_constructs() {
        let evaluator = MetalCandidateEvaluator::new();
        #[cfg(feature = "metal-dispatch")]
        {
            assert!(evaluator.device.is_some() || cfg!(not(target_os = "macos")));
        }
        #[cfg(not(feature = "metal-dispatch"))]
        let _ = evaluator;
    }
}
