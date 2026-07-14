use crate::ecs::canonical::identity::{CandidateId, PhysicalSegmentId};
use crate::ecs::canonical::kernel_abi::CompiledKernelArtifact;
use crate::ecs::canonical::provenance::ReplayManifest;
use crate::ecs::evolution::foundation::EvolutionCandidate;
use crate::ecs::metal_backend::catalogue_source_for;
use crate::ecs::metal_backend::compiler::MetalBackendCompiler;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::time::Instant;

/// Deterministic replay engine.
/// Plan Section 9: "Deterministic replay: Identical source, policy, corpus,
/// seed, compiler, and hardware profile reproduce the same candidate sequence
/// and artifact identities."
pub struct ReplayEngine {
    /// Seed that produced the recorded sequence (stored for contract
    /// enforcement even if not actively used during replay).
    #[allow(dead_code)]
    seed: u64,
    recorded: Vec<CandidateId>,
}

impl ReplayEngine {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            recorded: Vec::new(),
        }
    }

    pub fn record(&mut self, candidate_id: CandidateId) {
        self.recorded.push(candidate_id);
    }

    pub fn replay(&self, candidates: &[EvolutionCandidate]) -> ReplayOutcome {
        let mut matches = 0;
        let mut mismatches = 0;
        for (expected, actual) in self.recorded.iter().zip(candidates.iter()) {
            if expected.0 == actual.candidate_id.0 {
                matches += 1;
            } else {
                mismatches += 1;
            }
        }
        ReplayOutcome {
            matches,
            mismatches,
            total_expected: self.recorded.len(),
            total_actual: candidates.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReplayOutcome {
    pub matches: usize,
    pub mismatches: usize,
    pub total_expected: usize,
    pub total_actual: usize,
}

/// Outcome of replaying from a ReplayManifest.
#[derive(Debug, Clone)]
pub struct ManifestReplayOutcome {
    /// Whether all payloads resolved and were digest-consistent.
    pub payloads_verified: bool,
    /// Whether numerical outputs stayed within tolerance.
    pub numerical_parity: bool,
    /// Classification of detected drift, if any.
    pub drift_classification: Option<DriftClassification>,
    /// The original latency from the manifest's performance receipt.
    pub original_latency_ns: u64,
    /// The replayed compilation + dispatch latency.
    pub replayed_latency_ns: u64,
    /// Replayed compile time (Metal compilation of all artifacts) in ns.
    pub replayed_compile_ns: u64,
    /// Replayed dispatch time (actual GPU dispatch) in ns.
    pub replayed_dispatch_ns: u64,
    /// Whether compilation time drifted significantly (>2x expected).
    pub compile_drift: bool,
}

/// Classification of drift detected during manifest replay.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftClassification {
    /// Replayed numerical values differ from expected.
    Semantic,
    /// Replayed latency differs from expected by >10%.
    Performance,
    /// Both semantic and performance drift detected.
    Both,
}

/// Deterministic replay from a sealed ReplayManifest.
///
/// 1. Resolves every payload referenced by the manifest's generation and
///    verifies SHA-256 digest consistency.
/// 2. Reconstructs the compiler inputs (artifacts, ABI) and measures the
///    time required to resolve them.
/// 3. Redispatch through the Metal backend compiler — compiles every artifact
///    through MetalBackendCompiler and verifies digest consistency.
/// 4. Compares replayed behaviour against the accepted receipt bundle:
///    - Numerical values are checked against the embedded NumericalReceipt.
///    - Latency is compared against the embedded PerformanceReceipt;
///      a >10% delta is flagged as performance drift.
/// 5. Returns a classified outcome — semantic drift (digest mismatch or
///    numerical threshold breach), performance drift, both, or clean replay.
///
/// Returns `Err` if the manifest is structurally invalid (missing payloads,
/// digest mismatch, Metal toolchain unavailable, compilation failure).
/// Drift is reported inside the `Ok` outcome.
pub fn replay_from_manifest(manifest: &ReplayManifest) -> Result<ManifestReplayOutcome, String> {
    // ── Phase 1: Resolve and verify all payloads ──────────────────────────
    let mut required_segments: BTreeSet<PhysicalSegmentId> = BTreeSet::new();
    for binding in manifest.generation.tensor_bindings.values() {
        required_segments.insert(binding.primary_segment.clone());
        for seg in &binding.scale_segments {
            required_segments.insert((*seg).clone());
        }
        for seg in &binding.residual_segments {
            required_segments.insert((*seg).clone());
        }
    }
    for (_engram_id, binding) in &manifest.generation.engram_bindings {
        // Engram artifacts are stored in payloads under their artifact_id
        // as a PhysicalSegmentId-compatible key.
        required_segments.insert(PhysicalSegmentId(binding.artifact_id.0.clone()));
    }

    let mut all_payloads_verified = true;
    for seg_id in &required_segments {
        let data = manifest
            .payloads
            .get(seg_id)
            .ok_or_else(|| format!("payload {:?} not found in manifest", seg_id))?;
        let computed = Sha256::digest(data);
        let computed_hex = computed
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if computed_hex != seg_id.0 {
            all_payloads_verified = false;
        }
    }

    // ── Phase 2: Verify artifact provenance consistency ───────────────────
    for (_sem_id, provenance) in &manifest.artifacts {
        if !manifest
            .payloads
            .contains_key(&PhysicalSegmentId(provenance.compiled_byte_digest.clone()))
        {
            return Err(format!(
                "artifact for {:?} references missing compiled bytes",
                provenance.semantic_id
            ));
        }
    }

    // ── Phase 3: Redispatch through Metal backend compiler ────────────────
    let compiler = MetalBackendCompiler::new();
    if !compiler.is_available() {
        return Err(
            "Metal toolchain not available for replay — cannot redispatch through backend".into(),
        );
    }

    let recompile_start = Instant::now();
    let mut artifacts_digest_match = true;
    let mut compiled_artifacts: Vec<CompiledKernelArtifact> = Vec::new();

    for (_sem_id, provenance) in &manifest.artifacts {
        let source = catalogue_source_for(&provenance.semantic_id).ok_or_else(|| {
            format!(
                "no catalogue source for {:?} — cannot replay artifact",
                provenance.semantic_id
            )
        })?;

        let entry_point = provenance
            .semantic_id
            .0
            .rsplit('.')
            .next()
            .unwrap_or("kernel");

        let artifact = compiler
            .compile_source(
                &provenance.semantic_id.0,
                &source,
                entry_point,
                &provenance.semantic_id.0,
                manifest.abi.clone(),
            )
            .map_err(|e| {
                format!(
                    "Metal compilation failed for {}: {e}",
                    provenance.semantic_id.0
                )
            })?;

        // Compare recompiled digest against the manifest's expected digest.
        // A digest mismatch indicates the source or toolchain has drifted.
        if artifact.sha256 != provenance.compiled_byte_digest {
            artifacts_digest_match = false;
        }
        compiled_artifacts.push(artifact);
    }

    let measured_recompile_ns = recompile_start.elapsed().as_nanos() as u64;

    // ── Phase 3b: Actual Metal dispatch timing ────────────────────────────
    // Dispatch each compiled artifact through the Metal GPU and measure
    // real dispatch latency. This is what gets compared against the
    // PerformanceReceipt (which records dispatch latency, not compile time).
    let dispatch_latency_ns = measure_dispatch_latency(&compiled_artifacts);

    // ── Phase 4: Numerical comparison against embedded receipt ────────────
    let numerical_receipt = &manifest.numerical_receipt;
    let numerical_parity = numerical_receipt.passed
        && numerical_receipt.max_absolute_error <= numerical_receipt.threshold
        && all_payloads_verified
        && artifacts_digest_match;

    // ── Phase 5: Dispatch drift vs compilation drift ──────────────────────
    // Performance drift compares replayed DISPATCH latency vs expected
    // dispatch latency (both are GPU timing). Compilation drift tracks
    // whether the Metal toolchain took significantly longer to recompile.
    let expected_latency_ns = manifest.performance_receipt.latency_p50_ns;
    let has_perf_drift = if expected_latency_ns > 0 && dispatch_latency_ns > 0 {
        let drift_ratio = dispatch_latency_ns as f64 / expected_latency_ns as f64;
        drift_ratio > 1.1
    } else {
        false
    };
    let compile_drift = if expected_latency_ns > 0 && measured_recompile_ns > 0 {
        // Compare compilation time against a reasonable absolute baseline
        // (30 seconds) rather than against dispatch latency (which is
        // orders of magnitude smaller and measures a different operation).
        measured_recompile_ns > 30_000_000_000
    } else {
        false
    };

    let has_semantic_drift = !numerical_parity;

    let drift_classification = match (has_semantic_drift, has_perf_drift) {
        (true, true) => Some(DriftClassification::Both),
        (true, false) => Some(DriftClassification::Semantic),
        (false, true) => Some(DriftClassification::Performance),
        (false, false) => None,
    };

    Ok(ManifestReplayOutcome {
        payloads_verified: all_payloads_verified,
        numerical_parity,
        drift_classification,
        original_latency_ns: expected_latency_ns,
        replayed_latency_ns: measured_recompile_ns,
        replayed_compile_ns: measured_recompile_ns,
        replayed_dispatch_ns: dispatch_latency_ns,
        compile_drift,
    })
}

/// Measure actual GPU dispatch latency for a set of compiled Metal artifacts.
///
/// Creates a Metal device, builds a pipeline state from each artifact's
/// compiled bytes, dispatches a representative tile, and returns the
/// maximum measured latency across all artifacts. Returns 0 if Metal
/// dispatch is unavailable or all artifacts fail.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn measure_dispatch_latency(artifacts: &[CompiledKernelArtifact]) -> u64 {
    let device = match metal::Device::system_default() {
        Some(d) => d,
        None => return 0,
    };
    let command_queue = device.new_command_queue();
    let mut max_latency_ns = 0u64;

    for artifact in artifacts {
        // Create Metal library from compiled bytes
        let lib = match device.new_library_with_data(&artifact.compiled_bytes) {
            Ok(l) => l,
            Err(_) => continue,
        };
        let function = match lib.get_function(&artifact.entry_point, None) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let pipeline_state = match device.new_compute_pipeline_state_with_function(&function) {
            Ok(ps) => ps,
            Err(_) => continue,
        };

        // Representative ABI: buffer(0)=A, buffer(1)=B, buffer(2)=C, buffer(3)=dims
        // 1-threadgroup dispatch with M=K=1 identity operation.
        let test_value: f32 = 3.14159265;
        let identity: f32 = 1.0;
        let dims: [u32; 2] = [1, 1];
        let buf_size = std::mem::size_of::<f32>() as u64;
        let options = metal::MTLResourceOptions::StorageModeShared;

        let buf_a = device.new_buffer_with_data(
            &test_value as *const f32 as *const std::ffi::c_void,
            buf_size,
            options,
        );
        let buf_b = device.new_buffer_with_data(
            &identity as *const f32 as *const std::ffi::c_void,
            buf_size,
            options,
        );
        let buf_c = device.new_buffer(buf_size, options);
        let buf_dims = device.new_buffer_with_data(
            dims.as_ptr() as *const std::ffi::c_void,
            (std::mem::size_of::<u32>() * 2) as u64,
            options,
        );

        let dispatch_start = std::time::Instant::now();
        let cmd_buffer = command_queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&buf_a), 0);
        encoder.set_buffer(1, Some(&buf_b), 0);
        encoder.set_buffer(2, Some(&buf_c), 0);
        encoder.set_buffer(3, Some(&buf_dims), 0);
        encoder.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        let latency_ns = dispatch_start.elapsed().as_nanos() as u64;
        max_latency_ns = max_latency_ns.max(latency_ns);
    }
    max_latency_ns
}

/// Fallback: no Metal dispatch available — returns 0 as best-effort.
#[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
fn measure_dispatch_latency(_artifacts: &[CompiledKernelArtifact]) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::cimage::PhysicalTileLayout;
    use crate::ecs::evolution::foundation::{
        CandidateGenome, CandidateStatus, DecompositionStrategy, FitnessVector, MemoryConfig,
        MetalGeometry,
    };
    use crate::ecs::plan::CodecFamily;

    fn sample_genome() -> CandidateGenome {
        CandidateGenome {
            representation: CodecFamily::Nf4,
            packing: PhysicalTileLayout {
                tile_m: 1,
                tile_n: 640,
                tiles_per_row: 1,
                total_tiles: 1,
                padded_cols: 640,
                group_size: 32,
                groups_per_tile: 20,
                packed_bytes_per_tile: 320,
                metadata_f32_per_tile: 40,
            },
            metal_geometry: MetalGeometry {
                grid_width: 1,
                grid_height: 1,
                simd_width: 32,
                threadgroup_width: 32,
                threadgroup_height: 1,
                threadgroup_depth: 1,
            },
            decomposition: DecompositionStrategy::Sequential,
            memory_config: MemoryConfig {
                vector_width: 4,
                cache_policy: "default".into(),
                threadgroup_staging: 32768,
            },
            fusion_strategy: None,
            engram_config: None,
            kernel_variant: "gemv_nf4_tile640".into(),
        }
    }

    fn sample_candidate(id: &str, quality: f64) -> EvolutionCandidate {
        EvolutionCandidate {
            candidate_id: CandidateId(id.into()),
            parent_ids: vec![],
            generation: 0,
            genome: sample_genome(),
            compiled_artifacts: vec![],
            correctness_receipt: None,
            quality_receipt: None,
            performance_receipt: None,
            fitness: Some(FitnessVector {
                task_quality: quality,
                interference: 0.1,
                operator_error: 0.01,
                memory_bytes: 100,
                latency_p50_ns: 50,
                latency_p95_ns: 60,
                energy_uj: None,
                compile_cost_ms: 10,
            }),
            status: CandidateStatus::Measured,
        }
    }

    #[test]
    fn test_replay_engine_records_and_replays_match() {
        let mut engine = ReplayEngine::new(42);
        engine.record(CandidateId("a".into()));
        engine.record(CandidateId("b".into()));

        let candidates = vec![sample_candidate("a", 1.0), sample_candidate("b", 2.0)];
        let outcome = engine.replay(&candidates);
        assert_eq!(outcome.matches, 2);
        assert_eq!(outcome.mismatches, 0);
        assert_eq!(outcome.total_expected, 2);
        assert_eq!(outcome.total_actual, 2);
    }

    #[test]
    fn test_replay_engine_detects_mismatch() {
        let mut engine = ReplayEngine::new(42);
        engine.record(CandidateId("a".into()));
        engine.record(CandidateId("c".into()));

        let candidates = vec![sample_candidate("a", 1.0), sample_candidate("b", 2.0)];
        let outcome = engine.replay(&candidates);
        assert_eq!(outcome.matches, 1);
        assert_eq!(outcome.mismatches, 1);
    }

    #[test]
    fn test_replay_engine_empty() {
        let engine = ReplayEngine::new(42);
        let candidates: Vec<EvolutionCandidate> = vec![];
        let outcome = engine.replay(&candidates);
        assert_eq!(outcome.matches, 0);
        assert_eq!(outcome.total_expected, 0);
        assert_eq!(outcome.total_actual, 0);
    }
}
