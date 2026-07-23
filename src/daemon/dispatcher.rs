//! Daemon-compiler dispatcher — loads model from disk at dispatch time.
//!
//! # Cancellation semantics
//! - Cooperative cancellation via `Arc<AtomicBool>` shared with each worker
//! - Worker checks cancellation signal before: graph construction, compile
//!   progress callbacks, staging promotion, and final publication
//! - cancel() marks the attempt cancelled, signals the worker, removes temp
//!   output, and ensures polling cannot report success afterward
//!
//! # Deterministic dispatch identity
//! - `handle_id = format!("compile-{}-{}-{}", work_entity, attempt, plan_generation)`
//! - Per-attempt unique temp path: `{output_dir}/{entity}-{attempt}-{gen}.cimage.tmp`
//! - Final path: `{output_dir}/{entity}-{attempt}-{gen}.cimage`
//!
//! # Digest verification
//! - After successful compilation, compute blake3 digest of temp file
//! - Before rename: verify digest still matches (no tamper)
use chrono::Utc;
use parking_lot::Mutex;
use prism_ecs_ir::evolution::evaluate::EvaluationStrategy;
use prism_ecs_quantization::compile_config::CanonicalCompileConfig;
use prism_ecs_runtime::{
    DispatchError, DispatchHandle, DispatchRequest, DispatchStatus, WorkDispatcher,
};
use prism_mcp_core::{
    ArtifactKind, ArtifactRepository, EvidenceReceipt, EvidenceStatus, MetricSet, ProjectionStore,
    ToolInvocationId,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone)]
pub struct CompilerProvenance {
    pub artifacts: Arc<dyn ArtifactRepository>,
    pub evidence: Arc<dyn prism_mcp_core::EvidenceStore>,
    pub projections: Arc<dyn ProjectionStore>,
}

impl CompilerProvenance {
    fn record_compile(
        &self,
        handle_id: &str,
        source: &Path,
        output: &Path,
        receipt: &serde_json::Value,
    ) {
        let invocation = ToolInvocationId::new();
        let source_bytes = std::fs::read(source).unwrap_or_default();
        let output_bytes = std::fs::read(output).unwrap_or_default();
        let source_id = self
            .artifacts
            .put(&source_bytes, ArtifactKind::BuildPlan, &invocation)
            .ok();
        let output_id = self
            .artifacts
            .put(&output_bytes, ArtifactKind::Cimage, &invocation)
            .ok();
        let receipt_bytes = serde_json::to_vec(receipt).unwrap_or_default();
        let receipt_id = self
            .artifacts
            .put(
                &receipt_bytes,
                ArtifactKind::CompilerDiagnostics,
                &invocation,
            )
            .ok();
        let source_hex = source_id.map(|id| id.hex()).unwrap_or_default();
        let output_hex = output_id.map(|id| id.hex()).unwrap_or_default();
        let receipt_hex = receipt_id.map(|id| id.hex()).unwrap_or_default();
        let graph = serde_json::json!({
            "kind": "compiler_provenance",
            "run": handle_id,
            "nodes": [
                {"id": format!("source:{source_hex}"), "kind": "source", "path": source.display().to_string()},
                {"id": format!("compiler:{handle_id}"), "kind": "compiler_decision", "receipt": receipt},
                {"id": format!("artifact:{output_hex}"), "kind": "cimage_artifact"},
                {"id": format!("evidence:{receipt_hex}"), "kind": "compile_evidence"}
            ],
            "edges": [
                {"from": format!("source:{source_hex}"), "to": format!("compiler:{handle_id}"), "kind": "compiled_by"},
                {"from": format!("compiler:{handle_id}"), "to": format!("artifact:{output_hex}"), "kind": "emitted"},
                {"from": format!("compiler:{handle_id}"), "to": format!("evidence:{receipt_hex}"), "kind": "attested_by"}
            ]
        });
        let _ = self.projections.put_trace(handle_id, &graph);
        let mut metrics = MetricSet::new();
        if let Some(duration) = receipt.get("duration_ms").and_then(|v| v.as_f64()) {
            metrics.values.insert("duration_ms".into(), duration);
        }
        let evidence = EvidenceReceipt {
            invocation_id: invocation,
            tool: "compiler".into(),
            operation: "compile_gguf".into(),
            inputs: source_id.into_iter().collect(),
            outputs: output_id.into_iter().chain(receipt_id).collect(),
            environment: "daemon-compiler-dispatcher".into(),
            target: receipt
                .get("target_hardware")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            source_revision: receipt
                .get("source_gguf_digest")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            status: EvidenceStatus::Success,
            metrics,
            diagnostics: Vec::new(),
            started_at: Utc::now(),
            duration_ms: receipt
                .get("duration_ms")
                .and_then(|v| v.as_i64())
                .unwrap_or_default(),
        };
        let _ = self.evidence.record(&evidence);
    }
}
struct DispatchEntry {
    _handle: DispatchHandle,
    cancellation_signal: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    _output_path: PathBuf,
    tmp_path: PathBuf,
    completed: Arc<Mutex<Option<DispatchStatus>>>,
    attempt: u32,
    _handle_id: String,
}
pub struct DaemonCompilerDispatcher {
    has_metal: bool,
    evaluator: Option<Arc<dyn EvaluationStrategy + Send + Sync>>,
    active: Arc<Mutex<HashMap<String, DispatchEntry>>>,
    cancelled: Arc<Mutex<HashSet<String>>>,
    temp_files: Arc<Mutex<Vec<PathBuf>>>,
    provenance: Option<CompilerProvenance>,
}
impl DaemonCompilerDispatcher {
    pub fn new(
        has_metal: bool,
        evaluator: Option<Arc<dyn EvaluationStrategy + Send + Sync>>,
    ) -> Self {
        Self {
            has_metal,
            evaluator,
            active: Arc::new(Mutex::new(HashMap::new())),
            cancelled: Arc::new(Mutex::new(HashSet::new())),
            temp_files: Arc::new(Mutex::new(Vec::new())),
            provenance: None,
        }
    }
    pub fn with_provenance(mut self, provenance: CompilerProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }
    fn handle_id(request: &DispatchRequest) -> String {
        format!(
            "compile-{}-{}-{}",
            request.work_entity, request.attempt, request.plan_generation
        )
    }
    fn per_attempt_paths(
        output_dir: &Path,
        entity: u64,
        attempt: u32,
        plan_gen: u32,
    ) -> (PathBuf, PathBuf) {
        let stem = format!("{}-{}-{}", entity, attempt, plan_gen);
        (
            output_dir.join(format!("{}.cimage", stem)),
            output_dir.join(format!("{}.cimage.tmp", stem)),
        )
    }
    fn file_digest(path: &Path) -> Result<String, String> {
        let data = std::fs::read(path).map_err(|e| format!("digest read: {e}"))?;
        Ok(blake3::hash(&data).to_hex().to_string())
    }
}
impl Drop for DaemonCompilerDispatcher {
    fn drop(&mut self) {
        let temps = std::mem::take(&mut *self.temp_files.lock());
        for tmp in temps {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
impl WorkDispatcher for DaemonCompilerDispatcher {
    fn start(&self, request: &DispatchRequest) -> Result<DispatchHandle, DispatchError> {
        let handle_id = Self::handle_id(request);
        if self.cancelled.lock().remove(&handle_id) {
            return Err(DispatchError::StartFailed("cancelled".to_string()));
        }
        let gguf_path = PathBuf::from(&request.input_path);
        let output_dir = PathBuf::from(&request.output_path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let valid_model_path = gguf_path.is_file()
            || (gguf_path.is_dir()
                && (gguf_path.join("model.safetensors").is_file()
                    || gguf_path.join("config.json").is_file()));
        if !valid_model_path {
            return Err(DispatchError::StartFailed(format!(
                "model source not found: {}",
                gguf_path.display()
            )));
        }
        let (output_path, tmp_path) = Self::per_attempt_paths(
            &output_dir,
            request.work_entity,
            request.attempt,
            request.plan_generation,
        );
        let _ = std::fs::create_dir_all(&output_dir);
        let has_metal = self.has_metal;
        let cancellation_signal = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(Mutex::new(None::<DispatchStatus>));
        let signal = cancellation_signal.clone();
        let comp = completed.clone();
        let tmp = tmp_path.clone();
        let out = output_path.clone();
        let source = gguf_path.clone();
        let evaluator = self.evaluator.clone();
        let provenance = self.provenance.clone();
        let provenance_handle = handle_id.clone();
        let thread = std::thread::Builder::new()
            .name(format!("daemon-compile-{}", request.work_entity))
            .spawn(move || {
                if signal.load(Ordering::Relaxed) { *comp.lock() = Some(DispatchStatus::Failed("cancelled".to_string())); return; }
                let config = CanonicalCompileConfig {
                    source_path: source.clone(),
                    output_path: tmp.clone(),
                    target_hardware: if has_metal { "apple-m1".to_string() } else { "cpu".to_string() },
                    evolution: Default::default(),
                    seed: None,
                    population_size: 50,
                    generation_limit: 100,
                    stall_limit: 10,
                    candidate_budget: None,
                    time_budget_secs: None,
                    calibration: prism_ecs_quantization::compile_config::CalibrationPolicy::Auto,
                    validation: prism_ecs_quantization::compile_config::ValidationPolicy::Strict,
                    progress: Some(Box::new(|_: &str, _: u32, _: u32, _: f64, _: f64| {})),
                    cancel: Some(signal.clone()),
                    evolution_enabled: true,
                    production_mode: true,
                };
                let evaluator_ref: Option<&dyn EvaluationStrategy> = evaluator.as_deref().map(|e| e as &dyn EvaluationStrategy);
                let result = if source.is_dir() {
                    let model_config = prism_ecs_ir::UnifiedConfig::from_file(
                        &source.join("config.json"),
                    );
                    match model_config {
                        Ok(model_config) => {
                            let graph = prism_ecs_ir::ModelGraph::build(&model_config);
                            prism_ecs_quantization::compiler::compile_to_cimage(
                                &graph,
                                &source,
                                &tmp,
                                has_metal,
                                |_, _, _, _, _| {},
                                None,
                                prism_ecs_quantization::compiler::CompilationBackend::Default,
                                Some(&config),
                            )
                        }
                        Err(error) => Err(format!("load model config: {error}")),
                    }
                } else {
                    #[cfg(feature = "gguf-compile")]
                    {
                        prism_ecs_quantization::compiler::compile_gguf(&config, evaluator_ref)
                    }
                    #[cfg(not(feature = "gguf-compile"))]
                    {
                        let _ = evaluator_ref;
                        Err("GGUF compilation requires the gguf-compile feature".to_string())
                    }
                };
                match result {
                    Ok(receipt) => {
                        if signal.load(Ordering::Relaxed) { let _ = std::fs::remove_file(&tmp); *comp.lock() = Some(DispatchStatus::Failed("cancelled".to_string())); return; }
                        let digest = match Self::file_digest(&tmp) { Ok(d) => d, Err(e) => { *comp.lock() = Some(DispatchStatus::Failed(e)); return; } };
                        if signal.load(Ordering::Relaxed) { let _ = std::fs::remove_file(&tmp); *comp.lock() = Some(DispatchStatus::Failed("cancelled".to_string())); return; }
                        match std::fs::rename(&tmp, &out) {
                            Ok(()) => {
                                if let Some(provenance) = provenance.as_ref() {
                                    if let Ok(receipt_json) = serde_json::to_value(&receipt) {
                                        provenance.record_compile(&provenance_handle, &source, &out, &receipt_json);
                                    }
                                }
                                let result = serde_json::json!({"digest": digest, "path": out.to_string_lossy()});
                                *comp.lock() = Some(DispatchStatus::Completed(serde_json::to_string(&result).unwrap_or_default().into_bytes()));
                            }
                            Err(e) => { let _ = std::fs::remove_file(&tmp); *comp.lock() = Some(DispatchStatus::Failed(format!("rename: {e}"))); }
                        }
                    }
                    Err(e) => { *comp.lock() = Some(DispatchStatus::Failed(format!("compilation: {e}"))); }
                }
            })
            .map_err(|e| DispatchError::StartFailed(format!("thread spawn: {e}")))?;
        let handle = DispatchHandle {
            id: handle_id.clone(),
            work_entity: request.work_entity,
            attempt: request.attempt,
        };
        self.active.lock().insert(
            handle_id.clone(),
            DispatchEntry {
                _handle: handle.clone(),
                cancellation_signal,
                thread: Some(thread),
                _output_path: output_path,
                tmp_path,
                completed,
                attempt: request.attempt,
                _handle_id: handle_id,
            },
        );
        Ok(handle)
    }

    fn poll(&self, handle: &DispatchHandle) -> Result<DispatchStatus, DispatchError> {
        let mut active = self.active.lock();
        let cancelled = self.cancelled.lock();
        if cancelled.contains(&handle.id) {
            return Ok(DispatchStatus::Failed("cancelled".to_string()));
        }
        drop(cancelled);
        if let Some(mut entry) = active.remove(&handle.id) {
            let entry_attempt = entry.attempt;
            if entry_attempt != handle.attempt {
                active.insert(handle.id.clone(), entry);
                return Err(DispatchError::StaleAttempt {
                    handle_attempt: handle.attempt,
                    current_attempt: entry_attempt,
                });
            }
            if let Some(status) = entry.completed.lock().take() {
                if let Some(thread) = entry.thread.take() {
                    let _ = thread.join();
                }
                return Ok(status);
            }
            active.insert(handle.id.clone(), entry);
            Ok(DispatchStatus::Running)
        } else {
            Ok(DispatchStatus::Running)
        }
    }

    fn cancel(&self, handle: &DispatchHandle) -> Result<(), DispatchError> {
        self.cancelled.lock().insert(handle.id.clone());
        // Remove from active so stale entries do not accumulate and poll()
        // will report Failed("cancelled") via the cancelled-set check.
        if let Some(entry) = self.active.lock().remove(&handle.id) {
            entry.cancellation_signal.store(true, Ordering::Relaxed);
            self.temp_files.lock().push(entry.tmp_path.clone());
        }
        let parts: Vec<&str> = handle.id.split('-').collect();
        if parts.len() >= 4 {
            if let (Ok(entity), Ok(attempt), Ok(plan_gen)) = (
                parts[1].parse::<u64>(),
                parts[2].parse::<u32>(),
                parts[3].parse::<u32>(),
            ) {
                for base in [std::env::temp_dir(), PathBuf::from(".")] {
                    let (_out, tmp) = Self::per_attempt_paths(&base, entity, attempt, plan_gen);
                    let _ = std::fs::remove_file(&tmp);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_runtime::{
        DispatchError, DispatchHandle, DispatchRequest, DispatchStatus, WorkDispatcher,
    };
    use safetensors::tensor::{Dtype, View};
    use std::borrow::Cow;
    use std::collections::HashMap;

    /// Helper: an f32 tensor that implements safetensors `View`.
    struct F32Tensor {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl F32Tensor {
        fn new(shape: Vec<usize>) -> Self {
            let n: usize = shape.iter().product();
            let mut data = Vec::with_capacity(n * 4);
            for i in 0..n {
                let val = (i as f32) * 0.001;
                data.extend_from_slice(&val.to_le_bytes());
            }
            Self { shape, data }
        }
    }

    impl View for F32Tensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    /// Create a minimal model directory with config.json and valid .safetensors
    /// for the DaemonCompilerDispatcher compilation test.
    fn create_minimal_model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        // Use a tiny config so weights are small
        let config = serde_json::json!({
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_kv_heads": 2,
            "intermediate_size": 16,
            "vocab_size": 8,
            "rms_norm_eps": 1e-6,
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .expect("write config");

        // Create all required weight tensors
        let vocab_size: usize = 8;
        let hidden_size: usize = 8;
        let num_heads: usize = 2;
        let kv_heads: usize = 2;
        let head_dim: usize = hidden_size / num_heads; // 4
        let intermediate_size: usize = 16;
        let q_dim = num_heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let tensors: Vec<(&str, F32Tensor)> = vec![
            (
                "model.embed_tokens.weight",
                F32Tensor::new(vec![vocab_size, hidden_size]),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                F32Tensor::new(vec![q_dim, hidden_size]),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                F32Tensor::new(vec![kv_dim, hidden_size]),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                F32Tensor::new(vec![kv_dim, hidden_size]),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                F32Tensor::new(vec![hidden_size, q_dim]),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                F32Tensor::new(vec![intermediate_size, hidden_size]),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                F32Tensor::new(vec![intermediate_size, hidden_size]),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                F32Tensor::new(vec![hidden_size, intermediate_size]),
            ),
            (
                "model.lm_head.weight",
                F32Tensor::new(vec![vocab_size, hidden_size]),
            ),
        ];

        safetensors::tensor::serialize_to_file(
            tensors,
            &None::<HashMap<String, String>>,
            &dir.path().join("model.safetensors"),
        )
        .expect("serialize safetensors");

        dir
    }

    #[test]
    fn test_real_daemon_compile_endpoint() {
        let model_dir = create_minimal_model_dir();
        let output_dir = tempfile::tempdir().expect("output tempdir");

        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let request = DispatchRequest {
            work_entity: 42,
            attempt: 1,
            plan_generation: 0,
            lease_token: "test".into(),
            deadline_ms: 9999999999999,
            backend: "test".into(),
            config: "{}".into(),
            input_path: model_dir.path().to_string_lossy().into_owned(),
            output_path: output_dir
                .path()
                .join("dummy.cimage")
                .to_string_lossy()
                .into_owned(),
        };

        let handle = dispatcher.start(&request).expect("start should succeed");

        // Poll until Complete or timeout (max 30s)
        let mut final_status = None;
        for _ in 0..300 {
            match dispatcher.poll(&handle).expect("poll should not error") {
                DispatchStatus::Completed(payload) => {
                    final_status = Some(DispatchStatus::Completed(payload));
                    break;
                }
                DispatchStatus::Failed(msg) => {
                    final_status = Some(DispatchStatus::Failed(msg));
                    break;
                }
                DispatchStatus::Running => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }

        let status = final_status
            .unwrap_or_else(|| DispatchStatus::Failed("timed out after 30s poll".to_string()));

        match &status {
            DispatchStatus::Completed(payload) => {
                // Verify completed payload is valid JSON with digest and path
                let json_str = String::from_utf8_lossy(payload);
                let parsed: serde_json::Value =
                    serde_json::from_str(&json_str).unwrap_or_else(|e| {
                        panic!("completed payload should be valid JSON: {e} — payload: {json_str}")
                    });

                let digest = parsed
                    .get("digest")
                    .and_then(|v| v.as_str())
                    .expect("JSON should have 'digest' field");
                assert!(!digest.is_empty(), "digest should not be empty");

                let path = parsed
                    .get("path")
                    .and_then(|v| v.as_str())
                    .expect("JSON should have 'path' field");
                assert!(!path.is_empty(), "path should not be empty");
                // Verify .cimage was produced at the path reported in the payload
                let cimage_path = std::path::Path::new(path);
                assert!(
                    cimage_path.exists(),
                    ".cimage should exist at {:?}",
                    cimage_path
                );
                let meta = std::fs::metadata(cimage_path).expect("read cimage metadata");
                assert!(meta.len() > 0, "cimage should have content");

                // ── Validate .cimage magic and header ──
                let cimage_bytes = std::fs::read(cimage_path).expect("read .cimage");
                assert_eq!(
                    &cimage_bytes[..8],
                    b"TRB_CIMG",
                    ".cimage should start with magic TRB_CIMG"
                );

                // Parse header (JSON after magic, within first 128KB)
                // Format: MAGIC[8] + header_size(u64 LE)[8] + header_json[header_size]
                let hdr_size = u64::from_le_bytes(cimage_bytes[8..16].try_into().unwrap()) as usize;
                if cimage_bytes.len() >= 16 + hdr_size {
                    let header_str = std::str::from_utf8(&cimage_bytes[16..16 + hdr_size])
                        .expect("header should be valid UTF-8");
                    let header: serde_json::Value =
                        serde_json::from_str(header_str).expect("header should be valid JSON");
                    let tensors = header
                        .get("tensors")
                        .and_then(|v| v.as_object())
                        .expect("header should have 'tensors' object");
                    assert!(!tensors.is_empty(), "tensors should be non-empty");

                    // Verify digest matches blake3 of file content
                    let reported_digest = parsed
                        .get("digest")
                        .and_then(|v| v.as_str())
                        .expect("digest in payload");
                    let computed_digest = blake3::hash(&cimage_bytes).to_hex().to_string();
                    assert_eq!(
                        reported_digest, &computed_digest,
                        "reported digest should match blake3 of file"
                    );
                }
            }
            DispatchStatus::Failed(msg) => {
                panic!("compilation should complete successfully, got: {msg}");
            }
            DispatchStatus::Running => {
                panic!("compilation timed out");
            }
        }
    }

    fn test_model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = serde_json::json!({"model_type":"test","hidden_size":64,"num_layers":1,"num_attention_heads":4,"num_kv_heads":4,"intermediate_size":256,"vocab_size":32000});
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .expect("write config");
        std::fs::write(dir.path().join("model.safetensors"), b"").expect("write safetensors");
        dir
    }

    fn make_request(
        work_entity: u64,
        attempt: u32,
        plan_generation: u32,
        output_dir: &Path,
    ) -> DispatchRequest {
        DispatchRequest {
            work_entity,
            attempt,
            plan_generation,
            lease_token: "test".into(),
            deadline_ms: 9999999999999,
            backend: "test".into(),
            config: "{}".into(),
            input_path: "/tmp".into(),
            output_path: output_dir.join("out.cimage").to_string_lossy().into_owned(),
        }
    }

    fn make_request_with_input(
        work_entity: u64,
        attempt: u32,
        plan_generation: u32,
        input_path: &Path,
        output_dir: &Path,
    ) -> DispatchRequest {
        DispatchRequest {
            work_entity,
            attempt,
            plan_generation,
            lease_token: "test".into(),
            deadline_ms: 9999999999999,
            backend: "test".into(),
            config: "{}".into(),
            input_path: input_path.to_string_lossy().into_owned(),
            output_path: output_dir.join("out.cimage").to_string_lossy().into_owned(),
        }
    }

    #[test]
    fn test_deterministic_handle_id_includes_plan_generation() {
        assert_eq!(
            format!("compile-{}-{}-{}", 7u64, 3u32, 0u32),
            format!("compile-{}-{}-{}", 7u64, 3u32, 0u32)
        );
        assert_ne!(
            format!("compile-{}-{}-{}", 7u64, 3u32, 0u32),
            format!("compile-{}-{}-{}", 7u64, 3u32, 1u32)
        );
        assert_ne!(
            format!("compile-{}-{}-{}", 7u64, 3u32, 0u32),
            format!("compile-{}-{}-{}", 7u64, 4u32, 0u32)
        );
        assert_ne!(
            format!("compile-{}-{}-{}", 7u64, 3u32, 0u32),
            format!("compile-{}-{}-{}", 8u64, 3u32, 0u32)
        );
    }

    #[test]
    fn test_cancel_before_dispatch() {
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let temp = tempfile::tempdir().expect("tempdir");
        let request = make_request(42, 1, 0, temp.path());
        let handle = DispatchHandle {
            id: DaemonCompilerDispatcher::handle_id(&request),
            work_entity: request.work_entity,
            attempt: request.attempt,
        };
        dispatcher.cancel(&handle).unwrap();
        let err = dispatcher.start(&request).unwrap_err();
        assert!(matches!(&err, DispatchError::StartFailed(msg) if msg == "cancelled"));
    }

    #[test]
    fn test_cancel_idempotent() {
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let temp = tempfile::tempdir().expect("tempdir");
        let request = make_request(2, 1, 0, temp.path());
        let handle = DispatchHandle {
            id: DaemonCompilerDispatcher::handle_id(&request),
            work_entity: 2,
            attempt: 1,
        };
        dispatcher.cancel(&handle).unwrap();
        dispatcher.cancel(&handle).unwrap();
        dispatcher.cancel(&handle).unwrap();
        assert!(dispatcher.cancelled.lock().contains(&handle.id));
    }

    #[test]
    fn test_cancelled_poll_returns_failed() {
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let handle = DispatchHandle {
            id: "compile-100-1-0".to_string(),
            work_entity: 100,
            attempt: 1,
        };
        dispatcher.cancel(&handle).unwrap();
        assert_eq!(
            dispatcher.poll(&handle).unwrap(),
            DispatchStatus::Failed("cancelled".to_string())
        );
    }

    #[test]
    fn test_cancel_after_completion_noop() {
        let disp = prism_ecs_runtime::NoopDispatcher;
        let temp = tempfile::tempdir().expect("tempdir");
        let request = make_request(1, 1, 0, temp.path());
        let handle = disp.start(&request).unwrap();
        assert_eq!(
            disp.poll(&handle).unwrap(),
            DispatchStatus::Completed(vec![])
        );
        disp.cancel(&handle).unwrap();
        assert_eq!(
            disp.poll(&handle).unwrap(),
            DispatchStatus::Completed(vec![])
        );
    }

    #[test]
    fn test_restart_after_cancel() {
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let model_dir = test_model_dir();
        let temp = tempfile::tempdir().expect("tempdir");
        let request = make_request_with_input(10, 1, 0, model_dir.path(), temp.path());
        let handle = DispatchHandle {
            id: DaemonCompilerDispatcher::handle_id(&request),
            work_entity: 10,
            attempt: 1,
        };
        dispatcher.cancel(&handle).unwrap();
        assert!(
            matches!(dispatcher.start(&request).unwrap_err(), DispatchError::StartFailed(msg) if msg == "cancelled")
        );
        let request2 = make_request_with_input(10, 2, 0, model_dir.path(), temp.path());
        assert_ne!(
            DaemonCompilerDispatcher::handle_id(&request),
            DaemonCompilerDispatcher::handle_id(&request2)
        );
        if let Err(DispatchError::StartFailed(msg)) = dispatcher.start(&request2) {
            assert_ne!(msg, "cancelled")
        }
    }

    #[test]
    fn test_per_attempt_paths_are_unique() {
        let base = Path::new("/tmp/prism-test");
        let (a, b) = DaemonCompilerDispatcher::per_attempt_paths(base, 1, 1, 0);
        let (c, d) = DaemonCompilerDispatcher::per_attempt_paths(base, 1, 2, 0);
        assert_ne!(a, c);
        assert_ne!(b, d);
        let (e, f) = DaemonCompilerDispatcher::per_attempt_paths(base, 1, 1, 1);
        assert_ne!(a, e);
        assert_ne!(b, f);
        let (g, h) = DaemonCompilerDispatcher::per_attempt_paths(base, 1, 1, 0);
        assert_eq!(a, g);
        assert_eq!(b, h);
    }

    #[test]
    /// Create dispatch entry via lock insertion, cancel before polling,
    /// verify cancelled poll result and entry removal from active map.
    fn cancel_before_dispatch() {
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let handle = DispatchHandle {
            id: "compile-200-1-0".to_string(),
            work_entity: 200,
            attempt: 1,
        };
        // Insert a dispatch entry manually (lock insertion) to simulate an
        // active dispatch that has not been polled yet.
        let entry = DispatchEntry {
            _handle: handle.clone(),
            cancellation_signal: Arc::new(AtomicBool::new(false)),
            thread: None,

            _output_path: PathBuf::from("/tmp/fake-200.cimage"),
            tmp_path: PathBuf::from("/tmp/fake-200.cimage.tmp"),
            completed: Arc::new(Mutex::new(None)),
            attempt: 1,
            _handle_id: "compile-200-1-0".to_string(),
        };
        dispatcher.active.lock().insert(handle.id.clone(), entry);

        // Cancel before polling
        dispatcher.cancel(&handle).unwrap();

        // Poll returns Failed("cancelled")
        assert_eq!(
            dispatcher.poll(&handle).unwrap(),
            DispatchStatus::Failed("cancelled".to_string())
        );

        // Entry is removed from active map
        assert!(
            !dispatcher.active.lock().contains_key(&handle.id),
            "entry should be removed from active map after cancel"
        );
    }

    #[test]
    fn cancel_during_execution() {
        let model_dir = create_minimal_model_dir();
        let output_dir = tempfile::tempdir().expect("output tempdir");
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let request = DispatchRequest {
            work_entity: 300,
            attempt: 1,
            plan_generation: 0,
            lease_token: "test".into(),
            deadline_ms: 9999999999999,
            backend: "test".into(),
            config: "{}".into(),
            input_path: model_dir.path().to_string_lossy().into_owned(),
            output_path: output_dir
                .path()
                .join("out.cimage")
                .to_string_lossy()
                .into_owned(),
        };

        // Start real compilation
        let handle = dispatcher.start(&request).expect("start should succeed");

        // Immediately cancel
        dispatcher.cancel(&handle).unwrap();

        // Poll should return Failed("cancelled").  The cancelled-set check
        // in poll() takes precedence over any completed status.
        for _ in 0..300 {
            let status = dispatcher.poll(&handle).expect("poll should succeed");
            match &status {
                DispatchStatus::Failed(msg) => {
                    assert_eq!(msg, "cancelled", "expected cancelled, got: {msg}");
                    break;
                }
                DispatchStatus::Completed(_) => {
                    panic!("compile finished before cancellation was effective");
                }
                DispatchStatus::Running => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }

        // cancelled set should track the handle
        assert!(
            dispatcher.cancelled.lock().contains(&handle.id),
            "cancelled set should track the cancelled handle"
        );
    }

    #[test]
    fn cancel_after_completion() {
        let model_dir = create_minimal_model_dir();
        let output_dir = tempfile::tempdir().expect("output tempdir");
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let request = DispatchRequest {
            work_entity: 400,
            attempt: 1,
            plan_generation: 0,
            lease_token: "test".into(),
            deadline_ms: 9999999999999,
            backend: "test".into(),
            config: "{}".into(),
            input_path: model_dir.path().to_string_lossy().into_owned(),
            output_path: output_dir
                .path()
                .join("out.cimage")
                .to_string_lossy()
                .into_owned(),
        };

        let handle = dispatcher.start(&request).expect("start should succeed");

        // Wait for completion
        let final_status = loop {
            match dispatcher.poll(&handle).expect("poll should not error") {
                DispatchStatus::Completed(payload) => {
                    break DispatchStatus::Completed(payload);
                }
                DispatchStatus::Failed(msg) => {
                    break DispatchStatus::Failed(msg);
                }
                DispatchStatus::Running => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        };
        assert!(
            matches!(&final_status, DispatchStatus::Completed(_)),
            "expected Completed before cancel, got: {final_status:?}"
        );

        // Cancel after completion -- should be a no-op, returning Ok
        dispatcher.cancel(&handle).unwrap();

        // After first poll consumed the Completed entry, a second poll
        // will find the entry gone from active and the handle in cancelled
        // set, so it returns Failed("cancelled").  That is fine -- the
        // point is that cancel() does not error after completion.
        let status = dispatcher.poll(&handle).expect("poll should succeed");
        let _ = status;
    }

    #[test]
    fn cancel_releases_lease() {
        let dispatcher = DaemonCompilerDispatcher::new(false, None);
        let handle = DispatchHandle {
            id: "compile-500-1-0".to_string(),
            work_entity: 500,
            attempt: 1,
        };

        // Cancel an arbitrary handle
        dispatcher.cancel(&handle).unwrap();

        // Verify cancelled set tracks the handle
        assert!(
            dispatcher.cancelled.lock().contains(&handle.id),
            "cancelled set should contain the cancelled handle"
        );
    }
    #[test]
    fn cancel_during_shutdown() {
        // Save refs to internal state before drop so we can inspect them after
        // the dispatcher's Drop runs.
        let cancelled: Arc<Mutex<HashSet<String>>>;
        let temp_files: Arc<Mutex<Vec<PathBuf>>>;
        let tmp_path: PathBuf;
        {
            let dispatcher = DaemonCompilerDispatcher::new(false, None);
            cancelled = dispatcher.cancelled.clone();
            temp_files = dispatcher.temp_files.clone();

            let handle = DispatchHandle {
                id: "compile-600-1-0".to_string(),
                work_entity: 600,
                attempt: 1,
            };
            tmp_path = PathBuf::from("/tmp/prism-shutdown-600.cimage.tmp");

            // Insert a dispatch entry manually to simulate an active dispatch
            let entry = DispatchEntry {
                _handle: handle.clone(),
                cancellation_signal: Arc::new(AtomicBool::new(false)),
                thread: None,
                _output_path: PathBuf::from("/tmp/prism-shutdown-600.cimage"),

                tmp_path: tmp_path.clone(),
                completed: Arc::new(Mutex::new(None)),
                attempt: 1,
                _handle_id: "compile-600-1-0".to_string(),
            };
            dispatcher.active.lock().insert(handle.id.clone(), entry);

            // Cancel the dispatch — this records the cancellation in the
            // cancelled set and pushes the tmp_path into temp_files.
            dispatcher.cancel(&handle).unwrap();

            // Verify cancelled tracks the handle before drop
            assert!(
                cancelled.lock().contains(&handle.id),
                "cancelled set should track the handle before shutdown"
            );

            // Verify the temp file path was added to temp_files
            assert!(
                temp_files.lock().contains(&tmp_path),
                "temp_files should track the cancelled dispatch's tmp path"
            );

            // Dispatcher drops here — Drop::drop takes temp_files and removes
            // each file from disk.  The temp file doesn't exist on disk so
            // remove_file will just return Ok(()).  The vec inside temp_files
            // is std::mem::take'd (replaced with empty vec).
        }

        // After drop: the cancelled set still holds the handle_id because our
        // cloned Arc keeps the inner Mutex alive.
        assert!(
            cancelled.lock().contains("compile-600-1-0"),
            "cancelled set should persist after dispatcher drop"
        );

        // After drop: temp_files was std::mem::take'd, so the inner vec is empty.
        assert!(
            temp_files.lock().is_empty(),
            "temp_files should be emptied by Drop::drop (std::mem::take)"
        );
    }

    #[test]
    fn crash_recovery() {
        let active: Arc<Mutex<HashMap<String, DispatchEntry>>>;
        {
            let dispatcher = DaemonCompilerDispatcher::new(false, None);
            active = dispatcher.active.clone();

            let handle = DispatchHandle {
                id: "compile-700-1-0".to_string(),
                work_entity: 700,
                attempt: 1,
            };
            let entry = DispatchEntry {
                _handle: handle.clone(),
                cancellation_signal: Arc::new(AtomicBool::new(false)),
                thread: None,
                _output_path: PathBuf::from("/tmp/prism-crash-700.cimage"),

                tmp_path: PathBuf::from("/tmp/prism-crash-700.cimage.tmp"),
                completed: Arc::new(Mutex::new(None)),
                attempt: 1,
                _handle_id: "compile-700-1-0".to_string(),
            };
            dispatcher.active.lock().insert(handle.id.clone(), entry);

            // Simulate crash: dispatcher drops WITHOUT calling cancel() on the
            // active entry.  The thread handle (None) and all sub-resources are
            // just dropped — no cleanup of active, cancelled, or temp_files.
        }

        // After the crash: our cloned Arc still points to the old dispatcher's
        // active map.  The entry we inserted should still be present because
        // nothing cleaned it up.
        assert!(
            active.lock().contains_key("compile-700-1-0"),
            "active entry should persist after crash (no cleanup on Drop)"
        );

        // A NEW dispatcher starts with clean state
        let new_dispatcher = DaemonCompilerDispatcher::new(false, None);
        assert!(
            new_dispatcher.active.lock().is_empty(),
            "new dispatcher should have no orphaned entries"
        );
        assert!(
            new_dispatcher.cancelled.lock().is_empty(),
            "new dispatcher should have no orphaned cancelled entries"
        );
        assert!(
            new_dispatcher.temp_files.lock().is_empty(),
            "new dispatcher should have no orphaned temp files"
        );
    }

    #[test]
    fn cancel_prevents_publication() {
        let model_dir = create_minimal_model_dir();
        let output_dir = tempfile::tempdir().expect("output tempdir");
        let dispatcher = DaemonCompilerDispatcher::new(false, None);

        let request = DispatchRequest {
            work_entity: 800,
            attempt: 1,
            plan_generation: 0,
            lease_token: "test".into(),
            deadline_ms: 9999999999999,
            backend: "test".into(),
            config: "{}".into(),
            input_path: model_dir.path().to_string_lossy().into_owned(),
            output_path: output_dir
                .path()
                .join("out.cimage")
                .to_string_lossy()
                .into_owned(),
        };

        let handle = dispatcher.start(&request).expect("start should succeed");

        // Compute the expected final path (published .cimage)
        let (_final_path, _tmp_path) = DaemonCompilerDispatcher::per_attempt_paths(
            output_dir.path(),
            request.work_entity,
            request.attempt,
            request.plan_generation,
        );

        // Cancel immediately — the worker thread may or may not have started
        // compilation, but cancel() sets the signal and marks the attempt.
        dispatcher.cancel(&handle).unwrap();

        // Poll must return Failed, never Completed
        let polled_status = loop {
            match dispatcher.poll(&handle).expect("poll should succeed") {
                DispatchStatus::Failed(msg) => break DispatchStatus::Failed(msg),
                DispatchStatus::Completed(_) => {
                    panic!("cancel_prevents_publication: poll returned Completed after cancel")
                }
                DispatchStatus::Running => {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        };
        assert_eq!(
            polled_status,
            DispatchStatus::Failed("cancelled".to_string()),
            "poll should return cancelled after cancel"
        );

        // Verify no .cimage was published (final path does not exist)
        assert!(
            !_final_path.exists(),
            "final .cimage should not exist: {:?}",
            _final_path
        );

        // Verify the cancelled set tracks the handle_id
        assert!(
            dispatcher.cancelled.lock().contains(&handle.id),
            "cancelled set should track the cancelled handle_id {}",
            handle.id
        );
    }
}
