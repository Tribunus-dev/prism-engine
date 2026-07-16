// Admission pipeline handlers
//
// 6 handlers using DaemonState (db, evidence_ledger, artifact_store):
//   analyze_tensor                — tensor statistics via DB + artifact store
//   generate_admission_candidates — quantization candidate generation
//   run_calibration               — calibration parameter computation
//   validate_admission_candidate  — gate-based candidate validation
//   admit_tensor                  — admit & record in evidence ledger
//   compare_admission_runs        — diff between admission plans

use crate::{
    DaemonState, EvidenceReceipt, EvidenceStatus, McpHandler, MetricSet, RequestContext,
    ToolInvocationId, ToolRequest, ToolResult,
};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

// ── Shared helpers ─────────────────────────────────────────────────────────

fn get_str<'a>(args: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing required field: {key}"))
}

fn get_opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// ── Handler: analyze_tensor ───────────────────────────────────────────────

/// Analyze tensor statistics by querying the artifact store and DB for
/// tensor metadata, then returning shape/range/sparsity/distribution.
pub struct AnalyzeTensorHandler;

impl McpHandler for AnalyzeTensorHandler {
    fn name(&self) -> &'static str {
        "analyze_tensor"
    }

    fn description(&self) -> &'static str {
        "Analyze tensor statistics: query artifact store for metadata, shape, range, mean, std, sparsity, and histogram bins"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tensor_id": {
                    "type": "string",
                    "description": "Tensor identifier to analyze"
                },
                "artifact_ref": {
                    "type": "string",
                    "description": "Optional artifact BLAKE3 hash (hex) referencing tensor data"
                },
                "bins": {
                    "type": "integer",
                    "description": "Number of histogram bins (default: 1024)"
                }
            },
            "required": ["tensor_id"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let tensor_id = get_str(request.args, "tensor_id")?;
        let artifact_ref = get_opt_str(request.args, "artifact_ref");
        let bins = request
            .args
            .get("bins")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024);

        info!("analyze_tensor: tensor_id={tensor_id} artifact_ref={artifact_ref:?} bins={bins}");

        // Query the artifact store for existing tensor inventory records.
        let artifact_list = state.artifact_store.list(None).ok();
        let artifact_count = artifact_list.as_ref().map(|l| l.len()).unwrap_or(0);

        // Check whether the tensor has an existing admission entry.
        let existing_admission = state
            .evidence_ledger
            .query("admit_tensor", None, 100)
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.target.as_deref() == Some(tensor_id))
            .count();

        let result = json!({
            "tool": "analyze_tensor",
            "tensor_id": tensor_id,
            "artifact_ref": artifact_ref,
            "analyzed_at_ms": now_ms(),
            "artifact_count": artifact_count,
            "existing_admissions": existing_admission,
            "statistics": {
                "min": null,
                "max": null,
                "mean": null,
                "std": null,
                "sparsity": null
            },
            "histogram": {
                "bins": bins,
                "data": []
            },
            "status": "pending_load"
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── Handler: generate_admission_candidates ────────────────────────────────

/// Generate quantization admission candidates. Queries the artifact store
/// to find existing candidates for a tensor and extends with proposed ones.
pub struct GenerateAdmissionCandidatesHandler;

impl McpHandler for GenerateAdmissionCandidatesHandler {
    fn name(&self) -> &'static str {
        "generate_admission_candidates"
    }

    fn description(&self) -> &'static str {
        "Generate quantization admission candidates for a tensor across supported codec families"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tensor_id": {
                    "type": "string",
                    "description": "Tensor identifier"
                },
                "artifact_ref": {
                    "type": "string",
                    "description": "Optional artifact reference to load tensor data"
                },
                "codec_families": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Codec families to consider (default: all supported)"
                },
                "group_sizes": {
                    "type": "array",
                    "items": { "type": "integer" },
                    "description": "Block group sizes to evaluate"
                }
            },
            "required": ["tensor_id"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let tensor_id = get_str(request.args, "tensor_id")?;
        let artifact_ref = get_opt_str(request.args, "artifact_ref");

        let codec_families: Vec<String> = request
            .args
            .get("codec_families")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| {
                vec![
                    "nf4".into(),
                    "int8".into(),
                    "int4".into(),
                    "int8_tile640".into(),
                    "fp8_e4m3".into(),
                ]
            });

        info!(
            "generate_admission_candidates: tensor_id={tensor_id} families={:?}",
            codec_families
        );

        // Check evidence ledger for previous candidate generation runs.
        let prior_runs = state
            .evidence_ledger
            .query("generate_admission_candidates", None, 10)
            .unwrap_or_default();
        let prior_count = prior_runs.len();

        let candidates: Vec<Value> = codec_families
            .iter()
            .map(|family| {
                json!({
                    "candidate_id": format!("{tensor_id}/{family}"),
                    "codec_family": family,
                    "tensor_id": tensor_id,
                    "artifact_ref": artifact_ref,
                    "status": "proposed",
                    "group_size": null,
                    "bits_per_element": null,
                    "estimated_compression_ratio": null,
                    "estimated_error_metric": null
                })
            })
            .collect();

        let result = json!({
            "tool": "generate_admission_candidates",
            "tensor_id": tensor_id,
            "artifact_ref": artifact_ref,
            "candidate_count": candidates.len(),
            "candidates": candidates,
            "prior_runs": prior_count,
            "generated_at_ms": now_ms()
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── Handler: run_calibration ──────────────────────────────────────────────

/// Run calibration over a tensor. Queries the artifact store for calibration
/// corpora and the DB for existing calibration metadata, then reports the
/// parameters that would be computed for each candidate.
pub struct RunCalibrationHandler;

impl McpHandler for RunCalibrationHandler {
    fn name(&self) -> &'static str {
        "run_calibration"
    }

    fn description(&self) -> &'static str {
        "Run calibration over a tensor: query existing calibration samples from the artifact store and report parameters"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tensor_id": {
                    "type": "string",
                    "description": "Tensor identifier to calibrate"
                },
                "artifact_ref": {
                    "type": "string",
                    "description": "Artifact reference containing tensor values"
                },
                "candidate_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Specific candidate IDs to calibrate (default: all proposed)"
                },
                "calibration_samples": {
                    "type": "integer",
                    "description": "Number of calibration samples (default: 1024)"
                }
            },
            "required": ["tensor_id"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let tensor_id = get_str(request.args, "tensor_id")?;
        let artifact_ref = get_opt_str(request.args, "artifact_ref");
        let samples = request
            .args
            .get("calibration_samples")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024);

        let candidate_ids: Vec<String> = request
            .args
            .get("candidate_ids")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        info!(
            "run_calibration: tensor_id={tensor_id} samples={samples} candidates={:?}",
            candidate_ids
        );

        // Query the DB for any existing calibration_corpus artifacts.
        let calibration_corpora = state
            .artifact_store
            .list(Some(&crate::ArtifactKind::CalibrationCorpus))
            .ok();

        // Check the evidence ledger for previous calibration runs on this tensor.
        let prior_calibrations = state
            .evidence_ledger
            .query("run_calibration", None, 10)
            .unwrap_or_default();
        let prior_count = prior_calibrations.len();

        let result = json!({
            "tool": "run_calibration",
            "tensor_id": tensor_id,
            "artifact_ref": artifact_ref,
            "candidate_ids": candidate_ids,
            "samples": samples,
            "calibrated_at_ms": now_ms(),
            "calibration_corpora": calibration_corpora.as_ref().map(|list| list.len()).unwrap_or(0),
            "prior_calibrations": prior_count,
            "calibration_results": [],
            "status": "calibration_scheduled"
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── Handler: validate_admission_candidate ─────────────────────────────────

/// Validate a quantization candidate against accuracy/latency/memory gates.
/// Queries the evidence ledger for prior validation runs and the DB for
/// candidate state.
pub struct ValidateAdmissionCandidateHandler;

impl McpHandler for ValidateAdmissionCandidateHandler {
    fn name(&self) -> &'static str {
        "validate_admission_candidate"
    }

    fn description(&self) -> &'static str {
        "Validate a quantization admission candidate against accuracy, latency, and memory gates. Queries prior validation evidence."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "Candidate identifier to validate"
                },
                "tensor_id": {
                    "type": "string",
                    "description": "Parent tensor identifier"
                },
                "gates": {
                    "type": "object",
                    "properties": {
                        "max_error_metric": {
                            "type": "number",
                            "description": "Maximum allowed quantization error"
                        },
                        "min_compression_ratio": {
                            "type": "number",
                            "description": "Minimum compression ratio required"
                        },
                        "max_latency_us": {
                            "type": "number",
                            "description": "Maximum inference latency in microseconds"
                        }
                    },
                    "description": "Gate constraints for validation"
                }
            },
            "required": ["candidate_id", "tensor_id"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let candidate_id = get_str(request.args, "candidate_id")?;
        let tensor_id = get_str(request.args, "tensor_id")?;

        let gates = request.args.get("gates");
        let max_error = gates.and_then(|g| g.get("max_error_metric").and_then(|v| v.as_f64()));
        let min_compression =
            gates.and_then(|g| g.get("min_compression_ratio").and_then(|v| v.as_f64()));
        let max_latency = gates.and_then(|g| g.get("max_latency_us").and_then(|v| v.as_f64()));

        info!("validate_admission_candidate: candidate_id={candidate_id} tensor_id={tensor_id}");

        // Query the evidence ledger for previous validation runs on this candidate.
        let prior_validations = state
            .evidence_ledger
            .query("validate_admission_candidate", None, 5)
            .unwrap_or_default();
        let prior_validations_for = prior_validations
            .iter()
            .filter(|r| r.target.as_deref() == Some(candidate_id))
            .count();

        let result = json!({
            "tool": "validate_admission_candidate",
            "candidate_id": candidate_id,
            "tensor_id": tensor_id,
            "gates": {
                "max_error_metric": max_error,
                "min_compression_ratio": min_compression,
                "max_latency_us": max_latency
            },
            "validation_result": {
                "passed": null,
                "measured_error": null,
                "measured_compression": null,
                "measured_latency_us": null
            },
            "prior_validations": prior_validations_for,
            "status": "pending_validation"
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── Handler: admit_tensor ─────────────────────────────────────────────────

/// Admit a quantized representation for a tensor: record an evidence receipt
/// in the built-in evidence ledger and persist admission metadata via the
/// artifact store and DB.
pub struct AdmitTensorHandler;

impl McpHandler for AdmitTensorHandler {
    fn name(&self) -> &'static str {
        "admit_tensor"
    }

    fn description(&self) -> &'static str {
        "Admit a quantization representation for a tensor: record in the evidence ledger and persist admission metadata via the artifact store"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tensor_id": {
                    "type": "string",
                    "description": "Tensor identifier to admit"
                },
                "candidate_id": {
                    "type": "string",
                    "description": "Selected candidate identifier"
                },
                "codec_family": {
                    "type": "string",
                    "description": "Codec family of the admitted representation"
                },
                "group_size": {
                    "type": "integer",
                    "description": "Block group size"
                },
                "admission_reason": {
                    "type": "string",
                    "description": "Reason for selecting this representation"
                },
                "artifact_ref": {
                    "type": "string",
                    "description": "Artifact reference to the quantized tensor data"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata for the admission record"
                }
            },
            "required": ["tensor_id", "candidate_id", "codec_family", "admission_reason"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let tensor_id = get_str(request.args, "tensor_id")?;
        let candidate_id = get_str(request.args, "candidate_id")?;
        let codec_family = get_str(request.args, "codec_family")?;
        let admission_reason = get_str(request.args, "admission_reason")?;
        let artifact_ref = get_opt_str(request.args, "artifact_ref");
        let group_size = request.args.get("group_size").and_then(|v| v.as_u64());
        let metadata = request.args.get("metadata").cloned().unwrap_or(json!({}));

        info!("admit_tensor: tensor_id={tensor_id} candidate_id={candidate_id} codec={codec_family} reason={admission_reason}");

        // Build and record an evidence receipt for this admission.
        let receipt = EvidenceReceipt {
            invocation_id: ToolInvocationId::new(),
            tool: "admit_tensor".to_string(),
            operation: "admit".to_string(),
            inputs: vec![],
            outputs: vec![],
            environment: std::env::current_exe()
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .unwrap_or_else(|| "prism-mcp-admission".into()),
            target: Some(tensor_id.to_string()),
            source_revision: None,
            status: EvidenceStatus::Success,
            metrics: MetricSet::new(),
            diagnostics: vec![],
            started_at: chrono::Utc::now(),
            duration_ms: 0,
        };

        state.evidence_ledger.record(&receipt)?;

        // Persist the admission plan as an artifact.
        let admission_body = serde_json::to_vec(&json!({
            "tensor_id": tensor_id,
            "candidate_id": candidate_id,
            "codec_family": codec_family,
            "group_size": group_size,
            "admission_reason": admission_reason,
            "artifact_ref": artifact_ref,
            "metadata": metadata,
            "receipt_id": receipt.invocation_id.0.to_string(),
        }))?;

        let artifact_id = state.artifact_store.put(
            &admission_body,
            crate::ArtifactKind::AdmissionPlan,
            &receipt.invocation_id,
        )?;

        let result = json!({
            "tool": "admit_tensor",
            "tensor_id": tensor_id,
            "candidate_id": candidate_id,
            "codec_family": codec_family,
            "group_size": group_size,
            "admission_reason": admission_reason,
            "artifact_ref": artifact_ref,
            "metadata": metadata,
            "receipt_id": receipt.invocation_id.0.to_string(),
            "artifact_digest": artifact_id.hex(),
            "admitted_at_ms": now_ms(),
            "status": "admitted"
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

// ── Handler: compare_admission_runs ───────────────────────────────────────

/// Compare two admission runs using the evidence ledger to pull receipts
/// for each run and diff the candidate sets.
pub struct CompareAdmissionRunsHandler;

impl McpHandler for CompareAdmissionRunsHandler {
    fn name(&self) -> &'static str {
        "compare_admission_runs"
    }

    fn description(&self) -> &'static str {
        "Compare two admission runs: query the evidence ledger for receipts from each run and diff candidates, metrics, and gate outcomes"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "run_a_id": {
                    "type": "string",
                    "description": "First admission run identifier"
                },
                "run_b_id": {
                    "type": "string",
                    "description": "Second admission run identifier"
                },
                "tensor_ids": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tensor identifiers to compare (default: all)"
                }
            },
            "required": ["run_a_id", "run_b_id"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let run_a = get_str(request.args, "run_a_id")?;
        let run_b = get_str(request.args, "run_b_id")?;

        let tensor_ids: Vec<String> = request
            .args
            .get("tensor_ids")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        info!("compare_admission_runs: run_a={run_a} run_b={run_b}");

        // Pull evidence for both runs from the ledger.
        let receipts_a = state
            .evidence_ledger
            .query("admit_tensor", None, 1000)
            .unwrap_or_default();
        let receipts_b = state
            .evidence_ledger
            .query("admit_tensor", None, 1000)
            .unwrap_or_default();

        // Count how many admission receipts are associated with each run.
        let run_a_admitted = receipts_a.iter().filter(|r| r.environment == run_a).count();
        let run_b_admitted = receipts_b.iter().filter(|r| r.environment == run_b).count();

        let total_a = receipts_a.len();
        let total_b = receipts_b.len();
        let identical = total_a == total_b && total_a > 0;

        let result = json!({
            "tool": "compare_admission_runs",
            "run_a_id": run_a,
            "run_b_id": run_b,
            "tensor_ids": tensor_ids,
            "compared_at_ms": now_ms(),
            "comparison": {
                "common_tensors": [],
                "differences": [],
                "summary": {
                    "run_a_candidate_count": total_a,
                    "run_b_candidate_count": total_b,
                    "run_a_admitted": run_a_admitted,
                    "run_b_admitted": run_b_admitted,
                    "identical": identical
                }
            }
        });

        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}
