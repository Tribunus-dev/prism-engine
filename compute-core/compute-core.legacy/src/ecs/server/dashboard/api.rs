//! Axum route handlers for the evidence dashboard API.
//!
//! All endpoints are gated behind `cfg(feature = "server-dashboard")`.
//! Uses `sqlx` (PostgreSQL) for persistence.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Response as AxumResponse,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::Row;
use std::sync::Arc;

use crate::ecs::server::dashboard::cache::DashboardCache;
use crate::ecs::server::dashboard::indexer::EvidenceIndexer;
use crate::ecs::server::dashboard::models::*;

// ── Shared application state ──────────────────────────────────────────────

/// Shared application state injected into every dashboard handler via
/// `axum::extract::State`.
pub struct DashboardState {
    /// Evidence indexer — holds the PostgreSQL pool and
    /// optional Valkey client.
    pub indexer: EvidenceIndexer,
    /// High-level Valkey-backed cache facade.
    pub cache: DashboardCache,
}

// ── Query-string filter types ─────────────────────────────────────────────

/// Optional filters for the admissions listing endpoint.
#[derive(Debug, Deserialize)]
pub struct AdmissionFilter {
    pub artifact_digest: Option<String>,
    pub codec: Option<String>,
    pub status: Option<String>,
}

/// Optional filters for the sweep listing endpoint.
#[derive(Debug, Deserialize)]
pub struct SweepFilter {
    pub artifact_digest: Option<String>,
    pub tensor_key: Option<String>,
}

/// Optional filters for the execution listing endpoint.
#[derive(Debug, Deserialize)]
pub struct ExecutionFilter {
    pub artifact_digest: Option<String>,
    pub backend: Option<String>,
}

/// Optional filters for the evidence listing endpoint.
#[derive(Debug, Deserialize)]
pub struct EvidenceFilter {
    pub scope: Option<String>,
    pub kind: Option<String>,
}

// ── API error type ────────────────────────────────────────────────────────

/// Unified error type returned by all dashboard API handlers.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn internal(msg: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
    pub fn not_found(msg: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }
    pub fn bad_request(msg: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> AxumResponse {
        let body = serde_json::json!({ "error": self.message });
        (self.status, Json(body)).into_response()
    }
}

// ── Helper: cache get/set with JSON serialisation ─────────────────────────

/// Attempt to read a typed JSON value from the Valkey cache.
/// Returns `Ok(None)` when the key is absent or deserialisation fails.
async fn cache_get<T>(cache: &DashboardCache, key: &str) -> Result<Option<T>, ApiError>
where
    T: serde::de::DeserializeOwned,
{
    match cache.get_cached(key).await {
        Some(raw) => match serde_json::from_str::<T>(&raw) {
            Ok(val) => Ok(Some(val)),
            Err(_) => Ok(None),
        },
        None => Ok(None),
    }
}

/// Store a typed JSON value in the Valkey cache with the given TTL.
async fn cache_set<T>(cache: &DashboardCache, key: &str, value: &T, ttl: u64)
where
    T: Serialize,
{
    if let Ok(raw) = serde_json::to_string(value) {
        cache.set_cached(key, &raw, ttl).await;
    }
}

// ── Handler: list CImage artifacts ───────────────────────────────────────

/// `GET /v1/cimages`
///
/// Returns all CImage artifacts ordered by creation time (newest first).
/// Results are cached in Valkey with a 60-second TTL.
pub async fn list_cimages(
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<Vec<DashboardCImageSummary>>, ApiError> {
    if let Some(cached) =
        cache_get::<Vec<DashboardCImageSummary>>(&state.cache, "dashboard:cimages:list").await?
    {
        return Ok(Json(cached));
    }

    let rows = sqlx::query(
        "SELECT digest, path, artifact_kind, model_family, schema_version, \
                tensor_count, receipt_count, validation_status, \
                compiler_policy_digest, hardware_profile, created_at \
         FROM cimage_artifacts ORDER BY created_at DESC",
    )
    .fetch_all(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let summaries: Vec<DashboardCImageSummary> = rows
        .iter()
        .map(|r| DashboardCImageSummary {
            digest: r.get("digest"),
            path: r.get("path"),
            artifact_kind: r.get("artifact_kind"),
            model_family: r.get("model_family"),
            schema_version: r.get("schema_version"),
            tensor_count: r.get("tensor_count"),
            receipt_count: r.get("receipt_count"),
            validation_status: r.get("validation_status"),
            compiler_policy_digest: r.get("compiler_policy_digest"),
            hardware_profile: r.get("hardware_profile"),
            created_at: r.get("created_at"),
        })
        .collect();

    cache_set(&state.cache, "dashboard:cimages:list", &summaries, 60).await;
    Ok(Json(summaries))
}

// ── Handler: get single CImage ────────────────────────────────────────────

/// `GET /v1/cimages/{digest}`
pub async fn get_cimage(
    Path(digest): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<DashboardCImageSummary>, ApiError> {
    let cache_key = format!("dashboard:cimage:{digest}");
    if let Some(cached) = cache_get::<DashboardCImageSummary>(&state.cache, &cache_key).await? {
        return Ok(Json(cached));
    }

    let row = sqlx::query(
        "SELECT digest, path, artifact_kind, model_family, schema_version, \
                tensor_count, receipt_count, validation_status, \
                compiler_policy_digest, hardware_profile, created_at \
         FROM cimage_artifacts WHERE digest = $1",
    )
    .bind(&digest)
    .fetch_optional(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?
    .ok_or_else(|| ApiError::not_found(format!("CImage not found: {digest}")))?;

    let summary = DashboardCImageSummary {
        digest: row.get("digest"),
        path: row.get("path"),
        artifact_kind: row.get("artifact_kind"),
        model_family: row.get("model_family"),
        schema_version: row.get("schema_version"),
        tensor_count: row.get("tensor_count"),
        receipt_count: row.get("receipt_count"),
        validation_status: row.get("validation_status"),
        compiler_policy_digest: row.get("compiler_policy_digest"),
        hardware_profile: row.get("hardware_profile"),
        created_at: row.get("created_at"),
    };

    cache_set(&state.cache, &cache_key, &summary, 60).await;
    Ok(Json(summary))
}

// ── Handler: get tensors for a CImage ─────────────────────────────────────

/// `GET /v1/cimages/{digest}/tensors`
pub async fn get_cimage_tensors(
    Path(digest): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<Vec<DashboardTensorSummary>>, ApiError> {
    let cache_key = format!("dashboard:tensors:{digest}");
    if let Some(cached) = cache_get::<Vec<DashboardTensorSummary>>(&state.cache, &cache_key).await?
    {
        return Ok(Json(cached));
    }

    let rows = sqlx::query(
        "SELECT artifact_digest, tensor_key, tensor_class, codec, \
                group_size, effective_bpw, logical_shape, payload_size, \
                promotion_status \
         FROM cimage_tensors WHERE artifact_digest = $1 ORDER BY tensor_key",
    )
    .bind(&digest)
    .fetch_all(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let tensors: Vec<DashboardTensorSummary> = rows
        .iter()
        .map(|r| {
            let shape_str: Option<String> = r.get("logical_shape");
            let logical_shape: Vec<u32> = shape_str
                .as_deref()
                .and_then(|s| serde_json::from_str::<Vec<u32>>(s).ok())
                .unwrap_or_default();

            DashboardTensorSummary {
                artifact_digest: r.get("artifact_digest"),
                tensor_key: r.get("tensor_key"),
                tensor_class: r.get("tensor_class"),
                codec: r.get("codec"),
                group_size: r.get("group_size"),
                effective_bpw: r.get("effective_bpw"),
                logical_shape,
                payload_size: r.get("payload_size"),
                promotion_status: r.get("promotion_status"),
            }
        })
        .collect();

    cache_set(&state.cache, &cache_key, &tensors, 60).await;
    Ok(Json(tensors))
}

// ── Handler: list admission receipts ──────────────────────────────────────

/// `GET /v1/admissions`
///
/// Queries `admission_receipts` with optional filters on `artifact_digest`,
/// `codec`, and `promotion_status` (serialised as `status` in the query).
pub async fn list_admissions(
    Query(f): Query<AdmissionFilter>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<Vec<DashboardAdmissionSummary>>, ApiError> {
    let mut sql = String::from(
        "SELECT receipt_id, tensor_key, codec, group_size, effective_bpw, \
                zero_fraction, neg_fraction, pos_fraction, scale_mean, \
                scale_std, scale_max, operator_nrmse, output_cosine, \
                activation_shift_l2, deadzone_collapse, rescue_required, \
                rescue_codec, promotion_status \
         FROM admission_receipts WHERE 1=1",
    );

    // Build WHERE clauses with positional $N parameters.
    // We track how many are bound so far to compute the right $N.
    let n_digest = f.artifact_digest.is_some();
    let n_codec = f.codec.is_some();
    let n_status = f.status.is_some();

    if n_digest {
        sql.push_str(" AND artifact_digest = $1");
    }
    if n_codec {
        if n_digest {
            sql.push_str(" AND codec = $2");
        } else {
            sql.push_str(" AND codec = $1");
        }
    }
    if n_status {
        let idx = 1 + n_digest as i32 + n_codec as i32;
        sql.push_str(&format!(" AND promotion_status = ${idx}"));
    }

    sql.push_str(" ORDER BY tensor_key, effective_bpw");

    let mut q = sqlx::query(&sql);
    if let Some(ref digest) = f.artifact_digest {
        q = q.bind(digest);
    }
    if let Some(ref codec) = f.codec {
        q = q.bind(codec);
    }
    if let Some(ref status) = f.status {
        q = q.bind(status);
    }

    let rows = q
        .fetch_all(&state.indexer.pool)
        .await
        .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let results: Vec<DashboardAdmissionSummary> = rows
        .iter()
        .map(|r| {
            let deadzone_raw: bool = r.get("deadzone_collapse");
            let rescue_raw: bool = r.get("rescue_required");
            DashboardAdmissionSummary {
                receipt_id: r.get("receipt_id"),
                tensor_key: r.get("tensor_key"),
                codec: r.get("codec"),
                group_size: r.get("group_size"),
                effective_bpw: r.get("effective_bpw"),
                zero_fraction: r.get("zero_fraction"),
                neg_fraction: r.get("neg_fraction"),
                pos_fraction: r.get("pos_fraction"),
                scale_mean: r.get("scale_mean"),
                scale_std: r.get("scale_std"),
                scale_max: r.get("scale_max"),
                operator_nrmse: r.get("operator_nrmse"),
                output_cosine: r.get("output_cosine"),
                activation_shift_l2: r.get("activation_shift_l2"),
                deadzone_collapse: deadzone_raw,
                rescue_required: rescue_raw,
                rescue_codec: r.get("rescue_codec"),
                promotion_status: r.get("promotion_status"),
            }
        })
        .collect();

    Ok(Json(results))
}

// ── Handler: get single admission receipt ─────────────────────────────────

/// `GET /v1/admissions/{id}`
pub async fn get_admission(
    Path(id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<DashboardAdmissionSummary>, ApiError> {
    let row = sqlx::query(
        "SELECT receipt_id, tensor_key, codec, group_size, effective_bpw, \
                zero_fraction, neg_fraction, pos_fraction, scale_mean, \
                scale_std, scale_max, operator_nrmse, output_cosine, \
                activation_shift_l2, deadzone_collapse, rescue_required, \
                rescue_codec, promotion_status \
         FROM admission_receipts WHERE receipt_id = $1",
    )
    .bind(&id)
    .fetch_optional(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?
    .ok_or_else(|| ApiError::not_found(format!("Admission receipt not found: {id}")))?;

    let deadzone_raw: bool = row.get("deadzone_collapse");
    let rescue_raw: bool = row.get("rescue_required");

    let summary = DashboardAdmissionSummary {
        receipt_id: row.get("receipt_id"),
        tensor_key: row.get("tensor_key"),
        codec: row.get("codec"),
        group_size: row.get("group_size"),
        effective_bpw: row.get("effective_bpw"),
        zero_fraction: row.get("zero_fraction"),
        neg_fraction: row.get("neg_fraction"),
        pos_fraction: row.get("pos_fraction"),
        scale_mean: row.get("scale_mean"),
        scale_std: row.get("scale_std"),
        scale_max: row.get("scale_max"),
        operator_nrmse: row.get("operator_nrmse"),
        output_cosine: row.get("output_cosine"),
        activation_shift_l2: row.get("activation_shift_l2"),
        deadzone_collapse: deadzone_raw,
        rescue_required: rescue_raw,
        rescue_codec: row.get("rescue_codec"),
        promotion_status: row.get("promotion_status"),
    };

    Ok(Json(summary))
}

// ── Handler: list sweeps ──────────────────────────────────────────────────

/// `GET /v1/sweeps`
pub async fn list_sweeps(
    Query(f): Query<SweepFilter>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<Vec<DashboardSweepSummary>>, ApiError> {
    let mut sql = String::from(
        "SELECT sweep_id, artifact_digest, tensor_key, candidate_count, \
                winner_candidate_id \
         FROM sweeps WHERE 1=1",
    );
    let n_digest = f.artifact_digest.is_some();
    let n_tkey = f.tensor_key.is_some();

    if n_digest {
        sql.push_str(" AND artifact_digest = $1");
    }
    if n_tkey {
        let idx = if n_digest { 2 } else { 1 };
        sql.push_str(&format!(" AND tensor_key = ${idx}"));
    }

    sql.push_str(" ORDER BY created_at DESC");

    let mut q = sqlx::query(&sql);
    if let Some(ref digest) = f.artifact_digest {
        q = q.bind(digest);
    }
    if let Some(ref tkey) = f.tensor_key {
        q = q.bind(tkey);
    }

    let rows = q
        .fetch_all(&state.indexer.pool)
        .await
        .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let sweeps: Vec<DashboardSweepSummary> = rows
        .iter()
        .map(|r| DashboardSweepSummary {
            sweep_id: r.get("sweep_id"),
            artifact_digest: r.get("artifact_digest"),
            tensor_key: r.get("tensor_key"),
            candidate_count: r.get("candidate_count"),
            winner_candidate_id: r.get("winner_candidate_id"),
        })
        .collect();

    Ok(Json(sweeps))
}

// ── Handler: get sweep candidates ────────────────────────────────────────

/// `GET /v1/sweeps/{id}/candidates`
pub async fn get_sweep_candidates(
    Path(id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<Vec<DashboardSweepCandidate>>, ApiError> {
    let rows = sqlx::query(
        "SELECT candidate_id, sweep_id, codec, group_size, \
                calibration_steps, nrmse, cosine, bytes, passed \
         FROM sweep_candidates WHERE sweep_id = $1 \
         ORDER BY group_size, calibration_steps",
    )
    .bind(&id)
    .fetch_all(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let candidates: Vec<DashboardSweepCandidate> = rows
        .iter()
        .map(|r| {
            let passed_raw: bool = r.get("passed");
            DashboardSweepCandidate {
                candidate_id: r.get("candidate_id"),
                codec: r.get("codec"),
                group_size: r.get("group_size"),
                calibration_steps: r.get("calibration_steps"),
                nrmse: r.get("nrmse"),
                cosine: r.get("cosine"),
                bytes: r.get("bytes"),
                passed: passed_raw,
            }
        })
        .collect();

    Ok(Json(candidates))
}

// ── Handler: list execution receipts ──────────────────────────────────────

/// `GET /v1/execution`
pub async fn list_execution(
    Query(f): Query<ExecutionFilter>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<Vec<DashboardExecutionSummary>>, ApiError> {
    let mut sql = String::from(
        "SELECT receipt_id, tensor_key, kernel_name, backend, \
                command_buffer_ms, effective_bandwidth_gbps, \
                validation_passed \
         FROM execution_receipts WHERE 1=1",
    );
    let n_digest = f.artifact_digest.is_some();
    let n_backend = f.backend.is_some();

    if n_digest {
        sql.push_str(" AND artifact_digest = $1");
    }
    if n_backend {
        let idx = if n_digest { 2 } else { 1 };
        sql.push_str(&format!(" AND backend = ${idx}"));
    }

    sql.push_str(" ORDER BY created_at DESC");

    let mut q = sqlx::query(&sql);
    if let Some(ref digest) = f.artifact_digest {
        q = q.bind(digest);
    }
    if let Some(ref backend) = f.backend {
        q = q.bind(backend);
    }

    let rows = q
        .fetch_all(&state.indexer.pool)
        .await
        .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let executions: Vec<DashboardExecutionSummary> = rows
        .iter()
        .map(|r| {
            let valid_raw: bool = r.get("validation_passed");
            DashboardExecutionSummary {
                receipt_id: r.get("receipt_id"),
                tensor_key: r.get("tensor_key"),
                kernel_name: r.get("kernel_name"),
                backend: r.get("backend"),
                command_buffer_ms: r.get("command_buffer_ms"),
                bandwidth_gbps: r.get("effective_bandwidth_gbps"),
                validation_passed: valid_raw,
            }
        })
        .collect();

    Ok(Json(executions))
}

// ── Handler: list evidence entries ───────────────────────────────────────

/// `GET /v1/evidence`
pub async fn list_evidence(
    Query(f): Query<EvidenceFilter>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<Vec<DashboardEvidenceEntry>>, ApiError> {
    let mut sql = String::from(
        "SELECT receipt_id, artifact_digest, scope, kind, \
                validation_passed, json_data \
         FROM evidence_ledger WHERE 1=1",
    );
    let n_scope = f.scope.is_some();
    let n_kind = f.kind.is_some();

    if n_scope {
        sql.push_str(" AND scope = $1");
    }
    if n_kind {
        let idx = if n_scope { 2 } else { 1 };
        sql.push_str(&format!(" AND kind = ${idx}"));
    }

    sql.push_str(" ORDER BY created_at DESC");

    let mut q = sqlx::query(&sql);
    if let Some(ref scope) = f.scope {
        q = q.bind(scope);
    }
    if let Some(ref kind) = f.kind {
        q = q.bind(kind);
    }

    let rows = q
        .fetch_all(&state.indexer.pool)
        .await
        .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let entries: Vec<DashboardEvidenceEntry> = rows
        .iter()
        .map(|r| {
            let valid_raw: bool = r.get("validation_passed");
            let json_str: String = r.get("json_data");
            let json: Value = serde_json::from_str(&json_str).unwrap_or(Value::Null);

            DashboardEvidenceEntry {
                receipt_id: r.get("receipt_id"),
                artifact_digest: r.get("artifact_digest"),
                scope: r.get("scope"),
                kind: r.get("kind"),
                validation_passed: valid_raw,
                json,
            }
        })
        .collect();

    Ok(Json(entries))
}

// ── Handler: get promotion gate explanation ──────────────────────────────

/// `GET /v1/promotion-gates/{scope_id}`
///
/// Aggregates the promotion status across all tensors associated with the
/// scope (either by matching artifact digest or compiler policy digest).
pub async fn get_promotion_gate(
    Path(scope_id): Path<String>,
    State(state): State<Arc<DashboardState>>,
) -> Result<Json<DashboardPromotionExplanation>, ApiError> {
    let cache_key = format!("dashboard:promotion:{scope_id}");
    if let Some(cached) =
        cache_get::<DashboardPromotionExplanation>(&state.cache, &cache_key).await?
    {
        return Ok(Json(cached));
    }

    // Determine the worst (lowest-ranked) promotion status across all
    // tensors that are part of this scope.
    let status_rank: i32 = sqlx::query(
        "SELECT COALESCE(MIN(rank), 5) AS worst_rank
         FROM (
             SELECT DISTINCT ct.promotion_status,
                 CASE ct.promotion_status
                     WHEN 'ProductionEligible' THEN 1
                     WHEN 'SyntheticPassed'    THEN 2
                     WHEN 'ResearchOnly'       THEN 3
                     WHEN 'Rejected'           THEN 4
                     ELSE 5
                 END AS rank
             FROM cimage_tensors ct
             JOIN cimage_artifacts ca ON ca.digest = ct.artifact_digest
             WHERE ca.digest = $1 OR ca.compiler_policy_digest = $1
         ) sub",
    )
    .bind(&scope_id)
    .fetch_optional(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?
    .map(|r| r.get::<i32, _>("worst_rank"))
    .unwrap_or(5);

    let current_status = match status_rank {
        1 => "ProductionEligible",
        2 => "SyntheticPassed",
        3 => "ResearchOnly",
        4 => "Rejected",
        _ => "Unknown",
    }
    .to_string();

    // Tensors without a matching admission receipt.
    let missing_rows = sqlx::query(
        "SELECT ct.tensor_key
         FROM cimage_tensors ct
         JOIN cimage_artifacts ca ON ca.digest = ct.artifact_digest
         WHERE (ca.digest = $1 OR ca.compiler_policy_digest = $1)
           AND NOT EXISTS (
               SELECT 1 FROM admission_receipts ar
               WHERE ar.artifact_digest = ct.artifact_digest
                 AND ar.tensor_key = ct.tensor_key
           )
         ORDER BY ct.tensor_key",
    )
    .bind(&scope_id)
    .fetch_all(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let missing_receipts: Vec<String> = missing_rows.iter().map(|r| r.get("tensor_key")).collect();

    // Execution-receipt kernels where validation did not pass.
    let failed_rows = sqlx::query(
        "SELECT DISTINCT er.kernel_name
         FROM execution_receipts er
         JOIN cimage_artifacts ca ON ca.digest = er.artifact_digest
         WHERE (ca.digest = $1 OR ca.compiler_policy_digest = $1)
           AND er.validation_passed = FALSE
         ORDER BY er.kernel_name",
    )
    .bind(&scope_id)
    .fetch_all(&state.indexer.pool)
    .await
    .map_err(|e| ApiError::internal(format!("Database query error: {e}")))?;

    let failed_gates: Vec<String> = failed_rows.iter().map(|r| r.get("kernel_name")).collect();

    let recommendation = if !failed_gates.is_empty() {
        format!(
            "{} gate(s) failing: {}. Re-run execution validation before promotion.",
            failed_gates.len(),
            failed_gates.join(", "),
        )
    } else if !missing_receipts.is_empty() {
        format!(
            "{} tensor(s) missing admission receipts. Run quantization sweep and admission pipeline.",
            missing_receipts.len(),
        )
    } else if current_status == "ProductionEligible" || current_status == "SyntheticPassed" {
        "All gates passed. Ready for promotion.".to_string()
    } else {
        format!("Current status is {current_status}. Review evidence before promotion.")
    };

    let explanation = DashboardPromotionExplanation {
        scope_id: scope_id.clone(),
        current_status,
        missing_receipts,
        failed_gates,
        recommendation,
    };

    cache_set(&state.cache, &cache_key, &explanation, 60).await;
    Ok(Json(explanation))
}

// ── Handler: OpenAPI schema ──────────────────────────────────────────────

/// `GET /v1/openapi.json`
pub async fn openapi_schema() -> Json<Value> {
    Json(generate_openapi_schema())
}

fn generate_openapi_schema() -> Value {
    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Prism Evidence Dashboard API",
            "version": "0.1.0",
            "description": "REST API for the Prism evidence dashboard — cimage inspection, admission receipts, sweep results, execution benchmarks, and promotion gates."
        },
        "servers": [{ "url": "/v1", "description": "Dashboard API v1" }],
        "paths": {
            "/cimages": {
                "get": {
                    "summary": "List all CImage artifacts",
                    "operationId": "listCImages",
                    "responses": {
                        "200": {
                            "description": "List of CImage artifact summaries",
                            "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/DashboardCImageSummary" } } } }
                        }
                    }
                }
            },
            "/cimages/{digest}": {
                "get": {
                    "summary": "Get a single CImage artifact",
                    "operationId": "getCImage",
                    "parameters": [{ "name": "digest", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "CImage artifact summary", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/DashboardCImageSummary" } } } },
                        "404": { "description": "CImage not found" }
                    }
                }
            },
            "/cimages/{digest}/tensors": {
                "get": {
                    "summary": "Get tensors for a CImage",
                    "operationId": "getCImageTensors",
                    "parameters": [{ "name": "digest", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "List of tensor summaries", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/DashboardTensorSummary" } } } } }
                    }
                }
            },
            "/admissions": {
                "get": {
                    "summary": "List admission receipts",
                    "operationId": "listAdmissions",
                    "parameters": [
                        { "name": "artifact_digest", "in": "query", "required": false, "schema": { "type": "string" } },
                        { "name": "codec", "in": "query", "required": false, "schema": { "type": "string" } },
                        { "name": "status", "in": "query", "required": false, "schema": { "type": "string" } }
                    ],
                    "responses": {
                        "200": { "description": "List of admission summaries", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/DashboardAdmissionSummary" } } } } }
                    }
                }
            },
            "/admissions/{id}": {
                "get": {
                    "summary": "Get a single admission receipt",
                    "operationId": "getAdmission",
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Admission receipt", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/DashboardAdmissionSummary" } } } },
                        "404": { "description": "Admission not found" }
                    }
                }
            },
            "/sweeps": {
                "get": {
                    "summary": "List quantization sweeps",
                    "operationId": "listSweeps",
                    "parameters": [
                        { "name": "artifact_digest", "in": "query", "required": false, "schema": { "type": "string" } },
                        { "name": "tensor_key", "in": "query", "required": false, "schema": { "type": "string" } }
                    ],
                    "responses": {
                        "200": { "description": "List of sweep summaries", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/DashboardSweepSummary" } } } } }
                    }
                }
            },
            "/sweeps/{id}/candidates": {
                "get": {
                    "summary": "Get candidates for a sweep",
                    "operationId": "getSweepCandidates",
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "List of sweep candidates", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/DashboardSweepCandidate" } } } } }
                    }
                }
            },
            "/execution": {
                "get": {
                    "summary": "List execution receipts",
                    "operationId": "listExecution",
                    "parameters": [
                        { "name": "artifact_digest", "in": "query", "required": false, "schema": { "type": "string" } },
                        { "name": "backend", "in": "query", "required": false, "schema": { "type": "string" } }
                    ],
                    "responses": {
                        "200": { "description": "List of execution summaries", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/DashboardExecutionSummary" } } } } }
                    }
                }
            },
            "/evidence": {
                "get": {
                    "summary": "List evidence entries",
                    "operationId": "listEvidence",
                    "parameters": [
                        { "name": "scope", "in": "query", "required": false, "schema": { "type": "string" } },
                        { "name": "kind", "in": "query", "required": false, "schema": { "type": "string" } }
                    ],
                    "responses": {
                        "200": { "description": "List of evidence entries", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/DashboardEvidenceEntry" } } } } }
                    }
                }
            },
            "/promotion-gates/{scope_id}": {
                "get": {
                    "summary": "Get promotion gate explanation",
                    "operationId": "getPromotionGate",
                    "parameters": [{ "name": "scope_id", "in": "path", "required": true, "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "Promotion gate explanation", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/DashboardPromotionExplanation" } } } }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "DashboardCImageSummary": {
                    "type": "object",
                    "properties": {
                        "digest":                 { "type": "string" },
                        "path":                   { "type": "string" },
                        "artifact_kind":          { "type": "string" },
                        "model_family":           { "type": "string" },
                        "schema_version":         { "type": "integer" },
                        "tensor_count":           { "type": "integer" },
                        "receipt_count":          { "type": "integer" },
                        "validation_status":      { "type": "string" },
                        "compiler_policy_digest": { "type": "string", "nullable": true },
                        "hardware_profile":       { "type": "string", "nullable": true },
                        "created_at":             { "type": "string", "format": "date-time" }
                    }
                },
                "DashboardTensorSummary": {
                    "type": "object",
                    "properties": {
                        "artifact_digest":  { "type": "string" },
                        "tensor_key":       { "type": "string" },
                        "tensor_class":     { "type": "string" },
                        "codec":            { "type": "string" },
                        "group_size":       { "type": "integer", "nullable": true },
                        "effective_bpw":    { "type": "number", "nullable": true },
                        "logical_shape":    { "type": "array", "items": { "type": "integer" } },
                        "payload_size":     { "type": "integer", "nullable": true },
                        "promotion_status": { "type": "string" }
                    }
                },
                "DashboardAdmissionSummary": {
                    "type": "object",
                    "properties": {
                        "receipt_id":          { "type": "string" },
                        "tensor_key":          { "type": "string" },
                        "codec":               { "type": "string" },
                        "group_size":          { "type": "integer" },
                        "effective_bpw":       { "type": "number", "nullable": true },
                        "zero_fraction":       { "type": "number", "nullable": true },
                        "neg_fraction":        { "type": "number", "nullable": true },
                        "pos_fraction":        { "type": "number", "nullable": true },
                        "scale_mean":          { "type": "number", "nullable": true },
                        "scale_std":           { "type": "number", "nullable": true },
                        "scale_max":           { "type": "number", "nullable": true },
                        "operator_nrmse":      { "type": "number", "nullable": true },
                        "output_cosine":       { "type": "number", "nullable": true },
                        "activation_shift_l2": { "type": "number", "nullable": true },
                        "deadzone_collapse":   { "type": "boolean" },
                        "rescue_required":     { "type": "boolean" },
                        "rescue_codec":        { "type": "string", "nullable": true },
                        "promotion_status":    { "type": "string" }
                    }
                },
                "DashboardSweepSummary": {
                    "type": "object",
                    "properties": {
                        "sweep_id":            { "type": "string" },
                        "artifact_digest":     { "type": "string" },
                        "tensor_key":          { "type": "string" },
                        "candidate_count":     { "type": "integer" },
                        "winner_candidate_id": { "type": "string", "nullable": true }
                    }
                },
                "DashboardSweepCandidate": {
                    "type": "object",
                    "properties": {
                        "candidate_id":      { "type": "string" },
                        "codec":             { "type": "string" },
                        "group_size":        { "type": "integer" },
                        "calibration_steps": { "type": "integer" },
                        "nrmse":             { "type": "number" },
                        "cosine":            { "type": "number" },
                        "bytes":             { "type": "integer" },
                        "passed":            { "type": "boolean" }
                    }
                },
                "DashboardExecutionSummary": {
                    "type": "object",
                    "properties": {
                        "receipt_id":        { "type": "string" },
                        "tensor_key":        { "type": "string" },
                        "kernel_name":       { "type": "string" },
                        "backend":           { "type": "string" },
                        "command_buffer_ms": { "type": "number", "nullable": true },
                        "bandwidth_gbps":    { "type": "number", "nullable": true },
                        "validation_passed": { "type": "boolean" }
                    }
                },
                "DashboardEvidenceEntry": {
                    "type": "object",
                    "properties": {
                        "receipt_id":        { "type": "string" },
                        "artifact_digest":   { "type": "string" },
                        "scope":             { "type": "string" },
                        "kind":              { "type": "string" },
                        "validation_passed": { "type": "boolean" },
                        "json":              { "type": "object" }
                    }
                },
                "DashboardPromotionExplanation": {
                    "type": "object",
                    "properties": {
                        "scope_id":          { "type": "string" },
                        "current_status":    { "type": "string" },
                        "missing_receipts":  { "type": "array", "items": { "type": "string" } },
                        "failed_gates":      { "type": "array", "items": { "type": "string" } },
                        "recommendation":    { "type": "string" }
                    }
                }
            }
        }
    })
}
