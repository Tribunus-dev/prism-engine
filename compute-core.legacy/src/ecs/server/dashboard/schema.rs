/// CImage artifact metadata table.
pub const CREATE_ARTIFACTS: &str = "CREATE TABLE IF NOT EXISTS cimage_artifacts (
    digest TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    model_family TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    tensor_count INTEGER NOT NULL DEFAULT 0,
    receipt_count INTEGER NOT NULL DEFAULT 0,
    validation_status TEXT NOT NULL DEFAULT 'Unknown',
    compiler_policy_digest TEXT,
    hardware_profile TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    manifest_json TEXT
);";

/// Tensor metadata table within a CImage artifact.
pub const CREATE_TENSORS: &str = "CREATE TABLE IF NOT EXISTS cimage_tensors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_digest TEXT NOT NULL REFERENCES cimage_artifacts(digest),
    tensor_key TEXT NOT NULL,
    tensor_class TEXT NOT NULL,
    codec TEXT NOT NULL,
    group_size INTEGER,
    effective_bpw REAL,
    logical_shape TEXT,
    payload_size INTEGER,
    promotion_status TEXT NOT NULL DEFAULT 'ResearchOnly',
    UNIQUE(artifact_digest, tensor_key)
);
CREATE INDEX IF NOT EXISTS idx_tensors_artifact ON cimage_tensors(artifact_digest);
CREATE INDEX IF NOT EXISTS idx_tensors_codec ON cimage_tensors(codec);
CREATE INDEX IF NOT EXISTS idx_tensors_class ON cimage_tensors(tensor_class);";

/// Admission receipt table — records each quantization trial for a tensor.
pub const CREATE_ADMISSIONS: &str = "CREATE TABLE IF NOT EXISTS admission_receipts (
    receipt_id TEXT PRIMARY KEY,
    artifact_digest TEXT NOT NULL REFERENCES cimage_artifacts(digest),
    tensor_key TEXT NOT NULL,
    codec TEXT NOT NULL,
    group_size INTEGER NOT NULL,
    effective_bpw REAL,
    zero_fraction REAL,
    neg_fraction REAL,
    pos_fraction REAL,
    scale_mean REAL,
    scale_std REAL,
    scale_max REAL,
    operator_nrmse REAL,
    output_cosine REAL,
    activation_shift_l2 REAL,
    deadzone_collapse INTEGER DEFAULT 0,
    rescue_required INTEGER DEFAULT 0,
    rescue_codec TEXT,
    promotion_status TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_admissions_artifact ON admission_receipts(artifact_digest);
CREATE INDEX IF NOT EXISTS idx_admissions_codec ON admission_receipts(codec);
CREATE INDEX IF NOT EXISTS idx_admissions_status ON admission_receipts(promotion_status);";

/// Execution receipt table — records Metal dispatch timing and validation results.
pub const CREATE_EXECUTION: &str = "CREATE TABLE IF NOT EXISTS execution_receipts (
    receipt_id TEXT PRIMARY KEY,
    artifact_digest TEXT REFERENCES cimage_artifacts(digest),
    tensor_key TEXT NOT NULL,
    kernel_name TEXT NOT NULL,
    backend TEXT NOT NULL,
    command_buffer_ms REAL,
    effective_bandwidth_gbps REAL,
    metal_vs_cpu_nrmse REAL,
    validation_passed INTEGER DEFAULT 0,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);";

/// Sweep table — groups quantization candidates explored for a single tensor.
pub const CREATE_SWEEPS: &str = "CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id TEXT PRIMARY KEY,
    artifact_digest TEXT NOT NULL REFERENCES cimage_artifacts(digest),
    tensor_key TEXT NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    winner_candidate_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);";

/// Sweep candidate table — individual quantization configuration trialled in a sweep.
pub const CREATE_CANDIDATES: &str = "CREATE TABLE IF NOT EXISTS sweep_candidates (
    candidate_id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(sweep_id),
    codec TEXT NOT NULL,
    group_size INTEGER NOT NULL,
    calibration_steps INTEGER NOT NULL,
    nrmse REAL NOT NULL,
    cosine REAL NOT NULL,
    bytes INTEGER NOT NULL,
    passed INTEGER NOT NULL DEFAULT 0
);";

/// Evidence ledger — immutable log of validation outcomes across scopes.
pub const CREATE_EVIDENCE: &str = "CREATE TABLE IF NOT EXISTS evidence_ledger (
    receipt_id TEXT PRIMARY KEY,
    artifact_digest TEXT NOT NULL REFERENCES cimage_artifacts(digest),
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    validation_passed INTEGER NOT NULL DEFAULT 0,
    json_data TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);";

/// Calibration run summary table.
pub const CREATE_CALIBRATIONS: &str = "CREATE TABLE IF NOT EXISTS calibrations (
    calibration_id TEXT PRIMARY KEY,
    stage_1_codec TEXT NOT NULL,
    stage_2_codec TEXT,
    loss_operator_nrmse REAL,
    loss_cosine REAL,
    execution_environment TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);";

/// Returns all CREATE TABLE statements in dependency order.
pub fn all_schema() -> Vec<&'static str> {
    vec![
        CREATE_ARTIFACTS,
        CREATE_TENSORS,
        CREATE_ADMISSIONS,
        CREATE_EXECUTION,
        CREATE_SWEEPS,
        CREATE_CANDIDATES,
        CREATE_EVIDENCE,
        CREATE_CALIBRATIONS,
    ]
}
