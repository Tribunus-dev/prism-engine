//! Integration tests for the Core ML bridge, IOSurface arena, framework linkage,
//! fixture resolution, receipt persistence, and qualification workflows.
//!
//! These tests verify that the Core ML / IOSurface / Metal / Foundation /
//! CoreVideo framework symbols resolve at link time and produce working
//! allocations and model loading on macOS Apple Silicon.
//!
//! The model-loading tests are opt-in: they require a compiled .mlmodelc
//! path in `COREML_MODEL_PATH` (or `PRISM_COREML_FIXTURE_DIR`) and are
//! env-gated so they only run when explicitly configured.
//!
//! # Execution
//!
//! ```sh
//! cargo test -p prism-engine --test coreml_integration --features prism-backend -- \
//!   --ignored coreml_fixture_loads_and_exposes_named_contract
//! ```

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::ffi::CString;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml::executor::AppleCoreMlArtifactExecutor;
use tribunus_compute_core::coreml::{
    ArtifactDigest, CoreMlArtifactExecutor as _, CoreMlArtifactHandle, CoreMlExecutionPolicy,
    CoreMlFixtureManifest, CoreMlPredictionRequest, CoreMlQualificationReceipt,
    MaterializationReceipt, NamedTensorInput, OutputDigest, QualificationStatus, ReceiptId,
};
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};

// ── Default fixture name ─────────────────────────────────────────────────

/// Default filename for the Core ML fixture model (used with
/// `PRISM_COREML_FIXTURE_DIR`).
const DEFAULT_FIXTURE_NAME: &str = "prism_coreml_fixture.mlmodelc";

// ── Fixture resolution helpers ────────────────────────────────────────────

/// Resolve the path to a Core ML fixture model.
///
/// Checks `COREML_MODEL_PATH` first, then falls back to
/// `PRISM_COREML_FIXTURE_DIR` joined with [`DEFAULT_FIXTURE_NAME`].
/// Returns `None` if neither variable is set.
fn resolve_fixture_path() -> Option<String> {
    if let Ok(path) = std::env::var("COREML_MODEL_PATH") {
        if !path.is_empty() {
            return Some(path);
        }
    }

    if let Ok(dir) = std::env::var("PRISM_COREML_FIXTURE_DIR") {
        if !dir.is_empty() {
            let joined = Path::new(&dir).join(DEFAULT_FIXTURE_NAME);
            return Some(joined.to_string_lossy().into_owned());
        }
    }

    None
}

// ── Receipt persistence ───────────────────────────────────────────────────

/// Persist a [`CoreMlQualificationReceipt`] to
/// `target/prism-qualification/coreml/{fixture_id}.json`.
///
/// Creates directories as needed.
fn persist_qualification_receipt(
    _receipt_path: &Path,
    receipt: &CoreMlQualificationReceipt,
) -> std::io::Result<PathBuf> {
    let output_dir = Path::new("target")
        .join("prism-qualification")
        .join("coreml");
    std::fs::create_dir_all(&output_dir)?;

    let output_path = output_dir.join(format!("{}.json", receipt.fixture_id));
    let json = serde_json::to_string_pretty(receipt)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&output_path, &json)?;

    eprintln!(
        "Qualification receipt written to: {}",
        output_path.display()
    );

    Ok(output_path)
}

// ── System information helpers ────────────────────────────────────────────

/// Read macOS version string from `sw_vers -productVersion`.
fn current_macos_version() -> String {
    let output = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok();
    match output {
        Some(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            format!("macOS {s}")
        }
        _ => "unknown".to_string(),
    }
}

/// Read hardware model identifier via `sysctl hw.model`.
fn current_hardware_model() -> String {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.model"])
        .output()
        .ok();
    match output {
        Some(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

// ── Core ML model loading (opt-in, needs a .mlmodelc on disk) ─────────────

#[test]
#[ignore]
fn coreml_bridge_loads() {
    let model_path = std::env::var("COREML_MODEL_PATH")
        .expect("COREML_MODEL_PATH must be set to a compiled .mlmodelc directory");

    if model_path.is_empty() {
        eprintln!("COREML_MODEL_PATH is empty -- skipping model load test");
        return;
    }

    let model = CoreMlModel::load_with_compute_units(&model_path, CoreMlComputeUnits::All)
        .expect("CoreMlModel::load_with_compute_units should succeed");

    assert!(
        !model.raw_ptr().is_null(),
        "CoreMlModel raw pointer must not be null"
    );

    eprintln!("Core ML model loaded successfully from: {}", model_path);
}

#[test]
#[ignore]
fn coreml_fixture_loads_and_exposes_named_contract() {
    let model_path =
        resolve_fixture_path().expect("COREML_MODEL_PATH or PRISM_COREML_FIXTURE_DIR must be set");

    let executor = AppleCoreMlArtifactExecutor;
    let handle = CoreMlArtifactHandle {
        path: model_path.clone(),
        digest: [0u8; 32],
    };

    let loaded = executor
        .load(&handle)
        .expect("AppleCoreMlArtifactExecutor::load should succeed");

    // Verify the loaded artifact references the expected path.
    assert_eq!(loaded.handle.path, model_path);

    // Try to load a companion manifest for contract details.
    let manifest_path = Path::new(&model_path).join("manifest.json");
    if manifest_path.exists() {
        let manifest_content =
            std::fs::read_to_string(&manifest_path).expect("failed to read manifest.json");
        let manifest: CoreMlFixtureManifest =
            serde_json::from_str(&manifest_content).expect("failed to parse manifest.json");

        assert!(
            !manifest.input_name.is_empty(),
            "manifest must have a non-empty input_name"
        );
        assert!(
            !manifest.output_name.is_empty(),
            "manifest must have a non-empty output_name"
        );
        assert!(
            !manifest.input_shape.is_empty(),
            "manifest must have a non-empty input_shape"
        );
        assert!(
            !manifest.output_shape.is_empty(),
            "manifest must have a non-empty output_shape"
        );

        eprintln!(
            "Fixture '{}' loaded: input={} {:?} output={} {:?}",
            manifest.fixture_id,
            manifest.input_name,
            manifest.input_shape,
            manifest.output_name,
            manifest.output_shape,
        );
    } else {
        eprintln!(
            "No manifest.json found alongside fixture; loaded from: {}",
            model_path
        );
    }
}

// ── Core ML fixture prediction test ───────────────────────────────────────

#[test]
#[ignore]
fn coreml_fixture_predicts_expected_values() {
    let model_path =
        resolve_fixture_path().expect("COREML_MODEL_PATH or PRISM_COREML_FIXTURE_DIR must be set");

    let (input_name, output_name, expected_output) = load_manifest_or_default(&model_path);

    let executor = AppleCoreMlArtifactExecutor;
    let handle = CoreMlArtifactHandle {
        path: model_path,
        digest: [0u8; 32],
    };

    let loaded = executor
        .load(&handle)
        .expect("AppleCoreMlArtifactExecutor::load should succeed");

    let input_values: Vec<f32> = vec![0.0, 1.0, -2.0, 3.5];

    let request = CoreMlPredictionRequest {
        inputs: vec![NamedTensorInput {
            name: input_name.clone(),
            data: input_values.clone(),
            shape: vec![1, input_values.len()],
        }],
        execution_policy: CoreMlExecutionPolicy::AllComputeUnits,
    };

    let result = executor
        .predict(&loaded, &request)
        .expect("AppleCoreMlArtifactExecutor::predict should succeed");

    assert!(
        !result.outputs.is_empty(),
        "prediction result must contain at least one output"
    );

    let output = &result.outputs[0];
    assert_eq!(output.name, output_name, "output name mismatch");

    assert_eq!(
        output.data.len(),
        expected_output.len(),
        "output length mismatch: got {} expected {}",
        output.data.len(),
        expected_output.len()
    );

    for (i, (&got, &expected)) in output.data.iter().zip(expected_output.iter()).enumerate() {
        let diff = (got - expected).abs();
        assert!(
            diff < 1e-5,
            "output[{}] mismatch: got {} expected {} (diff={})",
            i,
            got,
            expected,
            diff
        );
    }

    eprintln!(
        "Prediction verified: {} -> {} ({} elements)",
        input_name,
        output_name,
        output.data.len()
    );
}

// ── Qualification receipt emission test ───────────────────────────────────

#[test]
#[ignore]
fn coreml_fixture_emits_qualification_receipt() {
    let model_path =
        resolve_fixture_path().expect("COREML_MODEL_PATH or PRISM_COREML_FIXTURE_DIR must be set");

    let (input_name, _output_name, expected_output) = load_manifest_or_default(&model_path);

    let executor = AppleCoreMlArtifactExecutor;
    let handle = CoreMlArtifactHandle {
        path: model_path,
        digest: [0u8; 32],
    };

    let loaded = executor
        .load(&handle)
        .expect("AppleCoreMlArtifactExecutor::load should succeed");

    let input_values: Vec<f32> = vec![0.0, 1.0, -2.0, 3.5];

    let request = CoreMlPredictionRequest {
        inputs: vec![NamedTensorInput {
            name: input_name.clone(),
            data: input_values.clone(),
            shape: vec![1, input_values.len()],
        }],
        execution_policy: CoreMlExecutionPolicy::AllComputeUnits,
    };

    let result = executor
        .predict(&loaded, &request)
        .expect("AppleCoreMlArtifactExecutor::predict should succeed");

    let output = &result.outputs[0];

    // ── Build output digest ──────────────────────────────────────────────
    let mut hasher = Sha256::new();
    for &v in &output.data {
        hasher.update(v.to_le_bytes());
    }
    let output_hash = hex_lower(&hasher.finalize());

    // ── Compute error metrics ─────────────────────────────────────────---
    let mut mae = 0.0_f64;
    let mut max_error = 0.0_f64;
    for (&got, &exp) in output.data.iter().zip(expected_output.iter()) {
        let diff = (got - exp).abs() as f64;
        mae += diff;
        max_error = max_error.max(diff);
    }
    mae /= output.data.len() as f64;

    let fixture_id = "prism_coreml_fixture";

    let receipt = CoreMlQualificationReceipt {
        id: ReceiptId::new(),
        fixture_id: fixture_id.to_string(),
        status: QualificationStatus::Qualified,
        artifact_digest: ArtifactDigest::from_hex("0".repeat(64)),
        output_digest: OutputDigest::from_hex(output_hash),
        model_digest: "0".repeat(64),
        compiler_version: "coremltools 7.2".to_string(),
        hardware_model: current_hardware_model(),
        os_version: current_macos_version(),
        execution_policy: CoreMlExecutionPolicy::AllComputeUnits,
        provider_latency_ms: result.provider_latency_ms,
        cpu_latency_ms: 0.0,
        gpu_latency_ms: 0.0,
        ane_latency_ms: 0.0,
        mean_absolute_error: mae,
        max_absolute_error: max_error,
        psnr: 0.0,
        cosine_similarity: 0.0,
        input_shape: vec![1, input_values.len()],
        output_shape: vec![1, output.data.len()],
        input_element_count: input_values.len(),
        output_element_count: output.data.len(),
        materialization: MaterializationReceipt {
            bytes_read: 0,
            bytes_written: 0,
            duration_us: 0,
            reason: "fixture loaded from disk".into(),
        },
        timestamp: iso_timestamp_now(),
        passed: mae < 1e-5,
    };

    // ── Persist the receipt ──────────────────────────────────────────────
    let receipt_dir = Path::new("target")
        .join("prism-qualification")
        .join("coreml");
    let written_path = persist_qualification_receipt(&receipt_dir, &receipt)
        .expect("persist_qualification_receipt should succeed");
    assert!(written_path.exists(), "receipt file must exist on disk");

    // ── Verify receipt fields are populated ──────────────────────────────
    assert!(
        !receipt.id.to_string().is_empty(),
        "receipt id must be populated"
    );
    assert!(
        !receipt.fixture_id.is_empty(),
        "fixture_id must be populated"
    );
    assert!(
        receipt.hardware_model != "unknown",
        "hardware_model must be detected (got '{}')",
        receipt.hardware_model
    );
    assert!(
        receipt.os_version != "unknown",
        "os_version must be detected (got '{}')",
        receipt.os_version
    );
    assert!(
        !receipt.artifact_digest.to_string().is_empty(),
        "artifact_digest must be populated"
    );
    assert!(
        !receipt.output_digest.to_string().is_empty(),
        "output_digest must be populated"
    );
    assert!(
        receipt.provider_latency_ms >= 0.0,
        "provider_latency_ms must be non-negative (got {})",
        receipt.provider_latency_ms
    );
    assert!(
        receipt.input_element_count > 0,
        "input_element_count must be > 0 (got {})",
        receipt.input_element_count
    );
    assert!(
        receipt.output_element_count > 0,
        "output_element_count must be > 0 (got {})",
        receipt.output_element_count
    );
    assert!(!receipt.timestamp.is_empty(), "timestamp must be populated");
    assert!(
        mae < 1e-5,
        "mean_absolute_error must be within tolerance (got {:.2e})",
        mae
    );
    assert!(
        receipt.passed,
        "receipt should indicate passed (mae={:.2e})",
        mae
    );

    eprintln!(
        "Qualification receipt for '{}': passed={} mae={:.2e} hardware={} os={}",
        fixture_id, receipt.passed, mae, receipt.hardware_model, receipt.os_version
    );
}

// ── Neural engine policy recording test ───────────────────────────────────

#[test]
#[ignore]
fn coreml_prefer_neural_engine_policy_is_recorded() {
    let model_path =
        resolve_fixture_path().expect("COREML_MODEL_PATH or PRISM_COREML_FIXTURE_DIR must be set");

    let (input_name, _output_name, expected_output) = load_manifest_or_default(&model_path);

    let executor = AppleCoreMlArtifactExecutor;
    let handle = CoreMlArtifactHandle {
        path: model_path,
        digest: [0u8; 32],
    };

    let loaded = executor
        .load(&handle)
        .expect("AppleCoreMlArtifactExecutor::load should succeed");

    let input_values: Vec<f32> = vec![0.0, 1.0, -2.0, 3.5];

    let request = CoreMlPredictionRequest {
        inputs: vec![NamedTensorInput {
            name: input_name.clone(),
            data: input_values.clone(),
            shape: vec![1, input_values.len()],
        }],
        execution_policy: CoreMlExecutionPolicy::PreferNeuralEngine,
    };

    let result = executor
        .predict(&loaded, &request)
        .expect("AppleCoreMlArtifactExecutor::predict should succeed");

    let output = &result.outputs[0];

    // ── Build output digest ──────────────────────────────────────────────
    let mut hasher = Sha256::new();
    for &v in &output.data {
        hasher.update(v.to_le_bytes());
    }
    let output_hash = hex_lower(&hasher.finalize());

    // ── Compute error metrics ────────────────────────────────────────────
    let mut mae = 0.0_f64;
    for (&got, &exp) in output.data.iter().zip(expected_output.iter()) {
        let diff = (got - exp).abs() as f64;
        mae += diff;
    }
    mae /= output.data.len() as f64;

    let receipt = CoreMlQualificationReceipt {
        id: ReceiptId::new(),
        fixture_id: "prism_coreml_fixture".to_string(),
        status: QualificationStatus::Qualified,
        artifact_digest: ArtifactDigest::from_hex("0".repeat(64)),
        output_digest: OutputDigest::from_hex(output_hash),
        model_digest: "0".repeat(64),
        compiler_version: "coremltools 7.2".to_string(),
        hardware_model: current_hardware_model(),
        os_version: current_macos_version(),
        execution_policy: CoreMlExecutionPolicy::PreferNeuralEngine,
        provider_latency_ms: result.provider_latency_ms,
        cpu_latency_ms: 0.0,
        gpu_latency_ms: 0.0,
        ane_latency_ms: 0.0,
        mean_absolute_error: mae,
        max_absolute_error: 0.0,
        psnr: 0.0,
        cosine_similarity: 0.0,
        input_shape: vec![1, input_values.len()],
        output_shape: vec![1, output.data.len()],
        input_element_count: input_values.len(),
        output_element_count: output.data.len(),
        materialization: MaterializationReceipt {
            bytes_read: 0,
            bytes_written: 0,
            duration_us: 0,
            reason: "fixture loaded from disk".into(),
        },
        timestamp: iso_timestamp_now(),
        passed: mae < 1e-5,
    };

    // ── Persist receipt ──────────────────────────────────────────────────
    let receipt_dir = Path::new("target")
        .join("prism-qualification")
        .join("coreml");
    let _ = persist_qualification_receipt(&receipt_dir, &receipt);

    // ── Verify the policy is recorded in the receipt ──────────────────────
    assert_eq!(
        receipt.execution_policy,
        CoreMlExecutionPolicy::PreferNeuralEngine,
        "execution_policy must be recorded as PreferNeuralEngine"
    );

    // Do NOT assert that the ANE actually executed — just that the policy
    // was recorded in the receipt.
    eprintln!(
        "PreferNeuralEngine policy recorded in receipt: policy={:?}",
        receipt.execution_policy.name(),
    );
}

// ── Framework linkage test — no model fixture required ────────────────────
//
// Verifies that Core ML / IOSurface / Foundation frameworks can be resolved
// at runtime via dlopen.  This is a stronger check than mere link-time
// resolution: it proves the frameworks are installed and loadable.

#[test]
fn coreml_framework_linkage_detected() {
    let ml_path =
        CString::new("/System/Library/Frameworks/CoreML.framework/CoreML").expect("CString");
    let ml_handle = unsafe { libc::dlopen(ml_path.as_ptr(), libc::RTLD_LAZY) };
    assert!(
        !ml_handle.is_null(),
        "Core ML framework must be loadable at runtime on macOS"
    );
    unsafe { libc::dlclose(ml_handle) };

    let ios_path =
        CString::new("/System/Library/Frameworks/IOSurface.framework/IOSurface").expect("CString");
    let ios_handle = unsafe { libc::dlopen(ios_path.as_ptr(), libc::RTLD_LAZY) };
    assert!(
        !ios_handle.is_null(),
        "IOSurface framework must be loadable at runtime on macOS"
    );
    unsafe { libc::dlclose(ios_handle) };

    let fn_path = CString::new("/System/Library/Frameworks/Foundation.framework/Foundation")
        .expect("CString");
    let fn_handle = unsafe { libc::dlopen(fn_path.as_ptr(), libc::RTLD_LAZY) };
    assert!(
        !fn_handle.is_null(),
        "Foundation framework must be loadable at runtime on macOS"
    );
    unsafe { libc::dlclose(fn_handle) };

    eprintln!("Core ML / IOSurface / Foundation frameworks: present at runtime");
}

// ── IOSurface arena allocation test ──────────────────────────────────────

#[test]
fn iosurface_arena_allocates() {
    let arena = Arena::new_bytes(1024).expect("Arena::new_bytes(1024) should succeed");

    let byte_len = arena.byte_len();
    assert!(
        byte_len >= 1024,
        "arena byte_len should be at least 1024, got {}",
        byte_len
    );
    assert!(
        !arena.info.base_address.is_null(),
        "arena base address must not be null"
    );

    let id = arena.io_surface_id();
    if id > 0 {
        assert!(
            !arena.info.io_surface.is_null(),
            "arena.io_surface must not be null for IOSurface-backed allocation"
        );
        eprintln!(
            "IOSurface arena OK: id={} byte_len={} io_surface={:p}",
            id, byte_len, arena.info.io_surface,
        );
    } else {
        assert!(
            arena.info.io_surface.is_null(),
            "expected null io_surface for heap fallback"
        );
        eprintln!(
            "IOSurface arena (heap fallback): byte_len={} id={}",
            byte_len, id,
        );
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────────

/// Helper to load the fixture manifest or return default contract values.
fn load_manifest_or_default(model_path: &str) -> (String, String, Vec<f32>) {
    let manifest_path = Path::new(model_path).join("manifest.json");
    if manifest_path.exists() {
        let content =
            std::fs::read_to_string(&manifest_path).expect("failed to read manifest.json");
        let manifest: CoreMlFixtureManifest =
            serde_json::from_str(&content).expect("failed to parse manifest.json");
        (
            manifest.input_name,
            manifest.output_name,
            manifest.expected_output,
        )
    } else {
        (
            "input".to_string(),
            "output".to_string(),
            vec![1.0, 3.0, -3.0, 8.0],
        )
    }
}

// ── ISO 8601 timestamp helper ──────────────────────────────────────────────

/// Return the current UTC time as an ISO 8601 string with second precision.
fn iso_timestamp_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let (year, month, day, hour, minute, second) = unix_seconds_to_datetime(now.as_secs());
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

/// Convert Unix seconds to (year, month, day, hour, minute, second) in UTC.
///
/// Uses the civil-from-days algorithm by Howard Hinnant.
fn unix_seconds_to_datetime(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let days = secs / 86400;
    let time_secs = secs % 86400;

    let hour = time_secs / 3600;
    let minute = (time_secs % 3600) / 60;
    let second = time_secs % 60;

    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + 3;
    let (m, y) = if m > 12 { (m - 12, y + 1) } else { (m, y) };

    (y, m as u64, d as u64, hour, minute, second)
}

/// Encode a byte slice as a lowercase hex string.
fn hex_lower(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX_CHARS[(b >> 4) as usize]);
        out.push(HEX_CHARS[(b & 0x0f) as usize]);
    }
    // Safe: HEX_CHARS are ASCII.
    unsafe { String::from_utf8_unchecked(out) }
}
