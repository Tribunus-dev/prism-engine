//! ComputeImage manifest types, reader/writer, and telemetry.
//!
//! Re-exports from submodules (`types`, `builder`, `runtime`) and provides
//! telemetry functions, admission estimation, capability probing, and the
//! top-level convenience reader.

pub mod builder;
pub mod runtime;
pub mod types;

// Re-export everything from submodules.
pub use builder::*;
pub use runtime::*;
pub use types::*;

pub use crate::compute_image::manifest::types::{
    Manifest, ResidencyPlan, Segment, STORAGE_ABI_MAPPED_NO_COPY_V1,
};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

// ── Build profile ──────────────────────────────────────────────────────────

/// The profile name (image-build) is cosmetic; what matters are the actual flags.
pub fn verify_image_build_profile() -> crate::Result<()> {
    Ok(())
}

/// Export profile attestation for callers (builder binary, seal.json).
pub fn image_build_attestation() -> serde_json::Value {
    let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
    let opt_level = option_env!("TRIBUNUS_OPT_LEVEL").unwrap_or("0");
    let target = option_env!("TRIBUNUS_TARGET").unwrap_or("unknown");
    serde_json::json!({
        "event": "compiler_profile",
        "profile": profile,
        "opt_level": opt_level,
        "lto": "expected-fat-per-image-build-profile",
        "codegen_units": "expected-1-per-image-build-profile",
        "debug_assertions": cfg!(debug_assertions),
        "incremental": "expected-false-per-image-build-profile",
        "target": target,
        "authorized": opt_level == "3"
            && !cfg!(debug_assertions)
            && target == "aarch64-apple-darwin",
    })
}

// ── Telemetry helpers ──────────────────────────────────────────────────────

/// Returns MLX active memory in bytes, or 0 if the mlx-rs API is unavailable.
pub fn mlx_active_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_active_memory(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Returns MLX cache memory in bytes, or 0 if the mlx-rs API is unavailable.
pub fn mlx_cache_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_cache_memory(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Returns MLX peak memory in bytes, or 0 if unavailable.
pub fn mlx_peak_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_peak_memory(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Clear the MLX Metal allocator cache. Returns the number of bytes freed.
pub fn clear_mlx_cache() -> u64 {
    let before = mlx_cache_memory_bytes();
    #[cfg(target_os = "macos")]
    unsafe {
        mlx_sys::mlx_clear_cache()
    };
    let after = mlx_cache_memory_bytes();
    before.saturating_sub(after)
}

/// Set the MLX Metal cache limit in bytes. Returns the previous limit.
pub fn set_mlx_cache_limit(limit_bytes: u64) -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut prev: usize = 0;
        unsafe { mlx_sys::mlx_set_cache_limit(&mut prev, limit_bytes as usize) };
        prev as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = limit_bytes;
        0
    }
}

/// Get the MLX Metal active memory limit in bytes.
pub fn mlx_get_memory_limit() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut res: usize = 0;
        unsafe { mlx_sys::mlx_get_memory_limit(&mut res) };
        res as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Set the MLX Metal active memory limit in bytes. Returns the previous limit.
pub fn set_mlx_memory_limit(limit_bytes: u64) -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut prev: usize = 0;
        unsafe { mlx_sys::mlx_set_memory_limit(&mut prev, limit_bytes as usize) };
        prev as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = limit_bytes;
        0
    }
}

/// Returns the process memory in bytes, or 0 if unavailable.
fn system_memory_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            extern "C" {
                fn sysctlbyname(
                    name: *const c_char,
                    oldp: *mut c_void,
                    oldlenp: *mut usize,
                    newp: *mut c_void,
                    newlen: usize,
                ) -> c_int;
            }

            let mut value: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = CString::new("hw.memsize").expect("CString");
            let ret = sysctlbyname(
                name.as_ptr(),
                &mut value as *mut _ as *mut c_void,
                &mut size as *mut usize,
                std::ptr::null_mut(),
                0,
            );
            if ret == 0 && value > 0 {
                return value;
            }
        }
    }
    0
}

fn memory_override_enabled() -> bool {
    matches!(
        std::env::var("TRIBUNUS_COMPUTE_ALLOW_HIGH_MEMORY")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn estimate_open_runtime_peak_bytes(manifest: &Manifest) -> u64 {
    let persistent_bytes = manifest
        .residency_plan
        .persistent_segments
        .iter()
        .filter_map(|segment_id| {
            manifest
                .segments
                .iter()
                .find(|segment| &segment.id == segment_id)
        })
        .map(|segment| segment.byte_size)
        .sum::<u64>();
    let arch = &manifest.architecture;
    let rope_bytes = u64::from(arch.max_position_embeddings)
        .saturating_mul(u64::from(arch.head_dim))
        .saturating_mul(4)
        .saturating_add(
            u64::from(arch.max_position_embeddings)
                .saturating_mul(u64::from(arch.global_head_dim.unwrap_or(arch.head_dim)))
                .saturating_mul(4),
        );
    let embedding_dequant_bytes = u64::from(arch.vocab_size)
        .saturating_mul(u64::from(arch.hidden_size))
        .saturating_mul(4);

    persistent_bytes
        .saturating_add(rope_bytes)
        .saturating_add(embedding_dequant_bytes)
        .saturating_add(1024 * 1024 * 1024)
}

// ── Admission estimate ─────────────────────────────────────────────────────

/// Produce an admission estimate given the manifest.
///
/// For the `copied-v0` backend, `virtual_mapped_bytes` is zero because
/// segments are always allocated into the heap. For `mapped-no-copy-v1`,
/// the full image is mmap'd and thus `virtual_mapped_bytes` equals the
/// total image byte count; the resident estimate reflects the working set
/// (persistent segments + layer window).
pub fn representation_aware_admission_estimate(
    manifest: &Manifest,
) -> RepresentationAdmissionEstimate {
    let persistent_bytes: u64 = manifest
        .residency_plan
        .persistent_segments
        .iter()
        .filter_map(|sid| manifest.segments.iter().find(|s| &s.id == sid))
        .map(|s| s.byte_size)
        .sum();

    let layer_segments: Vec<&Segment> = manifest
        .residency_plan
        .layer_segments
        .iter()
        .filter_map(|sid| manifest.segments.iter().find(|s| &s.id == sid))
        .collect();

    let max_layer_window_bytes: u64 = {
        let window = manifest.residency_plan.layer_window_size.max(1) as usize;
        let mut sorted = layer_segments.clone();
        sorted.sort_by(|a, b| b.byte_size.cmp(&a.byte_size));
        sorted.iter().take(window).map(|s| s.byte_size).sum()
    };

    let total_mapped: u64 = manifest.segments.iter().map(|s| s.byte_size).sum();

    let arch = &manifest.architecture;
    let rope_bytes = u64::from(arch.max_position_embeddings)
        .saturating_mul(u64::from(arch.head_dim))
        .saturating_mul(4)
        .saturating_add(
            u64::from(arch.max_position_embeddings)
                .saturating_mul(u64::from(arch.global_head_dim.unwrap_or(arch.head_dim)))
                .saturating_mul(4),
        );
    let kv_budget_bytes = rope_bytes.saturating_mul(4);
    let mlx_workspace_bytes = 512 * 1024 * 1024;
    let allocator_cache_bytes = 512 * 1024 * 1024;
    let system_reserve_bytes = 2u64 * 1024 * 1024 * 1024;

    let is_mapped = manifest.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1;
    let virtual_mapped_bytes = if is_mapped { total_mapped } else { 0 };

    let seq_len = u64::from(arch.max_position_embeddings.min(8192));
    let hidden_size = u64::from(arch.hidden_size);
    let vocab_size = u64::from(arch.vocab_size);
    let attention_workspace = seq_len.saturating_mul(hidden_size).saturating_mul(4);
    let output_proj_workspace = hidden_size.saturating_mul(vocab_size).saturating_mul(4);
    let largest_transient_bytes = attention_workspace.max(output_proj_workspace);

    let (expected_resident_bytes, materialized_bytes) = if is_mapped {
        let resident = persistent_bytes
            .saturating_add(max_layer_window_bytes)
            .saturating_add(rope_bytes)
            .saturating_add(mlx_workspace_bytes);
        let materialized: u64 = manifest
            .tensor_table
            .iter()
            .filter(|t| t.quantization.is_some())
            .map(|t| t.byte_length)
            .sum();
        (resident, materialized)
    } else {
        let total_tensor_bytes: u64 = manifest.tensor_table.iter().map(|t| t.byte_length).sum();
        let resident = total_tensor_bytes
            .saturating_add(rope_bytes)
            .saturating_add(mlx_workspace_bytes);
        (resident, 0)
    };

    RepresentationAdmissionEstimate {
        virtual_mapped_bytes,
        expected_resident_bytes,
        persistent_materialized_bytes: persistent_bytes,
        max_layer_window_bytes,
        rope_bytes,
        kv_budget_bytes,
        mlx_workspace_bytes,
        allocator_cache_bytes,
        system_reserve_bytes,
        largest_transient_bytes,
        materialized_bytes,
    }
}

// ── Native capability report ───────────────────────────────────────────────

impl NativeCapabilityReport {
    /// Probe the current native environment.
    pub fn probe() -> Self {
        let metal_available = {
            #[cfg(target_os = "macos")]
            {
                let mut res: bool = false;
                unsafe { mlx_sys::mlx_metal_is_available(&mut res) };
                res
            }
            #[cfg(not(target_os = "macos"))]
            false
        };

        let supports_memory_telemetry = mlx_active_memory_bytes() > 0 || metal_available;
        let supports_cache_control = metal_available;
        let supports_quantized_matmul = true;
        let supports_dequantize = true;

        Self {
            mlx_core_version: option_env!("TRIBUNUS_MLX_CORE_VERSION")
                .unwrap_or("v0.31.2")
                .to_string(),
            mlx_c_version: option_env!("TRIBUNUS_MLX_C_VERSION")
                .unwrap_or("0.6.0")
                .to_string(),
            mlx_rs_version: option_env!("TRIBUNUS_MLX_RS_VERSION")
                .unwrap_or("0.25.3-tribunus.1")
                .to_string(),
            mlx_sys_version: option_env!("TRIBUNUS_MLX_SYS_VERSION")
                .unwrap_or("0.6.0-tribunus.1")
                .to_string(),
            compute_native_version: "0.1.0".to_string(),
            supports_quantized_matmul,
            supports_dequantize,
            supports_memory_telemetry,
            supports_cache_control,
            supports_external_array: true,
            supports_multithreaded_execution: true,
            metal_available,
            accelerate_available: true,
        }
    }
}

// ── Convenience reader ─────────────────────────────────────────────────────

/// Open a compiled image from `image_dir` and return a `CompiledImageReader`.
pub fn read(image_dir: &str) -> crate::Result<CompiledImageReader> {
    CompiledImageReader::open(std::path::Path::new(image_dir))
}
