//! Metal kernel source, architecture constants, and on-the-fly compilation.
//!
//! Provides the full Gemma 4 48-layer transformer Metal shader source string
//! alongside Rust-side compilation helpers and shared layout constants.

use metal::*;

// ── Architecture constants (Gemma 4 12B Unified) ───────────────────
#[allow(dead_code)]
pub const HIDDEN_DIM: u32 = 3840;
pub const LAYERS: u32 = 48;
#[allow(dead_code)]
pub const NUM_Q_HEADS: u32 = 16;
pub const NUM_KV_HEADS: u32 = 8;
#[allow(dead_code)]
pub const HEAD_DIM: u32 = 256;
pub const GLOBAL_HEAD_DIM: u32 = 512;
#[allow(dead_code)]
pub const FFN_INTERMEDIATE: u32 = 15360;
pub const VOCAB_SIZE: u32 = 262144;
pub const MAX_CONTEXT: u32 = 2048; // KV cache slots (limited by 16 GB SRAM + device mem)
pub const NUM_CENTROIDS: u32 = 256;
pub const NUM_MTP_HEADS: u32 = 4;
pub const MTP_HIDDEN: u32 = 2048;
pub const MTP_FFN_INTER: u32 = 8192;
pub const MTP_TILES: u32 = (MTP_HIDDEN + 640) / 640; // 4
pub const MTP_TILES_FFN: u32 = (MTP_FFN_INTER + 640) / 640; // 13
pub const MAX_DRAFT_CANDIDATES: u32 = 5;
pub const DRAFT_HIDDEN: u32 = 768;
pub const TILE: u32 = 640;
#[allow(dead_code)]
const MAGIC_DIV3: u32 = 2863311531;

// ── Work queue constants ───────────────────────────────────────────
pub const NUM_SLOTS: u32 = 256;
pub const SLOT_U32_COUNT: u32 = 4 + VOCAB_SIZE; // 262148
pub const SLOT_BYTE_COUNT: u64 = SLOT_U32_COUNT as u64 * 4; // 1,048,592

// ── Ternary KV block constants ────────────────────────────────────

// ── Weight-offset constants for the per-layer matrix layout ────────
// Each matrix's flat element count BEFORE Base-3 nibble packing.
// For tile-GEMV indexing we compute nibble offsets at runtime.
#[allow(dead_code)]
const Q_COLS: u32 = NUM_Q_HEADS * HEAD_DIM; // 4096
#[allow(dead_code)]
const KV_COLS: u32 = NUM_KV_HEADS * HEAD_DIM; // 2048
#[allow(dead_code)]
const O_ROWS: u32 = Q_COLS; // 4096
#[allow(dead_code)]
const DOWN_ROWS: u32 = FFN_INTERMEDIATE; // 15360

// ====================================================================
//  Metal Shader Source
// ====================================================================

pub const SHADER_SRC: &str = include_str!("shaders/gemma4_full.metal");

// ====================================================================
//  INT4 Fused Ternary Variant (M5-optimized, 5-per-byte ternary)
// ====================================================================

pub const SHADER_SRC_INT4: &str = include_str!("shaders/gemma4_full_int4.metal");

/// T32 coalesced uint4 GEMV production kernel.
/// 4 rows per TG, 128 threads (4 SIMD groups × 32 lanes).
/// Activation loaded once into SRAM, shared across all 4 rows.
/// Weights read via uint4 vector loads (32 threads read same block, broadcast).
/// Per-lane `/3` and `%3` trit extraction (no magic-division overflow).
/// SRAM-based reduction across SIMD group (no simd_sum issues).
pub const PERSISTENT_GEMV_SRC: &str = include_str!("shaders/persistent_gemv.metal");
// ====================================================================
//  Compilation
// ====================================================================
pub(crate) fn compile_kernel(device: &Device, int4: bool) -> Result<ComputePipelineState, String> {
    let shader_src = if int4 { SHADER_SRC_INT4 } else { SHADER_SRC };
    let tmp = std::env::temp_dir().join("tribunus-full-transformer");
    let _ = std::fs::create_dir_all(&tmp);

    let src_path = tmp.join("gemma4_full.metal");
    let air_path = tmp.join("gemma4_full.air");
    let lib_path = tmp.join("gemma4_full.metallib");

    std::fs::write(&src_path, shader_src)
        .map_err(|e| format!("failed to write Metal source: {e}"))?;

    // Step 1: Compile .metal → .air via metal compiler
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-c"]);
    cmd.arg(src_path.to_str().unwrap())
        .arg("-o")
        .arg(air_path.to_str().unwrap());
    let status = cmd.status().map_err(|e| format!("xcrun metal: {e}"))?;
    if !status.success() {
        return Err("Metal source compilation failed".into());
    }

    // Step 2: Link .air → .metallib via metallib linker
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib", "-o"]);
    cmd.arg(lib_path.to_str().unwrap())
        .arg(air_path.to_str().unwrap());
    let status = cmd.status().map_err(|e| format!("xcrun metallib: {e}"))?;
    if !status.success() {
        return Err("Metal library linking failed".into());
    }

    let lib_data = std::fs::read(&lib_path).map_err(|e| format!("read metallib: {e}"))?;
    let library = device
        .new_library_with_data(&lib_data)
        .map_err(|e| format!("new_library: {:?}", e))?;
    let function = library
        .get_function("gemma4_full_decode_persistent", None)
        .map_err(|e| format!("get_function: {:?}", e))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state: {:?}", e))
}

/// Extract a named function from an already-loaded library and return its pipeline state.
pub fn compile_function_from_lib(
    device: &Device,
    library: &LibraryRef,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|e| format!("get_function({:?}): {:?}", name, e))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state for {:?}: {:?}", name, e))
}

/// Load a pre-compiled .metallib from bytes (alias for INT4 variant — same shader).
pub fn compile_kernel_from_metallib_int4(
    device: &Device,
    data: &[u8],
) -> Result<ComputePipelineState, String> {
    compile_kernel_from_metallib(device, data)
}

/// Load a pre-compiled .metallib from bytes and create a pipeline state.
pub fn compile_kernel_from_metallib(
    device: &Device,
    data: &[u8],
) -> Result<ComputePipelineState, String> {
    let library = device
        .new_library_with_data(data)
        .map_err(|e| format!("new_library_with_data: {:?}", e))?;
    let function = library
        .get_function("gemma4_full_decode_persistent", None)
        .map_err(|e| format!("get_function: {:?}", e))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state for {:?}: {:?}", "gemma4_full_decode_persistent", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_src_is_non_empty() {
        assert!(!SHADER_SRC.is_empty(), "SHADER_SRC must not be empty");
    }

    #[test]
    fn shader_src_contains_kernel_entry_point() {
        assert!(
            SHADER_SRC.contains("gemma4_full_decode_persistent"),
            "SHADER_SRC must contain the kernel entry point"
        );
    }

    #[test]
    fn shader_src_contains_metal_header() {
        assert!(
            SHADER_SRC.contains("<metal_stdlib>"),
            "SHADER_SRC must include metal_stdlib"
        );
    }

    #[test]
    fn shader_src_int4_is_non_empty() {
        assert!(!SHADER_SRC_INT4.is_empty(), "SHADER_SRC_INT4 must not be empty");
    }

    #[test]
    fn shader_src_int4_contains_kernel_entry_point() {
        assert!(
            SHADER_SRC_INT4.contains("gemma4_full_decode_persistent"),
            "SHADER_SRC_INT4 must contain the kernel entry point"
        );
    }

    #[test]
    fn persistent_gemv_src_is_non_empty() {
        assert!(!PERSISTENT_GEMV_SRC.is_empty(), "PERSISTENT_GEMV_SRC must not be empty");
    }

    #[test]
    fn persistent_gemv_contains_kernel_entry_point() {
        assert!(
            PERSISTENT_GEMV_SRC.contains("matvec_persistent_t32_coalesced"),
            "PERSISTENT_GEMV_SRC must contain the kernel entry point"
        );
    }
}
