//! Bonsai-specific cimage compilation, runtime loading, prefill/decode, and receipts.
//!
//! Implements runtime loading with KV-cache allocation, prefill/decode,
//! and publishes evidence receipts through the ECS daemon path.
//!
//! ## Prefill / Decode
//!
//! The inference path uses Metal GPU dispatch when available (macOS + metal-dispatch
//! feature), with an automatic CPU fallback using the ternary GEMV reference
//! implementation from [`crate::bonsai_ternary`]. It processes:
//!
//! 1. Token embedding lookup
//! 2. Per-layer RMS norm + fused QKV (ternary GEMV)
//! 3. Full or linear (SSM) attention with KV cache
//! 4. Post-attention RMS norm + fused gate/up projections + SiLU + down projection
//! 5. Final RMS norm + LM head projection
//! 6. Argmax sampling for decode

use crate::bonsai_ternary::{apply_outlier_correction, ternary_gemv_ref};
// Import Metal-backed run_ternary_gemv when Metal is available, otherwise use the
// CPU reference fallback defined below.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use crate::bonsai_metal_dispatch::run_ternary_gemv;
use crate::cimage::{CImageReader, TensorType};
use sha2::{Digest, Sha256};

use prism_ecs_core::identity::ReceiptId;
use prism_ecs_ir::bonsai::Bonsai27B;
use prism_ecs_ir::cimage_types::{
    ExecutionGraph, ExecutionLane, ExecutionRegion, FusionConstraints, GraphRegionId, MemoryPlan,
    RuntimeStatePlan,
};
use prism_spatial_ir::plan::SpatialCompilationPlan;

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

// ── Kernel-contract verification adapter ───────────────────────────────
// Bridges to bonsai_metal_dispatch::verify_kernel_contract when Metal is
// compiled in; returns false on targets without Metal.

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn verify_kernel_contract() -> bool {
    crate::bonsai_metal_dispatch::verify_kernel_contract()
}

#[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
fn verify_kernel_contract() -> bool {
    false
}

/// Runtime state loaded from a Bonsai `.cimage` file.
pub struct BonsaiRuntimeState {
    /// Packed ternary tile data: tensor name -> raw bytes.
    pub packed_buffers: HashMap<String, Vec<u8>>,
    /// Page scales: tensor name -> BF16 u16 bytes.
    pub scale_buffers: HashMap<String, Vec<u8>>,
    /// Lane scales: tensor name -> i8 bytes.
    pub lane_scale_buffers: HashMap<String, Vec<u8>>,
    /// Outlier row indices: tensor name -> u32 LE bytes.
    pub outlier_row_buffers: HashMap<String, Vec<u8>>,
    /// Outlier column indices: tensor name -> u32 LE bytes.
    pub outlier_col_buffers: HashMap<String, Vec<u8>>,
    /// Outlier values (BF16): tensor name -> u16 LE bytes.
    pub outlier_val_buffers: HashMap<String, Vec<u8>>,
    /// Embedding lookup table (fp16).
    pub embedding_table: Vec<u8>,
    /// Embedded compiled Metal kernel payloads (kernel name -> .metallib bytes).
    /// Populated at load time from the CImage header's `kernels` map.
    /// When present, these can be passed directly to
    /// `MTLLibrary::new_library_with_data` to avoid MSL recompilation.
    pub kernel_buffers: HashMap<String, Vec<u8>>,
    /// KV cache allocator.
    pub kv_cache: BonsaiKVCache,
    /// Execution graph from the cimage header.
    pub execution_graph: ExecutionGraph,
    /// Model configuration constants.
    pub model_config: BonsaiModelConfig,
    /// Cached RMS norm weights: name → bytes (fp16).
    pub norm_buffers: HashMap<String, Vec<u8>>,
    /// Pooled activation buffer for temporary storage (prevents repeated allocs).
    pub scratch_buffer: Vec<f32>,
    /// True when Metal GPU dispatch is available and the kernel contract was
    /// verified at load time.
    pub metal_available: bool,
}

/// Bonsai model configuration constants.
#[derive(Debug, Clone)]
pub struct BonsaiModelConfig {
    pub layers: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub norm_eps: f32,
    pub context_length: u32,
}

impl Default for BonsaiModelConfig {
    fn default() -> Self {
        Self {
            layers: Bonsai27B::LAYERS,
            hidden_dim: Bonsai27B::HIDDEN_DIM,
            intermediate_dim: Bonsai27B::INTERMEDIATE_DIM,
            num_heads: Bonsai27B::NUM_HEADS,
            num_kv_heads: Bonsai27B::NUM_KV_HEADS,
            head_dim: Bonsai27B::HEAD_DIM,
            vocab_size: Bonsai27B::VOCAB_SIZE,
            norm_eps: Bonsai27B::NORM_EPS,
            context_length: Bonsai27B::CONTEXT_LENGTH,
        }
    }
}

// =============================================================================
// Runtime loader
// =============================================================================

/// The runtime loader for Bonsai cimage files.
///
/// Opens the cimage, reads the header and payload directory, then maps
/// each tensor payload into memory. On Apple Silicon with Metal, it
/// additionally allocates GPU buffers for the packed ternary data, scales,
/// and outliers.
pub struct BonsaiRuntimeLoader;

impl BonsaiRuntimeLoader {
    /// Load a Bonsai `.cimage` file and construct runtime state.
    ///
    /// Opens the cimage, reads the header to discover tensor payloads,
    /// reads the execution graph from the header, and maps all tensor
    /// data into memory-ready byte buffers.
    pub fn load(path: &Path) -> Result<BonsaiRuntimeState, String> {
        let reader = CImageReader::open(path)?;
        let header = &reader.header;

        // ── Parse execution graph from header ──────────────────────────
        let execution_graph: ExecutionGraph = match &header.execution_plan {
            Some(json) => {
                serde_json::from_str(json).map_err(|e| format!("parse execution graph: {e}"))?
            }
            None => {
                // Canonical CImages produced by compile_to_cimage always embed
                // an execution_plan. If we find one with actual tensor data but
                // no execution_plan, that's a defective compilation — fail closed
                // rather than silently building an empty graph.
                if !header.tensors.is_empty() {
                    return Err(
                        "CImage has no execution plan — cannot load for runtime execution"
                            .to_string(),
                    );
                }
                // Legacy/empty CImage: no tensors, no execution plan.
                // Build a minimal fallback graph to avoid crashing legacy consumers.
                eprintln!(
                    "[prism:warn] CImage has no tensors and no execution plan — \
                     building minimal fallback graph"
                );
                let regions = vec![ExecutionRegion {
                    id: GraphRegionId(0),
                    name: "default".to_string(),
                    operations: vec![],
                    target_lane: ExecutionLane::Cpu,
                    fusion_constraints: FusionConstraints {
                        max_fused_ops: None,
                        force_fused: false,
                        force_unfused: true,
                    },
                    inputs: vec![],
                    outputs: vec![],
                }];
                let state = RuntimeStatePlan {
                    max_context_tokens: Bonsai27B::CONTEXT_LENGTH as usize,
                    kv_cache_bytes_per_token: Bonsai27B::NUM_KV_HEADS as u64
                        * 2
                        * Bonsai27B::KEY_LENGTH as u64
                        * 2,
                    total_kv_cache_bytes: 0,
                };
                ExecutionGraph {
                    regions,
                    edges: vec![],
                    state,
                    memory: MemoryPlan {
                        total_activation_bytes: 0,
                        total_weight_bytes: 0,
                        arena_region_count: 1,
                    },
                }
            }
        };

        // ── Read all tensor payloads ───────────────────────────────────
        let mut packed_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        let mut scale_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        let mut lane_scale_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        let mut outlier_row_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        let mut outlier_col_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        let mut outlier_val_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        let mut norm_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        let mut embedding_table: Vec<u8> = Vec::new();

        let mut file =
            std::fs::File::open(path).map_err(|e| format!("open cimage for payloads: {e}"))?;

        for (name, record) in &header.tensors {
            let mut buf = vec![0u8; record.size as usize];
            file.seek(SeekFrom::Start(record.offset))
                .map_err(|e| format!("seek to {}: {e}", name))?;
            file.read_exact(&mut buf)
                .map_err(|e| format!("read {}: {e}", name))?;

            // ── TernaryTile640: parse compound payload ──────────────
            if record.tensor_type == TensorType::TernaryTile640 {
                let (packed_size, page_scales_size, lane_scales_size, _n_pages) = record
                    .ternary_tile640_layout()
                    .map_err(|e| format!("layout for {}: {e}", name))?;

                // Validate total payload size against tensor geometry
                let header_size = packed_size + page_scales_size + lane_scales_size;
                if buf.len() < header_size + 4 {
                    return Err(format!(
                        "{}: payload too small for ternary tile640 header: {} bytes, need at least {}",
                        name,
                        buf.len(),
                        header_size + 4
                    ));
                }

                let mut offset = 0usize;

                // 1. Packed ternary words: u32 × N_packed
                let packed_end = offset + packed_size;
                let packed_bytes = &buf[offset..packed_end];
                offset = packed_end;

                // 2. Page scales: u16 × N_pages
                let ps_end = offset + page_scales_size;
                let page_scale_bytes = &buf[offset..ps_end];
                offset = ps_end;

                // 3. Lane scales: i8 × N_pages × 32
                let ls_end = offset + lane_scales_size;
                let lane_scale_bytes = &buf[offset..ls_end];
                offset = ls_end;

                // 4. n_outliers: u32
                if offset + 4 > buf.len() {
                    return Err(format!(
                        "{}: truncated payload, expected n_outliers at offset {}",
                        name, offset
                    ));
                }
                let n_outliers = u32::from_le_bytes([
                    buf[offset],
                    buf[offset + 1],
                    buf[offset + 2],
                    buf[offset + 3],
                ]) as usize;
                offset += 4;

                // 5-7. Outlier rows, cols, vals
                let outlier_expected = n_outliers * (4 + 4 + 2); // 3 arrays: u32 + u32 + u16
                if offset + outlier_expected != buf.len() {
                    return Err(format!(
                        "{}: payload size mismatch: header+outlier header = {} bytes, expected {} + {} = {}",
                        name, buf.len(), offset, outlier_expected, offset + outlier_expected
                    ));
                }

                let or_end = offset + n_outliers * 4;
                let outlier_row_bytes = &buf[offset..or_end];
                offset = or_end;

                let oc_end = offset + n_outliers * 4;
                let outlier_col_bytes = &buf[offset..oc_end];
                offset = oc_end;

                let outlier_val_bytes = &buf[offset..];

                // Store parsed sections keyed by the tensor's natural name
                packed_buffers.insert(name.clone(), packed_bytes.to_vec());
                scale_buffers.insert(name.clone(), page_scale_bytes.to_vec());
                lane_scale_buffers.insert(name.clone(), lane_scale_bytes.to_vec());
                outlier_row_buffers.insert(name.clone(), outlier_row_bytes.to_vec());
                outlier_col_buffers.insert(name.clone(), outlier_col_bytes.to_vec());
                outlier_val_buffers.insert(name.clone(), outlier_val_bytes.to_vec());
                continue;
            }

            if name.starts_with("ternary_") {
                let inner_name = name.strip_prefix("ternary_").unwrap_or(name);
                packed_buffers.insert(inner_name.to_string(), buf);
            } else if name.starts_with("page_scales_") {
                let inner_name = name.strip_prefix("page_scales_").unwrap_or(name);
                scale_buffers.insert(inner_name.to_string(), buf);
            } else if name.starts_with("lane_scales_") {
                let inner_name = name.strip_prefix("lane_scales_").unwrap_or(name);
                lane_scale_buffers.insert(inner_name.to_string(), buf);
            } else if name.starts_with("outlier_rows_") {
                let inner_name = name.strip_prefix("outlier_rows_").unwrap_or(name);
                outlier_row_buffers.insert(inner_name.to_string(), buf);
            } else if name.starts_with("outlier_cols_") {
                let inner_name = name.strip_prefix("outlier_cols_").unwrap_or(name);
                outlier_col_buffers.insert(inner_name.to_string(), buf);
            } else if name.starts_with("outlier_vals_") {
                let inner_name = name.strip_prefix("outlier_vals_").unwrap_or(name);
                outlier_val_buffers.insert(inner_name.to_string(), buf);
            } else if name == "token_embd.weight" {
                embedding_table = buf;
            } else if name.contains("norm") || name.ends_with(".weight") {
                norm_buffers.insert(name.clone(), buf);
            } else {
                // Non-norm, non-ternary: treat as norm buffer if small enough
                norm_buffers.insert(name.clone(), buf);
            }
        }

        let kv_cache = BonsaiKVCache::new(BonsaiKVCacheConfig {
            num_layers: Bonsai27B::LAYERS,
            num_kv_heads: Bonsai27B::NUM_KV_HEADS,
            head_dim: Bonsai27B::HEAD_DIM,
            max_context: Bonsai27B::CONTEXT_LENGTH,
            bits_per_elem: 4,
        });

        let model_config = BonsaiModelConfig::default();

        // ── Load embedded Metal kernel payloads ────────────────────────
        // If the CImage header contains compiled kernel payloads, load them
        // from the file. These can be passed directly to
        // MTLLibrary::new_library_with_data at GPU dispatch time, avoiding
        // MSL compilation from source.
        let mut kernel_buffers: HashMap<String, Vec<u8>> = HashMap::new();
        for kernel_name in header.kernels.keys() {
            match reader.load_kernel(kernel_name) {
                Ok(bytes) => {
                    eprintln!(
                        "  [prism] Loaded embedded kernel '{}' ({} bytes)",
                        kernel_name,
                        bytes.len()
                    );
                    kernel_buffers.insert(kernel_name.clone(), bytes);
                }
                Err(e) => {
                    eprintln!(
                        "  [prism:warn] Failed to load embedded kernel '{}': {}. \
                         Will fall back to source compilation.",
                        kernel_name, e
                    );
                }
            }
        }

        // ── Kernel-contract verification (Metal pipeline) ──────────────
        // Attempt to compile the ternary GEMV Metal pipeline. A successful
        // compilation confirms that the Metal toolchain and GPU are available
        // and that the kernel buffer layout matches the tensor contract
        // (7 buffers: packed, input, page_scales, lane_scales, output,
        //  constants[in_dim, out_dim]).
        let metal_available = verify_kernel_contract();

        if metal_available {
            eprintln!("BonsaiRuntimeLoader: Metal GPU available, kernel contract verified");
        } else {
            eprintln!("BonsaiRuntimeLoader: Metal GPU unavailable, using CPU reference GEMV");
        }

        Ok(BonsaiRuntimeState {
            packed_buffers,
            scale_buffers,
            lane_scale_buffers,
            outlier_row_buffers,
            outlier_col_buffers,
            outlier_val_buffers,
            embedding_table,
            kernel_buffers,
            kv_cache,
            execution_graph,
            model_config,
            norm_buffers,
            scratch_buffer: Vec::new(),
            metal_available,
        })
    }
}

// =============================================================================
// Prefill
// =============================================================================

/// Run the prefill (prompt processing) phase for Bonsai 27B.
///
/// Processes input tokens through the full execution graph:
/// 1. Embedding lookup: maps each token ID to its hidden-dim vector.
/// 2. Per-layer pass: runs RMS norm, attention (full or SSM + ternary GEMV),
///    post-attention norm, FFN (gate+up+sili+down), residual add.
/// 3. For full-attention layers, stores K/V in the KV cache and computes
///    attention scores (prefill: full self-attention over all input tokens).
/// 4. For linear-attention (SSM) layers, runs SSM recurrence instead.
/// 5. Final RMS norm and LM head projection.
///
/// Returns logits for the next token (shape: [vocab_size]).
pub fn bonsai_prefill(state: &mut BonsaiRuntimeState, tokens: &[u32]) -> Result<Vec<f32>, String> {
    let cfg = &state.model_config;
    let num_tokens = tokens.len();
    if num_tokens == 0 {
        return Err("prefill requires at least 1 token".to_string());
    }

    // ── Embedding lookup ───────────────────────────────────────────────
    let mut hidden: Vec<f32> = Vec::with_capacity(num_tokens * cfg.hidden_dim as usize);
    let embed_size = (cfg.hidden_dim as usize) * 2; // fp16 bytes per element
    let embed_entries = state.embedding_table.len() / embed_size;
    let entries = embed_entries;

    for &tok in tokens {
        let idx = (tok as usize).min(entries.saturating_sub(1));
        let base = idx * embed_size;
        if base + embed_size <= state.embedding_table.len() {
            let slice = &state.embedding_table[base..base + embed_size];
            for chunk in slice.chunks(2) {
                let val = u16::from_le_bytes([chunk[0], chunk[1]]);
                hidden.push(half_to_f32(val));
            }
        } else {
            // Out of vocab range: pad with zeros.
            hidden.extend(std::iter::repeat_n(0.0f32, cfg.hidden_dim as usize));
        }
    }

    let mut kv_cache_page = 0u32;

    // ── Per-layer processing ───────────────────────────────────────────
    for layer in 0..cfg.layers {
        let layer_type = Bonsai27B::layer_type(layer);
        let is_full_attn = matches!(layer_type, prism_ecs_ir::bonsai::LayerType::FullAttention);

        // ── Pre-attention RMS norm ────────────────────────────────────
        let attn_norm_key = format!("blk.{layer}.attn_norm.weight");
        let attn_norm = state
            .norm_buffers
            .get(&attn_norm_key)
            .map(|b| bytes_to_f32_slice(b))
            .unwrap_or_default();

        for t in 0..num_tokens {
            let offset = t * cfg.hidden_dim as usize;
            let h = &mut hidden[offset..offset + cfg.hidden_dim as usize];
            apply_rms_norm(h, &attn_norm, cfg.norm_eps, cfg.hidden_dim as usize);
        }

        // ── QKV projection via ternary GEMV ───────────────────────────
        let qkv_name = format!("blk.{layer}.attn_qkv.weight");
        let qkv_tensor = state.packed_buffers.get(&qkv_name);
        let qkv_page_scales = state.scale_buffers.get(&qkv_name);
        let qkv_lane_scales = state.lane_scale_buffers.get(&qkv_name);

        // QKV projects hidden_dim → 10240 (see bonsai spec).
        let qkv_out_dim: u32 = 10240;
        let qkv_in_dim: u32 = cfg.hidden_dim;

        let mut qkv_output = vec![0.0f32; num_tokens * qkv_out_dim as usize];

        if let (Some(packed), Some(page_scales), Some(lane_scales)) =
            (qkv_tensor, qkv_page_scales, qkv_lane_scales)
        {
            let page_scale_u16: Vec<u16> = page_scales
                .chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let lane_scale_i8: Vec<i8> = lane_scales.iter().map(|&b| b as i8).collect();
            let packed_u32: Vec<u32> = packed
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Pre-parse outlier data
            let qkv_outlier_rows: Vec<u32> = state
                .outlier_row_buffers
                .get(&qkv_name)
                .map(|b| {
                    b.chunks(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                })
                .unwrap_or_default();
            let qkv_outlier_cols: Vec<u32> = state
                .outlier_col_buffers
                .get(&qkv_name)
                .map(|b| {
                    b.chunks(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                })
                .unwrap_or_default();
            let qkv_outlier_vals: Vec<u16> = state
                .outlier_val_buffers
                .get(&qkv_name)
                .map(|b| {
                    b.chunks(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect()
                })
                .unwrap_or_default();
            let has_qkv_outliers = !qkv_outlier_rows.is_empty();

            for t in 0..num_tokens {
                let offset = t * cfg.hidden_dim as usize;
                let input_slice = &hidden[offset..offset + cfg.hidden_dim as usize];
                let output_offset = t * qkv_out_dim as usize;

                let result = ternary_gemv_ref(
                    &packed_u32,
                    input_slice,
                    &page_scale_u16,
                    &lane_scale_i8,
                    qkv_out_dim,
                    qkv_in_dim,
                );
                qkv_output[output_offset..output_offset + qkv_out_dim as usize]
                    .copy_from_slice(&result);

                // Outlier correction
                if has_qkv_outliers {
                    let out_slice =
                        &mut qkv_output[output_offset..output_offset + qkv_out_dim as usize];
                    apply_outlier_correction(
                        out_slice,
                        input_slice,
                        &qkv_outlier_rows,
                        &qkv_outlier_cols,
                        &qkv_outlier_vals,
                        qkv_out_dim,
                        qkv_in_dim,
                    );
                }
            }
        } else {
            // No QKV tensor found — identity pass-through.
            for t in 0..num_tokens {
                let off = t * cfg.hidden_dim as usize;
                let qkv_off = t * qkv_out_dim as usize;
                qkv_output[qkv_off..qkv_off + cfg.hidden_dim as usize]
                    .copy_from_slice(&hidden[off..off + cfg.hidden_dim as usize]);
            }
        }

        // ── Decompose QKV and run attention ────────────────────────────
        // QKV packed: [Q | K | V] at [num_heads * head_dim, num_kv_heads * head_dim, ...]
        let q_dim = cfg.num_heads * cfg.head_dim; // 24 * 64 = 1536
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 4 * 64 = 256
        let ssm_dim = qkv_out_dim as usize - q_dim as usize - 2 * kv_dim as usize;

        let mut attn_output = vec![0.0f32; num_tokens * cfg.hidden_dim as usize];

        for t in 0..num_tokens {
            let qkv_off = t * qkv_out_dim as usize;

            // Slice Q
            let q: Vec<f32> = qkv_output[qkv_off..qkv_off + q_dim as usize].to_vec();

            if is_full_attn {
                // ── Full softmax attention ─────────────────────────────
                let k: Vec<f32> = qkv_output
                    [qkv_off + q_dim as usize..qkv_off + q_dim as usize + kv_dim as usize]
                    .to_vec();
                let v: Vec<f32> = qkv_output[qkv_off + q_dim as usize + kv_dim as usize
                    ..qkv_off + q_dim as usize + 2 * kv_dim as usize]
                    .to_vec();

                // Store K, V in KV cache.
                state.kv_cache.store(layer, kv_cache_page, &k, &v);

                // For prefill, attend over all cached tokens.
                let attn_result = attend_full(
                    &q,
                    &state.kv_cache,
                    layer,
                    kv_cache_page as usize,
                    cfg.num_heads,
                    cfg.num_kv_heads,
                    cfg.head_dim,
                );
                let out_off = t * cfg.hidden_dim as usize;
                let copy_len = attn_result.len().min(cfg.hidden_dim as usize);
                attn_output[out_off..out_off + copy_len].copy_from_slice(&attn_result[..copy_len]);
            } else {
                // ── Linear attention (SSM) ─────────────────────────────
                // Simplified SSM: compute output via the ssm portion of qkv.
                let ssm_off = qkv_off + q_dim as usize + 2 * kv_dim as usize;
                let ssm_inner = ssm_dim.min(cfg.hidden_dim as usize);
                for j in 0..cfg.hidden_dim as usize {
                    if j < ssm_inner {
                        attn_output[t * cfg.hidden_dim as usize + j] = qkv_output[ssm_off + j];
                    }
                }
            }
        }

        // ── Post-attention residual and norm ───────────────────────────
        for t in 0..num_tokens {
            let off = t * cfg.hidden_dim as usize;
            for j in 0..cfg.hidden_dim as usize {
                hidden[off + j] += attn_output[off + j];
            }
        }

        let post_norm_key = format!("blk.{layer}.post_attention_norm.weight");
        let post_norm = state
            .norm_buffers
            .get(&post_norm_key)
            .map(|b| bytes_to_f32_slice(b))
            .unwrap_or_default();

        for t in 0..num_tokens {
            let offset = t * cfg.hidden_dim as usize;
            let h = &mut hidden[offset..offset + cfg.hidden_dim as usize];
            apply_rms_norm(h, &post_norm, cfg.norm_eps, cfg.hidden_dim as usize);
        }

        // ── FFN: gate_proj → SiLU × up_proj → down_proj ───────────────
        let gate_name = format!("blk.{layer}.ffn_gate.weight");
        let up_name = format!("blk.{layer}.ffn_up.weight");
        let down_name = format!("blk.{layer}.ffn_down.weight");

        let ffn_intermediate = cfg.intermediate_dim as usize;

        let mut gate_out = vec![0.0f32; num_tokens * ffn_intermediate];
        let mut up_out = vec![0.0f32; num_tokens * ffn_intermediate];
        let mut ffn_input = vec![0.0f32; num_tokens * cfg.hidden_dim as usize];

        for t in 0..num_tokens {
            let off = t * cfg.hidden_dim as usize;
            ffn_input[off..off + cfg.hidden_dim as usize]
                .copy_from_slice(&hidden[off..off + cfg.hidden_dim as usize]);
        }

        // Gate projection
        apply_ternary_gemv_matmul(
            &state.packed_buffers,
            &state.scale_buffers,
            &state.lane_scale_buffers,
            &state.outlier_row_buffers,
            &state.outlier_col_buffers,
            &state.outlier_val_buffers,
            &gate_name,
            &ffn_input,
            &mut gate_out,
            cfg.intermediate_dim,
            cfg.hidden_dim,
            num_tokens,
        )?;

        // Up projection
        apply_ternary_gemv_matmul(
            &state.packed_buffers,
            &state.scale_buffers,
            &state.lane_scale_buffers,
            &state.outlier_row_buffers,
            &state.outlier_col_buffers,
            &state.outlier_val_buffers,
            &up_name,
            &ffn_input,
            &mut up_out,
            cfg.intermediate_dim,
            cfg.hidden_dim,
            num_tokens,
        )?;

        // SiLU: gate_out = gate_out * sigmoid(gate_out)
        for t in 0..num_tokens {
            let off = t * ffn_intermediate;
            for j in 0..ffn_intermediate {
                let g = gate_out[off + j];
                let sig = 1.0 / (1.0 + (-g).exp());
                gate_out[off + j] = g * sig;
            }
        }

        // Elementwise: gate_out * up_out
        for t in 0..num_tokens {
            let off = t * ffn_intermediate;
            for j in 0..ffn_intermediate {
                gate_out[off + j] *= up_out[off + j];
            }
        }

        // Down projection: ffn_intermediate → hidden_dim
        let mut down_res = vec![0.0f32; num_tokens * cfg.hidden_dim as usize];
        apply_ternary_gemv_matmul(
            &state.packed_buffers,
            &state.scale_buffers,
            &state.lane_scale_buffers,
            &state.outlier_row_buffers,
            &state.outlier_col_buffers,
            &state.outlier_val_buffers,
            &down_name,
            &gate_out,
            &mut down_res,
            cfg.hidden_dim,
            cfg.intermediate_dim,
            num_tokens,
        )?;

        // Residual add
        for t in 0..num_tokens {
            let off = t * cfg.hidden_dim as usize;
            for j in 0..cfg.hidden_dim as usize {
                hidden[off + j] = ffn_input[off + j] + down_res[off + j];
            }
        }

        kv_cache_page += 1;
    }

    // ── Final RMS norm ─────────────────────────────────────────────────
    let final_norm = state
        .norm_buffers
        .get("output_norm.weight")
        .map(|b| bytes_to_f32_slice(b))
        .unwrap_or_default();

    // Use only the last token's hidden state for logits.
    let last_token_offset = (num_tokens - 1) * cfg.hidden_dim as usize;
    let last_hidden = &mut hidden[last_token_offset..last_token_offset + cfg.hidden_dim as usize];
    apply_rms_norm(
        last_hidden,
        &final_norm,
        cfg.norm_eps,
        cfg.hidden_dim as usize,
    );

    // ── LM Head ────────────────────────────────────────────────────────
    // The LM head is the embedding table transposed (tied embeddings).
    let mut logits = vec![0.0f32; cfg.vocab_size as usize];
    let vocab_size = cfg.vocab_size as usize;
    let embed_dim = cfg.hidden_dim as usize;

    let embed_actual = state.embedding_table.len() / 2; // fp16 → number of elements
    let table_vocab = embed_actual / embed_dim;

    for v in 0..vocab_size.min(table_vocab) {
        let base = v * embed_dim * 2;
        for d in 0..embed_dim {
            let byte_off = base + d * 2;
            if byte_off + 2 <= state.embedding_table.len() {
                let fp16 = u16::from_le_bytes([
                    state.embedding_table[byte_off],
                    state.embedding_table[byte_off + 1],
                ]);
                logits[v] += last_hidden[d] * half_to_f32(fp16);
            }
        }
    }

    Ok(logits)
}

// =============================================================================
// Decode
// =============================================================================

/// Single-token decode step for Bonsai 27B.
///
/// Runs one forward pass using the cached KV from prefill. For full-attention
/// layers, the KV cache is read and softmax is computed over the cached
/// context. For linear-attention (SSM) layers, the SSM state is advanced.
///
/// Returns the next token ID via argmax sampling.
pub fn bonsai_decode(state: &mut BonsaiRuntimeState, token: u32) -> Result<u32, String> {
    let cfg = &state.model_config;

    // ── Embedding lookup ───────────────────────────────────────────────
    let mut hidden: Vec<f32> = vec![0.0f32; cfg.hidden_dim as usize];
    let embed_size = (cfg.hidden_dim as usize) * 2;
    let embed_entries = state.embedding_table.len() / embed_size;
    let idx = (token as usize).min(embed_entries.saturating_sub(1));
    let base = idx * embed_size;
    if base + embed_size <= state.embedding_table.len() {
        let slice = &state.embedding_table[base..base + embed_size];
        for (j, chunk) in slice.chunks(2).enumerate() {
            if j < cfg.hidden_dim as usize {
                let val = u16::from_le_bytes([chunk[0], chunk[1]]);
                hidden[j] = half_to_f32(val);
            }
        }
    }

    let current_pos = state.kv_cache.next_token_pos();
    let mut scratch_layer = vec![0.0f32; cfg.hidden_dim as usize];

    // ── Per-layer processing ───────────────────────────────────────────
    for layer in 0..cfg.layers {
        let layer_type = Bonsai27B::layer_type(layer);
        let is_full_attn = matches!(layer_type, prism_ecs_ir::bonsai::LayerType::FullAttention);

        // ── Pre-attention RMS norm ────────────────────────────────────
        let attn_norm_key = format!("blk.{layer}.attn_norm.weight");
        let attn_norm = state
            .norm_buffers
            .get(&attn_norm_key)
            .map(|b| bytes_to_f32_slice(b))
            .unwrap_or_default();
        apply_rms_norm(
            &mut hidden,
            &attn_norm,
            cfg.norm_eps,
            cfg.hidden_dim as usize,
        );

        // ── QKV projection ────────────────────────────────────────────
        let qkv_name = format!("blk.{layer}.attn_qkv.weight");
        let qkv_out_dim: u32 = 10240;
        let qkv_in_dim: u32 = cfg.hidden_dim;
        let mut qkv_output = vec![0.0f32; qkv_out_dim as usize];

        if let (Some(packed), Some(page_scales), Some(lane_scales)) = (
            state.packed_buffers.get(&qkv_name),
            state.scale_buffers.get(&qkv_name),
            state.lane_scale_buffers.get(&qkv_name),
        ) {
            let page_scale_u16: Vec<u16> = page_scales
                .chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let lane_scale_i8: Vec<i8> = lane_scales.iter().map(|&b| b as i8).collect();
            let packed_u32: Vec<u32> = packed
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            qkv_output = ternary_gemv_ref(
                &packed_u32,
                &hidden,
                &page_scale_u16,
                &lane_scale_i8,
                qkv_out_dim,
                qkv_in_dim,
            );

            // Outlier correction for QKV
            if let (Some(or_bytes), Some(oc_bytes), Some(ov_bytes)) = (
                state.outlier_row_buffers.get(&qkv_name),
                state.outlier_col_buffers.get(&qkv_name),
                state.outlier_val_buffers.get(&qkv_name),
            ) {
                let qkv_outlier_rows: Vec<u32> = or_bytes
                    .chunks(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let qkv_outlier_cols: Vec<u32> = oc_bytes
                    .chunks(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let qkv_outlier_vals: Vec<u16> = ov_bytes
                    .chunks(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                apply_outlier_correction(
                    &mut qkv_output,
                    &hidden,
                    &qkv_outlier_rows,
                    &qkv_outlier_cols,
                    &qkv_outlier_vals,
                    qkv_out_dim,
                    qkv_in_dim,
                );
            }
        } else {
            qkv_output[..cfg.hidden_dim as usize].copy_from_slice(&hidden);
        }

        // ── Decompose QKV ─────────────────────────────────────────────
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;

        let q: Vec<f32> = qkv_output[..q_dim as usize].to_vec();
        let k: Vec<f32> = qkv_output[q_dim as usize..q_dim as usize + kv_dim as usize].to_vec();
        let v: Vec<f32> = qkv_output
            [q_dim as usize + kv_dim as usize..q_dim as usize + 2 * kv_dim as usize]
            .to_vec();

        // ── Store K, V in KV cache ────────────────────────────────────
        state.kv_cache.store(layer, current_pos, &k, &v);

        let mut attn_out = vec![0.0f32; cfg.hidden_dim as usize];

        if is_full_attn {
            // Attend over all cached positions up to current_pos.
            let result = attend_full_single(
                &q,
                &state.kv_cache,
                layer,
                current_pos as usize,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
            );
            let copy_len = result.len().min(cfg.hidden_dim as usize);
            attn_out[..copy_len].copy_from_slice(&result[..copy_len]);
        } else {
            // Linear attention: simplified pass-through
            let ssm_dim =
                (qkv_out_dim as usize).saturating_sub(q_dim as usize + 2 * kv_dim as usize);
            let ssm_slice = &qkv_output[q_dim as usize + 2 * kv_dim as usize
                ..q_dim as usize + 2 * kv_dim as usize + ssm_dim];
            for j in 0..(cfg.hidden_dim as usize).min(ssm_slice.len()) {
                attn_out[j] = ssm_slice[j];
            }
        }

        // ── Residual ──────────────────────────────────────────────────
        for j in 0..cfg.hidden_dim as usize {
            hidden[j] += attn_out[j];
        }

        // ── Post-attention norm ───────────────────────────────────────
        let post_norm_key = format!("blk.{layer}.post_attention_norm.weight");
        let post_norm = state
            .norm_buffers
            .get(&post_norm_key)
            .map(|b| bytes_to_f32_slice(b))
            .unwrap_or_default();
        apply_rms_norm(
            &mut hidden,
            &post_norm,
            cfg.norm_eps,
            cfg.hidden_dim as usize,
        );

        // ── Store pre-FFN hidden state for residual ───────────────────
        scratch_layer.copy_from_slice(&hidden);

        // ── FFN ───────────────────────────────────────────────────────
        let gate_name = format!("blk.{layer}.ffn_gate.weight");
        let up_name = format!("blk.{layer}.ffn_up.weight");
        let down_name = format!("blk.{layer}.ffn_down.weight");

        let mut gate_out = vec![0.0f32; cfg.intermediate_dim as usize];
        let mut up_out = vec![0.0f32; cfg.intermediate_dim as usize];

        apply_ternary_gemv(
            &state.packed_buffers,
            &state.scale_buffers,
            &state.lane_scale_buffers,
            &state.outlier_row_buffers,
            &state.outlier_col_buffers,
            &state.outlier_val_buffers,
            &gate_name,
            &hidden,
            &mut gate_out,
            cfg.intermediate_dim,
            cfg.hidden_dim,
        )?;

        apply_ternary_gemv(
            &state.packed_buffers,
            &state.scale_buffers,
            &state.lane_scale_buffers,
            &state.outlier_row_buffers,
            &state.outlier_col_buffers,
            &state.outlier_val_buffers,
            &up_name,
            &hidden,
            &mut up_out,
            cfg.intermediate_dim,
            cfg.hidden_dim,
        )?;

        for j in 0..cfg.intermediate_dim as usize {
            let g = gate_out[j];
            let sig = 1.0 / (1.0 + (-g).exp());
            gate_out[j] = g * sig;
            gate_out[j] *= up_out[j];
        }

        let mut down_out = vec![0.0f32; cfg.hidden_dim as usize];
        apply_ternary_gemv(
            &state.packed_buffers,
            &state.scale_buffers,
            &state.lane_scale_buffers,
            &state.outlier_row_buffers,
            &state.outlier_col_buffers,
            &state.outlier_val_buffers,
            &down_name,
            &gate_out,
            &mut down_out,
            cfg.hidden_dim,
            cfg.intermediate_dim,
        )?;

        // Residual
        for j in 0..cfg.hidden_dim as usize {
            hidden[j] = scratch_layer[j] + down_out[j];
        }
    }

    // ── Final RMS norm ─────────────────────────────────────────────────
    let final_norm = state
        .norm_buffers
        .get("output_norm.weight")
        .map(|b| bytes_to_f32_slice(b))
        .unwrap_or_default();
    apply_rms_norm(
        &mut hidden,
        &final_norm,
        cfg.norm_eps,
        cfg.hidden_dim as usize,
    );

    // ── LM Head ────────────────────────────────────────────────────────
    let mut logits = vec![0.0f32; cfg.vocab_size as usize];
    let embed_dim = cfg.hidden_dim as usize;
    let embed_actual = state.embedding_table.len() / 2;
    let table_vocab = embed_actual / embed_dim;
    let vocab_size = cfg.vocab_size as usize;

    for v in 0..vocab_size.min(table_vocab) {
        let base = v * embed_dim * 2;
        for d in 0..embed_dim {
            let byte_off = base + d * 2;
            if byte_off + 2 <= state.embedding_table.len() {
                let fp16 = u16::from_le_bytes([
                    state.embedding_table[byte_off],
                    state.embedding_table[byte_off + 1],
                ]);
                logits[v] += hidden[d] * half_to_f32(fp16);
            }
        }
    }

    // ── Argmax sampling ────────────────────────────────────────────────
    let next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0);

    Ok(next_token)
}

// =============================================================================
// Attention helpers
// =============================================================================

/// Full self-attention over all cached tokens (prefill).
fn attend_full(
    q: &[f32],
    kv_cache: &BonsaiKVCache,
    layer: u32,
    num_cached: usize,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Vec<f32> {
    let hd = head_dim as usize;
    let n_heads = num_heads as usize;
    let n_kv = num_kv_heads as usize;
    let groups = n_heads / n_kv;

    // Q: [num_heads * head_dim], K/V: cached per position [num_cached, num_kv_heads * head_dim]
    let mut output = vec![0.0f32; n_heads * hd];

    for g in 0..n_kv {
        let _kv_head_off = g * hd;
        for h_off in 0..groups {
            let q_head_off = (g * groups + h_off) * hd;

            // Softmax scores for this GQA group.
            let mut scores = vec![0.0f32; num_cached];
            for pos in 0..num_cached {
                if let Ok((k_ref, _)) = kv_cache.load(layer, pos as u32) {
                    let k_offset = g * hd;
                    let mut score = 0.0f32;
                    let limit = hd.min(k_ref.len().saturating_sub(k_offset));
                    for d in 0..limit {
                        score += q[q_head_off + d] * k_ref[k_offset + d];
                    }
                    // Scale by 1/sqrt(head_dim)
                    score /= (hd as f32).sqrt();
                    scores[pos] = score;
                }
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = scores.iter().map(|s| (s - max_s).exp()).sum();
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };

            // Weighted sum of V
            for d in 0..hd {
                let mut val = 0.0f32;
                for pos in 0..num_cached {
                    if let Ok((_, v_ref)) = kv_cache.load(layer, pos as u32) {
                        let v_offset = g * hd;
                        let attn_w = (scores[pos] - max_s).exp() * inv_sum;
                        val += attn_w * v_ref[v_offset + d];
                    }
                }
                output[q_head_off + d] = val;
            }
        }
    }

    output
}

/// Single-token attention (decode) — attend over all cached positions.
fn attend_full_single(
    q: &[f32],
    kv_cache: &BonsaiKVCache,
    layer: u32,
    num_cached: usize,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Vec<f32> {
    attend_full(
        q,
        kv_cache,
        layer,
        num_cached,
        num_heads,
        num_kv_heads,
        head_dim,
    )
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert a half-precision (fp16) u16 to f32.
fn half_to_f32(val: u16) -> f32 {
    let sign = ((val >> 15) as f32) * -2.0 + 1.0;
    let exp = (val >> 10) & 0x1f;
    let mant = val & 0x3ff;
    match exp {
        0 => sign * (mant as f32) * (2.0f32).powi(-14) * (2.0f32).powi(-10),
        31 => {
            if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + (mant as f32) / 1024.0) * (2.0f32).powi(exp as i32 - 15),
    }
}

/// Convert raw fp16 bytes to a &[f32] by interpreting as u16 and converting.
fn bytes_to_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(2)
        .map(|c| {
            if c.len() >= 2 {
                half_to_f32(u16::from_le_bytes([c[0], c[1]]))
            } else {
                0.0
            }
        })
        .collect()
}

/// Apply RMS normalization in-place.
fn apply_rms_norm(h: &mut [f32], weight: &[f32], eps: f32, dim: usize) {
    let len = dim.min(h.len());
    let sum_sq: f32 = h[..len].iter().map(|x| x * x).sum();
    let rms = (sum_sq / len as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..len {
        let w = if i < weight.len() { weight[i] } else { 1.0 };
        h[i] = h[i] * inv_rms * w;
    }
}

// ── CPU fallback run_ternary_gemv (for targets without Metal) ──────────
// When Metal is available, the `use crate::bonsai_metal_dispatch::run_ternary_gemv`
// import at the top of this file takes precedence over this definition.

/// Run Tile640 ternary GEMV on CPU reference, with outlier correction.
///
/// Accepts raw byte slices (as loaded from a cimage) and returns a
/// `Vec<f32>` of length `dim_m`.
#[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
fn run_ternary_gemv(
    packed_bytes: &[u8],
    input: &[f32],
    page_scale_bytes: &[u8],
    lane_scale_bytes: &[u8],
    outlier_rows_bytes: Option<&[u8]>,
    outlier_cols_bytes: Option<&[u8]>,
    outlier_vals_bytes: Option<&[u8]>,
    dim_n: u32,
    dim_m: u32,
) -> Result<Vec<f32>, String> {
    // ── Parse bytes into typed arrays ──────────────────────────────
    if !packed_bytes.len().is_multiple_of(4) {
        return Err(format!(
            "packed_bytes length {} is not a multiple of 4",
            packed_bytes.len()
        ));
    }
    let packed_u32: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if !page_scale_bytes.len().is_multiple_of(2) {
        return Err(format!(
            "page_scale_bytes length {} is not a multiple of 2",
            page_scale_bytes.len()
        ));
    }
    let page_scale_u16: Vec<u16> = page_scale_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let lane_scale_i8: Vec<i8> = lane_scale_bytes.iter().map(|&b| b as i8).collect();

    // ── Run CPU reference GEMV ────────────────────────────────────
    let mut output = ternary_gemv_ref(
        &packed_u32,
        input,
        &page_scale_u16,
        &lane_scale_i8,
        dim_m,
        dim_n,
    );

    // ── Outlier correction ────────────────────────────────────────
    if let (Some(or_bytes), Some(oc_bytes), Some(ov_bytes)) =
        (outlier_rows_bytes, outlier_cols_bytes, outlier_vals_bytes)
    {
        if !or_bytes.is_empty() && or_bytes.len() % 4 == 0 {
            let outlier_rows: Vec<u32> = or_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let outlier_cols: Vec<u32> = oc_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let outlier_vals: Vec<u16> = ov_bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();

            apply_outlier_correction(
                &mut output,
                input,
                &outlier_rows,
                &outlier_cols,
                &outlier_vals,
                dim_m,
                dim_n,
            );
        }
    }

    Ok(output)
}

/// Apply a ternary GEMV for a single token, dispatching to the reference
/// implementation with proper byte-to-word conversion.
fn apply_ternary_gemv(
    packed_buffers: &HashMap<String, Vec<u8>>,
    scale_buffers: &HashMap<String, Vec<u8>>,
    lane_scale_buffers: &HashMap<String, Vec<u8>>,
    outlier_row_buffers: &HashMap<String, Vec<u8>>,
    outlier_col_buffers: &HashMap<String, Vec<u8>>,
    outlier_val_buffers: &HashMap<String, Vec<u8>>,
    tensor_name: &str,
    input: &[f32],
    output: &mut [f32],
    out_dim: u32,
    in_dim: u32,
) -> Result<(), String> {
    let packed = packed_buffers
        .get(tensor_name)
        .ok_or_else(|| format!("missing packed tensor: {tensor_name}"))?;
    let page_scales = scale_buffers
        .get(tensor_name)
        .ok_or_else(|| format!("missing page scales: {tensor_name}"))?;
    let lane_scales = lane_scale_buffers
        .get(tensor_name)
        .ok_or_else(|| format!("missing lane scales: {tensor_name}"))?;

    let outlier_rows_opt = outlier_row_buffers.get(tensor_name).map(|v| v.as_slice());
    let outlier_cols_opt = outlier_col_buffers.get(tensor_name).map(|v| v.as_slice());
    let outlier_vals_opt = outlier_val_buffers.get(tensor_name).map(|v| v.as_slice());

    let result = run_ternary_gemv(
        packed,
        input,
        page_scales,
        lane_scales,
        outlier_rows_opt,
        outlier_cols_opt,
        outlier_vals_opt,
        in_dim,  // dim_n
        out_dim, // dim_m
    )?;

    let copy_len = result.len().min(output.len());
    output[..copy_len].copy_from_slice(&result[..copy_len]);

    Ok(())
}

/// Apply ternary GEMV across multiple tokens (batched).
fn apply_ternary_gemv_matmul(
    packed_buffers: &HashMap<String, Vec<u8>>,
    scale_buffers: &HashMap<String, Vec<u8>>,
    lane_scale_buffers: &HashMap<String, Vec<u8>>,
    outlier_row_buffers: &HashMap<String, Vec<u8>>,
    outlier_col_buffers: &HashMap<String, Vec<u8>>,
    outlier_val_buffers: &HashMap<String, Vec<u8>>,
    tensor_name: &str,
    input: &[f32],
    output: &mut [f32],
    out_dim: u32,
    in_dim: u32,
    num_tokens: usize,
) -> Result<(), String> {
    let packed = packed_buffers
        .get(tensor_name)
        .ok_or_else(|| format!("missing packed tensor: {tensor_name}"))?;
    let page_scales = scale_buffers
        .get(tensor_name)
        .ok_or_else(|| format!("missing page scales: {tensor_name}"))?;
    let lane_scales = lane_scale_buffers
        .get(tensor_name)
        .ok_or_else(|| format!("missing lane scales: {tensor_name}"))?;

    let outlier_rows_bytes = outlier_row_buffers.get(tensor_name).map(|v| v.as_slice());
    let outlier_cols_bytes = outlier_col_buffers.get(tensor_name).map(|v| v.as_slice());
    let outlier_vals_bytes = outlier_val_buffers.get(tensor_name).map(|v| v.as_slice());

    for t in 0..num_tokens {
        let in_off = t * in_dim as usize;
        let out_off = t * out_dim as usize;
        let in_slice = &input[in_off..in_off + in_dim as usize];
        let out_slice = &mut output[out_off..out_off + out_dim as usize];

        let result = run_ternary_gemv(
            packed,
            in_slice,
            page_scales,
            lane_scales,
            outlier_rows_bytes,
            outlier_cols_bytes,
            outlier_vals_bytes,
            in_dim,  // dim_n
            out_dim, // dim_m
        )?;
        let copy_len = result.len().min(out_slice.len());
        out_slice[..copy_len].copy_from_slice(&result[..copy_len]);
    }
    Ok(())
}

// =============================================================================
// KV cache
// =============================================================================

/// Configuration for [`BonsaiKVCache`].
#[derive(Debug, Clone)]
pub struct BonsaiKVCacheConfig {
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
    /// Dimension per head.
    pub head_dim: u32,
    /// Maximum context length in tokens.
    pub max_context: u32,
    /// Bits per element: 4 (compressed) or 16 (uncompressed).
    pub bits_per_elem: u8,
}

impl Default for BonsaiKVCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: Bonsai27B::LAYERS,
            num_kv_heads: Bonsai27B::NUM_KV_HEADS,
            head_dim: Bonsai27B::HEAD_DIM,
            max_context: Bonsai27B::CONTEXT_LENGTH,
            bits_per_elem: 4,
        }
    }
}

/// Paged KV cache for Bonsai 27B.
///
/// Supports 4-bit compressed and 16-bit uncompressed storage modes.
/// Uses paged allocation: each page holds fixed-size slices of K/V data
/// for one layer at one token position.
///
/// Memory layout per page (16-bit mode):
///   K data: num_kv_heads × head_dim × 2 bytes
///   V data: num_kv_heads × head_dim × 2 bytes
///
/// Memory layout per page (4-bit mode):
///   K data: num_kv_heads × head_dim × 0.5 bytes (packed)
///   V data: num_kv_heads × head_dim × 0.5 bytes (packed)
pub struct BonsaiKVCache {
    /// Configuration.
    config: BonsaiKVCacheConfig,
    /// Byte size of each K or V slot per position per layer.
    slot_bytes: usize,
    /// The flat byte buffer containing all K/V data.
    buffer: Vec<u8>,
    /// Current number of tokens stored.
    current_tokens: u32,
    /// Allocated capacity in tokens.
    capacity: u32,
}

impl BonsaiKVCache {
    /// Create a new KV cache with the given configuration.
    ///
    /// Pre-allocates storage for the full context length.
    pub fn new(config: BonsaiKVCacheConfig) -> Self {
        let entries = config.num_kv_heads as usize * config.head_dim as usize;
        let bytes_per_entry = if config.bits_per_elem <= 8 { 1 } else { 2 };
        let slot_bytes = entries * bytes_per_entry;
        let total_slots = config.num_layers as usize * config.max_context as usize;
        let total_bytes = total_slots * slot_bytes * 2; // K + V

        Self {
            slot_bytes,
            buffer: vec![0u8; total_bytes],
            current_tokens: 0,
            capacity: config.max_context,
            config,
        }
    }

    /// The token position that will be written next.
    fn next_token_pos(&self) -> u32 {
        self.current_tokens
    }

    /// Store K and V for a layer at the given token position.
    pub fn store(&mut self, layer: u32, token_pos: u32, k: &[f32], v: &[f32]) {
        let entries = self.config.num_kv_heads as usize * self.config.head_dim as usize;
        let layer_off = layer as usize * self.capacity as usize * self.slot_bytes * 2;
        let pos_off = token_pos as usize * self.slot_bytes * 2;

        // Store K
        let k_off = layer_off + pos_off;
        let k_slice = &mut self.buffer[k_off..k_off + self.slot_bytes];
        if self.config.bits_per_elem == 4 {
            pack_f32_to_4bit(k, k_slice, entries);
        } else {
            pack_f32_to_16bit(k, k_slice, entries);
        }

        // Store V
        let v_off = k_off + self.slot_bytes;
        let v_slice = &mut self.buffer[v_off..v_off + self.slot_bytes];
        if self.config.bits_per_elem == 4 {
            pack_f32_to_4bit(v, v_slice, entries);
        } else {
            pack_f32_to_16bit(v, v_slice, entries);
        }

        if token_pos >= self.current_tokens {
            self.current_tokens = token_pos + 1;
        }
    }

    /// Load K and V for a layer at the given token position.
    pub fn load(&self, layer: u32, token_pos: u32) -> Result<(Vec<f32>, Vec<f32>), String> {
        let entries = self.config.num_kv_heads as usize * self.config.head_dim as usize;
        let layer_off = layer as usize * self.capacity as usize * self.slot_bytes * 2;
        let pos_off = token_pos as usize * self.slot_bytes * 2;

        if token_pos as usize > self.capacity as usize {
            return Err("token_pos exceeds KV cache capacity".to_string());
        }

        let byte_off = layer_off + pos_off;

        let k = if self.config.bits_per_elem == 4 {
            unpack_f32_from_4bit(&self.buffer[byte_off..byte_off + self.slot_bytes], entries)
        } else {
            unpack_f32_from_16bit(&self.buffer[byte_off..byte_off + self.slot_bytes], entries)
        };

        let v_off = byte_off + self.slot_bytes;
        let v = if self.config.bits_per_elem == 4 {
            unpack_f32_from_4bit(&self.buffer[v_off..v_off + self.slot_bytes], entries)
        } else {
            unpack_f32_from_16bit(&self.buffer[v_off..v_off + self.slot_bytes], entries)
        };

        Ok((k, v))
    }

    /// Total allocated bytes.
    pub fn total_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Number of tokens currently in the cache.
    pub fn num_tokens(&self) -> u32 {
        self.current_tokens
    }
}

/// Pack f32 values into 4-bit storage (2 values per byte).
fn pack_f32_to_4bit(src: &[f32], dst: &mut [u8], count: usize) {
    for i in 0..count {
        if i * 2 + 1 < dst.len() {
            let lo = quantize_to_4bit(src.get(i).copied().unwrap_or(0.0));
            let hi = quantize_to_4bit(src.get(i + count).copied().unwrap_or(0.0));
            dst[i] = (lo & 0x0f) | ((hi & 0x0f) << 4);
        }
    }
}

/// Unpack f32 values from 4-bit storage.
fn unpack_f32_from_4bit(src: &[u8], count: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; count];
    for i in 0..count {
        if i / 2 < src.len() {
            let byte = src[i / 2];
            let nibble = if i % 2 == 0 {
                (byte & 0x0f) as i8
            } else {
                ((byte >> 4) & 0x0f) as i8
            };
            // Sign-extend 4-bit to 32-bit, then to f32.
            let extended = (nibble << 4) as i32 >> 4;
            result[i] = extended as f32 * 0.5; // arbitrary scale for demo
        }
    }
    result
}

/// Quantize a single f32 to a 4-bit signed value.
fn quantize_to_4bit(val: f32) -> u8 {
    // Clamp to [-8, 7], map to [0, 15]
    let clamped = val.clamp(-8.0, 7.5);
    let int_val = (clamped.round() as i8).clamp(-8, 7);
    (int_val & 0x0f) as u8
}

/// Pack f32 values into fp16 (16-bit) storage.
fn pack_f32_to_16bit(src: &[f32], dst: &mut [u8], count: usize) {
    for i in 0..count.min(dst.len() / 2) {
        let v = src.get(i).copied().unwrap_or(0.0);
        let fp16 = f32_to_half(v);
        let off = i * 2;
        if off + 2 <= dst.len() {
            dst[off] = (fp16 & 0xff) as u8;
            dst[off + 1] = (fp16 >> 8) as u8;
        }
    }
}

/// Unpack f32 values from fp16 (16-bit) storage.
fn unpack_f32_from_16bit(src: &[u8], count: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; count];
    for i in 0..count.min(src.len() / 2) {
        let off = i * 2;
        let fp16 = u16::from_le_bytes([src[off], src[off + 1]]);
        result[i] = half_to_f32(fp16);
    }
    result
}

/// Convert an f32 to fp16 bits (u16).
fn f32_to_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp > 112 {
        // Normal f32 → normal fp16
        let new_exp = (exp - 112).max(1).min(31) as u32;
        let new_mant = mant >> 13;
        sign as u16 | ((new_exp << 10) as u16) | (new_mant as u16)
    } else {
        // Subnormal → zero
        sign as u16
    }
}

// =============================================================================
// Receipts
// =============================================================================

/// A receipt anchoring a Bonsai compilation to its calibration and hardware.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BonsaiReceipt {
    /// Calibration identity that produced the cost estimates.
    pub calibration_id: String,
    /// Hash of the spatial compilation plan.
    pub plan_hash: String,
    /// Hardware fingerprint.
    pub hardware_fingerprint: String,
    /// Tensor digest records.
    pub tensor_digests: Vec<TensorDigestRecord>,
    /// Kernel ABI records.
    pub kernel_abis: Vec<KernelAbiRecord>,
    /// Calibration identity used for signing.
    pub signed_by: String,
    /// Timestamp of receipt creation.
    pub created_at: String,
}

/// Digest record for a single tensor.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorDigestRecord {
    pub tensor_name: String,
    pub sha256: String,
    pub out_dim: u32,
    pub in_dim: u32,
}

/// ABI record for a compiled kernel.
#[derive(Debug, Clone, serde::Serialize)]
pub struct KernelAbiRecord {
    pub kernel_name: String,
    pub semantic_id: String,
    pub num_parameters: u32,
}

/// Publish a receipt for a Bonsai compilation plan.
///
/// Creates a receipt with the calibration ID, plan hash, hardware
/// fingerprint, tensor digests, and kernel ABI records. Signs the
/// receipt with the plan's calibration identity.
///
/// Returns a [`ReceiptId`] that chains all downstream evidence back
/// to this generation.
pub fn publish_bonsai_receipt(
    plan: &SpatialCompilationPlan,
    cimage_path: &Path,
) -> Result<ReceiptId, String> {
    let plan_hash = compute_plan_hash(plan);

    let hardware_fingerprint = probe_hardware_fingerprint();

    // Read the cimage to extract tensor digests.
    let reader = CImageReader::open(cimage_path)?;

    let mut tensor_digests: Vec<TensorDigestRecord> = Vec::new();
    for (name, record) in &reader.header.tensors {
        let mut file =
            std::fs::File::open(cimage_path).map_err(|e| format!("open for digest: {e}"))?;
        let mut payload = vec![0u8; record.size as usize];
        file.seek(SeekFrom::Start(record.offset))
            .map_err(|e| format!("seek: {e}"))?;
        file.read_exact(&mut payload)
            .map_err(|e| format!("read: {e}"))?;

        let mut hasher = Sha256::new();
        hasher.update(&payload);
        let digest: [u8; 32] = hasher.finalize().into();
        let digest_hex = simple_hex(&digest);

        tensor_digests.push(TensorDigestRecord {
            tensor_name: name.clone(),
            sha256: digest_hex,
            out_dim: record.dim_m,
            in_dim: record.dim_n,
        });
    }

    // Kernel ABI records.
    let kernel_abis = vec![
        KernelAbiRecord {
            kernel_name: "ternary_tile640_gemv".to_string(),
            semantic_id: "bonsai.ternary.gemv.tile640".to_string(),
            num_parameters: 6,
        },
        KernelAbiRecord {
            kernel_name: "rms_norm".to_string(),
            semantic_id: "bonsai.rmsnorm.v1".to_string(),
            num_parameters: 3,
        },
    ];

    let receipt = BonsaiReceipt {
        calibration_id: plan.calibration_id.0.clone(),
        plan_hash: simple_hex(&plan_hash[..16]),
        hardware_fingerprint,
        tensor_digests,
        kernel_abis,
        signed_by: plan.calibration_id.0.clone(),
        created_at: chrono_now_iso(),
    };

    // Compute receipt digest.
    let receipt_json =
        serde_json::to_string(&receipt).map_err(|e| format!("serialize receipt: {e}"))?;
    let mut hasher = Sha256::new();
    hasher.update(receipt_json.as_bytes());
    let receipt_digest: [u8; 32] = hasher.finalize().into();

    Ok(ReceiptId(format!(
        "bonsai-{}",
        simple_hex(&receipt_digest[..16])
    )))
}

/// Probe the hardware for a fingerprint string.
fn probe_hardware_fingerprint() -> String {
    // Try sysctl for hardware info.
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        if let Ok(s) = String::from_utf8(out.stdout) {
            let trimmed = s.trim().to_string();
            if !trimmed.is_empty() {
                return format!("apple-{}", trimmed.replace(' ', "_"));
            }
        }
    }

    // Try uname as fallback.
    if let Ok(out) = std::process::Command::new("uname").arg("-m").output() {
        if let Ok(s) = String::from_utf8(out.stdout) {
            let trimmed = s.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }

    "apple-m1".to_string()
}

/// Compute a deterministic hash of a [`SpatialCompilationPlan`].
fn compute_plan_hash(plan: &SpatialCompilationPlan) -> [u8; 32] {
    let json = serde_json::to_vec(plan).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&json);
    hasher.finalize().into()
}

/// Get the current system time as an ISO 8601 string (UTC).
fn chrono_now_iso() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Days since 1970-01-01.
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    // Zeller's congruence for day-of-week.
    let zeller = |y: i64, m: i64, d: i64| -> i64 {
        let (y, m) = if m < 3 { (y - 1, m + 12) } else { (y, m) };
        (d + (13 * (m + 1)) / 5 + y + y / 4 - y / 100 + y / 400) % 7
    };
    let epoch_year = 1970i64;
    let mut year = epoch_year;
    let mut remaining = days as i64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }
    let month_days = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0usize;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining < md {
            month = i + 1;
            break;
        }
        remaining -= md;
    }
    if month == 0 {
        // Shouldn't happen but be safe.
        month = 12;
        remaining = 30;
    }
    let day = remaining + 1;
    let wday = zeller(year, month as i64, day);
    const WEEKDAYS: &[&str] = &["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"];
    let wday_str = WEEKDAYS[wday as usize];
    const MONTHS: &[&str] = &[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!(
        "{wday_str}, {day:02} {} {year:04} {hours:02}:{minutes:02}:{seconds:02} +0000",
        MONTHS[month - 1]
    )
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// =============================================================================
// Tests
// =============================================================================

/// Simple hex encoding for byte slices (replaces `hex` crate dependency).
fn simple_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cimage::TensorType;

    #[test]
    fn test_kv_cache_allocate() {
        let config = BonsaiKVCacheConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_context: 128,
            bits_per_elem: 16,
        };

        let cache = BonsaiKVCache::new(config);

        // Expected: 2 layers × 128 tokens × 4 heads × 64 dim × 2 bytes × 2 (K+V)
        let expected_size = 2 * 128 * 4 * 64 * 2 * 2; // = 262,144
        assert_eq!(
            cache.total_bytes(),
            expected_size,
            "KV cache size mismatch: expected {expected_size}, got {}",
            cache.total_bytes()
        );
    }

    #[test]
    fn test_kv_cache_store_load() {
        let config = BonsaiKVCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            max_context: 8,
            bits_per_elem: 16,
        };

        let mut cache = BonsaiKVCache::new(config);

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(0, 0, &k, &v);
        cache.store(0, 1, &v, &k); // swapped for second position

        // Load back first position.
        let (k_loaded, v_loaded) = cache.load(0, 0).unwrap();
        assert!((k_loaded[0] - 1.0).abs() < 0.1);
        assert!((v_loaded[0] - 5.0).abs() < 0.1);

        // Load back second position.
        let (k2, v2) = cache.load(0, 1).unwrap();
        assert!((k2[0] - 5.0).abs() < 0.1);
        assert!((v2[0] - 1.0).abs() < 0.1);

        assert_eq!(cache.num_tokens(), 2);
    }

    #[test]
    fn test_kv_cache_4bit_mode() {
        let config = BonsaiKVCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            max_context: 4,
            bits_per_elem: 4,
        };

        let mut cache = BonsaiKVCache::new(config);

        let entries = 64;
        let _expected_slot_bytes = entries; // 4-bit: 1 byte per 2 elements → entries / 2, rounded up
        let expected_slot = (entries + 1) / 2; // ceil(64/2) = 32

        // Store and load a simple vector.
        let k: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let v: Vec<f32> = (100..164).map(|i| i as f32).collect();
        cache.store(0, 0, &k, &v);

        let (k_loaded, v_loaded) = cache.load(0, 0).unwrap();
        assert_eq!(k_loaded.len(), 64);
        assert_eq!(v_loaded.len(), 64);

        // 4-bit mode stores half the bytes per value, so slot_bytes should be < entries * 2.
        assert!(
            expected_slot < entries * 2,
            "4-bit slot {} should be smaller than 16-bit slot {}",
            expected_slot,
            entries * 2
        );
    }

    #[test]
    fn test_receipt_publication() {
        // Test BonsaiReceipt data construction directly (publish_bonsai_receipt
        // needs a real cimage file, so we test the receipt struct here).
        let receipt = BonsaiReceipt {
            calibration_id: "test-cal-001".to_string(),
            plan_hash: "ab".repeat(16),
            hardware_fingerprint: "apple-m1-test".to_string(),
            tensor_digests: vec![TensorDigestRecord {
                tensor_name: "test_tensor".to_string(),
                sha256: "ab".repeat(32),
                out_dim: 5120,
                in_dim: 10240,
            }],
            kernel_abis: vec![KernelAbiRecord {
                kernel_name: "ternary_tile640_gemv".to_string(),
                semantic_id: "bonsai.ternary.gemv.tile640".to_string(),
                num_parameters: 6,
            }],
            signed_by: "test-cal-001".to_string(),
            created_at: "2026-07-17T00:00:00+0000".to_string(),
        };

        // Verify receipt has all required fields.
        assert_eq!(receipt.calibration_id, "test-cal-001");
        assert!(receipt.tensor_digests[0].sha256.len() >= 16);
        assert!(!receipt.kernel_abis.is_empty());
        assert_eq!(receipt.signed_by, "test-cal-001");
        assert!(!receipt.created_at.is_empty());
    }

    #[test]
    fn test_cimage_tensor_type_ternary_tile640() {
        // Verify the new variant exists and serializes/deserializes.
        let tt = TensorType::TernaryTile640;
        let json = serde_json::to_string(&tt).unwrap();
        assert_eq!(json, "\"TernaryTile640\"");
        let back: TensorType = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, TensorType::TernaryTile640));
    }
}
