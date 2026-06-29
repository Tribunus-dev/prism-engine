//! KV cache compaction — StreamingLLM-style heuristic position selection.
//!
//! When compressing a long KV cache (e.g. 1M tokens → 20K) for ternary
//! packing, we must decide which token positions to keep.  Instead of
//! running an expensive attention-based importance model (which would
//! require an ANE / GPU pass over the entire prefill), we use the
//! empirically proven StreamingLLM heuristic:
//!
//! 1. **Attention sinks** (first N\_SINKS positions) — always kept.
//!    Early tokens absorb out-of-distribution attention mass and are
//!    critical for stable generation.
//! 2. **Recent context** (last N\_RECENT positions) — always kept.
//!    Local coherence depends on the most recent tokens.
//! 3. **Middle region** — uniformly sampled to fill the remaining budget.
//!
//! Reference: Xiao et al., "Efficient Streaming Language Models with
//! Attention Sinks" (https://arxiv.org/abs/2309.17453).

use half::f16;

// ── Default constants ────────────────────────────────────────────

/// Number of initial token positions treated as attention sinks.
const N_SINKS: usize = 4;

/// Number of most recent token positions always retained.
const N_RECENT: usize = 1024;

// ── Public API ───────────────────────────────────────────────────

/// Select which KV-cache positions to keep when compacting from
/// `seq_len` down to `target_count` positions.
///
/// Uses the StreamingLLM heuristic:
/// - Sink tokens   `[0, N_SINKS)`  unconditionally kept.
/// - Recent tokens `[seq_len - N_RECENT, seq_len)` unconditionally kept
///   (or fewer if `seq_len < N_SINKS + N_RECENT`).
/// - Middle tokens uniformly subsampled to reach `target_count`.
///
/// # Panics
///
/// Panics if `seq_len == 0` (callers with an empty prompt must handle
/// that before calling).
///
/// # Examples
///
/// ```ignore
/// // use prism::compute_image::compaction::select_compaction_positions;
/// let positions = select_compaction_positions(100, 20);
/// assert_eq!(positions.len(), 20);
/// // First 4 are sinks
/// assert_eq!(positions[..4], vec![0, 1, 2, 3]);
/// // Because N_RECENT = 1024 overflows seq_len=100, recent_start
/// // gets clamped to seq_len-sinks = 96; with target_count=20 and
/// // n_sinks=4, budget for recent = 16, so positions[4..] = [84..100).
/// ```
pub fn select_compaction_positions(seq_len: usize, target_count: usize) -> Vec<u32> {
    assert!(seq_len > 0, "seq_len must be > 0");

    // Fast path: nothing to compact, or target covers everything.
    if target_count >= seq_len {
        return (0..seq_len as u32).collect();
    }

    // Clamp sinks to available length.
    let n_sinks = N_SINKS.min(seq_len).min(target_count);
    let sink_end = n_sinks;

    // Remaining budget after reserving sinks.
    let remaining = target_count - n_sinks;

    // Keep the sink positions.
    let mut result: Vec<u32> = (0..sink_end as u32).collect();

    // Number of recent tokens we can afford.
    let n_recent = remaining
        .min(N_RECENT)
        .min(seq_len.saturating_sub(sink_end));
    let recent_start = seq_len.saturating_sub(n_recent);
    // Ensure recent doesn't overlap sinks.
    let recent_start = recent_start.max(sink_end);
    let n_recent_actual = seq_len - recent_start;

    // Update remaining after reserving recent.
    let remaining = remaining.saturating_sub(n_recent_actual);
    if remaining == 0 {
        // No budget for middle positions.
        result.extend(recent_start as u32..seq_len as u32);
        debug_assert!(result.len() == target_count);
        return result;
    }

    // Middle region: [sink_end, recent_start).
    let middle_len = recent_start - sink_end;

    if remaining >= middle_len {
        // Enough budget for all middle positions.
        result.extend(sink_end as u32..recent_start as u32);
    } else if remaining == 0 {
        // No budget for middle positions at all.
    } else {
        // Uniform stride through middle region.
        let step = (middle_len as f64) / (remaining as f64);
        let middle: Vec<u32> = (0..remaining)
            .map(|i| {
                let idx = sink_end + (i as f64 * step + 0.5).floor() as usize;
                idx.min(recent_start - 1) as u32
            })
            .collect();
        result.extend(middle);
    };

    // Add recent tokens.
    result.extend(recent_start as u32..seq_len as u32);

    result.truncate(target_count);
    result
}

// ── Entropy-guided compaction ────────────────────────────────────

/// Select positions to keep based on per-token entropy scores.
/// High-entropy tokens (uncertain predictions) are preserved;
/// low-entropy tokens (filler, boilerplate) are evicted first.
///
/// Always keeps: attention sinks (first N_SINKS) + recent window
/// (last N_RECENT).  Fills remaining budget with highest-entropy
/// positions from the middle.
pub fn select_entropy_compaction_positions(
    entropy_map: &[f16], // per-position entropy [0,1], len = seq_len
    target_count: usize, // e.g. 20480
) -> Vec<u32> {
    let seq_len = entropy_map.len();
    let mut positions = Vec::with_capacity(target_count);

    // 1. Always keep attention sinks
    for i in 0..N_SINKS.min(seq_len) {
        positions.push(i as u32);
    }

    if seq_len <= N_SINKS + N_RECENT {
        // Short sequence: keep everything
        for i in (N_SINKS as usize)..seq_len {
            positions.push(i as u32);
        }
        return positions;
    }

    let recent_start = seq_len.saturating_sub(N_RECENT);

    // 2. Collect middle positions with their entropy scores
    let middle_end = recent_start;

    // 3. Sort middle positions by entropy descending
    let mut middle_entropy: Vec<(u32, f32)> = (N_SINKS..middle_end)
        .map(|i| (i as u32, entropy_map[i].to_f32()))
        .collect();
    middle_entropy.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 4. Take highest-entropy positions up to budget
    let remaining_budget = target_count.saturating_sub(positions.len() + N_RECENT);
    let take_count = remaining_budget.min(middle_entropy.len());
    for i in 0..take_count {
        positions.push(middle_entropy[i].0);
    }

    // 5. Add recent window (always keep)
    for i in recent_start..seq_len {
        positions.push(i as u32);
    }

    // 6. Sort final positions for sequential gather (gather along axis=2)
    positions.sort();
    positions.truncate(target_count);

    positions
}

// ═══════════════════════════════════════════════════════════════════════════
// ANE Gather Model — MIL program generation
// ═══════════════════════════════════════════════════════════════════════════

/// Pad dimension to nearest multiple of 64 bytes worth of elements.
/// For FP16 (2 bytes/element): pad to mult of 32.
/// For Int32 (4 bytes/element): pad to mult of 16.
pub fn align_dim(dim: u32, element_bytes: u32) -> u32 {
    let align_bytes = 64u32;
    let bytes = dim * element_bytes;
    let padded = ((bytes + align_bytes - 1) / align_bytes) * align_bytes;
    padded / element_bytes
}

use crate::arena::Arena;
use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};

/// Default compaction target (50x at 1M tokens).
pub const DEFAULT_TARGET_COUNT: u32 = 20480;

/// Generate MIL text for the ANE KV compaction gather program.
///
/// The program reads KV cache from IOSurface inputs, selects survivors
/// by indices written by CPU, and writes compacted KV to IOSurface outputs.
///
/// n_kv_heads: number of KV heads (GQA)
/// head_dim:   dimension per KV head
/// max_seq_len: maximum KV cache length
/// target_count: number of positions to keep after compaction
pub fn generate_compaction_mil(
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    target_count: u32,
) -> String {
    let mut mil = String::new();

    mil.push_str("// ANE KV Compaction Gather\n");
    mil.push_str(&format!(
        "// n_kv_heads: {} head_dim: {} max_seq_len: {} target_count: {}\n\n",
        n_kv_heads, head_dim, max_seq_len, target_count
    ));

    // Inputs: KV cache (4D) and indices (1D Int32)
    mil.push_str(&format!(
        "input @ \"key_cache\" (float16, [1, {}, {}, {}]) read write\n",
        n_kv_heads, max_seq_len, head_dim
    ));
    mil.push_str(&format!(
        "input @ \"value_cache\" (float16, [1, {}, {}, {}]) read write\n",
        n_kv_heads, max_seq_len, head_dim
    ));
    mil.push_str(&format!(
        "input @ \"indices\" (int32, [{}]) read write\n",
        target_count
    ));

    // Outputs: compacted KV (4D, sequence dimension = target_count)
    mil.push_str(&format!(
        "output @ \"compacted_key\" (float16, [1, {}, {}, {}]) read write\n",
        n_kv_heads, target_count, head_dim
    ));
    mil.push_str(&format!(
        "output @ \"compacted_value\" (float16, [1, {}, {}, {}]) read write\n\n",
        n_kv_heads, target_count, head_dim
    ));

    // Gather operation for K: gather along sequence dimension (axis=2)
    mil.push_str(
        "layer gather_k {\n  type: gather\n  input: \"key_cache\"\n  input: \"indices\"\n  output: \"compacted_key\"\n  axis: 2\n}\n\n",
    );

    // Gather operation for V
    mil.push_str(
        "layer gather_v {\n  type: gather\n  input: \"value_cache\"\n  input: \"indices\"\n  output: \"compacted_value\"\n  axis: 2\n}\n",
    );

    mil
}

/// Compile the compaction MIL program and load as a Core ML model on ANE.
///
/// Writes MIL text to a temp .mlmodel file, loads via Core ML's model loader
/// with CpuAndNeuralEngine compute units, and returns the loaded model.
pub fn compile_compaction_model(
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    target_count: u32,
) -> Result<CoreMlModel, String> {
    let mil_text = generate_compaction_mil(n_kv_heads, head_dim, max_seq_len, target_count);
    compile_compaction_mil_inner(&mil_text)
}

/// Generate MIL text for the ANE KV compaction gather program (optimized layout).
///
/// Uses [B, C, 1, S] NCHW layout with 64-byte alignment and axis=3 gather.
///
/// n_kv_heads: number of KV heads (GQA)
/// head_dim:   dimension per KV head
/// max_seq_len: maximum KV cache length
/// target_count: number of positions to keep after compaction
pub fn generate_compaction_mil_optimized(
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    target_count: u32,
) -> String {
    let mut mil = String::new();

    // Compute aligned dimensions
    let c = align_dim(n_kv_heads * head_dim, 2); // channels = heads x dim, FP16
    let s_in = align_dim(max_seq_len, 2); // sequence = cache length, FP16
    let s_out = align_dim(target_count, 2); // survivors, FP16

    mil.push_str("program(1.3)\n");
    mil.push_str("[buildInfo = dict<string, string>({{");
    mil.push_str("{\"coremltools-version\", \"9.0\"}, ");
    mil.push_str("{\"coremlc-component-MIL\", \"3510.2.1\"}");
    mil.push_str("}})]\n");
    mil.push_str(&format!("void compaction_gather<ios18>(tensor<fp16, [1, {}, 1, {}]> key_cache, tensor<fp16, [1, {}, 1, {}]> value_cache, tensor<int32, [{}]> indices) {{\n", c, s_in, c, s_in, target_count));
    mil.push_str(&format!("  tensor<fp16, [1, {}, 1, {}]> compacted_key = gather(axis = 3L, x = key_cache, indices = indices)[name = string(\"gather_k\")];\n", c, s_out));
    mil.push_str(&format!("  tensor<fp16, [1, {}, 1, {}]> compacted_value = gather(axis = 3L, x = value_cache, indices = indices)[name = string(\"gather_v\")];\n", c, s_out));
    mil.push_str(&format!("}} -> (compacted_key, compacted_value);\n"));

    mil
}

/// Compile the optimized compaction MIL program and load as a Core ML model on ANE.
pub fn compile_compaction_model_optimized(
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    target_count: u32,
) -> Result<CoreMlModel, String> {
    let mil_text =
        generate_compaction_mil_optimized(n_kv_heads, head_dim, max_seq_len, target_count);
    compile_compaction_mil_inner(&mil_text)
}

/// Shared compile logic for compaction MIL programs.
fn compile_compaction_mil_inner(mil_text: &str) -> Result<CoreMlModel, String> {
    // Wrap MIL text in a .mlpackage and compile via coremlc, then load.
    // This supports the program(1.3) format that coremlcompiler accepts.
    let tag = format!("ane_compaction_{:x}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
    let tmp = std::env::temp_dir().join(&tag);
    let _ = std::fs::create_dir_all(&tmp);
    let _ = std::fs::create_dir_all(tmp.join("Data"));

    // Write MIL text to Data/model.mil
    std::fs::write(tmp.join("Data").join("model.mil"), mil_text)
        .map_err(|e| format!("write model.mil: {}", e))?;

    // Write minimal Manifest.json (needed for .mlpackage format)
    let manifest = serde_json::json!({
        "fileFormatVersion": "1.0.0",
        "specificationVersion": 9
    });
    std::fs::write(tmp.join("Manifest.json"), serde_json::to_string_pretty(&manifest).unwrap())
        .map_err(|e| format!("write Manifest.json: {}", e))?;

    let out_dir = tmp.join("compiled");
    let output = std::process::Command::new("xcrun")
        .args(["coremlc", "compile", &tmp.to_string_lossy(), &out_dir.to_string_lossy()])
        .output()
        .map_err(|e| format!("coremlc: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("coremlc compile failed: {}", stderr));
    }

    // Find .mlmodelc directory
    let modelc_name = format!("{}.modelc", tag);
    let modelc_dir = out_dir.join(&modelc_name);
    if !modelc_dir.exists() {
        return Err(format!(".mlmodelc not found at {:?}", modelc_dir));
    }

    let model = CoreMlModel::load_with_compute_units(
        &modelc_dir.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .map_err(|e| format!("load compaction model: {}", e))?;

    // Keep tmp dir alive (Core ML references it)
    let _ = std::mem::ManuallyDrop::new(tmp);

    Ok(model)
}

/// Run the ANE compaction gather on a full KV cache.
///
/// # Arguments
/// * `model` - The loaded compaction Core ML model
/// * `k_arena` - Arena containing full key cache [1, n_kv_heads, seq_len, head_dim] FP16
/// * `v_arena` - Arena containing full value cache [1, n_kv_heads, seq_len, head_dim] FP16
/// * `indices` - Survivor positions (must match the model's target_count)
/// * `indices_arena` - Arena for indices (Int32 data, written before predict)
/// * `compacted_k_arena` - Output arena for compacted key
/// * `compacted_v_arena` - Output arena for compacted value
pub fn run_compaction(
    model: &CoreMlModel,
    k_arena: &Arena,
    v_arena: &Arena,
    indices: &[u32],
    indices_arena: &Arena,
    compacted_k_arena: &mut Arena,
    compacted_v_arena: &mut Arena,
) -> Result<(), String> {
    // ── 1. Write indices to indices arena ─────────────────────────────
    {
        indices_arena.lock()?;
        let ptr = unsafe { indices_arena.base_ptr() as *mut u32 };
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr, indices.len()) };
        dst.copy_from_slice(indices);
        indices_arena.unlock()?;
    }

    // ── 2. Run multi-IO prediction ────────────────────────────────────
    model
        .predict_multi(
            &["key_cache", "value_cache", "indices"],
            &[&k_arena.info, &v_arena.info, &indices_arena.info],
            &["compacted_key", "compacted_value"],
            &mut [&mut compacted_k_arena.info, &mut compacted_v_arena.info],
        )
        .map_err(|e| format!("compaction predict: {}", e))?;

    Ok(())
}

/// Select positions with entropy-weighted stride.
///
/// High-entropy regions: stride=1 (keep all).
/// Low-entropy regions: stride scales inversely with entropy.
/// Falls back to [`select_entropy_compaction_positions`] if adaptive
/// selection yields fewer than half the target count.
pub fn select_entropy_adaptive_positions(entropy_map: &[f16], target_count: usize) -> Vec<u32> {
    let seq_len = entropy_map.len();
    let mut positions = Vec::with_capacity(target_count);

    // Always keep sinks
    for i in 0..N_SINKS.min(seq_len) {
        positions.push(i as u32);
    }

    if seq_len <= N_SINKS + N_RECENT {
        for i in (N_SINKS as usize)..seq_len {
            positions.push(i as u32);
        }
        return positions;
    }

    let recent_start = seq_len.saturating_sub(N_RECENT);

    // Adaptive stride: stride = max(1, (1.0 - entropy) * max_stride)
    // entropy ∈ [0,1], high entropy → small stride (keep more)
    const MAX_STRIDE: u32 = 64;
    const MIN_STRIDE: u32 = 1;

    for i in (N_SINKS as usize)..recent_start {
        let e = entropy_map[i].to_f32().clamp(0.0, 1.0);
        let stride = (MIN_STRIDE as f32 + (1.0 - e) * (MAX_STRIDE - MIN_STRIDE) as f32) as u32;
        if (i as u32 - N_SINKS as u32) % stride.max(1) == 0 {
            if positions.len() < target_count - N_RECENT {
                positions.push(i as u32);
            }
        }
    }

    // Recent window
    for i in recent_start..seq_len {
        positions.push(i as u32);
    }

    positions.truncate(target_count);

    // Fall back to entropy-sorted selection if we got too few
    if positions.len() < target_count / 2 {
        return select_entropy_compaction_positions(entropy_map, target_count);
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_sequence_all_kept() {
        // seq_len <= target_count → all positions.
        let pos = select_compaction_positions(10, 20);
        assert_eq!(pos, (0..10).collect::<Vec<u32>>());
        assert_eq!(pos.len(), 10);
    }

    #[test]
    fn test_exact_fit() {
        let pos = select_compaction_positions(10, 10);
        assert_eq!(pos, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn test_sinks_and_recent_only() {
        // Short enough that sinks + recent cover the whole sequence.
        let pos = select_compaction_positions(50, 20);
        // seq_len=50: sinks=4, recent=min(1024, 46)=46, but overlap since
        // recent_start = 50-46 = 4 which == sink_end=4 → no middle region.
        // Dedup gives all 50 positions (since we have more budget than seq_len,
        // but target_count < seq_len — wait, target_count=20 < seq_len=50.
        // So we must compact. But sinks(4) + recent(46) = 50 > target=20.
        // The function keeps what fits: sinks + recent trimmed to target.
        assert_eq!(pos.len(), 20);
        assert_eq!(pos[..4], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_large_sequence_middle_sampling() {
        // seq_len=100k, target_count=10k
        let seq_len = 100_000;
        let target = 10_000;
        let pos = select_compaction_positions(seq_len, target);

        // Should have exactly target_count positions.
        assert_eq!(pos.len(), target);

        // First N_SINKS are the sinks.
        for i in 0..N_SINKS {
            assert_eq!(pos[i], i as u32);
        }

        // Last N_RECENT (with possible truncation to fit target) are recent.
        let recent_start_idx = target.saturating_sub(N_RECENT).max(N_SINKS);
        // The recent portion should start at seq_len - N_RECENT.
        assert!(pos[recent_start_idx] >= (seq_len - N_RECENT) as u32);
    }

    #[test]
    fn test_sinks_always_present() {
        // Even with tiny target, sinks are kept.
        let pos = select_compaction_positions(100, 4);
        assert_eq!(pos.len(), 4);
        assert_eq!(pos, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_target_less_than_sinks() {
        let pos = select_compaction_positions(100, 2);
        assert_eq!(pos.len(), 2);
        assert_eq!(pos, vec![0, 1]);
    }

    #[test]
    fn test_no_duplicates() {
        let pos = select_compaction_positions(2000, 500);
        assert_eq!(pos.len(), 500);
        // All unique and sorted.
        let mut sorted = pos.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 500);
        assert_eq!(pos, sorted);
    }

    #[test]
    fn test_middle_stride_uniform() {
        // seq_len=2000, sinks=4, recent=1024
        // middle_len = 2000 - 4 - 1024 = 972
        // target=100 → sinks(4) + recent(96) → no middle budget
        // Let's test with target large enough for middle.
        let seq_len = 10_000;
        let target = 3_000;
        let pos = select_compaction_positions(seq_len, target);

        let sinks = &pos[..N_SINKS];
        assert_eq!(sinks, &[0, 1, 2, 3]);

        // Recent tokens should be at the end.
        let recent_start = target.saturating_sub(N_RECENT);
        for i in recent_start..target {
            assert!(pos[i] >= (seq_len - N_RECENT) as u32);
        }

        // Middle should be strictly between sink_end and recent_start.
        for i in N_SINKS..recent_start {
            let v = pos[i];
            assert!(v >= N_SINKS as u32);
            assert!(v < (seq_len - N_RECENT) as u32);
        }
    }

    #[test]
    fn test_single_token() {
        let pos = select_compaction_positions(1, 1);
        assert_eq!(pos, vec![0]);
    }

    #[test]
    fn test_roundtrip_midpoint() {
        // Verify that middle sampling produces a sequence where
        // the gap between sampled positions doesn't massively exceed
        // the expected stride.
        let seq_len = 1_000_000;
        let target = 20_000;
        let pos = select_compaction_positions(seq_len, target);

        assert_eq!(pos.len(), target);
        assert!(pos.windows(2).all(|w| w[0] < w[1]));

        // Average middle stride should be ~(1M - 1028) / (20K - 1028) ≈ 50
        let middle_start = N_SINKS;
        let middle_end = target.saturating_sub(N_RECENT);
        if middle_end > middle_start {
            let mid = &pos[middle_start..middle_end];
            if mid.len() >= 2 {
                let avg_gap = (mid.last().unwrap() - mid[0]) as f64 / (mid.len() - 1) as f64;
                // Average gap shouldn't deviate wildly from expected.
                let expected_gap = (seq_len - N_SINKS - N_RECENT) as f64 / mid.len() as f64;
                assert!(
                    (avg_gap / expected_gap - 1.0).abs() < 2.0,
                    "avg_gap={avg_gap}, expected_gap={expected_gap}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "seq_len must be > 0")]
    fn test_zero_seq_len_panics() {
        select_compaction_positions(0, 10);
    }
}

// ── Entropy-guided compaction tests ────────────────────────────

#[test]
fn test_entropy_short_sequence_all_kept() {
    let entropies = vec![f16::ZERO; 500];
    let pos = select_entropy_compaction_positions(&entropies, 1000);
    assert_eq!(pos.len(), 500);
    assert_eq!(pos, (0..500).collect::<Vec<u32>>());
}

#[test]
fn test_entropy_sinks_and_recent_only() {
    // seq_len large enough that sinks+recent fits within target budget, no middle.
    // seq_len=2000, target=1028 -> sinks(4)+recent(1024)=1028 fits exactly.
    let entropies = vec![f16::ZERO; 2000];
    let pos = select_entropy_compaction_positions(&entropies, 1028);
    assert_eq!(pos.len(), 1028);
    for i in 0..4 {
        assert_eq!(pos[i], i as u32);
    }
    for i in 4..1028 {
        assert!(pos[i] >= 976); // 2000-1024 = 976
    }
}

#[test]
fn test_entropy_high_entropy_middle_selected() {
    let seq_len = 5000;
    let target = 1500;
    let mut entropies = vec![f16::ZERO; seq_len];
    for i in 4..2000 {
        entropies[i] = f16::from_f32(0.9);
    }
    for i in 2000..5000 {
        entropies[i] = f16::from_f32(0.1);
    }
    let pos = select_entropy_compaction_positions(&entropies, target);
    assert_eq!(pos.len(), target);
    assert_eq!(pos[..4], vec![0, 1, 2, 3]);
    for i in 4..(target - 1024) {
        assert!(pos[i] < 2000, "pos[{}]={} should be < 2000", i, pos[i]);
    }
}

#[test]
fn test_entropy_sorted_output() {
    let mut entropies = vec![f16::ZERO; 10_000];
    for i in 0..10_000 {
        entropies[i] = f16::from_f32(((i % 100) as f32) / 100.0);
    }
    let pos = select_entropy_compaction_positions(&entropies, 2000);
    assert_eq!(pos.len(), 2000);
    let mut sorted = pos.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 2000, "duplicate positions in output");
    assert_eq!(pos, sorted, "output not sorted");
    assert_eq!(pos[..N_SINKS], vec![0, 1, 2, 3]);
}

#[test]
fn test_entropy_all_zero() {
    let entropies = vec![f16::ZERO; 10_000];
    let pos = select_entropy_compaction_positions(&entropies, 2000);
    assert_eq!(pos.len(), 2000);
    assert_eq!(pos[..N_SINKS], vec![0, 1, 2, 3]);
}

#[test]
fn test_optimized_mil_contains_axis3_and_b_c_1_s() {
    let mil = generate_compaction_mil_optimized(8, 512, 2048, 20480);
    assert!(mil.contains("axis = 3L"), "MIL should have axis=3 in program(1.3) format");
    // Verify [B, C, 1, S] layout — inputs start with [1,
    assert!(
        mil.contains("[1, "),
        "MIL should use [B,C,1,S] layout starting with [1,, got:\n{mil}"
    );
    assert!(!mil.contains("axis: 2"), "MIL should NOT use old axis=2");
    // 8*512=4096 FP16 -> 8192 bytes, already 64-byte aligned (128*64), so C=4096.
    let c = align_dim(8 * 512, 2);
    assert_eq!(c, 4096, "8*512=4096 FP16 -> 8192B, already 64B-aligned");
    // Verify that an odd-length dim DOES get padded (e.g. 100 FP16 elements = 200B, pads to 256B = 128 FP16)
    let padded = align_dim(100, 2);
    assert_eq!(
        padded, 128,
        "100 FP16 -> 200B, padded to 256B = 128 FP16 elements"
    );
    assert!(
        mil.contains(&format!("[1, {}, 1, ", c)),
        "MIL should use aligned C={}",
        c
    );
}
