#include <metal_stdlib>
using namespace metal;

// ── Struct definitions (must match kernel_types.rs #[repr(C)]) ──────────────

struct ProjectionParams {
    uint   in_dim;            // vocab_size (vocabulary dimension)
    uint   out_dim;           // num_heads (number of attention heads)
    uint   page_count;        // num_tokens (number of KV tokens to probe)
    uint   page_width;
    uint   mode_flags;
    uint   probe_seed;        // seed for deterministic (token,head) sampling
    uint   reserved[5];
};

struct AttentionProbe {
    uint   head_id;
    uint   token_index;
    float  teacher_max_logit;
    float  student_max_logit;
    float  teacher_entropy;
    float  student_entropy;
    float  sampled_probability_l1;
    float  sampled_probability_kl;
};

// ── Attention score probe ───────────────────────────────────────────────────
//
// For sampled (token, head) pairs, compute teacher/student softmax statistics
// over the full vocabulary.  Produces diagnostic records for analysing
// attention-spread divergence between the teacher and student models.
//
// buffer(0): teacher_logits  [num_tokens * num_heads * vocab_size] half
// buffer(1): student_logits  [num_tokens * num_heads * vocab_size] half
// buffer(2): output_probes   [num_tokens * num_heads]              AttentionProbe
// buffer(3): ProjectionParams  (page_count == num_tokens)
//
// Grid: one threadgroup (64 threads) per sampled (token, head) pair.
//   token = gid / num_heads,  head = gid % num_heads
//
// Entropy is derived mathematically from log(Z) - sum(exp * t_norm) / Z
// rather than a separate softmax pass, saving one full vocabulary load.
kernel void attention_score_probe(
    device const half*         teacher_logits  [[buffer(0)]],
    device const half*         student_logits  [[buffer(1)]],
    device AttentionProbe*     output_probes   [[buffer(2)]],
    constant ProjectionParams& params          [[buffer(3)]],
    uint gid                                   [[threadgroup_position_in_grid]],
    uint tid                                   [[thread_position_in_threadgroup]],
    uint simd_lane                             [[thread_index_in_simdgroup]],
    uint simd_id                               [[simdgroup_index_in_threadgroup]])
{
    const uint num_tokens  = params.page_count;
    const uint num_heads   = params.out_dim;
    const uint vocab_size  = params.in_dim;
    const uint total_pairs = num_tokens * num_heads;

    if (gid >= total_pairs) return;

    const uint token = gid / num_heads;
    const uint head  = gid % num_heads;

    // Defensive bound: token must be within [0, num_tokens).
    // (Redundant given total_pairs guard above, but protects against
    // host misconfiguration or grid alignment padding.)
    if (token >= num_tokens) return;

    // Base offset into flattened logit tensors: [token][head][vocab].
    const uint base_offset = (token * num_heads + head) * vocab_size;
    device const half* tbase = teacher_logits + base_offset;
    device const half* sbase = student_logits + base_offset;

    // ── Phase 1: find max logit (teacher + student in one loop) ───────────
    float t_max = -INFINITY;
    float s_max = -INFINITY;
    for (uint i = tid; i < vocab_size; i += 64) {
        float t = float(tbase[i]);
        float s = float(sbase[i]);
        t_max = max(t_max, t);
        s_max = max(s_max, s);
    }
    t_max = simd_max(t_max);
    s_max = simd_max(s_max);

    // Cross-SIMD reduction (64 threads → 2 SIMD groups of 32).
    threadgroup float tg_t_max[2];
    threadgroup float tg_s_max[2];
    if (simd_lane == 0) {
        tg_t_max[simd_id] = t_max;
        tg_s_max[simd_id] = s_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        tg_t_max[0] = max(tg_t_max[0], tg_t_max[1]);
        tg_s_max[0] = max(tg_s_max[0], tg_s_max[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    t_max = tg_t_max[0];
    s_max = tg_s_max[0];

    // ── Phase 2: sum exp + entropy intermediates ──────────────────────────
    // Accumulate:
    //   sum_exp_t, sum_exp_s        → softmax denominator Z
    //   sum_exp_t_tn, sum_exp_t     → entropy = log(Z) - sum(exp*t_norm)/Z
    //
    // Notation: tn = t - t_max,  sn = s - s_max  (shifted by phase-1 max).
    float sum_t = 0.0f;
    float sum_s = 0.0f;
    float sum_exp_t_tn = 0.0f;  // Σ exp(tn) * tn  for teacher
    float sum_exp_s_sn = 0.0f;  // Σ exp(sn) * sn  for student

    for (uint i = tid; i < vocab_size; i += 64) {
        float tn = float(tbase[i]) - t_max;
        float sn = float(sbase[i]) - s_max;
        float et = exp(tn);
        float es = exp(sn);
        sum_t += et;
        sum_s += es;
        sum_exp_t_tn += et * tn;
        sum_exp_s_sn += es * sn;
    }

    sum_t        = simd_sum(sum_t);
    sum_s        = simd_sum(sum_s);
    sum_exp_t_tn = simd_sum(sum_exp_t_tn);
    sum_exp_s_sn = simd_sum(sum_exp_s_sn);

    threadgroup float tg_sum_t[2];
    threadgroup float tg_sum_s[2];
    threadgroup float tg_sum_et_tn[2];
    threadgroup float tg_sum_es_sn[2];
    if (simd_lane == 0) {
        tg_sum_t[simd_id]     = sum_t;
        tg_sum_s[simd_id]     = sum_s;
        tg_sum_et_tn[simd_id] = sum_exp_t_tn;
        tg_sum_es_sn[simd_id] = sum_exp_s_sn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        tg_sum_t[0]     = tg_sum_t[0]     + tg_sum_t[1];
        tg_sum_s[0]     = tg_sum_s[0]     + tg_sum_s[1];
        tg_sum_et_tn[0] = tg_sum_et_tn[0] + tg_sum_et_tn[1];
        tg_sum_es_sn[0] = tg_sum_es_sn[0] + tg_sum_es_sn[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_t        = tg_sum_t[0];
    sum_s        = tg_sum_s[0];
    sum_exp_t_tn = tg_sum_et_tn[0];
    sum_exp_s_sn = tg_sum_es_sn[0];

    // Derive entropy without a second softmax pass.
    //   entropy = log(Z) - Σ(exp(tn) * tn) / Z
    //   where tn = t - t_max is the max-shifted logit.
    const float inv_sum_t = 1.0f / sum_t;
    const float inv_sum_s = 1.0f / sum_s;
    const float entropy_t = log(sum_t) - sum_exp_t_tn * inv_sum_t;
    const float entropy_s = log(sum_s) - sum_exp_s_sn * inv_sum_s;

    // ── Phase 3: L1 and KL divergence ─────────────────────────────────────
    float l1 = 0.0f;
    float kl = 0.0f;

    for (uint i = tid; i < vocab_size; i += 64) {
        float tn = float(tbase[i]) - t_max;
        float sn = float(sbase[i]) - s_max;
        float p_t = exp(tn) * inv_sum_t;
        float p_s = exp(sn) * inv_sum_s;

        // L1: Σ|p_t - p_s|
        l1 += abs(p_t - p_s);

        // KL: Σ p_t * log(p_t / p_s).
        // Lim_{p_t→0} p_t * log(p_t / p_s) = 0; skip zero p_t.
        // Use clamped p_s within KL to avoid log(0) → inf when student
        // assigns no probability (p_s underflows to zero).
        float p_s_clamped = max(p_s, 1e-10f);
        kl += (p_t > 0.0f) ? p_t * log(p_t / p_s_clamped) : 0.0f;
    }

    l1 = simd_sum(l1);
    kl = simd_sum(kl);

    threadgroup float tg_l1[2];
    threadgroup float tg_kl[2];
    if (simd_lane == 0) {
        tg_l1[simd_id] = l1;
        tg_kl[simd_id] = kl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        tg_l1[0] = tg_l1[0] + tg_l1[1];
        tg_kl[0] = tg_kl[0] + tg_kl[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Write the probe record ────────────────────────────────────────────
    if (tid == 0) {
        device AttentionProbe& out = output_probes[gid];
        out.head_id                = head;
        out.token_index            = token;
        out.teacher_max_logit      = t_max;
        out.student_max_logit      = s_max;
        out.teacher_entropy        = entropy_t;
        out.student_entropy        = entropy_s;
        out.sampled_probability_l1 = tg_l1[0];
        out.sampled_probability_kl = tg_kl[0];
    }
}
