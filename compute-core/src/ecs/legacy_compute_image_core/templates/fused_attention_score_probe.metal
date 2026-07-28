// ── Fused Attention Score Probe ──────────────────────────────────────────────
//
// Computes QK dot products for sampled (query_token, head) pairs and writes
// AttentionProbe records. Reads Q and K from activation arena. Avoids storing
// full attention matrices — only sampled positions produce probe output.
//
// Each threadgroup (64 threads) handles one query token:
//   1. Deterministically select ~1/32 of heads for probing
//   2. For each sampled head, compute Q[query][head] · K[kv][head]
//      for all KV positions (causal mask: kv ≤ query)
//   3. Apply scaling factor (1/sqrt(head_dim))
//   4. Write AttentionProbe records
//
// 3 personalities via mode_flags:
//   deployment: compute scores only, no probe output (elided)
//   diagnostic (MODE_DIAGNOSTIC): full AttentionProbe with entropy
//   fused_scoring (MODE_FUSED_SCORING): compact ErrorPartial per sampled head
//
// Buffer binding:
//   0 = q_activations   half[]              — Q activations [num_tokens × q_dim]
//   1 = k_activations   half[]              — K activations [num_kv × k_dim]
//   2 = output_probes   AttentionProbe[]    — diagnostic probe output (optional)
//   3 = error_partials  ErrorPartial[]      — fused_scoring output (optional)
//   4 = params          ProjectionParams (constant)
//       → in_dim  = head_dim (dimension per attention head)
//       → out_dim = num_heads (number of query heads)
//       → page_count = num_tokens (query token count)
//       → page_width = num_kv (KV token count)
//   5 = receipt         KernelReceipt       — instrumentation (optional)
//
// Function constants:
//   HEAD_DIM (index 0) — dimension per head, default 128
//   SCALE    (index 1) — 1/sqrt(head_dim), pre-computed at compile-image build
//
// Grid: threadgroups = num_tokens, threads_per_threadgroup = 64

#include <metal_stdlib>
using namespace metal;

// ── Mode flag constants ─────────────────────────────────────────────────────
constant uint MODE_SIDECAR       = 1u;    // bit 0 (unused in this kernel)
constant uint MODE_RECEIPT       = 2u;    // bit 1
constant uint MODE_CANDIDATE     = 4u;    // bit 2 (unused in this kernel)
constant uint MODE_DIAGNOSTIC    = 8u;    // bit 3 — full AttentionProbe
constant uint MODE_FUSED_SCORING = 16u;   // bit 4 — compact ErrorPartial

// ── Function constants ──────────────────────────────────────────────────────
constant uint   HEAD_DIM_FC [[function_constant(0)]];
constant float  SCALE_FC    [[function_constant(1)]];

// ── Data structures (repr(C) matching Rust kernel_types.rs) ────────────────

struct ProjectionParams {
    uint  in_dim;            // head_dim
    uint  out_dim;           // num_heads
    uint  page_count;        // num_tokens (query)
    uint  page_width;        // num_kv (KV tokens)
    uint  mode_flags;
    uint  probe_seed;
    uint  reserved[5];
};

struct AttentionProbe {
    uint  head_id;
    uint  token_index;
    float teacher_max_logit;
    float student_max_logit;
    float teacher_entropy;
    float student_entropy;
    float sampled_probability_l1;
    float sampled_probability_kl;
};

struct ErrorPartial {
    float sum_sq_error;
    float sum_abs_error;
    float dot_teacher_student;
    float sum_teacher_sq;
    float sum_student_sq;
    float max_abs_error;
    uint  element_count;
    uint  _pad;
};

struct KernelReceipt {
    uint     kernel_id;
    uint     phase_id;
    uint     page_count;
    uint     sidecar_hits;
    uint     sidecar_entries_read;
    uint     threadgroups;
    uint     threads_per_threadgroup;
    uint     output_elements;
    uint     flags;
    uint     _pad_receipt;
    uint64_t logical_weight_bytes;
    uint64_t logical_sidecar_bytes;
    uint64_t logical_activation_bytes;
};

// ── Helper: dot product for one head slice ──────────────────────────────────
//
// Computes Q[query][head] · K[kv][head] over head_dim elements.
// Each thread handles a subset of head_dim elements (stride 64).

METAL_FUNC float qk_dot_slice(
    device const half* q_activations,
    uint               q_head_offset,
    device const half* k_activations,
    uint               k_head_offset,
    uint               head_dim,
    uint               tid)
{
    float dot = 0.0f;
    for (uint i = tid; i < head_dim; i += 64) {
        dot = fma(float(q_activations[q_head_offset + i]),
                  float(k_activations[k_head_offset + i]), dot);
    }
    return dot;
}

// ── Kernel ──────────────────────────────────────────────────────────────────

kernel void fused_attention_score_probe(
    device const half*          q_activations  [[buffer(0)]],
    device const half*          k_activations  [[buffer(1)]],
    device AttentionProbe*      output_probes  [[buffer(2)]],
    device ErrorPartial*        error_partials [[buffer(3)]],
    constant ProjectionParams&  params         [[buffer(4)]],
    device KernelReceipt*       receipt        [[buffer(5)]],
    uint gid                                   [[threadgroup_position_in_grid]],
    uint tid                                   [[thread_position_in_threadgroup]],
    uint simd_lane                             [[thread_index_in_simdgroup]],
    uint simd_id                               [[simdgroup_index_in_threadgroup]])
{
    const uint flags      = params.mode_flags;
    const uint head_dim   = HEAD_DIM_FC > 0 ? HEAD_DIM_FC : params.in_dim;
    const uint num_heads  = params.out_dim;
    const uint num_tokens = params.page_count;
    const uint num_kv     = params.page_width;

    if (gid >= num_tokens) return;

    const float scale = SCALE_FC > 0.0f ? SCALE_FC : (1.0f / sqrt(float(head_dim)));

    // ── Phase 1: deterministic head sampling via thread 0 ──────────────────
    // Select ~1/32 of heads for probing using a hash per (token, head) pair.
    // Store selected head indices in threadgroup memory.
    threadgroup uint sampled_heads[64];
    threadgroup uint n_sampled = 0;

    if (tid == 0) {
        uint ns = 0;
        for (uint h = 0; h < num_heads && ns < 64; ++h) {
            uint hash = (gid * num_heads + h) * 2654435761u;
            hash ^= params.probe_seed;
            if (hash % 32u == 0u) {
                sampled_heads[ns++] = h;
            }
        }
        n_sampled = ns;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: compute QK scores for sampled heads ───────────────────────
    // Q layout: [token × (num_heads * head_dim)]
    // K layout: [kv_token × (num_heads * head_dim)]

    const uint q_token_offset = gid * num_heads * head_dim;

    for (uint si = 0; si < n_sampled; ++si) {
        const uint head = sampled_heads[si];
        const uint q_head_off = q_token_offset + head * head_dim;

        // ── Pass 1: find max score across all KV tokens ──────────────────
        float max_score = -INFINITY;

        for (uint kv = 0; kv < num_kv; ++kv) {
            if (kv > gid) break;  // causal mask

            const uint k_head_off = (kv * num_heads + head) * head_dim;
            float dot = qk_dot_slice(q_activations, q_head_off,
                                     k_activations, k_head_off,
                                     head_dim, tid);
            dot = simd_sum(dot) * scale;

            // Cross-SIMD reduction (2 SIMD groups × 32 lanes).
            threadgroup float tg_dot[2];
            if (simd_lane == 0) { tg_dot[simd_id] = dot; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) { dot = tg_dot[0] + tg_dot[1]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Broadcast reduced value to all threads via TG memory.
            threadgroup float shared_val = 0.0f;
            if (tid == 0) { shared_val = dot; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            max_score = max(max_score, shared_val);
        }

        // ── Pass 2: softmax denominator (sum_exp) ────────────────────────
        float sum_exp = 0.0f;

        for (uint kv = 0; kv < num_kv; ++kv) {
            if (kv > gid) break;

            const uint k_head_off = (kv * num_heads + head) * head_dim;
            float dot = qk_dot_slice(q_activations, q_head_off,
                                     k_activations, k_head_off,
                                     head_dim, tid);
            dot = simd_sum(dot) * scale;

            threadgroup float tg_dot[2];
            if (simd_lane == 0) { tg_dot[simd_id] = dot; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) { dot = tg_dot[0] + tg_dot[1]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float sv;
            if (tid == 0) { sv = dot; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) { sum_exp += exp(sv - max_score); }
        }

        // ── Pass 3: entropy and divergence (diagnostic only) ─────────────
        if ((flags & MODE_DIAGNOSTIC) && output_probes) {
            float l1 = 0.0f;
            float kl = 0.0f;
            float psum = 0.0f;

            for (uint kv = 0; kv < num_kv; ++kv) {
                if (kv > gid) break;

                const uint k_head_off = (kv * num_heads + head) * head_dim;
                float dot = qk_dot_slice(q_activations, q_head_off,
                                         k_activations, k_head_off,
                                         head_dim, tid);
                dot = simd_sum(dot) * scale;

                threadgroup float tg_dot[2];
                if (simd_lane == 0) { tg_dot[simd_id] = dot; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) { dot = tg_dot[0] + tg_dot[1]; }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0 && sum_exp > 0.0f) {
                    float p = exp(dot - max_score) / sum_exp;
                    psum += p;
                    if (p > 0.0f) {
                        l1 += p;  // sum of probs (should converge to 1)
                        kl += p * log(p);  // negative entropy accumulator
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                const float entropy = (psum > 0.0f)
                    ? -kl / psum + log(psum)
                    : 0.0f;

                device AttentionProbe& out = output_probes[gid * num_heads + head];
                out.head_id                = head;
                out.token_index            = gid;
                out.teacher_max_logit      = max_score;
                out.student_max_logit      = max_score;
                out.teacher_entropy        = entropy;
                out.student_entropy        = entropy;
                out.sampled_probability_l1 = l1;
                out.sampled_probability_kl = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Fused scoring: write ErrorPartial ──────────────────────────
        if ((flags & MODE_FUSED_SCORING) && error_partials) {
            if (tid == 0) {
                device ErrorPartial& ep = error_partials[gid * 64 + si];
                ep.sum_sq_error          = 0.0f;
                ep.sum_abs_error         = 0.0f;
                ep.dot_teacher_student   = max_score;
                ep.sum_teacher_sq        = sum_exp;
                ep.sum_student_sq        = max_score;
                ep.max_abs_error         = 0.0f;
                ep.element_count         = num_kv;
                ep._pad                  = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ── Phase 3: Instrumentation (MODE_RECEIPT) ─────────────────────────
    if ((flags & MODE_RECEIPT) && gid == 0 && tid == 0) {
        receipt->kernel_id              = 0;
        receipt->phase_id               = 0;
        receipt->page_count             = num_tokens;
        receipt->sidecar_hits           = 0;
        receipt->sidecar_entries_read   = 0;
        receipt->threadgroups           = num_tokens;
        receipt->threads_per_threadgroup = 64;
        receipt->output_elements        = n_sampled;
        receipt->flags                  = flags;
        receipt->_pad_receipt           = 0;
        receipt->logical_weight_bytes   = 0;
        receipt->logical_sidecar_bytes  = 0;
        receipt->logical_activation_bytes = ulong(n_sampled) * ulong(sizeof(AttentionProbe));
    }
}
