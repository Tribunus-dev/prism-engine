// DS4 MoE Router — top-K expert selection with warp-level primitives.
// AOT compilation uses {{PLACEHOLDER}} for tunable parameters:
//   {{N_EXPERT}}          — total number of experts
//   {{N_EXPERT_USED}}     — number of experts selected per token
//   {{WARP_SIZE}}         — warp width (64 for AMD, 32 for CUDA)
//   {{WARP_MASK}}         — shuffle mask matching warp width
//   {{MAX_EXPERT_GROUPS}} — max(1, ceil(N_EXPERT / WARP_SIZE))
//   {{HIDDEN_DIM}}        — hidden state dimension

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Device helpers — mirrors ds4_rocm_router.cuh
__device__ static float ds4_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -10.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ static bool score_better(
    float av, uint32_t ai,
    float bv, uint32_t bi)
{
    return av > bv || (av == bv && ai < bi);
}

// ---------------------------------------------------------------------------
// Kernel: ds4_moe_router_topk
//
// Given hidden states [batch, hidden_dim] and a router weight matrix,
// compute router logits via matmul, then select the top-{{N_EXPERT_USED}}
// experts per token using warp-level shuffle reduction.
//
// Inputs:
//   hidden      — [batch, hidden_dim] f32 hidden states
//   router_w    — [N_EXPERT, hidden_dim] f32 router weight matrix
//   bias        — [N_EXPERT] f32 optional bias (NULL if none)
//
// Outputs:
//   expert_ids  — [batch, N_EXPERT_USED] int32 selected expert indices
//   weights     — [batch, N_EXPERT_USED] f32 softmax-normalized weights
//   logits_out  — [batch, N_EXPERT] f32 intermediate logits (may be NULL)
// ---------------------------------------------------------------------------
extern "C" __global__ void ds4_moe_router_topk(
    const float* __restrict__ hidden,
    const float* __restrict__ router_w,
    const float* __restrict__ bias,
    int32_t* __restrict__ expert_ids,
    float* __restrict__ weights,
    float* __restrict__ logits_out,
    uint32_t batch,
    uint32_t hidden_dim,
    uint32_t n_expert,
    float expert_weight_scale,
    int32_t has_bias)
{
    // Each block handles one token; blockDim.y tokens per block.
    // threadIdx.x = lane within warp (0..WARP_SIZE-1)
    // threadIdx.y = row within block
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= batch) return;

    constexpr uint32_t active_n_expert  = {{N_EXPERT}};
    constexpr uint32_t active_n_used    = {{N_EXPERT_USED}};
    constexpr uint32_t warp_size        = {{WARP_SIZE}};
    constexpr uint32_t warp_mask        = {{WARP_MASK}};
    constexpr uint32_t groups           = {{MAX_EXPERT_GROUPS}};
    const float active_scale            = expert_weight_scale > 0.0f
                                              ? expert_weight_scale
                                              : 1.0f;

    // Lane must be within the warp (static assert via compile-time guard)
    // Not a runtime check — the AOT generator must ensure blockDim.x == warp_size.
    if (lane >= warp_size) return;

    // Fixed-size local storage for per-stripe logits/probs/scores.
    // MAX_EXPERT_GROUPS is validated at AOT compile time.
    float logits_buf[MAX_EXPERT_GROUPS];
    float prob_buf[MAX_EXPERT_GROUPS];
    float score_buf[MAX_EXPERT_GROUPS];

    // --- Step 1: matmul hidden[t,:] * router_w → logits per expert ---
    // Each lane processes one expert per group; groups = ceil(N_EXPERT / WARP_SIZE).
    #pragma unroll
    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t e = lane + g * warp_size;
        if (e < active_n_expert) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < hidden_dim; d++) {
                sum += hidden[(uint64_t)t * hidden_dim + d]
                     * router_w[(uint64_t)e * hidden_dim + d];
            }
            logits_buf[g] = sum;
        } else {
            logits_buf[g] = -INFINITY;
        }
    }

    // --- Step 2: softplus activation + optional bias ---
    #pragma unroll
    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t e = lane + g * warp_size;
        if (e < active_n_expert) {
            const float p = sqrtf(ds4_softplus(logits_buf[g]));
            prob_buf[g]  = p;
            score_buf[g] = p + (has_bias && bias ? bias[e] : 0.0f);
        } else {
            prob_buf[g]  = 0.0f;
            score_buf[g] = -INFINITY;
        }
    }

    // Write intermediate logits if requested
    if (logits_out) {
        #pragma unroll
        for (uint32_t g = 0; g < groups; g++) {
            const uint32_t e = lane + g * warp_size;
            if (e < active_n_expert) {
                logits_out[(uint64_t)t * active_n_expert + e] = logits_buf[g];
            }
        }
    }

    // --- Step 3: top-K selection via warp-level butterfly reduction ---
    // Iteratively select the best remaining expert using __shfl_xor_sync
    // across the full warp (mask respects warp_size).
    float out_prob[{{N_EXPERT_USED}}];
    uint32_t out_idx[{{N_EXPERT_USED}}];

    #pragma unroll
    for (uint32_t k = 0; k < active_n_used; k++) {
        // Local best in this lane's stripe
        float best_score = -INFINITY;
        float best_prob  = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t g = 0; g < groups; g++) {
            const uint32_t e = lane + g * warp_size;
            if (e < active_n_expert) {
                const float s = score_buf[g];
                if (score_better(s, e, best_score, best_idx)) {
                    best_score = s;
                    best_prob  = prob_buf[g];
                    best_idx   = e;
                }
            }
        }

        // Warp-level butterfly: reduce across all lanes
        #pragma unroll
        for (uint32_t mask = warp_size / 2u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(warp_mask, best_score, mask);
            const float other_prob  = __shfl_xor_sync(warp_mask, best_prob,  mask);
            const uint32_t other_idx = __shfl_xor_sync(warp_mask, best_idx,   mask);
            if (score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob  = other_prob;
                best_idx   = other_idx;
            }
        }

        // Remove selected expert from the pool
        #pragma unroll
        for (uint32_t g = 0; g < groups; g++) {
            const uint32_t e = lane + g * warp_size;
            if (e == best_idx) score_buf[g] = -INFINITY;
        }

        // Lane 0 records the global best
        if (lane == 0) {
            out_idx[k]  = best_idx;
            out_prob[k] = best_prob;
        }
    }

    // --- Step 4: softmax over selected probabilities + write outputs ---
    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < active_n_used; j++) {
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f); // 1/16384 prevent div-by-zero
        #pragma unroll
        for (uint32_t j = 0; j < active_n_used; j++) {
            expert_ids[(uint64_t)t * active_n_used + j] = (int32_t)out_idx[j];
            weights[(uint64_t)t * active_n_used + j]    = out_prob[j] / sum * active_scale;
        }
    }
}
