// DS4 DeepSeek V4 compressed sparse attention (CSA/HCA) HIP kernel.
//
// Compile-time constants (substituted by AOT template expander):
//   {{HEAD_DIM}}        — 128 for DS4
//   {{NUM_HEADS}}       — varies by layer
//   {{COMPRESS_RATIO}}  — 0 (raw window only), 4 (ratio-4 indexed), 128 (heavy)
//   {{USE_INDEXER}}     — 0 or 1
//
// Three variants controlled by COMPRESS_RATIO/USE_INDEXER:
//   1. Raw sliding-window attention (latest 128 tokens, COMPRESS_RATIO=0)
//   2. Ratio-4 compressed with indexer (even layers, COMPRESS_RATIO=4, USE_INDEXER=1)
//   3. Ratio-128 heavily compressed (odd layers, COMPRESS_RATIO=128, USE_INDEXER=0)
//
// All use single-pass online softmax with warp shuffle reductions
// (no KV reload — rescale on the fly).
// Grid: (n_tokens, n_heads), block: 256 threads (8 warps x 32 lanes).
// HEAD_DIM must be a multiple of 128; each lane processes 4 floats via float4.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── Constants (vanilla C preprocessor — no `u` suffix on #if lines) ──

#define HD            {{HEAD_DIM}}
#define NH            {{NUM_HEADS}}
#define CR            {{COMPRESS_RATIO}}
#define USE_IDX       {{USE_INDEXER}}

#define ROWS_PER_STAGE 4
#define RAW_WINDOW_MAX 128
#define TOPK_MAX 512

// ── Device helpers (warp-aware for both HIP and CUDA) ─────────────────

__device__ static float ds4_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        v += __shfl_down(v, offset, 32);
#else
        v += __shfl_down_sync(0xFFFFFFFF, v, offset, 32);
#endif
    }
    return v;
}

__device__ static float ds4_warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        v = fmaxf(v, __shfl_down(v, offset, 32));
#else
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, offset, 32));
#endif
    }
    return v;
}

__device__ static float ds4_dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Warp broadcast: get value from src_lane to all lanes.
__device__ static float ds4_warp_broadcast(float v, int src_lane) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    return __shfl(v, src_lane, 32);
#else
    return __shfl_sync(0xFFFFFFFF, v, src_lane, 32);
#endif
}

// ── Kernel 1: raw sliding-window attention ────────────────────────────
// COMPRESS_RATIO = 0: only the latest RAW_WINDOW_MAX raw KV tokens.
// Single-pass online softmax with warp reductions.

#if CR == 0

extern "C" __global__ void compressed_attn_kernel(
    float* __restrict__ heads,
    const float* __restrict__ sinks,
    const float* __restrict__ q,
    const float* __restrict__ raw_kv,
    uint32_t n_tokens,
    uint32_t raw_capacity,
    uint32_t raw_start,
    uint32_t window)
{
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= NH) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t hd_d4 = HD >> 2;

    // Compute raw KV range within sliding window.
    uint32_t raw_cnt = 0u;
    uint32_t raw_off = 0u;
    {
        const uint32_t n_avail = t + 1u;
        const uint32_t w = (window != 0u && window < RAW_WINDOW_MAX)
                               ? window
                               : RAW_WINDOW_MAX;
        raw_cnt = (w != 0u && n_avail > w) ? w : n_avail;
        raw_off = n_avail - raw_cnt;
    }

    // Per-lane Q: each of 32 lanes loads one float4 (HD=128 → 32 float4).
    const float4* q4 = (const float4*)(q + ((uint64_t)t * NH + h) * HD);
    float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (lane < hd_d4) qv = q4[lane];

    // Shared memory.
    __shared__ uint32_t raw_rows[RAW_WINDOW_MAX];
    __shared__ float4 kv_shared[ROWS_PER_STAGE * (HD >> 2)];

    const float scale = rsqrtf((float)HD);

    // Translate raw offsets to circular-buffer indices.
    if (threadIdx.x < raw_cnt) {
        raw_rows[threadIdx.x] = (raw_start + raw_off + threadIdx.x) % raw_capacity;
    }
    __syncthreads();

    // Single-pass online softmax:
    //   o = o * old_scale + k * row_scale
    //   sum_s = sum_s * old_scale + row_scale
    // where old_scale = expf(max_s - new_m), row_scale = expf(score - new_m).
    //
    // sum_s starts at 1.0 = expf(sinks[h] - max_s) since max_s = sinks[h]
    // initially.  The sink contribution is folded into the online
    // recurrence and does not need separate handling at the end.
    float max_s = sinks[h];
    float sum_s = 1.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t row0 = 0u; row0 < raw_cnt; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = (raw_cnt - row0 < ROWS_PER_STAGE)
                                ? (raw_cnt - row0)
                                : ROWS_PER_STAGE;
        // Load KV tile into shared.
        for (uint32_t off = threadIdx.x; off < nr * hd_d4; off += blockDim.x) {
            const uint32_t rr = off / hd_d4;
            const uint32_t d4 = off % hd_d4;
            const uint32_t sr = row0 + rr;
            if (sr < raw_cnt) {
                const uint32_t ridx = raw_rows[sr];
                const float4* src =
                    (const float4*)(raw_kv + (uint64_t)ridx * HD);
                kv_shared[off] = src[d4];
            }
        }
        __syncthreads();

        if (lane < hd_d4) {
            for (uint32_t rr = 0u; rr < nr; rr++) {
                const float4* kv4 = kv_shared + rr * hd_d4;
                float score = ds4_dot4_f32(qv, kv4[lane]);
                score = ds4_warp_sum_f32(score);
                score = ds4_warp_broadcast(score, 0);
                score *= scale;

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + kv4[lane].x * row_scale;
                o0.y = o0.y * old_scale + kv4[lane].y * row_scale;
                o0.z = o0.z * old_scale + kv4[lane].z * row_scale;
                o0.w = o0.w * old_scale + kv4[lane].w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    // Normalize: o /= sum_s.
    // sum_s already contains the sink contribution (initialised at 1.0
    // and rescaled through the online recurrence).
    if (lane < hd_d4) {
        const float inv_s = (sum_s != 0.0f) ? (1.0f / sum_s) : 0.0f;
        float4* out4 = (float4*)(heads + ((uint64_t)t * NH + h) * HD);
        out4[lane] = make_float4(
            o0.x * inv_s, o0.y * inv_s, o0.z * inv_s, o0.w * inv_s);
    }
}

#endif  // CR == 0

// ── Kernel 2: ratio-4 compressed attention with indexer ──────────────
// CR = 4, USE_IDX = 1: raw window + top-k selected compressed KV rows.
// Indexer provides up to TOPK_MAX selected row indices.

#if CR == 4 && USE_IDX == 1

extern "C" __global__ void compressed_attn_kernel(
    float* __restrict__ heads,
    const float* __restrict__ sinks,
    const float* __restrict__ q,
    const float* __restrict__ raw_kv,
    const float* __restrict__ comp_kv,
    const int32_t* __restrict__ topk,
    uint32_t n_tokens,
    uint32_t n_heads,
    uint32_t head_dim,
    uint32_t n_raw,
    uint32_t raw_capacity,
    uint32_t raw_start,
    uint32_t n_comp,
    uint32_t top_k,
    uint32_t window)
{
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= NH) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t hd_d4 = HD >> 2;

    const float4* q4 = (const float4*)(q + ((uint64_t)t * NH + h) * HD);
    float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (lane < hd_d4) qv = q4[lane];

    // Visible compressed KV count at ratio 4.
    const uint32_t qpos = t;
    uint32_t visible_comp = (qpos + 1u) / 4u;
    if (visible_comp > n_comp) visible_comp = n_comp;

    // Raw sliding window.
    uint32_t raw_cnt = 0u;
    uint32_t raw_off = 0u;
    {
        const uint32_t n_avail = t + 1u;
        const uint32_t w = (window != 0u && window < RAW_WINDOW_MAX)
                               ? window
                               : RAW_WINDOW_MAX;
        raw_cnt = (w != 0u && n_avail > w) ? w : n_avail;
        raw_off = n_avail - raw_cnt;
    }

    // Shared memory.
    __shared__ uint32_t raw_rows[RAW_WINDOW_MAX];
    __shared__ uint32_t comp_rows[TOPK_MAX];
    __shared__ uint32_t comp_cnt_s;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * (HD >> 2)];

    // Prepare raw row indices.
    if (threadIdx.x < raw_cnt) {
        raw_rows[threadIdx.x] = (raw_start + raw_off + threadIdx.x) % raw_capacity;
    }

    // Indexer: thread 0 selects top-k valid compressed rows.
    if (threadIdx.x == 0u) {
        uint32_t cc = 0u;
        const uint32_t k_lim = (top_k < TOPK_MAX) ? top_k : TOPK_MAX;
        for (uint32_t i = 0u; i < k_lim && cc < TOPK_MAX; i++) {
            const int32_t ci = topk[(uint64_t)t * top_k + i];
            if (ci < 0) continue;
            const uint32_t c = (uint32_t)ci;
            if (c < visible_comp) comp_rows[cc++] = c;
        }
        comp_cnt_s = cc;
    }
    __syncthreads();

    const uint32_t comp_cnt = comp_cnt_s;
    const uint32_t n_score = raw_cnt + comp_cnt;
    const float scale = rsqrtf((float)HD);

    // Single-pass online softmax.
    float max_s = sinks[h];
    float sum_s = 1.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t row0 = 0u; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = (n_score - row0 < ROWS_PER_STAGE)
                                ? (n_score - row0)
                                : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * hd_d4; off += blockDim.x) {
            const uint32_t rr = off / hd_d4;
            const uint32_t d4 = off % hd_d4;
            const uint32_t sr = row0 + rr;
            if (sr < n_score) {
                const float4* src;
                uint64_t kv_off;
                if (sr < raw_cnt) {
                    src = (const float4*)raw_kv;
                    kv_off = (uint64_t)raw_rows[sr] * HD;
                } else {
                    src = (const float4*)comp_kv;
                    kv_off = (uint64_t)comp_rows[sr - raw_cnt] * HD;
                }
                kv_shared[off] = src[kv_off / 4u + d4];
            }
        }
        __syncthreads();

        if (lane < hd_d4) {
            for (uint32_t rr = 0u; rr < nr; rr++) {
                const float4* kv4 = kv_shared + rr * hd_d4;
                float score = ds4_dot4_f32(qv, kv4[lane]);
                score = ds4_warp_sum_f32(score);
                score = ds4_warp_broadcast(score, 0);
                score *= scale;

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + kv4[lane].x * row_scale;
                o0.y = o0.y * old_scale + kv4[lane].y * row_scale;
                o0.z = o0.z * old_scale + kv4[lane].z * row_scale;
                o0.w = o0.w * old_scale + kv4[lane].w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    // Normalize: o /= sum_s.
    if (lane < hd_d4) {
        const float inv_s = (sum_s != 0.0f) ? (1.0f / sum_s) : 0.0f;
        float4* out4 = (float4*)(heads + ((uint64_t)t * NH + h) * HD);
        out4[lane] = make_float4(
            o0.x * inv_s, o0.y * inv_s, o0.z * inv_s, o0.w * inv_s);
    }
}

#endif  // CR == 4 && USE_IDX == 1

// ── Kernel 3: ratio-128 heavily compressed attention ─────────────────
// CR = 128, USE_IDX = 0: raw window + all compressed KV rows at ratio 128.
// Deterministic stride (no indexer).

#if CR == 128 && USE_IDX == 0

extern "C" __global__ void compressed_attn_kernel(
    float* __restrict__ heads,
    const float* __restrict__ sinks,
    const float* __restrict__ q,
    const float* __restrict__ raw_kv,
    const float* __restrict__ comp_kv,
    uint32_t n_tokens,
    uint32_t n_heads,
    uint32_t head_dim,
    uint32_t n_raw,
    uint32_t raw_capacity,
    uint32_t raw_start,
    uint32_t n_comp,
    uint32_t window)
{
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= NH) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t hd_d4 = HD >> 2;

    const float4* q4 = (const float4*)(q + ((uint64_t)t * NH + h) * HD);
    float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (lane < hd_d4) qv = q4[lane];

    // Compressed rows at ratio 128: indices 0..n_comp-1, max = (qpos+1)/128.
    const uint32_t qpos = t;
    uint32_t comp_cnt = (qpos + 1u) / 128u;
    if (comp_cnt > n_comp) comp_cnt = n_comp;

    // Raw sliding window.
    uint32_t raw_cnt = 0u;
    uint32_t raw_off = 0u;
    {
        const uint32_t n_avail = t + 1u;
        const uint32_t w = (window != 0u && window < RAW_WINDOW_MAX)
                               ? window
                               : RAW_WINDOW_MAX;
        raw_cnt = (w != 0u && n_avail > w) ? w : n_avail;
        raw_off = n_avail - raw_cnt;
    }

    __shared__ uint32_t raw_rows[RAW_WINDOW_MAX];
    __shared__ float4 kv_shared[ROWS_PER_STAGE * (HD >> 2)];

    if (threadIdx.x < raw_cnt) {
        raw_rows[threadIdx.x] = (raw_start + raw_off + threadIdx.x) % raw_capacity;
    }
    __syncthreads();

    const uint32_t n_score = raw_cnt + comp_cnt;
    const float scale = rsqrtf((float)HD);

    // Single-pass online softmax.
    float max_s = sinks[h];
    float sum_s = 1.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t row0 = 0u; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = (n_score - row0 < ROWS_PER_STAGE)
                                ? (n_score - row0)
                                : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * hd_d4; off += blockDim.x) {
            const uint32_t rr = off / hd_d4;
            const uint32_t d4 = off % hd_d4;
            const uint32_t sr = row0 + rr;
            if (sr < n_score) {
                const float4* src;
                uint64_t kv_off;
                if (sr < raw_cnt) {
                    src = (const float4*)raw_kv;
                    kv_off = (uint64_t)raw_rows[sr] * HD;
                } else {
                    src = (const float4*)comp_kv;
                    kv_off = (uint64_t)(sr - raw_cnt) * HD;
                }
                kv_shared[off] = src[kv_off / 4u + d4];
            }
        }
        __syncthreads();

        if (lane < hd_d4) {
            for (uint32_t rr = 0u; rr < nr; rr++) {
                const float4* kv4 = kv_shared + rr * hd_d4;
                float score = ds4_dot4_f32(qv, kv4[lane]);
                score = ds4_warp_sum_f32(score);
                score = ds4_warp_broadcast(score, 0);
                score *= scale;

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + kv4[lane].x * row_scale;
                o0.y = o0.y * old_scale + kv4[lane].y * row_scale;
                o0.z = o0.z * old_scale + kv4[lane].z * row_scale;
                o0.w = o0.w * old_scale + kv4[lane].w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    // Normalize: o /= sum_s.
    if (lane < hd_d4) {
        const float inv_s = (sum_s != 0.0f) ? (1.0f / sum_s) : 0.0f;
        float4* out4 = (float4*)(heads + ((uint64_t)t * NH + h) * HD);
        out4[lane] = make_float4(
            o0.x * inv_s, o0.y * inv_s, o0.z * inv_s, o0.w * inv_s);
    }
}

#endif  // CR == 128 && USE_IDX == 0
