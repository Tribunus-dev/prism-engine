// DS4 Flash Persistent Megakernel — Per-Layer Operation Functions (MI300X)
// ============================================================================
// AOT-compiled with {{PLACEHOLDER}} substitution for:
//   {{HIDDEN_DIM}}       — 4096
//   {{HEAD_DIM}}         — 128
//   {{N_EXPERT}}         — 256
//   {{TOP_K}}            — 6
//   {{EXPERT_MID_DIM}}   — 2560
//   {{VOCAB_SIZE}}       — 128256
//   {{LAYERS}}           — 43
//   {{MAX_CONTEXT}}      — 1048576
//   {{TOP_K_MAX}}        — 512
//   {{QK_K}}             — 256 (GGML super-block size)
//   {{N_ROT}}            — 64 (DS4 rotary half-dim)
//
// All functions are __device__ — they run inside the single persistent
// threadblock launched by ds4_persistent_decode. Every call shares:
//   h_buf   — half[DIM]  (8 KB for DIM=4096, __shared__)
//   scratch — half[16384] (32 KB __shared__, usable as 4096 f32)
//   tid/tg_sz — from the outer loop (threadIdx.x, blockDim.x)
//
// scratch is reusable across calls — each function may use the full 16 KB
// for its own temporaries.
// ============================================================================

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// ===========================================================================
// Compile-time constants (AOT substitution via {{PLACEHOLDER}})
// ===========================================================================
#ifndef HIDDEN_DIM
#define HIDDEN_DIM       {{HIDDEN_DIM}}
#endif
#ifndef HEAD_DIM
#define HEAD_DIM         {{HEAD_DIM}}
#endif
#ifndef N_EXPERT
#define N_EXPERT         {{N_EXPERT}}
#endif
#ifndef TOP_K
#define TOP_K            {{TOP_K}}
#endif
#ifndef EXPERT_MID_DIM
#define EXPERT_MID_DIM   {{EXPERT_MID_DIM}}
#endif
#ifndef MAX_CONTEXT
#define MAX_CONTEXT      {{MAX_CONTEXT}}
#endif
#ifndef TOP_K_MAX
#define TOP_K_MAX        {{TOP_K_MAX}}
#endif
#ifndef QK_K
#define QK_K             256
#endif
#ifndef N_ROT
#define N_ROT            64
#endif

#define N_HEADS          (HIDDEN_DIM / HEAD_DIM)
#define RAW_WINDOW       128
#define SHARED_FF        (4 * HIDDEN_DIM)

// ===========================================================================
// Device helpers
// ===========================================================================

__device__ static float wsum_f32(float v) {
    for (int off = 16; off > 0; off >>= 1) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        v += __shfl_down(v, off, 32);
#else
        v += __shfl_down_sync(0xFFFFFFFF, v, off, 32);
#endif
    }
    return v;
}

__device__ static float wmax_f32(float v) {
    for (int off = 16; off > 0; off >>= 1) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        v = fmaxf(v, __shfl_down(v, off, 32));
#else
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, off, 32));
#endif
    }
    return v;
}

__device__ static float bsum_f32(float v) {
    __shared__ float sh[256];
    uint32_t lane = threadIdx.x & 31u;
    uint32_t wid  = threadIdx.x >> 5u;
    uint32_t nw   = (blockDim.x + 31u) >> 5u;
    v = wsum_f32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? sh[lane] : 0.0f;
    if (wid == 0u) v = wsum_f32(v);
    if (threadIdx.x == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ static float bmax_f32(float v) {
    __shared__ float sh[256];
    uint32_t lane = threadIdx.x & 31u;
    uint32_t wid  = threadIdx.x >> 5u;
    uint32_t nw   = (blockDim.x + 31u) >> 5u;
    v = wmax_f32(v);
    if (lane == 0u) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? sh[lane] : -3.4e38f;
    if (wid == 0u) v = wmax_f32(v);
    if (threadIdx.x == 0u) sh[0] = v;
    __syncthreads();
    return sh[0];
}

__device__ static int is_finite_f32(float x) {
    return (x > -3.4e38f && x < 3.4e38f) || (x == 0.0f);
}

__device__ static float q8_0_scale(const uint8_t* blk) {
    return __half2float(*(const __half*)blk);
}

__device__ static int32_t dot_i8x32(const int8_t* a, const int8_t* b) {
    int32_t dot = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(*(const int32_t*)(a + i), *(const int32_t*)(b + i), dot);
    }
    return dot;
}

__device__ static int32_t clip_i8(int v) {
    return (v < -128) ? -128 : ((v > 127) ? 127 : v);
}

// ===========================================================================
// GGML block types
// ===========================================================================
struct block_q4_K {
    __half   d;
    __half   dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(block_q4_K) == 140, "block_q4_K size");

struct block_q2_K {
    __half   d;
    __half   dmin;
    uint8_t  scales[16];
    uint8_t  qs[64];
};
static_assert(sizeof(block_q2_K) == 72, "block_q2_K size");

struct block_q8_K {
    __half   d;
    int16_t  bsums[16];
    int8_t   qs[256];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K size");

// ===========================================================================
// Q4_K / Q2_K device helpers
// ===========================================================================

__device__ static void q4k_scales(uint32_t j, const uint8_t* scales,
                                  uint8_t* d_out, uint8_t* m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ static int32_t dot_q4_32(const uint8_t* qs, const int8_t* q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        int32_t v = (*(const int32_t*)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t*)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dot_q2_16(const uint8_t* q2, const int8_t* q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        int32_t v = (*(const int32_t*)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t*)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dot_q4K_q8K(const block_q4_K* x, const block_q8_K* y) {
    float xd   = __half2float(x->d);
    float xmin = __half2float(x->dmin);
    int isum = 0, summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        q4k_scales(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        uint32_t bo = (j >> 1u) * 32u;
        int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dot_q4_32(x->qs + bo, y->qs + j * 32u, shift);
    }
    return __half2float(y->d) * xd * (float)isum
         - __half2float(y->d) * xmin * (float)summs;
}

__device__ static float dot_q2K_q8K(const block_q2_K* x, const block_q8_K* y) {
    const uint8_t* q2 = x->qs;
    const int8_t*  q8 = y->qs;
    const uint8_t* sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    float dall = __half2float(y->d) * __half2float(x->d);
    float dmin = __half2float(y->d) * __half2float(x->dmin);
    int isum = 0, is_idx = 0;
    for (int k = 0; k < QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is_idx++] & 0x0f;
            isum += d * dot_q2_16(q2, q8, shift);
            d = sc[is_idx++] & 0x0f;
            isum += d * dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

// ===========================================================================
// 1. rms_norm<DIM>
// ===========================================================================
// Standard RMS normalization with per-element weight.
// Reference: ds4_rocm_norm_rope.cuh (rms_norm_weight_kernel)
// scratch[0..HIDDEN_DIM-1] = f32 conversion of h_buf (temporary)

template <uint32_t DIM>
__device__ void rms_norm(
    half* h_buf,
    half* scratch,
    const void* weight_ptr,
    uint tid,
    uint tg_sz)
{
    const float* weight = (const float*)weight_ptr;
    float* sf = (float*)scratch;

    float ssq = 0.0f;
    for (uint i = tid; i < DIM; i += tg_sz) {
        float v = __half2float(h_buf[i]);
        sf[i] = v;
        ssq += v * v;
    }

    float total = bsum_f32(ssq);
    float scale = rsqrtf(total / (float)DIM + 1e-6f);

    for (uint i = tid; i < DIM; i += tg_sz) {
        h_buf[i] = __float2half(sf[i] * scale * weight[i]);
    }
}

// ===========================================================================
// 2. compressed_attention<HEAD_DIM, COMPRESS_RATIO>
// ===========================================================================
// Decode attention: one head per warp.
// COMPRESS_RATIO determines KV compression:
//   0   = raw 128-token window only
//   4   = + ratio-4 indexed via TOP_K (even layers)
//   128 = + ratio-128 heavy (odd layers)
//
// Reference: ds4_rocm_attention.cuh (attention_decode_mixed_*_kernel)
//
// scratch is used for scores[] (f32 up to RAW_WINDOW + n_comp entries).
// Layer weights (for Q projection) and per-layer KV are addressed
// via kv_cache + layer_id in the persistent kernel's namespace.

template <uint32_t HD, uint32_t CR>
__device__ void compressed_attention(
    half* h_buf,
    half* scratch,
    void* kv_cache,
    uint  layer,
    uint  seq_pos,
    uint  tid,
    uint  tg_sz)
{
    constexpr uint32_t DIM     = HIDDEN_DIM;
    constexpr uint32_t NH      = N_HEADS;
    constexpr bool     USE_IDX = (CR == 4u);
    constexpr uint32_t HD4     = HD / 4u;

    uint32_t lane = tid & 31u;
    uint32_t warp = tid >> 5u;
    uint32_t nw   = (tg_sz + 31u) >> 5u;
    float* scores = (float*)scratch;

    // KV cache offsets per layer
    uint64_t lk = (uint64_t)layer * (uint64_t)MAX_CONTEXT * NH * HD;
    const float* raw_kv  = (const float*)((const char*)kv_cache + lk * sizeof(float));
    uint32_t raw_cap     = RAW_WINDOW;
    uint32_t raw_start   = (seq_pos >= raw_cap) ? (seq_pos - raw_cap + 1u) : 0u;
    uint32_t n_raw       = (seq_pos + 1u < raw_cap) ? (seq_pos + 1u) : raw_cap;

    uint64_t raw_bytes   = (uint64_t)raw_cap * NH * HD * sizeof(float);
    const float* comp_kv = NULL;
    uint32_t n_comp      = 0;
    const int32_t* topk  = NULL;
    uint32_t topk_cnt    = 0;
    const float* cmask   = NULL;

    if constexpr (CR > 0u) {
        comp_kv = (const float*)((const char*)raw_kv + raw_bytes);
        n_comp  = (seq_pos + 1u) / CR;
        uint32_t maxc = MAX_CONTEXT / CR;
        if (n_comp > maxc) n_comp = maxc;

        uint64_t comp_bytes = (uint64_t)n_comp * NH * HD * sizeof(float);
        if constexpr (USE_IDX) {
            topk = (const int32_t*)((const char*)comp_kv + comp_bytes);
            topk_cnt = 0;
            for (uint32_t i = 0; i < TOP_K_MAX; i++) {
                if (topk[i] >= 0) topk_cnt++; else break;
            }
            uint64_t tkb = (uint64_t)TOP_K_MAX * sizeof(int32_t);
            cmask = (const float*)((const char*)topk + tkb);
        } else {
            cmask = (const float*)((const char*)comp_kv + comp_bytes);
        }
    }

    float sc = rsqrtf((float)HD);

    for (uint32_t h = warp; h < NH; h += nw) {
        // Load Q for this head from h_buf
        float qh[HD];
        for (uint32_t d = lane; d < HD; d += 32u) {
            qh[d] = __half2float(h_buf[h * HD + d]);
        }

        uint32_t n_total = n_raw + n_comp;
        float lmax = -3.4e38f;

        // Score raw window tokens
        for (uint32_t r = lane; r < n_raw; r += 32u) {
            uint32_t kr = (raw_start + r) % raw_cap;
            const float* kv = raw_kv + ((uint64_t)kr * NH + h) * HD;
            float dot = 0.0f;
            for (uint32_t d = 0; d < HD; d++) dot += qh[d] * kv[d];
            float s = dot * sc;
            scores[r] = s;
            lmax = fmaxf(lmax, s);
        }

        // Score compressed tokens
        if constexpr (CR > 0u) {
            uint32_t cl = n_comp;
            if constexpr (USE_IDX) { cl = (topk_cnt < n_comp) ? topk_cnt : n_comp; }
            for (uint32_t c = lane; c < cl; c += 32u) {
                uint32_t ci = c;
                if constexpr (USE_IDX) {
                    ci = (uint32_t)topk[c];
                    if ((int32_t)ci < 0) break;
                }
                const float* kv = comp_kv + ((uint64_t)ci * NH + h) * HD;
                float dot = 0.0f;
                for (uint32_t d = 0; d < HD; d++) dot += qh[d] * kv[d];
                float s = dot * sc;
                if (cmask && c < n_comp) s += cmask[ci];
                if (s > -3.4e37f) {
                    scores[n_raw + c] = s;
                    lmax = fmaxf(lmax, s);
                } else {
                    scores[n_raw + c] = -3.4e38f;
                }
            }
        }

        // Online softmax
        float ms = bmax_f32(lmax);
        float ls = 0.0f;
        for (uint32_t i = lane; i < n_total; i += 32u) {
            float w = expf(scores[i] - ms);
            scores[i] = w;
            ls += w;
        }
        float denom = bsum_f32(ls);
        float inv_denom = 1.0f / fmaxf(denom, 1e-10f);

        // Weighted sum of V
        float out_h[HD];
        for (uint32_t d = lane; d < HD; d += 32u) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_raw; r++) {
                uint32_t kr = (raw_start + r) % raw_cap;
                acc += scores[r] * raw_kv[((uint64_t)kr * NH + h) * HD + d];
            }
            if constexpr (CR > 0u) {
                uint32_t cl = n_comp;
                if constexpr (USE_IDX) { cl = (topk_cnt < n_comp) ? topk_cnt : n_comp; }
                for (uint32_t c = 0; c < cl; c++) {
                    uint32_t ci = c;
                    if constexpr (USE_IDX) ci = (uint32_t)topk[c];
                    float w = scores[n_raw + c];
                    acc += w * comp_kv[((uint64_t)ci * NH + h) * HD + d];
                }
            }
            out_h[d] = acc * inv_denom;
        }

        // Write head output
        half* hh = h_buf + h * HD;
        for (uint32_t d = lane; d < HD; d += 32u) {
            hh[d] = __float2half(out_h[d]);
        }
        __syncthreads();
    }
}

// ===========================================================================
// 3. hyper_connection — 4-stream residual w/ Sinkhorn routing
// ===========================================================================
// DS4 hierarchical composition: 4 hidden streams merged via per-token
// routing (Sinkhorn-normalized 4x4 + pre/post scales).
// scratch[0..23] = hc split coefficients (f32)
//
// Reference: ds4_rocm_hc.cuh (hc4_split_one, hc_expand4_kernel)

__device__ void hyper_connection(
    half* h_buf,
    half* scratch,
    const void* hc_weight,
    uint tid,
    uint tg_sz)
{
    constexpr uint32_t DIM  = HIDDEN_DIM;
    constexpr float    EPSV = 1e-6f;

    const float* mix  = (const float*)hc_weight;       // [24]
    const float* base = (const float*)hc_weight + 24u; // [24]
    const half*  hc_res = (const half*)(base + 24u);   // [4 * DIM]

    float* split = (float*)scratch;  // 24 floats

    // Thread 0 computes Sinkhorn coefficients
    if (tid == 0) {
        float out[24], c[16];

        for (int i = 0; i < 4; i++)
            out[i] = 1.0f / (1.0f + expf(-mix[i] - base[i])) + EPSV;
        for (int i = 0; i < 4; i++)
            out[4 + i] = 2.0f / (1.0f + expf(-mix[4 + i] - base[4 + i]));

        for (int r = 0; r < 4; r++) {
            float m = -3.4e38f;
            for (int col = 0; col < 4; col++) {
                float v = mix[8 + r * 4 + col] + base[8 + r * 4 + col];
                c[r * 4 + col] = v;
                m = fmaxf(m, v);
            }
            float s = 0.0f;
            for (int col = 0; col < 4; col++) { c[r * 4 + col] = expf(c[r * 4 + col] - m); s += c[r * 4 + col]; }
            for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + EPSV;
        }
        for (int col = 0; col < 4; col++) { float s = EPSV; for (int r = 0; r < 4; r++) s += c[r * 4 + col]; for (int r = 0; r < 4; r++) c[r * 4 + col] /= s; }
        for (int r = 0; r < 4; r++) { float s = EPSV; for (int col = 0; col < 4; col++) s += c[r * 4 + col]; for (int col = 0; col < 4; col++) c[r * 4 + col] /= s; }
        for (int col = 0; col < 4; col++) { float s = EPSV; for (int r = 0; r < 4; r++) s += c[r * 4 + col]; for (int r = 0; r < 4; r++) c[r * 4 + col] /= s; }

        for (int i = 0; i < 16; i++) out[8 + i] = c[i];
        for (int i = 0; i < 24; i++) split[i] = out[i];
    }
    __syncthreads();

    // Weighted blend
    const float* post = split + 4u;
    for (uint32_t d = tid; d < DIM; d += tg_sz) {
        float mv = __half2float(h_buf[d]);
        float a = 0.0f;
        for (uint32_t s = 0; s < 4u; s++) {
            a += __half2float(hc_res[s * DIM + d]) * post[s];
        }
        h_buf[d] = __float2half(mv + a);
    }
}

// ===========================================================================
// 4. moe_router<N_EXPERT, TOP_K>
// ===========================================================================
// Router logits -> softplus -> sqrt -> top-6 -> normalize.
// Results stored in scratch for routed_expert_matmul to consume:
//   scratch[0..N_EXPERT-1]            f32 = per-expert probabilities
//   scratch[N_EXPERT..+TOP_K-1]       int32 = selected expert IDs
//   scratch[N_EXPERT+TOP_K..]          f32 = per-expert weights

template <uint N_EXP, uint TOP>
__device__ void moe_router(
    half* h_buf,
    half* scratch,
    const void* router_weight,
    uint tid,
    uint tg_sz)
{
    constexpr uint32_t DIM = HIDDEN_DIM;
    constexpr uint32_t GRS = (N_EXP + 31u) / 32u;

    uint32_t lane = tid & 31u;
    const float* rw = (const float*)router_weight;

    // scratch: logits broadcast buffer (N_EXP f32, also stores results)
    float* logits = (float*)scratch;

    // Per-lane top-K state (GRS <= 8 for N_EXP <= 256)
    float lprob[8];
    for (uint32_t g = 0; g < GRS; g++) {
        uint32_t e = lane + g * 32u;
        if (e < N_EXP) {
            float sum = 0.0f;
            const float* row = rw + (uint64_t)e * DIM;
            for (uint32_t d = 0; d < DIM; d++)
                sum += __half2float(h_buf[d]) * row[d];
            float sp = (sum > 20.0f) ? sum
                      : ((sum < -10.0f) ? expf(sum) : log1pf(expf(sum)));
            lprob[g] = sqrtf(sp);
            logits[e] = lprob[g];
        } else {
            lprob[g] = -3.4e38f;
        }
    }

    // Top-K via warp shuffle
    auto better = [](float av, uint32_t ai, float bv, uint32_t bi) -> bool {
        return av > bv || (av == bv && ai < bi);
    };

    uint32_t out_idx[TOP] = {0};
    float    out_prob[TOP] = {0.0f};

    for (uint32_t k = 0; k < TOP; k++) {
        float bs = -3.4e38f, bp = 0.0f;
        uint32_t bi = ~0u;
        for (uint32_t g = 0; g < GRS; g++) {
            uint32_t e = lane + g * 32u;
            if (e < N_EXP && better(lprob[g], e, bs, bi)) { bs = lprob[g]; bp = lprob[g]; bi = e; }
        }
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
            float os = __shfl_xor(bs, mask, 32);
            float ob = __shfl_xor(bp, mask, 32);
            uint32_t oi = __shfl_xor(bi, mask, 32);
#else
            float os = __shfl_xor_sync(0xFFFFFFFF, bs, mask, 32);
            float ob = __shfl_xor_sync(0xFFFFFFFF, bp, mask, 32);
            uint32_t oi = __shfl_xor_sync(0xFFFFFFFF, bi, mask, 32);
#endif
            if (better(os, oi, bs, bi)) { bs = os; bp = ob; bi = oi; }
        }
        for (uint32_t g = 0; g < GRS; g++) {
            uint32_t e = lane + g * 32u;
            if (e == bi) lprob[g] = -3.4e38f;
        }
        if (lane == 0) { out_idx[k] = bi; out_prob[k] = bp; }
    }

    if (lane == 0) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < TOP; k++) sum += out_prob[k];
        sum = fmaxf(sum, 1e-10f);
        for (uint32_t k = 0; k < TOP; k++) out_prob[k] /= sum;
    }

    int32_t* sel = (int32_t*)((float*)scratch + N_EXP);
    float*   wgt = (float*)(sel + TOP);
    for (uint32_t i = tid; i < TOP; i += tg_sz) {
        sel[i] = (int32_t)out_idx[i];
        wgt[i] = out_prob[i];
    }
    __syncthreads();
}

// ===========================================================================
// 5. shared_expert_mlp — Q8_0 gate+up fused SwiGLU
// ===========================================================================
// Quantizes h_buf -> Q8_0, fused gate+up matmul (tiled), SiLU(gate)*up.
// TILED: scratch can only hold ~8K floats (32 KB); SHARED_FF=16384 won't fit.
// Process rows in tiles of TILE_SIZE (2048) — 8 tiles cover SHARED_FF=16384.
//
// scratch layout:
//   [0..blocks*32-1]           int8   = Q8_0 quantized activations (4 KB)
//   [blocks*32..+blocks]       float  = per-block scales (128*4=512 B)
//   [+blocks+pad..+TILE_SIZE]  float  = gate output (tile)
//   [+TILE_SIZE..+2*TILE]      float  = up output (tile)
//   Total: ~4.5 KB + 2*TILE*4 <= 32 KB; TILE=2048 fits (4.5+16=20.5 KB).

__device__ void shared_expert_mlp(
    half* h_buf,
    half* scratch,
    const void* shared_weight,
    uint tid,
    uint tg_sz)
{
    constexpr uint32_t DIM = HIDDEN_DIM;
    constexpr uint32_t FF  = SHARED_FF;
    const uint32_t blocks  = (DIM + 31u) / 32u;
    uint32_t lane = tid & 31u;

    int8_t* xq     = (int8_t*)scratch;
    float*  xscale = (float*)(xq + blocks * 32u);

    const uint8_t* gate_w = (const uint8_t*)shared_weight;
    const uint8_t* up_w   = gate_w + (uint64_t)FF * blocks * 36u;

    float* gate_out = (float*)(xscale + blocks);

    // --- Quantize h_buf to Q8_0 ---
    for (uint32_t b = lane; b < blocks; b += 32u) {
        uint32_t i0 = b * 32u;
        uint32_t bn = (DIM - i0 < 32u) ? (DIM - i0) : 32u;
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++)
            amax = fmaxf(amax, fabsf(__half2float(h_buf[i0 + i])));
        amax = wmax_f32(amax);
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        float d = __shfl(amax, 0, 32) / 127.0f;
#else
        float d = __shfl_sync(0xFFFFFFFF, amax, 0, 32) / 127.0f;
#endif
        float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        if (lane == 0) xscale[b] = d;
        int8_t* dst = xq + b * 32u;
        for (uint32_t i = lane; i < bn; i += 32u) {
            dst[i] = (int8_t)clip_i8((int)lrintf(__half2float(h_buf[i0 + i]) * id));
        }
        uint32_t tail0 = (bn > lane) ? bn : lane;
        for (uint32_t i = tail0; i < 32u; i += 32u) dst[i] = 0;
    }
    __syncthreads();

    // --- Fused gate+up matmul (tiled, warp-level) ---
    uint32_t warp  = tid >> 5u;
    uint32_t nw    = (tg_sz + 31u) >> 5u;
    constexpr uint32_t RPB = 8u;
    constexpr uint32_t TILE_SIZE = 2048u;
    float* tgate = (float*)(xscale + blocks);
    float* tup   = tgate + TILE_SIZE;

    for (uint32_t ts = 0u; ts < FF; ts += TILE_SIZE) {
        uint32_t te = ts + TILE_SIZE;
        if (te > FF) te = FF;
        uint32_t tr = te - ts;

        // Matmul (gate + up) for this tile
        for (uint32_t rb = warp * RPB; rb < tr; rb += nw * RPB) {
            for (uint32_t rr = 0; rr < RPB; rr++) {
                uint32_t row = ts + rb + rr;
                if (row >= FF) break;
                float ag = 0.0f, au = 0.0f;
                for (uint32_t b = lane; b < blocks; b += 32u) {
                    const uint8_t* gb = gate_w + ((uint64_t)row * blocks + b) * 36u;
                    const uint8_t* ub = up_w   + ((uint64_t)row * blocks + b) * 36u;
                    const int8_t* xb = xq + b * 32u;
                    ag += q8_0_scale(gb) * xscale[b] * (float)dot_i8x32((const int8_t*)(gb + 4u), xb);
                    au += q8_0_scale(ub) * xscale[b] * (float)dot_i8x32((const int8_t*)(ub + 4u), xb);
                }
                ag = wsum_f32(ag);
                au = wsum_f32(au);
                if (lane == 0) {
                    tgate[rb + rr] = ag;
                    tup[rb + rr]   = au;
                }
            }
        }
        __syncthreads();

        // SwiGLU for this tile, accumulate to h_buf (first DIM rows)
        for (uint32_t i = tid; i < tr; i += tg_sz) {
            uint32_t ai = ts + i;
            if (ai >= DIM) break;
            float g = tgate[i];
            float u = tup[i];
            h_buf[ai] = __float2half(__half2float(h_buf[ai])
                         + (g / (1.0f + expf(-g))) * u);
        }
    }
}

// ===========================================================================
// 6. routed_expert_matmul — per-expert Q4_K/Q2_K matmul
// ===========================================================================
// One selected expert: Q4_K gate+up -> SwiGLU -> Q2_K down -> h_buf accum.
// Batch-4 WMMA path (USE_WMMA=1): processes 4 tokens via rocwmma 16x16 tiles.
// Scalar path (USE_WMMA=0): per-token Q8_K quant + GEMV.
//
// scratch layout:
//   [0..N_EXPERT+2*TOP_K]           = router output (preserved)
//   [ROUTER_SKIP..+xq_sb*sizeof(Q8K)] = Q8_K quant (scalar) OR tile buf (WMMA)
//   [after..]                        = mid_out[MID] f32

__device__ void routed_expert_matmul(
    half* h_buf,
    half* scratch,
    const void* expert_weights,
    uint32_t expert_id,
    uint tid,
    uint tg_sz)
{
    constexpr uint32_t DIM = HIDDEN_DIM;
    constexpr uint32_t MID = EXPERT_MID_DIM;
    uint32_t lane = tid & 31u;

    // expert_id is the slot index (0..TOP_K-1); the actual expert index
    // was written by moe_router into scratch[N_EXPERT + slot]
    int32_t actual_id = ((const int32_t*)((const float*)scratch + N_EXPERT))[expert_id];
    if (actual_id < 0) return;                      // invalid expert
    uint32_t eid = (uint32_t)actual_id;
    float weight = ((const float*)((const int32_t*)((const float*)scratch + N_EXPERT) + TOP_K))[expert_id];

    const uint32_t xq_sb    = (DIM + QK_K - 1u) / QK_K;
    const uint32_t q4k_sb   = (DIM + QK_K - 1u) / QK_K;
    const uint32_t q2k_sb   = (MID + QK_K - 1u) / QK_K;

    // Router data occupies scratch[0..N_EXPERT*4+TOP_K*8] bytes.
    // Start xq/mid_out AFTER router data to avoid overwriting.
    constexpr uint32_t ROUTER_SKIP = ((N_EXPERT * 4 + TOP_K * 8 + 255u) / 256u) * 256u;
    char* scratch_bytes = (char*)scratch;

    const uint64_t q4k_rb = (uint64_t)q4k_sb * sizeof(block_q4_K);
    const uint64_t q2k_rb = (uint64_t)q2k_sb * sizeof(block_q2_K);
    const uint64_t ge_b   = (uint64_t)MID * q4k_rb;
    const uint64_t ue_b   = (uint64_t)MID * q4k_rb;
    const uint64_t de_b   = (uint64_t)DIM * q2k_rb;

    const char* base = (const char*)expert_weights;
    const block_q4_K* gate = (const block_q4_K*)(base + (uint64_t)eid * ge_b);
    uint64_t gate_tab = (uint64_t)N_EXPERT * ge_b;
    const block_q4_K* up   = (const block_q4_K*)(base + gate_tab + (uint64_t)eid * ue_b);
    uint64_t up_tab   = (uint64_t)N_EXPERT * ue_b;
    const block_q2_K* down = (const block_q2_K*)(base + gate_tab + up_tab + (uint64_t)eid * de_b);


    // Scalar path: Q8_K quant of h_buf + Q4_K/Q2_K GEMV.
    // When {{USE_WMMA}}=1, the AOT code generator should replace this section
    // with a batch-4 rocwmma implementation: dequant weight tiles -> half tile
    // buffer in scratch, then wmma::mma_sync for [4xDIM]x[MIDxDIM]^T matmuls.

    block_q8_K* xq = (block_q8_K*)(scratch_bytes + ROUTER_SKIP);

    for (uint32_t sb = lane; sb < xq_sb; sb += 32u) {
        uint32_t i0 = sb * QK_K;
        uint32_t bn = (DIM - i0 < QK_K) ? (DIM - i0) : QK_K;
        float amax = 0.0f;
        for (uint32_t i = 0; i < bn; i++)
            amax = fmaxf(amax, fabsf(__half2float(h_buf[i0 + i])));
        amax = wmax_f32(amax);
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
        float max_v = __shfl(amax, 0, 32);
#else
        float max_v = __shfl_sync(0xFFFFFFFF, amax, 0, 32);
#endif
        block_q8_K* blk = &xq[sb];
        if (max_v == 0.0f) {
            if (lane == 0) blk->d = 0.0f;
            for (uint32_t i = lane; i < QK_K; i += 32u) blk->qs[i] = 0;
            for (uint32_t i = lane; i < 16u; i += 32u) blk->bsums[i] = 0;
        } else {
            float isc = -127.0f / max_v;
            if (lane == 0) blk->d = __float2half(1.0f / isc);
            for (uint32_t i = lane; i < bn; i += 32u) {
                blk->qs[i] = (int8_t)clip_i8((int)lrintf(isc * __half2float(h_buf[i0 + i])));
            }
            uint32_t tail1 = (bn > lane) ? bn : lane;
            for (uint32_t i = tail1; i < QK_K; i += 32u) blk->qs[i] = 0;
            __syncthreads();
            if (lane < 16u) {
                int s = 0;
                for (int i = 0; i < 16; i++) s += blk->qs[lane * 16 + i];
                blk->bsums[lane] = (int16_t)s;
            }
        }
    }
    __syncthreads();

    // --- Gate + Up matmul -> f32 mid_out ---
    float* mid_out = (float*)(xq + xq_sb);
    for (uint32_t row = tid; row < MID; row += tg_sz) {
        float g = 0.0f, u = 0.0f;
        for (uint32_t sb = 0; sb < q4k_sb; sb++) {
            g += dot_q4K_q8K(&gate[(uint64_t)row * (uint64_t)q4k_sb + sb], &xq[sb]);
            u += dot_q4K_q8K(&up[(uint64_t)row * (uint64_t)q4k_sb + sb],   &xq[sb]);
        }
        mid_out[row] = (g / (1.0f + expf(-g))) * u;
    }
    __syncthreads();

    const uint32_t q2k_sb_mid = (MID + QK_K - 1u) / QK_K;
    for (uint32_t row = tid; row < DIM; row += tg_sz) {
        float acc = 0.0f;
        for (uint32_t sb = 0; sb < q2k_sb_mid; sb++) {
            const block_q2_K* w = &down[(uint64_t)row * (uint64_t)q2k_sb_mid + sb];
            uint32_t i0 = sb * QK_K;
            uint32_t bn = (MID - i0 < QK_K) ? (MID - i0) : QK_K;
            float d2 = __half2float(w->d);
            float dm = __half2float(w->dmin);
            const uint8_t* qs = w->qs;
            const uint8_t* sc = w->scales;
            for (uint32_t j = 0; j < 16 && (j * 16u) < bn; j++) {
                uint32_t joff = j * 16u;
                int s = (int)(sc[j] & 0x0f);
                uint32_t qb = j / 4;
                uint32_t qi = j % 4;
                for (uint32_t k = 0; k < 16; k++) {
                    uint32_t idx = i0 + joff + k;
                    if (idx >= MID) break;
                    int qv = (qs[qb * 8 + k] >> (qi * 2)) & 3;
                    acc += (d2 * (float)qv - dm) * mid_out[idx];
                }
            }
        }
        h_buf[row] = __float2half(__half2float(h_buf[row]) + acc * weight);
    }
}

// ===========================================================================
// 7. residual_add
// ===========================================================================
// h_buf[tid] += scratch[tid] for all HIDDEN_DIM elements.

__device__ void residual_add(
    half* h_buf,
    half* scratch,
    uint tid,
    uint tg_sz)
{
    for (uint32_t i = tid; i < HIDDEN_DIM; i += tg_sz) {
        h_buf[i] = __float2half(
            __half2float(h_buf[i]) + __half2float(scratch[i]));
    }
}
