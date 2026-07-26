// ── DS4 Flash Persistent Decode Megakernel ─────────────────────────────────
//
// Single persistent HIP kernel, one launch per decode session.
// Each threadblock polls a fixed ring-buffer slot (blockIdx.x == slot_id).
//
// Ring buffer protocol (5 × u32 per entry, RING_SIZE={{RING_SIZE}}):
//   entry[0]: state|kind  — low 2 bits = state, upper bits = token_id
//   entry[1]: seq_pos / num_prior
//   entry[2]: num_prior
//   entry[3]: kv_slot_id
//   entry[4]: reserved
//
// State encoding:
//   0 = EMPTY         (available for CPU submission)
//   1 = TOKEN_READY   (CPU → GPU, work pending)
//   2 = PROCESSING    (GPU → self, claimed)
//   3 = COMPLETED     (GPU → CPU, logits ready for readback)
//
// Shutdown: host sets shutdown_flag → 1; all threads exit at top of loop.
//
// Template placeholders (substituted by AOT compilation):
//   RING_SIZE         — ring buffer capacity (default 512)
//   VOCAB_SIZE        — vocabulary size for logits output
//   HIDDEN_DIM        — hidden dimension (default 4096)
//   THREADS_PER_TB    — threads per threadblock (default 256)
//   LAYERS            — number of transformer layers (default 43)
//   N_EXPERT          — number of MoE experts (default 256)
//   HEAD_DIM          — attention head dimension (default 128)
//   COMPRESS_RATIO_4  — compressed KV ratio-4 code path selector
//   COMPRESS_RATIO_128— compressed KV ratio-128 code path selector
//   TOP_K             — MoE top-k routing (default 6)
//   EXPERT_MID_DIM    — expert intermediate dimension (default 2560)
//
// Scratch buffer: fixed at 16 × 1024 half-elements = 32 KB
//   (h_buf = 8 KB, scratch = 32 KB → 40 KB total, within MI300X 64 KB LDS)
//
// Stub functions (defined by per-layer kernel agents):
//   embed, rms_norm, compressed_attention, hyper_connection,
//   moe_router, shared_expert_mlp, routed_expert_matmul,
//   residual_add, write_logits

#include <hip/hip_runtime.h>

// ── Ring buffer constants ──────────────────────────────────────────────────
#define RING_EMPTY       0u
#define RING_TOKEN_READY 1u
#define RING_PROCESSING  2u
#define RING_COMPLETED   3u

#define RING_SLOT_WORDS  5u

// ── Per-layer device function stubs (defined by MegakernelPerLayer agent) ──
// Each operates on __shared__ buffers h_buf/scratch with thread cooperativity
// controlled by tid and tg_sz.

// Embedding: load token embedding from device weight table into h_buf.
__device__ void embed(
    half*       h_buf,          // [HIDDEN_DIM] output
    const half* embed_weight,   // [VOCAB_SIZE × HIDDEN_DIM]
    uint        token_id,
    uint        tid,
    uint        tg_sz);

// RMSNorm: normalize h_buf in-place, write norm into buffer.
// Template parameter sets the normalization dimension.
template <uint DIM>
__device__ void rms_norm(
    half*       h_buf,          // [DIM] input/output residual
    half*       scratch,        // scratch buffer for reduction
    const void* layer_weights,  // per-layer weight bundle
    uint        tid,
    uint        tg_sz);

// Compressed KV-cache attention.
// COMPRESS_RATIO selects the ratio-code path (4 or 128).
template <uint HEAD_DIM, uint COMPRESS_RATIO>
__device__ void compressed_attention(
    half*       h_buf,          // [HIDDEN_DIM] residual stream
    half*       scratch,        // score buffer [~640 × f32]
    void*       kv_cache,       // per-layer compressed KV
    uint        layer,
    uint        seq_pos,
    uint        tid,
    uint        tg_sz);

// Hyper-connection (gated FFN bypass).
__device__ void hyper_connection(
    half*       h_buf,
    half*       scratch,        // split coefficients [24 × half]
    const void* layer_weights,
    uint        tid,
    uint        tg_sz);

// MoE router: compute expert selection logits, select top-k.
template <uint N_EXPERT, uint TOP_K>
__device__ void moe_router(
    half*       h_buf,          // [HIDDEN_DIM] router input
    half*       scratch,        // scratch for softmax/gather
    const void* layer_weights,  // per-layer weight bundle
    uint        tid,
    uint        tg_sz);

// Shared MLP (used by all experts as a common hidden projection).
__device__ void shared_expert_mlp(
    half*       h_buf,
    half*       scratch,        // Q8_0 quant + tile out (~21 KB)
    const void* layer_weights,
    uint        tid,
    uint        tg_sz);

// Per-expert matmul: apply the e-th selected expert's weights.
__device__ void routed_expert_matmul(
    half*       h_buf,
    half*       scratch,        // Q8_K quant + mid_out (~15 KB)
    const void* layer_weights,
    uint        expert_idx,
    uint        tid,
    uint        tg_sz);

// Residual addition with norm scratch.
__device__ void residual_add(
    half*       h_buf,          // output residual
    half*       scratch,        // input pre-layer hidden
    uint        tid,
    uint        tg_sz);

// Write logits from residual stream to slot output buffer.
__device__ void write_logits(
    half*       logits_out,     // [VOCAB_SIZE] output slot
    half*       h_buf,          // [HIDDEN_DIM] residual stream
    uint        tid,
    uint        tg_sz);

// ── Persistent decode kernel ───────────────────────────────────────────────
//
// Dispatch: gridDim.x = RING_SIZE (one threadblock per slot).
//           Each threadblock runs an infinite loop until shutdown.
//
// Host-side lifecycle:
//   1. hipMalloc ring_entries[RING_SIZE * 5] → zero-filled (EMPTY)
//   2. hipMemset completion_counter → 0
//   3. Launch ds4_persistent_decode<<<RING_SIZE, {{THREADS_PER_TB}}>>>(...)
//   4. For each decode step: fill ring_entries[slot], set state TOKEN_READY
//   5. Poll completion_counter; when incremented, read logits_out
//   6. Reset ring_entries[slot] → EMPTY, recycle slot
//   7. Host sets shutdown_flag → 1, hipDeviceSynchronize cleans up
//
extern "C" __global__
void ds4_persistent_decode(
    // ── Ring buffer ────────────────────────────────────────────────────────
    uint*       ring_entries,          // [RING_SIZE * RING_SLOT_WORDS]
    uint*       ring_tail,             // reserved for host-side polling

    // ── Weights (pre-loaded in VRAM at session startup) ────────────────────
    const half* embed_weight,          // [VOCAB_SIZE × HIDDEN_DIM]

    // ── KV cache (pre-allocated, per-layer) ────────────────────────────────
    void*       kv_cache,              // [LAYERS × compressed KV]

    // ── Output buffers ─────────────────────────────────────────────────────
    half*       logits_out,            // [RING_SIZE × VOCAB_SIZE]
    uint*       completion_counter,    // monotonic completion count

    // ── Shutdown signal ────────────────────────────────────────────────────
    uint*       shutdown_flag          // non-zero triggers clean exit
) {
    // ── Shared memory ──────────────────────────────────────────────────────
    // Budget: HIDDEN_DIM × 2 = 8 KB (h_buf) + 16K × 2 = 32 KB (scratch)
    //         = 40 KB  (within MI300X 64 KB LDS per CU)
    __shared__ half h_buf[{{HIDDEN_DIM}}];         // residual stream (8 KB)
    __shared__ half scratch[16 * 1024];             // general-purpose scratch (32 KB)

    // Each threadblock owns exactly one slot.
    const uint slot_offset = blockIdx.x;            // 0 .. RING_SIZE-1
    const uint slot_base   = slot_offset * RING_SLOT_WORDS;

    const uint tid   = threadIdx.x;
    const uint tg_sz = blockDim.x;

    // ── Persistent work loop ───────────────────────────────────────────────
    while (true) {
        // 1. Shutdown check — host sets shutdown_flag → 1 to exit cleanly.
        //    HIP portable atomic-load idiom (__ldg is CUDA-only).
        if (atomicAdd(shutdown_flag, 0u) != 0) {
            return;
        }

        // 2. Poll ring buffer slot for work.
        //    atomicAdd(ptr, 0) performs an atomic load in HIP device code.
        const uint entry_header = atomicAdd(&ring_entries[slot_base], 0u);
        const uint state = entry_header & 3u;

        if (state != RING_TOKEN_READY) {
            // No work available — spin (yield) and recheck.
            // Occupancy keeps the CU active; no explicit yield needed.
            continue;
        }

        // 3. Decode work token from entry fields.
        const uint token_id = entry_header >> 2u;
        // entry[1] = seq_pos per spec: [0]=state|token_id, [1]=seq_pos,
        // [2]=num_prior, [3]=kv_slot_id, [4]=reserved
        const uint seq_pos  = ring_entries[slot_base + 1u];

        // 4. Claim this slot by transitioning to PROCESSING.
        //    Prevents host from recycling the slot while GPU is computing.
        __threadfence();  // ensure seq_pos read completes before state write
        atomicExch(&ring_entries[slot_base], RING_PROCESSING);

        // ── Full decode pass ───────────────────────────────────────────────
        // Each layer's stub functions resolve their own weight pointers via a
        // device-resident weight LUT pre-loaded at session init (separate kernel
        // argument).  nullptr for layer_weights is a placeholder — the per-layer
        // agent bridges from the LUT to concrete pointers internally.

        // Stage 0: Embedding lookup
        embed(h_buf, embed_weight, token_id, tid, tg_sz);
        __syncthreads();

        // Stage 1–LAYERS: Transformer layers
        #pragma unroll
        for (uint layer = 0u; layer < {{LAYERS}}; layer++) {
            // Each layer processes:
            //   1. RMSNorm
            //   2. Compressed attention (head_dim=128, ratio={4,128})
            //   3. Hyper-connection
            //   4. MoE router → top-k expert selection
            //   5. Shared expert MLP
            //   6. Routed expert matmuls (top-k)
            //   7. Residual add

            // 1. RMSNorm
            rms_norm<{{HIDDEN_DIM}}>(h_buf, scratch, nullptr, tid, tg_sz);
            __syncthreads();

            // 2. Compressed attention (ratio varies per layer)
            {
                const uint cr = (layer & 1u) ? {{COMPRESS_RATIO_128}}
                                             : {{COMPRESS_RATIO_4}};
                if (cr == {{COMPRESS_RATIO_128}}) {
                    compressed_attention<{{HEAD_DIM}}, {{COMPRESS_RATIO_128}}>(
                        h_buf, scratch, kv_cache, layer, seq_pos, tid, tg_sz);
                } else {
                    compressed_attention<{{HEAD_DIM}}, {{COMPRESS_RATIO_4}}>(
                        h_buf, scratch, kv_cache, layer, seq_pos, tid, tg_sz);
                }
            }
            __syncthreads();

            // 3. Hyper-connection
            hyper_connection(h_buf, scratch, nullptr, tid, tg_sz);
            __syncthreads();

            // 4. MoE router → top-k expert selection
            moe_router<{{N_EXPERT}}, {{TOP_K}}>(
                h_buf, scratch, nullptr, tid, tg_sz);
            __syncthreads();

            // 5. Shared expert MLP
            shared_expert_mlp(h_buf, scratch, nullptr, tid, tg_sz);
            __syncthreads();

            // 6. Routed expert matmuls (top-k)
            #pragma unroll
            for (uint e = 0u; e < {{TOP_K}}; e++) {
                routed_expert_matmul(h_buf, scratch, nullptr, e, tid, tg_sz);
                __syncthreads();
            }

            // 7. Residual add
            residual_add(h_buf, scratch, tid, tg_sz);
            __syncthreads();
        }

        // Stage 2: Write logits to slot output region
        half* slot_logits = logits_out + slot_offset * {{VOCAB_SIZE}};
        write_logits(slot_logits, h_buf, tid, tg_sz);
        __syncthreads();

        // 5. Signal completion
        //    Order matters: logits must be globally visible before the host
        //    sees COMPLETED. __threadfence() before and after the state write
        //    ensures correct ordering.
        __threadfence();
        atomicExch(&ring_entries[slot_base], RING_COMPLETED);
        __threadfence();
        atomicAdd(completion_counter, 1u);

        // ── Spin until host recycles this slot back to EMPTY ───────────────
        // (Continues to while(true) loop top — polls ring slot for next token.)
    }
}
