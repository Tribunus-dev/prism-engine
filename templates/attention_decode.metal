// Attention decode kernel — single query token, multi-head, FP16.
// Each thread handles one head: Q@K^T/sqrt(d) → softmax → @V
// Two-pass: first pass finds max score, second pass computes softmax + weighted sum.
// Uses configurable kv_stride to handle per-layer dimension differences.

#include <metal_stdlib>
using namespace metal;

kernel void attention_decode(
    device const half* q       [[buffer(0)]],  // [num_heads * head_dim]
    device const half* k       [[buffer(1)]],  // [seq_len * kv_stride]
    device const half* v       [[buffer(2)]],  // [seq_len * kv_stride]
    device half* out           [[buffer(3)]],  // [num_heads * head_dim]
    constant uint& seq_len     [[buffer(4)]],
    constant uint& num_heads   [[buffer(5)]],
    constant uint& kv_heads    [[buffer(6)]],
    constant uint& head_dim    [[buffer(7)]],
    constant uint& kv_stride   [[buffer(8)]],  // elements per token in K/V buffer
    uint3 gid [[thread_position_in_grid]]
) {
    uint h = gid.x;
    if (h >= num_heads) return;

    uint grp = num_heads / max(kv_heads, 1u);
    uint kh = h / grp;
    uint stride = kv_stride > 0 ? kv_stride : kv_heads * head_dim;
    uint hd = head_dim;
    float inv_sqrt_d = 1.0 / sqrt(float(hd));

    // Pass 1: find max score
    float max_score = -INFINITY;
    for (uint t = 0; t < seq_len; t++) {
        float s = 0.0;
        uint base = t * stride + kh * hd;
        for (uint d = 0; d < hd; d++) {
            s += float(q[h * hd + d]) * float(k[base + d]);
        }
        s *= inv_sqrt_d;
        if (s > max_score) max_score = s;
    }

    // Pass 2: softmax denominator + weighted sum of V (recompute scores)
    float sum = 0.0;
    float acc[256];
    for (uint d = 0; d < hd; d++) acc[d] = 0.0;

    for (uint t = 0; t < seq_len; t++) {
        float s = 0.0;
        uint base = t * stride + kh * hd;
        for (uint d = 0; d < hd; d++) {
            s += float(q[h * hd + d]) * float(k[base + d]);
        }
        s = exp(s * inv_sqrt_d - max_score);
        sum += s;
        for (uint d = 0; d < hd; d++) {
            acc[d] += s * float(v[base + d]);
        }
    }

    float inv_sum = 1.0 / sum;
    for (uint d = 0; d < hd; d++) {
        out[h * hd + d] = half(acc[d] * inv_sum);
    }
}
