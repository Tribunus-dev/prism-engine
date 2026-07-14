#include <metal_stdlib>
using namespace metal;

/// Gemma 4 attention kernel.
kernel void gemma4_attention(
    device const float *q    [[buffer(0)]],
    device const float *k    [[buffer(1)]],
    device const float *v    [[buffer(2)]],
    device float *out        [[buffer(3)]],
    constant uint &seq_len   [[buffer(4)]],
    constant uint &head_dim  [[buffer(5)]],
    uint2 pos                 [[thread_position_in_grid]]
) {
    uint h = pos.x, t = pos.y;
    float score = 0.0;
    for (uint d = 0; d < head_dim; ++d)
        score += q[h * head_dim + d] * k[t * head_dim + d];
    score = exp(score / sqrt(float(head_dim)));
    float sum = 0.0;
    for (uint s = 0; s <= t; ++s) {
        float s2 = 0.0;
        for (uint d = 0; d < head_dim; ++d)
            s2 += q[h * head_dim + d] * k[s * head_dim + d];
        sum += exp(s2 / sqrt(float(head_dim)));
    }
    out[h * (seq_len * head_dim) + t] = score / (sum + 1e-6);
}
