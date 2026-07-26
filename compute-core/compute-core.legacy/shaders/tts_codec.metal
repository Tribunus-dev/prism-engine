#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════
// Mimi Codec — causal ConvNet decoder for 24 kHz PCM synthesis
// Part of Qwen3-TTS (Apache 2.0 licensed)
// ═══════════════════════════════════════════════════════════════════════════
//
// Architecture overview:
//   codebook_gather  →  conv1d_transpose (×N upsampling)  →  residual blocks
//   (causal_conv1d + seq_layernorm)  →  overlap_add
//
// Each codebook has 2048 entries, 128 dims each, 16 codebooks total.
// Tokens arrive at 12.5 Hz and are upsampled to 24 kHz.

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 1: codebook_gather
// ═══════════════════════════════════════════════════════════════════════════
//
// Gather 128-dim embeddings for each of 16 codebooks per token.
//
// Inputs:
//   indices  [num_tokens × 16]           — u32 codebook indices per token
//   codebook [16][2048][128]             — half-precision embedding table
// Output:
//   output   [num_tokens][16][128]       — gathered embeddings
//
// Grid: num_tokens × 16 × 128
//
kernel void codebook_gather(
    device const uint*  indices       [[buffer(0)]],
    device const half*  codebook      [[buffer(1)]],
    device half*        output        [[buffer(2)]],
    constant uint&      num_tokens    [[buffer(3)]],
    uint3               pos           [[thread_position_in_grid]]
) {
    uint token_idx  = pos.x;
    uint cb_idx     = pos.y;
    uint dim_idx    = pos.z;

    if (token_idx >= num_tokens || cb_idx >= 16 || dim_idx >= 128) { return; }

    // Read the codebook index for this token + codebook
    uint entry = indices[token_idx * 16 + cb_idx];

    // Clamp to valid range [0, 2047]
    entry = min(entry, 2047u);

    // Compute offset into codebook table: [cb_idx][entry][dim_idx]
    uint codebook_offset = cb_idx * 2048 * 128 + entry * 128 + dim_idx;

    output[token_idx * 16 * 128 + cb_idx * 128 + dim_idx] = codebook[codebook_offset];
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 2: conv1d_transpose
// ═══════════════════════════════════════════════════════════════════════════
//
// 1D transposed convolution for upsampling.
// Input frames:    L_in
// Output frames:   L_out = (L_in - 1) * stride + kernel_size
//
// Each thread computes one output element at [out_frame, out_channel].
//
// Output[p, oc] = Σ_ic Σ_i weight[oc, ic, i] * input[in_idx, ic]
//   where in_idx = (p - i) / stride when (p - i) >= 0 and (p - i) % stride == 0
//
kernel void conv1d_transpose(
    device const half*  input          [[buffer(0)]],   // [L_in][in_ch]
    device const half*  weight         [[buffer(1)]],   // [out_ch][in_ch][kernel]
    device const half*  bias           [[buffer(2)]],   // [out_ch] or null
    device half*        output         [[buffer(3)]],   // [L_out][out_ch]
    constant uint&      in_channels    [[buffer(4)]],
    constant uint&      out_channels   [[buffer(5)]],
    constant uint&      stride         [[buffer(6)]],
    constant uint&      kernel_size    [[buffer(7)]],
    constant uint&      input_frames   [[buffer(8)]],
    uint                tid            [[thread_position_in_grid]]
) {
    uint output_frames = (input_frames - 1) * stride + kernel_size;
    uint total = output_frames * out_channels;
    if (tid >= total) { return; }

    uint oc  = tid % out_channels;
    uint p   = tid / out_channels;   // output position

    float accum = 0.0f;

    // Iterate over kernel indices i where (p - i) >= 0 and (p - i) % stride == 0
    uint i_start = (p < kernel_size) ? 0 : (p - kernel_size + 1);
    uint i_min = (p >= kernel_size) ? (p - kernel_size + 1) : 0;

    for (uint i = 0; i < kernel_size; ++i) {
        uint q = p - i;
        // When (q >= 0) and (q % stride == 0), the input index is q / stride
        if (q % stride == 0) {
            uint in_idx = q / stride;
            if (in_idx < input_frames) {
                uint weight_base = oc * in_channels * kernel_size;
                for (uint ic = 0; ic < in_channels; ++ic) {
                    uint weight_off = weight_base + ic * kernel_size + i;
                    accum += (float)input[in_idx * in_channels + ic] *
                             (float)weight[weight_off];
                }
            }
        }
    }

    accum += (float)(bias ? bias[oc] : (half)0.0f);
    output[tid] = (half)accum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 3: causal_conv1d
// ═══════════════════════════════════════════════════════════════════════════
//
// Causal dilated 1D convolution for residual blocks.
// Output[t] depends only on input[t - i*dilation] for i = 0..(kernel_size-1)
// and t - i*dilation >= 0.
//
// Each thread computes one element at [frame, channel].
//
kernel void causal_conv1d(
    device const half*  input          [[buffer(0)]],   // [frames][channels]
    device const half*  weight         [[buffer(1)]],   // [out_ch][in_ch][kernel]
    device const half*  bias           [[buffer(2)]],   // [out_ch] or null
    device half*        output         [[buffer(3)]],   // [frames][out_ch]
    constant uint&      channels       [[buffer(4)]],   // in_channels (same as out_ch for residual)
    constant uint&      kernel_size    [[buffer(5)]],
    constant uint&      dilation       [[buffer(6)]],
    constant uint&      frames         [[buffer(7)]],
    uint                tid            [[thread_position_in_grid]]
) {
    uint total = frames * channels;
    if (tid >= total) { return; }

    uint ch = tid % channels;
    uint t  = tid / channels;   // time step

    float accum = 0.0f;

    // Accumulate causal window: input[t - i*dilation] * weight[ch, i] for i=0..k-1
    // Weight layout: [out_ch][in_ch][kernel] but since in_ch == out_ch for residual:
    uint weight_base = ch * channels * kernel_size;
    for (uint i = 0; i < kernel_size; ++i) {
        int src = (int)t - (int)(i * dilation);
        if (src >= 0) {
            uint src_pos = (uint)src;
            for (uint ic = 0; ic < channels; ++ic) {
                uint weight_off = weight_base + ic * kernel_size + i;
                accum += (float)input[src_pos * channels + ic] *
                         (float)weight[weight_off];
            }
        }
    }

    if (bias) { accum += (float)bias[ch]; }
    output[tid] = (half)accum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 4: seq_layernorm
// ═══════════════════════════════════════════════════════════════════════════
//
// Layer normalization along the feature dimension.
//
// output[t, f] = (input[t, f] - mean[t]) / sqrt(var[t] + eps) * weight[f] + bias[f]
//
// Grid: frames × 1
//
kernel void seq_layernorm(
    device const half*  input          [[buffer(0)]],   // [frames][dim]
    device const half*  weight         [[buffer(1)]],
    device const half*  bias           [[buffer(2)]],
    device half*        output         [[buffer(3)]],
    constant uint&      dim            [[buffer(4)]],
    constant uint&      frames         [[buffer(5)]],
    constant float&     eps            [[buffer(6)]],
    uint                tid            [[thread_position_in_grid]]
) {
    if (tid >= frames) { return; }

    uint base = tid * dim;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        sum += (float)input[base + i];
    }
    float mean = sum / (float)dim;

    // Compute variance
    float var = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        float diff = (float)input[base + i] - mean;
        var += diff * diff;
    }
    float rvar = 1.0f / sqrt(var / (float)dim + eps);

    // Normalize and write
    for (uint i = 0; i < dim; ++i) {
        float w = (float)(weight ? weight[i] : (half)1.0f);
        float b = (float)(bias   ? bias[i]   : (half)0.0f);
        float norm = ((float)input[base + i] - mean) * rvar;
        output[base + i] = (half)(norm * w + b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 5: overlap_add
// ═══════════════════════════════════════════════════════════════════════════
//
// Overlap-add for streaming waveform synthesis.
// Each frame contributes frame_size samples, offset by frame_idx * hop_size.
//
// Example usage: a sliding window synthesis where frames overlap.
//
// Grid: output_samples (the total output length)
//
kernel void overlap_add(
    device const half*  frames         [[buffer(0)]],   // [num_frames][frame_size]
    device atomic_float* output        [[buffer(1)]],   // [output_samples] (atomic accumulation)
    device float*        window        [[buffer(2)]],   // [frame_size] window function
    constant uint&      num_frames     [[buffer(3)]],
    constant uint&      hop_size       [[buffer(4)]],
    constant uint&      frame_size     [[buffer(5)]],
    constant uint&      output_samples [[buffer(6)]],
    uint                tid            [[thread_position_in_grid]]
) {
    if (tid >= num_frames) { return; }

    uint base = tid * frame_size;
    uint out_offset = tid * hop_size;

    for (uint i = 0; i < frame_size; ++i) {
        uint out_pos = out_offset + i;
        if (out_pos < output_samples) {
            float val = (float)frames[base + i];
            if (window) {
                val *= (float)window[i];
            }
            atomic_fetch_add_explicit(&output[out_pos], val, memory_order_relaxed);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 6: ifft_synthesis
// ═══════════════════════════════════════════════════════════════════════════
//
// IFFT-based synthesis from frequency-domain features.
// Computes an inverse real FFT to produce waveform samples.
//
// Uses a naive O(N²) IFFT — suitable only for small FFT sizes
// (typical for neural vocoder frames). For large FFT sizes, use
// the Metal vImage/Accelerate IFFT instead.
//
// Grid: num_frames × (fft_size / 2 + 1)
//
// Inputs:
//   real     [num_frames][fft_size/2 + 1]  — real parts
//   imag     [num_frames][fft_size/2 + 1]  — imaginary parts
// Output:
//   output   [num_frames][fft_size]         — time-domain waveform
//
kernel void ifft_synthesis(
    device const half*  real           [[buffer(0)]],
    device const half*  imag           [[buffer(1)]],
    device float*       output         [[buffer(2)]],
    constant uint&      num_frames     [[buffer(3)]],
    constant uint&      fft_size       [[buffer(4)]],
    uint2               pos            [[thread_position_in_grid]]
) {
    uint frame = pos.y;
    uint k      = pos.x;   // frequency bin, 0..fft_size/2

    uint num_bins = fft_size / 2 + 1;
    if (frame >= num_frames || k >= num_bins) { return; }

    uint base = frame * fft_size;

    // Contribution of bin k to all output samples n = 0..fft_size-1
    // X[n] = (1/N) * Σ_{k=0}^{N-1} (real[k]*cos(2πkn/N) - imag[k]*sin(2πkn/N))
    //
    // For real input: X[N-k] = conj(X[k]), so we only store bins 0..N/2.
    // For k=0 and k=N/2 (if N even), imag[k] should be 0.

    float re = (float)real[frame * num_bins + k];
    float im = (float)imag[frame * num_bins + k];
    float twopi = 6.283185307179586f;
    float inv_n = 1.0f / (float)fft_size;

    // Accumulate this bin's contribution to all output samples
    // (each thread writes only its bin's partial sum into the output buffer)
    for (uint n = 0; n < fft_size; ++n) {
        float angle = twopi * (float)(k * n) / (float)fft_size;
        float cos_a = cos(angle);
        float sin_a = sin(angle);
        float contrib = 2.0f * inv_n * (re * cos_a - im * sin_a);   // *2 for conjugate symmetry

        // Use atomic add since multiple bins write to the same output sample
        atomic_fetch_add_explicit(
            (device atomic_float*)(output + base + n),
            contrib,
            memory_order_relaxed
        );
    }
}
