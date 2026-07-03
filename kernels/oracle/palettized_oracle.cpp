// palettized_oracle.cpp — CPU reference + self-test for the fused palettized
// GEMV kernel. This defines the CANONICAL unpack + LUT + MAC semantics that
// every GPU port (CUDA / HIP / SYCL) must reproduce bit-for-bit in structure.
//
// ABI (identical to templates/palettized_gemv.metal):
//   input_vector   [in_dim]              fp (half on GPU; float here)
//   codebook_block [out_dim * 16]        16-entry LUT per output row
//   indices_block  [out_dim * in_dim/2]  4-bit indices, 2 per byte, LSB = even element
//   output_vector  [out_dim]
//   8 nibbles are consumed per 32-bit word: element (8w+j) uses nibble j,
//   where nibble j = (word >> (4*j)) & 0xF, j = 0..7  (little-endian byte order).
//
// Build & run:  g++ -O2 -std=c++17 palettized_oracle.cpp -o oracle && ./oracle
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
using namespace std;

// Straightforward, obviously-correct reference: dequantize each weight and dot.
static void reference_gemv(const float* x, const float* cb, const uint8_t* idx,
                           float* y, uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t o = 0; o < out_dim; ++o) {
        const float* row_cb = cb + o * 16;
        const uint8_t* row_idx = idx + o * (in_dim / 2);
        float acc = 0.0f;
        for (uint32_t i = 0; i < in_dim; ++i) {
            uint8_t byte = row_idx[i / 2];
            uint32_t nib = (i & 1) ? (byte >> 4) & 0xF : byte & 0xF; // LSB = even element
            acc += x[i] * row_cb[nib];
        }
        y[o] = acc;
    }
}

// Kernel-equivalent: replicates the exact word/shift unpack the GPU kernels use.
static void kernel_equiv_gemv(const float* x, const float* cb, const uint8_t* idx,
                              float* y, uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t o = 0; o < out_dim; ++o) {
        const float* row_cb = cb + o * 16;
        const uint32_t* idx_ptr =
            reinterpret_cast<const uint32_t*>(idx + o * (in_dim / 2));
        uint32_t num_words = in_dim / 8;
        float acc = 0.0f;
        for (uint32_t w = 0; w < num_words; ++w) {
            uint32_t packed = idx_ptr[w];
            uint32_t off = w * 8;
            acc += x[off+0] * row_cb[ packed        & 0xF]
                 + x[off+1] * row_cb[(packed >>  4) & 0xF]
                 + x[off+2] * row_cb[(packed >>  8) & 0xF]
                 + x[off+3] * row_cb[(packed >> 12) & 0xF]
                 + x[off+4] * row_cb[(packed >> 16) & 0xF]
                 + x[off+5] * row_cb[(packed >> 20) & 0xF]
                 + x[off+6] * row_cb[(packed >> 24) & 0xF]
                 + x[off+7] * row_cb[(packed >> 28) & 0xF];
        }
        y[o] = acc;
    }
}

int main() {
    const uint32_t in_dim = 2048, out_dim = 512; // in_dim multiple of 8
    mt19937 rng(1234);
    uniform_real_distribution<float> uf(-1.f, 1.f);
    uniform_int_distribution<int> ub(0, 255);

    vector<float> x(in_dim), cb(out_dim * 16), yref(out_dim), yker(out_dim);
    vector<uint8_t> idx(out_dim * (in_dim / 2));
    for (auto& v : x)  v = uf(rng);
    for (auto& v : cb) v = uf(rng);
    for (auto& b : idx) b = (uint8_t)ub(rng);

    reference_gemv(x.data(), cb.data(), idx.data(), yref.data(), in_dim, out_dim);
    kernel_equiv_gemv(x.data(), cb.data(), idx.data(), yker.data(), in_dim, out_dim);

    double max_abs = 0.0;
    for (uint32_t o = 0; o < out_dim; ++o)
        max_abs = fmax(max_abs, fabs((double)yref[o] - (double)yker[o]));

    printf("out_dim=%u in_dim=%u  max|ref - kernel_equiv| = %.3e\n", out_dim, in_dim, max_abs);
    bool ok = max_abs < 1e-3;
    printf("%s\n", ok ? "PASS: unpack/LUT/MAC convention verified" : "FAIL");
    return ok ? 0 : 1;
}
