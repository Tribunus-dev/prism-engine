// [[kernel]] calibration_forward_pass — FP16 forward pass with activation capture
//
// Used during PT2-LLM calibration to generate intermediate layer activations.
// Standard simdgroup matrix multiply-accumulate (identical to live inference).
// The key difference: writes the output activation to two destinations:
//   1. Next layer's input buffer (standard forward path)
//   2. A secondary MTLStorageModeShared buffer that the CPU reads via MTLSharedEvent
//
// buffer(0): input_activations  [M * K] half
// buffer(1): weight_matrix      [N * K] half
// buffer(2): output_activations [M * N] half  (next layer input — GPU-private)
// buffer(3): captured_output    [M * N] half  (shared — mapped for CPU read)
// buffer(4): M uint, K uint, N uint

#include <metal_stdlib>
using namespace metal;

kernel void calibration_forward_pass(
    device const half*  input_act  [[buffer(0)]],
    device const half*  weights    [[buffer(1)]],
    device half*        output_act [[buffer(2)]],
    device half*        captured   [[buffer(3)]],
    constant uint&      M          [[buffer(4)]],
    constant uint&      K          [[buffer(5)]],
    constant uint&      N          [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]])
{
    uint row = gid.x;
    uint col = gid.y;
    if (row >= M || col >= N) return;

    half acc = 0.0h;
    for (uint k = 0; k < K; k++) {
        acc += input_act[row * K + k] * weights[col * K + k];
    }

    // Write to both destinations
    output_act[row * N + col] = acc;
    captured[row * N + col] = acc;
}
