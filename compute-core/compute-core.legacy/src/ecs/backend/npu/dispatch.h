#pragma once

#include <cstddef>
#include <cstdint>
#include "../../runtime/memory/npu_unified.h"

namespace prism {
namespace npu {

/// Opaque handle wrapping the vendor-specific compiled graph.
using NpuGraphSession = void*;

/// Load an offline-compiled subgraph blob into the target NPU.
///   Apple: .mlmodelc directory  (Core ML)
///   Intel: .blob file           (OpenVINO)
///   AMD:   .xmodel file         (Vitis AI / XRT)
NpuGraphSession load_graph(TargetNpu target, const char* blob_path);

/// Submit a compiled graph for asynchronous NPU execution.
/// Returns a monotonically increasing submission ID for ECS tracking.
uint64_t submit_execution(
    TargetNpu target,
    NpuGraphSession session,
    NpuBuffer* input_tensors,
    NpuBuffer* output_tensors,
    size_t tensor_count);

/// Non-blocking poll for completion of a specific submission.
bool poll_completion(TargetNpu target, NpuGraphSession session, uint64_t submission_id);

/// Release a loaded graph session.
void destroy_session(TargetNpu target, NpuGraphSession session);

} // namespace npu
} // namespace prism
