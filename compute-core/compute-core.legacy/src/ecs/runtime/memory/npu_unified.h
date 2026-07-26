#pragma once

#include <cstddef>
#include <cstdint>

namespace prism {
namespace npu {

enum class TargetNpu {
    AppleAne = 0,
    IntelVpu = 1,  // OpenVINO / IVPU
    AmdXdna = 2,   // Ryzen AI / XRT
};

/// A memory buffer accessible by the NPU DMA engine.
/// Host pointer is valid for CPU read/write; vendor_handle is the
/// NPU-specific backing (MLMultiArray, OpenVINO Tensor, XRT BO).
struct NpuBuffer {
    void* ptr;
    size_t size;
    void* vendor_handle;
};

/// Allocate or register a buffer suitable for NPU DMA.
NpuBuffer allocate_npu_buffer(TargetNpu target, size_t size);
void free_npu_buffer(TargetNpu target, NpuBuffer* buffer);

} // namespace npu
} // namespace prism
