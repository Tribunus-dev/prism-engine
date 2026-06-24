// Copyright © 2023 Apple Inc.

#pragma once

#include <metal_stdlib>

using namespace metal;

// On macOS 26+, the Metal `bfloat` type might not be available.
// Fall back to half (FP16) which is the same 16-bit size and
// works with all Metal operators and as_type conversions.
#if __has_extension(metal_bfloat)
    typedef bfloat bfloat16_t;
#else
    typedef half bfloat16_t;
#endif

inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}
