/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/memory.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/memory.h"
#include "mlx/allocator.h"
#ifdef MLX_BUILD_METAL
#include "mlx/backend/metal/allocator.h"
#endif

extern "C" int mlx_clear_cache(void) {
  try {
    mlx::core::clear_cache();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_active_memory(size_t* res) {
  try {
    *res = mlx::core::get_active_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_cache_memory(size_t* res) {
  try {
    *res = mlx::core::get_cache_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_memory_limit(size_t* res) {
  try {
    *res = mlx::core::get_memory_limit();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_peak_memory(size_t* res) {
  try {
    *res = mlx::core::get_peak_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_reset_peak_memory(void) {
  try {
    mlx::core::reset_peak_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_set_cache_limit(size_t* res, size_t limit) {
  try {
    *res = mlx::core::set_cache_limit(limit);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_set_memory_limit(size_t* res, size_t limit) {
  try {
    *res = mlx::core::set_memory_limit(limit);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_set_wired_limit(size_t* res, size_t limit) {
  try {
    *res = mlx::core::set_wired_limit(limit);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// ── Output buffer hint (Tribunus) ─────────────────────────────────────────

extern "C" int mlx_set_output_buffer_hint(void* ptr, size_t size) {
  try {
    mlx::core::allocator::set_output_hint(ptr, size);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_clear_output_buffer_hint(void) {
  try {
    mlx::core::allocator::clear_output_hint();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_set_memory_plan(size_t num_slots, const mlx_memory_plan_slot* slots) {
#ifdef MLX_BUILD_METAL
  try {
    mlx::core::metal::allocator().set_memory_plan(
        num_slots,
        reinterpret_cast<const mlx::core::metal::PlanEntry*>(slots));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
}
#else
  (void)num_slots;
  (void)slots;
#endif
  return 0;
}

extern "C" int mlx_clear_memory_plan(void) {
#ifdef MLX_BUILD_METAL
  try {
    mlx::core::metal::allocator().clear_memory_plan();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
}
#endif
  return 0;
}
