/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_MEMORY_H
#define MLX_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup memory Memory operations
 */
/**@{*/

int mlx_clear_cache(void);
int mlx_get_active_memory(size_t* res);
int mlx_get_cache_memory(size_t* res);
int mlx_get_memory_limit(size_t* res);
int mlx_get_peak_memory(size_t* res);
int mlx_reset_peak_memory(void);
int mlx_set_cache_limit(size_t* res, size_t limit);
int mlx_set_memory_limit(size_t* res, size_t limit);
int mlx_set_wired_limit(size_t* res, size_t limit);

/**
 * Set the output buffer hint for the next Metal allocation.
 * When the next malloc(size) matches hint_size exactly, the Metal
 * allocator wraps ptr as an MTLBuffer instead of allocating new
 * memory.  The hint is consumed atomically on first use.
 */
int mlx_set_output_buffer_hint(void* ptr, size_t size);

/**
 * Clear the output buffer hint without consuming it.
 */
int mlx_clear_output_buffer_hint(void);
/**
 * Memory plan slot — pre-assigned IOSurface slice for one allocation.
 */
typedef struct {
    void*    iosurface_ptr;
    size_t   size;
} mlx_memory_plan_slot;

/**
 * Set the memory plan for the Metal allocator.
 */
int mlx_set_memory_plan(size_t num_slots, const mlx_memory_plan_slot* slots);

/**
 * Clear the memory plan.
 */
int mlx_clear_memory_plan(void);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
