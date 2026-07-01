/* Copyright © 2025 Apple Inc. */
/* Level Zero C API — capture dir for AOT SPIR-V extraction */

#ifndef MLX_LEVEL_ZERO_H
#define MLX_LEVEL_ZERO_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup level_zero Level Zero specific operations
 */
/**@{*/

int mlx_level_zero_is_available(bool* res);
int mlx_level_zero_set_capture_dir(const char* path);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
