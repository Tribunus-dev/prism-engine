#pragma once
#include <stdint.h>
#include <stddef.h>
typedef struct ArenaInfo {
    int32_t width, height, logical_dim0, logical_dim1, pixel_format, byte_size, bytes_per_row;
    void *base_address, *cv_buffer, *io_surface;
} ArenaInfo;
