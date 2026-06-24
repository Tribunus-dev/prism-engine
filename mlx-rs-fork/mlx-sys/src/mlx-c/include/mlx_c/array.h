#ifndef MLX_C_ARRAY_H
#define MLX_C_ARRAY_H

#include "mlx_c/status.h"
#include "mlx_c/context.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_c_array mlx_c_array_t;

typedef enum {
    MLX_C_DTYPE_FLOAT32 = 0,
    MLX_C_DTYPE_UNKNOWN = 100
} mlx_c_dtype_t;

mlx_c_status_t mlx_c_array_create_from_f32(
    mlx_c_context_t* ctx,
    const float* data,
    const int64_t* shape,
    size_t ndim,
    mlx_c_array_t** out_array);

void mlx_c_array_free(mlx_c_array_t* array);

mlx_c_status_t mlx_c_array_dtype(const mlx_c_array_t* array, mlx_c_dtype_t* out_dtype);
mlx_c_status_t mlx_c_array_ndim(const mlx_c_array_t* array, size_t* out_ndim);
mlx_c_status_t mlx_c_array_shape(const mlx_c_array_t* array, int64_t* dims_out, size_t dims_capacity, size_t* ndim_out);
mlx_c_status_t mlx_c_array_size(const mlx_c_array_t* array, size_t* out_size);
mlx_c_status_t mlx_c_array_copy_to_f32(const mlx_c_array_t* array, float* data_out, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_ARRAY_H
