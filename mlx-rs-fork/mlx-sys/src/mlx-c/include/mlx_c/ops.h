#ifndef MLX_C_OPS_H
#define MLX_C_OPS_H

#include "mlx_c/status.h"
#include "mlx_c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

mlx_c_status_t mlx_c_array_copy(const mlx_c_array_t* input, mlx_c_array_t** out);
mlx_c_status_t mlx_c_add(const mlx_c_array_t* lhs, const mlx_c_array_t* rhs, mlx_c_array_t** out);
mlx_c_status_t mlx_c_multiply(const mlx_c_array_t* lhs, const mlx_c_array_t* rhs, mlx_c_array_t** out);
mlx_c_status_t mlx_c_sigmoid(const mlx_c_array_t* input, mlx_c_array_t** out);
mlx_c_status_t mlx_c_silu(const mlx_c_array_t* input, mlx_c_array_t** out);
mlx_c_status_t mlx_c_matmul(const mlx_c_array_t* lhs, const mlx_c_array_t* rhs, mlx_c_array_t** out);
mlx_c_status_t mlx_c_reshape(const mlx_c_array_t* input, const int64_t* new_shape, size_t new_ndim, mlx_c_array_t** out);
mlx_c_status_t mlx_c_transpose(const mlx_c_array_t* input, const int64_t* axes, size_t axes_len, mlx_c_array_t** out);
mlx_c_status_t mlx_c_softmax(const mlx_c_array_t* input, int64_t axis, mlx_c_array_t** out);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_OPS_H
