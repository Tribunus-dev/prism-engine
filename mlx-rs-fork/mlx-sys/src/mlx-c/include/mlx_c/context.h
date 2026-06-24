#ifndef MLX_C_CONTEXT_H
#define MLX_C_CONTEXT_H

#include "mlx_c/status.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_c_context mlx_c_context_t;

mlx_c_status_t mlx_c_context_create(mlx_c_context_t** out_context);
void mlx_c_context_free(mlx_c_context_t* context);
mlx_c_status_t mlx_c_context_is_backend_available(const mlx_c_context_t* context, bool* out_available);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_CONTEXT_H
