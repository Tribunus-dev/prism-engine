#include "mlx_c/context.h"
#include <cstdlib>

struct mlx_c_context {
    int dummy;
};

extern "C" mlx_c_status_t mlx_c_context_create(mlx_c_context_t** out_context) {
    if (!out_context) {
        mlx_c_status_t status = {MLX_C_STATUS_NULL_POINTER, "mlx_c_context_create", "out_context cannot be null"};
        return status;
    }

    *out_context = new mlx_c_context();
    return mlx_c_status_ok();
}

extern "C" void mlx_c_context_free(mlx_c_context_t* context) {
    if (context) {
        delete context;
    }
}

extern "C" mlx_c_status_t mlx_c_context_is_backend_available(const mlx_c_context_t* context, bool* out_available) {
    if (!context || !out_available) {
        mlx_c_status_t status = {MLX_C_STATUS_NULL_POINTER, "mlx_c_context_is_backend_available", "context or out_available cannot be null"};
        return status;
    }

#ifdef MLX_C_ENABLE_MLX_BACKEND
    *out_available = true;
#else
    *out_available = false;
#endif

    return mlx_c_status_ok();
}
