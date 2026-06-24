#include "mlx_c/status.h"
#include <string.h>

int mlx_c_status_is_ok(mlx_c_status_t status) {
    return status.code == MLX_C_STATUS_OK;
}

mlx_c_status_t mlx_c_status_ok(void) {
    mlx_c_status_t status;
    status.code = MLX_C_STATUS_OK;
    status.function = "mlx_c_status_ok";
    status.message[0] = '\0';
    return status;
}

const char* mlx_c_status_code_name(mlx_c_status_code_t code) {
    switch (code) {
        case MLX_C_STATUS_OK: return "MLX_C_STATUS_OK";
        case MLX_C_STATUS_INVALID_ARGUMENT: return "MLX_C_STATUS_INVALID_ARGUMENT";
        case MLX_C_STATUS_NULL_POINTER: return "MLX_C_STATUS_NULL_POINTER";
        case MLX_C_STATUS_DEVICE_UNAVAILABLE: return "MLX_C_STATUS_DEVICE_UNAVAILABLE";
        case MLX_C_STATUS_BACKEND_UNAVAILABLE: return "MLX_C_STATUS_BACKEND_UNAVAILABLE";
        case MLX_C_STATUS_NOT_IMPLEMENTED: return "MLX_C_STATUS_NOT_IMPLEMENTED";
        case MLX_C_STATUS_UPSTREAM_EXCEPTION: return "MLX_C_STATUS_UPSTREAM_EXCEPTION";
        case MLX_C_STATUS_INTERNAL_ERROR: return "MLX_C_STATUS_INTERNAL_ERROR";
        case MLX_C_STATUS_SHAPE_ERROR: return "MLX_C_STATUS_SHAPE_ERROR";
        case MLX_C_STATUS_DTYPE_UNSUPPORTED: return "MLX_C_STATUS_DTYPE_UNSUPPORTED";
        default: return "UNKNOWN_STATUS_CODE";
    }
}
