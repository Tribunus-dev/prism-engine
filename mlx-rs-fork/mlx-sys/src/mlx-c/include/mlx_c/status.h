#ifndef MLX_C_STATUS_H
#define MLX_C_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MLX_C_STATUS_OK = 0,
    MLX_C_STATUS_INVALID_ARGUMENT = 1,
    MLX_C_STATUS_NULL_POINTER = 2,
    MLX_C_STATUS_DEVICE_UNAVAILABLE = 3,
    MLX_C_STATUS_BACKEND_UNAVAILABLE = 4,
    MLX_C_STATUS_NOT_IMPLEMENTED = 5,
    MLX_C_STATUS_UPSTREAM_EXCEPTION = 6,
    MLX_C_STATUS_INTERNAL_ERROR = 7,
    MLX_C_STATUS_SHAPE_ERROR = 8,
    MLX_C_STATUS_DTYPE_UNSUPPORTED = 9
} mlx_c_status_code_t;

typedef struct {
    mlx_c_status_code_t code;
    const char* function;
    char message[1024];
} mlx_c_status_t;

int mlx_c_status_is_ok(mlx_c_status_t status);
mlx_c_status_t mlx_c_status_ok(void);
const char* mlx_c_status_code_name(mlx_c_status_code_t code);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_STATUS_H
