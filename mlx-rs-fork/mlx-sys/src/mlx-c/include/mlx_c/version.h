#ifndef MLX_C_VERSION_H
#define MLX_C_VERSION_H

#include "mlx_c/status.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* mlx_c_version;
    const char* mlx_version;
    bool mlx_backend_enabled;
    const char* compiler_info;
} mlx_c_version_info_t;

mlx_c_status_t mlx_c_get_version_info(mlx_c_version_info_t* out_info);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_VERSION_H
