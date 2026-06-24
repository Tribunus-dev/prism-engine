#include "mlx_c/version.h"

extern "C" mlx_c_status_t mlx_c_get_version_info(mlx_c_version_info_t* out_info) {
    if (!out_info) {
        mlx_c_status_t status = {MLX_C_STATUS_NULL_POINTER, "mlx_c_get_version_info", "out_info cannot be null"};
        return status;
    }

    out_info->mlx_c_version = "0.6.0";
#ifdef MLX_C_ENABLE_MLX_BACKEND
    out_info->mlx_version = "0.31.2";
    out_info->mlx_backend_enabled = true;
#else
    out_info->mlx_version = "N/A";
    out_info->mlx_backend_enabled = false;
#endif

#if defined(__clang__)
    out_info->compiler_info = "Clang";
#elif defined(__GNUC__)
    out_info->compiler_info = "GCC";
#elif defined(_MSC_VER)
    out_info->compiler_info = "MSVC";
#else
    out_info->compiler_info = "Unknown";
#endif

    return mlx_c_status_ok();
}
