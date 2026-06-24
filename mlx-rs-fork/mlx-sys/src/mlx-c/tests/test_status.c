#include <stdio.h>
#include <assert.h>
#include <mlx_c/mlx_c.h>

int main() {
    mlx_c_status_t ok = mlx_c_status_ok();
    assert(ok.code == MLX_C_STATUS_OK);
    assert(mlx_c_status_is_ok(ok));

    mlx_c_status_t err;
    err.code = MLX_C_STATUS_INVALID_ARGUMENT;
    assert(!mlx_c_status_is_ok(err));

    printf("test_status passed\n");
    return 0;
}
