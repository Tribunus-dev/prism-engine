#include <stdio.h>
#include <assert.h>
#include <mlx_c/mlx_c.h>

int main() {
    mlx_c_version_info_t vinfo;
    mlx_c_status_t status = mlx_c_get_version_info(&vinfo);
    assert(mlx_c_status_is_ok(status));
    assert(vinfo.mlx_c_version != NULL);

    status = mlx_c_get_version_info(NULL);
    assert(!mlx_c_status_is_ok(status));
    assert(status.code == MLX_C_STATUS_NULL_POINTER);

    mlx_c_context_t* ctx = NULL;
    status = mlx_c_context_create(&ctx);
    assert(mlx_c_status_is_ok(status));
    assert(ctx != NULL);

    bool available = false;
    status = mlx_c_context_is_backend_available(ctx, &available);
    assert(mlx_c_status_is_ok(status));

    status = mlx_c_context_is_backend_available(NULL, &available);
    assert(!mlx_c_status_is_ok(status));

    mlx_c_context_free(ctx);

    printf("test_context passed\n");
    return 0;
}
