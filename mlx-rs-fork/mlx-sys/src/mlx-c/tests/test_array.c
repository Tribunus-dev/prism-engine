#include <stdio.h>
#include <stdlib.h>
#include <mlx_c/mlx_c.h>
#include "test_utils.h"

void test_adversarial() {
    mlx_c_context_t* ctx = NULL;
    mlx_c_status_t status = mlx_c_context_create(&ctx);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);

    mlx_c_array_t* arr = NULL;
    float data[4] = {1.0, 2.0, 3.0, 4.0};
    int64_t shape[2] = {2, 2};

    // Test null output pointer
    status = mlx_c_array_create_from_f32(ctx, data, shape, 2, NULL);
    EXPECT_EQ_STATUS(MLX_C_STATUS_NULL_POINTER, status);

    // Test null context
    status = mlx_c_array_create_from_f32(NULL, data, shape, 2, &arr);
    EXPECT_EQ_STATUS(MLX_C_STATUS_NULL_POINTER, status);
    EXPECT_TRUE(arr == NULL);

    // Test null data
    status = mlx_c_array_create_from_f32(ctx, NULL, shape, 2, &arr);
    EXPECT_EQ_STATUS(MLX_C_STATUS_NULL_POINTER, status);
    EXPECT_TRUE(arr == NULL);

    // Test zero ndim
    status = mlx_c_array_create_from_f32(ctx, data, shape, 0, &arr);
    EXPECT_EQ_STATUS(MLX_C_STATUS_INVALID_ARGUMENT, status);
    EXPECT_TRUE(arr == NULL);

    // Test negative dimensions
    int64_t bad_shape1[2] = {2, -2};
    status = mlx_c_array_create_from_f32(ctx, data, bad_shape1, 2, &arr);
    EXPECT_EQ_STATUS(MLX_C_STATUS_SHAPE_ERROR, status);
    EXPECT_TRUE(arr == NULL);

    // Test shape overflow (should fail gracefully)
    int64_t bad_shape2[2] = {10000000000LL, 10000000000LL};
    status = mlx_c_array_create_from_f32(ctx, data, bad_shape2, 2, &arr);
    EXPECT_EQ_STATUS(MLX_C_STATUS_SHAPE_ERROR, status);
    EXPECT_TRUE(arr == NULL);

    mlx_c_context_free(ctx);
}

void test_roundtrip_and_ownership() {
    mlx_c_context_t* ctx = NULL;
    mlx_c_status_t status = mlx_c_context_create(&ctx);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);

    bool available = false;
    mlx_c_context_is_backend_available(ctx, &available);

    mlx_c_array_t* arr = NULL;
    float source_data[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    int64_t shape[2] = {2, 2};

    status = mlx_c_array_create_from_f32(ctx, source_data, shape, 2, &arr);
    if (!available) {
        EXPECT_EQ_STATUS(MLX_C_STATUS_BACKEND_UNAVAILABLE, status);
        mlx_c_context_free(ctx);
        return;
    }

    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);
    EXPECT_TRUE(arr != NULL);

    // Mutate source_data to verify copy semantics
    source_data[0] = 99.9f;

    // Inspect ndim
    size_t ndim = 0;
    status = mlx_c_array_ndim(arr, &ndim);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);
    EXPECT_EQ_SIZE(2, ndim);

    // Inspect size
    size_t size = 0;
    status = mlx_c_array_size(arr, &size);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);
    EXPECT_EQ_SIZE(4, size);

    // Inspect dtype
    mlx_c_dtype_t dtype;
    status = mlx_c_array_dtype(arr, &dtype);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);
    EXPECT_EQ_INT(MLX_C_DTYPE_FLOAT32, dtype);

    // Inspect shape - Two pass query
    size_t queried_ndim = 0;
    status = mlx_c_array_shape(arr, NULL, 0, &queried_ndim);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);
    EXPECT_EQ_SIZE(2, queried_ndim);

    int64_t out_shape[2];
    status = mlx_c_array_shape(arr, out_shape, 2, &queried_ndim);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);
    EXPECT_EQ_INT(2, out_shape[0]);
    EXPECT_EQ_INT(2, out_shape[1]);

    // Array copy to host
    float copied_data[4] = {0};
    status = mlx_c_array_copy_to_f32(arr, copied_data, 4);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);

    // Verify values remain original (1.5) and not mutated (99.9)
    EXPECT_FLOAT_NEAR(1.5f, copied_data[0], 1e-6);
    EXPECT_FLOAT_NEAR(2.5f, copied_data[1], 1e-6);
    EXPECT_FLOAT_NEAR(3.5f, copied_data[2], 1e-6);
    EXPECT_FLOAT_NEAR(4.5f, copied_data[3], 1e-6);

    // Test too small capacity
    float tiny_data[2];
    status = mlx_c_array_copy_to_f32(arr, tiny_data, 2);
    EXPECT_EQ_STATUS(MLX_C_STATUS_SHAPE_ERROR, status);

    mlx_c_array_free(arr);
    mlx_c_context_free(ctx);
}

void test_create_free_loop() {
    mlx_c_context_t* ctx = NULL;
    mlx_c_status_t status = mlx_c_context_create(&ctx);
    EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);

    bool available = false;
    mlx_c_context_is_backend_available(ctx, &available);
    if (!available) {
        mlx_c_context_free(ctx);
        return;
    }

    float data[1] = {1.0f};
    int64_t shape[1] = {1};

    for (int i = 0; i < 1000; i++) {
        mlx_c_array_t* arr = NULL;
        status = mlx_c_array_create_from_f32(ctx, data, shape, 1, &arr);
        EXPECT_EQ_STATUS(MLX_C_STATUS_OK, status);
        EXPECT_TRUE(arr != NULL);
        mlx_c_array_free(arr);
    }

    mlx_c_context_free(ctx);
}

int main() {
    test_adversarial();
    test_roundtrip_and_ownership();
    test_create_free_loop();

    printf("test_array passed\n");
    return 0;
}
