#include "kv_cache.h"
#include "mlx/c/ops.h"
#include <stdio.h>
#include <assert.h>

// Helper test macros based on memories
#define EXPECT_EQ_STATUS(status, expected) if ((status) != (expected)) { printf("Expected status %d but got %d\n", (expected), (status)); return 1; }
#define EXPECT_TRUE(cond) if (!(cond)) { printf("Condition failed: %s\n", #cond); return 1; }
#define EXPECT_FLOAT_NEAR(a, b, tol) if ((a) < (b) - (tol) || (a) > (b) + (tol)) { printf("Expected %f near %f\n", (a), (b)); return 1; }

int main() {
    int n_layers = 2;
    int n_kv_heads = 4;
    int n_embd_k = 64;
    int n_embd_v = 64;
    int max_seq_len = 100;

    kv_cache* cache = kv_cache_create(n_layers, n_kv_heads, n_embd_k, n_embd_v, max_seq_len);
    EXPECT_TRUE(cache != NULL);
    EXPECT_TRUE(cache->seq_len == 0);

    // Test caching values
    int seq_len = 10;
    int shape_k[] = {1, n_kv_heads, seq_len, n_embd_k};
    int shape_v[] = {1, n_kv_heads, seq_len, n_embd_v};
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    mlx_array k_slice = mlx_array_new();
    mlx_zeros(&k_slice, shape_k, 4, MLX_FLOAT32, stream);
    
    mlx_array v_slice = mlx_array_new();
    mlx_zeros(&v_slice, shape_v, 4, MLX_FLOAT32, stream);

    int status = kv_cache_store(cache, 0, 0, k_slice, v_slice);
    EXPECT_EQ_STATUS(status, 0);
    EXPECT_TRUE(cache->seq_len == 10);

    // Retrieve slices
    mlx_array k_view;
    mlx_array v_view;
    status = kv_cache_slice(cache, 0, 0, cache->seq_len, &k_view, &v_view);
    EXPECT_EQ_STATUS(status, 0);

    // Check retrieved shapes
    const int* k_view_shape = mlx_array_shape(k_view);
    const int* v_view_shape = mlx_array_shape(v_view);
    
    EXPECT_TRUE(k_view_shape[2] == seq_len);
    EXPECT_TRUE(v_view_shape[2] == seq_len);

    kv_cache_clear(cache);
    EXPECT_TRUE(cache->seq_len == 0);

    mlx_array_free(k_slice);
    mlx_array_free(v_slice);
    mlx_array_free(k_view);
    mlx_array_free(v_view);
    mlx_stream_free(stream);
    kv_cache_free(cache);

    printf("KV cache tests passed!\n");
    return 0;
}
