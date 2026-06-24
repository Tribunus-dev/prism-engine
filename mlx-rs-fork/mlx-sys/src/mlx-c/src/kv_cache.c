#include "kv_cache.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/device.h"
#include <stdlib.h>

kv_cache* kv_cache_create(int n_layers, int n_kv_heads, int n_embd_k, int n_embd_v, int max_seq_len) {
    kv_cache* c = (kv_cache*)malloc(sizeof(kv_cache));
    if (!c) return NULL;

    c->n_layers = n_layers;
    c->n_kv_heads = n_kv_heads;
    c->n_embd_k = n_embd_k;
    c->n_embd_v = n_embd_v;
    c->max_seq_len = max_seq_len;
    c->seq_len = 0;

    c->k = (mlx_array*)malloc(n_layers * sizeof(mlx_array));
    c->v = (mlx_array*)malloc(n_layers * sizeof(mlx_array));

    mlx_stream stream = mlx_default_cpu_stream_new();

    for (int i = 0; i < n_layers; ++i) {
        int shape_k[] = {1, n_kv_heads, max_seq_len, n_embd_k};
        int shape_v[] = {1, n_kv_heads, max_seq_len, n_embd_v};
        c->k[i] = mlx_array_new();
        c->v[i] = mlx_array_new();
        mlx_zeros(&c->k[i], shape_k, 4, MLX_FLOAT32, stream);
        mlx_zeros(&c->v[i], shape_v, 4, MLX_FLOAT32, stream);
    }
    
    mlx_stream_free(stream);
    return c;
}

int kv_cache_store(kv_cache* c, int layer, int pos, mlx_array k_slice, mlx_array v_slice) {
    if (!c || layer < 0 || layer >= c->n_layers) return -1;
    
    const int* k_shape = mlx_array_shape(k_slice);
    const int* v_shape = mlx_array_shape(v_slice);
    size_t k_ndim = mlx_array_ndim(k_slice);
    size_t v_ndim = mlx_array_ndim(v_slice);
    
    int seq_len = 0;
    // Assume k_slice has shape [1, n_kv_heads, seq_len, n_embd_k] or similar. 
    // Wait, k_slice and v_slice seq_len is usually the size of the 2nd axis (index 2).
    if (k_ndim >= 3) {
        seq_len = k_shape[2]; // assuming [batch, heads, seq_len, embd]
    } else {
        return -1; // invalid shape
    }
    
    int start_k[] = {0, 0, pos, 0};
    int stop_k[] = {1, c->n_kv_heads, pos + seq_len, c->n_embd_k};
    int strides_k[] = {1, 1, 1, 1};

    int start_v[] = {0, 0, pos, 0};
    int stop_v[] = {1, c->n_kv_heads, pos + seq_len, c->n_embd_v};
    int strides_v[] = {1, 1, 1, 1};

    mlx_stream stream = mlx_default_cpu_stream_new();
    
    mlx_array new_k = mlx_array_new();
    int status = mlx_slice_update(&new_k, c->k[layer], k_slice, start_k, 4, stop_k, 4, strides_k, 4, stream);
    if (status == 0) {
        mlx_array_free(c->k[layer]);
        c->k[layer] = new_k;
    }

    mlx_array new_v = mlx_array_new();
    status = mlx_slice_update(&new_v, c->v[layer], v_slice, start_v, 4, stop_v, 4, strides_v, 4, stream);
    if (status == 0) {
        mlx_array_free(c->v[layer]);
        c->v[layer] = new_v;
    }

    mlx_stream_free(stream);
    
    if (pos + seq_len > c->seq_len) {
        c->seq_len = pos + seq_len;
    }
    
    return 0;
}

int kv_cache_slice(kv_cache* c, int layer, int start, int end, mlx_array* out_k_view, mlx_array* out_v_view) {
    if (!c || layer < 0 || layer >= c->n_layers) return -1;
    
    int start_idx[] = {0, 0, start, 0};
    int stop_k[] = {1, c->n_kv_heads, end, c->n_embd_k};
    int stop_v[] = {1, c->n_kv_heads, end, c->n_embd_v};
    int strides[] = {1, 1, 1, 1};

    mlx_stream stream = mlx_default_cpu_stream_new();
    
    *out_k_view = mlx_array_new();
    mlx_slice(out_k_view, c->k[layer], start_idx, 4, stop_k, 4, strides, 4, stream);

    *out_v_view = mlx_array_new();
    mlx_slice(out_v_view, c->v[layer], start_idx, 4, stop_v, 4, strides, 4, stream);

    mlx_stream_free(stream);
    return 0;
}

void kv_cache_clear(kv_cache* c) {
    if (c) {
        c->seq_len = 0;
    }
}

void kv_cache_free(kv_cache* c) {
    if (c) {
        for (int i = 0; i < c->n_layers; ++i) {
            mlx_array_free(c->k[i]);
            mlx_array_free(c->v[i]);
        }
        free(c->k);
        free(c->v);
        free(c);
    }
}
