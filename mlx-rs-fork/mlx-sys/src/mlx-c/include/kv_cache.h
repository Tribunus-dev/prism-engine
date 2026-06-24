#ifndef MLX_C_KV_CACHE_H
#define MLX_C_KV_CACHE_H

#include "mlx/c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kv_cache {
    int n_layers;
    int n_kv_heads;
    int n_embd_k;
    int n_embd_v;
    int max_seq_len;
    int seq_len;
    mlx_array* k;
    mlx_array* v;
} kv_cache;

kv_cache* kv_cache_create(int n_layers, int n_kv_heads, int n_embd_k, int n_embd_v, int max_seq_len);
int kv_cache_store(kv_cache* c, int layer, int pos, mlx_array k_slice, mlx_array v_slice);
int kv_cache_slice(kv_cache* c, int layer, int start, int end, mlx_array* out_k_view, mlx_array* out_v_view);
void kv_cache_clear(kv_cache* c);
void kv_cache_free(kv_cache* c);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_KV_CACHE_H
