#ifndef MLX_LLAMA_H
#define MLX_LLAMA_H

#ifdef MLX_BACKEND
#include "mlx/c/array.h"
#else
// Mock for tests
typedef struct mlx_array_ { void* ctx; } mlx_array;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_ctx;
    float rope_freq_base;
    float rope_freq_scale;
} llama_model_params;

typedef struct {
    mlx_array wq;
    mlx_array wk;
    mlx_array wv;
    mlx_array wo;
    mlx_array w1;
    mlx_array w2;
    mlx_array w3;
    mlx_array rms_att_w;
    mlx_array rms_ffn_w;
} llama_layer;

typedef struct {
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int n_embd;
    int n_ff;
    
    mlx_array tok_embeddings;
    mlx_array norm;
    mlx_array output;
    
    llama_layer *layers;
    llama_model_params params;
} llama_model;

llama_model* llama_model_load(const char *gguf_path);
void llama_model_free(llama_model *model);

#ifdef __cplusplus
}
#endif

#endif
