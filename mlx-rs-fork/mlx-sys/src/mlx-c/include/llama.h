#ifndef LLAMA_H
#define LLAMA_H

#include "mlx/c/array.h"

struct llama_layer {
    mlx_array attention_wq;
    mlx_array attention_wk;
    mlx_array attention_wv;
    mlx_array attention_wo;
    mlx_array ffn_w1; // gate
    mlx_array ffn_w2; // down
    mlx_array ffn_w3; // up
    mlx_array attention_norm_weight;
    mlx_array ffn_norm_weight;
};

struct llama_model {
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int dim;
    int hidden_dim;
    float norm_eps;
    float rope_freq_base;
    float rope_freq_scale;

    mlx_array tok_embeddings;
    struct llama_layer* layers;
    mlx_array norm_weight;
    mlx_array output_weight;
};

mlx_array rms_norm(mlx_array x, mlx_array weight, float eps);
mlx_array rope(mlx_array x, int n_heads, int n_kv_heads, int pos, float freq_base, float freq_scale);
mlx_array swiglu(mlx_array x, mlx_array gate, mlx_array up);
mlx_array attention(mlx_array q, mlx_array k, mlx_array v, mlx_array mask);
mlx_array llm_forward(struct llama_model* m, mlx_array tokens, int n_tokens);

#endif
