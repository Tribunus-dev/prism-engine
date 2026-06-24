#include "../include/llama.h"
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EXPECT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: Expected condition to be true\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

mlx_array create_dummy_array(int* shape, int ndim) {
    mlx_array a = mlx_array_new_float(0.01f);
    mlx_array b ={NULL};
    mlx_broadcast_to(&b, a, shape, ndim, (mlx_stream){NULL});
    mlx_array_free(a);
    return b;
}

int main() {
    struct llama_model model;
    model.n_layers = 1;
    model.n_heads = 2;
    model.n_kv_heads = 2;
    model.dim = 8;
    model.hidden_dim = 16;
    model.norm_eps = 1e-5f;
    model.rope_freq_base = 10000.0f;
    model.rope_freq_scale = 1.0f;

    int vocab_size = 32;
    int emb_shape[] = {vocab_size, model.dim};
    model.tok_embeddings = create_dummy_array(emb_shape, 2);

    int norm_shape[] = {model.dim};
    model.norm_weight = create_dummy_array(norm_shape, 1);

    int out_shape[] = {model.dim, vocab_size};
    model.output_weight = create_dummy_array(out_shape, 2);

    struct llama_layer layer;
    int wq_shape[] = {model.dim, model.dim};
    layer.attention_wq = create_dummy_array(wq_shape, 2);
    layer.attention_wk = create_dummy_array(wq_shape, 2);
    layer.attention_wv = create_dummy_array(wq_shape, 2);
    layer.attention_wo = create_dummy_array(wq_shape, 2);

    int w1_shape[] = {model.dim, model.hidden_dim};
    int w2_shape[] = {model.hidden_dim, model.dim};
    int w3_shape[] = {model.dim, model.hidden_dim};
    layer.ffn_w1 = create_dummy_array(w1_shape, 2);
    layer.ffn_w2 = create_dummy_array(w2_shape, 2);
    layer.ffn_w3 = create_dummy_array(w3_shape, 2);

    layer.attention_norm_weight = create_dummy_array(norm_shape, 1);
    layer.ffn_norm_weight = create_dummy_array(norm_shape, 1);

    model.layers = &layer;

    int tokens_data[] = {1, 5, 2};
    mlx_array tokens = mlx_array_new_data(tokens_data, (int[]){3}, 1, MLX_INT32);

    mlx_array logits = llm_forward(&model, tokens, 3);

    EXPECT_TRUE(logits.ctx != NULL);
    size_t ndim = mlx_array_ndim(logits);
    EXPECT_TRUE(ndim == 3);

    const int* shape = mlx_array_shape(logits);
    EXPECT_TRUE(shape[0] == 1);
    EXPECT_TRUE(shape[1] == 3);
    EXPECT_TRUE(shape[2] == 32);

    printf("LLaMA forward pass test passed.\n");

    mlx_array_free(logits);
    mlx_array_free(tokens);
    mlx_array_free(model.tok_embeddings);
    mlx_array_free(model.norm_weight);
    mlx_array_free(model.output_weight);
    mlx_array_free(layer.attention_wq);
    mlx_array_free(layer.attention_wk);
    mlx_array_free(layer.attention_wv);
    mlx_array_free(layer.attention_wo);
    mlx_array_free(layer.ffn_w1);
    mlx_array_free(layer.ffn_w2);
    mlx_array_free(layer.ffn_w3);
    mlx_array_free(layer.attention_norm_weight);
    mlx_array_free(layer.ffn_norm_weight);

    return 0;
}
