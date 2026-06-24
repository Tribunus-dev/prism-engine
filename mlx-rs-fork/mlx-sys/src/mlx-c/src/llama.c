#include "../include/llama.h"
#include "mlx/c/ops.h"
#include "mlx/c/fast.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

mlx_array rms_norm(mlx_array x, mlx_array weight, float eps) {
    mlx_array res ={NULL};
    mlx_fast_rms_norm(&res, x, weight, eps, (mlx_stream){NULL});
    return res;
}

mlx_array rope(mlx_array x, int n_heads, int n_kv_heads, int pos, float freq_base, float freq_scale) {
    mlx_optional_float base = {true, freq_base};
    mlx_array res ={NULL};
    int head_dim = 0;
    mlx_fast_rope(&res, x, head_dim, false, base, freq_scale, pos, (mlx_array){NULL}, (mlx_stream){NULL});
    return res;
}

mlx_array swiglu(mlx_array x, mlx_array gate, mlx_array up) {
    mlx_array sig_x ={NULL};
    mlx_sigmoid(&sig_x, x, (mlx_stream){NULL});
    
    mlx_array x_sig_x ={NULL};
    mlx_multiply(&x_sig_x, x, sig_x, (mlx_stream){NULL});
    mlx_array_free(sig_x);

    mlx_array res ={NULL};
    mlx_multiply(&res, x_sig_x, up, (mlx_stream){NULL});
    mlx_array_free(x_sig_x);
    
    return res;
}

mlx_array attention(mlx_array q, mlx_array k, mlx_array v, mlx_array mask) {
    mlx_array res ={NULL};
    mlx_fast_scaled_dot_product_attention(&res, q, k, v, 1.0f, "causal", mask, (mlx_array){NULL}, (mlx_stream){NULL});
    return res;
}

mlx_array llm_forward(struct llama_model* m, mlx_array tokens, int n_tokens) {
    mlx_array x ={NULL};
    mlx_take(&x, m->tok_embeddings, tokens, (mlx_stream){NULL});

    int head_dim = m->dim / m->n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int i = 0; i < m->n_layers; i++) {
        struct llama_layer* l = &m->layers[i];

        mlx_array normed_x = rms_norm(x, l->attention_norm_weight, m->norm_eps);

        mlx_array q ={NULL}, k ={NULL}, v ={NULL};
        mlx_matmul(&q, normed_x, l->attention_wq, (mlx_stream){NULL});
        mlx_matmul(&k, normed_x, l->attention_wk, (mlx_stream){NULL});
        mlx_matmul(&v, normed_x, l->attention_wv, (mlx_stream){NULL});

        int q_shape[] = {1, n_tokens, m->n_heads, head_dim};
        int kv_shape[] = {1, n_tokens, m->n_kv_heads, head_dim};

        mlx_array q_reshaped ={NULL}, k_reshaped ={NULL}, v_reshaped ={NULL};
        mlx_reshape(&q_reshaped, q, q_shape, 4, (mlx_stream){NULL});
        mlx_reshape(&k_reshaped, k, kv_shape, 4, (mlx_stream){NULL});
        mlx_reshape(&v_reshaped, v, kv_shape, 4, (mlx_stream){NULL});
        mlx_array_free(q);
        mlx_array_free(k);
        mlx_array_free(v);

        mlx_array q_rope = rope(q_reshaped, m->n_heads, m->n_kv_heads, 0, m->rope_freq_base, m->rope_freq_scale);
        mlx_array k_rope = rope(k_reshaped, m->n_heads, m->n_kv_heads, 0, m->rope_freq_base, m->rope_freq_scale);
        mlx_array_free(q_reshaped);
        mlx_array_free(k_reshaped);

        mlx_array attn_out ={NULL};
        mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, k_rope, v_reshaped, scale, "causal", (mlx_array){NULL}, (mlx_array){NULL}, (mlx_stream){NULL});
        mlx_array_free(q_rope);
        mlx_array_free(k_rope);
        mlx_array_free(v_reshaped);

        int out_shape[] = {1, n_tokens, m->dim};
        mlx_array attn_flat ={NULL};
        mlx_reshape(&attn_flat, attn_out, out_shape, 3, (mlx_stream){NULL});
        mlx_array_free(attn_out);

        mlx_array proj_out ={NULL};
        mlx_matmul(&proj_out, attn_flat, l->attention_wo, (mlx_stream){NULL});
        mlx_array_free(attn_flat);

        mlx_array x_added1 ={NULL};
        mlx_add(&x_added1, x, proj_out, (mlx_stream){NULL});
        mlx_array_free(x);
        mlx_array_free(proj_out);
        mlx_array_free(normed_x);
        x = x_added1;

        mlx_array normed_ffn = rms_norm(x, l->ffn_norm_weight, m->norm_eps);

        mlx_array gate ={NULL}, up ={NULL};
        mlx_matmul(&gate, normed_ffn, l->ffn_w1, (mlx_stream){NULL});
        mlx_matmul(&up, normed_ffn, l->ffn_w3, (mlx_stream){NULL});

        mlx_array swiglu_out = swiglu(gate, gate, up);
        mlx_array_free(gate);
        mlx_array_free(up);

        mlx_array down ={NULL};
        mlx_matmul(&down, swiglu_out, l->ffn_w2, (mlx_stream){NULL});
        mlx_array_free(swiglu_out);

        mlx_array x_added2 ={NULL};
        mlx_add(&x_added2, x, down, (mlx_stream){NULL});
        mlx_array_free(x);
        mlx_array_free(down);
        mlx_array_free(normed_ffn);
        x = x_added2;
    }

    mlx_array final_norm = rms_norm(x, m->norm_weight, m->norm_eps);
    mlx_array_free(x);

    mlx_array logits ={NULL};
    mlx_matmul(&logits, final_norm, m->output_weight, (mlx_stream){NULL});
    mlx_array_free(final_norm);

    return logits;
}
