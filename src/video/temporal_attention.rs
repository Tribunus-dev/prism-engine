pub fn temporal_attention(
    x: &[f32], // [N, F, H*W, D] — flattened spatial
    heads: usize,
    is_causal: bool,
    n: usize,
    f: usize,
    hw: usize,
    d: usize,
) -> Vec<f32> {
    // For each spatial position (h, w):
    //   Q = x[:, :, pos, :]  — query all frames at this position
    //   K = x[:, :, pos, :]  — key all frames
    //   V = x[:, :, pos, :]  — value all frames
    //   attn = softmax(Q @ K^T / sqrt(d)) @ V  — attention across time
    //
    // Result: each spatial position aggregates information across frames

    let mut output = vec![0.0; n * f * hw * d];
    let head_dim = d / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0.0; f * f];
    for batch in 0..n {
        for pos in 0..hw {
            for h in 0..heads {
                // Compute attention scores for each frame
                for q_t in 0..f {
                    for k_t in 0..f {
                        if is_causal && k_t > q_t {
                            scores[q_t * f + k_t] = f32::NEG_INFINITY;
                            continue;
                        }
                        let mut score = 0.0;
                        for i in 0..head_dim {
                            let q_idx =
                                batch * (f * hw * d) + q_t * (hw * d) + pos * d + h * head_dim + i;
                            let k_idx =
                                batch * (f * hw * d) + k_t * (hw * d) + pos * d + h * head_dim + i;
                            score += x[q_idx] * x[k_idx];
                        }
                        scores[q_t * f + k_t] = score * scale;
                    }
                }

                // Softmax
                for q_t in 0..f {
                    let mut max_score = f32::NEG_INFINITY;
                    for k_t in 0..f {
                        if scores[q_t * f + k_t] > max_score {
                            max_score = scores[q_t * f + k_t];
                        }
                    }
                    let mut sum = 0.0;
                    for k_t in 0..f {
                        scores[q_t * f + k_t] = (scores[q_t * f + k_t] - max_score).exp();
                        sum += scores[q_t * f + k_t];
                    }
                    for k_t in 0..f {
                        scores[q_t * f + k_t] /= sum;
                    }
                }

                // Output
                for q_t in 0..f {
                    for i in 0..head_dim {
                        let mut out_val = 0.0;
                        for k_t in 0..f {
                            let v_idx =
                                batch * (f * hw * d) + k_t * (hw * d) + pos * d + h * head_dim + i;
                            out_val += scores[q_t * f + k_t] * x[v_idx];
                        }
                        let out_idx =
                            batch * (f * hw * d) + q_t * (hw * d) + pos * d + h * head_dim + i;
                        output[out_idx] = out_val;
                    }
                }
            }
        }
    }
    output
}
