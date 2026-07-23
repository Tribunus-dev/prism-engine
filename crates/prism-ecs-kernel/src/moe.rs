use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MoeDispatchRequest {
    pub expert_indices: Vec<usize>,
    pub weights: Vec<f32>,
}

pub fn request_from_router_logits(logits: &[f32], top_k: usize) -> MoeDispatchRequest {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(top_k.min(indices.len()));
    let weights = indices.iter().map(|&i| logits[i]).collect();
    MoeDispatchRequest {
        expert_indices: indices,
        weights,
    }
}

pub fn weighted_aggregate(outputs: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    let width = outputs.first().map_or(0, Vec::len);
    let mut result = vec![0.0; width];
    for (output, &weight) in outputs.iter().zip(weights) {
        for (dst, &value) in result.iter_mut().zip(output) {
            *dst += value * weight;
        }
    }
    result
}
