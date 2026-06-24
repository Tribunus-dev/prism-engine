pub struct RvqState {
    codebook_size: usize,
    n_q: usize,
    dim: usize,
}

impl RvqState {
    pub fn new(codebook_size: usize, n_q: usize, dim: usize) -> Self {
        Self { codebook_size, n_q, dim }
    }
    
    pub fn quantize(&self, _input: &[f32]) -> Vec<u32> {
        vec![]
    }
}
