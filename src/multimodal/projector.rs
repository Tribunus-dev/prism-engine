pub struct ProjectorConfig {
    pub input_dim: u32,
    pub output_dim: u32,
}

impl ProjectorConfig {
    pub fn forward(&self, features: &[u16]) -> Vec<u16> {
        // Dummy projection layer implementation
        features.to_vec()
    }
}
