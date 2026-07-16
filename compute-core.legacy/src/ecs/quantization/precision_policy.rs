use serde::{Deserialize, Serialize};

/// Precision policy for each tensor class in a Gemma 4 model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionPolicy {
    /// Large projection matrices (Q, K, V, O, gate, up, down)
    pub large_projections: TensorPrecision,
    /// Embeddings and output head
    pub embeddings: TensorPrecision,
    /// Normalization weights (RMSNorm)
    pub norms: TensorPrecision,
    /// Biases and small vectors
    pub biases: TensorPrecision,
    /// KV cache
    pub kv_cache: TensorPrecision,
    /// MTP projections
    pub mtp_projections: TensorPrecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorPrecision {
    Nf4,
    Int8,
    Fp16,
    Fp32,
}

impl PrecisionPolicy {
    pub fn nf4_default() -> Self {
        Self::for_context_length(131072)
    }

    /// Build a policy adapted to a maximum context length.
    ///
    /// Context length determines KV cache precision: short contexts (<32K)
    /// use FP16, longer contexts switch to NF4 to fit M1 16GB memory budget.
    pub fn for_context_length(max_context: u32) -> Self {
        // NF4 KV cache: ~21 KB/token vs FP16's ~84 KB/token for Gemma 4 12B
        // At 256K context FP16 KV cache alone would be 21 GB — exceeds 16 GB
        let kv_precision = if max_context > 32768 {
            TensorPrecision::Nf4
        } else {
            TensorPrecision::Fp16
        };
        Self {
            large_projections: TensorPrecision::Nf4,
            embeddings: TensorPrecision::Nf4,
            norms: TensorPrecision::Fp16,
            biases: TensorPrecision::Fp16,
            kv_cache: kv_precision,
            mtp_projections: TensorPrecision::Nf4,
        }
    }
}

/// Memory budget check for M1 MacBook Pro (16 GB).
pub struct M1MemoryBudget {
    pub total_ram_gb: f64,
    pub os_overhead_gb: f64,
    pub max_model_gb: f64,
}

impl M1MemoryBudget {
    pub fn default_16gb() -> Self {
        // 16 GB total, ~2 GB for OS, ~1 GB for Metal GPU memory
        Self {
            total_ram_gb: 16.0,
            os_overhead_gb: 2.0,
            max_model_gb: 13.0,
        }
    }

    pub fn can_fit_model(&self, model_gb: f64, kv_cache_gb: f64) -> bool {
        let total = model_gb + kv_cache_gb + self.os_overhead_gb;
        total <= self.total_ram_gb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nf4_default_precision_policy() {
        let policy = PrecisionPolicy::nf4_default();
        assert_eq!(policy.large_projections, TensorPrecision::Nf4);
        assert_eq!(policy.embeddings, TensorPrecision::Nf4);
        assert_eq!(policy.norms, TensorPrecision::Fp16);
        assert_eq!(policy.biases, TensorPrecision::Fp16);
        // nf4_default uses 131K context, which exceeds the 32K FP16 threshold
        assert_eq!(policy.kv_cache, TensorPrecision::Nf4);
        assert_eq!(policy.mtp_projections, TensorPrecision::Nf4);
    }

    #[test]
    fn tensor_precision_eq() {
        assert_eq!(TensorPrecision::Nf4, TensorPrecision::Nf4);
        assert_ne!(TensorPrecision::Nf4, TensorPrecision::Fp16);
        assert_ne!(TensorPrecision::Int8, TensorPrecision::Fp32);
    }

    #[test]
    fn m1_budget_fits_within_limit() {
        let budget = M1MemoryBudget::default_16gb();
        // 12 GB model + 0.5 GB KV cache + 2 GB OS = 14.5 GB → within 16 GB
        assert!(budget.can_fit_model(12.0, 0.5));
    }

    #[test]
    fn m1_budget_exceeds_limit() {
        let budget = M1MemoryBudget::default_16gb();
        // 15 GB model + 0.5 GB KV cache + 2 GB OS = 17.5 GB → exceeds 16 GB
        assert!(!budget.can_fit_model(15.0, 0.5));
    }

    #[test]
    fn m1_budget_at_exact_boundary() {
        let budget = M1MemoryBudget::default_16gb();
        // 14 GB model + 0 GB KV cache + 2 GB OS = 16 GB → at boundary
        assert!(budget.can_fit_model(14.0, 0.0));
    }

    #[test]
    fn m1_budget_custom_config() {
        let budget = M1MemoryBudget {
            total_ram_gb: 8.0,
            os_overhead_gb: 1.0,
            max_model_gb: 7.0,
        };
        assert!(budget.can_fit_model(6.0, 0.5));
        assert!(!budget.can_fit_model(7.5, 0.0));
    }
}
