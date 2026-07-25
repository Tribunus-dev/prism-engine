//! Evolutionary search mutation operator scaffolding.
//!
//! Defines per-tensor (format, operation) mutation operators for the
//! evolutionary search. This is the design scaffolding that will be
//! wired into AlphaEvolve after Waves 15-16 produce a working compiler
//! pipeline.
//!
//! The search jointly optimizes (format, operation) per tensor — not per
//! layer or per model. Each tensor in a compilation plan can independently
//! mutate its quantization format AND its computation operation.

use serde::{Deserialize, Serialize};

// ── TensorFormat ────────────────────────────────────────────────────────────

/// The quantization format for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorFormat {
    /// No quantization — full FP16 precision.
    Fp16,
    /// BFloat16 (truncated FP32).
    Bf16,
    /// 8-bit integer with per-tensor scale.
    Int8,
    /// 4-bit integer with per-group scale.
    Int4,
    /// Normal float 4 (4-bit, non-uniform, per-group).
    Nf4,
    /// Normal float 8.
    Nf8,
    /// Palettized 4-bit (k-means with 16 centroids per 16-element block).
    /// Used as the default AOT compilation format when no evolution plan
    /// is provided. Route: `compile_to_cimage` -> `quantize_and_append` ->
    /// k-means palettization -> `CImageWriter::append_palettized`.
    Palettized4Bit,
    /// Ternary 1.58-bit {-1, 0, +1} with FP16 scale per 128 elements.
    Ternary158,
    /// Binary 1-bit {0, +1} with FP16 scale.
    Binary1,
}

impl TensorFormat {
    /// Stable single-byte discriminant for content addressing.
    ///
    /// The order of variants in this enum is part of the public ABI of
    /// the CImage format. Do not reorder existing variants — append new
    /// variants at the end with a fresh discriminant byte.
    pub fn discriminant_byte(&self) -> u8 {
        match self {
            TensorFormat::Fp16 => 0,
            TensorFormat::Bf16 => 1,
            TensorFormat::Int8 => 2,
            TensorFormat::Int4 => 3,
            TensorFormat::Nf4 => 4,
            TensorFormat::Nf8 => 5,
            TensorFormat::Palettized4Bit => 6,
            TensorFormat::Ternary158 => 7,
            TensorFormat::Binary1 => 8,
        }
    }

    /// Reverse mapping for `discriminant_byte`. Returns `None` for an
    /// unknown byte so callers can detect stale or corrupted schema
    /// versions without panicking.
    pub fn from_discriminant_byte(byte: u8) -> Option<TensorFormat> {
        match byte {
            0 => Some(TensorFormat::Fp16),
            1 => Some(TensorFormat::Bf16),
            2 => Some(TensorFormat::Int8),
            3 => Some(TensorFormat::Int4),
            4 => Some(TensorFormat::Nf4),
            5 => Some(TensorFormat::Nf8),
            6 => Some(TensorFormat::Palettized4Bit),
            7 => Some(TensorFormat::Ternary158),
            8 => Some(TensorFormat::Binary1),
            _ => None,
        }
    }

    /// Lookup a format by its `Debug` name. Used to parse the
    /// `default_format` field on the constitutional
    /// `QuantizationResultComponent`, which is stored as a string for
    /// forward compatibility.
    pub fn from_name(name: &str) -> Option<TensorFormat> {
        match name {
            "Fp16" => Some(TensorFormat::Fp16),
            "Bf16" => Some(TensorFormat::Bf16),
            "Int8" => Some(TensorFormat::Int8),
            "Int4" => Some(TensorFormat::Int4),
            "Nf4" => Some(TensorFormat::Nf4),
            "Nf8" => Some(TensorFormat::Nf8),
            "Palettized4Bit" => Some(TensorFormat::Palettized4Bit),
            "Ternary158" => Some(TensorFormat::Ternary158),
            "Binary1" => Some(TensorFormat::Binary1),
            _ => None,
        }
    }

    /// All available formats.
    pub fn all() -> &'static [TensorFormat] {
        &[
            TensorFormat::Fp16,
            TensorFormat::Bf16,
            TensorFormat::Int8,
            TensorFormat::Int4,
            TensorFormat::Nf4,
            TensorFormat::Nf8,
            TensorFormat::Palettized4Bit,
            TensorFormat::Ternary158,
            TensorFormat::Binary1,
        ]
    }

    /// Bits per weight (approximate).
    pub fn bits_per_weight(&self) -> u32 {
        match self {
            TensorFormat::Fp16 => 16,
            TensorFormat::Bf16 => 16,
            TensorFormat::Int8 => 8,
            TensorFormat::Int4 => 4,
            TensorFormat::Nf4 => 4,
            TensorFormat::Nf8 => 8,
            TensorFormat::Palettized4Bit => 4,
            TensorFormat::Ternary158 => 2, // 1.58 rounded up
            TensorFormat::Binary1 => 1,
        }
    }

    /// Whether this format requires group-wise dequantization.
    pub fn requires_group_dequant(&self) -> bool {
        matches!(
            self,
            TensorFormat::Int4
                | TensorFormat::Nf4
                | TensorFormat::Palettized4Bit
                | TensorFormat::Ternary158
        )
    }
}

// ── TensorOperation ─────────────────────────────────────────────────────────

/// The operation kind for computing a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorOperation {
    /// Standard float matmul (FP16, BF16).
    Matmul,
    /// Ternary Gemm: {−1, 0, +1} weights × FP16 activation.
    /// Pure addition/subtraction — no multiplication by weight.
    TernaryGemm,
    /// Binary popcount Gemm: 1-bit weights × FP16 activation.
    /// Uses popcount for dot product.
    BinaryPopcountGemm,
    /// INT4 matmul with dequantization on load.
    Int4DequantMatmul,
}

impl TensorOperation {
    /// All available operations.
    pub fn all() -> &'static [TensorOperation] {
        &[
            TensorOperation::Matmul,
            TensorOperation::TernaryGemm,
            TensorOperation::BinaryPopcountGemm,
            TensorOperation::Int4DequantMatmul,
        ]
    }
}

// ── TensorMutation ──────────────────────────────────────────────────────────

/// A mutation that changes one tensor's format and/or operation.
#[derive(Debug, Clone)]
pub struct TensorMutation {
    pub tensor_id: String,
    pub new_format: Option<TensorFormat>,
    pub new_operation: Option<TensorOperation>,
}

impl TensorMutation {
    pub fn new_format(tensor_id: impl Into<String>, format: TensorFormat) -> Self {
        Self {
            tensor_id: tensor_id.into(),
            new_format: Some(format),
            new_operation: None,
        }
    }

    pub fn new_operation(tensor_id: impl Into<String>, operation: TensorOperation) -> Self {
        Self {
            tensor_id: tensor_id.into(),
            new_format: None,
            new_operation: Some(operation),
        }
    }

    pub fn new_both(
        tensor_id: impl Into<String>,
        format: TensorFormat,
        operation: TensorOperation,
    ) -> Self {
        Self {
            tensor_id: tensor_id.into(),
            new_format: Some(format),
            new_operation: Some(operation),
        }
    }
}

// ── MutationTable ───────────────────────────────────────────────────────────

/// Mutation table — registered mutation operators for evolution.
///
/// Each operator is a function that takes the current value and returns
/// a mutated value. The evolution engine calls these to explore the
/// search space.
pub struct MutationTable {
    pub format_mutations: Vec<fn(&TensorFormat) -> TensorFormat>,
    pub operation_mutations: Vec<fn(&TensorOperation) -> TensorOperation>,
}

impl MutationTable {
    pub fn new() -> Self {
        Self {
            format_mutations: Vec::new(),
            operation_mutations: Vec::new(),
        }
    }

    /// Register a format mutation operator.
    pub fn add_format_mutation(&mut self, op: fn(&TensorFormat) -> TensorFormat) {
        self.format_mutations.push(op);
    }

    /// Register an operation mutation operator.
    pub fn add_operation_mutation(&mut self, op: fn(&TensorOperation) -> TensorOperation) {
        self.operation_mutations.push(op);
    }

    /// Apply a format mutation: shift to the next format in the cycle.
    pub fn mutate_format(&self, format: &TensorFormat) -> TensorFormat {
        if self.format_mutations.is_empty() {
            // Default: cycle to next format
            let all = TensorFormat::all();
            let pos = all.iter().position(|f| f == format).unwrap_or(0);
            all[(pos + 1) % all.len()]
        } else {
            (self.format_mutations[0])(format)
        }
    }

    /// Apply an operation mutation: shift to the next operation in the cycle.
    pub fn mutate_operation(&self, op: &TensorOperation) -> TensorOperation {
        if self.operation_mutations.is_empty() {
            // Default: cycle to next operation
            let all = TensorOperation::all();
            let pos = all.iter().position(|o| o == op).unwrap_or(0);
            all[(pos + 1) % all.len()]
        } else {
            (self.operation_mutations[0])(op)
        }
    }
}

impl Default for MutationTable {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_cycle_mutation() {
        let table = MutationTable::new();
        let f = TensorFormat::Fp16;
        let mutated = table.mutate_format(&f);
        assert_eq!(mutated, TensorFormat::Bf16);
    }

    #[test]
    fn operation_cycle_mutation() {
        let table = MutationTable::new();
        let op = TensorOperation::Matmul;
        let mutated = table.mutate_operation(&op);
        assert_eq!(mutated, TensorOperation::TernaryGemm);
    }

    #[test]
    fn custom_format_mutation() {
        let mut table = MutationTable::new();
        table.add_format_mutation(|f| match f {
            TensorFormat::Fp16 => TensorFormat::Binary1,
            TensorFormat::Binary1 => TensorFormat::Fp16,
            _ => TensorFormat::Fp16,
        });

        assert_eq!(
            table.mutate_format(&TensorFormat::Fp16),
            TensorFormat::Binary1
        );
        assert_eq!(
            table.mutate_format(&TensorFormat::Binary1),
            TensorFormat::Fp16
        );
    }

    #[test]
    fn format_properties() {
        assert_eq!(TensorFormat::Fp16.bits_per_weight(), 16);
        assert_eq!(TensorFormat::Binary1.bits_per_weight(), 1);
        assert_eq!(TensorFormat::Ternary158.bits_per_weight(), 2);
        assert!(TensorFormat::Ternary158.requires_group_dequant());
        assert!(!TensorFormat::Fp16.requires_group_dequant());
    }

    #[test]
    fn tensor_mutation_creation() {
        let m1 = TensorMutation::new_format("q_proj", TensorFormat::Int4);
        assert_eq!(m1.tensor_id, "q_proj");
        assert_eq!(m1.new_format, Some(TensorFormat::Int4));
        assert!(m1.new_operation.is_none());

        let m2 = TensorMutation::new_operation("k_proj", TensorOperation::TernaryGemm);
        assert_eq!(m2.new_operation, Some(TensorOperation::TernaryGemm));
    }
}
