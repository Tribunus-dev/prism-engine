//! Target-independent MLIR execution contracts for Prism ECS.
//!
//! This module deliberately does not depend on the MLIR C++ libraries. It is
//! the stable ECS-side contract that can be lowered to MLIR text or bytecode by
//! a platform adapter while Metal remains the production backend.

use serde::{Deserialize, Serialize};

use crate::ecs::canonical::identity::LogicalTensorId;
use crate::ecs::canonical::kernel_abi::KernelSemanticId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlirDialect {
    Linalg,
    Gpu,
    Arith,
    MemRef,
    Prism,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlirLoweringTarget {
    Metal,
    Nvidia,
    Amd,
    Cpu,
    HetGpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationAttribute {
    F32,
    F16,
    Int8 { group_size: u32 },
    Nf4 { group_size: u32 },
    Ternary { group_size: u32, packed: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlirTensorType {
    pub id: LogicalTensorId,
    pub shape: Vec<u64>,
    pub quantization: QuantizationAttribute,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformStep {
    Tile { m: u32, n: u32, k: u32 },
    Fuse { region: String },
    Parallelize { dimension: String, lanes: u32 },
    Vectorize { width: u32 },
    DecomposeReduction { strategy: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlirTransformSchedule {
    pub steps: Vec<TransformStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlirExecutionContract {
    pub semantic_id: KernelSemanticId,
    pub inputs: Vec<MlirTensorType>,
    pub output: MlirTensorType,
    pub dialects: Vec<MlirDialect>,
    pub schedule: MlirTransformSchedule,
    pub target: MlirLoweringTarget,
}

impl MlirExecutionContract {
    pub fn validate(&self) -> Result<(), String> {
        if self.inputs.is_empty() || self.output.shape.is_empty() {
            return Err("MLIR contract must declare inputs and an output shape".into());
        }
        if self.dialects.is_empty() || self.schedule.steps.is_empty() {
            return Err("MLIR contract must declare dialects and a transform schedule".into());
        }
        for dimension in self.output.shape.iter().chain(
            self.inputs.iter().flat_map(|tensor| tensor.shape.iter()),
        ) {
            if *dimension == 0 {
                return Err("MLIR tensor dimensions must be non-zero".into());
            }
        }
        for step in &self.schedule.steps {
            match step {
                TransformStep::Tile { m, n, k } if *m == 0 || *n == 0 || *k == 0 => {
                    return Err("MLIR tile dimensions must be non-zero".into())
                }
                TransformStep::Parallelize { lanes, .. } | TransformStep::Vectorize { width: lanes }
                    if *lanes == 0 => return Err("MLIR parallel width must be non-zero".into()),
                _ => {}
            }
        }
        Ok(())
    }

    /// Emit a deterministic MLIR-shaped module for backend adapters and
    /// receipt hashing. The actual vendor lowering remains outside this ECS
    /// contract.
    pub fn module_text(&self) -> Result<String, String> {
        self.validate()?;
        let mut text = format!("module @{} {{\n", self.semantic_id.0.replace('.', "_"));
        text.push_str("  // prism dialect contract\n");
        for tensor in self.inputs.iter().chain(std::iter::once(&self.output)) {
            text.push_str(&format!("  // tensor {} shape {:?}\n", tensor.id.0, tensor.shape));
        }
        for step in &self.schedule.steps {
            text.push_str(&format!("  // transform {:?}\n", step));
        }
        text.push_str("}\n");
        Ok(text)
    }
}

/// Canonical contract for the existing NF4 tile640 evaluator fixture.
pub fn nf4_tile640_contract(target: MlirLoweringTarget) -> MlirExecutionContract {
    let input = MlirTensorType {
        id: LogicalTensorId("nf4.input".into()),
        shape: vec![2, 4],
        quantization: QuantizationAttribute::F32,
    };
    let weights = MlirTensorType {
        id: LogicalTensorId("nf4.weights".into()),
        shape: vec![4, 640],
        quantization: QuantizationAttribute::Nf4 { group_size: 128 },
    };
    let output = MlirTensorType {
        id: LogicalTensorId("nf4.output".into()),
        shape: vec![2, 640],
        quantization: QuantizationAttribute::F32,
    };
    MlirExecutionContract {
        semantic_id: KernelSemanticId("prism.nf4tile640.dequant_mul.v1".into()),
        inputs: vec![input, weights],
        output,
        dialects: vec![MlirDialect::Prism, MlirDialect::Linalg, MlirDialect::Gpu, MlirDialect::Arith],
        schedule: MlirTransformSchedule {
            steps: vec![
                TransformStep::Tile { m: 2, n: 640, k: 4 },
                TransformStep::Vectorize { width: 16 },
                TransformStep::Parallelize { dimension: "n".into(), lanes: 16 },
                TransformStep::DecomposeReduction { strategy: "sequential".into() },
            ],
        },
        target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nf4_tile640_contract_is_valid_and_deterministic() {
        let contract = nf4_tile640_contract(MlirLoweringTarget::Metal);
        contract.validate().unwrap();
        let first = contract.module_text().unwrap();
        assert_eq!(first, contract.module_text().unwrap());
        assert!(first.contains("prism_nf4tile640_dequant_mul_v1"));
    }

    #[test]
    fn invalid_schedule_is_rejected() {
        let mut contract = nf4_tile640_contract(MlirLoweringTarget::Metal);
        contract.schedule.steps[0] = TransformStep::Tile { m: 0, n: 640, k: 4 };
        assert!(contract.validate().is_err());
    }
}
