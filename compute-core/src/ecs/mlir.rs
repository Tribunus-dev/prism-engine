//! Target-independent MLIR execution contracts for Prism ECS.
//!
//! The ECS-facing contract is independent of the vendor backend. With the
//! `mlir-runtime` feature enabled, its textual module is parsed and verified by
//! the real MLIR runtime before the precision implementation is selected.

use serde::{Deserialize, Serialize};

use crate::ecs::canonical::identity::LogicalTensorId;
use crate::ecs::canonical::kernel_abi::{KernelAbi, KernelSemanticId};
use crate::ecs::metal_backend::{catalogue_source_for, MetalImplementationCatalogue};

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
    Palettized { group_size: u32 },
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

/// Backend artifact produced by lowering a target-independent contract.
///
/// The Metal source remains authoritative in the implementation catalogue;
/// this adapter binds the contract's semantic identity to that source so the
/// existing compiler and evaluator can compile and measure the result.
#[derive(Debug, Clone, PartialEq)]
pub struct MetalLoweringArtifact {
    pub semantic_id: KernelSemanticId,
    pub module_text: String,
    pub source: String,
    pub entry_point: String,
    pub abi: KernelAbi,
    pub dispatch: MetalDispatchContract,
}

/// Concrete dispatch contract emitted by the MLIR-to-Metal translator. This
/// is intentionally separate from the catalogue registration: the catalogue
/// supplies the executable implementation, while MLIR supplies the verified
/// tensor geometry and schedule used to invoke it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDispatchContract {
    pub semantic_id: KernelSemanticId,
    pub grid: [u32; 3],
    pub threadgroup: [u32; 3],
    pub vector_width: u32,
    pub tile: [u32; 3],
}

/// Build the runtime dispatch contract from the geometry requested by a
/// precision binder. The binder supplies only physical launch dimensions;
/// this function turns them back into a valid tensor-level MLIR matmul,
/// verifies it when the MLIR runtime is enabled, and returns the translated
/// Metal launch contract used by the encoder.
pub fn runtime_dispatch_contract(
    semantic_id: &KernelSemanticId,
    grid: [u32; 3],
    threadgroup: [u32; 3],
) -> Result<MetalDispatchContract, String> {
    runtime_lowering_artifact(semantic_id, grid, threadgroup).map(|artifact| artifact.dispatch)
}

/// Lower a runtime launch request through the same MLIR path used by AOT
/// compilation, including translated Metal source and ABI selection.
pub fn runtime_lowering_artifact(
    semantic_id: &KernelSemanticId,
    grid: [u32; 3],
    threadgroup: [u32; 3],
) -> Result<MetalLoweringArtifact, String> {
    let weight_quantization = match semantic_id.0.as_str() {
        "prism.linear.rawf32.v1" => QuantizationAttribute::F32,
        "prism.linear.int8.v1" => QuantizationAttribute::Int8 { group_size: 1 },
        "prism.linear.nf4.v1"
        | "prism.nf4.tile640.gemv.v1"
        | "prism.nf4tile640.dequant_mul.v1"
        | "prism.q4.block_sym.gemv.v1" => QuantizationAttribute::Nf4 { group_size: 128 },
        "prism.palettized.gemv.v1" | "prism.palettized.gemm.v1" | "prism.palettized.swiglu.v1" => {
            QuantizationAttribute::Palettized { group_size: 16 }
        }
        "prism.ternary.gemv.v1"
        | "prism.ternary.gemv.v2"
        | "prism.ternary.gemm.v1"
        | "prism.ternary.cimage.gemv.v1" => QuantizationAttribute::Ternary {
            group_size: 128,
            packed: true,
        },
        _ => {
            return Err(format!(
                "MLIR runtime has no quantization mapping for {}",
                semantic_id.0
            ))
        }
    };
    let contract = MlirExecutionContract {
        semantic_id: semantic_id.clone(),
        inputs: vec![
            MlirTensorType {
                id: LogicalTensorId("runtime.input".into()),
                shape: vec![grid[1].max(1) as u64, 1],
                quantization: QuantizationAttribute::F16,
            },
            MlirTensorType {
                id: LogicalTensorId("runtime.weights".into()),
                shape: vec![1, grid[0].max(1) as u64],
                quantization: weight_quantization,
            },
        ],
        output: MlirTensorType {
            id: LogicalTensorId("runtime.output".into()),
            shape: vec![grid[1].max(1) as u64, grid[0].max(1) as u64],
            quantization: QuantizationAttribute::F16,
        },
        dialects: vec![MlirDialect::Prism, MlirDialect::Linalg, MlirDialect::Gpu],
        schedule: MlirTransformSchedule {
            steps: vec![
                TransformStep::Tile {
                    m: threadgroup[0].max(1),
                    n: threadgroup[1].max(1),
                    k: threadgroup[2].max(1),
                },
                TransformStep::Vectorize {
                    width: threadgroup[0].max(1),
                },
            ],
        },
        target: MlirLoweringTarget::Metal,
    };
    contract.lower_to_metal()
}

/// MLIR-to-Metal adapter. The optional runtime performs real MLIR parsing and
/// verification; the final precision-specific Metal implementation is selected
/// from Prism's authoritative catalogue so its ABI and executable source stay
/// identical to the non-MLIR path.
pub struct MlirToMetalAdapter;

#[cfg(feature = "mlir-runtime")]
fn verify_with_real_mlir(module_text: &str) -> Result<(), String> {
    use melior::{
        dialect::DialectRegistry,
        ir::{operation::OperationLike, Module},
        utility, Context,
    };

    let registry = DialectRegistry::new();
    utility::register_all_dialects(&registry);
    let context = Context::new_with_registry(&registry, true);
    context.set_allow_unregistered_dialects(true);
    let module = Module::parse(&context, module_text)
        .ok_or_else(|| "MLIR parser rejected module".to_string())?;
    if !module.as_operation().verify() {
        return Err("MLIR verifier rejected module".into());
    }
    Ok(())
}

#[cfg(not(feature = "mlir-runtime"))]
fn verify_with_real_mlir(_module_text: &str) -> Result<(), String> {
    Ok(())
}

impl MlirToMetalAdapter {
    pub fn lower(contract: &MlirExecutionContract) -> Result<MetalLoweringArtifact, String> {
        contract.lower_to_metal()
    }
}

impl MlirExecutionContract {
    pub fn validate(&self) -> Result<(), String> {
        if self.inputs.is_empty() || self.output.shape.is_empty() {
            return Err("MLIR contract must declare inputs and an output shape".into());
        }
        if self.dialects.is_empty() || self.schedule.steps.is_empty() {
            return Err("MLIR contract must declare dialects and a transform schedule".into());
        }
        for dimension in self
            .output
            .shape
            .iter()
            .chain(self.inputs.iter().flat_map(|tensor| tensor.shape.iter()))
        {
            if *dimension == 0 {
                return Err("MLIR tensor dimensions must be non-zero".into());
            }
        }
        for step in &self.schedule.steps {
            match step {
                TransformStep::Tile { m, n, k } if *m == 0 || *n == 0 || *k == 0 => {
                    return Err("MLIR tile dimensions must be non-zero".into())
                }
                TransformStep::Parallelize { lanes, .. }
                | TransformStep::Vectorize { width: lanes }
                    if *lanes == 0 =>
                {
                    return Err("MLIR parallel width must be non-zero".into())
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Emit a deterministic MLIR module for parsing, verification, and receipt
    /// hashing.
    pub fn module_text(&self) -> Result<String, String> {
        self.validate()?;
        let mut text = format!("module @{} {{\n", self.semantic_id.0.replace('.', "_"));
        text.push_str("  // prism dialect contract\n");
        for tensor in self.inputs.iter().chain(std::iter::once(&self.output)) {
            text.push_str(&format!(
                "  // tensor {} shape {:?}\n",
                tensor.id.0, tensor.shape
            ));
        }
        for step in &self.schedule.steps {
            text.push_str(&format!("  // transform {:?}\n", step));
        }
        let input = &self.inputs[0];
        let output = &self.output;
        let input_shape = input
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x");
        let output_shape = output
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x");
        let weight_shape = if let Some(weight) = self.inputs.get(1) {
            weight
                .shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join("x")
        } else {
            format!(
                "{}x{}",
                input.shape.last().copied().unwrap_or(1),
                output.shape.last().copied().unwrap_or(1)
            )
        };
        text.push_str(&format!(
            "  func.func @{}(%input: tensor<{}xf32>, %weights: tensor<{}xf32>) -> tensor<{}xf32> {{\n",
            self.semantic_id.0.replace('.', "_"),
            input_shape,
            weight_shape,
            output_shape
        ));
        text.push_str(&format!(
            "    %init = tensor.empty() : tensor<{}xf32>\n    %result = linalg.matmul ins(%input, %weights : tensor<{}xf32>, tensor<{}xf32>) outs(%init : tensor<{}xf32>) -> tensor<{}xf32>\n",
            output_shape, input_shape, weight_shape, output_shape, output_shape
        ));
        text.push_str(&format!(
            "    \"prism.metal.dispatch\"() {{semantic = \"{}\"}} : () -> ()\n",
            self.semantic_id.0
        ));
        text.push_str(&format!(
            "    return %result : tensor<{}xf32>\n  }}\n",
            output_shape
        ));
        text.push_str("}\n");
        Ok(text)
    }

    /// Parse and verify the contract with MLIR when enabled, then lower the
    /// verified precision operation to its production Metal implementation.
    pub fn lower_to_metal(&self) -> Result<MetalLoweringArtifact, String> {
        self.validate()?;
        if self.target != MlirLoweringTarget::Metal {
            return Err(format!(
                "Metal lowering requires Metal target, got {:?}",
                self.target
            ));
        }
        let module_text = self.module_text()?;
        verify_with_real_mlir(&module_text)?;
        if !module_text.contains("linalg.matmul") {
            return Err("MLIR lowering requires a linalg.matmul operation".into());
        }
        if !module_text.contains("\"prism.metal.dispatch\"")
            || !module_text.contains(&format!("semantic = \"{}\"", self.semantic_id.0))
        {
            return Err("MLIR lowering requires a semantic prism.metal.dispatch operation".into());
        }
        let source = catalogue_source_for(&self.semantic_id)
            .ok_or_else(|| format!("Metal catalogue has no source for {}", self.semantic_id.0))?;
        if !source.contains("kernel") {
            return Err(format!(
                "Metal source for {} is not executable",
                self.semantic_id.0
            ));
        }
        let catalogue = MetalImplementationCatalogue::default();
        let registration = catalogue
            .for_semantic(&self.semantic_id)
            .into_iter()
            .find(|registration| registration.source_entry_point.is_some())
            .ok_or_else(|| {
                format!(
                    "Metal catalogue has no executable entry point for {}",
                    self.semantic_id.0
                )
            })?;
        let dispatch = self.translate_dispatch()?;
        let source = self.translate_source(&source, &dispatch);
        Ok(MetalLoweringArtifact {
            semantic_id: self.semantic_id.clone(),
            module_text,
            source,
            entry_point: registration.source_entry_point.clone().unwrap(),
            abi: registration.abi.clone(),
            dispatch,
        })
    }

    fn translate_dispatch(&self) -> Result<MetalDispatchContract, String> {
        let output_m = *self.output.shape.first().unwrap_or(&1) as u32;
        let output_n = *self.output.shape.last().unwrap_or(&1) as u32;
        let mut tile = [output_m, output_n, 1];
        let mut vector_width = 1;
        for step in &self.schedule.steps {
            match step {
                TransformStep::Tile { m, n, k } => tile = [*m, *n, *k],
                TransformStep::Vectorize { width } => vector_width = *width,
                _ => {}
            }
        }
        if tile.iter().any(|dimension| *dimension == 0) || vector_width == 0 {
            return Err("MLIR translator produced an invalid Metal dispatch shape".into());
        }
        Ok(MetalDispatchContract {
            semantic_id: self.semantic_id.clone(),
            grid: [output_n.max(1), output_m.max(1), 1],
            threadgroup: [tile[0].min(32), tile[1].min(32), 1],
            vector_width,
            tile,
        })
    }

    fn translate_source(&self, source: &str, dispatch: &MetalDispatchContract) -> String {
        format!(
            "// MLIR-to-Metal translated dispatch\n// semantic: {}\n// grid: {} {} {}\n// threadgroup: {} {} {}\n// tile: {} {} {}\n// vector_width: {}\n{}",
            dispatch.semantic_id.0,
            dispatch.grid[0],
            dispatch.grid[1],
            dispatch.grid[2],
            dispatch.threadgroup[0],
            dispatch.threadgroup[1],
            dispatch.threadgroup[2],
            dispatch.tile[0],
            dispatch.tile[1],
            dispatch.tile[2],
            dispatch.vector_width,
            source
        )
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
        dialects: vec![
            MlirDialect::Prism,
            MlirDialect::Linalg,
            MlirDialect::Gpu,
            MlirDialect::Arith,
        ],
        schedule: MlirTransformSchedule {
            steps: vec![
                TransformStep::Tile { m: 2, n: 640, k: 4 },
                TransformStep::Vectorize { width: 16 },
                TransformStep::Parallelize {
                    dimension: "n".into(),
                    lanes: 16,
                },
                TransformStep::DecomposeReduction {
                    strategy: "sequential".into(),
                },
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

    #[test]
    fn metal_lowering_resolves_authoritative_executable_source() {
        let contract = nf4_tile640_contract(MlirLoweringTarget::Metal);
        let module = contract.module_text().unwrap();
        assert!(module.contains("linalg.matmul"));
        assert!(module.contains("tensor.empty"));
        let artifact = MlirToMetalAdapter::lower(&contract).unwrap();
        assert_eq!(artifact.entry_point, "dequant_mul_nf4tile640");
        assert!(artifact.module_text.contains("linalg.matmul"));
        assert!(artifact
            .source
            .contains("MLIR-to-Metal translated dispatch"));
        assert_eq!(artifact.dispatch.vector_width, 16);
        assert!(artifact.source.contains("dequant_mul_nf4tile640"));
        assert!(!artifact.source.trim().is_empty());
    }

    #[test]
    fn non_metal_contract_does_not_cross_metal_boundary() {
        let contract = nf4_tile640_contract(MlirLoweringTarget::Cpu);
        let error = contract.lower_to_metal().unwrap_err();
        assert!(error.contains("requires Metal target"));
    }

    #[test]
    fn all_registered_precision_targets_lower_through_one_adapter() {
        let targets = [
            "prism.linear.rawf32.v1",
            "prism.linear.nf4.v1",
            "prism.linear.int8.v1",
            "prism.ternary.gemv.v1",
            "prism.ternary.gemv.v2",
            "prism.ternary.gemm.v1",
            "prism.nf4.tile640.gemv.v1",
            "prism.nf4tile640.dequant_mul.v1",
            "prism.q4.block_sym.gemv.v1",
            "prism.palettized.gemv.v1",
            "prism.palettized.swiglu.v1",
            "prism.palettized.gemm.v1",
            "prism.ternary.cimage.gemv.v1",
        ];
        for semantic_id in targets {
            let contract = MlirExecutionContract {
                semantic_id: KernelSemanticId(semantic_id.into()),
                inputs: vec![MlirTensorType {
                    id: LogicalTensorId("input".into()),
                    shape: vec![2, 4],
                    quantization: QuantizationAttribute::F32,
                }],
                output: MlirTensorType {
                    id: LogicalTensorId("output".into()),
                    shape: vec![2, 640],
                    quantization: QuantizationAttribute::F32,
                },
                dialects: vec![MlirDialect::Prism, MlirDialect::Linalg, MlirDialect::Gpu],
                schedule: MlirTransformSchedule {
                    steps: vec![TransformStep::Tile { m: 2, n: 640, k: 4 }],
                },
                target: MlirLoweringTarget::Metal,
            };
            let artifact = contract
                .lower_to_metal()
                .unwrap_or_else(|error| panic!("{semantic_id} did not lower: {error}"));
            assert!(!artifact.source.is_empty());
            assert!(!artifact.entry_point.is_empty());
        }
    }

    #[test]
    fn runtime_dispatch_uses_translated_mlir_geometry() {
        let semantic = KernelSemanticId("prism.linear.nf4.v1".into());
        let dispatch = runtime_dispatch_contract(&semantic, [640, 2, 1], [16, 16, 1]).unwrap();
        assert_eq!(dispatch.semantic_id, semantic);
        assert_eq!(dispatch.grid, [640, 2, 1]);
        assert_eq!(dispatch.threadgroup, [16, 16, 1]);
        assert_eq!(dispatch.vector_width, 16);
    }
}
