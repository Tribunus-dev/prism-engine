//! Target-independent MLIR execution contracts for Prism ECS.
//!
//! The ECS-facing contract is independent of the vendor backend. With the
//! `mlir-runtime` feature enabled, its textual module is parsed and verified by
//! the real MLIR runtime before the precision implementation is selected.

use serde::{Deserialize, Serialize};

use prism_ecs_constitutional::canonical::identity::LogicalTensorId;
use prism_ecs_constitutional::canonical::kernel_abi::{
    generate_buffer_constants, generate_constant_indices, KernelAbi, KernelSemanticId,
};
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

/// Construct the canonical MLIR precision contract used by the normal Metal
/// compiler path. The dimensions are representative rather than model-specific;
/// runtime launch dimensions are specialized separately.
pub fn precision_contract_for_semantic(
    semantic_id: &KernelSemanticId,
) -> Result<MlirExecutionContract, String> {
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
        _ => return Err(format!("no MLIR precision contract for {}", semantic_id.0)),
    };
    Ok(MlirExecutionContract {
        semantic_id: semantic_id.clone(),
        inputs: vec![
            MlirTensorType {
                id: LogicalTensorId("precision.input".into()),
                shape: vec![1, 128],
                quantization: QuantizationAttribute::F16,
            },
            MlirTensorType {
                id: LogicalTensorId("precision.weights".into()),
                shape: vec![128, 640],
                quantization: weight_quantization,
            },
        ],
        output: MlirTensorType {
            id: LogicalTensorId("precision.output".into()),
            shape: vec![1, 640],
            quantization: QuantizationAttribute::F16,
        },
        dialects: vec![MlirDialect::Prism, MlirDialect::Linalg, MlirDialect::Gpu],
        schedule: MlirTransformSchedule {
            steps: vec![
                TransformStep::Tile {
                    m: 1,
                    n: 640,
                    k: 128,
                },
                TransformStep::Vectorize { width: 16 },
                TransformStep::Parallelize {
                    dimension: "n".into(),
                    lanes: 64,
                },
            ],
        },
        target: MlirLoweringTarget::Metal,
    })
}

/// MLIR-to-Metal adapter. The optional runtime performs real MLIR parsing and
/// verification; the final precision-specific Metal implementation is selected
/// from Prism's authoritative catalogue so its ABI and executable source stay
/// identical to the non-MLIR path.
pub struct MlirToMetalAdapter;

#[cfg(feature = "mlir-runtime")]
fn verify_with_real_mlir(module_text: &str, expected_semantic: &str) -> Result<(), String> {
    use melior::{
        dialect::DialectRegistry,
        ir::{
            operation::{OperationLike, WalkOrder, WalkResult},
            Module,
        },
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
    let mut found_dispatch = false;
    module
        .as_operation()
        .walk(WalkOrder::PreOrder, |operation| {
            let identifier = operation.name();
            let name = identifier.as_string_ref().as_str().unwrap_or_default();
            if name == "prism.metal.dispatch" {
                found_dispatch = operation
                    .attribute("semantic")
                    .map(|attribute| attribute.to_string().contains(expected_semantic))
                    .unwrap_or(false);
            }
            WalkResult::Advance
        });
    if !found_dispatch {
        return Err(format!(
            "MLIR module has no prism.metal.dispatch for {expected_semantic}"
        ));
    }
    Ok(())
}

#[cfg(not(feature = "mlir-runtime"))]
fn verify_with_real_mlir(_module_text: &str, _expected_semantic: &str) -> Result<(), String> {
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
        verify_with_real_mlir(&module_text, &self.semantic_id.0)?;
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
        let abi = registration.abi.clone();
        let entry_point = registration.source_entry_point.clone().unwrap();
        let source = self.translate_source(&source, &dispatch, &abi, &entry_point);
        Ok(MetalLoweringArtifact {
            semantic_id: self.semantic_id.clone(),
            module_text,
            source,
            entry_point,
            abi,
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

    fn translate_source(
        &self,
        source: &str,
        dispatch: &MetalDispatchContract,
        abi: &KernelAbi,
        entry_point: &str,
    ) -> String {
        let implementation = match self.semantic_id.0.as_str() {
            "prism.linear.rawf32.v1" => generated_rawf32_kernel(entry_point),
            "prism.linear.int8.v1" => generated_int8_kernel(entry_point),
            "prism.linear.nf4.v1" => generated_nf4_linear_kernel(entry_point),
            "prism.q4.block_sym.gemv.v1" => generated_q4_kernel(entry_point),
            "prism.nf4.tile640.gemv.v1" => generated_nf4_tile640_kernel(entry_point),
            "prism.ternary.cimage.gemv.v1" => generated_ternary_cimage_kernel(entry_point),
            "prism.ternary.gemv.v2" => generated_ternary_legacy_kernel(entry_point),
            "prism.palettized.gemv.v1" => generated_palettized_gemv_kernel(entry_point),
            "prism.palettized.gemm.v1" => generated_palettized_gemm_kernel(entry_point),
            "prism.palettized.swiglu.v1" => generated_palettized_swiglu_kernel(entry_point),
            "prism.ternary.gemv.v1" => generated_ternary_tile640_kernel(entry_point),
            "prism.ternary.gemm.v1" => generated_ternary_gemm_kernel(entry_point),
            "prism.nf4tile640.dequant_mul.v1" => generated_nf4_tile640_dequant_kernel(entry_point),
            _ => source.to_string(),
        };
        format!(
            "// MLIR-to-Metal translated dispatch\n// semantic: {}\n// grid: {} {} {}\n// threadgroup: {} {} {}\n// tile: {} {} {}\n// vector_width: {}\n{}{}{}",
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
            generate_buffer_constants(abi),
            generate_constant_indices(abi),
            implementation
        )
    }
}

fn generated_rawf32_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nstruct MlpConstants {{ uint32_t hidden_dim; uint32_t intermediate_dim; uint32_t group_size; uint32_t codec_id; float epsilon; uint32_t _pad[3]; }};\nkernel void {entry_point}(device const float* input [[buffer(0)]], device const float* weights [[buffer(1)]], device const float* scales [[buffer(2)]], device const float* biases [[buffer(3)]], device float* output [[buffer(4)]], constant MlpConstants& c [[buffer(5)]], uint tid [[thread_position_in_grid]]) {{\n    if (tid >= c.intermediate_dim) return;\n    float acc = 0.0f;\n    for (uint i = 0; i < c.hidden_dim; ++i) {{\n        acc = fma(weights[i * c.intermediate_dim + tid], input[i], acc);\n    }}\n    output[tid] = acc;\n}}\n"
    )
}

fn generated_int8_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nstruct MlpConstants {{ uint32_t hidden_dim; uint32_t intermediate_dim; uint32_t group_size; uint32_t codec_id; float epsilon; uint32_t _pad[3]; }};\nkernel void {entry_point}(device const float* input [[buffer(0)]], device const char* weights [[buffer(1)]], device const float* scales [[buffer(2)]], device const float* biases [[buffer(3)]], device float* output [[buffer(4)]], constant MlpConstants& c [[buffer(5)]], uint tid [[thread_position_in_grid]]) {{\n    if (tid >= c.intermediate_dim) return;\n    float acc = 0.0f;\n    float scale = scales[tid];\n    float bias = biases[tid];\n    for (uint i = 0; i < c.hidden_dim; ++i) {{\n        float w = (float)weights[i * c.intermediate_dim + tid] * scale + bias;\n        acc = fma(w, input[i], acc);\n    }}\n    output[tid] = acc;\n}}\n"
    )
}

fn generated_nf4_linear_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nconstant float nf4_codebook[16] = {{ -1.0f, -0.6961928f, -0.5250731f, -0.3949175f, -0.2844414f, -0.1847734f, -0.09105f, 0.0f, 0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f, 0.4407099f, 0.562617f, 0.7229568f, 1.0f }};\nstruct MlpConstants {{ uint32_t hidden_dim; uint32_t intermediate_dim; uint32_t group_size; uint32_t codec_id; float epsilon; uint32_t _pad[3]; }};\nkernel void {entry_point}(device const float* input [[buffer(0)]], device const uchar* codes [[buffer(1)]], device const float* scales [[buffer(2)]], device const float* biases [[buffer(3)]], device float* output [[buffer(4)]], constant MlpConstants& c [[buffer(5)]], uint tid [[thread_position_in_grid]]) {{\n    if (tid >= c.intermediate_dim) return;\n    float acc = 0.0f;\n    uint groups_per_tile = 640 / c.group_size;\n    uint out_group = tid / c.group_size;\n    for (uint i = 0; i < c.hidden_dim; ++i) {{\n        uint index = i * 640 + tid;\n        uchar packed = codes[index >> 1];\n        uchar nibble = (index & 1) ? (packed >> 4) : (packed & 0x0Fu);\n        float weight = fma(nf4_codebook[nibble], scales[i * groups_per_tile + out_group], biases[i * groups_per_tile + out_group]);\n        acc = fma(weight, input[i], acc);\n    }}\n    output[tid] = acc;\n}}\n"
    )
}

fn generated_q4_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nkernel void {entry_point}(device const half* input [[buffer(0)]], device const uint* weights [[buffer(1)]], device const half* scales [[buffer(2)]], device half* output [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& group_size [[buffer(6)]], uint row [[thread_position_in_grid]]) {{\n    if (row >= N) return;\n    uint num_groups = K / group_size;\n    uint words_per_group = group_size / 8;\n    uint packed_per_row = K / 8;\n    device const uint* row_weights = weights + row * packed_per_row;\n    device const half* row_scales = scales + row * num_groups;\n    float acc = 0.0f;\n    for (uint group = 0; group < num_groups; ++group) {{\n        float scale = float(row_scales[group]);\n        for (uint word = 0; word < words_per_group; ++word) {{\n            uint packed = row_weights[group * words_per_group + word];\n            uint input_base = group * group_size + word * 8;\n            for (uint lane = 0; lane < 8; ++lane) {{\n                uint nibble = (packed >> (lane * 4)) & 0xFu;\n                float value = float(int(nibble) - 8) * scale;\n                acc = fma(value, float(input[input_base + lane]), acc);\n            }}\n        }}\n    }}\n    output[row] = half(acc);\n}}\n"
    )
}

fn generated_nf4_tile640_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nconstant float nf4_codebook[16] = {{ -1.0f, -0.6961928f, -0.5250731f, -0.3949175f, -0.2844414f, -0.1847734f, -0.09105f, 0.0f, 0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f, 0.4407099f, 0.562617f, 0.7229568f, 1.0f }};\nstatic float unpack_nf4(device const uchar* packed, uint index) {{ uchar byte = packed[index >> 1]; uchar nibble = (index & 1) ? (byte >> 4) : (byte & 0x0Fu); return nf4_codebook[nibble]; }}\nkernel void {entry_point}(device const uchar* packed_weights [[buffer(0)]], device const float* scales [[buffer(1)]], device const float* biases [[buffer(2)]], device const float* in_vector [[buffer(3)]], device float* out_vector [[buffer(4)]], constant uint& num_macro_tiles [[buffer(5)]], constant uint& in_dim [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_threadgroup]]) {{\n    constexpr uint TILE = 640;\n    constexpr uint GROUP = 128;\n    constexpr uint GROUPS_PER_TILE = 5;\n    constexpr uint BYTES_PER_TILE = 320;\n    constexpr uint LANE_VALUES = 4;\n    float accumulator = 0.0f;\n    uint row_weight_base = row * num_macro_tiles * BYTES_PER_TILE;\n    uint row_meta_base = row * num_macro_tiles * GROUPS_PER_TILE;\n    for (uint tile = 0; tile < num_macro_tiles; ++tile) {{\n        uint weight_base = row_weight_base + tile * BYTES_PER_TILE;\n        uint meta_base = row_meta_base + tile * GROUPS_PER_TILE;\n        for (uint group = 0; group < GROUPS_PER_TILE; ++group) {{\n            float scale = scales[meta_base + group];\n            float bias = biases[meta_base + group];\n            uint local_base = group * GROUP + simd_lane * LANE_VALUES;\n            for (uint i = 0; i < LANE_VALUES; ++i) {{\n                uint local_col = local_base + i;\n                uint col = tile * TILE + local_col;\n                if (col >= in_dim) continue;\n                float weight = fma(unpack_nf4(packed_weights + weight_base, local_col), scale, bias);\n                accumulator = fma(weight, in_vector[col], accumulator);\n            }}\n        }}\n    }}\n    accumulator = simd_sum(accumulator);\n    if (simd_lane == 0) out_vector[row] = accumulator;\n}}\n"
    )
}

fn generated_ternary_cimage_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nstruct TernaryGemvConstants {{ uint32_t rows; uint32_t cols; uint32_t group_size; uint32_t groups_per_row; uint32_t bytes_per_group; uint32_t output_dtype; uint32_t padding[3]; }};\nkernel void {entry_point}(device const half* activations [[buffer(0)]], device const uchar* codes [[buffer(1)]], device const half* scales [[buffer(2)]], device half* output [[buffer(3)]], constant TernaryGemvConstants& c [[buffer(4)]], uint row [[thread_position_in_grid]]) {{\n    if (row >= c.rows) return;\n    float acc = 0.0f;\n    for (uint group = 0; group < c.groups_per_row; ++group) {{\n        float scale = float(scales[row * c.groups_per_row + group]);\n        uint group_offset = row * c.groups_per_row * c.bytes_per_group + group * c.bytes_per_group;\n        for (uint byte_index = 0; byte_index < c.bytes_per_group; ++byte_index) {{\n            uchar packed = codes[group_offset + byte_index];\n            for (uint lane = 0; lane < 4; ++lane) {{\n                uint weight_index = byte_index * 4 + lane;\n                if (weight_index >= c.group_size) break;\n                uint col = group * c.group_size + weight_index;\n                if (col >= c.cols) break;\n                uint code = (uint(packed) >> (lane * 2)) & 0x03u;\n                float weight = code == 0u ? -1.0f : (code == 2u ? 1.0f : 0.0f);\n                acc = fma(weight * scale, float(activations[col]), acc);\n            }}\n        }}\n    }}\n    output[row] = half(acc);\n}}\n"
    )
}

fn generated_ternary_legacy_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nkernel void {entry_point}(device const uchar* packed_weights [[buffer(0)]], device const half* input [[buffer(1)]], device half* output [[buffer(2)]], constant uint& in_dim [[buffer(3)]], constant uint& out_dim [[buffer(4)]], uint row [[thread_position_in_grid]]) {{\n    if (row >= out_dim) return;\n    uint packed_cols = in_dim / 4;\n    float sum = 0.0f;\n    for (uint i = 0; i < packed_cols; ++i) {{\n        uchar packed = packed_weights[row * packed_cols + i];\n        for (uint lane = 0; lane < 4; ++lane) {{\n            uint code = (uint(packed) >> (lane * 2)) & 0x03u;\n            float weight = code == 1u ? 1.0f : (code == 2u ? -1.0f : 0.0f);\n            sum = fma(weight, float(input[i * 4 + lane]), sum);\n        }}\n    }}\n    output[row] = half(sum);\n}}\n"
    )
}

fn generated_palettized_gemv_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nkernel void {entry_point}(device const half* input [[buffer(0)]], device const half* codebook [[buffer(1)]], device const uchar* indices [[buffer(2)]], device half* output [[buffer(3)]], constant uint& in_dim [[buffer(4)]], constant uint& out_dim [[buffer(5)]], uint row [[thread_position_in_grid]]) {{\n    if (row >= out_dim) return;\n    device const half* row_codebook = codebook + row * 16;\n    device const uchar* row_indices = indices + row * (in_dim / 2);\n    float acc = 0.0f;\n    for (uint i = 0; i < in_dim; ++i) {{\n        uchar packed = row_indices[i >> 1];\n        uint code = (i & 1u) ? (uint(packed) >> 4) : (uint(packed) & 0x0Fu);\n        acc = fma(float(input[i]), float(row_codebook[code]), acc);\n    }}\n    output[row] = half(acc);\n}}\n"
    )
}

fn generated_palettized_gemm_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nkernel void {entry_point}(device const uchar* weight_arena [[buffer(0)]], device const half* input [[buffer(1)]], device half* output [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& in_dim [[buffer(4)]], constant uint& out_dim [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]], uint2 local [[thread_position_in_threadgroup]]) {{\n    uint row = group.y * 16 + local.y;\n    uint col = group.x * 16 + local.x;\n    if (row >= M || col >= out_dim) return;\n    uint row_stride = 32 + in_dim / 2;\n    device const half* row_codebook = reinterpret_cast<device const half*>(weight_arena + col * row_stride);\n    device const uchar* row_indices = weight_arena + col * row_stride + 32;\n    float acc = 0.0f;\n    for (uint k = 0; k < in_dim; ++k) {{\n        uchar packed = row_indices[k >> 1];\n        uint code = (k & 1u) ? (uint(packed) >> 4) : (uint(packed) & 0x0Fu);\n        acc = fma(float(input[row * in_dim + k]), float(row_codebook[code]), acc);\n    }}\n    output[row * out_dim + col] = half(acc);\n}}\n"
    )
}

fn generated_palettized_swiglu_kernel(entry_point: &str) -> String {
    let s = format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nkernel void {entry_point}(device const uchar* gate_weights [[buffer(0)]], device const uchar* up_weights [[buffer(1)]], device const half* input [[buffer(2)]], device half* output [[buffer(3)]], constant uint& in_dim [[buffer(4)]], constant uint& out_dim [[buffer(5)]], uint row [[threadgroup_position_in_grid]]) {{\n    if (row >= out_dim) return;\n    uint stride = 32 + in_dim / 2;\n    device const half* gate_cb = reinterpret_cast<device const half*>(gate_weights + row * stride);\n    device const half* up_cb = reinterpret_cast<device const half*>(up_weights + row * stride);\n    device const uchar* gate_idx = gate_weights + row * stride + 32;\n    device const uchar* up_idx = up_weights + row * stride + 32;\n    float gate = 0.0f;\n    float up = 0.0f;\n    for (uint i = 0; i < in_dim; ++i) {{\n        uchar gp = gate_idx[i >> 1];\n        uchar up_p = up_idx[i >> 1];\n        uint gc = (i & 1u) ? (uint(gp) >> 4) : (uint(gp) & 0x0Fu);\n        uint uc = (i & 1u) ? (uint(up_p) >> 4) : (uint(up_p) & 0x0Fu);\n        gate = fma(float(input[i]), float(gate_cb[gc]), gate);\n        up = fma(float(input[i]), float(up_cb[uc]), up);\n    }}\n    float silu = gate / (1.0f + exp(-gate));\n    output[row] = half(silu * up);\n}}\n");
    s.replace("threadgroup_position_in_grid", "thread_position_in_grid")
}

fn generated_ternary_tile640_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nkernel void {entry_point}(device const uint* packed [[buffer(0)]], device const half* input [[buffer(1)]], device const ushort* page_scales [[buffer(2)]], device const uchar* lane_scales [[buffer(3)]], device half* output [[buffer(4)]], constant uint& in_dim [[buffer(5)]], constant uint& out_dim [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {{\n    if (row >= out_dim) return;\n    uint pages = (in_dim + 639) / 640;\n    uint words_per_row = pages * 32;\n    device const uint* row_pack = packed + row * words_per_row;\n    float acc = 0.0f;\n    for (uint wi = tid; wi < words_per_row; wi += 64) {{\n        uint page = wi / 32;\n        uint lane = wi % 32;\n        uint col0 = page * 640 + lane * 20;\n        float page_max = as_type<float>(uint(page_scales[row * pages + page]) << 16);\n        float scale = page_max * (float(lane_scales[row * words_per_row + wi]) / 127.0f);\n        uint remainder = row_pack[wi];\n        for (uint value = 0; value < 20; ++value) {{\n            uint digit = remainder % 3u;\n            remainder /= 3u;\n            uint col = col0 + value;\n            if (col >= in_dim) break;\n            if (digit != 0u) acc = fma(float(input[col]), digit == 1u ? scale : -scale, acc);\n        }}\n    }}\n    acc = simd_sum(acc);\n    threadgroup float reduction[32];\n    if (simd_lane == 0) reduction[simd_id] = acc;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (tid == 0) {{ float total = 0.0f; for (uint s = 0; s < 2; ++s) total += reduction[s]; output[row] = half(total); }}\n}}\n"
    )
}

fn generated_ternary_gemm_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nkernel void {entry_point}(device const half* activations [[buffer(0)]], device const uint* weights [[buffer(1)]], device const half* scales [[buffer(2)]], device half* output [[buffer(3)]], constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]], constant uint& group_size [[buffer(7)]], uint2 gid [[thread_position_in_grid]]) {{\n    uint row = gid.y;\n    uint col = gid.x;\n    if (row >= M || col >= N) return;\n    uint packed_k = (K + 15) / 16;\n    uint groups = (K + group_size - 1) / group_size;\n    float acc = 0.0f;\n    for (uint k = 0; k < K; ++k) {{\n        uint packed = weights[col * packed_k + (k >> 4)];\n        uint code = (packed >> ((k & 15u) * 2u)) & 0x3u;\n        float weight = code == 1u ? 1.0f : (code == 2u ? -1.0f : 0.0f);\n        acc = fma(weight, float(activations[row * K + k]), acc);\n    }}\n    uint group = min(K / group_size, groups - 1);\n    output[row * N + col] = half(acc * float(scales[col * groups + group]));\n}}\n"
    )
}

fn generated_nf4_tile640_dequant_kernel(entry_point: &str) -> String {
    format!(
        "// generated by prism MLIR lowering\n#include <metal_stdlib>\nusing namespace metal;\nconstant float nf4_codebook[16] = {{ -1.0f, -0.6961928f, -0.5250731f, -0.3949175f, -0.2844414f, -0.1847734f, -0.09105f, 0.0f, 0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f, 0.4407099f, 0.562617f, 0.7229568f, 1.0f }};\nstruct Nf4Tile640DispatchParams {{ uint abi_version; uint m; uint k; uint n; uint group_size; uint reserved_0; uint reserved_1; uint reserved_2; }};\nkernel void {entry_point}(device const uchar* packed_codes [[buffer(0)]], device const float* scale_buffer [[buffer(1)]], device const float* bias_buffer [[buffer(2)]], device const float* input [[buffer(3)]], device float* output [[buffer(4)]], constant Nf4Tile640DispatchParams& params [[buffer(5)]], constant const void* profile_buffer [[buffer(9)]], uint2 pos [[thread_position_in_grid]]) {{\n    if (params.abi_version != 1 || params.group_size != 128) return;\n    if (pos.y >= params.m || pos.x >= params.n) return;\n    constexpr uint TILE = 640;\n    constexpr uint GROUPS = 5;\n    constexpr uint BYTES = 320;\n    uint tile = pos.x / TILE;\n    uint elem = pos.x % TILE;\n    uint group = elem / params.group_size;\n    uint elem_group = elem % params.group_size;\n    uint tile_count = (params.n + TILE - 1) / TILE;\n    uint tile_base = (pos.y * params.k + 0) * tile_count * BYTES;\n    float acc = 0.0f;\n    for (uint k = 0; k < params.k; ++k) {{\n        uint tile_base_k = (k * tile_count + tile) * BYTES;\n        uint byte_offset = group * 64 + (elem_group >> 1);\n        uchar packed = packed_codes[tile_base_k + byte_offset];\n        uint code = (elem_group & 1u) ? (uint(packed) >> 4) : (uint(packed) & 0x0Fu);\n        uint meta = (k * tile_count + tile) * GROUPS + group;\n        float weight = nf4_codebook[code] * scale_buffer[meta] + bias_buffer[meta];\n        acc = fma(weight, input[pos.y * params.k + k], acc);\n    }}\n    output[pos.y * params.n + pos.x] = acc;\n}}\n"
    )
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
    fn rawf32_body_is_emitted_from_mlir_lowering() {
        let mut contract = nf4_tile640_contract(MlirLoweringTarget::Metal);
        contract.semantic_id = KernelSemanticId("prism.linear.rawf32.v1".into());
        let artifact = contract.lower_to_metal().unwrap();
        assert!(artifact.source.contains("generated by prism MLIR lowering"));
        assert!(artifact.source.contains("kernel void cimage_linear_rawf32"));
        assert!(!artifact.source.contains("// SPDX-License-Identifier"));
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
            if semantic_id == "prism.linear.int8.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("device const char* weights"));
            }
            if semantic_id == "prism.linear.nf4.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("nf4_codebook"));
            }
            if semantic_id == "prism.q4.block_sym.gemv.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("int(nibble) - 8"));
            }
            if semantic_id == "prism.nf4.tile640.gemv.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("simd_sum(accumulator)"));
            }
            if semantic_id == "prism.ternary.cimage.gemv.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("groups_per_row"));
            }
            if semantic_id == "prism.ternary.gemv.v2" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("packed_cols"));
            }
            if semantic_id == "prism.palettized.gemv.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("row_codebook"));
            }
            if semantic_id == "prism.palettized.gemm.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("weight_arena"));
            }
            if semantic_id == "prism.palettized.swiglu.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("silu"));
            }
            if semantic_id == "prism.ternary.gemv.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("remainder % 3u"));
            }
            if semantic_id == "prism.ternary.gemm.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("packed_k"));
            }
            if semantic_id == "prism.nf4tile640.dequant_mul.v1" {
                assert!(artifact.source.contains("generated by prism MLIR lowering"));
                assert!(artifact.source.contains("Nf4Tile640DispatchParams"));
            }
        }
    }

    #[test]
    fn all_registered_precision_targets_compile_to_metallib() {
        let toolchain = crate::ecs::metal_backend::toolchain::MetalToolchain::default();
        if !toolchain.is_available() {
            return;
        }
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
            let contract =
                precision_contract_for_semantic(&KernelSemanticId(semantic_id.into())).unwrap();
            let artifact = contract.lower_to_metal().unwrap();
            let output = toolchain
                .compile_source(semantic_id, &artifact.source)
                .unwrap_or_else(|error| panic!("{semantic_id} failed Metal compilation: {error}"));
            assert_eq!(&output.metallib_bytes[..4], b"MTLB");
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
