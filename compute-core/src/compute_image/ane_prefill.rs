//! ANE batched prefill model — one transformer layer.
//!
//! One transformer layer compiled as an optimized MIL program with all
//! hardware optimizations:
//!
//!   - **4D [B, C, 1, S] NCHW layout** — exposes the S dimension for ANE
//!     pipeline efficiency
//!   - **64-byte alignment** — pads all dimensions to 64-byte IOSurface
//!     alignment boundaries
//!   - **Conv2d 1x1** — replaces matmuls with convolution ops that map
//!     better to ANE compute units
//!   - **IOSurface weight inputs** — weights are runtime inputs (not
//!     constants), so the same compiled program runs for all 48 layers
//!     with per-layer weights streamed from the .cimage
//!
//! # Layer Ops Graph
//!
//! ```text
//! hidden_state [B, C, 1, S] FP16
//!   ├── conv2d(w_q)       ──→ Q [B, d_q, 1, S] FP16  ──→ GPU attention
//!   ├── conv2d(w_k)       ──→ K [B, d_k, 1, S] FP16  ──→ GPU attention
//!   ├── conv2d(w_v)       ──→ V [B, d_v, 1, S] FP16  ──→ GPU attention
//!   ├── conv2d(w_gate)    ──→ SiLU ──→ gate_silu
//!   ├── conv2d(w_up)      ──→ up
//!   ├── mul(gate_silu, up) ──→ gated
//!   ├── conv2d(w_down)    ──→ mlp_out
//!   └── add(mlp_out, hidden) ──→ next_hidden [B, C, 1, S] FP16
//! ```
//!
//! # GPU handles (separate from this program)
//!
//! - Attention: QK^T, softmax, weighted V sum
//! - O projection: attention_output @ w_o
//! - Final residual: O_output + next_hidden

use coreml_proto::proto::mil_spec;
use std::path::Path;

use crate::mil_builder::MilBuilder;
use crate::mlpackage::{self, ModelMeta};

// ── Alignment ──────────────────────────────────────────────────────────────

/// Pad dimension to nearest 64-byte boundary given element byte size.
///
/// For FP16 (2 bytes/element): pads to multiple of 32 elements.
/// For INT8 (1 byte/element): pads to multiple of 64 elements.
pub fn align_dim(dim: u32, element_bytes: u32) -> u32 {
    let align_bytes = 64u32;
    let bytes = dim * element_bytes;
    let padded = ((bytes + align_bytes - 1) / align_bytes) * align_bytes;
    padded / element_bytes
}

// ── Architecture metadata ──────────────────────────────────────────────────

/// Aligned dimension layout for one transformer layer's ANE prefill.
///
/// All dimensions are 64-byte aligned. The MIL program accepts these
/// as IOSurface inputs and produces the corresponding outputs.
pub struct PrefillLayerInfo {
    /// Aligned hidden (channel) dimension — [B, C, 1, S] input.
    pub c: u32,
    /// Aligned S dimension (always 32 for FP16 batch=1 on 64-byte boundary).
    pub s: u32,
    /// Aligned Q projection output dimension.
    pub d_q: u32,
    /// Aligned K projection output dimension.
    pub d_k: u32,
    /// Aligned V projection output dimension.
    pub d_v: u32,
    /// Aligned intermediate FFN dimension.
    pub d_ff: u32,
    /// Raw (unaligned) hidden dimension — for weight offset calculations.
    pub hidden_raw: u32,
    /// Raw Q output dimension.
    pub d_q_raw: u32,
    /// Raw K output dimension.
    pub d_k_raw: u32,
    /// Raw V output dimension.
    pub d_v_raw: u32,
    /// Raw FFN intermediate dimension.
    pub d_ff_raw: u32,
}

impl PrefillLayerInfo {
    /// Compute aligned dimensions for given architecture constants.
    pub fn for_architecture(hidden: u32, d_q: u32, d_k: u32, d_v: u32, d_ff: u32) -> Self {
        Self {
            c: align_dim(hidden, 2),
            s: align_dim(1, 2),
            d_q: align_dim(d_q, 2),
            d_k: align_dim(d_k, 2),
            d_v: align_dim(d_v, 2),
            d_ff: align_dim(d_ff, 2),
            hidden_raw: hidden,
            d_q_raw: d_q,
            d_k_raw: d_k,
            d_v_raw: d_v,
            d_ff_raw: d_ff,
        }
    }

    /// Gemma4 12B default dimensions.
    pub fn gemma4_12b() -> Self {
        Self::for_architecture(3840, 4096, 2048, 2048, 15360)
    }
}

// ── MIL program builder ───────────────────────────────────────────────────

/// Build the MIL program for one transformer layer.
///
/// Inputs (7 IOSurface FP16 tensors):
///   - `hidden_state` — [B, C, 1, S] input activations
///   - `w_q` — [d_q, C, 1, 1] Q projection Conv2d weight
///   - `w_k` — [d_k, C, 1, 1] K projection Conv2d weight
///   - `w_v` — [d_v, C, 1, 1] V projection Conv2d weight
///   - `w_gate` — [d_ff, C, 1, 1] Gate projection Conv2d weight
///   - `w_up` — [d_ff, C, 1, 1] Up projection Conv2d weight
///   - `w_down` — [C, d_ff, 1, 1] Down projection Conv2d weight
///
/// Outputs (4 FP16 tensors):
///   - `Q` — [B, d_q, 1, S] to GPU attention
///   - `K` — [B, d_k, 1, S] to GPU attention
///   - `V` — [B, d_v, 1, S] to GPU attention
///   - `next_hidden` — [B, C, 1, S] residual output (MLP + skip connection)
pub fn generate_prefill_mil(
    batch: u32,
    info: &PrefillLayerInfo,
) -> Result<mil_spec::Program, String> {
    let b = batch as i64;
    let c = info.c as i64;
    let s = info.s as i64;
    let d_q = info.d_q as i64;
    let d_k = info.d_k as i64;
    let d_v = info.d_v as i64;
    let d_ff = info.d_ff as i64;

    // ── Declare all inputs (hidden state + 6 weight matrices) ─────────
    let mut builder = MilBuilder::new("main")
        .set_opset("CoreML9")
        // Hidden state [B, C, 1, S] FP16
        .input("hidden_state", mil_spec::DataType::Float16, &[b, c, 1, s])
        // Weight matrices [C_out, C_in, 1, 1] FP16 for Conv2d 1x1
        .input("w_q", mil_spec::DataType::Float16, &[d_q, c, 1, 1])
        .input("w_k", mil_spec::DataType::Float16, &[d_k, c, 1, 1])
        .input("w_v", mil_spec::DataType::Float16, &[d_v, c, 1, 1])
        .input("w_gate", mil_spec::DataType::Float16, &[d_ff, c, 1, 1])
        .input("w_up", mil_spec::DataType::Float16, &[d_ff, c, 1, 1])
        .input("w_down", mil_spec::DataType::Float16, &[c, d_ff, 1, 1]);

    // ── Step 1: Q, K, V projections via Conv2d 1x1 ────────────────────
    builder = builder.conv("conv_q", "hidden_state", "w_q", &[1, 1], "valid");
    let q_name = builder.last_name().ok_or("conv_q SSA name")?.to_string();

    builder = builder.conv("conv_k", "hidden_state", "w_k", &[1, 1], "valid");
    let k_name = builder.last_name().ok_or("conv_k SSA name")?.to_string();

    builder = builder.conv("conv_v", "hidden_state", "w_v", &[1, 1], "valid");
    let v_name = builder.last_name().ok_or("conv_v SSA name")?.to_string();

    // ── Step 2: Gate projection → SiLU activation ─────────────────────
    builder = builder.conv("conv_gate", "hidden_state", "w_gate", &[1, 1], "valid");
    let gate_name = builder.last_name().ok_or("conv_gate SSA name")?.to_string();
    builder = builder.silu("gate_silu", &gate_name);
    let gate_silu_name = builder.last_name().ok_or("gate_silu SSA name")?.to_string();

    // ── Step 3: Up projection ────────────────────────────────────────
    builder = builder.conv("conv_up", "hidden_state", "w_up", &[1, 1], "valid");
    let up_name = builder.last_name().ok_or("conv_up SSA name")?.to_string();

    // ── Step 4: Gate * Up (element-wise multiply) ────────────────────
    builder = builder.mul(&gate_silu_name, &up_name);
    let gated_name = builder.last_name().ok_or("gated mul SSA name")?.to_string();

    // ── Step 5: Down projection ──────────────────────────────────────
    builder = builder.conv("conv_down", &gated_name, "w_down", &[1, 1], "valid");
    let down_name = builder.last_name().ok_or("conv_down SSA name")?.to_string();

    // ── Step 6: Residual add (mlp_out + hidden_state) ────────────────
    builder = builder.add(&down_name, "hidden_state");
    let output_name = builder.last_name().ok_or("add SSA name")?.to_string();

    // ── Declare all 4 outputs ─────────────────────────────────────────
    builder = builder
        .output(&q_name)
        .output(&k_name)
        .output(&v_name)
        .output(&output_name);

    builder
        .build()
        .map_err(|e| format!("prefill MIL build error: {e}"))
}

// ── Compilation ───────────────────────────────────────────────────────────

/// Compile a prefill MIL program to a `.mlmodelc` bundle.
///
/// Writes the MIL program as an `.mlpackage`, compiles with `xcrun
/// coremlcompiler`, and returns the path to the compiled `.mlmodelc`
/// directory.
pub fn compile_prefill_mil(
    prog: mil_spec::Program,
    output_dir: &Path,
    tag: &str,
) -> Result<std::path::PathBuf, String> {
    let meta = ModelMeta {
        model_name: format!("prefill_{tag}"),
        function_name: "main".into(),
        short_description: format!("ANE prefill layer ({tag})"),
        version: env!("CARGO_PKG_VERSION").to_string(),
        author: "prism-engine".to_string(),
        output_name: "next_hidden".into(),
        inputs: vec![],
        outputs: vec![
            ("Q".into(), vec![1, 4096, 1, 32]),      // 16 heads × 256 dim
            ("K".into(), vec![1, 2048, 1, 32]),      // 8 KV heads × 256 dim
            ("V".into(), vec![1, 2048, 1, 32]),      // 8 KV heads × 256 dim
            ("next_hidden".into(), vec![1, 3840, 1, 32]),  // hidden dim
        ],
    };

    let pkg = mlpackage::write_mlpackage(prog, output_dir, &meta)
        .map_err(|e| format!("mlpackage: {e}"))?;

    let compiled_dir = output_dir.join("compiled");
    std::fs::create_dir_all(&compiled_dir).map_err(|e| format!("create compiled dir: {e}"))?;

    let receipt = crate::coreml_pipeline::compile_mlpackage(
        &pkg,
        &compiled_dir,
        tag,
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("compile: {e}"))?;

    Ok(std::path::PathBuf::from(&receipt.compiled_modelc_path))
}

/// Read the compiled model.mlmodel protobuf bytes from a `.mlmodelc` bundle.
pub fn read_modelc_proto(modelc_path: &Path) -> Result<Vec<u8>, String> {
    let model_file = modelc_path.join("model.mlmodel");
    if !model_file.exists() {
        return Err(format!("model.mlmodel not found in {:?}", modelc_path));
    }
    std::fs::read(&model_file).map_err(|e| format!("read {:?}: {e}", model_file))
}

// ── Runtime execution ─────────────────────────────────────────────────────

/// Weight buffer indices for one prefill layer.
///
/// Order matches the MIL program's weight input names:
///   0: w_q, 1: w_k, 2: w_v, 3: w_gate, 4: w_up, 5: w_down
#[derive(Debug, Clone, Copy)]
pub enum WeightInput {
    WQ = 0,
    WK = 1,
    WV = 2,
    WGate = 3,
    WUp = 4,
    WDown = 5,
}

/// Number of weight inputs per layer.
pub const NUM_WEIGHT_INPUTS: usize = 6;

/// Run one layer prefill on the ANE.
///
/// Sends the hidden state and 6 weight matrices as IOSurface inputs,
/// receives 4 output tensors (Q, K, V, next_hidden).
///
/// # Arguments
/// * `model` — Loaded CoreML model (the compiled MIL program)
/// * `hidden_arena` — Input [B, C, 1, S] FP16 hidden state
/// * `weight_arenas` — 6 weight arenas indexed by [`WeightInput`]
/// * `q_arena` — Output Q [B, d_q, 1, S]
/// * `k_arena` — Output K [B, d_k, 1, S]
/// * `v_arena` — Output V [B, d_v, 1, S]
/// * `next_hidden_arena` — Output next_hidden [B, C, 1, S]
pub fn run_prefill_layer(
    model: &crate::coreml_bridge::CoreMlModel,
    hidden_arena: &crate::arena::Arena,
    weight_arenas: &[&crate::arena::Arena; NUM_WEIGHT_INPUTS],
    q_arena: &mut crate::arena::Arena,
    k_arena: &mut crate::arena::Arena,
    v_arena: &mut crate::arena::Arena,
    next_hidden_arena: &mut crate::arena::Arena,
) -> Result<(), String> {
    let input_names = [
        "hidden_state",
        "w_q",
        "w_k",
        "w_v",
        "w_gate",
        "w_up",
        "w_down",
    ];
    let input_arenas: [&crate::arena::ArenaInfo; 7] = [
        &hidden_arena.info,
        &weight_arenas[WeightInput::WQ as usize].info,
        &weight_arenas[WeightInput::WK as usize].info,
        &weight_arenas[WeightInput::WV as usize].info,
        &weight_arenas[WeightInput::WGate as usize].info,
        &weight_arenas[WeightInput::WUp as usize].info,
        &weight_arenas[WeightInput::WDown as usize].info,
    ];
    // Need to collect because we need mutable refs for output arenas
    let mut q_info = q_arena.info;
    let mut k_info = k_arena.info;
    let mut v_info = v_arena.info;
    let mut next_info = next_hidden_arena.info;
    let output_names = ["Q", "K", "V", "next_hidden"];
    let mut output_infos = [&mut q_info, &mut k_info, &mut v_info, &mut next_info];

    model
        .predict_multi(
            &input_names,
            &input_arenas,
            &output_names,
            &mut output_infos,
        )
        .map_err(|e| format!("ANE prefill layer predict: {e}"))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn generate_prefill_mil_acceptance() {
        let info = PrefillLayerInfo::gemma4_12b();
        assert_eq!(info.c, 3840, "hidden 3840 already 64B-aligned for FP16");
        assert_eq!(info.s, 32, "S=1 padded to 32 for FP16 64B alignment");
        assert_eq!(info.d_q, 4096);
        assert_eq!(info.d_k, 2048);
        assert_eq!(info.d_v, 2048);
        assert_eq!(info.d_ff, 15360);

        let prog = generate_prefill_mil(1, &info).unwrap();
        assert_eq!(prog.version, 1);

        let func = prog.functions.get("main").unwrap();
        let block = func.block_specializations.get("CoreML9").unwrap();
        let ops = &block.operations;

        // Expected ops in order:
        //   0: conv(conv_q, hidden_state, w_q)  → Q
        //   1: conv(conv_k, hidden_state, w_k)  → K
        //   2: conv(conv_v, hidden_state, w_v)  → V
        //   3: conv(conv_gate, hidden_state, w_gate) → gate
        //   4: silu(gate_silu, gate)             → gate_silu
        //   5: conv(conv_up, hidden_state, w_up) → up
        //   6: mul(mul_6, gate_silu, up)         → gated
        //   7: conv(conv_down, gated, w_down)    → down
        //   8: add(add_8, down, hidden_state)    → output
        assert_eq!(ops.len(), 9, "expected 9 ops, got {}", ops.len());

        // Verify operation types in order
        let op_types: Vec<&str> = ops.iter().map(|o| o.r#type.as_str()).collect();
        assert_eq!(
            op_types,
            vec!["conv", "conv", "conv", "conv", "silu", "conv", "mul", "conv", "add"],
            "op types must match expected MLP+projection pipeline"
        );

        // Verify outputs
        assert_eq!(block.outputs.len(), 4, "expected 4 outputs");
        // Outputs are auto-generated SSA names:
        //   conv_q_0, conv_k_1, conv_v_2 (from conv ops), add_8 (from add op)
        assert!(
            block.outputs.iter().any(|o| o.starts_with("conv_q")),
            "Q projection output"
        );
        assert!(
            block.outputs.iter().any(|o| o.starts_with("conv_k")),
            "K projection output"
        );
        assert!(
            block.outputs.iter().any(|o| o.starts_with("conv_v")),
            "V projection output"
        );
        assert!(
            block.outputs.iter().any(|o| o.starts_with("add")),
            "residual add output"
        );

        // Verify input count (hidden + 6 weight inputs)
        // Inputs are stored at the function level, not block level
        let func_inputs = &func.inputs;
        assert_eq!(func_inputs.len(), 7, "expected 7 function inputs");
        // Verify specific input names
        let input_names: Vec<&str> = func_inputs.iter().map(|nv| nv.name.as_str()).collect();
        assert!(
            input_names.contains(&"hidden_state"),
            "hidden_state input present"
        );
        assert!(input_names.contains(&"w_q"), "w_q input present");
        assert!(input_names.contains(&"w_k"), "w_k input present");
        assert!(input_names.contains(&"w_v"), "w_v input present");
        assert!(input_names.contains(&"w_gate"), "w_gate input present");
        assert!(input_names.contains(&"w_up"), "w_up input present");
        assert!(input_names.contains(&"w_down"), "w_down input present");

        // Verify protobuf serialization
        let bytes = prog.encode_to_vec();
        assert!(!bytes.is_empty(), "protobuf must not be empty");
        assert!(bytes.len() > 100, "protobuf should have meaningful size");
    }

    #[test]
    fn align_dim_correctness() {
        // FP16: 2 bytes per element, 64-byte boundary = 32 elements
        assert_eq!(align_dim(1, 2), 32, "S=1 → 32 for FP16");
        assert_eq!(align_dim(32, 2), 32, "S=32 exact fit");
        assert_eq!(align_dim(33, 2), 64, "S=33 → 64");
        assert_eq!(align_dim(3840, 2), 3840, "hidden 3840 already aligned");
        assert_eq!(align_dim(4096, 2), 4096, "d_q 4096 already aligned");
        assert_eq!(align_dim(15360, 2), 15360, "d_ff 15360 already aligned");

        // INT8: 1 byte per element, 64-byte boundary = 64 elements
        assert_eq!(align_dim(1, 1), 64, "1 INT8 element → 64");
        assert_eq!(align_dim(64, 1), 64);
        assert_eq!(align_dim(65, 1), 128);

        // Edge: zero dims
        assert_eq!(align_dim(0, 2), 0);
        assert_eq!(align_dim(0, 1), 0);
    }

    #[test]
    fn prefill_layer_info_gemma4() {
        let info = PrefillLayerInfo::gemma4_12b();
        assert_eq!(info.hidden_raw, 3840);
        assert_eq!(info.d_q_raw, 4096);
        assert_eq!(info.d_k_raw, 2048);
        assert_eq!(info.d_v_raw, 2048);
        assert_eq!(info.d_ff_raw, 15360);
    }
}
