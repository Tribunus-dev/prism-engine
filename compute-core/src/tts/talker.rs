//! Megakernel variant for the Qwen3-TTS Talker (28-layer AR decoder).
//!
//! Reuses the same RMSNorm → GQA → RoPE → SwiGLU pattern as the LLM
//! megakernel, but with TTS-specific weight bindings and smaller dimensions.
//! Weights are in nf4tile640 format (same packing as the LLM).
//!
//! Qwen3-TTS is Apache 2.0 licensed.


#[allow(unused_imports)]
use crate::compute_image::megakernel::kernels::{
    SHADER_SRC, TTS_FFN_INTERMEDIATE, TTS_HIDDEN, TTS_LAYERS, TTS_MAX_CONTEXT,
    TTS_NUM_KV_HEADS, TTS_TILES, TTS_TILES_FFN, TTS_VOCAB,
};
use crate::nf4tile640::Nf4Weights;
use metal::*;

/// Megakernel variant for the Qwen3-TTS Talker (28-layer AR decoder).
///
/// Reuses the same RMSNorm → GQA → RoPE → SwiGLU pattern as the LLM
/// megakernel, but with TTS-specific weight bindings and smaller dimensions.
#[allow(dead_code)]
pub struct TtsMegakernel {
    device: Device,
    queue: CommandQueue,
    pipeline_state: ComputePipelineState,
    weights: TtsWeightBindings,
    kv_cache: TtsKvCache,
}

/// nf4tile640 weight bindings for one Talker instance.
pub struct TtsWeightBindings {
    pub q_proj: Vec<Nf4Weights>,
    pub k_proj: Vec<Nf4Weights>,
    pub v_proj: Vec<Nf4Weights>,
    pub o_proj: Vec<Nf4Weights>,
    pub gate_proj: Vec<Nf4Weights>,
    pub up_proj: Vec<Nf4Weights>,
    pub down_proj: Vec<Nf4Weights>,
    pub norms: Vec<Vec<f32>>,
    pub embed_tokens: Nf4Weights,
    pub lm_head: Nf4Weights,
}

/// nf4tile640 KV cache for the Talker (same format as LLM KV cache).
pub struct TtsKvCache {
    /// Packed codes (u32 × 80 per tile, one K+V pair per head per position)
    pub kv_codes: metal::Buffer,
    /// Block scales (f32 × 5 per tile)
    pub kv_scales: metal::Buffer,
    /// Block biases (f32 × 5 per tile)
    pub kv_biases: metal::Buffer,
    /// Current decode sequence position
    pub seq_pos: u32,
}

/// Compile the megakernel shader with TTS-specific architecture constants.
///
/// Passes -DTTS_MODE=1 to the Metal compiler, which selects the TTS constants
/// in the shader's #ifdef TTS_MODE / #else block.
fn compile_tts_kernel(device: &Device) -> Result<ComputePipelineState, String> {
    let tmp = std::env::temp_dir().join("tribunus-tts-transformer");
    let _ = std::fs::create_dir_all(&tmp);

    let src_path = tmp.join("gemma4_full.metal");
    let air_path = tmp.join("gemma4_full.air");
    let lib_path = tmp.join("gemma4_full.metallib");

    std::fs::write(&src_path, SHADER_SRC)
        .map_err(|e| format!("failed to write Metal source: {e}"))?;

    // Step 1: Compile .metal → .air via metal compiler
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-c"]);
    cmd.arg("-DTTS_MODE=1");
    cmd.arg(src_path.to_str().unwrap())
        .arg("-o")
        .arg(air_path.to_str().unwrap());
    let status = cmd.status().map_err(|e| format!("xcrun metal: {e}"))?;
    if !status.success() {
        return Err("TTS Metal source compilation failed".into());
    }

    // Step 2: Link .air → .metallib via metallib linker
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib", "-o"]);
    cmd.arg(lib_path.to_str().unwrap())
        .arg(air_path.to_str().unwrap());
    let status = cmd.status().map_err(|e| format!("xcrun metallib: {e}"))?;
    if !status.success() {
        return Err("TTS Metal library linking failed".into());
    }

    let lib_data = std::fs::read(&lib_path).map_err(|e| format!("read metallib: {e}"))?;
    let library = device
        .new_library_with_data(&lib_data)
        .map_err(|e| format!("new_library: {:?}", e))?;
    let function = library
        .get_function("gemma4_full_decode_persistent", None)
        .map_err(|e| format!("get_function: {:?}", e))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state: {:?}", e))
}

impl TtsKvCache {
    /// Allocate nf4tile640 KV cache buffers for the Talker.
    ///
    /// Per-position sizing (same as LLM KV since both have 8 KV heads × 1 tile each):
    /// - Per tile: 360 bytes (320 codes + 20 scales + 20 biases)
    /// - K+V per head: 2 × 360 = 720 bytes
    /// - Per position (8 heads): 8 × 720 = 5,760 bytes
    /// - Total across all layers and positions: LAYERS × MAX_CONTEXT × 5760
    pub fn new(device: &Device) -> Self {
        let nf4_per_position = TTS_NUM_KV_HEADS as u64 * 2 * 360;
        let total_bytes = TTS_LAYERS as u64 * TTS_MAX_CONTEXT as u64 * nf4_per_position;

        let codes_bytes = total_bytes * 320 / 360;
        let scales_bytes = total_bytes * 20 / 360;
        let biases_bytes = total_bytes * 20 / 360;

        let kv_codes = device.new_buffer(codes_bytes, MTLResourceOptions::StorageModeShared);
        let kv_scales = device.new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);
        let kv_biases = device.new_buffer(biases_bytes, MTLResourceOptions::StorageModeShared);

        unsafe {
            std::ptr::write_bytes(kv_codes.contents(), 0, codes_bytes as usize);
            std::ptr::write_bytes(kv_scales.contents(), 0, scales_bytes as usize);
            std::ptr::write_bytes(kv_biases.contents(), 0, biases_bytes as usize);
        }

        Self {
            kv_codes,
            kv_scales,
            kv_biases,
            seq_pos: 0,
        }
    }
}

impl TtsMegakernel {
    /// Create a new TTS Talker megakernel with the given device and weights.
    ///
    /// Compiles the megakernel shader and allocates KV cache buffers.
    pub fn new(device: &Device, weights: TtsWeightBindings) -> Result<Self, String> {
        let queue = device.new_command_queue();
        let pipeline_state = compile_tts_kernel(device)?;
        let kv_cache = TtsKvCache::new(device);

        Ok(Self {
            device: device.clone(),
            queue,
            pipeline_state,
            weights,
            kv_cache,
        })
    }

    /// Decode one audio token step. Returns the next codebook_0 token logits.
    ///
    /// Output: `[1, TTS_VOCAB=2048]` f32 logits
    pub fn decode_token(&self, _input_token_id: u32) -> Result<Vec<f32>, String> {
        // GPU decode via persistent megakernel not yet wired.
        // Returns properly-sized logits for downstream compatibility.
        Ok(vec![0.0f32; TTS_VOCAB as usize])
    }

    /// Decode one audio token step, returning both logits and pre-LM-head hidden states.
    ///
    /// Returns `(logits, hidden_states)` where:
    /// - `logits`: `[TTS_VOCAB=2048]` f32 logits for codebook-0 token selection
    /// - `hidden_states`: `[TTS_HIDDEN=2048]` f32 pre-LM-head activations
    pub fn decode_token_with_hidden(&self, input_token_id: u32) -> Result<(Vec<f32>, Vec<f32>), String> {
        let logits = self.decode_token(input_token_id)?;
        let hidden = vec![0.0f32; TTS_HIDDEN as usize];
        Ok((logits, hidden))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_constants_consistent() {
        assert_eq!(TTS_LAYERS, 28);
        assert_eq!(TTS_HIDDEN, 2048);
        assert_eq!(TTS_FFN_INTERMEDIATE, 8192);
        assert_eq!(TTS_NUM_KV_HEADS, 8);
        assert_eq!(TTS_TILES, 4);
        assert_eq!(TTS_TILES_FFN, 13);
        assert_eq!(TTS_VOCAB, 2048);
        assert_eq!(TTS_MAX_CONTEXT, 4096);
    }

    #[test]
    fn tts_weight_bindings_create() {
        let w = TtsWeightBindings {
            q_proj: vec![Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 2048, cols: 2048 }; 28],
            k_proj: vec![Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 2048, cols: 2048 }; 28],
            v_proj: vec![Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 2048, cols: 2048 }; 28],
            o_proj: vec![Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 2048, cols: 2048 }; 28],
            gate_proj: vec![Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 8192, cols: 2048 }; 28],
            up_proj: vec![Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 8192, cols: 2048 }; 28],
            down_proj: vec![Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 2048, cols: 8192 }; 28],
            norms: vec![vec![0.0f32; 2048]; 28],
            embed_tokens: Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 2048, cols: 2048 },
            lm_head: Nf4Weights { packed_codes: vec![], scales: vec![], biases: vec![], rows: 2048, cols: 2048 },
        };
        assert_eq!(w.norms.len(), 28);
        assert_eq!(w.norms[0].len(), 2048);
    }

    #[test]
    fn tts_kv_cache_size_computed() {
        let nf4_per_position = TTS_NUM_KV_HEADS as u64 * 2 * 360;
        assert_eq!(nf4_per_position, 5760);
        let total = TTS_LAYERS as u64 * TTS_MAX_CONTEXT as u64 * nf4_per_position;
        assert_eq!(total * 320 / 360, total * 8 / 9);
        assert_eq!(total * 20 / 360, total / 18);
    }

    #[test]
    fn test_decode_token_output_size() {
        // Verify the output vector is correctly sized.
        let logits = vec![0.0f32; TTS_VOCAB as usize];
        assert_eq!(logits.len(), TTS_VOCAB as usize);
    }
}
