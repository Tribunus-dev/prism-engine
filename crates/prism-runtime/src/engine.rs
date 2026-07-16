//! Prism Engine — unified inference runtime for `.cimage` models.
//!
//! The full PrismEngine (tensor loading, Metal/ANE backends, generate) will
//! be ported here over subsequent milestones. Currently a minimal loader that
//! opens the `.cimage` via Model::load and stores the graph for later use.

use crate::inference::{InferenceEngine, KvCache};
use crate::model::Model;
use crate::multimodal::{MultimodalCallback, MultimodalInput, StreamEvent, TokenMetrics};
use crate::streaming::StreamingLayerLoader;
use prism_ecs_ir::model_graph::ModelGraph;
use prism_multimodal::multimodal::vision_encoder::{
    self, MatmulProvider, VisionArch, VisionEncoderConfig,
};
use prism_video::{generate_video, VideoParams};
use std::collections::HashMap;
use std::path::Path;

/// Per-inference statistics returned by [`PrismEngine::generate`].
#[derive(Debug, Default)]
pub struct InferenceStats {
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub total_time_ms: f64,
}

/// Inference engine for `.cimage` models.
///
/// Wraps the loaded [`Model`] together with a [`ModelGraph`] that describes
/// the compute topology.  `load()` parses the header and records tensor
/// metadata; payloads are read on demand.
pub struct PrismEngine {
    /// Wrapped model — tensor metadata and header info.
    pub model: Model,
    /// Compute graph describing the transformer architecture.
    pub graph: ModelGraph,
    /// Vision encoder weights, keyed by tensor name.
    /// Loaded separately from the main model for multimodal inference.
    pub vision_weights: HashMap<String, Vec<f32>>,
    /// Vision encoder architecture configuration.
    pub vision_config: VisionEncoderConfig,
    /// Matmul provider for f32 vision-encoder linear projections.
    pub matmul_provider: MatmulProvider,
}

impl PrismEngine {
    /// Generate video frames from a text prompt and stream them via callback.
    ///
    /// Uses the image-generation provider from `prism-video` to produce
    /// keyframes, then fills intermediate frames via interpolation.
    /// Each frame is emitted as a [`StreamEvent::VideoFrame`] on the
    /// given callback.
    ///
    /// # Arguments
    /// - `prompt`: text description of the desired video.
    /// - `params`: video generation parameters (frame count, FPS, seed).
    /// - `callback`: streaming callback that receives `VideoFrame` events.
    pub fn generate_video(
        &self,
        prompt: &str,
        params: VideoParams,
        callback: &mut dyn MultimodalCallback,
    ) -> Result<(), String> {
        // image_provider callback: generate a flat RGBA8888 frame at the
        // requested dimensions. For now returns a solid-color frame to keep
        // the pipeline live; a real vision model would be wired here.
        let image_provider = |_prompt: &str, width: u32, height: u32| {
            let size = (width * height * 4) as usize;
            Ok(vec![128u8; size]) // mid-gray placeholder
        };

        // No model path needed for the pipeline fallback — pass empty string
        let receipt = generate_video("", prompt, params.clone(), &image_provider)
            .map_err(|e| format!("video generation failed: {e}"))?;

        let frame_duration_ns = if params.fps > 0 {
            1_000_000_000u64 / params.fps as u64
        } else {
            33_333_333 // ~30 fps default
        };

        for (i, (_width, _height, frame_data)) in receipt.frames.iter().enumerate() {
            callback.on_event(StreamEvent::VideoFrame {
                bytes: frame_data.clone(),
                width: *_width,
                height: *_height,
                timestamp_ns: i as u64 * frame_duration_ns,
            });
        }

        callback.on_done();
        Ok(())
    }

    /// Multimodal generation with streaming callback.
    /// Open a `.cimage` path and pair it with a `ModelGraph`.
    ///
    /// Parses the .cimage header via [`Model::load`] and stores the
    /// graph for subsequent inference dispatch.
    pub fn load(path: &Path, graph: ModelGraph) -> Result<Self, String> {
        let model = Model::load(path)?;
        Ok(PrismEngine {
            model,
            graph,
            vision_weights: HashMap::new(),
            vision_config: VisionEncoderConfig {
                arch: VisionArch::ClipVitL,
                input_size: (224, 224),
                patch_size: 14,
                num_layers: 24,
                hidden_dim: 1024,
                num_heads: 16,
            },
            matmul_provider: make_f32_matmul_provider(),
        })
    }

    /// Set the vision encoder configuration for multimodal inference.
    pub fn with_vision_encoder(mut self, config: VisionEncoderConfig) -> Self {
        self.vision_config = config;
        self
    }

    /// Load vision encoder weights from an f32 weight map.
    pub fn load_vision_weights(&mut self, weights: HashMap<String, Vec<f32>>) {
        self.vision_weights = weights;
    }

    /// Enable Metal GPU acceleration.
    #[cfg(feature = "metal-dispatch")]
    pub fn with_metal(&mut self) -> Result<(), String> {
        eprintln!("[prism] Metal dispatch not yet wired in prism-runtime stub");
        Ok(())
    }

    /// Generate tokens from a prompt.
    ///
    /// # Arguments
    /// - `prompt_tokens`: input token IDs.
    /// - `max_tokens`: maximum number of tokens to generate.
    ///
    /// # Returns
    /// Generation statistics including the generated token sequence.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<InferenceStats, String> {
        if prompt_tokens.is_empty() {
            return Err("generate: empty prompt".to_string());
        }

        // Build an inference engine from the loaded model.
        // The engine auto-detects model architecture from tensor metadata.
        let engine = InferenceEngine::new(self.model.clone());

        // Estimate KV cache parameters from the model graph.
        let num_layers = self.graph.num_layers as usize;
        let head_dim = Self::detect_head_dim(&engine);
        let num_kv_heads = Self::detect_num_kv_heads(&engine);
        let max_seq_len = prompt_tokens.len() + max_tokens;

        let mut kv_cache = KvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);
        let t0 = std::time::Instant::now();

        // Prefill: run forward on all prompt tokens at once.
        // This populates the KV cache for the entire prompt.
        let mut logits = engine.forward(prompt_tokens, &mut kv_cache)?;

        // Autoregressive generation loop
        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next_token = crate::sampling::sample(&logits, &Default::default());
            generated.push(next_token);

            // EOS token (typically 0, but models vary; 0 is the most common pad/EOS)
            if next_token == 0 {
                break;
            }

            // Forward pass for the single new token
            logits = engine.forward(&[next_token], &mut kv_cache)?;
        }

        Ok(InferenceStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            total_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Generate tokens using streamed layer loading.
    ///
    /// Same as generate() but loads one layer at a time via
    /// StreamingLayerLoader for models exceeding available RAM.
    pub fn generate_streamed(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        streamer: &StreamingLayerLoader,
    ) -> Result<InferenceStats, String> {
        if prompt_tokens.is_empty() {
            return Err("generate_streamed: empty prompt".to_string());
        }

        let engine = InferenceEngine::new(self.model.clone());

        let num_layers = self.graph.num_layers as usize;
        let head_dim = Self::detect_head_dim(&engine);
        let num_kv_heads = Self::detect_num_kv_heads(&engine);
        let max_seq_len = prompt_tokens.len() + max_tokens;

        let mut kv_cache = KvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);
        let t0 = std::time::Instant::now();

        // Prefill: process all prompt tokens
        let mut logits = engine.forward_streamed(prompt_tokens, &mut kv_cache, streamer)?;

        // Autoregressive generation
        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next_token = crate::sampling::sample(&logits, &Default::default());
            generated.push(next_token);
            if next_token == 0 {
                break;
            }
            logits = engine.forward_streamed(&[next_token], &mut kv_cache, streamer)?;
        }

        Ok(InferenceStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            total_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Detect head_dim from engine config.
    fn detect_head_dim(_engine: &InferenceEngine) -> usize {
        // Use the ctx from inference.InferenceEngine which auto-detects this
        // from the model tensor metadata. Default to 128 if somehow missing.
        // TODO: expose config publicly for cleaner access.
        // For now just use a reasonable default — most LLaMA-family models
        // have head_dim = hidden_size / num_heads.
        128
    }

    /// Detect num_kv_heads from engine config.
    fn detect_num_kv_heads(_engine: &InferenceEngine) -> usize {
        // Same as head_dim — the engine's config auto-detects this.
        // Default to num_heads (no GQA) which is the common case for smaller models.
        32
    }

    /// Return the hidden dimension of the embedding layer.
    pub fn embedding_dim(&self) -> u32 {
        for node in &self.graph.nodes {
            if let prism_ecs_ir::model_graph::ComputeNode::TokenEmbedding { hidden_dim, .. } = node
            {
                return *hidden_dim;
            }
        }
        896
    }

    /// Tokenize text using the model's tokenizer.
    /// Falls back to simple whitespace splitting if no tokenizer path in metadata.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let tokenizer_path = self.model.metadata["tokenizer_path"].as_str().unwrap_or("");
        if tokenizer_path.is_empty() {
            // Fallback: simple whitespace split for testing
            return Ok(text.split_whitespace().map(|_| 1u32).collect());
        }
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| format!("encode: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Detokenize token IDs back to text.
    pub fn detokenize(&self, ids: &[u32]) -> Result<String, String> {
        let tokenizer_path = self.model.metadata["tokenizer_path"].as_str().unwrap_or("");
        if tokenizer_path.is_empty() {
            return Ok(format!("<token {}>", ids.first().copied().unwrap_or(0)));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        tokenizer
            .decode(ids, true)
            .map_err(|e| format!("decode: {e}"))
    }

    /// Multimodal generation with streaming callback.
    ///
    /// Accepts text + optional images/audio, runs inference, and streams
    /// results via the callback trait.
    pub fn generate_multimodal(
        &mut self,
        input: MultimodalInput,
        max_tokens: usize,
        callback: &mut dyn MultimodalCallback,
    ) -> Result<(), String> {
        // Video output — if the user requested video generation, produce frames
        if input.text.contains("generate_video") || input.text.contains("--video") {
            let params = VideoParams {
                num_frames: 16,
                fps: 8,
                seed: 42,
            };
            return self.generate_video(&input.text, params, callback);
        }

        // Tokenize text
        let prompt_tokens = self.tokenize(&input.text)?;

        // Fuse vision embeddings if images are present
        let fused_input: Vec<u32> = if !input.images.is_empty() {
            if self.vision_weights.is_empty() {
                return Err(
                    "generate_multimodal: image input requires vision weights, but none loaded"
                        .to_string(),
                );
            }

            // Encode each image through the vision encoder
            for img in &input.images {
                let _embeddings = vision_encoder::encode_image(
                    &img.data,
                    &self.vision_config,
                    &self.vision_weights,
                    &self.matmul_provider,
                )?;
                // TODO: fuse image embeddings with text tokens
                // This requires injecting image embeddings into the token embedding
                // sequence at the appropriate positions. The image embeddings should
                // be projected to the LLM's hidden dimension and placed where the
                // <image> placeholder tokens are in the text token sequence.
                //
                // Future work: replace <image> placeholder tokens in the tokenized
                // text with the vision encoder's patch embeddings, creating a fused
                // input sequence that the LLM processes as its hidden state.
                let _ = _embeddings;
            }

            prompt_tokens
        } else {
            prompt_tokens
        };

        if input.audio.is_some() {
            eprintln!(
                "[prism] multimodal: audio input received but audio encoder not yet implemented"
            );
        }

        let stats = self.generate(&fused_input, max_tokens)?;

        let total_time_s = (stats.total_time_ms / 1000.0).max(0.001);
        let tps = stats.prompt_tokens as f64 / total_time_s;

        // Stream tokens via callback
        for (i, token_id) in stats.generated_tokens.iter().enumerate() {
            let token_str = self.detokenize(&[*token_id])?;
            callback.on_event(StreamEvent::Text {
                token: token_str,
                index: i as u64,
                metrics: TokenMetrics {
                    tokens_per_sec: tps,
                    time_ms: stats.total_time_ms,
                    layer: 0,
                },
            });
        }

        callback.on_done();
        Ok(())
    }
}

/// Build a MatmulProvider that does simple f32 GEMV (no quantization).
///
/// For real quantized weights, use [`cpu::matmul`] with proper TensorFormat
/// dispatch instead. This f32-only provider is the minimum for vision-encoder
/// forward passes where weights are already dequantized to f32.
fn make_f32_matmul_provider() -> MatmulProvider {
    MatmulProvider {
        matmul: Box::new(|input: &[f32], weight: &[f32], dim_m: u32, dim_n: u32| {
            let m = dim_m as usize;
            let n = dim_n as usize;
            if input.len() != n {
                return Err(format!(
                    "f32 matmul: input len {} != dim_n {}",
                    input.len(),
                    n
                ));
            }
            if weight.len() < n * m {
                return Err(format!(
                    "f32 matmul: weight len {} < dim_n {} * dim_m {}",
                    weight.len(),
                    n,
                    m
                ));
            }
            let mut out = vec![0.0f32; m];
            // GEMV: out[j] = sum_i input[i] * weight[i * m + j]
            for j in 0..m {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += input[i] as f64 * weight[i * m + j] as f64;
                }
                out[j] = s as f32;
            }
            Ok(out)
        }),
    }
}
