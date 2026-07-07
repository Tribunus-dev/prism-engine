//! Full TTS pipeline orchestrator: text token IDs → PCM waveform.
//!
//! Owns the three TTS sub-components (Talker, Code Predictor, Mimi Codec)
//! and wires them into a single `generate()` call.

use crate::compute_image::cimage_loader::CimageDeployment;
use crate::compute_image::compile::ternary::SegmentEntry;
use crate::nf4tile640::{self, Nf4Weights};
use crate::tts::code_predictor::TtsCodePredictor;
use crate::tts::codec::MimiCodec;
use crate::tts::talker::TtsMegakernel;
use crate::tts::talker::TtsWeightBindings;
use metal::Device;

/// TTS pipeline: text tokens → PCM waveform via Talker AR decode,
/// Code Predictor RVQ completion, and Mimi Codec waveform synthesis.
pub struct TtsPipeline {
    talker: TtsMegakernel,
    code_predictor: TtsCodePredictor,
    codec: MimiCodec,
}

/// TTS hidden dimension (pre-LM-head state size).
const TTS_HIDDEN: usize = 2048;

/// TTS vocabulary size (codebook-0 logits).
const TTS_VOCAB: usize = 2048;

/// BOS token ID for TTS.
const TTS_BOS: u32 = 0;
/// EOS token ID for TTS.
const TTS_EOS: u32 = 1;

/// Number of RVQ codebooks produced by the Code Predictor.
const NUM_CODEBOOKS: usize = 16;

// ── TTS segment kind constants (matching TtsCimageSegments) ────────────────
const TTS_TALKER_WEIGHT: u32 = 30;
const TTS_TALKER_SCALE: u32 = 31;
const TTS_TALKER_BIAS: u32 = 32;
const TTS_CP_WEIGHT: u32 = 33;
const TTS_CP_SCALE: u32 = 34;
const TTS_CP_BIAS: u32 = 35;
const TTS_CODEC_WEIGHT: u32 = 36;
const TTS_CODEC_CODEBOOK: u32 = 37;

// ── Talker architecture constants ──────────────────────────────────────────
/// Talker hidden dimension (model width).
const TALKER_HIDDEN: u32 = 2048;
/// Talker FFN intermediate dimension (SwiGLU).
const TALKER_FFN_INTERMEDIATE: u32 = 8192;
/// Number of Talker transformer layers.
const TTS_NUM_LAYERS: usize = 28;
/// Talker vocabulary size (embedding table rows).
const TALKER_VOCAB_SIZE: u32 = 151936;

impl TtsPipeline {
    /// Build the TTS pipeline from TTS weight segments in a cimage deployment.
    ///
    /// Expects TTS segments in the cimage (SegmentKind values 30-37):
    /// - TtsTalkerWeight(30), TtsTalkerScale(31), TtsTalkerBias(32)
    /// - TtsCodePredictorWeight(33), TtsCodePredictorScale(34), TtsCodePredictorBias(35)
    /// - TtsCodecWeight(36), TtsCodebook(37)
    pub fn from_cimage(deployment: &CimageDeployment, device: &Device) -> Result<Self, String> {
        let header = deployment
            .prism_header()
            .ok_or_else(|| "TTS requires v2 (Prism) cimage format".to_string())?;

        // ---- 1. Talker (weights/scales/biases in three parallel segments) --
        let talker_w = find_seg(&header.segments, TTS_TALKER_WEIGHT)
            .ok_or_else(|| "cimage missing TtsTalkerWeight segment".to_string())?;
        let talker_s = find_seg(&header.segments, TTS_TALKER_SCALE)
            .ok_or_else(|| "cimage missing TtsTalkerScale segment".to_string())?;
        let talker_b = find_seg(&header.segments, TTS_TALKER_BIAS)
            .ok_or_else(|| "cimage missing TtsTalkerBias segment".to_string())?;

        let talker_w_bytes = segment_data(deployment, talker_w)?;
        let talker_s_bytes = segment_data(deployment, talker_s)?;
        let talker_b_bytes = segment_data(deployment, talker_b)?;

        let weight_bindings = parse_talker_weights(talker_w_bytes, talker_s_bytes, talker_b_bytes)?;

        let talker = TtsMegakernel::new(device, weight_bindings)?;

        // ---- 2. Code Predictor (rejoin three segments into flat bytes) --
        let cp_w = find_seg(&header.segments, TTS_CP_WEIGHT)
            .ok_or_else(|| "cimage missing TtsCodePredictorWeight segment".to_string())?;
        let cp_s = find_seg(&header.segments, TTS_CP_SCALE)
            .ok_or_else(|| "cimage missing TtsCodePredictorScale segment".to_string())?;
        let cp_b = find_seg(&header.segments, TTS_CP_BIAS)
            .ok_or_else(|| "cimage missing TtsCodePredictorBias segment".to_string())?;

        let cp_w_bytes = segment_data(deployment, cp_w)?;
        let cp_s_bytes = segment_data(deployment, cp_s)?;
        let cp_b_bytes = segment_data(deployment, cp_b)?;

        // Concatenate weight + scale + bias for the single-buffer API.
        let mut cp_flat =
            Vec::with_capacity(cp_w_bytes.len() + cp_s_bytes.len() + cp_b_bytes.len());
        cp_flat.extend_from_slice(cp_w_bytes);
        cp_flat.extend_from_slice(cp_s_bytes);
        cp_flat.extend_from_slice(cp_b_bytes);

        let code_predictor = TtsCodePredictor::from_segments(&cp_flat)?;

        // ---- 3. Mimi Codec (conv weight + codebook) ---------------------
        let codec_conv_seg = find_seg(&header.segments, TTS_CODEC_WEIGHT)
            .ok_or_else(|| "cimage missing TtsCodecWeight segment".to_string())?;
        let codec_conv_bytes = segment_data(deployment, codec_conv_seg)?;

        let codec_cb_seg = find_seg(&header.segments, TTS_CODEC_CODEBOOK)
            .ok_or_else(|| "cimage missing TtsCodebook segment".to_string())?;
        let codec_cb_bytes = segment_data(deployment, codec_cb_seg)?;

        let codec = MimiCodec::from_segments(device, codec_conv_bytes, codec_cb_bytes)?;

        Ok(Self {
            talker,
            code_predictor,
            codec,
        })
    }

    /// Generate audio from text token IDs.
    ///
    /// `text_token_ids` — UTF-32 token IDs for TTS text encoder.
    /// `max_audio_tokens` — maximum number of autoregressive audio token steps.
    ///
    /// Returns `(PCM samples at 24 kHz, sample_rate == 24000)`.
    pub fn generate(
        &self,
        text_token_ids: &[u32],
        max_audio_tokens: usize,
    ) -> Result<(Vec<f32>, u32), String> {
        if text_token_ids.is_empty() {
            return Err("text_token_ids must not be empty".to_string());
        }
        if max_audio_tokens == 0 {
            return Err("max_audio_tokens must be > 0".to_string());
        }

        // 1. Prepend BOS, append EOS
        let mut token_sequence = Vec::with_capacity(text_token_ids.len() + 2);
        token_sequence.push(TTS_BOS);
        token_sequence.extend_from_slice(text_token_ids);
        token_sequence.push(TTS_EOS);

        // 2. Autoregressive Talker decode
        let mut audio_tokens: Vec<u32> = Vec::with_capacity(max_audio_tokens);
        // Hidden states: [num_audio_tokens x TTS_HIDDEN] f32 (pre-LM-head)
        let mut hidden_states: Vec<f32> = Vec::with_capacity(max_audio_tokens * TTS_HIDDEN);
        let mut current_token = TTS_BOS;

        for _step in 0..max_audio_tokens {
            // Prefer decode_token_with_hidden; fallback to decode_token
            // using logits as hidden-state proxy.
            // (Talker agent is adding decode_token_with_hidden.)
            #[cfg(feature = "talker_hidden_state")]
            let (logits, hs) = { self.talker.decode_token_with_hidden(current_token)? };
            #[cfg(not(feature = "talker_hidden_state"))]
            let (logits, hs) = {
                let logits = self.talker.decode_token(current_token)?;
                (logits.clone(), logits)
            };
            hidden_states.extend_from_slice(&hs);

            // Argmax on logits
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(TTS_EOS);

            audio_tokens.push(next_token);
            current_token = next_token;

            if next_token == TTS_EOS {
                break;
            }
        }

        if audio_tokens.is_empty() {
            return Err("Talker produced zero audio tokens".to_string());
        }

        let num_audio_tokens = audio_tokens.len();

        // 3. Code Predictor → [num_tokens × 16] codebook indices
        let codebook_indices = self
            .code_predictor
            .predict(&hidden_states, num_audio_tokens)?;

        // codebook_indices layout: [cb0_t0..cb0_tN, cb1_t0..cb1_tN, ...]
        // MimiCodec expects interleaved: [t0_cb0..t0_cb15, t1_cb0..t1_cb15, ...]
        let mut interleaved = Vec::with_capacity(num_audio_tokens * NUM_CODEBOOKS);
        for t in 0..num_audio_tokens {
            for cb in 0..NUM_CODEBOOKS {
                interleaved.push(codebook_indices[cb * num_audio_tokens + t]);
            }
        }

        // 4. Mimi Codec → PCM waveform
        let samples = self.codec.decode(&interleaved)?;
        if samples.is_empty() {
            return Err("Mimi Codec produced zero samples".to_string());
        }

        Ok((samples, 24000))
    }

    /// Generate audio in streaming chunks, emitting PCM segments
    /// as each chunk's Talker tokens complete.
    ///
    /// `text_token_ids` — text tokens for TTS.
    /// `max_audio_tokens` — maximum number of autoregressive audio token steps.
    /// `chunk_tokens` — number of Talker tokens per chunk (clamped 5-50).
    ///
    /// Returns one flat `Vec<f32>` PCM chunk per completed chunk.
    pub fn generate_streaming(
        &self,
        text_token_ids: &[u32],
        max_audio_tokens: usize,
        chunk_tokens: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        if text_token_ids.is_empty() {
            return Err("text_token_ids must not be empty".to_string());
        }
        if max_audio_tokens == 0 {
            return Err("max_audio_tokens must be > 0".to_string());
        }

        let chunk_size = chunk_tokens.max(5).min(50); // 5-50 tokens per chunk

        // 1. Autoregressive Talker decode — collect hidden states
        let mut all_hidden: Vec<f32> = Vec::with_capacity(max_audio_tokens * TTS_HIDDEN);
        let mut current_token = TTS_BOS;
        let mut num_audio_tokens = 0usize;

        for step in 0..max_audio_tokens {
            #[cfg(feature = "talker_hidden_state")]
            let (logits, hs) = { self.talker.decode_token_with_hidden(current_token)? };
            #[cfg(not(feature = "talker_hidden_state"))]
            let (logits, hs) = {
                let logits = self.talker.decode_token(current_token)?;
                (logits.clone(), logits)
            };
            all_hidden.extend_from_slice(&hs);

            // Argmax on logits
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(TTS_EOS);

            current_token = next_token;
            num_audio_tokens = step + 1;

            if next_token == TTS_EOS {
                break;
            }
        }

        if num_audio_tokens == 0 {
            return Err("Talker produced zero audio tokens".to_string());
        }

        // 2. Process hidden states in chunks through Code Predictor + Mimi Codec
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        let mut chunk_start_tok = 0usize;

        while chunk_start_tok < num_audio_tokens {
            let chunk_end_tok = (chunk_start_tok + chunk_size).min(num_audio_tokens);
            let chunk_token_count = chunk_end_tok - chunk_start_tok;

            // Slice hidden states for this chunk: [chunk_token_count x TTS_HIDDEN]
            let chunk_start_flat = chunk_start_tok * TTS_HIDDEN;
            let chunk_end_flat = chunk_end_tok * TTS_HIDDEN;
            let chunk_hidden = &all_hidden[chunk_start_flat..chunk_end_flat];

            // 3. Code Predictor on this chunk's hidden states
            let codebook_indices = self
                .code_predictor
                .predict(chunk_hidden, chunk_token_count)?;

            // 4. Interleave for Mimi Codec (same pattern as generate())
            let mut interleaved = Vec::with_capacity(chunk_token_count * NUM_CODEBOOKS);
            for t in 0..chunk_token_count {
                for cb in 0..NUM_CODEBOOKS {
                    interleaved.push(codebook_indices[cb * chunk_token_count + t]);
                }
            }

            // 5. Mimi Codec → PCM waveform
            let pcm = self.codec.decode(&interleaved)?;
            chunks.push(pcm);

            chunk_start_tok = chunk_end_tok;
        }

        Ok(chunks)
    }
}

// ── Segment helpers ────────────────────────────────────────────────────────

/// Extract a byte slice from a deployment's mmap_data for a SegmentEntry.
fn segment_data<'a>(
    deployment: &'a CimageDeployment,
    seg: &SegmentEntry,
) -> Result<&'a [u8], String> {
    let start = seg.offset as usize;
    let end = start + seg.length as usize;
    if end > deployment.mmap_data.len() {
        return Err(format!(
            "TTS segment at offset {} length {} extends past end of cimage ({} bytes)",
            seg.offset,
            seg.length,
            deployment.mmap_data.len()
        ));
    }
    Ok(&deployment.mmap_data[start..end])
}

/// Find a segment by kind value in the segment directory.
fn find_seg<'a>(segments: &'a [SegmentEntry], kind: u32) -> Option<&'a SegmentEntry> {
    segments.iter().find(|s| s.kind == kind && s.length > 0)
}

// ── Weight parsing ────────────────────────────────────────────────────────

/// Parse Talker weights from three parallel segments (codes, scales, biases).
///
/// Each segment contains all matrices in the same order:
///   [28 layers × 7 matrices][embed_tokens][lm_head]
/// Per-layer order: Q, K, V, O, Gate, Up, Down.
///
/// Matches are formed by taking equal-length chunks from codes, scales,
/// and biases for each matrix.
fn parse_talker_weights(
    codes: &[u8],
    scales: &[u8],
    biases: &[u8],
) -> Result<TtsWeightBindings, String> {
    if codes.len() != scales.len() || codes.len() != biases.len() {
        return Err(format!(
            "Talker segment length mismatch: codes={} scales={} biases={}",
            codes.len(),
            scales.len(),
            biases.len()
        ));
    }

    let mut offset_c = 0;

    let mut next_matrix = |rows: u32, cols: u32| -> Result<Nf4Weights, String> {
        let raw = rows as usize * cols as usize;
        let num_tiles = raw.div_ceil(nf4tile640::TILE_ELEMENTS);
        let codes_len = num_tiles * nf4tile640::PACKED_BYTES_PER_TILE;
        let meta_per_tile = nf4tile640::SCALES_F32_PER_TILE * 4; // f32 → 4 bytes
        let scales_len = num_tiles * meta_per_tile;
        let biases_len = num_tiles * meta_per_tile;

        let end_c = offset_c + codes_len;
        let end_s = offset_c + scales_len;
        let end_b = offset_c + biases_len;

        if end_c > codes.len() || end_s > scales.len() || end_b > biases.len() {
            return Err(format!(
                "Talker segment too short for matrix [{rows}x{cols}] at offset {offset_c}"
            ));
        }

        let packed_codes = codes[offset_c..end_c].to_vec();
        let s_bytes = &scales[offset_c..end_s];
        let b_bytes = &biases[offset_c..end_b];

        let s_vals: Vec<f32> = s_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_vals: Vec<f32> = b_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        offset_c += codes_len;

        Ok(Nf4Weights {
            packed_codes,
            scales: s_vals,
            biases: b_vals,
            rows,
            cols,
        })
    };

    // ---- 28 layers × 7 matrices -----------------------------------------
    let mut q_proj = Vec::with_capacity(TTS_NUM_LAYERS);
    let mut k_proj = Vec::with_capacity(TTS_NUM_LAYERS);
    let mut v_proj = Vec::with_capacity(TTS_NUM_LAYERS);
    let mut o_proj = Vec::with_capacity(TTS_NUM_LAYERS);
    let mut gate_proj = Vec::with_capacity(TTS_NUM_LAYERS);
    let mut up_proj = Vec::with_capacity(TTS_NUM_LAYERS);
    let mut down_proj = Vec::with_capacity(TTS_NUM_LAYERS);

    for _layer in 0..TTS_NUM_LAYERS {
        q_proj.push(next_matrix(TALKER_HIDDEN, TALKER_HIDDEN)?);
        k_proj.push(next_matrix(TALKER_HIDDEN, TALKER_HIDDEN)?);
        v_proj.push(next_matrix(TALKER_HIDDEN, TALKER_HIDDEN)?);
        o_proj.push(next_matrix(TALKER_HIDDEN, TALKER_HIDDEN)?);
        gate_proj.push(next_matrix(TALKER_FFN_INTERMEDIATE, TALKER_HIDDEN)?);
        up_proj.push(next_matrix(TALKER_FFN_INTERMEDIATE, TALKER_HIDDEN)?);
        down_proj.push(next_matrix(TALKER_HIDDEN, TALKER_FFN_INTERMEDIATE)?);
    }

    // ---- Embed tokens ---------------------------------------------------
    let embed_tokens = next_matrix(TALKER_VOCAB_SIZE, TALKER_HIDDEN)?;

    // ---- LM head --------------------------------------------------------
    let lm_head = next_matrix(TTS_VOCAB as u32, TALKER_HIDDEN)?;

    // ---- Norm weights (f32 flat at end of bias/scale segment) -----------
    // 28 layers × (input_norm + post_attn_norm) + final_norm = 57 × 2048 f32
    let num_norm_groups = TTS_NUM_LAYERS * 2 + 1; // 57
    let norm_size = TALKER_HIDDEN as usize; // 2048 f32
    let norms_offset = offset_c; // aligned to where codes data ends
    let norms_expected = num_norm_groups * norm_size * 4; // f32 = 4 bytes
    let norms_end = norms_offset + norms_expected;

    // Norms are appended after the last matrix's bias data in the biases
    // segment (or at end of scales segment — same offset convention).
    let norms_src = if norms_end <= biases.len() {
        &biases[norms_offset..norms_end]
    } else if norms_end <= scales.len() {
        &scales[norms_offset..norms_end]
    } else {
        return Err(format!(
            "Talker segment too short for norms: need offset+{} = {}, have {} bytes",
            norms_expected,
            norms_end,
            codes.len()
        ));
    };

    let mut norms = Vec::with_capacity(num_norm_groups);
    for i in 0..num_norm_groups {
        let start = i * norm_size * 4;
        let end = start + norm_size * 4;
        let group: Vec<f32> = norms_src[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        norms.push(group);
    }

    Ok(TtsWeightBindings {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        gate_proj,
        up_proj,
        down_proj,
        embed_tokens,
        lm_head,
        norms,
    })
}

// ── WAV encoder ──────────────────────────────────────────────────────────

/// Encode 24 kHz f32 PCM samples as 16-bit mono WAV bytes.
pub fn pcm_to_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    if samples.is_empty() {
        return Err("no samples to encode".to_string());
    }

    let num_samples = samples.len();
    let data_size = num_samples * 2; // 16-bit mono
    let file_size = 36 + data_size; // total minus 8-byte RIFF header

    let mut wav = Vec::with_capacity(44 + data_size);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(file_size as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_size as u32).to_le_bytes());

    // PCM sample data (f32 → i16)
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * i16::MAX as f32) as i16;
        wav.extend_from_slice(&i16_sample.to_le_bytes());
    }

    Ok(wav)
}

/// Encode a PCM chunk as a self-contained WAV segment (header + body).
/// Each chunk is a valid WAV for independent streaming playback.
pub fn pcm_chunk_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let data_size = num_samples * 2; // 16-bit PCM
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(file_size + 8);
    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(file_size as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
                                                 // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_size as u32).to_le_bytes());
    for &s in samples {
        let clamped = (s.max(-1.0).min(1.0) * 32767.0) as i16;
        wav.extend_from_slice(&clamped.to_le_bytes());
    }
    wav
}
