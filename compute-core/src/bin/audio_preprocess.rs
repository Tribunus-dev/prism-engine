//! Audio preprocessor for Gemma 4 Unified.
//!
//! Reads a 16 kHz mono PCM WAV file, generates synthetic audio features at
//! 20 ms frame granularity using a 640-point DFT magnitude spectrum, projects
//! each 640-d feature vector through the model's `embed_audio.embedding_projection.weight`
//! ([3840, 640] BF16), and writes rows of 3840 FP16 values to a flat .bin.
//!
//! Usage:
//!   cargo run --bin audio-preprocess --features prism-backend -- \
//!     --audio /path/to/speech.wav \
//!     --model /path/to/model.safetensors \
//!     --output /path/to/embeddings.bin

use clap::Parser;
use std::fs;
use std::path::PathBuf;

// ── Constants ────────────────────────────────────────────────────────────

/// Audio feature dimension coming out of the upstream encoder.
const AUDIO_FEATURE_DIM: usize = 640;

/// Hidden dimension after projection.
const HIDDEN_DIM: usize = 3840;

/// 20 ms frame at 16 kHz.
const FRAME_SIZE: usize = 320;

/// DFT size — zero-pad 320-sample frames to 640 for frequency resolution.
const DFT_SIZE: usize = 640;

// ── CLI ──────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "audio-preprocess",
    about = "Audio preprocessor for Gemma 4 Unified"
)]
struct Args {
    /// Path to input WAV file (16 kHz, mono, PCM 16-bit).
    #[arg(long)]
    audio: PathBuf,

    /// Path to model.safetensors containing embed_audio.embedding_projection.weight.
    #[arg(long)]
    model: PathBuf,

    /// Output .bin path for FP16 hidden vectors (num_frames x 3840).
    #[arg(long)]
    output: PathBuf,
}

// ── WAV parsing (no external crates) ─────────────────────────────────────

/// Hand-parse a simple WAV header: PCM 16-bit, mono.
///
/// Returns (sample_rate, f32-normalized samples in [-1, 1]).
fn read_wav_pcm16(path: &PathBuf) -> Result<(u32, Vec<f32>), String> {
    let data = fs::read(path).map_err(|e| format!("read WAV: {e}"))?;
    if data.len() < 44 {
        return Err(format!("WAV too short: {} bytes (need >= 44)", data.len()));
    }
    if &data[0..4] != b"RIFF" {
        return Err("Not a RIFF file".into());
    }
    if &data[8..12] != b"WAVE" {
        return Err("Not a WAVE file".into());
    }
    if &data[12..16] != b"fmt " {
        return Err("Missing fmt chunk".into());
    }

    let audio_fmt = u16::from_le_bytes([data[20], data[21]]);
    if audio_fmt != 1 {
        return Err(format!(
            "Unsupported audio format {} (only PCM = 1 supported)",
            audio_fmt
        ));
    }

    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    if num_channels != 1 {
        return Err(format!(
            "Unsupported channel count {} (only mono = 1 supported)",
            num_channels
        ));
    }

    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);

    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);
    if bits_per_sample != 16 {
        return Err(format!(
            "Unsupported bits per sample {} (only 16-bit supported)",
            bits_per_sample
        ));
    }

    // Walk sub-chunks to find "data"
    let mut offset: usize = 36; // end of fmt sub-chunk header fields
    let (data_offset, data_size) = loop {
        if offset + 8 > data.len() {
            return Err("No data chunk found".into());
        }
        let chunk_id = &data[offset..offset + 4];
        let chunk_len = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        if chunk_id == b"data" {
            break (offset + 8, chunk_len);
        }
        // Skip past this chunk (word-aligned)
        offset = offset + 8 + chunk_len;
        if chunk_len % 2 != 0 {
            offset += 1;
        }
    };

    if data_size == 0 {
        return Err("Empty data chunk".into());
    }
    if data_offset + data_size > data.len() {
        return Err("Data chunk extends beyond file".into());
    }

    let sample_count = data_size / 2;
    let mut samples = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let pos = data_offset + i * 2;
        let sample_i16 = i16::from_le_bytes([data[pos], data[pos + 1]]);
        samples.push(sample_i16 as f32 / 32768.0);
    }

    Ok((sample_rate, samples))
}

// ── Safetensors loading ─────────────────────────────────────────────────

/// Load the audio embedding projection weight from a .safetensors file.
///
/// Expected shape: [3840, 640] (output_dim × input_dim), dtype BF16 or F32.
/// Returns row-major f32 values.
fn load_projection_weight(path: &PathBuf) -> Result<Vec<f32>, String> {
    let buf = fs::read(path).map_err(|e| format!("read model: {e}"))?;
    let tensors = safetensors::SafeTensors::deserialize(&buf)
        .map_err(|e| format!("parse safetensors: {e}"))?;

    let view = tensors
        .tensor("model.embed_audio.embedding_projection.weight")
        .map_err(|e| {
            format!("tensor 'model.embed_audio.embedding_projection.weight' not found: {e}")
        })?;

    let shape: Vec<usize> = view.shape().to_vec();
    if shape.len() != 2 || shape[0] != HIDDEN_DIM || shape[1] != AUDIO_FEATURE_DIM {
        return Err(format!(
            "Expected weight shape [{}, {}], got [{}, {}]",
            HIDDEN_DIM, AUDIO_FEATURE_DIM, shape[0], shape[1]
        ));
    }

    // Convert BF16 or F32 → f32
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        safetensors::Dtype::BF16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|c| {
                let u = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((u as u32) << 16)
            })
            .collect()),
        dt => Err(format!(
            "unsupported weight dtype: {dt:?} (need BF16 or F32)"
        )),
    }
}

// ── DFT magnitude spectrum ───────────────────────────────────────────────

/// Compute 640-point DFT magnitude spectrum from up to `FRAME_SIZE` samples.
///
/// Samples beyond `len` are implicitly zero (the caller zero-padded the frame
/// buffer).  The result is the per-bin magnitude |X[k]| for k = 0 .. 639.
fn compute_dft_magnitudes(frame: &[f32; FRAME_SIZE], len: usize) -> [f32; AUDIO_FEATURE_DIM] {
    const N: f32 = DFT_SIZE as f32;
    let mut mags = [0.0f32; AUDIO_FEATURE_DIM];

    for k in 0..AUDIO_FEATURE_DIM {
        let kf = k as f32;
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        for n in 0..len {
            let angle = 2.0 * std::f32::consts::PI * kf * n as f32 / N;
            let s = frame[n];
            sum_re += s * angle.cos();
            sum_im -= s * angle.sin();
        }
        mags[k] = (sum_re * sum_re + sum_im * sum_im).sqrt();
    }

    mags
}

// ── Matrix-vector projection ─────────────────────────────────────────────

/// Project a 640-d feature vector through the [3840, 640] weight matrix
/// (row-major) into 3840-d hidden space.
fn project_feature(feat: &[f32; AUDIO_FEATURE_DIM], weight: &[f32]) -> [f32; HIDDEN_DIM] {
    let mut out = [0.0f32; HIDDEN_DIM];
    for row in 0..HIDDEN_DIM {
        let base = row * AUDIO_FEATURE_DIM;
        let mut sum = 0.0f32;
        for col in 0..AUDIO_FEATURE_DIM {
            sum += weight[base + col] * feat[col];
        }
        out[row] = sum;
    }
    out
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() -> Result<(), String> {
    let args = Args::parse();

    // ── 1. Read audio ────────────────────────────────────────────────────
    eprintln!("[audio-preprocess] Reading WAV: {}", args.audio.display());
    let (sample_rate, samples) = read_wav_pcm16(&args.audio)?;
    let duration = samples.len() as f64 / sample_rate as f64;
    eprintln!(
        "[audio-preprocess]  {} samples, {:.3}s, {} Hz, mono",
        samples.len(),
        duration,
        sample_rate
    );

    // The model expects 16 kHz.  If the file has a different rate, warn but
    // proceed — this is a development tool.
    if sample_rate != 16000 {
        eprintln!(
            "[audio-preprocess]  [WARNING] expected 16000 Hz, got {} Hz",
            sample_rate
        );
    }

    // ── 2. Load projection weight ────────────────────────────────────────
    eprintln!(
        "[audio-preprocess] Loading projection from: {}",
        args.model.display()
    );
    let weight = load_projection_weight(&args.model)?;

    // ── 3. Frame the audio, compute features, project ────────────────────
    let num_frames = samples.len().div_ceil(FRAME_SIZE);
    eprintln!(
        "[audio-preprocess]  {num_frames} frame(s) ({} samples at 20 ms)",
        FRAME_SIZE
    );

    let mut output_buf = Vec::with_capacity(num_frames * HIDDEN_DIM * 2);

    for fi in 0..num_frames {
        let start = fi * FRAME_SIZE;
        let end = std::cmp::min(start + FRAME_SIZE, samples.len());
        let frame_len = end - start;

        // Zero-padded frame buffer
        let mut frame = [0.0f32; FRAME_SIZE];
        frame[..frame_len].copy_from_slice(&samples[start..end]);

        // 640-d DFT magnitude feature
        let features = compute_dft_magnitudes(&frame, frame_len);

        // 3840-d projection
        let projected = project_feature(&features, &weight);

        // Serialise as FP16 → little-endian bytes
        for &v in &projected {
            output_buf.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
    }

    // ── 4. Write output ──────────────────────────────────────────────────
    let total_fp16 = output_buf.len() / 2;
    eprintln!(
        "[audio-preprocess] Writing {} FP16 values ({} frames x {}) to: {}",
        total_fp16,
        num_frames,
        HIDDEN_DIM,
        args.output.display()
    );
    fs::write(&args.output, &output_buf).map_err(|e| format!("write output: {e}"))?;

    eprintln!("[audio-preprocess] Done");
    Ok(())
}
