//! Audio preprocessing for Gemma 4 Unified — Accelerate vDSP accelerated.
//!
//! Zero MLX dependency. Uses Apple Accelerate framework's vDSP for FFT
//! and real-time signal processing primitives.
//!
//! # Pipeline
//! 1. `load_wav_to_f32` — parse raw WAV bytes → f32 mono + metadata
//! 2. `resample_linear` — resample to 16 kHz
//! 3. `compute_mel_spectrogram_vdsp` — FFT via vDSP → mel filterbank → log
//!
//! # Gemma 4 Audio Configuration
//! - sample_rate: 16000
//! - n_fft: 400 (25 ms at 16 kHz, zero-padded to FFT_SIZE=512)
//! - hop_length: 160 (10 ms at 16 kHz)
//! - num_mel_bins: 640
//! - max_audio_length_s: 30

use crate::backend::accelerate_ffi::{self, DSPSplitComplex, FFT_FORWARD};

// ── Gemma 4 constants ───────────────────────────────────────────────────────

/// Target sample rate (16 kHz).
const TARGET_SAMPLE_RATE: u32 = 16000;

/// FFT window size (25 ms at 16 kHz).
const N_FFT: usize = 400;

/// STFT hop length (10 ms at 16 kHz).
const HOP_LENGTH: usize = 160;

/// Number of mel filterbank bins.
const NUM_MEL_BINS: usize = 640;

/// Maximum audio duration in seconds.
const MAX_AUDIO_LENGTH_S: u32 = 30;

/// FFT size — next power of 2 >= N_FFT.
const FFT_SIZE: usize = 512;

/// log2(FFT_SIZE) = 9.
const LOG2N: u32 = 9;

/// Number of unique frequency bins after real FFT: FFT_SIZE/2 + 1 = 257.
const NUM_FREQ_BINS: usize = FFT_SIZE / 2 + 1;

// ── Public API ──────────────────────────────────────────────────────────────

/// Parse raw WAV bytes into f32 mono samples, returning the WAV's metadata.
///
/// Returns `(samples, sample_rate, num_channels)`.
/// Does **not** resample — caller (`preprocess_audio_gemma4`) handles that.
///
/// Supports PCM 8/16/24/32-bit integer, IEEE float 32-bit, mono/stereo.
/// Stereo is averaged to mono.
pub fn load_wav_to_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32, u16), String> {
    if bytes.len() < 44 {
        return Err("WAV file too short (no header)".into());
    }

    // RIFF header
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    let mut channels: u16 = 1;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 16;
    let mut data_start: usize = 44;
    let mut data_size: usize = 0;
    let mut fmt_found = false;

    let mut offset: usize = 12;
    while offset + 8 <= bytes.len() {
        let chunk_id: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
        let chunk_size =
            u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        match &chunk_id {
            b"fmt " => {
                if offset + 16 > bytes.len() {
                    return Err("WAV fmt chunk truncated".into());
                }
                let audio_format =
                    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                channels =
                    u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap());
                sample_rate =
                    u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
                bits_per_sample =
                    u16::from_le_bytes(bytes[offset + 14..offset + 16].try_into().unwrap());
                if audio_format != 1 && audio_format != 3 {
                    return Err(format!(
                        "Unsupported WAV format code: {} (only PCM/float)",
                        audio_format
                    ));
                }
                fmt_found = true;
            }
            b"data" => {
                data_start = offset;
                data_size = chunk_size.min(bytes.len() - offset);
                break;
            }
            _ => {}
        }
        offset += chunk_size;
        if chunk_size % 2 != 0 {
            offset += 1;
        }
    }

    if !fmt_found {
        return Err("WAV fmt chunk not found".into());
    }
    if data_size == 0 {
        return Err("WAV data chunk not found or empty".into());
    }

    let byte_depth = (bits_per_sample / 8) as usize;
    let frame_size = byte_depth * channels as usize;
    let num_frames = data_size / frame_size;
    let data = &bytes[data_start..data_start + num_frames * frame_size];

    let mut samples = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let frame_offset = frame_idx * frame_size;
        let mut frame_sum: f32 = 0.0;

        for ch in 0..channels as usize {
            let ch_offset = frame_offset + ch * byte_depth;
            let sample_f32 = match bits_per_sample {
                8 => (data[ch_offset] as f32 - 128.0) / 128.0,
                16 => {
                    let val =
                        i16::from_le_bytes(data[ch_offset..ch_offset + 2].try_into().unwrap());
                    val as f32 / i16::MAX as f32
                }
                24 => {
                    let mut buf = [0u8; 4];
                    buf[..3].copy_from_slice(&data[ch_offset..ch_offset + 3]);
                    if buf[2] & 0x80 != 0 {
                        buf[3] = 0xFF;
                    }
                    let val = i32::from_le_bytes(buf);
                    val as f32 / 8_388_607.0f32
                }
                32 => {
                    let val =
                        i32::from_le_bytes(data[ch_offset..ch_offset + 4].try_into().unwrap());
                    val as f32 / i32::MAX as f32
                }
                _ => 0.0,
            };
            frame_sum += sample_f32;
        }
        samples.push(frame_sum / channels as f32);
    }

    Ok((samples, sample_rate, channels))
}

/// Full Gemma 4 preprocessing: resample + mel spectrogram.
///
/// Returns flattened `[NUM_MEL_BINS, num_frames]` — caller adds batch dim.
pub fn preprocess_audio_gemma4(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, String> {
    // 1. Resample to 16 kHz via linear interpolation
    let samples_16k = resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE);

    // 2. Compute mel spectrogram via vDSP
    let (mel_spec, _num_frames) =
        compute_mel_spectrogram_vdsp(&samples_16k, N_FFT, HOP_LENGTH, NUM_MEL_BINS, TARGET_SAMPLE_RATE)?;

    Ok(mel_spec)
}

// ── Resampling ──────────────────────────────────────────────────────────────

/// Linear interpolation resampler from `src_rate` to `dst_rate`.
fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || input.is_empty() {
        return input.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let output_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    let last = *input.last().unwrap_or(&0.0);
    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        if src_idx + 1 < input.len() {
            let sample = input[src_idx] as f64 * (1.0 - frac) + input[src_idx + 1] as f64 * frac;
            output.push(sample as f32);
        } else {
            output.push(last);
        }
    }

    output
}

// ── Mel spectrogram via vDSP ────────────────────────────────────────────────

/// Compute mel spectrogram using Accelerate vDSP for FFT.
///
/// Parameters mirror Gemma 4 config:
/// - `samples` — PCM f32 mono audio at `sample_rate` Hz.
/// - `n_fft` — window size (400).
/// - `hop_length` — stride between frames (160).
/// - `num_mel_bins` — output mel bins (640).
/// - `sample_rate` — sample rate in Hz (16000).
///
/// Returns `(Vec<f32>, usize)` — flattened `[num_mel_bins, num_frames]` and
/// the frame count.
fn compute_mel_spectrogram_vdsp(
    samples: &[f32],
    n_fft: usize,
    hop_length: usize,
    num_mel_bins: usize,
    sample_rate: u32,
) -> Result<(Vec<f32>, usize), String> {
    if samples.is_empty() {
        return Err("Empty audio samples".into());
    }

    // Truncate to max duration
    let max_samples = (MAX_AUDIO_LENGTH_S as usize) * (sample_rate as usize);
    let samples = if samples.len() > max_samples {
        &samples[..max_samples]
    } else {
        samples
    };

    let num_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        return Err(format!(
            "Audio too short: {} samples, need at least {}",
            samples.len(),
            n_fft
        ));
    };

    // Step 1 — Hann window via vDSP
    let mut hann_window = vec![0.0f32; n_fft];
    unsafe {
        accelerate_ffi::vDSP_hann_window(hann_window.as_mut_ptr(), n_fft as u32, 0);
    }

    // Step 2 — Build mel filterbank
    let mel_fb = build_mel_filterbank(FFT_SIZE, num_mel_bins, sample_rate);

    // Step 3 — Create FFT setup (reused across all frames)
    let fft_setup = unsafe { accelerate_ffi::vDSP_create_fftsetup(LOG2N) };
    if fft_setup.is_null() {
        return Err("vDSP_create_fftsetup failed — invalid FFT size".into());
    }

    // Per-frame scratch buffers
    let mut frame_buf = vec![0.0f32; FFT_SIZE];
    let mut realp = vec![0.0f32; FFT_SIZE / 2];
    let mut imagp = vec![0.0f32; FFT_SIZE / 2];
    let mut mag2 = vec![0.0f32; NUM_FREQ_BINS];

    let mut mel_spec = vec![0.0f32; num_mel_bins * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        // a. Copy n_fft samples to frame buffer, zero-fill the rest
        frame_buf[..n_fft].copy_from_slice(&samples[start..start + n_fft]);

        // b. Apply Hann window in-place (first n_fft elements)
        unsafe {
            accelerate_ffi::vDSP_vmul(
                frame_buf.as_ptr(),
                1,
                hann_window.as_ptr(),
                1,
                frame_buf.as_mut_ptr(),
                1,
                n_fft as i32,
            );
        }

        // c. Pack into split complex — realp[k] = buf[2k], imagp[k] = buf[2k+1]
        for k in 0..FFT_SIZE / 2 {
            realp[k] = frame_buf[2 * k];
            imagp[k] = frame_buf[2 * k + 1];
        }

        // d. Real FFT via vDSP_fft_zrip
        let mut split = DSPSplitComplex {
            realp: realp.as_mut_ptr(),
            imagp: imagp.as_mut_ptr(),
        };
        unsafe {
            accelerate_ffi::vDSP_fft_zrip(fft_setup, &mut split, 1, LOG2N, FFT_FORWARD);
        }

        // e. Magnitude² from zrip output
        //    zrip forward packs:
        //      realp[0] = X[0] (DC, purely real)
        //      imagp[0] = X[N/2] (Nyquist, purely real)
        //      realp[k] = Re(X[k]), imagp[k] = Im(X[k]) for k=1..N/2-1
        {
            let half_n = FFT_SIZE / 2;
            mag2[0] = realp[0] * realp[0];                         // DC
            for k in 1..half_n {
                mag2[k] = realp[k] * realp[k] + imagp[k] * imagp[k];
            }
            mag2[half_n] = imagp[0] * imagp[0];                    // Nyquist
        }

        // f. Mel filterbank + log10
        for mel_idx in 0..num_mel_bins {
            let mut energy = 0.0f64;
            let row_off = mel_idx * NUM_FREQ_BINS;
            for fft_idx in 0..NUM_FREQ_BINS {
                energy += mag2[fft_idx] as f64 * mel_fb[row_off + fft_idx];
            }
            // log10, clipped at 1e-10
            mel_spec[mel_idx * num_frames + frame_idx] = (energy as f32).max(1e-10).log10();
        }
    }

    // Cleanup
    unsafe {
        accelerate_ffi::vDSP_destroy_fftsetup(fft_setup);
    }

    Ok((mel_spec, num_frames))
}

// ── Mel filterbank ──────────────────────────────────────────────────────────

/// Build triangular mel filterbank matrix.
///
/// Returns a flattened `[num_mel_bins][fft_size/2 + 1]` f64 matrix (row-major).
fn build_mel_filterbank(fft_size: usize, num_mel_bins: usize, sample_rate: u32) -> Vec<f64> {
    let fft_bins = fft_size / 2 + 1;
    let mut filterbank = vec![0.0f64; num_mel_bins * fft_bins];

    let low_freq_mel = 0.0;
    let high_freq_mel = hz_to_mel(sample_rate as f64 / 2.0);
    let mel_step = (high_freq_mel - low_freq_mel) / (num_mel_bins + 1) as f64;

    let mel_points: Vec<f64> = (0..num_mel_bins + 2)
        .map(|i| mel_to_hz(low_freq_mel + i as f64 * mel_step))
        .collect();

    for mel_idx in 0..num_mel_bins {
        let left = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let right = mel_points[mel_idx + 2];
        let row_off = mel_idx * fft_bins;

        for fft_idx in 0..fft_bins {
            let freq = fft_idx as f64 * sample_rate as f64 / fft_size as f64;
            if freq >= left && freq <= center {
                filterbank[row_off + fft_idx] = (freq - left) / (center - left);
            } else if freq >= center && freq <= right {
                filterbank[row_off + fft_idx] = (right - freq) / (right - center);
            }
        }
    }

    filterbank
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_mel_roundtrip() {
        for f in [0.0, 100.0, 1000.0, 8000.0] {
            let m = hz_to_mel(f);
            let f_back = mel_to_hz(m);
            assert!((f - f_back).abs() < 1.0, "Hz-mel error at {} Hz: {}", f, f_back);
        }
    }

    #[test]
    fn test_resample_identity() {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let out = resample_linear(&input, 16000, 16000);
        assert_eq!(out.len(), input.len());
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resample_downsample() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = resample_linear(&input, 16000, 8000);
        assert!(out.len() < input.len());
        assert!(out.len() > 40);
    }

    #[test]
    fn test_load_wav_too_short() {
        assert!(load_wav_to_f32(&[]).is_err());
        assert!(load_wav_to_f32(&[0u8; 10]).is_err());
        assert!(load_wav_to_f32(&[0u8; 44]).is_err());
    }

    #[test]
    fn test_mel_filterbank_dimensions() {
        let fb = build_mel_filterbank(512, 640, 16000);
        assert_eq!(fb.len(), 640 * 257);
    }

    #[test]
    fn test_vdsp_short_audio_error() {
        assert!(compute_mel_spectrogram_vdsp(&[], 400, 160, 640, 16000).is_err());
    }

    #[test]
    fn test_preprocess_short_audio_error() {
        assert!(preprocess_audio_gemma4(&[0.0f32; 100], 16000).is_err());
    }

    #[test]
    fn test_preprocess_basic() {
        let num = N_FFT + HOP_LENGTH; // 2 frames minimum
        let s: Vec<f32> = (0..num).map(|i| (i as f32 * 0.1).sin()).collect();
        let mel = preprocess_audio_gemma4(&s, 16000).unwrap();
        assert_eq!(mel.len(), NUM_MEL_BINS * 2);
    }

    #[test]
    fn test_preprocess_with_resample() {
        let s: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.05).sin()).collect();
        let mel = preprocess_audio_gemma4(&s, 44100).unwrap();
        assert!(!mel.is_empty());
    }
}
