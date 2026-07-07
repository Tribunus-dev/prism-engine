//! Sliding-window temporal KV cache for acoustic streaming (Qwen3-TTS).
//!
//! Unlike [`LogQuant`](crate::quantization::turboquant_kv) which uses
//! power-of-two spacing for text, acoustic models need continuous local
//! context to preserve prosody and pitch. This cache keeps the most recent
//! K frames at full FP16 precision and compresses older frames uniformly
//! with asymmetric quantization.

use half::f16;

// ── Int2PackedGroup ─────────────────────────────────────────────────────────

/// A packed group of 64 values quantized to 2 bits each, with per-group
/// asymmetric scaling.
///
/// # Layout
///
/// | Field | Size |
/// |---|---|
/// | `packed_elements` | 16 bytes (64 × 2 bits) |
/// | `scale` | 2 bytes (f16) |
/// | `min_value` | 2 bytes (f16) |
/// | **Total** | **20 bytes** |
#[derive(Debug, Clone)]
pub struct Int2PackedGroup {
    /// 64 tokens worth of 2-bit values packed into 16 bytes.
    ///
    /// Byte `b` stores elements `[b*4 .. b*4+4)`:
    /// - bits `[1:0]` = element `b*4`
    /// - bits `[3:2]` = element `b*4 + 1`
    /// - bits `[5:4]` = element `b*4 + 2`
    /// - bits `[7:6]` = element `b*4 + 3`
    pub packed_elements: [u8; 16],
    /// FP16 scale factor: `(max - min) / 3.0`.
    pub scale: f16,
    /// FP16 minimum tracking point.
    pub min_value: f16,
}

impl Int2PackedGroup {
    /// Pack 64 f32 values into a 2-bit quantized group.
    ///
    /// Computes per-group scale and min, then uniformly quantizes each
    /// element to `{0, 1, 2, 3}`.
    pub fn pack(values: &[f32; 64]) -> Self {
        let mut min = values[0];
        let mut max = values[0];
        for &v in &values[1..] {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        let range = max - min;
        let scale = if range == 0.0 {
            f16::from_f32(1.0)
        } else {
            f16::from_f32(range / 3.0)
        };
        let scale_f32 = scale.to_f32();

        let mut packed = [0u8; 16];
        for (chunk_idx, byte) in packed.iter_mut().enumerate() {
            let base = chunk_idx * 4;
            let mut b = 0u8;
            for j in 0..4 {
                let v = values[base + j];
                let q = if scale_f32 == 0.0 {
                    0u8
                } else {
                    ((v - min) / scale_f32).round().clamp(0.0, 3.0) as u8
                };
                b |= q << (j * 2);
            }
            *byte = b;
        }

        Self {
            packed_elements: packed,
            scale,
            min_value: f16::from_f32(min),
        }
    }

    /// Unpack back to 64 f32 values using the stored scale and min.
    pub fn unpack(&self) -> [f32; 64] {
        let scale_f32 = self.scale.to_f32();
        let min_f32 = self.min_value.to_f32();
        let mut out = [0.0f32; 64];

        for (chunk_idx, &byte) in self.packed_elements.iter().enumerate() {
            let base = chunk_idx * 4;
            for j in 0..4 {
                let q = (byte >> (j * 2)) & 0x03;
                out[base + j] = min_f32 + (q as f32) * scale_f32;
            }
        }

        out
    }

    /// Total packed size in bytes.
    pub const fn packed_size() -> usize {
        20 // 16 + 2 + 2
    }
}

// ── SlidingWindowCache ──────────────────────────────────────────────────────

/// Sliding-window temporal cache for acoustic streaming.
///
/// Unlike [`LogQuant`](crate::quantization::turboquant_kv) which uses
/// power-of-two spacing for text, acoustic models need continuous local
/// context to preserve prosody and pitch. This cache keeps the most recent
/// K frames at full FP16 precision and compresses older frames uniformly
/// with asymmetric quantization.
#[derive(Debug, Clone)]
pub struct SlidingWindowCache {
    /// Maximum number of frames the cache can hold.
    pub capacity: usize,
    /// Number of most-recent frames kept at full precision.
    pub window_size: usize,
    /// Number of quantization bits for compressed frames (currently only
    /// `2` is supported).
    pub quantization_bits: u8,
    /// Current write position (append-only).
    pub position: usize,

    // ── internal storage ──
    /// FP16 keys for the most recent `window_size` frames.
    /// Index `0` is the oldest in-window frame.
    fp16_keys: Vec<Vec<f16>>,
    /// FP16 values for the most recent `window_size` frames.
    fp16_values: Vec<Vec<f16>>,
    /// Quantized keys for all frames older than the window.
    /// Index `i` corresponds to original frame index `i`.
    quantized_keys: Vec<Int2PackedGroup>,
    /// Quantized values for all frames older than the window.
    quantized_values: Vec<Int2PackedGroup>,
}

impl SlidingWindowCache {
    /// Create a new cache with the given capacity, window size, and
    /// quantization bits.
    ///
    /// # Arguments
    ///
    /// * `capacity` — Maximum number of frames the cache can hold before
    ///   stores begin returning errors.
    /// * `window_size` — Number of most-recent frames kept at full FP16
    ///   precision. Must be ≤ `capacity`.
    /// * `quant_bits` — Must be `2` (the only currently supported value).
    ///
    /// # Panics
    ///
    /// Panics if `quant_bits != 2` or `window_size > capacity`.
    pub fn new(capacity: usize, window_size: usize, quant_bits: u8) -> Self {
        assert!(quant_bits == 2, "only 2-bit quantization is supported");
        assert!(
            window_size <= capacity,
            "window_size ({}) must not exceed capacity ({})",
            window_size,
            capacity
        );
        Self {
            capacity,
            window_size,
            quantization_bits: quant_bits,
            position: 0,
            fp16_keys: Vec::with_capacity(window_size),
            fp16_values: Vec::with_capacity(window_size),
            quantized_keys: Vec::new(),
            quantized_values: Vec::new(),
        }
    }

    /// Append a key-value pair at the current write position.
    ///
    /// Each slice must be exactly 64 elements long (matching
    /// [`Int2PackedGroup`] packing granularity).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the cache is full (`position >= capacity`) or the
    /// slices are not exactly 64 elements.
    pub fn store_key_value(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        if self.position >= self.capacity {
            return Err("SlidingWindowCache is full".into());
        }
        if key.len() != 64 || value.len() != 64 {
            return Err(format!(
                "key/value must be exactly 64 elements; got key={}, val={}",
                key.len(),
                value.len(),
            ));
        }

        // Store as FP16
        let key_f16: Vec<f16> = key.iter().map(|&x| f16::from_f32(x)).collect();
        let val_f16: Vec<f16> = value.iter().map(|&x| f16::from_f32(x)).collect();
        self.fp16_keys.push(key_f16);
        self.fp16_values.push(val_f16);

        // If the FP16 buffer exceeds window_size, move the oldest frame
        // to quantized storage.
        if self.fp16_keys.len() > self.window_size {
            // Pop the oldest FP16 frame
            let oldest_key_f16 = self.fp16_keys.remove(0);
            let oldest_val_f16 = self.fp16_values.remove(0);

            // Convert back to f32 for packing
            let mut key_arr = [0.0f32; 64];
            for (i, &v) in oldest_key_f16.iter().enumerate() {
                key_arr[i] = v.to_f32();
            }
            let mut val_arr = [0.0f32; 64];
            for (i, &v) in oldest_val_f16.iter().enumerate() {
                val_arr[i] = v.to_f32();
            }

            self.quantized_keys.push(Int2PackedGroup::pack(&key_arr));
            self.quantized_values.push(Int2PackedGroup::pack(&val_arr));
        }

        self.position += 1;
        Ok(())
    }

    /// Read a key at the given frame index.
    ///
    /// Frames within `window_size` of the last written position are
    /// returned at full FP16 precision (converted to `f32`). Older frames
    /// are decompressed from quantized storage.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `frame_index >= position`.
    pub fn read_key(&self, frame_index: usize) -> Result<Vec<f32>, String> {
        self.read_frame(frame_index, /* is_key */ true)
    }

    /// Read a value at the given frame index.
    ///
    /// Behaves identically to [`read_key`](Self::read_key) but reads from
    /// the value buffer.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `frame_index >= position`.
    pub fn read_value(&self, frame_index: usize) -> Result<Vec<f32>, String> {
        self.read_frame(frame_index, /* is_key */ false)
    }

    /// Reset the cache, discarding all stored frames.
    pub fn clear(&mut self) {
        self.position = 0;
        self.fp16_keys.clear();
        self.fp16_values.clear();
        self.quantized_keys.clear();
        self.quantized_values.clear();
    }

    // ── internal helpers ──

    fn num_quantized(&self) -> usize {
        self.quantized_keys.len()
    }

    fn read_frame(&self, frame_index: usize, is_key: bool) -> Result<Vec<f32>, String> {
        if frame_index >= self.position {
            return Err(format!(
                "frame_index {} out of range (position = {})",
                frame_index, self.position,
            ));
        }

        let nq = self.num_quantized();
        if frame_index < nq {
            // Decompress from quantized storage
            let group = if is_key {
                &self.quantized_keys[frame_index]
            } else {
                &self.quantized_values[frame_index]
            };
            let arr = group.unpack();
            Ok(arr.to_vec())
        } else {
            // Read from FP16 storage
            let fp16_idx = frame_index - nq;
            let slice = if is_key {
                &self.fp16_keys[fp16_idx]
            } else {
                &self.fp16_values[fp16_idx]
            };
            Ok(slice.iter().map(|&x| x.to_f32()).collect())
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int2_pack_unpack_roundtrip() {
        // Constant input — no loss.
        let values = [42.0f32; 64];
        let group = Int2PackedGroup::pack(&values);
        let unpacked = group.unpack();
        for (i, &v) in unpacked.iter().enumerate() {
            assert!(
                (v - 42.0).abs() < 0.1,
                "element {i}: expected ~42.0, got {v}",
            );
        }

        // Linear ramp — slight quantization error.
        let mut ramp = [0.0f32; 64];
        for (i, v) in ramp.iter_mut().enumerate() {
            *v = i as f32;
        }
        let group = Int2PackedGroup::pack(&ramp);
        let unpacked = group.unpack();
        for (i, &v) in unpacked.iter().enumerate() {
            let expected = i as f32;
            let err = (v - expected).abs();
            // With 2-bit quantization over a range of 63, each step is
            // ~21, so error can be up to ~10.5.
            assert!(
                err < 12.0,
                "element {i}: expected {expected}, got {v} (err={err})",
            );
        }
    }

    #[test]
    fn test_int2_packed_size() {
        assert_eq!(Int2PackedGroup::packed_size(), 20);
    }

    #[test]
    fn test_sliding_window_roundtrip() {
        let mut cache = SlidingWindowCache::new(100, 10, 2);
        for i in 0..20 {
            let key = vec![i as f32; 64];
            let val = vec![i as f32 * 2.0; 64];
            cache.store_key_value(&key, &val).unwrap();
        }

        // Most recent 10 should be full precision.
        let recent = cache.read_key(19).unwrap();
        assert!((recent[0] - 19.0).abs() < 0.01);

        // Older should still round-trip (with quantization noise).
        let older = cache.read_key(5).unwrap();
        assert!(
            (older[0] - 5.0).abs() < 2.0,
            "quantized read should approx match"
        );
    }

    #[test]
    fn test_out_of_range() {
        let mut cache = SlidingWindowCache::new(10, 3, 2);
        let key = vec![1.0f32; 64];
        let val = vec![2.0f32; 64];
        cache.store_key_value(&key, &val).unwrap();
        assert!(cache.read_key(1).is_err());
        assert!(cache.read_value(1).is_err());
    }

    #[test]
    fn test_capacity_full() {
        let mut cache = SlidingWindowCache::new(3, 1, 2);
        for i in 0..3 {
            let key = vec![i as f32; 64];
            let val = vec![i as f32; 64];
            cache.store_key_value(&key, &val).unwrap();
        }
        // Fourth store should fail.
        let err = cache
            .store_key_value(&[0.0f32; 64], &[0.0f32; 64])
            .unwrap_err();
        assert!(err.contains("full"), "expected full error, got: {err}",);
    }

    #[test]
    fn test_clear_resets() {
        let mut cache = SlidingWindowCache::new(10, 3, 2);
        for i in 0..5 {
            let key = vec![i as f32; 64];
            let val = vec![i as f32 * 10.0; 64];
            cache.store_key_value(&key, &val).unwrap();
        }
        assert_eq!(cache.position, 5);
        cache.clear();
        assert_eq!(cache.position, 0);
        assert!(cache.read_key(0).is_err());
    }

    #[test]
    fn test_value_separate_from_key() {
        let mut cache = SlidingWindowCache::new(20, 5, 2);
        for i in 0..12 {
            let key = vec![i as f32; 64];
            let val = vec![i as f32 * 100.0; 64];
            cache.store_key_value(&key, &val).unwrap();
        }
        // Recent value
        let val = cache.read_value(11).unwrap();
        assert!((val[0] - 1100.0).abs() < 0.1);
        // Quantized value
        let val = cache.read_value(3).unwrap();
        assert!(
            (val[0] - 300.0).abs() < 10.0,
            "quantized value approx match: {}",
            val[0]
        );
    }
}
