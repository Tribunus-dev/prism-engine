//! Mixed tile rescue family candidate generation for QuantSweep.
//!
//! Packs weight tiles using the base 4-bit codec, measures per-tile
//! reconstruction error, and rescues the worst tiles by storing their
//! raw f32 values in the extra payload. The rescue fraction controls
//! the number of tiles replaced. Rescue routing is tracked via a
//! serialized routing table in the extra bytes rather than sentinel
//! all-zero code blocks.

use serde_json::json;

use crate::contract::NF4_TILE640_CODE_BYTES;
use crate::nf4tile640::nf4_dequantize;
use crate::nf4tile640::NF4_CODEBOOK;
use crate::nf4tile640::{pack_nf4_tile_with_group_size, TILE_ELEMENTS};
use crate::sweep::families::FamilyCandidate;
use crate::sweep::spec::{MixedTileSweepGrid, OverlayMode, RescueGranularity, RescueSchedule};

#[cfg(test)]
mod tests {
    use super::*;
    // ── Test 1: MixedTile routing table round-trip ──────────────────────

    #[test]
    fn test_mixed_tile_routing_table_roundtrip() {
        let entries = vec![
            MixedTileRoutingEntry {
                unit_id: 0,
                granularity: RescueGranularity::Tile640,
                overlay_mode: OverlayMode::FullReplacement,
                rescue_format: 0, // f32
                payload_offset: 0,
                payload_len: 2560, // 640 * 4
            },
            MixedTileRoutingEntry {
                unit_id: 1,
                granularity: RescueGranularity::Group,
                overlay_mode: OverlayMode::DeltaCorrection,
                rescue_format: 1, // fp16
                payload_offset: 2560,
                payload_len: 1280, // 640 * 2
            },
        ];

        let serialized = serialize_routing_table(&entries);
        let (deserialized, bytes_consumed) = deserialize_routing_table(&serialized);

        assert_eq!(deserialized.len(), entries.len());
        assert_eq!(bytes_consumed, 4 + entries.len() * 24);

        for (orig, deser) in entries.iter().zip(deserialized.iter()) {
            assert_eq!(orig.unit_id, deser.unit_id);
            assert_eq!(orig.granularity, deser.granularity);
            assert_eq!(orig.overlay_mode, deser.overlay_mode);
            assert_eq!(orig.rescue_format, deser.rescue_format);
            assert_eq!(orig.payload_offset, deser.payload_offset);
            assert_eq!(orig.payload_len, deser.payload_len);
        }
    }

    // ── Test 2: no false rescue for all-zero NF4 tiles ──────────────────

    #[test]
    fn test_mixed_tile_no_false_rescue_for_all_zero_nf4_tiles() {
        const ROWS: usize = 1;
        const COLS: usize = TILE_ELEMENTS; // exactly one tile (640)
        let weights = vec![0.0f32; ROWS * COLS];
        let group_size: usize = 128;
        let rescue_fraction = 0.1;
        let (codes, scales, biases, extra_bytes) =
            pack_mixed_tile_matrix(&weights, ROWS, COLS, group_size, rescue_fraction);

        let reconstructed = unpack_mixed_tile(
            &codes,
            &scales,
            &biases,
            &extra_bytes,
            ROWS,
            COLS,
            group_size,
            rescue_fraction,
        );

        // All-zero input MUST reconstruct to all-zero output.
        // This proves the sentinel false-positive bug is fixed: the routing
        // table is the sole authority for identifying rescued tiles. An
        // all-zero weight tile will never be mistaken for a sentinel.
        for (i, &val) in reconstructed.iter().enumerate() {
            assert!(val.abs() < 1e-6, "expected zero at index {i}, got {val}");
        }
    }

    // ── Test 3: rescue preserves rescued tile values ────────────────────

    #[test]
    fn test_mixed_tile_rescue_preserves_rescued_tile_values() {
        const ROWS: usize = 1;
        const COLS: usize = TILE_ELEMENTS * 2; // two tiles
        let group_size: usize = 128;
        let rescue_fraction = 0.5; // ceil(2 * 0.5) = 1 tile rescued

        let mut weights = vec![0.0f32; ROWS * COLS];

        // Tile 0: all zeros → MSE = 0.
        // Tile 1: pattern with intra-group variance → MSE > 0.
        // With alternating values inside each group, the max-abs quantizer
        // produces non-zero error, so tile 1 is selected for rescue.
        for i in 0..TILE_ELEMENTS {
            weights[TILE_ELEMENTS + i] = match i % 3 {
                0 => 0.3,
                1 => -0.2,
                _ => 0.7,
            };
        }

        let (codes, scales, biases, extra_bytes) =
            pack_mixed_tile_matrix(&weights, ROWS, COLS, group_size, rescue_fraction);

        let reconstructed = unpack_mixed_tile(
            &codes,
            &scales,
            &biases,
            &extra_bytes,
            ROWS,
            COLS,
            group_size,
            rescue_fraction,
        );

        // Tile 0 was NOT rescued — NF4 reconstruction should be near zero.
        for i in 0..TILE_ELEMENTS {
            assert!(
                reconstructed[i].abs() < 1e-4,
                "tile 0 element {i} should be ~0, got {}",
                reconstructed[i]
            );
        }

        // Tile 1 WAS rescued — raw f32 payload gives exact values.
        for i in 0..TILE_ELEMENTS {
            let expected = match i % 3 {
                0 => 0.3f32,
                1 => -0.2f32,
                _ => 0.7f32,
            };
            assert!(
                (reconstructed[TILE_ELEMENTS + i] - expected).abs() < 1e-6,
                "tile 1 element {i}: expected {expected}, got {}",
                reconstructed[TILE_ELEMENTS + i]
            );
        }
    }
}

// ── Routing table types ────────────────────────────────────────────────

/// Explicit rescue routing entry for mixed-tile representation.
#[derive(Debug, Clone, Copy)]
pub struct MixedTileRoutingEntry {
    pub unit_id: u64,
    pub granularity: RescueGranularity,
    pub overlay_mode: OverlayMode,
    pub rescue_format: u16, // 0 = f32, 1 = fp16
    pub payload_offset: u64,
    pub payload_len: u32,
}

/// Serialize a slice of routing entries to LE bytes.
///
/// Format: [entry_count: u32 LE] [entries...]
/// Each entry (24 bytes):
///   [unit_id: u64 LE] [granularity: u8] [overlay_mode: u8]
///   [rescue_format: u16 LE] [payload_offset: u64 LE] [payload_len: u32 LE]
fn serialize_routing_table(entries: &[MixedTileRoutingEntry]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + entries.len() * 24);
    bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        bytes.extend_from_slice(&e.unit_id.to_le_bytes());
        bytes.push(e.granularity as u8);
        bytes.push(e.overlay_mode as u8);
        bytes.extend_from_slice(&e.rescue_format.to_le_bytes());
        bytes.extend_from_slice(&e.payload_offset.to_le_bytes());
        bytes.extend_from_slice(&e.payload_len.to_le_bytes());
    }
    bytes
}

/// Deserialize routing entries from raw bytes.
///
/// Returns (entries, bytes_consumed). Handles truncation gracefully
/// by parsing only the complete entries available in the input.
fn deserialize_routing_table(bytes: &[u8]) -> (Vec<MixedTileRoutingEntry>, usize) {
    if bytes.len() < 4 {
        return (Vec::new(), bytes.len());
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    // Sanity-cap allocation to avoid pathological inputs.
    let count = count.min(8192);

    const ENTRY_BYTES: usize = 24;
    const HEADER_SIZE: usize = 4;
    let max_entries = (bytes.len().saturating_sub(HEADER_SIZE)) / ENTRY_BYTES;
    let actual = count.min(max_entries);

    let mut entries = Vec::with_capacity(actual);
    let mut offset = HEADER_SIZE;
    for _ in 0..actual {
        if offset + ENTRY_BYTES > bytes.len() {
            break;
        }
        let unit_id = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        let granularity = match bytes[offset + 8] {
            0 => RescueGranularity::Group,
            1 => RescueGranularity::Tile640,
            _ => RescueGranularity::OutputChannel,
        };
        let overlay_mode = match bytes[offset + 9] {
            0 => OverlayMode::FullReplacement,
            _ => OverlayMode::DeltaCorrection,
        };
        let rescue_format = u16::from_le_bytes(bytes[offset + 10..offset + 12].try_into().unwrap());
        let payload_offset =
            u64::from_le_bytes(bytes[offset + 12..offset + 20].try_into().unwrap());
        let payload_len = u32::from_le_bytes(bytes[offset + 20..offset + 24].try_into().unwrap());

        entries.push(MixedTileRoutingEntry {
            unit_id,
            granularity,
            overlay_mode,
            rescue_format,
            payload_offset,
            payload_len,
        });
        offset += ENTRY_BYTES;
    }

    (entries, offset)
}

// ── Byte-count estimators ────────────────────────────────────────────────

/// Mixed tile code bytes: base 4-bit codes plus rescued f32 tiles.
/// The estimate assumes standard 4-bit codes for non-rescued tiles.
fn mixed_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    // Upper bound: all tiles packed as NF4 (320 bytes each), plus worst-case
    // routing table overhead and f32 payload for rescued tiles.
    let nf4_bytes = (total_tiles as u64) * (NF4_TILE640_CODE_BYTES as u64);
    // Worst case: every tile rescued = TILE_ELEMENTS * 4 bytes of f32 payload.
    let payload_bytes = (total_tiles as u64) * (TILE_ELEMENTS as u64) * 4;
    nf4_bytes + payload_bytes
}

/// Mixed tile metadata bytes: base NF4 metadata plus rescue metadata.
fn mixed_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    // Base estimate: 5 groups x 8 bytes = 40 bytes per tile for NF4 metadata.
    let nf4_meta = (total_tiles as u64) * 40;
    // Routing table overhead: 4-byte header + 24 bytes per entry (worst case = every tile).
    let routing_overhead = 4u64 + (total_tiles as u64) * 24;
    nf4_meta + routing_overhead
}

// ── Tile error computation ───────────────────────────────────────────────

/// Compute the MSE for a single tile after NF4 quantization.
fn tile_mse(original: &[f32; TILE_ELEMENTS], group_size: usize) -> f32 {
    let groups = TILE_ELEMENTS / group_size;
    let mut total_sq_err = 0.0f32;

    for g in 0..groups {
        let base = g * group_size;
        // Compute scale (max-abs)
        let max_abs = original[base..base + group_size]
            .iter()
            .map(|v| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let scale = if max_abs < 1e-30 { 1.0f32 } else { max_abs };

        for i in 0..group_size {
            let orig = original[base + i];
            // Quantize and dequantize inline
            let norm = (orig / scale).clamp(-1.0, 1.0);
            let code = nf4_quantize_closest(norm);
            let decoded = nf4_dequantize(code) * scale;
            let err = orig - decoded;
            total_sq_err += err * err;
        }
    }

    total_sq_err / (TILE_ELEMENTS as f32)
}

/// Find the closest NF4 codebook index for a value in [-1, 1].
fn nf4_quantize_closest(value: f32) -> u8 {
    let mut best_idx = 7u8; // default: 0.0
    let mut best_dist = f32::MAX;
    for (i, &cb_val) in NF4_CODEBOOK.iter().enumerate() {
        let d = (value - cb_val).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }
    best_idx
}

// ── Tile rescuing logic ──────────────────────────────────────────────────

/// Pack a weight matrix with mixed tile rescue.
///
/// 1. Pack all tiles using NF4 (standard tile packer).
/// 2. Compute reconstruction error per tile.
/// 3. Identify the worst `rescue_count` tiles by highest MSE.
/// 4. Build a routing table in the extra bytes and store each rescued
///    tile's raw f32 values in the extra payload.
/// 5. Non-rescued tiles use their normal NF4 codes/scales/biases.
///
/// The extra bytes encode the routing table followed by f32 payloads.
/// The packer returns `(Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>)` where
/// the extra `Vec<f32>` encodes the routing table + payload bytes by
/// reinterpreting each 4-byte LE chunk as an f32.
fn pack_mixed_tile_matrix(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    rescue_fraction: f32,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>) {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let codes_per_tile = TILE_ELEMENTS / 2;

    // Step 1: Pack all tiles normally.
    let mut all_tiles: Vec<[f32; TILE_ELEMENTS]> = Vec::with_capacity(total_tiles);
    let mut all_codes: Vec<Vec<u8>> = Vec::with_capacity(total_tiles);
    let mut all_scales: Vec<Vec<f32>> = Vec::with_capacity(total_tiles);
    let mut all_biases: Vec<Vec<f32>> = Vec::with_capacity(total_tiles);
    let mut tile_errors: Vec<(usize, f32)> = Vec::with_capacity(total_tiles);

    for row in 0..in_features {
        let row_base = row * out_features;
        for t in 0..tiles_per_row {
            let col_start = t * TILE_ELEMENTS;
            let mut tile_buf = [0.0f32; TILE_ELEMENTS];
            let remaining = out_features.saturating_sub(col_start);
            let copy_len = remaining.min(TILE_ELEMENTS);
            for i in 0..copy_len {
                tile_buf[i] = weights[row_base + col_start + i];
            }

            let (t_codes, t_scales, t_biases) =
                pack_nf4_tile_with_group_size(&tile_buf, group_size);

            let mse = tile_mse(&tile_buf, group_size);
            let tile_idx = all_tiles.len();

            all_tiles.push(tile_buf);
            all_codes.push(t_codes);
            all_scales.push(t_scales);
            all_biases.push(t_biases);
            tile_errors.push((tile_idx, mse));
        }
    }

    // Step 2: Select worst tiles for rescue.
    tile_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let rescue_count = ((total_tiles as f32) * rescue_fraction).ceil() as usize;
    let rescue_count = rescue_count.min(total_tiles);

    let mut rescued_set = std::collections::HashSet::new();
    for i in 0..rescue_count {
        rescued_set.insert(tile_errors[i].0);
    }

    // Step 3: Build routing entries and payload bytes.
    let mut entries: Vec<MixedTileRoutingEntry> = Vec::new();
    let mut payload_bytes: Vec<u8> = Vec::new();
    for tile_idx in 0..total_tiles {
        if rescued_set.contains(&tile_idx) {
            let payload_offset = payload_bytes.len() as u64;
            // Store the original tile data as raw f32 LE bytes.
            for &val in &all_tiles[tile_idx] {
                payload_bytes.extend_from_slice(&val.to_le_bytes());
            }
            entries.push(MixedTileRoutingEntry {
                unit_id: tile_idx as u64,
                granularity: RescueGranularity::Tile640,
                overlay_mode: OverlayMode::FullReplacement,
                rescue_format: 0, // f32
                payload_offset,
                payload_len: (TILE_ELEMENTS * 4) as u32,
            });
        }
    }

    // Step 4: Concatenate routing table header + payload.
    let routing_bytes = serialize_routing_table(&entries);
    let mut extra_bytes = Vec::with_capacity(routing_bytes.len() + payload_bytes.len());
    extra_bytes.extend_from_slice(&routing_bytes);
    extra_bytes.extend_from_slice(&payload_bytes);

    // Step 5: Build output buffers — all tiles keep their NF4 codes.
    // The routing table is the sole authority for rescued tiles, so
    // there is no need for sentinel values.
    let mut codes = Vec::with_capacity(total_tiles * codes_per_tile);
    let mut scales = Vec::with_capacity(total_tiles * groups_per_tile);
    let mut biases = Vec::with_capacity(total_tiles * groups_per_tile);

    for tile_idx in 0..total_tiles {
        codes.extend(&all_codes[tile_idx]);
        scales.extend(&all_scales[tile_idx]);
        biases.extend(&all_biases[tile_idx]);
    }

    (codes, scales, biases, extra_bytes)
}

// ── Mixed tile unpacker ──────────────────────────────────────────────────

/// Unpack a mixed-tile encoded matrix.
///
/// Reads the routing table from the extra bytes to identify rescued tiles.
/// Non-rescued tiles are decoded via standard NF4 group-wise dequantization.
fn unpack_mixed_tile(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    extra: &[u8],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    _rescue_fraction: f32,
) -> Vec<f32> {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let codes_per_tile = TILE_ELEMENTS / 2;

    // Parse the routing table from extra bytes.
    let (entries, header_size) = deserialize_routing_table(extra);
    // Build a lookup: unit_id -> (byte_offset_in_extra, payload_len)
    let mut rescue_lookup = std::collections::HashMap::new();
    for e in &entries {
        let byte_offset = header_size + e.payload_offset as usize;
        rescue_lookup.insert(e.unit_id, (byte_offset, e.payload_len as usize));
    }

    // Decompose into normal NF4 tiles and rescued tiles.
    let mut output = vec![0.0f32; in_features * out_features];

    for tile_idx in 0..total_tiles {
        let row = tile_idx / tiles_per_row;
        let tile_in_row = tile_idx % tiles_per_row;
        let col_base = tile_in_row * TILE_ELEMENTS;

        if let Some(&(byte_offset, payload_len)) = rescue_lookup.get(&(tile_idx as u64)) {
            // Reconstruct from extra f32 payload (LE bytes).
            let payload_end = (byte_offset + payload_len).min(extra.len());
            let mut f32_idx = 0usize;
            let mut offset = byte_offset;
            while offset + 4 <= payload_end && f32_idx < TILE_ELEMENTS {
                let val = f32::from_le_bytes(extra[offset..offset + 4].try_into().unwrap());
                let out_pos = row * out_features + col_base + f32_idx;
                if out_pos < in_features * out_features && col_base + f32_idx < out_features {
                    output[out_pos] = val;
                }
                f32_idx += 1;
                offset += 4;
            }
        } else {
            // Regular NF4 tile — reuse standard unpacker logic.
            let code_start = tile_idx * codes_per_tile;
            let scale_start = tile_idx * groups_per_tile;
            let mut tile_out = [0.0f32; TILE_ELEMENTS];

            for g in 0..groups_per_tile {
                let scale = scales[scale_start + g];
                let bias = biases[scale_start + g];
                let cb_base = code_start + g * (group_size / 2);
                let out_base = g * group_size;

                for i in 0..(group_size / 2) {
                    let packed = codes[cb_base + i];
                    let code0 = packed & 0x0F;
                    let code1 = (packed >> 4) & 0x0F;
                    tile_out[out_base + 2 * i] = nf4_dequantize(code0) * scale + bias;
                    tile_out[out_base + 2 * i + 1] = nf4_dequantize(code1) * scale + bias;
                }
            }

            for i in 0..TILE_ELEMENTS {
                let out_pos = row * out_features + col_base + i;
                if out_pos < in_features * out_features && col_base + i < out_features {
                    output[out_pos] = tile_out[i];
                }
            }
        }
    }

    output
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all MixedTile family candidates from the sweep grid.
pub fn generate_mixed_tile_candidates(grid: &MixedTileSweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    // Default group_size = 128 for the base 4-bit codec.
    let default_group_size: usize = 128;

    for base_policy in &grid.base_policies {
        for schedule in &grid.schedules {
            // Extract a scalar rescue fraction from the schedule (total across all rounds).
            let rf_total: f32 = rescue_fraction_total(schedule);

            let params = json!({
                "family": "MixedTile",
                "base_policy": base_policy,
                "rescue_fraction": rf_total,
                "group_size": default_group_size,
                "rescue_schedule": schedule,
            });

            let gs = default_group_size;
            let rf = rf_total;

            let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
                pack_mixed_tile_matrix(w, r, c, gs, rf)
            });

            let unpacker = Box::new(
                move |codes: &[u8],
                      scales: &[f32],
                      biases: &[f32],
                      extra: &[u8],
                      rows: usize,
                      cols: usize| {
                    unpack_mixed_tile(codes, scales, biases, extra, rows, cols, gs, rf)
                },
            );

            candidates.push(FamilyCandidate {
                label: format!("MixedTile_{}_rescue{:.2}", base_policy.family, rf_total),
                parameters: params,
                packer,
                unpacker,
                code_bytes_fn: mixed_code_bytes,
                metadata_bytes_fn: Box::new(mixed_metadata_bytes),
            });
        }
    }

    candidates
}

/// Total rescue fraction from a schedule (sum of all rounds).
fn rescue_fraction_total(schedule: &RescueSchedule) -> f32 {
    match schedule {
        RescueSchedule::OneShot { fraction } => *fraction as f32,
        RescueSchedule::FixedPerRound {
            fraction_per_round,
            rounds,
        } => (*fraction_per_round * *rounds as f64) as f32,
        RescueSchedule::Geometric { fractions } => fractions.iter().sum::<f64>() as f32,
    }
}
