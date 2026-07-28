//! BitNet b1.58 2B4T native ternary weight importer.
//!
//! This module owns the canonical authority for ingesting already-ternary
//! {-1, 0, +1} weights from HuggingFace safetensors checkpoints into
//! Prism's cimage format, emitting cimage shards through the phased
//! builder system.
//!
//! The constitutional re-implementation is functionally identical to
//! the engine's pre-absorption `compute-core/src/ecs/bitnet/` module.
//! The engine's parallel `bitnet` directory has been deleted; engine
//! callers now read from `prism_ecs_quantization::bitnet`.
//!
//! # Module structure
//!
//! - `ternary_codec` — `TernaryPackedTensor`, `TernaryCodecError`,
//!   `pack_ternary_codes`, `unpack_ternary_codes`,
//!   `validate_no_reserved_codes`.
//! - `cimage_shim` — minimal self-contained copy of the cimage
//!   manifest / payload / writer types the bitnet module needs
//!   (structurally identical to the engine's, to be re-merged when
//!   the cimage subsystem is itself absorbed).
//! - `importer` — `BitNetImporter` and its deterministic
//!   pseudo-random ternary weight generation.
//! - `checkpoint` — `BitNetCheckpoint` safetensors loader and
//!   `make_ternary_from_checkpoint`.
//! - `reference` — pure-Rust CPU reference for a single decoder
//!   layer (`bitnet_decoder_layer_reference`,
//!   `bitnet_decoder_logits`).
//! - `kv` — `BitNetKvCache` and the cimage KV cache manifest entry
//!   helper.
//! - `phases` — phased cimage emission:
//!   `emit_single_bitnet_linear`, `emit_bitnet_mlp_block`,
//!   `emit_bitnet_decoder_layer`, `emit_bitnet_full_model`,
//!   `emit_bitnet_from_checkpoint`,
//!   plus `BitNetDecoderLayerShardConfig`.
//! - `text` — auto-regressive token-wise inference loop
//!   (`prefill`, `decode_single`, `greedy_sample`, `run_text`,
//!   `BitNetTokenizer`).

pub mod checkpoint;
pub mod cimage_shim;
pub mod importer;
pub mod kv;
pub mod phases;
pub mod reference;
pub mod ternary_codec;
pub mod text;

#[cfg(test)]
mod tests;
