//! ANE warmup MIL program.
//!
//! This module owns the canonical authority for the embedded MIL
//! program used to warm the Apple Neural Engine firmware via the
//! Core ML compilation path. The bytes are exposed as a `&'static
//! [u8]` so other constitutional crates and the engine can embed
//! the same MIL source without duplicating it.

/// MIL program for ANE warmup (x * x element-wise multiply on a
/// `tensor<fp32, [1, 1, 1, 1]>`). The program is in Core ML MIL
/// 1.3 and casts to fp16, multiplies, then casts back to fp32 so
/// the ANE compiler is forced to exercise the full fp16/fp32
/// convert + multiply path.
pub const ANE_WARMUP_MIL: &[u8] = include_bytes!("ane_warmup_mil.bytes");

/// Borrow the ANE warmup MIL program as a byte slice.
///
/// This is the canonical accessor used by
/// [`coreai_warmup::build_warmup_mlpackage`](super::coreai_warmup::build_warmup_mlpackage)
/// when assembling the `.mlpackage` bundle.
pub fn ane_warmup_mil() -> &'static [u8] {
    ANE_WARMUP_MIL
}
