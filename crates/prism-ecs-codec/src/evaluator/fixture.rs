//! EvaluationFixture — codec-correct evaluation fixtures.
//!
//! This module owns the canonical authority for the input/output
//! data and reference oracles that drive a single evaluation run.
//! Each variant owns its packed payloads, scales or codebooks,
//! input data, decoded-weight oracle, output oracle, dimensions,
//! and payload digest. The digest is the fixture's content
//! address; any payload change invalidates the fixture.

use serde::{Deserialize, Serialize};

/// Codec-correct evaluation fixture.
///
/// Every variant owns its packed payloads, scales or codebooks, input data,
/// decoded-weight oracle, output oracle, dimensions, and payload digests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationFixture {
    /// NF4 Tile640 fixture.
    Nf4 {
        codes: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        input: Vec<f32>,
        reference: Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        digest: [u8; 32],
    },
    /// Ternary { -1, 0, +1 } fixture.
    Ternary {
        packed: Vec<u8>,
        scale: f32,
        input: Vec<f32>,
        reference: Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        digest: [u8; 32],
    },
    /// INT8 block-scaled fixture.
    Int8 {
        weights: Vec<i8>,
        scales: Vec<f32>,
        zero_points: Vec<i8>,
        input: Vec<f32>,
        reference: Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        digest: [u8; 32],
    },
    /// FP16 fixture.
    Fp16 {
        weights: Vec<u16>,
        input: Vec<f32>,
        reference: Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        digest: [u8; 32],
    },
}

impl EvaluationFixture {
    /// Returns the fixture's content-addressed digest.
    pub fn digest(&self) -> [u8; 32] {
        match self {
            Self::Nf4 { digest, .. }
            | Self::Ternary { digest, .. }
            | Self::Int8 { digest, .. }
            | Self::Fp16 { digest, .. } => *digest,
        }
    }

    /// Returns the codec family of this fixture.
    pub fn codec_id(&self) -> &'static str {
        match self {
            Self::Nf4 { .. } => "nf4",
            Self::Ternary { .. } => "ternary",
            Self::Int8 { .. } => "int8",
            Self::Fp16 { .. } => "fp16",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nf4_fixture() -> EvaluationFixture {
        EvaluationFixture::Nf4 {
            codes: vec![0xAB; 128],
            scales: vec![1.0; 4],
            biases: vec![0.0; 4],
            input: vec![0.5; 256],
            reference: vec![0.0; 64],
            m: 4,
            k: 256,
            n: 16,
            digest: [7u8; 32],
        }
    }

    fn ternary_fixture() -> EvaluationFixture {
        EvaluationFixture::Ternary {
            packed: vec![0x55; 32],
            scale: 0.5,
            input: vec![0.5; 128],
            reference: vec![0.0; 32],
            m: 4,
            k: 128,
            n: 8,
            digest: [11u8; 32],
        }
    }

    fn int8_fixture() -> EvaluationFixture {
        EvaluationFixture::Int8 {
            weights: vec![1; 128],
            scales: vec![1.0; 4],
            zero_points: vec![0; 4],
            input: vec![0.5; 128],
            reference: vec![0.0; 32],
            m: 4,
            k: 128,
            n: 8,
            digest: [13u8; 32],
        }
    }

    fn fp16_fixture() -> EvaluationFixture {
        EvaluationFixture::Fp16 {
            weights: vec![0x3C00; 64], // 1.0 in FP16
            input: vec![0.5; 128],
            reference: vec![0.0; 32],
            m: 4,
            k: 64,
            n: 8,
            digest: [17u8; 32],
        }
    }

    #[test]
    fn all_variants_constructible_and_serializable() {
        for fixture in [
            nf4_fixture(),
            ternary_fixture(),
            int8_fixture(),
            fp16_fixture(),
        ] {
            let json = serde_json::to_string(&fixture).expect("serialize");
            let restored: EvaluationFixture = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(restored, fixture);
        }
    }

    #[test]
    fn digest_returns_payload_digest() {
        assert_eq!(nf4_fixture().digest(), [7u8; 32]);
        assert_eq!(ternary_fixture().digest(), [11u8; 32]);
        assert_eq!(int8_fixture().digest(), [13u8; 32]);
        assert_eq!(fp16_fixture().digest(), [17u8; 32]);
    }

    #[test]
    fn codec_id_matches_variant() {
        assert_eq!(nf4_fixture().codec_id(), "nf4");
        assert_eq!(ternary_fixture().codec_id(), "ternary");
        assert_eq!(int8_fixture().codec_id(), "int8");
        assert_eq!(fp16_fixture().codec_id(), "fp16");
    }
}
