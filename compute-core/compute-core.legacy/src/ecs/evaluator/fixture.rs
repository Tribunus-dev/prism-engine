//! Evaluation fixture types — codec-correct packed bytes, inputs, oracles.

use serde::{Deserialize, Serialize};

/// Codec-correct evaluation fixture.
///
/// Every variant owns its packed payloads, scales or codebooks, input data,
/// decoded-weight oracle, output oracle, dimensions, and payload digests.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
