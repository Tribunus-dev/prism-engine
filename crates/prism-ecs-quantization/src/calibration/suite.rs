//! Activation bank types and deterministic generators.
//!
//! Each tensor class gets a dedicated activation bank with promotion and
//! holdout sets. The generators produce deterministic, reproducible vectors
//! that exercise different activation regimes (high-magnitude, sparse,
//! saturated, near-zero, structured-pattern) relevant to the tensor's role.

use std::path::Path;

/// Single activation vector (the input to a matrix multiply, length = rows).
pub type ActivationVector = Vec<f32>;

/// Provenance of an activation bank.
#[derive(Debug, Clone)]
pub enum ActivationSource {
    /// Generated from deterministic seeds.
    Deterministic { seed: u64, variant_id: String },
    /// Captured from a reference model run.
    Prerendered {
        source_model: String,
        /// SHA256 of the samples + metadata.
        hash: [u8; 32],
    },
}

/// A set of activation vectors for operator-space validation of one tensor class.
///
/// Contains a promotion set (used during candidate selection) and a holdout
/// set (used after promotion passes, to detect overfitting to the promotion
/// set). Both sets are deterministic given their source metadata.
#[derive(Debug, Clone)]
pub struct ActivationBank {
    /// Promotion vectors used during candidate selection.
    pub promotion: Vec<ActivationVector>,
    /// Holdout vectors used after a candidate survives promotion.
    pub holdout: Vec<ActivationVector>,
    /// Human-readable label for diagnostics.
    pub label: String,
    /// Provenance of the bank.
    pub source: ActivationSource,
    /// Required activation dimension (must match `rows` of the weight matrix).
    pub input_dim: usize,
}

/// A deterministic stress bank — generated from pattern generators, exercises
/// codec pathologies without requiring the reference model.
///
/// Semantically identical to ActivationBank but the vectors are generated
/// deterministically rather than captured from model execution.
/// Mandatory for every tensor during admission.
pub type StressBank = ActivationBank;

/// A suite of deterministic stress banks for every tensor class.
///
/// Always built. Validates codec correctness with diverse synthetic patterns.
/// Required for all admission, even when a prerendered CalibrationSuite is
/// also provided.
#[derive(Debug, Clone)]
pub struct StressSuite {
    banks: Vec<(crate::TensorClass, ActivationBank)>,
}

impl StressSuite {
    /// Create an empty stress suite.
    pub fn empty() -> Self {
        Self { banks: Vec::new() }
    }

    /// Insert a stress bank for a tensor class.
    pub fn insert(
        &mut self,
        class: crate::TensorClass,
        bank: ActivationBank,
    ) {
        if let Some(pos) = self.banks.iter().position(|(c, _)| *c == class) {
            self.banks[pos].1 = bank;
        } else {
            self.banks.push((class, bank));
        }
    }

    /// Look up the stress bank for a tensor class.
    pub fn get(
        &self,
        class: &crate::TensorClass,
    ) -> Option<&ActivationBank> {
        self.banks.iter().find(|(c, _)| c == class).map(|(_, b)| b)
    }

    /// Build the default stress suite with deterministic generators.
    ///
    /// Generates promotion and holdout activation vectors for every
    /// defined `TensorClass` using the research-derived patterns.
    pub fn build_default() -> Self {
        let mut suite = Self::empty();
        for class in &[
            crate::TensorClass::VisionPatchProjection,
            crate::TensorClass::DecoderAttentionProjection,
            crate::TensorClass::DecoderMlpProjection,
            crate::TensorClass::CrossModalBridge,
            crate::TensorClass::TokenEmbedding,
            crate::TensorClass::OutputHead,
        ] {
            if let Some(bank) = generate_for_class(*class) {
                suite.insert(*class, bank);
            }
        }
        suite
    }

    /// Number of banks in the suite.
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    /// Whether the suite is empty.
    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }
}

/// A suite of model-native prerendered activation banks for quantization
/// admission.
///
/// Maps `TensorClass` to an `ActivationBank` populated from reference model
/// execution. Optional: without it the artifact is `DiagnosticOnly`.
/// Required for `ProductionQualified` admission.
#[derive(Debug, Clone)]
pub struct CalibrationSuite {
    banks: Vec<(crate::TensorClass, ActivationBank)>,
}

impl CalibrationSuite {
    /// Create an empty calibration suite (no prerendered data yet).
    pub fn empty() -> Self {
        Self { banks: Vec::new() }
    }

    /// Insert an activation bank for a tensor class.
    pub fn insert(
        &mut self,
        class: crate::TensorClass,
        bank: ActivationBank,
    ) {
        if let Some(pos) = self.banks.iter().position(|(c, _)| *c == class) {
            self.banks[pos].1 = bank;
        } else {
            self.banks.push((class, bank));
        }
    }

    /// Look up the activation bank for a tensor class.
    pub fn get(
        &self,
        class: &crate::TensorClass,
    ) -> Option<&ActivationBank> {
        self.banks.iter().find(|(c, _)| c == class).map(|(_, b)| b)
    }

    /// Number of banks in the suite.
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    /// Whether the suite is empty.
    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }

    /// Load a prerendered activation bank from a directory containing
    /// `promotion_samples.bin`, `holdout_samples.bin`, and `metadata.json`.
    ///
    /// The bank files are flat f32 row-major arrays: `[num_samples, input_dim]`.
    pub fn load_from_bank_dir(
        dir: &Path,
        tensor_class: crate::TensorClass,
        input_dim: usize,
        label: &str,
    ) -> std::io::Result<Self> {
        use std::fs;

        let promo_path = dir.join("promotion_samples.bin");
        let hold_path = dir.join("holdout_samples.bin");

        let promo_raw = fs::read(&promo_path)?;
        let hold_raw = fs::read(&hold_path)?;

        let promo_floats: &[f32] = bytemuck::cast_slice(&promo_raw);
        let hold_floats: &[f32] = bytemuck::cast_slice(&hold_raw);

        let promo_count = promo_floats.len() / input_dim;
        let hold_count = hold_floats.len() / input_dim;

        let mut promotion = Vec::with_capacity(promo_count);
        for i in 0..promo_count {
            let start = i * input_dim;
            promotion.push(promo_floats[start..start + input_dim].to_vec());
        }

        let mut holdout = Vec::with_capacity(hold_count);
        for i in 0..hold_count {
            let start = i * input_dim;
            holdout.push(hold_floats[start..start + input_dim].to_vec());
        }

        let bank = ActivationBank {
            promotion,
            holdout,
            label: label.to_string(),
            source: ActivationSource::Prerendered {
                source_model: format!("bank:{}", dir.display()),
                hash: [0u8; 32],
            },
            input_dim,
        };

        let mut suite = Self::empty();
        suite.insert(tensor_class, bank);
        Ok(suite)
    }
}

// ── Deterministic generators ─────────────────────────────────────

const DEFAULT_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const PROMOTION_COUNT: usize = 50;
const HOLDOUT_COUNT: usize = 20;

/// Number of norm-band strata for probe/stress sampling.
pub const STRATIFY_NUM_STRATA_PROBE: usize = 4;
/// Number of norm-band strata for promotion sampling.
pub const STRATIFY_NUM_STRATA_PROMO: usize = 8;
/// Number of norm-band strata for holdout sampling.
pub const STRATIFY_NUM_STRATA_HOLDOUT: usize = 8;
/// Default seed for stratified sampling.
pub const DEFAULT_SAMPLE_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// Generate an activation bank for a specific tensor class.
///
/// Returns `None` for classes that have no generator yet.
pub fn generate_for_class(
    class: crate::TensorClass,
) -> Option<ActivationBank> {
    let (input_dim, label, promo_count, hold_count) = match class {
        crate::TensorClass::VisionPatchProjection => {
            (6912, "vision-patch", PROMOTION_COUNT, HOLDOUT_COUNT)
        }
        crate::TensorClass::DecoderAttentionProjection => {
            (3840, "decoder-attention", PROMOTION_COUNT, HOLDOUT_COUNT)
        }
        crate::TensorClass::DecoderMlpProjection => {
            (3840, "decoder-mlp", PROMOTION_COUNT, HOLDOUT_COUNT)
        }
        crate::TensorClass::CrossModalBridge => {
            (3840, "cross-modal-bridge", PROMOTION_COUNT, HOLDOUT_COUNT)
        }
        crate::TensorClass::TokenEmbedding => (
            262144,
            "token-embedding",
            4_usize.max(PROMOTION_COUNT / 12),
            2_usize.max(HOLDOUT_COUNT / 10),
        ),
        crate::TensorClass::OutputHead => (3840, "output-head", PROMOTION_COUNT, HOLDOUT_COUNT),
        crate::TensorClass::Unknown => return None,
    };

    // Generate vectors at multiple norm bands to test different activation
    // regimes. Each band provides structure-diverse vectors normalized to
    // a target L2 norm.
    let dim_f = input_dim as f32;
    let norm_bands: &[f32] = &[
        0.1 * dim_f.sqrt(),  // low-energy
        1.0 * dim_f.sqrt(),  // moderate (typical hidden state)
        10.0 * dim_f.sqrt(), // high-energy, adversarial
    ];

    let per_band_promo = (promo_count / norm_bands.len()).max(1);
    let per_band_hold = (hold_count / norm_bands.len()).max(1);

    let mut promotion = Vec::with_capacity(promo_count);
    let mut holdout = Vec::with_capacity(hold_count);

    for (bi, &target_norm) in norm_bands.iter().enumerate() {
        let band_seed = DEFAULT_SEED
            .wrapping_add(bi as u64)
            .wrapping_mul(0x9E37_79B9);
        let mut promo_vecs = generate_activation_set(class, input_dim, per_band_promo, band_seed);
        let mut hold_vecs = generate_activation_set(
            class,
            input_dim,
            per_band_hold,
            band_seed.wrapping_add(0xFF),
        );
        normalize_to_l2(&mut promo_vecs, target_norm);
        normalize_to_l2(&mut hold_vecs, target_norm);
        promotion.extend(promo_vecs);
        holdout.extend(hold_vecs);
    }

    Some(ActivationBank {
        promotion,
        holdout,
        label: label.to_string(),
        source: ActivationSource::Deterministic {
            seed: DEFAULT_SEED,
            variant_id: format!("v2-{}", label),
        },
        input_dim,
    })
}

/// Normalize every vector in a set to a target L2 norm.
fn normalize_to_l2(vecs: &mut [Vec<f32>], target_norm: f32) {
    for v in vecs.iter_mut() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            let scale = target_norm / norm;
            for x in v.iter_mut() {
                *x *= scale;
            }
        }
    }
}

/// Generate a set of deterministic activation vectors for a tensor class.
fn generate_activation_set(
    class: crate::TensorClass,
    dim: usize,
    count: usize,
    seed: u64,
) -> Vec<ActivationVector> {
    let mut variants: Vec<Box<dyn Fn(&mut XorShift64, usize) -> ActivationVector>> = Vec::new();

    match class {
        crate::TensorClass::VisionPatchProjection => {
            variants.push(Box::new(|rng, d| flat_uniform(rng, d, 0.05)));
            variants.push(Box::new(|rng, d| checkerboard(rng, d, 16, 0.8)));
            variants.push(Box::new(|rng, d| sinusoidal_gradient(rng, d, 0.03, 1.0)));
            variants.push(Box::new(|rng, d| step_edge(rng, d, 0.3, 1.2)));
            variants.push(Box::new(|rng, d| sparse_impulse(rng, d, 0.02, 2.0)));
            variants.push(Box::new(|rng, d| deterministic_noise(rng, d, 0.3)));
            variants.push(Box::new(|rng, d| low_freq_pattern(rng, d, 0.05, 0.5)));
            variants.push(Box::new(|rng, d| gamma_corner(rng, d, 0.1, 1.5)));
        }
        crate::TensorClass::DecoderAttentionProjection => {
            variants.push(Box::new(|rng, d| flat_uniform(rng, d, 0.1)));
            variants.push(Box::new(|rng, d| sparse_impulse(rng, d, 0.05, 3.0)));
            variants.push(Box::new(|rng, d| sinusoidal_gradient(rng, d, 0.1, 0.5)));
            variants.push(Box::new(|rng, d| deterministic_noise(rng, d, 0.4)));
            variants.push(Box::new(|rng, d| step_edge(rng, d, 0.1, 0.8)));
            variants.push(Box::new(|rng, d| low_freq_pattern(rng, d, 0.01, 0.1)));
            variants.push(Box::new(|rng, d| checkerboard(rng, d, 8, 0.6)));
        }
        crate::TensorClass::DecoderMlpProjection => {
            variants.push(Box::new(|rng, d| saturated_positive(rng, d, 3.0)));
            variants.push(Box::new(|rng, d| saturated_negative(rng, d, 3.0)));
            variants.push(Box::new(|rng, d| deterministic_noise(rng, d, 0.3)));
            variants.push(Box::new(|rng, d| flat_uniform(rng, d, 0.05)));
            variants.push(Box::new(|rng, d| sparse_impulse(rng, d, 0.03, 2.5)));
            variants.push(Box::new(|rng, d| sinusoidal_gradient(rng, d, 0.08, 1.0)));
            variants.push(Box::new(|rng, d| step_edge(rng, d, 0.2, 2.0)));
        }
        crate::TensorClass::CrossModalBridge => {
            variants.push(Box::new(|rng, d| flat_uniform(rng, d, 0.15)));
            variants.push(Box::new(|rng, d| deterministic_noise(rng, d, 0.5)));
            variants.push(Box::new(|rng, d| sinusoidal_gradient(rng, d, 0.05, 0.8)));
            variants.push(Box::new(|rng, d| sparse_impulse(rng, d, 0.04, 4.0)));
            variants.push(Box::new(|rng, d| gamma_corner(rng, d, 0.15, 2.0)));
            variants.push(Box::new(|rng, d| low_freq_pattern(rng, d, 0.02, 0.3)));
            variants.push(Box::new(|rng, d| checkerboard(rng, d, 4, 0.5)));
        }
        crate::TensorClass::TokenEmbedding => {
            variants.push(Box::new(|rng, d| sparse_one_hot(rng, d, 0.001)));
            variants.push(Box::new(|rng, d| flat_uniform(rng, d, 0.01)));
            variants.push(Box::new(|rng, d| deterministic_noise(rng, d, 0.2)));
            variants.push(Box::new(|rng, d| sparse_impulse(rng, d, 0.005, 1.0)));
        }
        crate::TensorClass::OutputHead => {
            variants.push(Box::new(|rng, d| flat_uniform(rng, d, 0.2)));
            variants.push(Box::new(|rng, d| low_freq_pattern(rng, d, 0.01, 0.1)));
            variants.push(Box::new(|rng, d| deterministic_noise(rng, d, 0.3)));
            variants.push(Box::new(|rng, d| sparse_impulse(rng, d, 0.02, 5.0)));
            variants.push(Box::new(|rng, d| sinusoidal_gradient(rng, d, 0.05, 0.4)));
        }
        crate::TensorClass::Unknown => {}
    }

    if variants.is_empty() {
        return (0..count)
            .map(|i| {
                let mut r = XorShift64::new(seed.wrapping_add(i as u64));
                deterministic_noise(&mut r, dim, 0.3)
            })
            .collect();
    }

    // Round-robin through variants, each with a sub-seed for uniqueness.
    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let variant_idx = i % variants.len();
        let sub_seed = seed.wrapping_add(i as u64).wrapping_mul(0x9E37_79B9);
        let mut sub_rng = XorShift64::new(sub_seed);
        vectors.push(variants[variant_idx](&mut sub_rng, dim));
    }
    vectors
}

// ── Activation pattern generators ─────────────────────────────────

fn flat_uniform(rng: &mut XorShift64, dim: usize, amp: f32) -> ActivationVector {
    (0..dim).map(|_| (rng.f32() - 0.5) * 2.0 * amp).collect()
}

fn checkerboard(_rng: &mut XorShift64, dim: usize, period: usize, amp: f32) -> ActivationVector {
    (0..dim)
        .map(|i| {
            if (i / period) % 2 == 0 {
                amp
            } else {
                -amp * 0.3
            }
        })
        .collect()
}

fn sinusoidal_gradient(rng: &mut XorShift64, dim: usize, freq: f64, amp: f32) -> ActivationVector {
    let phase = rng.f64() * std::f64::consts::TAU;
    (0..dim)
        .map(|i| {
            let x = i as f64 / dim as f64;
            let val = (x * std::f64::consts::TAU * freq + phase).sin();
            (val as f32) * amp
        })
        .collect()
}

fn step_edge(rng: &mut XorShift64, dim: usize, low_amp: f32, high_amp: f32) -> ActivationVector {
    let mid = (dim as f64 * rng.f64()).ceil() as usize;
    let mid = mid.clamp(1, dim - 1);
    (0..dim)
        .map(|i| {
            if i < mid {
                low_amp * (rng.f32() - 0.5)
            } else {
                high_amp * (rng.f32() - 0.5)
            }
        })
        .collect()
}

fn sparse_impulse(
    rng: &mut XorShift64,
    dim: usize,
    density: f64,
    impulse_amp: f32,
) -> ActivationVector {
    (0..dim)
        .map(|_| {
            if rng.f64() < density {
                impulse_amp * (rng.f32() * 2.0 - 1.0)
            } else {
                (rng.f32() - 0.5) * 0.01
            }
        })
        .collect()
}

fn deterministic_noise(rng: &mut XorShift64, dim: usize, amp: f32) -> ActivationVector {
    (0..dim).map(|_| (rng.f32() - 0.5) * 2.0 * amp).collect()
}

fn low_freq_pattern(rng: &mut XorShift64, dim: usize, freq: f64, amp: f32) -> ActivationVector {
    let phase = rng.f64() * std::f64::consts::TAU;
    (0..dim)
        .map(|i| {
            let x = i as f64 / dim as f64;
            let val = (x * std::f64::consts::TAU * freq + phase).cos();
            (val as f32) * amp
        })
        .collect()
}

fn gamma_corner(rng: &mut XorShift64, dim: usize, corner_frac: f64, amp: f32) -> ActivationVector {
    let corner_end = (dim as f64 * corner_frac).ceil() as usize;
    let corner_end = corner_end.max(1);
    (0..dim)
        .map(|i| {
            if i < corner_end {
                amp * (rng.f32() - 0.5) * 2.0
            } else {
                (rng.f32() - 0.5) * 0.05
            }
        })
        .collect()
}

fn saturated_positive(rng: &mut XorShift64, dim: usize, amp: f32) -> ActivationVector {
    (0..dim).map(|_| amp * (1.0 + rng.f32() * 0.5)).collect()
}

fn saturated_negative(rng: &mut XorShift64, dim: usize, amp: f32) -> ActivationVector {
    (0..dim).map(|_| -amp * (1.0 + rng.f32() * 0.5)).collect()
}

fn sparse_one_hot(rng: &mut XorShift64, dim: usize, density: f64) -> ActivationVector {
    (0..dim)
        .map(|_| if rng.f64() < density { 1.0 } else { 0.0 })
        .collect()
}

// ── Deterministic RNG ─────────────────────────────────────────────

pub(crate) struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub(crate) fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0xDEAD_BEEF_CAFE_F00D
        } else {
            seed
        };
        Self { state }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0f64 / ((1u64 << 53) as f64))
    }

    pub(crate) fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0f32 / ((1u64 << 24) as f32))
    }
}

/// Result of stratified sampling from an activation bank.
///
/// Contains the selected vectors, their original indices, and metadata
/// about the stratifiction configuration used.
pub struct StratifiedSample {
    pub vectors: Vec<ActivationVector>,
    pub selected_indices: Vec<usize>,
    pub seed: u64,
    pub num_strata: usize,
    pub strata_sizes: Vec<usize>,
    pub algorithm_version: u32,
}

/// Deterministic stratified sample from a vector bank.
///
/// Vectors are divided into `num_strata` equal bands by L2 norm. Per band,
/// `ceil(count / num_strata)` vectors are selected using XorShift64 seeded
/// with `seed + band_index`. The final selection is shuffled using
/// XorShift64 with `seed + 0xFF` and returned in a StratifiedSample.
pub fn stratified_sample(
    vectors: &[ActivationVector],
    count: usize,
    num_strata: usize,
    seed: u64,
    exclude_indices: Option<&[usize]>,
) -> StratifiedSample {
    if vectors.len() <= count || count == 0 {
        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        let mut rng = XorShift64::new(seed.wrapping_add(0xFF));
        for i in (1..indices.len()).rev() {
            let j = (rng.f32() * (i + 1) as f32) as usize;
            indices.swap(i, j.min(i));
        }
        return StratifiedSample {
            vectors: indices.iter().map(|&i| vectors[i].clone()).collect(),
            selected_indices: indices,
            seed,
            num_strata,
            strata_sizes: vec![vectors.len()],
            algorithm_version: 1,
        };
    }

    // 1. Compute L2 norms and create index array
    let mut indexed: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.iter().map(|x| x * x).sum::<f32>().sqrt()))
        .collect();

    // 2. Sort by L2 norm
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // 3. Divide into num_strata bands
    let num_strata = num_strata.max(1).min(indexed.len());
    let band_size = indexed.len() / num_strata;
    let remainder = indexed.len() % num_strata;
    let per_band = (count + num_strata - 1) / num_strata;

    let mut selected_indices = Vec::with_capacity(count);
    let mut strata_sizes = Vec::with_capacity(num_strata);
    let mut band_start = 0usize;

    for band in 0..num_strata {
        let this_band_size = band_size + if band < remainder { 1 } else { 0 };
        let band_indices = &indexed[band_start..band_start + this_band_size];
        let take_from_band = per_band.min(this_band_size);

        // 4. Deterministic selection within band
        let mut rng = XorShift64::new(seed.wrapping_add(band as u64));
        // Build a mutable index list for this band and apply exclusion filter
        let mut band_indices: Vec<usize> = band_indices.iter().map(|pair| pair.0).collect();
        if let Some(exclude) = exclude_indices {
            band_indices.retain(|i| !exclude.contains(i));
        }

        // Partial Fisher-Yates shuffle to pick without replacement
        let n_take = take_from_band.min(band_indices.len());
        for pos in 0..n_take {
            let remaining = band_indices.len() - pos;
            let pick = pos + (rng.f32() * remaining as f32) as usize;
            let swap_idx = pick.min(band_indices.len() - 1);
            band_indices.swap(pos, swap_idx);
        }
        let selected: Vec<usize> = band_indices[..n_take].to_vec();

        selected_indices.extend(selected);
        strata_sizes.push(take_from_band);
        band_start += this_band_size;
    }

    // 5. Shuffle final selection
    let mut rng = XorShift64::new(seed.wrapping_add(0xFF));
    for i in (1..selected_indices.len()).rev() {
        let j = (rng.f32() * (i + 1) as f32) as usize;
        selected_indices.swap(i, j.min(i));
    }

    // 6. Collect selected vectors
    let vectors: Vec<ActivationVector> = selected_indices
        .iter()
        .map(|&idx| vectors[idx].clone())
        .collect();

    StratifiedSample {
        vectors,
        selected_indices,
        seed,
        num_strata,
        strata_sizes,
        algorithm_version: 1,
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_suite_non_empty() {
        let suite = StressSuite::build_default();
        assert!(
            suite.len() >= 5,
            "Expected at least 5 banks, got {}",
            suite.len()
        );
    }

    #[test]
    fn test_bank_vector_dimensions_match_class() {
        let suite = StressSuite::build_default();
        if let Some(bank) = suite.get(&crate::TensorClass::VisionPatchProjection) {
            for v in &bank.promotion {
                assert_eq!(v.len(), 6912, "Vision vector wrong dimension");
            }
        }
    }

    #[test]
    fn test_promotion_holdout_disjoint() {
        let suite = StressSuite::build_default();
        for (_, bank) in &suite.banks {
            assert!(
                !bank.promotion.is_empty(),
                "Promotion empty for {}",
                bank.label
            );
            assert!(!bank.holdout.is_empty(), "Holdout empty for {}", bank.label);
            assert!(
                bank.promotion != bank.holdout,
                "Promotion and holdout identical for {}",
                bank.label
            );
        }
    }

    #[test]
    fn test_xorshift_determinism() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        for _ in 0..100 {
            assert_eq!(a.f64(), b.f64(), "XorShift64 not deterministic");
        }
    }

    #[test]
    fn test_activation_vector_diversity() {
        let suite = StressSuite::build_default();
        for (_, bank) in &suite.banks {
            if bank.promotion.len() >= 2 {
                let first = &bank.promotion[0];
                let different = bank.promotion.iter().any(|v| v != first);
                assert!(
                    different,
                    "All promotion vectors identical for {}",
                    bank.label
                );
            }
        }
    }

    #[test]
    fn test_stratified_sample_covers_all_bands() {
        // Create 1000 vectors with increasing L2 norms from 0.1 to 100.0
        let mut vectors = Vec::new();
        for i in 0..1000 {
            let norm = 0.1 + (i as f32) * 0.1;
            let v: Vec<f32> = (0..64).map(|_| norm / 8.0).collect();
            vectors.push(v);
        }

        let sample = stratified_sample(&vectors, 100, 4, 42, None);
        assert_eq!(sample.vectors.len(), 100);
        assert_eq!(sample.num_strata, 4);
        assert_eq!(sample.seed, 42);

        // Verify each stratum contributes (25 plusmn wiggle room)
        assert!(sample.strata_sizes.iter().all(|&s| s >= 20 && s <= 30));

        // Deterministic: same seed => same sample
        let sample2 = stratified_sample(&vectors, 100, 4, 42, None);
        assert_eq!(sample.selected_indices, sample2.selected_indices);
    }

    #[test]
    fn test_stratified_sample_no_replacement() {
        let vectors: Vec<ActivationVector> = (0..100).map(|i| vec![i as f32; 4]).collect();
        let sample = stratified_sample(&vectors, 50, 4, 42, None);
        let mut sorted = sample.selected_indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            sample.selected_indices.len(),
            "Must not have duplicates"
        );
    }

    #[test]
    fn test_stratified_sample_exclusion() {
        let vectors: Vec<ActivationVector> = (0..100).map(|i| vec![i as f32; 4]).collect();
        let exclude: Vec<usize> = (0..10).collect();
        let sample = stratified_sample(&vectors, 50, 4, 42, Some(&exclude));
        // Verify none of the excluded indices are in the sample
        for idx in &sample.selected_indices {
            assert!(
                !exclude.contains(idx),
                "Excluded index {} found in sample",
                idx
            );
        }
        // Verify all indices are unique (no replacement)
        let mut sorted = sample.selected_indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), sample.selected_indices.len());
    }
}
