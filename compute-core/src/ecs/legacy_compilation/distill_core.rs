//! distill_core.rs — numeric core for the NF4-teacher → ternary-student loop.
//!
//! Pure, std-only metrics the block-by-block distillation loop
//! (`server::distill_worker`) calls to (a) measure how well the ternary student
//! reproduces the NF4 teacher and (b) decide whether a distilled block is
//! accepted. The teacher weights are made bindable to both the Metal and
//! stateless-ANE lanes by the NF4Tile640 shared arena (see
//! `compilation::apple_installation::derive_nf4_tile640_arena_abi`); the student
//! weights are produced by `compute_image::legacy_compute_image_compile::ternary_pipeline`.
//!
//! This module is intentionally dependency-free (no MLX / Metal / compute-core
//! types) so it compiles and unit-tests on every host, including Linux CI. The
//! actual gradient training (QAT / straight-through estimator backprop) runs on
//! the Mac via MLX; what lives here is the loss/agreement/acceptance math that
//! is identical on every platform and must be verified deterministically.

// ── Accelerate-optimised softmax (macOS) vs scalar fallback ──────────────

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// Vector exponential: y[i] = exp(x[i]).
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);

    /// Vector sum: result = sum(A[i]) over i in [0, N)
    fn vDSP_sve(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
    /// Vector-scalar add: C[i] = A[i] + B
    fn vDSP_vsadd(a: *const f32, a_stride: i32, b: *const f32, c: *mut f32, c_stride: i32, n: i32);

    /// Vector-scalar multiply: C[i] = A[i] * B
    fn vDSP_vsmul(a: *const f32, a_stride: i32, b: *const f32, c: *mut f32, c_stride: i32, n: i32);
}

/// Accelerated softmax body — one pre-allocated scratch buffer, in-place.
/// Computes `out[i] = exp((logits[i] - max_val) * inv_temp) / sum`.
#[cfg(target_os = "macos")]
fn softmax_impl(logits: &[f32], max_val: f32, inv_temp: f32, out: &mut [f32]) {
    let n = logits.len() as i32;
    unsafe {
        // Step 1: out[i] = logits[i] + (-max_val)  →  out[i] = logits[i] - max_val
        let neg_max = -max_val;
        vDSP_vsadd(logits.as_ptr(), 1, &neg_max, out.as_mut_ptr(), 1, n);

        // Step 2: out[i] *= inv_temp  →  out[i] = (logits[i] - max_val) / temperature
        vDSP_vsmul(out.as_ptr(), 1, &inv_temp, out.as_mut_ptr(), 1, n);

        // Step 3: out[i] = exp(out[i]) — vvexpf supports in-place
        vvexpf(out.as_mut_ptr(), out.as_ptr(), &n);

        // Step 4: sum = Σ out[i]
        let mut sum: f32 = 0.0;
        vDSP_sve(out.as_ptr(), 1, &mut sum, n);
        let inv_sum = (sum.max(1e-30_f32)).recip();

        // Step 5: out[i] *= inv_sum  →  out[i] = exp(...) / sum
        vDSP_vsmul(out.as_ptr(), 1, &inv_sum, out.as_mut_ptr(), 1, n);
    }
}

/// Scalar fallback softmax — same contract as the accelerated version.
#[cfg(not(target_os = "macos"))]
fn softmax_impl(logits: &[f32], max_val: f32, inv_temp: f32, out: &mut [f32]) {
    for i in 0..logits.len() {
        out[i] = ((logits[i] - max_val) * inv_temp).exp();
    }
    let sum: f32 = out.iter().copied().sum::<f32>().max(1e-30_f32);
    for e in out.iter_mut() {
        *e /= sum;
    }
}

/// Temperature-scaled softmax over one logit vector.
fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(1e-6);
    let inv_t = 1.0 / t;
    let max_val = logits.iter().cloned().fold(f32::MIN, f32::max);
    let mut result = vec![0.0_f32; logits.len()];
    softmax_impl(logits, max_val, inv_t, &mut result);
    result
}

/// Knowledge-distillation loss for one example: `T² · KL(p_teacher ‖ q_student)`
/// with both distributions softened by temperature `T` (Hinton et al.). Zero iff
/// the (temperature-scaled) distributions match. This is the signal the student
/// is trained to minimize, and the metric the loop logs per block.
pub fn kd_divergence(teacher_logits: &[f32], student_logits: &[f32], temperature: f32) -> f32 {
    assert_eq!(
        teacher_logits.len(),
        student_logits.len(),
        "logit length mismatch"
    );
    let p = softmax(teacher_logits, temperature);
    let q = softmax(student_logits, temperature);
    let kl: f32 = p
        .iter()
        .zip(&q)
        .map(|(&pi, &qi)| {
            if pi > 0.0 {
                pi * (pi / qi.max(1e-12)).ln()
            } else {
                0.0
            }
        })
        .sum();
    temperature * temperature * kl.max(0.0)
}

/// Mean KD loss over a batch of `[rows × vocab]` logits (row-major).
pub fn kd_divergence_batch(
    teacher: &[f32],
    student: &[f32],
    vocab: usize,
    temperature: f32,
) -> f32 {
    assert_eq!(teacher.len(), student.len(), "batch length mismatch");
    assert!(vocab > 0 && teacher.len() % vocab == 0, "ragged batch");
    let rows = teacher.len() / vocab;
    let mut acc = 0.0f32;
    for r in 0..rows {
        acc += kd_divergence(
            &teacher[r * vocab..(r + 1) * vocab],
            &student[r * vocab..(r + 1) * vocab],
            temperature,
        );
    }
    acc / rows as f32
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(
            (0, f32::MIN),
            |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
        )
        .0
}

/// Fraction of rows where the student's top-1 token matches the teacher's — the
/// coarse "does the student still pick the same answer" agreement metric.
pub fn top1_agreement(teacher: &[f32], student: &[f32], vocab: usize) -> f32 {
    assert!(vocab > 0 && teacher.len() % vocab == 0, "ragged batch");
    let rows = teacher.len() / vocab;
    if rows == 0 {
        return 1.0;
    }
    let mut agree = 0usize;
    for r in 0..rows {
        if argmax(&teacher[r * vocab..(r + 1) * vocab])
            == argmax(&student[r * vocab..(r + 1) * vocab])
        {
            agree += 1;
        }
    }
    agree as f32 / rows as f32
}

// ═══════════════════════════════════════════════════════════════════════════
// Joint primary + MTP distillation objective
//
// The megakernel carries NUM_MTP_HEADS=4 future-token predictors (MTP_HIDDEN
// 2048; logits written to `slot_logits + (head+1)·VOCAB`, read back via
// `read_slot_logits(slot, head)`). When the compile step co-distills the MTP
// heads with the shared torso, the objective is
//
//     L_total = L_primary + Σ_k λ_k · L_MTP_k
//
// with head 0 (primary) ALWAYS at weight 1.0 — λ only scales MTP heads.
//
// The struct keeps every component separate on purpose: the acceptance gates
// must threshold the PRIMARY term independently (a joint scalar can hide a
// primary regression behind MTP gains, which is exactly the failure mode that
// would silently destroy both base accuracy and spec-decode acceptance).
// ═══════════════════════════════════════════════════════════════════════════

/// Components of the joint primary+MTP KD objective.
#[derive(Debug, Clone, PartialEq)]
pub struct JointKd {
    /// `primary + Σ λ_k · mtp[k]` — the search/training objective.
    pub total: f32,
    /// Unweighted primary-head KD. Gate this one independently.
    pub primary: f32,
    /// Unweighted per-MTP-head KD (index 0 = the +2-token head).
    pub mtp: Vec<f32>,
    /// The λ schedule that produced `total` (echoed for receipts).
    pub lambdas: Vec<f32>,
}

/// Geometric MTP loss schedule: `λ_k = base · decay^(k−1)`, k = 1..=n.
/// Later heads predict further ahead, are intrinsically noisier, and matter
/// geometrically less to speculative throughput (a +k token only helps after
/// k−1 earlier acceptances), so their weight should decay to match.
pub fn geometric_lambdas(n_heads: usize, base: f32, decay: f32) -> Vec<f32> {
    (0..n_heads).map(|k| base * decay.powi(k as i32)).collect()
}

/// Joint KD over aligned per-head logit batches. `teacher_heads[h]` and
/// `student_heads[h]` are row-major `[rows × vocab]` for head `h`, with
/// **head 0 = primary** and heads 1.. = MTP (+2, +3, … predictors).
/// `lambdas.len()` must equal the number of MTP heads (`heads − 1`).
pub fn joint_kd_divergence(
    teacher_heads: &[Vec<f32>],
    student_heads: &[Vec<f32>],
    vocab: usize,
    temperature: f32,
    lambdas: &[f32],
) -> JointKd {
    assert!(!teacher_heads.is_empty(), "need at least the primary head");
    assert_eq!(
        teacher_heads.len(),
        student_heads.len(),
        "teacher/student head-count mismatch"
    );
    assert_eq!(
        lambdas.len(),
        teacher_heads.len() - 1,
        "one λ per MTP head (primary is always weight 1.0)"
    );

    let primary = kd_divergence_batch(&teacher_heads[0], &student_heads[0], vocab, temperature);
    let mtp: Vec<f32> = teacher_heads[1..]
        .iter()
        .zip(&student_heads[1..])
        .map(|(t, s)| kd_divergence_batch(t, s, vocab, temperature))
        .collect();
    let total = primary + mtp.iter().zip(lambdas).map(|(&kd, &l)| l * kd).sum::<f32>();

    JointKd {
        total,
        primary,
        mtp,
        lambdas: lambdas.to_vec(),
    }
}

/// Multi-codebook KL divergence for parallel codebook prediction.
///
/// Qwen3-TTS uses a 16-layer multi-codebook design: it predicts multiple
/// interleaved/parallel codebook indices simultaneously. Each codebook
/// produces its own logit distribution per step. This function computes
/// the KL divergence independently per codebook head, sums them, and
/// normalises by the number of heads, producing
/// `(1/H) · Σ_h KL(teacher_h || student_h)` so the magnitude is
/// independent of codebook count.
///
/// Args:
///   teacher_logits: &[Vec<f32>] — one logit vector per codebook head
///   student_logits: &[Vec<f32>] — one logit vector per codebook head
///   temperature: f32 — temperature for softening both distributions
///
/// Both slices must have the same length (same number of codebook heads).
pub fn multi_codebook_kd_divergence(
    teacher_logits: &[Vec<f32>],
    student_logits: &[Vec<f32>],
    temperature: f32,
) -> f32 {
    // Validate lengths
    assert_eq!(
        teacher_logits.len(),
        student_logits.len(),
        "teacher and student must have same number of codebook heads"
    );

    let n_heads = teacher_logits.len();
    if n_heads == 0 {
        return 0.0;
    }

    let mut total_kl = 0.0f32;
    // Compute per-codebook KL divergence and sum
    // This is equivalent to: (1/H) · Σ_h KL(p^teacher_h || q^student_h)
    for (t_logits, s_logits) in teacher_logits.iter().zip(student_logits.iter()) {
        total_kl += kd_divergence(t_logits, s_logits, temperature);
    }

    // Return mean across heads so the magnitude is independent of codebook count
    total_kl / n_heads as f32
}

/// Result of the per-block acceptance gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockAcceptance {
    /// Whether the block was accepted (relative error ≤ rel_tol).
    pub is_accepted: bool,
    /// L2 norm of teacher activations.
    pub teacher_rel_activation: f32,
    /// L2 norm of student activations.
    pub student_rel_activation: f32,
    /// Maximum absolute element-wise difference |t_i − s_i|.
    pub max_abs_diff: f32,
    /// Number of coordinates modified in this refinement step.
    pub modified_coords: u32,
}

/// Accept a distilled block iff the student's activations track the teacher's
/// within `rel_tol`. This is the activation-parity gate that decides whether a
/// ternary block is good enough to keep or must be re-distilled with a richer
/// config (more outliers, lower τ, AWQ). Ties into the loop's joint-acceptance.
pub fn block_accept(teacher_act: &[f32], student_act: &[f32], rel_tol: f32) -> BlockAcceptance {
    assert_eq!(
        teacher_act.len(),
        student_act.len(),
        "activation length mismatch"
    );
    let (mut se, mut den, mut max_diff) = (0.0f64, 0.0f64, 0.0f64);
    let (mut tnorm, mut snorm) = (0.0f64, 0.0f64);
    for (a, b) in teacher_act.iter().zip(student_act) {
        se += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
        tnorm += (*a as f64).powi(2);
        snorm += (*b as f64).powi(2);
        max_diff = max_diff.max((a - b).abs() as f64);
    }
    let rel_error = (se / den.max(1e-30)).sqrt() as f32;
    BlockAcceptance {
        is_accepted: rel_error <= rel_tol,
        teacher_rel_activation: tnorm.sqrt() as f32,
        student_rel_activation: snorm.sqrt() as f32,
        max_abs_diff: max_diff as f32,
        modified_coords: 0,
    }
}

// ── On-policy distillation refinement (§8.4) ──────────────────────────────

/// Temperature schedule for on-policy distillation refinement.
/// Starts at high temperature (soft exploration) and decays.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureSchedule {
    pub initial: f32,
    pub final_t: f32,
    pub decay_rate: f32, // multiplicative decay per round
}

impl TemperatureSchedule {
    pub fn default_r8() -> Self {
        Self {
            initial: 4.0,
            final_t: 1.0,
            decay_rate: 0.8,
        }
    }
    pub fn temperature(&self, round: usize) -> f32 {
        let t = self.initial * self.decay_rate.powi(round as i32);
        t.max(self.final_t)
    }
}

/// Bounded refinement configuration for the OPD loop (spec §8.4).
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Maximum refinement rounds (spec: 8)
    pub max_rounds: usize,
    /// Maximum fraction of coordinates selected per round (spec: 10%)
    pub max_selected_coords_fraction: f32,
    /// Maximum code choices per coordinate (spec: 3)
    pub max_code_choices_per_coord: usize,
    /// Minimum accepted improvement (policy-defined)
    pub min_improvement: f32,
    /// Plateau limit: stop if no improvement after this many rounds
    pub plateau_limit: usize,
    /// Loss weights per layer type
    pub logit_kl_weight: f32,
    pub activation_mse_weight: f32,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            max_rounds: 8,
            max_selected_coords_fraction: 0.10,
            max_code_choices_per_coord: 3,
            min_improvement: 1e-6,
            plateau_limit: 3,
            logit_kl_weight: 1.0,
            activation_mse_weight: 0.1,
        }
    }
}

/// Result of a full on-policy refinement pass.
#[derive(Debug, Clone)]
pub struct OnPolicyRefinementResult {
    pub rounds_completed: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub loss_per_round: Vec<f64>,
    pub coordinates_modified: usize,
    pub converged: bool,
    pub acceptance: Option<BlockAcceptance>,
    pub plateau_reached: bool,
}

/// Run bounded on-policy refinement for one matrix (spec §8.4).
///
/// - `initial_loss`: starting loss before any refinement
/// - `refine_step`: closure taking the round index, returning (new_loss, BlockAcceptance)
/// - `config`: refinement configuration
///
/// Returns the best refinement result after bounded rounds.
pub fn on_policy_refine(
    initial_loss: f64,
    mut refine_step: impl FnMut(usize) -> (f64, BlockAcceptance),
    config: &RefinementConfig,
) -> OnPolicyRefinementResult {
    let mut loss_per_round = Vec::with_capacity(config.max_rounds);
    loss_per_round.push(initial_loss);
    let mut best_loss = initial_loss;
    let mut best_acceptance = None;
    let mut plateau_count = 0;
    let mut coords_modified = 0;

    for round in 0..config.max_rounds {
        let (new_loss, acceptance) = refine_step(round);
        loss_per_round.push(new_loss);

        if acceptance.is_accepted {
            coords_modified += acceptance.modified_coords as usize;
        }

        let improvement = best_loss - new_loss;
        if improvement > config.min_improvement as f64 {
            best_loss = new_loss;
            best_acceptance = Some(acceptance);
            plateau_count = 0;
        } else {
            plateau_count += 1;
            if plateau_count >= config.plateau_limit {
                return OnPolicyRefinementResult {
                    rounds_completed: round + 1,
                    initial_loss,
                    final_loss: new_loss,
                    loss_per_round,
                    coordinates_modified: coords_modified,
                    converged: true,
                    acceptance: best_acceptance,
                    plateau_reached: true,
                };
            }
        }
    }

    OnPolicyRefinementResult {
        rounds_completed: config.max_rounds,
        initial_loss,
        final_loss: best_loss,
        loss_per_round,
        coordinates_modified: coords_modified,
        converged: false,
        acceptance: best_acceptance,
        plateau_reached: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_logits_have_zero_kd() {
        let t = [2.0f32, 1.0, 0.1, -0.5];
        assert!(kd_divergence(&t, &t, 1.0).abs() < 1e-5);
        assert!(kd_divergence(&t, &t, 4.0).abs() < 1e-5);
    }

    #[test]
    fn divergent_student_has_positive_kd() {
        let t = [2.0f32, 1.0, 0.1, -0.5];
        let s = [1.0f32, 1.0, 1.0, 1.0];
        assert!(kd_divergence(&t, &s, 1.0) > 0.0);
        assert!(kd_divergence(&t, &s, 4.0) > 0.0);
    }

    #[test]
    fn batch_kd_and_top1() {
        // 2 rows, vocab 3. row0 identical, row1 same argmax.
        let teacher = [3.0f32, 1.0, 0.0, /* row1 */ 0.5, 2.0, 1.0];
        let student = [3.0f32, 1.0, 0.0, /* row1 */ 0.4, 1.8, 1.1];
        assert!(kd_divergence_batch(&teacher, &student, 3, 2.0) >= 0.0);
        assert_eq!(top1_agreement(&teacher, &student, 3), 1.0); // both rows agree
    }

    #[test]
    fn acceptance_gate_thresholds() {
        let teacher = [2.0f32, 1.0, 0.1, -0.5];
        let close = [1.9f32, 1.1, 0.0, -0.4];
        let far = [1.0f32, 1.0, 1.0, 1.0];
        assert!(block_accept(&teacher, &close, 0.1).is_accepted);
        assert!(!block_accept(&teacher, &far, 0.1).is_accepted);
        // a perfect student is always accepted with ~0 error.
        let perfect = block_accept(&teacher, &teacher, 1e-6);
        assert!(perfect.is_accepted);
        assert!(perfect.max_abs_diff < 1e-6);
    }

    // ── joint primary + MTP objective ──────────────────────────────────

    #[test]
    fn geometric_lambda_schedule() {
        let l = geometric_lambdas(4, 0.3, 0.5);
        assert_eq!(l.len(), 4);
        assert!((l[0] - 0.3).abs() < 1e-7);
        assert!((l[1] - 0.15).abs() < 1e-7);
        assert!((l[2] - 0.075).abs() < 1e-7);
        assert!((l[3] - 0.0375).abs() < 1e-7);
    }

    #[test]
    fn joint_kd_perfect_student_is_zero_everywhere() {
        let heads: Vec<Vec<f32>> = (0..3)
            .map(|h| {
                vec![
                    2.0 + h as f32,
                    1.0,
                    0.0,
                    -1.0,
                    /* row1 */ 0.5,
                    2.0,
                    1.0,
                    0.0,
                ]
            })
            .collect();
        let j = joint_kd_divergence(&heads, &heads.clone(), 4, 2.0, &[0.3, 0.15]);
        assert!(j.total.abs() < 1e-6);
        assert!(j.primary.abs() < 1e-6);
        assert!(j.mtp.iter().all(|&k| k.abs() < 1e-6));
    }

    #[test]
    fn mtp_degradation_is_lambda_scaled_and_primary_isolated() {
        let vocab = 4;
        let base: Vec<f32> = vec![3.0, 1.0, 0.0, -1.0];
        let teacher: Vec<Vec<f32>> = vec![base.clone(), base.clone(), base.clone()];
        // Degrade ONLY MTP head 2 (index 2 overall): inverted logits.
        let mut student = teacher.clone();
        student[2] = base.iter().map(|&x| -x).collect();

        let lambdas = [0.3f32, 0.15];
        let j = joint_kd_divergence(&teacher, &student, vocab, 2.0, &lambdas);
        assert!(j.primary.abs() < 1e-6, "primary must be untouched");
        assert!(j.mtp[0].abs() < 1e-6);
        assert!(j.mtp[1] > 0.05, "degraded head must register: {}", j.mtp[1]);
        // total = primary + λ₂·mtp[1] exactly.
        let expect = j.primary + 0.15 * j.mtp[1];
        assert!((j.total - expect).abs() < 1e-6);
    }

    #[test]
    fn primary_regression_shows_regardless_of_lambdas() {
        // The gate rationale: a primary regression must be visible in
        // `primary` no matter how small the λs are — the joint total can
        // never be the only thing a gate looks at.
        let vocab = 4;
        let base: Vec<f32> = vec![3.0, 1.0, 0.0, -1.0];
        let teacher: Vec<Vec<f32>> = vec![base.clone(), base.clone()];
        let mut student = teacher.clone();
        student[0] = base.iter().map(|&x| -x).collect(); // wreck the primary
        let j = joint_kd_divergence(&teacher, &student, vocab, 2.0, &[1e-6]);
        assert!(
            j.primary > 0.05,
            "primary regression invisible: {}",
            j.primary
        );
        assert!((j.total - j.primary - 1e-6 * j.mtp[0]).abs() < 1e-6);
    }

    #[test]
    fn test_multi_codebook_kd_zero_when_equal() {
        // Two codebook heads with identical logits -> KL = 0
        let head1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let head2 = vec![4.0f32, 3.0, 2.0, 1.0];
        let teacher = vec![head1.clone(), head2.clone()];
        let student = vec![head1, head2];
        let kl = multi_codebook_kd_divergence(&teacher, &student, 1.0);
        assert!(kl.abs() < 1e-6, "identical logits should give zero KL");
    }

    #[test]
    fn test_multi_codebook_kd_divergent_heads() {
        // Two heads, one matches one diverges -> KL > 0 but not as large as both diverging
        let matching = vec![1.0f32, 2.0, 3.0, 4.0];
        let divergent = vec![100.0f32, 0.0, 0.0, 0.0];
        let teacher = vec![matching.clone(), divergent.clone()];
        // Only head 1 (divergent) diverges; the matching head is identical
        let student = vec![matching, vec![0.0f32, 100.0, 0.0, 0.0]];
        let kl = multi_codebook_kd_divergence(&teacher, &student, 1.0);
        assert!(kl > 0.0, "divergent head should yield positive KL");
        // Both divergent should be larger than one divergent
        let both_divergent = vec![divergent.clone(), divergent];
        let both_student = vec![vec![0.0f32, 100.0, 0.0, 0.0], vec![0.0f32, 0.0, 100.0, 0.0]];
        let kl_both = multi_codebook_kd_divergence(&both_divergent, &both_student, 1.0);
        assert!(kl_both > kl, "both heads diverging should give larger KL");
    }

    // ── on-policy refinement (§8.4) ────────────────────────────────────

    #[test]
    fn temperature_schedule_decays() {
        let sched = TemperatureSchedule::default_r8();
        assert!(sched.temperature(0) > sched.temperature(1));
        assert!(sched.temperature(7) >= 1.0);
    }

    #[test]
    fn refinement_plateau_early_exit() {
        let mut call_count = 0;
        let result = on_policy_refine(
            1.0,
            |_round| {
                call_count += 1;
                (
                    1.0,
                    BlockAcceptance {
                        is_accepted: true,
                        teacher_rel_activation: 0.0,
                        student_rel_activation: 0.0,
                        max_abs_diff: 0.0,
                        modified_coords: 0,
                    },
                )
            },
            &RefinementConfig {
                plateau_limit: 2,
                ..Default::default()
            },
        );
        assert!(result.plateau_reached);
        assert!(result.rounds_completed < 8);
    }

    #[test]
    fn refinement_improvement_tracked() {
        let result = on_policy_refine(
            1.0,
            |round| {
                // Simulate improvement: loss decreases each round
                let loss = 1.0 / (1 + round) as f64;
                (
                    loss,
                    BlockAcceptance {
                        is_accepted: true,
                        teacher_rel_activation: 0.0,
                        student_rel_activation: 0.0,
                        max_abs_diff: 0.0,
                        modified_coords: 10,
                    },
                )
            },
            &RefinementConfig::default(),
        );
        assert!(result.final_loss < result.initial_loss);
        assert!(result.coordinates_modified > 0);
    }
}
