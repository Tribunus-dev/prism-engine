//! distill_core.rs — numeric core for the NF4-teacher → ternary-student loop.
//!
//! Pure, std-only metrics the block-by-block distillation loop
//! (`server::distill_worker`) calls to (a) measure how well the ternary student
//! reproduces the NF4 teacher and (b) decide whether a distilled block is
//! accepted. The teacher weights are made bindable to both the Metal and
//! stateless-ANE lanes by the NF4Tile640 shared arena (see
//! `compilation::apple_installation::derive_nf4_tile640_arena_abi`); the student
//! weights are produced by `compute_image::compile::ternary_pipeline`.
//!
//! This module is intentionally dependency-free (no MLX / Metal / compute-core
//! types) so it compiles and unit-tests on every host, including Linux CI. The
//! actual gradient training (QAT / straight-through estimator backprop) runs on
//! the Mac via MLX; what lives here is the loss/agreement/acceptance math that
//! is identical on every platform and must be verified deterministically.

/// Temperature-scaled softmax over one logit vector.
fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(1e-6);
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = logits.iter().map(|&x| ((x - m) / t).exp()).collect();
    let s: f32 = ex.iter().sum::<f32>().max(1e-30);
    ex.iter().map(|&e| e / s).collect()
}

/// Knowledge-distillation loss for one example: `T² · KL(p_teacher ‖ q_student)`
/// with both distributions softened by temperature `T` (Hinton et al.). Zero iff
/// the (temperature-scaled) distributions match. This is the signal the student
/// is trained to minimize, and the metric the loop logs per block.
pub fn kd_divergence(teacher_logits: &[f32], student_logits: &[f32], temperature: f32) -> f32 {
    assert_eq!(teacher_logits.len(), student_logits.len(), "logit length mismatch");
    let p = softmax(teacher_logits, temperature);
    let q = softmax(student_logits, temperature);
    let kl: f32 = p
        .iter()
        .zip(&q)
        .map(|(&pi, &qi)| if pi > 0.0 { pi * (pi / qi.max(1e-12)).ln() } else { 0.0 })
        .sum();
    temperature * temperature * kl.max(0.0)
}

/// Mean KD loss over a batch of `[rows × vocab]` logits (row-major).
pub fn kd_divergence_batch(teacher: &[f32], student: &[f32], vocab: usize, temperature: f32) -> f32 {
    assert_eq!(teacher.len(), student.len(), "batch length mismatch");
    assert!(vocab > 0 && teacher.len() % vocab == 0, "ragged batch");
    let rows = teacher.len() / vocab;
    let mut acc = 0.0f32;
    for r in 0..rows {
        acc += kd_divergence(&teacher[r * vocab..(r + 1) * vocab], &student[r * vocab..(r + 1) * vocab], temperature);
    }
    acc / rows as f32
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

/// Fraction of rows where the student's top-1 token matches the teacher's — the
/// coarse "does the student still pick the same answer" agreement metric.
pub fn top1_agreement(teacher: &[f32], student: &[f32], vocab: usize) -> f32 {
    assert!(vocab > 0 && teacher.len() % vocab == 0, "ragged batch");
    let rows = teacher.len() / vocab;
    if rows == 0 { return 1.0; }
    let mut agree = 0usize;
    for r in 0..rows {
        if argmax(&teacher[r * vocab..(r + 1) * vocab]) == argmax(&student[r * vocab..(r + 1) * vocab]) {
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
    let total = primary
        + mtp
            .iter()
            .zip(lambdas)
            .map(|(&kd, &l)| l * kd)
            .sum::<f32>();

    JointKd {
        total,
        primary,
        mtp,
        lambdas: lambdas.to_vec(),
    }
}

/// Result of the per-block acceptance gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockAcceptance {
    /// Relative L2 activation error ‖teacher − student‖ / ‖teacher‖.
    pub rel_error: f32,
    pub accepted: bool,
}

/// Accept a distilled block iff the student's activations track the teacher's
/// within `rel_tol`. This is the activation-parity gate that decides whether a
/// ternary block is good enough to keep or must be re-distilled with a richer
/// config (more outliers, lower τ, AWQ). Ties into the loop's joint-acceptance.
pub fn block_accept(teacher_act: &[f32], student_act: &[f32], rel_tol: f32) -> BlockAcceptance {
    assert_eq!(teacher_act.len(), student_act.len(), "activation length mismatch");
    let (mut se, mut den) = (0.0f64, 0.0f64);
    for (a, b) in teacher_act.iter().zip(student_act) {
        se += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    let rel_error = (se / den.max(1e-30)).sqrt() as f32;
    BlockAcceptance { rel_error, accepted: rel_error <= rel_tol }
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
        assert!(block_accept(&teacher, &close, 0.1).accepted);
        assert!(!block_accept(&teacher, &far, 0.1).accepted);
        // a perfect student is always accepted with ~0 error.
        assert!(block_accept(&teacher, &teacher, 1e-6).rel_error < 1e-6);
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
            .map(|h| vec![2.0 + h as f32, 1.0, 0.0, -1.0, /* row1 */ 0.5, 2.0, 1.0, 0.0])
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
        assert!(j.primary > 0.05, "primary regression invisible: {}", j.primary);
        assert!((j.total - j.primary - 1e-6 * j.mtp[0]).abs() < 1e-6);
    }
}
