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
}
