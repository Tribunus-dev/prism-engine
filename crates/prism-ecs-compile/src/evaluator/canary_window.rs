//! Bounded active-layer working set for canary evaluation.
//!
//! This module owns the [`CanaryWindow`] type — a small, recycle-aware
//! container that admits exactly one reference tensor and one candidate
//! reconstruction at a time. It exists to bound the memory footprint of
//! the evaluator's canary pass: the window is explicitly recycled before
//! the next tensor is admitted, so per-tensor materialization does not
//! accumulate across the search. The window is canonical: it has no
//! hardware handles, no FFI, no `unsafe`, and no process-local state.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Failure modes for [`CanaryWindow`] admission.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CanaryWindowError {
    /// The reference tensor is empty — canary evaluation has no signal to score.
    #[error("canary reference tensor is empty")]
    EmptyReference,
    /// The reference tensor exceeds the active window's element budget.
    #[error("canary tensor exceeds {max_elements} element active window")]
    ExceedsBudget { max_elements: usize },
}

/// Bounded active-layer working set. Only the reference tensor and one
/// candidate representation are resident while a canary is evaluated; the
/// window is explicitly recycled before the next tensor is admitted.
#[derive(Debug, Default)]
pub struct CanaryWindow {
    reference: Vec<f32>,
    candidate: Vec<f32>,
    generation: u64,
    max_elements: usize,
}

impl CanaryWindow {
    /// Build a canary window with the given per-tensor element budget.
    pub fn new(max_elements: usize) -> Self {
        Self {
            max_elements,
            ..Self::default()
        }
    }

    /// Admit a single reference tensor into the window. The candidate slot
    /// is resized to match the reference length and zeroed so a fresh
    /// reconstruction can be scored against it.
    pub fn load(&mut self, reference: &[f32]) -> Result<u64, CanaryWindowError> {
        if reference.is_empty() {
            return Err(CanaryWindowError::EmptyReference);
        }
        if reference.len() > self.max_elements {
            return Err(CanaryWindowError::ExceedsBudget {
                max_elements: self.max_elements,
            });
        }
        self.reference.clear();
        self.reference.extend_from_slice(reference);
        self.candidate.resize(reference.len(), 0.0);
        self.generation = self.generation.wrapping_add(1);
        Ok(self.generation)
    }

    /// Borrow the reference slot — callers read it but do not mutate it.
    pub fn reference(&self) -> &[f32] {
        &self.reference
    }

    /// Borrow the candidate slot — the evaluator writes its reconstruction
    /// here for scoring.
    pub fn candidate_mut(&mut self) -> &mut [f32] {
        &mut self.candidate
    }

    /// The current generation counter. Increments on every successful
    /// `load` and every `recycle`; useful for tracing and for cache key
    /// disambiguation in callers that compose the window with an external
    /// reference cache.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The per-tensor element budget. Window rejects references that
    /// exceed this length.
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    /// Recycle the window. Both slots are cleared and the generation
    /// counter advances so a new canary can be admitted.
    pub fn recycle(&mut self) {
        self.reference.clear();
        self.candidate.clear();
        self.generation = self.generation.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canary_window_loads_reference_and_resizes_candidate() {
        let mut window = CanaryWindow::new(8);
        assert_eq!(window.generation(), 0);
        let g1 = window.load(&[1.0, 2.0, 3.0, 4.0]).expect("fits budget");
        assert_eq!(window.reference(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(window.candidate_mut().len(), 4);
        assert!(window.candidate_mut().iter().all(|&v| v == 0.0));
        assert_eq!(window.generation(), g1);
    }

    #[test]
    fn canary_window_rejects_empty_and_oversize() {
        let mut window = CanaryWindow::new(4);
        assert_eq!(window.load(&[]), Err(CanaryWindowError::EmptyReference));
        let big = vec![0.0f32; 8];
        assert_eq!(
            window.load(&big),
            Err(CanaryWindowError::ExceedsBudget { max_elements: 4 })
        );
    }

    #[test]
    fn canary_window_recycle_advances_generation_and_clears() {
        let mut window = CanaryWindow::new(4);
        let g0 = window.load(&[1.0, 2.0]).expect("fits");
        window.recycle();
        assert!(window.reference().is_empty());
        assert!(window.candidate_mut().is_empty());
        assert!(window.generation() > g0);
    }
}
