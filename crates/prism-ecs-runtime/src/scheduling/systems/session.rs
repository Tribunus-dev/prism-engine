//! Session system (constitutional home, runtime half).
//!
//! Per the inventory v2.1 step 32, the engine's `prism_session.rs`
//! is decomposed by authority. The runtime-scheduling system half
//! (this file) owns the per-session decode loop and the state-
//! transition logic. The aggregate PrismSession (step 14's
//! decomposition) was the state half; this is the system half.
//!
//! The actual `PrismSession` aggregate stays in the engine file as
//! the legacy duplicate; step 58 deletes it. When the engine
//! callers migrate, they invoke the constitutional systems here
//! instead.

use crate::scheduling::state::session::{DecodeScheduler, GenerationState};

/// Per-session system: advance the session's state machine.
pub fn advance_session_state(
    state: &mut GenerationState,
    scheduler: &mut DecodeScheduler,
) {
    match state {
        GenerationState::PromptProcessing => {
            *state = GenerationState::Decoding;
        }
        GenerationState::Decoding => {
            if scheduler.terminated {
                *state = GenerationState::Completed;
            }
            scheduler.advance();
        }
        GenerationState::Completed | GenerationState::Failed(_) => {
            // Terminal states — no further transitions.
        }
    }
}

/// Placeholder aggregate: per-session scheduling record. The full
/// `PrismSession` aggregate lives in the engine file as the legacy
/// duplicate; this is a constitutional-side skeleton for systems
/// to attach to.
#[derive(Debug, Clone)]
pub struct PrismSessionSystem {
    pub session_id: String,
    pub image_digest: String,
    pub scheduler: DecodeScheduler,
    pub generation_state: GenerationState,
}

impl PrismSessionSystem {
    pub fn new(session_id: String, image_digest: String, max_new_tokens: u32) -> Self {
        Self {
            session_id,
            image_digest,
            scheduler: DecodeScheduler::new(max_new_tokens),
            generation_state: GenerationState::PromptProcessing,
        }
    }

    /// Advance the session's state machine.
    pub fn step(&mut self) {
        advance_session_state(&mut self.generation_state, &mut self.scheduler);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_starts_in_prompt_processing() {
        let s = PrismSessionSystem::new("s1".into(), "img".into(), 10);
        assert_eq!(s.generation_state, GenerationState::PromptProcessing);
        assert_eq!(s.scheduler.epoch, 0);
    }

    #[test]
    fn first_step_transitions_to_decoding() {
        // Architectural invariant: the first step transitions the
        // session from PromptProcessing to Decoding. The scheduler
        // does NOT advance yet (the first step is the prompt-
        // processing → decoding transition itself).
        let mut s = PrismSessionSystem::new("s1".into(), "img".into(), 10);
        s.step();
        assert_eq!(s.generation_state, GenerationState::Decoding);
        assert_eq!(s.scheduler.epoch, 0);
    }

    #[test]
    fn session_terminates_at_completion() {
        // Architectural invariant: when the scheduler is terminated
        // (i.e. epoch > max_new_tokens), the next step transitions
        // the session to Completed. The engine's advance_state
        // logic: Decoding branch checks terminated BEFORE advancing;
        // advancement happens AFTER the check.
        let mut s = PrismSessionSystem::new("s1".into(), "img".into(), 2);
        s.step(); // → Decoding
        s.step(); // Decoding, not terminated, advance → epoch=1
        s.step(); // Decoding, not terminated, advance → epoch=2
        s.step(); // Decoding, not terminated, advance → epoch=3 (3>2, terminated)
        // Next step: Decoding, terminated=true → Completed
        s.step();
        assert_eq!(s.generation_state, GenerationState::Completed);
    }

    #[test]
    fn terminal_state_does_not_advance() {
        // Architectural invariant: a session in Completed or Failed
        // state does not advance further on subsequent step() calls.
        let mut s = PrismSessionSystem::new("s1".into(), "img".into(), 2);
        s.scheduler.terminate();
        s.generation_state = GenerationState::Completed;
        let before = s.scheduler.epoch;
        s.step();
        assert_eq!(s.scheduler.epoch, before);
        assert_eq!(s.generation_state, GenerationState::Completed);
    }
}
