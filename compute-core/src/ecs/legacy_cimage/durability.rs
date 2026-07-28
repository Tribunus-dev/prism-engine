use prism_ecs_constitutional::canonical::identity::GenerationId;

/// Crash recovery — ensure content-store integrity after interruption.
/// Plan Section 2: "Immutable history: Existing generations are never
/// overwritten. Changes produce new generations."
pub struct DurabilityManager {
    generation_log: Vec<GenerationId>,
}

impl DurabilityManager {
    pub fn new() -> Self {
        Self {
            generation_log: Vec::new(),
        }
    }

    /// Record a promotion for replay on recovery.
    pub fn record_promotion(&mut self, generation_id: GenerationId) {
        self.generation_log.push(generation_id);
    }

    /// Verify no corruption — ensure all promoted generations exist.
    pub fn verify_integrity(&self, available_generations: &[GenerationId]) -> Result<(), String> {
        for promoted in &self.generation_log {
            if !available_generations.contains(promoted) {
                return Err(format!(
                    "missing generation {:?} — corruption detected",
                    promoted
                ));
            }
        }
        Ok(())
    }

    /// Crash recovery — replay promotion log to restore latest valid state.
    pub fn recover(&self, generations: &[GenerationId]) -> Option<&GenerationId> {
        // Find the latest generation still present
        self.generation_log
            .iter()
            .rev()
            .find(|id| generations.contains(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrity_pass() {
        let mut mgr = DurabilityManager::new();
        let g1 = GenerationId("gen1".into());
        mgr.record_promotion(g1.clone());
        assert!(mgr.verify_integrity(&[g1]).is_ok());
    }

    #[test]
    fn test_integrity_fail() {
        let mut mgr = DurabilityManager::new();
        mgr.record_promotion(GenerationId("gen1".into()));
        assert!(mgr.verify_integrity(&[]).is_err());
    }

    #[test]
    fn test_crash_recovery() {
        let mut mgr = DurabilityManager::new();
        mgr.record_promotion(GenerationId("gen1".into()));
        mgr.record_promotion(GenerationId("gen2".into()));
        // gen2 was lost in crash — recover to gen1
        let recovered = mgr.recover(&[GenerationId("gen1".into())]);
        assert_eq!(recovered.map(|id| &id.0[..]), Some("gen1"));
    }
}
