use crate::inference_profile::{
    backend::{BackendKind, EvidenceStatus, PlacementClaim},
    evidence::{PhaseEvidenceReceipt, TimestampMs},
    ids::{MachineProfileDigest, ModelProfileDigest, PhaseId, ProfileId, ReceiptId},
    phase::PhaseKind,
};

#[derive(Debug, Clone)]
pub struct CoreAiOpaqueProbe {
    pub backend: BackendKind,
}

impl Default for CoreAiOpaqueProbe {
    fn default() -> Self {
        Self {
            backend: BackendKind::CoreML,
        }
    }
}

impl CoreAiOpaqueProbe {
    pub fn classify(&self) -> Vec<PhaseEvidenceReceipt> {
        let machine = MachineProfileDigest::from_hex("a".repeat(64)).unwrap();
        let model = ModelProfileDigest::from_hex("b".repeat(64)).unwrap();
        vec![
            receipt(
                self.backend,
                PhaseKind::KvView,
                EvidenceStatus::Unqualified,
                machine.clone(),
                model.clone(),
                "opaque",
            ),
            receipt(
                self.backend,
                PhaseKind::Prefill,
                EvidenceStatus::Unqualified,
                machine,
                model,
                "backend_managed",
            ),
        ]
    }

    pub fn placement_claim(&self) -> PlacementClaim {
        PlacementClaim::BackendManaged
    }
}

fn receipt(
    backend: BackendKind,
    phase_kind: PhaseKind,
    status: EvidenceStatus,
    machine: MachineProfileDigest,
    model: ModelProfileDigest,
    note: &str,
) -> PhaseEvidenceReceipt {
    let now = TimestampMs::now();
    PhaseEvidenceReceipt {
        receipt_id: ReceiptId::new_random(),
        phase_id: PhaseId::new(phase_kind.ordinal(), 0),
        phase_kind,
        profile_id: ProfileId::new_random(),
        backend,
        machine_profile_digest: machine,
        model_profile_digest: model,
        input_digest: "coreai-probe".into(),
        output_digest: None,
        started_at: now,
        finished_at: now,
        status,
        metrics: Default::default(),
        artifacts: vec![],
        gate_results: vec![],
        failure: None,
        notes: Some(note.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coreai_probe_classifies_backend_managed() {
        let probe = CoreAiOpaqueProbe::default();
        assert_eq!(probe.placement_claim(), PlacementClaim::BackendManaged);
        assert_eq!(probe.classify().len(), 2);
    }
}
