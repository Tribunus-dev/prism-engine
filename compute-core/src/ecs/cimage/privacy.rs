use crate::ecs::training_target::spec::PrivacyContract;

/// Privacy contract enforcement.
/// Plan Section 2: "Explicit privacy: Training corpora and engram artifacts
/// carry purpose, retention, disclosure, and deletion metadata."
pub struct PrivacyEnforcer {
    contracts: Vec<PrivacyContract>,
}

impl PrivacyEnforcer {
    pub fn new() -> Self {
        Self {
            contracts: Vec::new(),
        }
    }

    pub fn register(&mut self, contract: PrivacyContract) {
        self.contracts.push(contract);
    }

    pub fn check_purpose(&self, _artifact_id: &str, purpose: &str) -> bool {
        self.contracts
            .iter()
            .any(|c| c.purpose == purpose || c.assimilation_permitted)
    }

    pub fn can_delete(&self, _artifact_id: &str) -> bool {
        // Placeholder: return true for artifacts past retention
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_purpose_match() {
        let contract = PrivacyContract {
            purpose: "calibration".into(),
            retention: "30d".into(),
            disclosure_class: "internal".into(),
            assimilation_permitted: false,
        };
        let mut e = PrivacyEnforcer::new();
        e.register(contract);
        assert!(e.check_purpose("", "calibration"));
        assert!(!e.check_purpose("", "training"));
    }
}
