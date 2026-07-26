//! Resource governor — thermal-aware admission envelope adjustment.

use crate::ecs::registry::types::*;
use std::sync::Arc;

pub struct ResourceGovernor {
    pub contract: Arc<RuntimeContract>,
}

impl ResourceGovernor {
    pub fn new(contract: Arc<RuntimeContract>) -> Self {
        Self { contract }
    }

    /// Calculate the live admission envelope based on current system state.
    ///
    /// The RuntimeContract establishes the absolute ceiling. This function
    /// narrows it based on thermal state and power source. It never expands
    /// beyond the contract's limits.
    pub fn calculate_live_admission_envelope(&self) -> LiveAdmissionEnvelope {
        let sys_thermal = ThermalState::current();
        let power_source = PowerSource::current();

        let mut envelope = LiveAdmissionEnvelope::from_contract(&self.contract);

        match sys_thermal {
            ThermalState::Nominal => {}
            ThermalState::Fair => {
                envelope.active_batch_slots =
                    (self.contract.resource_budget.max_batch_slots * 3) / 4;
            }
            ThermalState::Serious => {
                envelope.active_batch_slots = self.contract.resource_budget.max_batch_slots / 2;
                envelope.allow_background_optimization = false;
                envelope.disable_speculative_decode = true;
            }
            ThermalState::Critical => {
                envelope.active_batch_slots = 1;
                envelope.allow_background_optimization = false;
                envelope.disable_speculative_decode = true;
                envelope.energy_mode = EnergyOptimizationProfile::MinimizePackageSustainedPower;
            }
        }

        if let PowerSource::Battery = power_source {
            envelope.active_batch_slots = std::cmp::min(envelope.active_batch_slots, 4);
            envelope.allow_background_optimization = false;
        }

        envelope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_contract() -> RuntimeContract {
        RuntimeContract {
            deployment_id: DeploymentId(Digest256::compute(b"t")),
            model_digest: Digest256::compute(b""),
            artifact_digest: Digest256::compute(b""),
            manifest_digest: Digest256::compute(b""),
            policy_digest: Digest256::compute(b""),
            hardware_digest: Digest256::compute(b""),
            effective_capabilities: CapabilitySet::empty(),
            resource_budget: ResourceBudget {
                max_context_tokens: 8192,
                max_batch_slots: 16,
                max_kv_bytes: 1_000_000_000,
                max_unified_bytes: 6_000_000_000,
                max_kv_blocks_per_sequence: 1280,
            },
            backend_plan: BackendPlan {
                primary: BackendTarget::Metal,
                fallbacks: vec![],
                validated_hardware: false,
            },
            admission_precision: AdmissionPrecision {
                base: PrecisionClass::NF4Tile640,
                allowed_escalations: vec![],
                allow_dynamic: false,
            },
            tool_authority: ToolAuthority::new(false),
            optimization_authority: OptimizationAuthority::new(true),
            issued_at: LogicalTimestamp(0),
            expires_at: None,
            registry_signature: RegistrySignature {
                bytes: vec![],
                key_fingerprint: KeyFingerprint(Digest256::compute(b"")),
            },
        }
    }

    #[test]
    fn test_nominal_batch_capacity() {
        let contract = Arc::new(test_contract());
        let governor = ResourceGovernor::new(contract);
        let envelope = governor.calculate_live_admission_envelope();
        assert!(envelope.active_batch_slots <= 16);
        assert!(envelope.active_batch_slots > 0);
    }

    #[test]
    fn test_governor_does_not_exceed_contract() {
        let contract = Arc::new(test_contract());
        let governor = ResourceGovernor::new(contract);
        let envelope = governor.calculate_live_admission_envelope();
        assert!(envelope.active_batch_slots <= 16);
    }
}
