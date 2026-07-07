use crate::registry::types::*;
use std::sync::Arc;

pub struct OverlayManager;

impl OverlayManager {
    pub fn validate_overlay(
        overlay: &OptimizationOverlay,
        current_handle: &DeploymentHandle,
    ) -> Result<(), DeploymentError> {
        // 1. Structural invariance — overlay must target the exact base artifact
        if overlay.base_artifact_digest != current_handle.contract.artifact_digest {
            return Err(DeploymentError::InvalidOverlayTarget);
        }

        // 2. Optimization must be permitted by contract
        if !current_handle
            .contract
            .optimization_authority
            .is_permitted()
        {
            return Err(DeploymentError::OptimizationForbidden);
        }

        // 3. Cryptographic validation of overlay signature
        overlay.verify_signature()?;

        // 4. Mutation scope check — can only mutate declared mutable profile slots
        for mutation in &overlay.mutations {
            let slot_id = match mutation {
                OverlayMutation::QuantizationScale { slot, .. } => slot,
                OverlayMutation::OutlierRouting { .. } => continue,
                OverlayMutation::PrecisionVariant { .. } => continue,
            };
            if !current_handle
                .execution_image
                .manifest
                .mutable_profile_slots
                .iter()
                .any(|s| &s.slot_id == slot_id)
            {
                return Err(DeploymentError::IllegalWeightMutation(slot_id.clone()));
            }
        }

        Ok(())
    }

    pub fn apply_overlay(
        overlay: OptimizationOverlay,
        current_handle: Arc<DeploymentHandle>,
    ) -> Result<Arc<DeploymentHandle>, DeploymentError> {
        Self::validate_overlay(&overlay, &current_handle)?;

        // Build upgraded handle with new profile generation
        let upgraded = DeploymentHandle {
            deployment_id: current_handle.deployment_id.clone(),
            contract: current_handle.contract.clone(),
            execution_image: current_handle.execution_image.clone(),
            #[cfg(target_os = "macos")]
            executor: current_handle.executor.clone(),
            live_state: Arc::new(DeploymentState::new_from_generation(
                overlay.profile_generation,
            )),
        };

        Ok(Arc::new(upgraded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_contract() -> RuntimeContract {
        RuntimeContract {
            deployment_id: DeploymentId(Digest256::compute(b"")),
            model_digest: Digest256::compute(b""),
            artifact_digest: Digest256::compute(b""),
            manifest_digest: Digest256::compute(b""),
            policy_digest: Digest256::compute(b""),
            hardware_digest: Digest256::compute(b""),
            effective_capabilities: CapabilitySet::empty(),
            resource_budget: ResourceBudget {
                max_context_tokens: 20480,
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
            precision_policy: PrecisionPolicy {
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

    fn dummy_manifest() -> ComputeImageManifest {
        ComputeImageManifest {
            format_version: 1,
            model_digest: Digest256::compute(b""),
            artifact_digest: Digest256::compute(b""),
            compiler_digest: Digest256::compute(b""),
            provider: ProviderIdentity(String::new()),
            model_license: LicenseDescriptor(String::new()),
            required_capabilities: CapabilitySet::empty(),
            optional_capabilities: CapabilitySet::empty(),
            execution_graph: PhaseGraphDigest(Digest256::compute(b"")),
            allowed_backends: vec![BackendTarget::Metal],
            required_precision_classes: vec![],
            memory_requirements: MemoryRequirementSet {
                min_unified_memory_bytes: 0,
                kv_cache_reservation_bytes: 0,
            },
            kv_cache_contract: KvCacheContract {
                max_blocks: 0,
                tokens_per_block: 0,
            },
            network_contract: NetworkContract {
                enabled: false,
                allowed_domains: vec![],
            },
            tool_contract: ToolContract {
                enabled: false,
                allowed_tools: vec![],
            },
            mutable_profile_slots: vec![],
            artifact_signature: ArtifactSignature {
                bytes: vec![],
                algorithm: String::new(),
            },
        }
    }

    #[test]
    fn test_reject_optimization_not_permitted() {
        let mut contract = dummy_contract();
        contract.optimization_authority = OptimizationAuthority::new(false);
        let handle = Arc::new(DeploymentHandle {
            deployment_id: contract.deployment_id.clone(),
            contract: Arc::new(contract),
            execution_image: Arc::new(LoadedComputeImage {
                manifest: dummy_manifest(),
            }),
            executor: Arc::new(std::sync::Mutex::new(None)),
            live_state: Arc::new(DeploymentState::new()),
        });

        let overlay = OptimizationOverlay {
            base_artifact_digest: Digest256::compute(b""),
            base_manifest_digest: Digest256::compute(b"test"),
            profile_generation: 1,
            mutations: vec![],
            validation_receipt_digest: Digest256::compute(b""),
            optimizer_identity: OptimizerIdentity(Digest256::compute(b"opt")),
            overlay_signature: RegistrySignature {
                bytes: vec![1],
                key_fingerprint: KeyFingerprint(Digest256::compute(b"k")),
            },
        };

        assert!(OverlayManager::validate_overlay(&overlay, &handle).is_err());
    }
}
