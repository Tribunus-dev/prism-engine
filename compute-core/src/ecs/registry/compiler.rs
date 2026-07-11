//! Deployment compiler — verifies artifact identity, computes capability
//! intersection, derives resource budgets, and produces a signed RuntimeContract.

#[cfg(feature = "legacy_mutations")]
use crate::ecs::registry::types::*;
#[cfg(feature = "legacy_mutations")]
use crate::ecs::registry::TrustStore;

#[cfg(feature = "legacy_mutations")]
pub struct DeploymentCompiler;

#[cfg(feature = "legacy_mutations")]
impl DeploymentCompiler {
    pub fn compile(
        manifest: &ComputeImageManifest,
        policy: &CapabilityPolicy,
        hardware: &HardwareProfile,
        trust_store: &TrustStore,
        manifest_bytes: &[u8],
    ) -> Result<RuntimeContract, DeploymentError> {
        // 1. Verify manifest signature via trust store
        trust_store.verify_manifest(manifest, manifest_bytes)?;

        // 2. Compute capability intersection using explicit separation.
        // Required capabilities: reject if missing from allowed OR explicitly denied.
        let required_missing = (manifest.required_capabilities & !policy.allowed_capabilities)
            | (manifest.required_capabilities & policy.denied_capabilities);
        if !required_missing.is_empty() {
            return Err(DeploymentError::RequiredCapabilityDenied(required_missing));
        }

        // Optional capabilities: only what policy allows AND doesn't deny.
        let optional_granted = manifest.optional_capabilities
            & policy.allowed_capabilities
            & !policy.denied_capabilities;
        let effective_capabilities = manifest.required_capabilities | optional_granted;

        // 3. Derive resource boundaries from hardware + policy.
        let resource_budget = ResourceBudget {
            max_context_tokens: std::cmp::min(
                policy.maximum_context_tokens,
                hardware.max_supported_tokens,
            ),
            max_batch_slots: std::cmp::min(
                policy.maximum_batch_slots,
                hardware.max_supported_slots,
            ),
            max_kv_bytes: std::cmp::min(
                policy.maximum_kv_bytes,
                hardware.max_kv_allocation_ceiling,
            ),
            max_unified_bytes: std::cmp::min(
                policy.maximum_unified_memory_bytes,
                hardware.available_memory_bytes,
            ),
            max_kv_blocks_per_sequence: manifest.kv_cache_contract.max_blocks,
        };

        // 4. Backend and precision plan.
        let policy_backends: std::collections::HashSet<BackendTarget> =
            policy.allowed_backends.clone();
        let backend_plan =
            BackendPlan::select(&manifest.allowed_backends, &policy_backends, hardware)?;
        let precision_policy = PrecisionPolicy::compile(
            &manifest.required_precision_classes,
            &policy.allowed_precision_modes,
        )?;

        // 5. Build dehydrated digest inputs for deployment identity.
        let input = DeploymentDigestInput {
            manifest_digest: manifest.manifest_digest(),
            artifact_digest: manifest.artifact_digest,
            policy_digest: policy.digest(),
            hardware_digest: hardware.digest(),
            backend_plan_digest: Digest256::compute(
                &serde_json::to_vec(&backend_plan).unwrap_or_default(),
            ),
            resource_budget_digest: resource_budget.digest(),
            precision_policy_digest: Digest256::compute(
                &serde_json::to_vec(&precision_policy).unwrap_or_default(),
            ),
            tool_authority_digest: Digest256::compute(&[if policy.allow_tool_execution {
                1u8
            } else {
                0u8
            }]),
            optimization_authority_digest: Digest256::compute(&[
                if policy.allow_background_optimization {
                    1u8
                } else {
                    0u8
                },
            ]),
        };
        let deployment_digest = hash_deployment_contract(&input);
        let manifest_digest = manifest.manifest_digest();

        // 6. Sign the contract.
        let sig_bytes = PlatformSecureSigner::sign(deployment_digest.as_bytes())
            .map_err(|_| DeploymentError::InternalError("signing failed".into()))?;

        Ok(RuntimeContract {
            deployment_id: DeploymentId(deployment_digest),
            model_digest: manifest.model_digest,
            artifact_digest: manifest.artifact_digest,
            manifest_digest,
            policy_digest: policy.digest(),
            hardware_digest: hardware.digest(),
            effective_capabilities,
            resource_budget,
            backend_plan,
            precision_policy,
            tool_authority: ToolAuthority::new(policy.allow_tool_execution),
            optimization_authority: OptimizationAuthority::new(
                policy.allow_background_optimization,
            ),
            issued_at: LogicalTimestamp::now(),
            expires_at: None,
            registry_signature: RegistrySignature {
                bytes: sig_bytes,
                key_fingerprint: KeyFingerprint(Digest256::compute(&[0u8; 32])),
            },
        })
    }
}
