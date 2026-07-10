use crate::quantization::contract::{
    BackendKind, RepresentationCapability, RuntimeRepresentationClass,
};
use std::collections::HashMap;

/// Registry of representation capabilities per backend.
#[derive(Debug)]
pub struct CapabilityRegistry {
    entries: HashMap<(RuntimeRepresentationClass, u16, BackendKind), RepresentationCapability>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a capability for a representation on a backend.
    pub fn register(&mut self, cap: RepresentationCapability) {
        let key = (cap.representation, cap.representation_version, cap.backend);
        self.entries.insert(key, cap);
    }

    /// Check if a representation is production-ready for a target backend.
    pub fn is_production_ready(
        &self,
        representation: RuntimeRepresentationClass,
        representation_version: u16,
        backend: BackendKind,
    ) -> bool {
        let key = (representation, representation_version, backend);
        self.entries
            .get(&key)
            .map_or(false, |cap| cap.production_ready)
    }

    /// Get the full capability entry if registered.
    pub fn get(
        &self,
        representation: RuntimeRepresentationClass,
        representation_version: u16,
        backend: BackendKind,
    ) -> Option<&RepresentationCapability> {
        let key = (representation, representation_version, backend);
        self.entries.get(&key)
    }

    /// Generate the ordered candidate ladder for a target (spec §12).
    /// Candidates are ordered by expected runtime cost (cheapest first).
    pub fn candidate_ladder(&self, backend: BackendKind) -> Vec<(RuntimeRepresentationClass, u16)> {
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, cap)| cap.backend == backend && cap.production_ready)
            .map(|(key, _)| (key.0, key.1))
            .collect();
        // Order by expected cost: Ternary(0) < NF4(1) < INT8(2) < RawF32(3)
        candidates.sort_by_key(|(rep, _)| match rep {
            RuntimeRepresentationClass::TernaryTile640Base => 0u8,
            RuntimeRepresentationClass::Nf4Tile640Base => 1,
            RuntimeRepresentationClass::Int8Tile640Base => 2,
            RuntimeRepresentationClass::RawF32 => 3,
        });
        candidates
    }

    /// Build the default V1 capability set for the Metal backend.
    /// This represents what's currently known to work.
    pub fn default_metal_v1() -> Self {
        let mut reg = Self::new();
        // Register Nf4Tile640Base as production-ready for Metal V1
        reg.register(RepresentationCapability {
            representation: RuntimeRepresentationClass::Nf4Tile640Base,
            representation_version: 1,
            backend: BackendKind::Metal,
            kernel_abi_digest: [0u8; 32],
            cpu_reference_ready: true,
            parser_ready: true,
            artifact_writer_ready: true,
            loader_ready: true,
            runtime_kernel_ready: true,
            nonzero_offset_test_passed: true,
            tail_mask_test_passed: true,
            mixed_format_test_passed: true,
            end_to_end_profile_test_passed: true,
            production_ready: true,
        });
        // Register Int8Tile640Base as production-ready for Metal V1
        reg.register(RepresentationCapability {
            representation: RuntimeRepresentationClass::Int8Tile640Base,
            representation_version: 1,
            backend: BackendKind::Metal,
            kernel_abi_digest: [0u8; 32],
            cpu_reference_ready: true,
            parser_ready: true,
            artifact_writer_ready: true,
            loader_ready: true,
            runtime_kernel_ready: true,
            nonzero_offset_test_passed: true,
            tail_mask_test_passed: true,
            mixed_format_test_passed: true,
            end_to_end_profile_test_passed: true,
            production_ready: true,
        });
        // Register RawF32 (always available - correctness fallback)
        reg.register(RepresentationCapability {
            representation: RuntimeRepresentationClass::RawF32,
            representation_version: 1,
            backend: BackendKind::Metal,
            kernel_abi_digest: [0u8; 32],
            cpu_reference_ready: true,
            parser_ready: true,
            artifact_writer_ready: true,
            loader_ready: true,
            runtime_kernel_ready: true,
            nonzero_offset_test_passed: true,
            tail_mask_test_passed: true,
            mixed_format_test_passed: true,
            end_to_end_profile_test_passed: true,
            production_ready: true,
        });
        // Ternary is NOT registered - packer exists but runtime kernel needs qualification.
        reg
    }

    /// Verify a candidate ladder entry meets all production requirements (spec §19).
    /// Rejects unsupported representations, missing runtime kernels, and incomplete conformance.
    pub fn verify_production_ladder(
        &self,
        ladder: &[(RuntimeRepresentationClass, u16)],
        backend: BackendKind,
    ) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for &(rep, version) in ladder {
            match self.get(rep, version, backend) {
                None => {
                    errors.push(format!(
                        "{:?} v{} not registered for {:?}",
                        rep, version, backend
                    ));
                }
                Some(cap) => {
                    if !cap.production_ready {
                        errors.push(format!(
                            "{:?} v{} not production-ready for {:?}",
                            rep, version, backend
                        ));
                    }
                    if !cap.cpu_reference_ready {
                        errors.push(format!("{:?} v{} missing CPU reference", rep, version));
                    }
                    if !cap.runtime_kernel_ready {
                        errors.push(format!("{:?} v{} missing runtime kernel", rep, version));
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::default_metal_v1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::contract::BackendKind;

    #[test]
    fn metal_v1_has_nf4_int8_rawf32() {
        let reg = CapabilityRegistry::default_metal_v1();
        assert!(reg.is_production_ready(
            RuntimeRepresentationClass::Nf4Tile640Base,
            1,
            BackendKind::Metal
        ));
        assert!(reg.is_production_ready(
            RuntimeRepresentationClass::Int8Tile640Base,
            1,
            BackendKind::Metal
        ));
        assert!(reg.is_production_ready(RuntimeRepresentationClass::RawF32, 1, BackendKind::Metal));
        assert!(!reg.is_production_ready(
            RuntimeRepresentationClass::TernaryTile640Base,
            1,
            BackendKind::Metal
        ));
    }

    #[test]
    fn candidate_ladder_order() {
        let reg = CapabilityRegistry::default_metal_v1();
        let ladder = reg.candidate_ladder(BackendKind::Metal);
        // Three candidates: NF4, INT8, RawF32 (ternary not production-ready)
        assert_eq!(ladder.len(), 3);
        // Order matches spec: cheaper formats first
        assert_eq!(ladder[0].0, RuntimeRepresentationClass::Nf4Tile640Base);
        assert_eq!(ladder[1].0, RuntimeRepresentationClass::Int8Tile640Base);
        assert_eq!(ladder[2].0, RuntimeRepresentationClass::RawF32);
    }

    #[test]
    fn ternary_not_in_ladder() {
        let reg = CapabilityRegistry::default_metal_v1();
        assert!(!reg.is_production_ready(
            RuntimeRepresentationClass::TernaryTile640Base,
            1,
            BackendKind::Metal
        ));
    }

    #[test]
    fn unsupported_backend_rejected() {
        let reg = CapabilityRegistry::default_metal_v1();
        assert!(!reg.is_production_ready(
            RuntimeRepresentationClass::Nf4Tile640Base,
            1,
            BackendKind::CpuReference
        ));
    }

    #[test]
    fn production_ladder_verification() {
        let reg = CapabilityRegistry::default_metal_v1();
        let good_ladder = reg.candidate_ladder(BackendKind::Metal);
        assert!(reg
            .verify_production_ladder(&good_ladder, BackendKind::Metal)
            .is_ok());
        let bad_ladder = vec![(RuntimeRepresentationClass::TernaryTile640Base, 1)];
        assert!(reg
            .verify_production_ladder(&bad_ladder, BackendKind::Metal)
            .is_err());
    }
}
