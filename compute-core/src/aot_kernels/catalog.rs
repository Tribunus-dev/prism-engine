//! Kernel variant catalog — CImageKernelCatalog with variant entries,
//! metallib payload references, and serialization for CImage embedding.

use serde::{Deserialize, Serialize};

use super::parameters::{KernelFamily, KernelParameters};
use super::profile_id::AppleSiliconProfileId;

/// The kernel catalog embedded in a CImage.
///
/// Contains all compiled kernel variants for a single model, indexed by
/// target profile, kernel family, and codec family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageKernelCatalog {
    pub catalog_version: u32,
    pub variants: Vec<KernelVariantEntry>,
    pub metallib_payloads: Vec<KernelMetallibPayloadRef>,
}

/// One compiled kernel variant in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelVariantEntry {
    pub variant_id: String,
    pub target_profile: AppleSiliconProfileId,
    pub fallback_profiles: Vec<AppleSiliconProfileId>,
    pub kernel_family: KernelFamily,
    pub entry_point: String,
    pub parameters: KernelParameters,
    pub metallib_payload_id: String,
    pub compile_receipt_id: String,
    pub validation_receipt_id: String,
    pub performance_receipt_id: Option<String>,
}

/// Reference to an embedded metallib payload within the CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetallibPayloadRef {
    pub payload_id: String,
    pub digest: String,
    pub byte_offset: u64,
    pub byte_length: u64,
}

impl CImageKernelCatalog {
    /// Create an empty catalog.
    pub fn empty() -> Self {
        Self {
            catalog_version: 1,
            variants: Vec::new(),
            metallib_payloads: Vec::new(),
        }
    }

    /// Add a variant entry to the catalog.
    pub fn add_variant(&mut self, entry: KernelVariantEntry) {
        self.variants.push(entry);
    }

    /// Find all variants for a given kernel family.
    pub fn variants_for_family(&self, family: KernelFamily) -> Vec<&KernelVariantEntry> {
        self.variants
            .iter()
            .filter(|v| v.kernel_family == family)
            .collect()
    }

    /// Find variants targeting a specific profile.
    pub fn variants_for_profile(&self, profile: AppleSiliconProfileId) -> Vec<&KernelVariantEntry> {
        self.variants
            .iter()
            .filter(|v| v.target_profile == profile)
            .collect()
    }

    /// Find the best variant for a given profile and kernel family.
    /// Returns variants ordered by preference: exact > same gen > fallback.
    pub fn best_variant(
        &self,
        profile: AppleSiliconProfileId,
        family: KernelFamily,
    ) -> Option<&KernelVariantEntry> {
        let mut candidates: Vec<&KernelVariantEntry> = self
            .variants
            .iter()
            .filter(|v| v.kernel_family == family)
            .collect();

        // Sort by match quality: exact profile first, then same
        // generation, then fallback profiles, then generic.
        candidates.sort_by_key(|v| {
            if v.target_profile == profile {
                0
            } else if v.target_profile.soc_generation() == profile.soc_generation() {
                1
            } else if v.fallback_profiles.contains(&profile) {
                2
            } else {
                3
            }
        });

        candidates.into_iter().next()
    }

    /// Number of unique profiles covered by this catalog.
    pub fn profile_coverage(&self) -> Vec<AppleSiliconProfileId> {
        let mut ids: Vec<_> = self.variants.iter().map(|v| v.target_profile).collect();
        ids.sort();
        ids.dedup();
        ids
    }
}
