//! Runtime kernel variant selector — picks the best embedded variant
//! from the CImage kernel catalog for the current Metal device.
//!
//! Selection rule:
//! 1. Exact profile match.
//! 2. Same SoC generation and same GPU tier match.
//! 3. Same or lower capability fallback.
//! 4. Generic AppleSilicon fallback.
//! 5. Built-in conservative kernel fallback.

use serde::{Deserialize, Serialize};

use super::catalog::{CImageKernelCatalog, KernelVariantEntry};
use super::device_match::RuntimeMetalDeviceProfile;
use super::parameters::KernelFamily;
use super::profile_id::AppleSiliconProfileId;

/// Result of variant selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantSelection {
    /// The selected variant, if found.
    pub variant: Option<KernelVariantEntry>,
    /// Whether this was an exact match or a fallback.
    pub match_type: MatchType,
    /// Whether the fallback to built-in conservative kernel was used.
    pub fallback_used: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchType {
    Exact,
    SameGeneration,
    FallbackList,
    GenericAppleSilicon,
    BuiltInConservative,
}

/// Kernel variant selector for runtime device matching.
pub struct KernelVariantSelector;

impl KernelVariantSelector {
    /// Select the best embedded variant for the given runtime device and kernel family.
    ///
    /// If no embedded variant is available, returns a selection with
    /// `fallback_used = true` and no variant (caller should dispatch
    /// the built-in conservative kernel).
    pub fn select_variant(
        catalog: &CImageKernelCatalog,
        runtime_device: &RuntimeMetalDeviceProfile,
        kernel_family: KernelFamily,
        profile_db: &super::profile_db::AppleSiliconProfileDb,
    ) -> VariantSelection {
        // 1. Match runtime device to a profile ID.
        let device_profile =
            super::device_match::match_device_to_profile(runtime_device, profile_db);

        // 2. Search the catalog for the best variant.
        match catalog.best_variant(device_profile, kernel_family) {
            Some(variant) => {
                let is_exact = variant.target_profile == device_profile;
                VariantSelection {
                    variant: Some(variant.clone()),
                    match_type: if is_exact {
                        MatchType::Exact
                    } else {
                        MatchType::SameGeneration
                    },
                    fallback_used: false,
                }
            }
            None => {
                // 3. No embedded variant found. Try generic fallback.
                let generic = AppleSiliconProfileId::UnknownAppleSilicon;
                if let Some(variant) = catalog.best_variant(generic, kernel_family) {
                    return VariantSelection {
                        variant: Some(variant.clone()),
                        match_type: MatchType::GenericAppleSilicon,
                        fallback_used: true,
                    };
                }

                // 4. No variant at all — caller must use built-in conservative kernel.
                VariantSelection {
                    variant: None,
                    match_type: MatchType::BuiltInConservative,
                    fallback_used: true,
                }
            }
        }
    }
}
