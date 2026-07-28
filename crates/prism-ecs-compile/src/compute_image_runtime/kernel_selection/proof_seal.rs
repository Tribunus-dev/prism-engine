//! Profile proof seal — compile-time artifact selection receipts with
//! numerical and resource-fit verification evidence.

use serde::{Deserialize, Serialize};

use super::selection::PreselectedKernelVariant;
use crate::compute_image_runtime::verification::numerical::NumericalVerificationReceipt;
use crate::compute_image_runtime::verification::resource_fit::ResourceFitReceipt;

/// Compile-time proof seal for a single target profile.
///
/// Records the identity of the profile, the selected kernel variants,
/// and the numerical and resource-fit receipts that certify each
/// selection. The `seal_hash` binds the entire profile seal into a
/// single digest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileProofSeal {
    /// Profile identifier.
    pub profile_id: String,
    /// Profile content hash (hex string).
    pub profile_hash: String,
    /// Per-shape-class kernel variant selections.
    pub variant_selections: Vec<PreselectedKernelVariant>,
    /// Numerical verification receipts.
    pub numerical_receipts: Vec<NumericalVerificationReceipt>,
    /// Resource-fit receipts.
    pub resource_fit_receipts: Vec<ResourceFitReceipt>,
    /// Binding hash for the entire profile seal.
    pub seal_hash: String,
}

/// A bundle of profile proof seals, aggregated for batch attestation.
///
/// Multiple profiles may target different hardware or runtime
/// constraints (e.g. Metal vs Core ML, low-power vs high-throughput).
/// The `bundle_hash` binds the set of profiles into a single
/// attestable digest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileProofSealBundle {
    /// All profile seals in the bundle.
    pub profiles: Vec<ProfileProofSeal>,
    /// Binding hash for the entire bundle.
    pub bundle_hash: String,
}
