//! Compilation authority and source-identity contract types.
//!
//! This module owns the canonical authority for the [`CompilationAuthority`]
//! discriminant — the typed contract that gates whether a compile is
//! allowed to run as a `TestFixture` (with the strict fixture ceiling)
//! or as a `SealedComputeImage` (with the production profile). The
//! module also owns [`ImageBuildAttestation`], the JSON-serializable
//! record emitted to `seal.json` and the build log that proves the
//! compile was performed by an authorized profile.
//!
//! The authority is **declarative** — the rest of the pipeline consults
//! the value but does not own the meaning. Changing the meaning of a
//! variant is a constitutional change and must be reviewed under the
//! authority gate.

use serde::{Deserialize, Serialize};
use serde_json::json;

/// Compilation authority discriminant.
///
/// Two variants are recognized:
///
/// - `TestFixture` — used by `cargo test` and ad-hoc invocations. Enforces
///   the fixture ceiling (max 4 layers, 256 tensors, 128 MB total source).
///   Refuses to run under the `image-build` profile.
/// - `SealedComputeImage` — used by the production image build. Requires
///   the `image-build` profile to be active at compile time.
///
/// The discriminant is the **single source of truth** for which profile
/// produced a CImage. Every receipt emitted by the pipeline records it;
/// the post-emission reader uses it to decide whether a CImage is
/// admissible to a runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompilationAuthority {
    /// Test fixture — strict ceiling, refuses `image-build` profile.
    TestFixture,
    /// Sealed compute image — production profile only.
    SealedComputeImage,
}

impl CompilationAuthority {
    /// The stable string form of the authority discriminant, used in
    /// receipts and manifests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TestFixture => "TestFixture",
            Self::SealedComputeImage => "SealedComputeImage",
        }
    }
}

impl Default for CompilationAuthority {
    fn default() -> Self {
        Self::SealedComputeImage
    }
}

/// Profile attestation record emitted to the build log and `seal.json`.
///
/// The attestation answers the question: *was this CImage produced by
/// an authorized profile?* The fields are populated from compile-time
/// `option_env!` lookups so the values are deterministic for a given
/// build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageBuildAttestation {
    /// Compile profile name (e.g. `image-build`).
    pub profile: String,
    /// Optimization level (`0`, `1`, `2`, `3`).
    pub opt_level: String,
    /// LTO setting (`expected-fat-per-image-build-profile` etc.).
    pub lto: String,
    /// Codegen-units setting.
    pub codegen_units: String,
    /// Whether debug assertions are enabled.
    pub debug_assertions: bool,
    /// Whether incremental compilation is enabled.
    pub incremental: String,
    /// Target triple (e.g. `aarch64-apple-darwin`).
    pub target: String,
    /// Whether the build is *authorized* — opt-level 3, no debug
    /// assertions, and target `aarch64-apple-darwin`.
    pub authorized: bool,
}

/// Construct an attestation record from compile-time environment
/// variables. The fields are deterministic for a given build.
pub fn image_build_attestation() -> ImageBuildAttestation {
    let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown").to_string();
    let opt_level = option_env!("TRIBUNUS_OPT_LEVEL").unwrap_or("0").to_string();
    let target = option_env!("TRIBUNUS_TARGET").unwrap_or("unknown").to_string();
    let authorized = opt_level == "3" && !cfg!(debug_assertions) && target == "aarch64-apple-darwin";
    ImageBuildAttestation {
        profile,
        opt_level,
        lto: "expected-fat-per-image-build-profile".to_string(),
        codegen_units: "expected-1-per-image-build-profile".to_string(),
        debug_assertions: cfg!(debug_assertions),
        incremental: "expected-false-per-image-build-profile".to_string(),
        target,
        authorized,
    }
}

/// Serialize the attestation as the canonical JSON form used in
/// `seal.json` and the build log.
pub fn image_build_attestation_json() -> serde_json::Value {
    let a = image_build_attestation();
    json!({
        "event": "compiler_profile",
        "profile": a.profile,
        "opt_level": a.opt_level,
        "lto": a.lto,
        "codegen_units": a.codegen_units,
        "debug_assertions": a.debug_assertions,
        "incremental": a.incremental,
        "target": a.target,
        "authorized": a.authorized,
    })
}
