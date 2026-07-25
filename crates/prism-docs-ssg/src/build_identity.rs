//! `build_identity` — emit the `build.json` artifact that the
//! post-production smoke check (A22) verifies against the
//! expected commit.
//!
//! See `OBSERVATORY_V1_SPEC.md` §12 A16 and ADR-032 D9. The
//! `build.json` is served at the site root so a smoke-test
//! visitor can read it and confirm the live build's identity.

use serde::{Deserialize, Serialize};

use crate::data_layer::{DataLayer, SiteSummary};

/// The build identity serialized to `build.json` at the site root.
/// Every field is populated at build time from the data layer
/// (`site.json` `build_identity`) plus the build environment
/// (the actual `cargo` invocation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildIdentity {
    pub commit: String,
    pub build_id: String,
    pub build_kind: String,
    pub recorded_at: String,
    pub ssg_version: String,
    pub toolchain: String,
    pub source_repo: String,
    pub deployment_compatibility_cutover: String,
    pub schema_version: String,
}

/// Build the build identity from a validated data layer plus the
/// runtime environment.
#[allow(unused_variables)]
pub fn build_identity(
    layer: &DataLayer,
    site: &SiteSummary,
    build_id_override: Option<String>,
    build_kind: &str,
) -> BuildIdentity {
    let build_id = build_id_override.unwrap_or_else(|| site.build_identity.build_id.clone());

    BuildIdentity {
        commit: site.build_identity.commit.clone(),
        build_id,
        build_kind: build_kind.to_string(),
        recorded_at: site.build_identity.recorded_at.clone(),
        ssg_version: env!("CARGO_PKG_VERSION").to_string(),
        toolchain: format!("rustc {}", rustc_version_runtime()),
        source_repo: "https://github.com/Tribunus-dev/prism-engine".to_string(),
        deployment_compatibility_cutover: site
            .deployment_compatibility_window
            .cutover_date
            .clone(),
        schema_version: site.schema_version.clone(),
    }
}

/// Try to get the rustc version at runtime. Falls back to a
/// static string when RUSTC_VERSION is not set (e.g., in tests
/// without a rustc on PATH).
fn rustc_version_runtime() -> String {
    std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string())
}

impl BuildIdentity {
    /// Serialize to the JSON the site root serves.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn load_layer() -> (DataLayer, SiteSummary) {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .to_path_buf();
        let data_root = base.join("docs/data");
        let schema_dir = base.join("schemas");
        let layer = DataLayer::load(&data_root, &schema_dir).expect("data layer");
        let site = layer
            .get("site")
            .expect("site.json")
            .as_site_summary()
            .expect("site summary");
        (layer, site)
    }

    #[test]
    fn build_identity_includes_commit() {
        let (layer, site) = load_layer();
        let id = build_identity(&layer, &site, None, "release");
        assert!(!id.commit.is_empty());
        assert!(!id.build_id.is_empty());
        assert_eq!(id.build_kind, "release");
    }

    #[test]
    fn build_identity_serializes() {
        let (layer, site) = load_layer();
        let id = build_identity(&layer, &site, Some("test-build-1".to_string()), "dev");
        let json = id.to_json().expect("serialize");
        assert!(json.contains("\"commit\""));
        assert!(json.contains("\"test-build-1\""));
    }
}
