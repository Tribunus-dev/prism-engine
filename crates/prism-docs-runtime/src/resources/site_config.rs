//! `SiteConfig` resource — site-wide build metadata.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiteConfig {
    pub site_title: String,
    pub site_tagline: String,
    pub canonical_origin: String,
    pub build_id: String,
}

impl Default for SiteConfig {
    fn default() -> Self {
        Self {
            site_title: "Prism Engine".into(),
            site_tagline: "Inspectable heterogeneous AI deployment".into(),
            canonical_origin: "https://prism-engine.example".into(),
            build_id: "dev".into(),
        }
    }
}

impl Component for SiteConfig {}
