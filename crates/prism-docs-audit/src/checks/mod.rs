//! The 22 axiom checks. Each check is a function that
//! takes an `AuditContext` and returns a `CheckResult`.
//!
//! The checks are organized by axiom number. They are
//! independent: each one is a pure function of the
//! context. The runner aggregates the results.

//! The 22 axiom checks. Each check is a function that
//! takes an `AuditContext` and returns a `CheckResult`.
//!
//! The checks are organized by axiom number. They are
//! independent: each one is a pure function of the
//! context. The runner aggregates the results.

#[path = "a01_route_integrity.rs"]
pub mod a01_route_integrity;
#[path = "a02_status_vocabulary.rs"]
pub mod a02_status_vocabulary;
#[path = "a03_data_layer_validation.rs"]
pub mod a03_data_layer_validation;
#[path = "a04_evidence_boundary.rs"]
pub mod a04_evidence_boundary;
#[path = "a05_chapter_locality.rs"]
pub mod a05_chapter_locality;
#[path = "a06_component_registration.rs"]
pub mod a06_component_registration;
#[path = "a07_manuscript_match.rs"]
pub mod a07_manuscript_match;
#[path = "a08_diagram_caption.rs"]
pub mod a08_diagram_caption;
#[path = "a09_reduced_motion.rs"]
pub mod a09_reduced_motion;
#[path = "a10_keyboard_parity.rs"]
pub mod a10_keyboard_parity;
#[path = "a11_screen_reader.rs"]
pub mod a11_screen_reader;
#[path = "a12_no_js_rendering.rs"]
pub mod a12_no_js_rendering;
#[path = "a13_schema_integrity.rs"]
pub mod a13_schema_integrity;
#[path = "a14_evidence_applicability.rs"]
pub mod a14_evidence_applicability;
#[path = "a15_canonical_urls.rs"]
pub mod a15_canonical_urls;
#[path = "a16_build_identity.rs"]
pub mod a16_build_identity;
#[path = "a17_status_not_color.rs"]
pub mod a17_status_not_color;
#[path = "a18_performance_budget.rs"]
pub mod a18_performance_budget;
#[path = "a19_security_privacy.rs"]
pub mod a19_security_privacy;
#[path = "a20_accessibility_extras.rs"]
pub mod a20_accessibility_extras;
#[path = "a21_allowlist.rs"]
pub mod a21_allowlist;
#[path = "a22_deployment_smoke.rs"]
pub mod a22_deployment_smoke;

/// Canonical routes per OBSERVATORY_V1_SPEC.md §7.1.
/// Used by A1 (route integrity) and A21 (allowlist).
pub const CANONICAL_ROUTES: &[&str] = &[
    "/",
    "/start/",
    "/architecture/",
    "/computeimage/",
    "/computeimage/specimen/",
    "/evidence/",
    "/status/",
    "/lab/",
    "/observatory/life/",
    "/roadmap/",
    "/run/",
    "/colophon/",
];

/// The forbidden status-bearing words per OBSERVATORY_V1_
/// SPEC.md §3 (the §3.3 vocabulary is closed; these are
/// not members). The linter flags them in page prose;
/// matches inside known-meta contexts are ignored.
pub const FORBIDDEN_STATUS_WORDS: &[&str] = &[
    "supported",
    "ready",
    "active",
    "available",
    "live",
];
