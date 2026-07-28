//! Level 3 — bridge provider routing, validation, and capability cache.
//!
//! Level 3 sits above the dense Metal (Level 1) and stateless Core ML (Level 2)
//! teachers. It is responsible for selecting, validating, and dispatching
//! bridge routes between student and teacher activation memory.
//!
//! Three providers are defined:
//!   - **MaterializationProvider** — explicit copy fallback, always available.
//!   - **SharedRouteProvider** — verified zero-copy shared-memory route, requires
//!     all four validation gates to pass before claiming zero-copy.
//!   - **StitchedProvider** — experimental model stitching for teacher-side region
//!     composition; disabled by default.
//!
//! Each provider implements `super::bridge_provider::BridgeProvider`. The
//! router (`routing::Level3Router`) maintains a capability fingerprint cache
//! and selects the best available route.
//!
//! ## Engine-feature gating
//!
//! These submodules are NOT feature-gated at the module-declaration level
//! because engine callers (level2/scheduler, level3/routing) reference them
//! across crate boundaries. The implementations carry their own internal
//! cfg-gates where the code depends on platform-specific libraries.

pub mod gates;
pub mod providers;
pub mod routing;
