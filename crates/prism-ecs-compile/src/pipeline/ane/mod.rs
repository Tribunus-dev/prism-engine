//! `pipeline::ane` — Apple Neural Engine compilation surface.
//!
//! This module owns the canonical authority for the ANE-specific
//! compile pipeline: legality rules derived from Orion's
//! `pass_ane_validate.c`, the fusion pass that merges adjacent ANE
//! regions, and the derived artifact schemas (MIL text, IOSurface
//! contracts, weight-blob plans).
//!
//! | File | Responsibility |
//! |---|---|
//! | [`legality`] | ANE rule trait, evaluator, receipts |
//! | [`rules`]    | Concrete ANE rules (Concat, F16, size, op limit) |
//! | [`fusion`]   | ANE fusion pass + `AneFusedArtifact` |
//! | [`artifacts`] | Derived ANE artifacts (MIL, IOSurface, BLOBFILE) |

pub mod artifacts;
pub mod fusion;
pub mod legality;
pub mod rules;
