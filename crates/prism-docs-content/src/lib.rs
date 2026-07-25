//! Prism docs content — typed content schema for the site.
//!
//! This crate owns the canonical authority for the docs site content
//! surface. It defines the manifest schema (entities, kinds, links), the
//! typed shapes of each entity kind, the markdown body parser, and the
//! validation rules. It is consumed by `prism-docs-runtime` to populate
//! a `prism-ecs-core::World` and by `prism-docs-ssg` to drive the build.
//!
//! The crate has no ECS dependency and no rendering dependency. The
//! typed shapes are the source of truth; the runtime and the SSG are
//! projections.

pub mod adr;
pub mod chapter;
pub mod claim;
pub mod error;
pub mod link;
pub mod manifest;
pub mod markdown;
pub mod ontology;
pub mod page;
pub mod source_ref;

pub use adr::{Adr, AdrStatus};
pub use chapter::Chapter;
pub use claim::Claim;
pub use error::ContentError;
pub use link::{Link, LinkKind};
pub use manifest::{
    ContentManifest, EntityEntry, EntityId, EntityKind, ManifestLoad, RawManifest,
};
pub use markdown::MarkdownDocument;
pub use ontology::{ClaimClass, ExistenceState, KnowledgeState, OpticalState, ObserverMode};
pub use page::Page;
pub use source_ref::SourceRef;
