//! Body components — the rendered HTML body of a chapter or ADR.
//!
//! `MarkdownBody` is the *rendered* HTML, computed at build time by
//! the markdown parser. The body is a derived projection: deleting
//! the body and rebuilding it from the source markdown must produce
//! the same bytes.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkdownBody(pub String);
impl Component for MarkdownBody {}

/// The source markdown, kept for replay and re-rendering. Without
/// this, the body is opaque to the architecture; with it, the body
/// can be re-derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkdownSource(pub String);
impl Component for MarkdownSource {}

/// The path to the source markdown file (relative to the content
/// root). Used for diagnostic messages and for the body source link
/// in the rendered page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkdownSourcePath(pub String);
impl Component for MarkdownSourcePath {}

/// Sections extracted from the markdown body. A section is one
/// heading and the body content until the next heading of equal or
/// higher level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    pub level: u8,
    pub anchor: String,
    pub title: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkdownSections(pub Vec<Section>);
impl Component for MarkdownSections {}
