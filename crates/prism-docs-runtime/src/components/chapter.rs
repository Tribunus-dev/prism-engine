//! Chapter components — the typed view of a chapter entity.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

/// The chapter's title. Required.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChapterTitle(pub String);

impl Component for ChapterTitle {}

/// The chapter's slug (URL fragment, e.g., "home-intent").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChapterSlug(pub String);

impl Component for ChapterSlug {}

/// Display order. Lower numbers come first in navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChapterOrder(pub u32);

impl Component for ChapterOrder {}

/// The one-sentence intent of the chapter. Required.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChapterIntent(pub String);

impl Component for ChapterIntent {}

/// Optional short blurb shown in nav and chapter index.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChapterBlurb(pub String);

impl Component for ChapterBlurb {}

/// Optional reading time in minutes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChapterReadingMinutes(pub u32);

impl Component for ChapterReadingMinutes {}

/// Path to the chapter's markdown body, relative to the content
/// root. The SSG reads this and attaches a `MarkdownBody` component
/// to the entity during the build.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChapterBodyPath(pub String);

impl Component for ChapterBodyPath {}
