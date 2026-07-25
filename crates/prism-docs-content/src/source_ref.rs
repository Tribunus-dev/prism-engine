//! `SourceRef` — typed reference to a source file location.
//!
//! Used by claims to point at the canonical evidence in the
//! repository. Mirrors the previous JS `claim.sourceRefs` strings but
//! with typed structure.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceRef {
    /// Repository-relative path to the source file.
    pub path: PathBuf,
    /// Optional section anchor (e.g., the markdown heading slug).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    /// Optional line number, 1-indexed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

impl SourceRef {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            section: None,
            line: None,
        }
    }

    pub fn with_section(mut self, section: impl Into<String>) -> Self {
        self.section = Some(section.into());
        self
    }

    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    /// Canonical string form used in rendered HTML and link targets.
    pub fn to_anchor(&self) -> String {
        let mut out = self.path.to_string_lossy().to_string();
        if let Some(section) = &self.section {
            out.push('#');
            out.push_str(section);
        }
        if let Some(line) = self.line {
            out.push(':');
            out.push_str(&line.to_string());
        }
        out
    }
}

impl fmt::Display for SourceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_anchor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_format() {
        let r = SourceRef::new("crates/prism-ecs-core/src/world.rs").with_section("spawn");
        assert_eq!(
            r.to_anchor(),
            "crates/prism-ecs-core/src/world.rs#spawn"
        );
    }

    #[test]
    fn anchor_with_line() {
        let r = SourceRef::new("crates/prism-ecs-core/src/world.rs").with_line(199);
        assert_eq!(r.to_anchor(), "crates/prism-ecs-core/src/world.rs:199");
    }
}
