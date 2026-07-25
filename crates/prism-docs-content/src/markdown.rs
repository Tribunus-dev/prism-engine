//! Markdown body parser.
//!
//! This module owns the canonical authority for converting the
//! markdown body of a chapter or ADR into an HTML string the renderer
//! can use.
//!
//! Frontmatter is `---`-delimited YAML at the top of the file. The body
//! is the remaining CommonMark.

use std::path::Path;

use pulldown_cmark::{html, Options, Parser};

use crate::error::ContentError;

/// Parsed frontmatter + body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownDocument {
    pub frontmatter: String,
    pub body_markdown: String,
    /// Rendered HTML for the body. Lazily computed by `render`.
    rendered_html: Option<String>,
}

impl MarkdownDocument {
    /// Parse raw file content into frontmatter + body.
    pub fn parse(path: &Path, raw: &str) -> Result<Self, ContentError> {
        let (frontmatter, body_markdown) = split_frontmatter(path, raw)?;
        Ok(Self {
            frontmatter,
            body_markdown,
            rendered_html: None,
        })
    }

    /// Render the body to HTML. Cached after first call.
    pub fn render(&mut self) -> &str {
        if self.rendered_html.is_none() {
            let mut options = Options::empty();
            options.insert(Options::ENABLE_TABLES);
            options.insert(Options::ENABLE_STRIKETHROUGH);
            options.insert(Options::ENABLE_TASKLISTS);
            options.insert(Options::ENABLE_FOOTNOTES);
            let parser = Parser::new_ext(&self.body_markdown, options);
            let mut out = String::new();
            html::push_html(&mut out, parser);
            self.rendered_html = Some(out);
        }
        self.rendered_html.as_deref().unwrap_or("")
    }
}

fn split_frontmatter(
    path: &Path,
    raw: &str,
) -> Result<(String, String), ContentError> {
    // Frontmatter is delimited by lines containing only `---`.
    // We split by line, but we keep the trailing newline of each
    // body line so the rendered markdown round-trips with the
    // original byte content.
    let mut lines = raw.split_inclusive('\n');
    let first = lines.next();
    if first.map(|s| s.trim_end()) != Some("---") {
        return Err(ContentError::FrontmatterInvalid {
            path: path.to_path_buf(),
            message: "file must begin with `---` frontmatter delimiter".into(),
        });
    }
    let mut fm_lines: Vec<&str> = Vec::new();
    let mut body_lines: Vec<&str> = Vec::new();
    let mut in_fm = true;
    for line in lines {
        let trimmed = line.trim_end_matches('\n');
        if in_fm && trimmed == "---" {
            in_fm = false;
            continue;
        }
        if in_fm {
            fm_lines.push(line);
        } else {
            body_lines.push(line);
        }
    }
    if in_fm {
        return Err(ContentError::FrontmatterInvalid {
            path: path.to_path_buf(),
            message: "frontmatter opened with `---` but never closed".into(),
        });
    }
    let body = if body_lines.is_empty() {
        String::new()
    } else {
        // Reassemble. We removed the trailing newline of the closing
        // `---` separator; everything from there on keeps its
        // original line endings.
        body_lines.join("")
    };
    Ok((fm_lines.join(""), body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parse_with_frontmatter() {
        let raw = "---\ntitle: Hello\n---\n\nThis is the body.\n";
        let path = PathBuf::from("test.md");
        let mut doc = MarkdownDocument::parse(&path, raw).unwrap();
        assert_eq!(doc.frontmatter, "title: Hello\n");
        assert_eq!(doc.body_markdown, "\nThis is the body.\n");
        let html = doc.render().to_string();
        assert!(html.contains("<p>This is the body.</p>"));
    }

    #[test]
    fn missing_frontmatter_is_error() {
        let raw = "no frontmatter here\n";
        let path = PathBuf::from("test.md");
        let err = MarkdownDocument::parse(&path, raw).unwrap_err();
        assert!(matches!(err, ContentError::FrontmatterInvalid { .. }));
    }

    #[test]
    fn unterminated_frontmatter_is_error() {
        let raw = "---\ntitle: Hello\n";
        let path = PathBuf::from("test.md");
        let err = MarkdownDocument::parse(&path, raw).unwrap_err();
        assert!(matches!(err, ContentError::FrontmatterInvalid { .. }));
    }

    #[test]
    fn render_is_idempotent() {
        let raw = "---\ntitle: x\n---\n\n# Heading\n\nBody.\n";
        let path = PathBuf::from("test.md");
        let mut doc = MarkdownDocument::parse(&path, raw).unwrap();
        let a = doc.render().to_string();
        let b = doc.render().to_string();
        assert_eq!(a, b);
    }
}
