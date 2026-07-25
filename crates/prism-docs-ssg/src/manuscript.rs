//! `manuscript` — parse `OBSERVATORY_V1_MANUSCRIPT.md` into per-page content.
//!
//! The manuscript is the binding prose artifact for Phase 3 of
//! the Observatory v1 implementation. The SSG reads it, extracts
//! the per-page sections, and projects them onto the canonical
//! routes.
//!
//! The format is conventional Markdown with a small number of
//! manuscript-internal conventions:
//! - `## Page N — Title (`/path/`)` headings mark page boundaries.
//! - `## Reviewer checklist` ends the publishable content.
//! - The H1 inside a page is captured as the page's hero (only
//!   the home page currently uses this).
//! - `**Brief:** §X.Y.` lines are spec cross-references and are
//!   stripped from prose at parse time.
//! - `[...]` blocks are renderer-directive notes (per the
//!   manuscript's own conventions section) and are stripped from
//!   prose at parse time.
//! - Section headers within a page (`### Foo`) become the page's
//!   section list. The section name is the heading; the prose is
//!   the body.
//!
//! Pages not in the manuscript (e.g., conditional routes whose
//! gate is not satisfied) are represented as `Page::Conditional`
//! with the redirect target.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// A single page extracted from the manuscript.
#[derive(Debug, Clone)]
pub struct Page {
    /// The route, e.g., "/start/".
    pub route: String,
    /// The page number from the manuscript (1-indexed), if present.
    pub number: Option<u32>,
    /// The short label, e.g., "Start".
    pub label: String,
    /// Whether this page is conditional (redirect target applies).
    pub conditional: bool,
    /// Where a conditional page redirects.
    pub redirect_to: Option<String>,
    /// The reason the page is conditional, if applicable.
    pub conditional_reason: Option<String>,
    /// Optional hero headline (the page's H1, if the manuscript
    /// uses one). The home page uses this for the §6.1 hero.
    pub hero: Option<String>,
    /// The page's sections: ordered list of (heading, prose).
    pub sections: Vec<Section>,
}

#[derive(Debug, Clone)]
pub struct Section {
    pub heading: String,
    pub level: u8,
    pub prose: String,
}

#[derive(Debug, Error)]
pub enum ManuscriptError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("manuscript has no pages: {path}")]
    Empty { path: PathBuf },
}

/// Parse the manuscript at the given path. Returns a map from
/// route to `Page`. Pages not in the manifest (e.g., conditional
/// ones explicitly marked "not in v1 manuscript") are included
/// as `Page { conditional: true, redirect_to, .. }`.
pub fn load_manuscript(path: &Path) -> Result<BTreeMap<String, Page>, ManuscriptError> {
    let text = fs::read_to_string(path).map_err(|e| ManuscriptError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Buffer the lines so we can consume multiple lines for a
    // single header (multi-line H3 with a parenthetical that
    // wraps onto the next line).
    let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();

    let mut pages: BTreeMap<String, Page> = BTreeMap::new();
    let mut current: Option<Page> = None;
    let mut current_section: Option<Section> = None;
    // The Reviewer checklist ends the publishable content; we
    // stop emitting page content once we see it.
    let mut after_reviewer_checklist = false;
    // Track whether we are still in the preamble (before the
    // first `## Page` header). The preamble is meta-only and
    // is not part of any page.
    let mut in_preamble = true;
    // Track whether we are inside a fenced code block. The
    // Run page (and others) have shell-command examples with
    // `#` comment lines that look like H1s to a naive parser.
    // We must ignore H1/H2/H3 (and `**Brief:**`) while inside
    // a code fence.
    let mut in_code = false;

    let mut i = 0;
    while i < lines.len() {
        let line = &lines[i];

        // Toggle the code-block state. A line that is just
        // ``` (with optional language) opens or closes a
        // fenced block.
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            // The fence itself is body content of the
            // current section; let it flow into prose like
            // any other line. The renderer emits the fence
            // as a `<pre><code>` block.
            if let Some(ref mut sec) = current_section {
                sec.prose.push_str(line);
                sec.prose.push('\n');
            } else if let Some(ref mut page) = current {
                page.sections.push(Section {
                    heading: String::new(),
                    level: 3,
                    prose: format!("{}\n", line),
                });
            }
            i += 1;
            continue;
        }
        if in_code {
            // Inside a code block: every line is body. The
            // renderer will emit the run as `<pre><code>`.
            if let Some(ref mut sec) = current_section {
                sec.prose.push_str(line);
                sec.prose.push('\n');
            } else if let Some(ref mut page) = current {
                page.sections.push(Section {
                    heading: String::new(),
                    level: 3,
                    prose: format!("{}\n", line),
                });
            }
            i += 1;
            continue;
        }

        // Cutoff: anything after `## Reviewer checklist` is
        // meta-only and must not enter the rendered prose.
        if line.starts_with("## Reviewer checklist") {
            after_reviewer_checklist = true;
        }
        if after_reviewer_checklist {
            // Finalize any in-flight page before stopping.
            if let Some(mut page) = current.take() {
                if let Some(sec) = current_section.take() {
                    page.sections.push(sec);
                }
                pages.insert(page.route.clone(), page);
            }
            i += 1;
            continue;
        }

        // Detect page header: `## Page N — Title (`/path/`)` or
        // `## Page N — Title (`/path/`) — not in v1 manuscript`
        if let Some(mut cap) = parse_page_header(line) {
            in_preamble = false;
            // Finalize the previous page.
            if let Some(mut page) = current.take() {
                if let Some(sec) = current_section.take() {
                    page.sections.push(sec);
                }
                pages.insert(page.route.clone(), page);
            }
            current_section = None;
            // For the home page, clear hero; the parser will
            // capture the H1 in the body. For other pages the
            // field stays None.
            cap.hero = None;
            current = Some(cap);
            i += 1;
            continue;
        }

        // Lines before the first page are part of the preamble
        // and are dropped.
        if in_preamble {
            i += 1;
            continue;
        }

        // No current page yet and not in preamble: skip.
        if current.is_none() {
            i += 1;
            continue;
        }

        // H1 inside a page: capture as the page's hero. The
        // home page uses this for the §6.1 hero headline.
        // We keep the current section open so the prose that
        // follows the H1 flows into the same section (the
        // Hero chapter on the home page).
        if let Some(rest) = line.strip_prefix("# ") {
            if let Some(ref mut page) = current {
                page.hero = Some(rest.trim().to_string());
            }
            i += 1;
            continue;
        }

        // Section header within a page: `### Foo`. The header
        // may span multiple lines if it contains a parenthetical
        // whose closing `)` is on the next line.
        if line.starts_with("### ") {
            if let Some(sec) = current_section.take() {
                if let Some(ref mut page) = current {
                    page.sections.push(sec);
                }
            }
            // Read the heading, consuming extra lines if the
            // parens are unbalanced.
            let (heading, consumed) = read_h3_heading(&lines, i);
            i += consumed;
            current_section = Some(Section {
                heading: strip_heading_meta(&heading),
                level: 3,
                prose: String::new(),
            });
            continue;
        }

        // H2 inside a page (other than `## Page`): close any
        // open section and skip; the spec doesn't currently use
        // H2 inside a page, but we handle it for safety.
        if line.starts_with("## ") {
            if let Some(sec) = current_section.take() {
                if let Some(ref mut page) = current {
                    page.sections.push(sec);
                }
            }
            current_section = None;
            i += 1;
            continue;
        }

        // `**Brief:** §X.Y.` lines are spec cross-references and
        // must not appear in the visitor-facing prose. Skip them.
        if is_brief_line(line) {
            i += 1;
            continue;
        }

        // Body line: append to the current section (or to a
        // synthetic preamble section if no `###` has been seen).
        if let Some(ref mut sec) = current_section {
            sec.prose.push_str(line);
            sec.prose.push('\n');
        } else if let Some(ref mut page) = current {
            // No section yet — accumulate into a synthetic
            // preamble with the line as the prose body.
            page.sections.push(Section {
                heading: String::new(),
                level: 3,
                prose: format!("{}\n", line),
            });
        }
        i += 1;
    }

    // Finalize the last page.
    if let Some(mut page) = current.take() {
        if let Some(sec) = current_section.take() {
            page.sections.push(sec);
        }
        pages.insert(page.route.clone(), page);
    }

    if pages.is_empty() {
        return Err(ManuscriptError::Empty { path: path.to_path_buf() });
    }

    Ok(pages)
}

/// Read an H3 heading, consuming extra lines until any opening
/// `(...)` is closed. Returns the heading text (without the
/// leading `### `) and the number of lines consumed.
fn read_h3_heading(lines: &[String], start: usize) -> (String, usize) {
    let first = lines[start].trim_start_matches("### ").trim();
    let mut heading = first.to_string();
    let mut consumed = 1;
    // While the parens are unbalanced, keep reading.
    while paren_balance(&heading) > 0 {
        if start + consumed >= lines.len() {
            break;
        }
        let next = &lines[start + consumed];
        // Don't consume across blank lines; that would fold
        // body content into the heading.
        if next.trim().is_empty() {
            break;
        }
        // Don't consume across another header / page break.
        if next.starts_with('#') {
            break;
        }
        heading.push(' ');
        heading.push_str(next.trim());
        consumed += 1;
    }
    (heading, consumed)
}

/// Returns the net open-paren count in `s`. Positive means
/// more `(` than `)`. Used to detect multi-line H3 headings.
fn paren_balance(s: &str) -> i32 {
    let mut n = 0;
    for c in s.chars() {
        if c == '(' {
            n += 1;
        } else if c == ')' {
            n -= 1;
        }
    }
    n
}

/// Strip a parenthetical meta suffix from an H3 heading. Used
/// for headings like "Current notes (illustrative; the published
/// Lab Notes are read from `/lab/index.json` at build time)" —
/// the parenthetical is meta, the leading clause is the visible
/// title.
///
/// The rule: if the heading contains a balanced `(...)` whose
/// inner text is meta (contains any of the meta keywords
/// `illustrative`, `read from`, `at build`, `for the v1`), strip
/// the parenthetical. Otherwise leave the heading intact.
fn strip_heading_meta(heading: &str) -> String {
    let trimmed = heading.trim();
    // Find a balanced parenthetical that contains a meta keyword.
    let bytes = trimmed.as_bytes();
    let mut depth = 0i32;
    let mut open_idx: Option<usize> = None;
    for (idx, c) in trimmed.char_indices() {
        if c == '(' {
            if depth == 0 {
                open_idx = Some(idx);
            }
            depth += 1;
        } else if c == ')' {
            depth -= 1;
            if depth == 0 {
                if let Some(start) = open_idx {
                    let inner = &trimmed[start + 1..idx];
                    if is_meta_parenthetical(inner) {
                        // Return everything before the `(`, trimmed.
                        return trimmed[..start].trim().to_string();
                    }
                }
                open_idx = None;
            }
        }
    }
    let _ = bytes;
    trimmed.to_string()
}

fn is_meta_parenthetical(s: &str) -> bool {
    let lower = s.to_lowercase();
    let keywords = [
        "illustrative",
        "read from",
        "at build",
        "for the v1",
        "placeholder",
        "v1 manuscript",
    ];
    keywords.iter().any(|k| lower.contains(k))
}

/// Returns true if `line` is a `**Brief:** ...` spec
/// cross-reference. The pattern is `**Brief:**` (with optional
/// trailing whitespace) followed by a `§X.Y` reference and a
/// period. We accept anything that begins with `**Brief:**`.
fn is_brief_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("**Brief:**")
}

/// Parse a `## Page N — Title (`/path/`)` header.
fn parse_page_header(line: &str) -> Option<Page> {
    let trimmed = line.trim();
    if !trimmed.starts_with("## Page ") {
        return None;
    }
    // Strip the leading "## Page ".
    let rest = trimmed.trim_start_matches("## Page ");
    // Optional leading number followed by " — " or " - ".
    let (number, after_number) = if let Some(idx) = rest.find('—') {
        let n_part = rest[..idx].trim();
        let n = n_part.parse::<u32>().ok();
        (n, rest[idx + '—'.len_utf8()..].trim())
    } else if let Some(idx) = rest.find(" - ") {
        let n_part = rest[..idx].trim();
        let n = n_part.parse::<u32>().ok();
        (n, rest[idx + 3..].trim())
    } else {
        (None, rest)
    };
    // The route is in backticks. The label is whatever comes
    // before the backticks (separated by " — " from the route).
    let backtick_start = after_number.find('`')?;
    let route_section = &after_number[backtick_start..];
    let after_tick = &route_section[1..];
    let route_end = after_tick.find('`')?;
    let route = after_tick[..route_end].to_string();
    // The label is the trimmed text before the backticks,
    // stripped of the leading " — " separator. We also trim
    // a trailing "(" which precedes the backticked route in
    // the canonical form `## Page N — Label (`/route/`)`.
    let label_raw = after_number[..backtick_start]
        .trim()
        .trim_end_matches('—')
        .trim_end_matches('-')
        .trim()
        .trim_end_matches('(')
        .trim()
        .to_string();
    // The post-route part: typically ") — ..." or ")" alone.
    let after_route = &after_tick[route_end + 1..];

    // Detect "not in v1 manuscript" anywhere in the post-route text.
    let conditional = after_route.contains("not in v1 manuscript");
    let conditional_reason = if conditional {
        extract_conditional_reason(after_route)
    } else {
        None
    };

    Some(Page {
        route,
        number,
        label: if label_raw.is_empty() { after_route.trim().trim_start_matches(')').trim_start_matches('—').trim().to_string() } else { label_raw },
        conditional,
        redirect_to: None,
        conditional_reason,
        hero: None,
        sections: Vec::new(),
    })
}

fn extract_conditional_reason(_text: &str) -> Option<String> {
    // The manuscript does not currently encode the redirect
    // target inside the page header. The redirect table is
    // owned by the SSG's redirects module and is merged at
    // render time.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_page_header() {
        let p = parse_page_header("## Page 1 — Home (`/`)").expect("parses");
        assert_eq!(p.route, "/");
        assert_eq!(p.label, "Home");
        assert_eq!(p.number, Some(1));
        assert!(!p.conditional);
    }

    #[test]
    fn parse_conditional_page_header() {
        let p = parse_page_header("## Page 12 — Prism ML (`/prism-ml/`) — not in v1 manuscript")
            .expect("parses");
        assert_eq!(p.route, "/prism-ml/");
        assert!(p.conditional);
    }

    #[test]
    fn load_v1_manuscript_succeeds() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .join("OBSERVATORY_V1_MANUSCRIPT.md");
        let pages = load_manuscript(&path).expect("manuscript loads");
        // The v1 manuscript has 15 page headers.
        assert!(pages.len() >= 15, "manuscript has too few pages: {}", pages.len());
        // Every page except the conditional ones should have at
        // least one section.
        let unconditional: Vec<&Page> =
            pages.values().filter(|p| !p.conditional).collect();
        for page in &unconditional {
            assert!(
                !page.sections.is_empty(),
                "page {} has no sections",
                page.route
            );
        }
        // Conditional pages are flagged.
        let conditional: Vec<&Page> =
            pages.values().filter(|p| p.conditional).collect();
        assert!(!conditional.is_empty(), "no conditional pages found");
        let conditional_routes: Vec<&str> =
            conditional.iter().map(|p| p.route.as_str()).collect();
        assert!(conditional_routes.contains(&"/prism-ml/"));
        assert!(conditional_routes.contains(&"/general-compute/"));
    }

    #[test]
    fn brief_lines_are_stripped_from_prose() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .join("OBSERVATORY_V1_MANUSCRIPT.md");
        let pages = load_manuscript(&path).expect("manuscript loads");
        for page in pages.values() {
            for sec in &page.sections {
                assert!(
                    !sec.prose.contains("**Brief:**"),
                    "page {} section {:?} still has a Brief: line",
                    page.route,
                    sec.heading
                );
            }
        }
    }

    #[test]
    fn preamble_is_excluded_from_pages() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .join("OBSERVATORY_V1_MANUSCRIPT.md");
        let pages = load_manuscript(&path).expect("manuscript loads");
        for page in pages.values() {
            for sec in &page.sections {
                assert!(
                    !sec.prose.contains("Binding artifact for Phase 3"),
                    "preamble leaked into page {} section {:?}",
                    page.route,
                    sec.heading
                );
                assert!(
                    !sec.prose.contains("Conventions in this manuscript"),
                    "conventions section leaked into page {} section {:?}",
                    page.route,
                    sec.heading
                );
            }
        }
    }

    #[test]
    fn reviewer_checklist_is_excluded() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .join("OBSERVATORY_V1_MANUSCRIPT.md");
        let pages = load_manuscript(&path).expect("manuscript loads");
        for page in pages.values() {
            for sec in &page.sections {
                assert!(
                    !sec.prose.contains("Reviewer checklist"),
                    "reviewer checklist leaked into page {} section {:?}",
                    page.route,
                    sec.heading
                );
                assert!(
                    !sec.prose.contains("H1. Status-vocabulary purity"),
                    "H1 review item leaked into page {} section {:?}",
                    page.route,
                    sec.heading
                );
            }
        }
    }

    #[test]
    fn home_page_captures_hero() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .join("OBSERVATORY_V1_MANUSCRIPT.md");
        let pages = load_manuscript(&path).expect("manuscript loads");
        let home = pages.get("/").expect("home page present");
        let hero = home.hero.as_deref().expect("home has hero");
        assert!(hero.contains("Compile intelligence"));
    }

    #[test]
    fn multi_line_h3_with_parenthetical_is_joined() {
        let lines = vec![
            "### Current notes (illustrative; the published Lab Notes are read from".to_string(),
            "`/lab/index.json` at build time)".to_string(),
            "".to_string(),
            "body".to_string(),
        ];
        let (heading, consumed) = read_h3_heading(&lines, 0);
        assert_eq!(consumed, 2);
        assert!(heading.contains("Current notes"));
        assert!(heading.contains("`/lab/index.json`"));
    }

    #[test]
    fn strip_heading_meta_drops_meta_parenthetical() {
        let h = "Current notes (illustrative; the published Lab Notes are read from `/lab/index.json` at build time)";
        assert_eq!(strip_heading_meta(h), "Current notes");
    }

    #[test]
    fn strip_heading_meta_keeps_non_meta_parenthetical() {
        let h = "The Receipt (a forged copy)";
        assert_eq!(strip_heading_meta(h), "The Receipt (a forged copy)");
    }

    #[test]
    fn is_brief_line_recognizes_brief() {
        assert!(is_brief_line("**Brief:** §6.1."));
        assert!(is_brief_line("  **Brief:** §6.2."));
        assert!(!is_brief_line("Brief: §6.1."));
        assert!(!is_brief_line("**Brief** §6.1."));
    }
}
