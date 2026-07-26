//! Audit context. The runner's input.
//!
//! The runner takes a `SiteSource` (a local directory or a
//! live URL) and produces an `AuditContext` that holds the
//! canonical paths, the data layer, the manuscript, and any
//! other artifact the checks need. The context is built
//! once and shared across all checks; checks are pure
//! functions of the context.

use std::path::{Path, PathBuf};

use crate::AuditError;

/// The source of the site to audit. A local directory of
/// built artifacts (the SSG output) or a live URL.
#[derive(Debug, Clone)]
pub enum SiteSource {
    /// A local directory containing the rendered site
    /// (`index.html`, `site.css`, `site.js`, the data
    /// layer, the manuscript, the schemas).
    LocalDir(PathBuf),
    /// A live URL. The runner fetches each canonical path
    /// over HTTPS and audits the responses.
    #[allow(dead_code)] // exercised in future browser-required checks
    Url(String),
}

impl SiteSource {
    pub fn describe(&self) -> String {
        match self {
            SiteSource::LocalDir(p) => format!("local:{}", p.display()),
            SiteSource::Url(u) => format!("url:{}", u),
        }
    }
}

/// The audit context. Built once; passed to every check.
#[derive(Debug, Clone)]
pub struct AuditContext {
    /// The site source.
    pub source: SiteSource,
    /// The canonical route list, derived from the SSG's
    /// emitted pages or from the data layer.
    pub canonical_routes: Vec<String>,
    /// All rendered HTML files (path, contents).
    pub html_files: Vec<(String, String)>,
    /// The CSS bundle, if found at the site root.
    pub css: Option<String>,
    /// The JS bundles referenced by the pages
    /// (selection-controller, theme, transitions, etc.).
    pub js_files: Vec<(String, String)>,
    /// The data layer directory (if the source is a local
    /// directory).
    pub data_layer_dir: Option<PathBuf>,
    /// The schema directory.
    pub schemas_dir: Option<PathBuf>,
    /// The manuscript file.
    pub manuscript: Option<String>,
    /// The `docs-publication.json` allowlist.
    pub publication_allowlist: Option<Vec<String>>,
    /// The build identity from `build.json`.
    pub build_json: Option<serde_json::Value>,
}

impl AuditContext {
    /// Build the context from a local directory.
    pub fn from_local_dir(root: &Path) -> Result<Self, AuditError> {
        if !root.exists() {
            return Err(AuditError::InvalidSource(format!(
                "directory does not exist: {}",
                root.display()
            )));
        }
        if !root.is_dir() {
            return Err(AuditError::InvalidSource(format!(
                "not a directory: {}",
                root.display()
            )));
        }

        let mut html_files: Vec<(String, String)> = Vec::new();
        let mut css: Option<String> = None;
        let mut js_files: Vec<(String, String)> = Vec::new();

        // Walk the rendered site. Every `index.html` becomes
        // a route; the root `site.css` becomes the bundle;
        // every `.js` at the site root becomes a JS file.
        for entry in walkdir::WalkDir::new(root)
            .max_depth(4)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let rel = path
                .strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            if path.file_name().and_then(|s| s.to_str()) == Some("index.html") {
                let contents = std::fs::read_to_string(path).map_err(|e| {
                    AuditError::Io {
                        path: path.display().to_string(),
                        source: e,
                    }
                })?;
                let route = if rel == "index.html" {
                    "/".to_string()
                } else {
                    let dir = rel.trim_end_matches("/index.html");
                    format!("/{}/", dir.trim_end_matches('/'))
                };
                html_files.push((route, contents));
            } else if rel == "site.css" {
                css = Some(
                    std::fs::read_to_string(path).map_err(|e| AuditError::Io {
                        path: path.display().to_string(),
                        source: e,
                    })?,
                );
            } else if rel.ends_with(".js") {
                let contents = std::fs::read_to_string(path).map_err(|e| {
                    AuditError::Io {
                        path: path.display().to_string(),
                        source: e,
                    }
                })?;
                js_files.push((rel, contents));
            }
        }

        // Sort for determinism.
        html_files.sort_by(|a, b| a.0.cmp(&b.0));
        js_files.sort();

        // The v1 surface is exactly the canonical routes
        // (per OBSERVATORY_V1_SPEC.md §7.1) plus the 404.
        // Filter the walked HTML to only those routes; any
        // other index.html under docs/ is legacy residue
        // from the previous constitutional-ECS demo and
        // must be cleaned up by a follow-on (not by the
        // audit runner). The runner reports on the v1
        // surface as the SSG emits it.
        let canonical: std::collections::BTreeSet<String> = crate::checks::CANONICAL_ROUTES
            .iter()
            .map(|r| r.to_string())
            .chain(std::iter::once("/404/".to_string()))
            .collect();
        html_files.retain(|(r, _)| canonical.contains(r));

        // The 404 page is emitted as `404.html` (not as
        // `index.html`), so the index.html walker does not
        // pick it up. Load it explicitly so checks that
        // count pages (A1, A7, A11, A16) see the full v1
        // surface.
        let four_oh_four_html = root.join("404.html");
        if four_oh_four_html.exists() {
            if let Ok(contents) = std::fs::read_to_string(&four_oh_four_html) {
                html_files.push(("/404/".to_string(), contents));
            }
        }

        let canonical_routes: Vec<String> =
            html_files.iter().map(|(r, _)| r.clone()).collect();

        // Look for the data layer, the schemas, the
        // manuscript, the publication allowlist, and the
        // build identity at the site root.
        let data_layer_dir = Some(root.join("data"));
        let schemas_dir = {
            // The schema directory is conventionally at
            // `<repo>/schemas`, not `<site>/schemas`. The
            // runner accepts a path override via the
            // `schemas` field if the caller wires one in.
            // For a local site root, we look for
            // `<root>/../schemas`.
            let p = root.join("../schemas");
            if p.exists() {
                Some(p.canonicalize().unwrap_or(p))
            } else {
                None
            }
        };
        let manuscript = {
            let p = root.join("../OBSERVATORY_V1_MANUSCRIPT.md");
            std::fs::read_to_string(&p).ok()
        };
        let publication_allowlist = {
            let p = root.join("data/docs-publication.json");
            if let Ok(s) = std::fs::read_to_string(&p) {
                serde_json::from_str::<serde_json::Value>(&s)
                    .ok()
                    .and_then(|v| {
                        v.get("allowlist")
                            .and_then(|a| a.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|x| x.as_str().map(String::from))
                                    .collect()
                            })
                    })
            } else {
                None
            }
        };
        let build_json = {
            let p = root.join("build.json");
            std::fs::read_to_string(&p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
        };

        Ok(AuditContext {
            source: SiteSource::LocalDir(root.to_path_buf()),
            canonical_routes,
            html_files,
            css,
            js_files,
            data_layer_dir,
            schemas_dir,
            manuscript,
            publication_allowlist,
            build_json,
        })
    }

    /// Look up a rendered HTML file by route.
    pub fn html(&self, route: &str) -> Option<&str> {
        self.html_files
            .iter()
            .find(|(r, _)| r == route)
            .map(|(_, h)| h.as_str())
    }

    /// Look up a JS file by relative path.
    pub fn js(&self, rel: &str) -> Option<&str> {
        self.js_files
            .iter()
            .find(|(r, _)| r == rel)
            .map(|(_, h)| h.as_str())
    }
}
