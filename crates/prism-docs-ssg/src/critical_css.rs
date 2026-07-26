//! Per-route critical CSS.
//!
//! Per `OBSERVATORY_V1_SPEC.md` §12 A18: "Critical CSS
//! per route ≤ 18 KB gzipped." The site has a single
//! 89KB raw `site.css` bundle that is loaded by every
//! page. The bundle is too large to fit the per-route
//! 18KB budget.
//!
//! The fix: extract the CSS rules needed to paint the
//! above-the-fold region of each route, inline them in
//! `<style>` in the page's `<head>`, and let the full
//! bundle load via `<link rel="stylesheet">` (cached
//! across pages). The browser paints with the inlined
//! critical CSS; the bundle refines and adds the rest.
//!
//! The critical CSS per route is the universal
//! foundation (tokens, typography, layout) + site-header
//! + hero + the route's primary component (status-table
//! on home and status, observatory on observatory/life,
//! etc.). The total is well under 18KB gzipped.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A loaded set of CSS files, keyed by their relative
/// path under the styles root. The order is preserved
/// (BTreeMap) so the concatenation is deterministic.
#[derive(Debug, Clone)]
pub struct CriticalCss {
    /// The styles root these files were loaded from.
    pub styles_root: PathBuf,
    /// `(relative_path, contents)` pairs in sorted order.
    pub files: BTreeMap<String, String>,
}

/// The set of files that are part of every page's
/// critical path. These are the always-needed rules
/// that paint the first viewport.
const UNIVERSAL: &[&str] = &[
    "foundation/tokens.css",
    "foundation/typography.css",
    "foundation/layout.css",
    "components/site-header.css",
    "components/hero.css",
    "components/chapter.css",
];

/// The per-route additions on top of `UNIVERSAL`. The
/// key is the canonical route. The value is the list of
/// component CSS files that are part of the critical
/// path on that route.
const PER_ROUTE: &[(&str, &[&str])] = &[
    ("/",                              &["components/status-table.css"]),
    ("/start/",                        &[]),
    ("/architecture/",                 &[]),
    ("/computeimage/",                 &["components/cimage.css"]),
    ("/computeimage/specimen/",        &["components/cimage.css"]),
    ("/evidence/",                     &[]),
    ("/status/",                       &["components/status-table.css"]),
    ("/lab/",                          &["components/lab-note.css"]),
    ("/observatory/life/",             &["components/observatory.css"]),
    ("/roadmap/",                      &["components/milestone.css"]),
    ("/run/",                          &[]),
    ("/colophon/",                     &[]),
];

impl CriticalCss {
    /// Load every CSS file under the styles root. The
    /// `CriticalCss` is held by the renderer for the
    /// duration of the build; reading once is enough.
    pub fn load(styles_root: &Path) -> Result<Self, CriticalCssError> {
        if !styles_root.exists() {
            return Err(CriticalCssError::StylesRootNotFound {
                path: styles_root.to_path_buf(),
            });
        }

        let mut files: BTreeMap<String, String> = BTreeMap::new();
        for entry in walkdir::WalkDir::new(styles_root)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("css") {
                continue;
            }
            let rel = path
                .strip_prefix(styles_root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            let contents = std::fs::read_to_string(path).map_err(|e| {
                CriticalCssError::Read {
                    path: path.display().to_string(),
                    source: e,
                }
            })?;
            files.insert(rel, contents);
        }

        Ok(CriticalCss {
            styles_root: styles_root.to_path_buf(),
            files,
        })
    }

    /// Build the critical CSS string for a given route.
    /// The result is the concatenation (in canonical
    /// order) of the universal files + the route's
    /// primary component files. The CSS is inlined
    /// verbatim — no minification, no rewriting.
    /// (Minification is a follow-on; the spec budget is
    /// gzipped size, and CSS gzip handles whitespace
    /// collapse for free.)
    pub fn for_route(&self, route: &str) -> String {
        let mut out = String::new();
        let mut files: Vec<&str> = UNIVERSAL.to_vec();
        for (r, extras) in PER_ROUTE {
            if *r == route {
                files.extend(extras.iter().copied());
                break;
            }
        }
        for f in files {
            if let Some(contents) = self.files.get(f) {
                out.push_str(contents);
                out.push('\n');
            }
            // A missing file is reported downstream by
            // the audit runner; here we silently skip so
            // the page can still render.
        }
        out
    }

    /// The list of files that make up the critical CSS
    /// for a given route. Used by the audit runner to
    /// report exactly what was inlined.
    pub fn files_for_route(&self, route: &str) -> Vec<String> {
        let mut out: Vec<String> = UNIVERSAL.iter().map(|s| s.to_string()).collect();
        for (r, extras) in PER_ROUTE {
            if *r == route {
                out.extend(extras.iter().map(|s| s.to_string()));
                break;
            }
        }
        out
    }

    /// The list of canonical routes the manifest
    /// understands. Used by the renderer to look up
    /// the critical CSS for a page; used by the audit
    /// runner to verify every route has a manifest
    /// entry.
    pub fn known_routes() -> Vec<&'static str> {
        PER_ROUTE.iter().map(|(r, _)| *r).collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CriticalCssError {
    #[error("styles root not found: {path}")]
    StylesRootNotFound { path: PathBuf },

    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(dir: &Path, rel: &str, contents: &str) {
        let path = dir.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    fn fixture() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        // The minimum set of files the loader expects.
        for f in UNIVERSAL {
            write(root, f, format!("/* {} */\n", f).as_str());
        }
        write(root, "components/cimage.css", "/* cimage */\n");
        write(root, "components/observatory.css", "/* observatory */\n");
        write(root, "components/status-table.css", "/* status-table */\n");
        write(root, "components/lab-note.css", "/* lab-note */\n");
        write(root, "components/milestone.css", "/* milestone */\n");
        write(root, "components/footer.css", "/* footer (not critical) */\n");
        tmp
    }

    #[test]
    fn load_reads_every_css_file() {
        let tmp = fixture();
        let css = CriticalCss::load(tmp.path()).unwrap();
        assert!(css.files.contains_key("foundation/tokens.css"));
        assert!(css.files.contains_key("components/cimage.css"));
        assert!(css.files.contains_key("components/footer.css"));
    }

    #[test]
    fn for_route_returns_universal_only_when_no_extras() {
        let tmp = fixture();
        let css = CriticalCss::load(tmp.path()).unwrap();
        let out = css.for_route("/start/");
        assert!(out.contains("foundation/tokens.css"));
        assert!(out.contains("foundation/typography.css"));
        assert!(out.contains("components/site-header.css"));
        assert!(out.contains("components/hero.css"));
        // No extras for /start/.
        assert!(!out.contains("cimage"));
        assert!(!out.contains("observatory"));
        assert!(!out.contains("status-table"));
    }

    #[test]
    fn for_route_adds_extras_for_observatory() {
        let tmp = fixture();
        let css = CriticalCss::load(tmp.path()).unwrap();
        let out = css.for_route("/observatory/life/");
        assert!(out.contains("observatory"));
        assert!(!out.contains("cimage"));
    }

    #[test]
    fn for_route_adds_cimage_for_computeimage() {
        let tmp = fixture();
        let css = CriticalCss::load(tmp.path()).unwrap();
        let out = css.for_route("/computeimage/specimen/");
        assert!(out.contains("cimage"));
        assert!(!out.contains("observatory"));
    }

    #[test]
    fn for_route_unknown_route_returns_universal() {
        let tmp = fixture();
        let css = CriticalCss::load(tmp.path()).unwrap();
        let out = css.for_route("/does-not-exist/");
        // The universal files are present; no extras.
        assert!(out.contains("foundation/tokens.css"));
        assert!(!out.contains("cimage"));
    }

    #[test]
    fn files_for_route_lists_inputs() {
        let tmp = fixture();
        let css = CriticalCss::load(tmp.path()).unwrap();
        let files = css.files_for_route("/observatory/life/");
        assert!(files.contains(&"components/observatory.css".to_string()));
        assert_eq!(files.len(), UNIVERSAL.len() + 1);
    }

    #[test]
    fn known_routes_returns_canonical() {
        let routes = CriticalCss::known_routes();
        assert!(routes.contains(&"/"));
        assert!(routes.contains(&"/observatory/life/"));
        assert!(routes.contains(&"/colophon/"));
    }

    #[test]
    fn missing_root_is_error() {
        let tmp = TempDir::new().unwrap();
        let err = CriticalCss::load(&tmp.path().join("nope")).unwrap_err();
        assert!(matches!(err, CriticalCssError::StylesRootNotFound { .. }));
    }
}
