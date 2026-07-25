//! CSS aggregation — one file per component kind, in
//! deterministic order.
//!
//! The SSG reads the CSS files from `docs/styles/`, sorts them
//! by path, and emits a `site.css` bundle. The aggregation is
//! itself a projection: the input is the source files, the
//! output is the bundle, and the order is part of the
//! propagation test.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// The CSS aggregation. Holds the per-file contents and the
/// combined bundle. Built by `aggregate_css`.
#[derive(Debug)]
pub struct CssBundle {
    /// `(relative_path, contents)` pairs in deterministic order.
    pub files: Vec<(String, String)>,
    /// The combined bundle: each file's contents, separated by
    /// a `/* --- <path> --- */` marker.
    pub combined: String,
    /// The output directory for the per-component files
    /// (relative to the SSG output dir).
    pub subdir: PathBuf,
}

/// Walk the styles directory, read every `.css` file, and
/// produce a deterministic bundle.
///
/// Order: the `foundation/` files come first, in lexical
/// order. Then the `components/` files, in lexical order. The
/// `system/` files come last. The bundle is a concatenation
/// with a marker comment per file.
pub fn aggregate_css(styles_root: &Path) -> Result<CssBundle, CssError> {
    if !styles_root.exists() {
        return Err(CssError::StylesRootNotFound {
            path: styles_root.to_path_buf(),
        });
    }

    let mut foundation: Vec<PathBuf> = Vec::new();
    let mut components: Vec<PathBuf> = Vec::new();
    let mut system: Vec<PathBuf> = Vec::new();

    for entry in WalkDir::new(styles_root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("css") {
            continue;
        }
        let rel = path.strip_prefix(styles_root).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.starts_with("foundation/") {
            foundation.push(path.to_path_buf());
        } else if rel_str.starts_with("components/") {
            components.push(path.to_path_buf());
        } else if rel_str.starts_with("system/") {
            system.push(path.to_path_buf());
        }
    }

    foundation.sort();
    components.sort();
    system.sort();

    let mut all: Vec<PathBuf> = Vec::new();
    all.extend(foundation);
    all.extend(components);
    all.extend(system);

    let mut files: Vec<(String, String)> = Vec::new();
    let mut combined = String::new();
    for path in &all {
        let rel = path.strip_prefix(styles_root).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let contents = std::fs::read_to_string(path).map_err(|e| CssError::Read {
            path: path.clone(),
            source: e,
        })?;
        combined.push_str(&format!("\n/* --- {} --- */\n", rel_str));
        combined.push_str(&contents);
        combined.push('\n');
        files.push((rel_str, contents));
    }

    Ok(CssBundle {
        files,
        combined,
        subdir: PathBuf::from("styles"),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum CssError {
    #[error("styles root not found: {path}")]
    StylesRootNotFound { path: PathBuf },

    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
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

    #[test]
    fn aggregates_in_canonical_order() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write(&root, "components/z.css", "/* z */\n");
        write(&root, "foundation/tokens.css", ":root{}\n");
        write(&root, "components/a.css", "/* a */\n");
        write(&root, "system/dynamic.css", "/* dynamic */\n");
        let bundle = aggregate_css(root).unwrap();
        let names: Vec<&str> = bundle.files.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "foundation/tokens.css",
                "components/a.css",
                "components/z.css",
                "system/dynamic.css"
            ]
        );
    }

    #[test]
    fn combined_has_file_markers() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write(&root, "foundation/tokens.css", ":root{}\n");
        write(&root, "components/a.css", ".a {}\n");
        let bundle = aggregate_css(root).unwrap();
        assert!(bundle.combined.contains("/* --- foundation/tokens.css --- */"));
        assert!(bundle.combined.contains("/* --- components/a.css --- */"));
        assert!(bundle.combined.contains(":root{}"));
        assert!(bundle.combined.contains(".a {}"));
    }

    #[test]
    fn combined_is_deterministic() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write(&root, "foundation/tokens.css", ":root{}\n");
        write(&root, "components/a.css", ".a {}\n");
        let a = aggregate_css(root).unwrap().combined;
        let b = aggregate_css(root).unwrap().combined;
        assert_eq!(a, b);
    }

    #[test]
    fn missing_root_is_error() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("does-not-exist");
        let err = aggregate_css(&root).unwrap_err();
        assert!(matches!(err, CssError::StylesRootNotFound { .. }));
    }
}
