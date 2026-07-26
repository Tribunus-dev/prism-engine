//! Styles aggregation — one file per CSS file in
//! deterministic order, plus a static-asset pass for
//! non-CSS files (fonts, images, anything else under
//! `docs/styles/`).
//!
//! The SSG reads every file under `docs/styles/`, sorts CSS
//! files by path, emits a `site.css` bundle, and copies
//! every other file (fonts, images) to the same `styles/`
//! output directory as-is. The aggregation is itself a
//! projection: the input is the source files, the output is
//! the bundle + assets, and the order is part of the
//! propagation test.
//!
//! Per spec §A19, the site serves its own font files; the
//! self-hosted Ubuntu TTFs live at `docs/styles/fonts/`. The
//! asset pass copies them to the output as-is so the live
//! site has no third-party requests.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

/// The styles aggregation. Holds the per-file CSS contents,
/// the combined bundle, and the list of non-CSS asset files
/// to copy through. Built by `aggregate_styles`.
#[derive(Debug)]
pub struct StylesBundle {
    /// `(relative_path, contents)` pairs for every CSS file,
    /// in deterministic order.
    pub css_files: Vec<(String, String)>,
    /// The combined CSS bundle: each file's contents,
    /// separated by a `/* --- <path> --- */` marker.
    pub combined: String,
    /// `(relative_path, source_path)` pairs for every
    /// non-CSS asset to copy through (fonts, images).
    /// `source_path` is the absolute path to the source file
    /// on disk; `relative_path` is its path under the styles
    /// root.
    pub assets: Vec<(String, PathBuf)>,
    /// The output directory for the per-component files
    /// (relative to the SSG output dir).
    pub subdir: PathBuf,
}

/// Walk the styles directory, classify every file, and
/// produce a deterministic CSS bundle plus a list of assets
/// to copy through.
///
/// Order: the `foundation/` CSS files come first, in lexical
/// order. Then the `components/` CSS files, in lexical
/// order. The `system/` CSS files come last. The CSS bundle
/// is a concatenation with a marker comment per file.
///
/// Non-CSS files (anything under `fonts/`, image files at
/// the root, etc.) are returned in the `assets` vector for
/// the SSG to copy through byte-for-byte.
pub fn aggregate_styles(styles_root: &Path) -> Result<StylesBundle, CssError> {
    if !styles_root.exists() {
        return Err(CssError::StylesRootNotFound {
            path: styles_root.to_path_buf(),
        });
    }

    let mut foundation: Vec<PathBuf> = Vec::new();
    let mut components: Vec<PathBuf> = Vec::new();
    let mut system: Vec<PathBuf> = Vec::new();
    let mut assets: Vec<PathBuf> = Vec::new();

    for entry in WalkDir::new(styles_root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let rel = path.strip_prefix(styles_root).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if path.extension().and_then(|s| s.to_str()) == Some("css") {
            if rel_str.starts_with("foundation/") {
                foundation.push(path.to_path_buf());
            } else if rel_str.starts_with("components/") {
                components.push(path.to_path_buf());
            } else if rel_str.starts_with("system/") {
                system.push(path.to_path_buf());
            } else {
                // Stray .css at the root; treat as a foundation
                // file. The taxonomy prefers explicit placement.
                foundation.push(path.to_path_buf());
            }
        } else {
            // Any non-CSS file under the styles tree is a
            // static asset. Fonts, images, etc. are copied
            // through.
            assets.push(path.to_path_buf());
        }
    }

    foundation.sort();
    components.sort();
    system.sort();
    assets.sort();

    let mut all_css: Vec<PathBuf> = Vec::new();
    all_css.extend(foundation);
    all_css.extend(components);
    all_css.extend(system);

    let mut css_files: Vec<(String, String)> = Vec::new();
    let mut combined = String::new();
    for path in &all_css {
        let rel = path.strip_prefix(styles_root).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let contents = std::fs::read_to_string(path).map_err(|e| CssError::Read {
            path: path.clone(),
            source: e,
        })?;
        combined.push_str(&format!("\n/* --- {} --- */\n", rel_str));
        combined.push_str(&contents);
        combined.push('\n');
        css_files.push((rel_str, contents));
    }

    let assets_out: Vec<(String, PathBuf)> = assets
        .into_iter()
        .map(|p| {
            let rel = p.strip_prefix(styles_root).unwrap_or(&p);
            (rel.to_string_lossy().replace('\\', "/"), p)
        })
        .collect();

    Ok(StylesBundle {
        css_files,
        combined,
        assets: assets_out,
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
        let bundle = aggregate_styles(root).unwrap();
        let names: Vec<&str> = bundle
            .css_files
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
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
        let bundle = aggregate_styles(root).unwrap();
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
        let a = aggregate_styles(root).unwrap().combined;
        let b = aggregate_styles(root).unwrap().combined;
        assert_eq!(a, b);
    }

    #[test]
    fn missing_root_is_error() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("does-not-exist");
        let err = aggregate_styles(&root).unwrap_err();
        assert!(matches!(err, CssError::StylesRootNotFound { .. }));
    }

    #[test]
    fn assets_are_collected_for_copy_through() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        write(&root, "foundation/tokens.css", ":root{}\n");
        write(&root, "fonts/Ubuntu-R.ttf", "FAKE_TTF_BYTES");
        write(&root, "components/a.css", ".a {}\n");
        write(&root, "fonts/UbuntuMono-R.ttf", "FAKE_MONO_BYTES");
        let bundle = aggregate_styles(root).unwrap();
        let names: Vec<&str> = bundle.assets.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["fonts/Ubuntu-R.ttf", "fonts/UbuntuMono-R.ttf"]);
    }
}
