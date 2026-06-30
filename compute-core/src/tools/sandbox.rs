//! Sandbox file-system tool implementations.
//!
//! All file operations are constrained to a sandbox root directory to prevent
//! path-traversal attacks. Paths are canonicalized and verified to be within
//! the sandbox before any read/write operation proceeds.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ── Constants ──────────────────────────────────────────────────────────────

/// Maximum bytes a single read_file call will return.
const MAX_READ_BYTES: u64 = 1_048_576; // 1 MB

/// Maximum number of search_file results.
const MAX_SEARCH_RESULTS: usize = 500;

/// Timeout for search_file operations.
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);

// ── Path resolution ────────────────────────────────────────────────────────

/// Resolve a path relative to the sandbox root.
///
/// Canonicalizes the joined path and verifies it falls within `root`.
/// Returns an error if:
/// - The path is empty or contains only whitespace
/// - The resolved path does not exist
/// - The resolved path is outside the sandbox root
pub fn resolve_sandbox_path(path_str: &str, root: &Path) -> Result<PathBuf, String> {
    let path_str = path_str.trim();
    if path_str.is_empty() {
        return Err("path is empty".into());
    }

    let root_canon = root
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize sandbox root: {e}"))?;

    // Join and canonicalize the requested path using the already-canonical root.
    // Joining a relative path onto a canonical root produces a non-canonical
    // path, so we canonicalize the result to resolve any symlinks or `..`.
    let joined = root_canon.join(path_str);
    let resolved = joined
        .canonicalize()
        .map_err(|e| format!("cannot resolve path '{path_str}': {e}"))?;

    // Ensure the resolved path is within the sandbox root.
    if !resolved.starts_with(&root_canon) {
        return Err(format!(
            "path '{path_str}' resolves outside the sandbox root"
        ));
    }

    Ok(resolved)
}

/// Resolve a path that may not exist yet (for write operations).
///
/// Validates the path is within the sandbox root **without requiring the file
/// to exist**. Sanitizes `..` components to prevent traversal, but does not
/// require canonicalization (which would fail on non-existent paths).
pub fn resolve_sandbox_path_relaxed(path_str: &str, root: &Path) -> Result<PathBuf, String> {
    let path_str = path_str.trim();
    if path_str.is_empty() {
        return Err("path is empty".into());
    }

    let root_canon = root
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize sandbox root: {e}"))?;

    // Join and normalize components to prevent traversal via `..`.
    let joined = root_canon.join(path_str);

    // Normalize the path using its components to strip `..` safely.
    let normalized = normalize_path(&joined);

    if !normalized.starts_with(&root_canon) {
        return Err(format!(
            "path '{path_str}' resolves outside the sandbox root"
        ));
    }

    Ok(normalized)
}

/// Strip `.` and resolve `..` components from a path without touching the
/// filesystem (no canonicalize).
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {
                // Skip `.`
            }
            std::path::Component::ParentDir => {
                // Pop the last component if available; don't pop the root.
                if components.len() > 1 {
                    components.pop();
                }
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

/// Get the sandbox root from an explicit parameter, the `PRISM_SANDBOX_ROOT`
/// environment variable, or the current working directory (in that order).
///
/// The returned path is canonicalized.
pub fn sandbox_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(ex) = explicit {
        if !ex.as_os_str().is_empty() {
            return ex
                .canonicalize()
                .unwrap_or_else(|_| ex.to_path_buf());
        }
    }

    if let Ok(env_root) = std::env::var("PRISM_SANDBOX_ROOT") {
        let p = PathBuf::from(env_root);
        if !p.as_os_str().is_empty() {
            return p.canonicalize().unwrap_or(p);
        }
    }

    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("/"))
}

// ── Error helper ───────────────────────────────────────────────────────────

/// Build a structured error JSON value.
fn err_json(code: &str, message: &str) -> serde_json::Value {
    serde_json::json!({
        "ok": false,
        "error": message,
        "code": code,
    })
}

// ── Binary / BOM / text helpers ────────────────────────────────────────────

/// Detect whether data is binary by scanning for null bytes in the first 8 KiB.
fn is_binary(data: &[u8]) -> bool {
    let scan_len = data.len().min(8192);
    data[..scan_len].contains(&0x00)
}

/// Detect a Byte Order Mark (BOM) and return the encoding name and number of
/// bytes to skip.
fn detect_bom(data: &[u8]) -> (&'static str, usize) {
    if data.len() >= 4 && data[..4] == [0x00, 0x00, 0xFE, 0xFF] {
        ("UTF-32BE", 4)
    } else if data.len() >= 4 && data[..4] == [0xFF, 0xFE, 0x00, 0x00] {
        ("UTF-32LE", 4)
    } else if data.len() >= 3 && data[..3] == [0xEF, 0xBB, 0xBF] {
        ("UTF-8", 3)
    } else if data.len() >= 2 && data[..2] == [0xFE, 0xFF] {
        ("UTF-16BE", 2)
    } else if data.len() >= 2 && data[..2] == [0xFF, 0xFE] {
        ("UTF-16LE", 2)
    } else {
        ("UTF-8", 0)
    }
}

/// Decode bytes to a String, stripping any BOM and replacing invalid sequences
/// with U+FFFD replacement characters.
fn decode_text(data: &[u8]) -> String {
    let (_encoding, skip) = detect_bom(data);
    let body = &data[skip..];
    String::from_utf8_lossy(body).into_owned()
}

// ── File walk helpers ──────────────────────────────────────────────────────

/// Recursively walk a directory collecting file paths.
///
/// - `base`: the sandbox root (used to compute relative paths)
/// - `dir`: the directory being walked (must be within `base`)
/// - `extension`: optional filter (e.g. `"rs"` to match `.rs` files)
/// - `matches`: output vector of relative paths
/// - `max`: maximum number of results to collect
fn walk_files(
    base: &Path,
    dir: &Path,
    extension: Option<&str>,
    matches: &mut Vec<String>,
    max: usize,
) -> Result<(), String> {
    if matches.len() >= max {
        return Ok(());
    }

    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read directory: {e}"))?;

    for entry in entries {
        if matches.len() >= max {
            break;
        }

        let entry = entry.map_err(|e| format!("cannot read entry: {e}"))?;
        let path = entry.path();

        if path.is_dir() {
            walk_files(base, &path, extension, matches, max)?;
        } else if path.is_file() {
            // Check extension filter.
            if let Some(ext) = extension {
                match path.extension() {
                    None => continue,
                    Some(e) if e != ext => continue,
                    _ => {}
                }
            }

            // Compute relative path.
            let rel = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            matches.push(rel);
        }
    }

    Ok(())
}

/// Recursively search files for a substring.
///
/// - `base`: the sandbox root (used to compute relative paths)
/// - `dir`: the directory being searched
/// - `pattern`: the substring to search for
/// - `extension`: optional file extension filter
/// - `results`: output vector of match objects (path, line, column, context)
/// - `total`: running count of matches found so far (checked against MAX)
/// - `start_time`: when the search began (checked against SEARCH_TIMEOUT)
fn search_in_files(
    base: &Path,
    dir: &Path,
    pattern: &str,
    extension: Option<&str>,
    results: &mut Vec<serde_json::Value>,
    total: &mut usize,
    start_time: &Instant,
) -> Result<(), String> {
    if start_time.elapsed() > SEARCH_TIMEOUT || *total >= MAX_SEARCH_RESULTS {
        return Ok(());
    }

    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read directory: {e}"))?;

    for entry in entries {
        if start_time.elapsed() > SEARCH_TIMEOUT || *total >= MAX_SEARCH_RESULTS {
            break;
        }

        let entry = entry.map_err(|e| format!("cannot read entry: {e}"))?;
        let path = entry.path();

        if path.is_dir() {
            search_in_files(base, &path, pattern, extension, results, total, start_time)?;
        } else if path.is_file() {
            // Check extension filter.
            if let Some(ext) = extension {
                match path.extension() {
                    None => continue,
                    Some(e) if e != ext => continue,
                    _ => {}
                }
            }

            // Compute relative path.
            let rel = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();

            // Read file content (limit to 1 MB per file for search).
            let data = match fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };

            if data.len() > MAX_READ_BYTES as usize {
                continue; // Skip files larger than 1 MB to avoid memory issues.
            }

            if is_binary(&data) {
                continue; // Skip binary files.
            }

            let content = decode_text(&data);

            for (line_idx, line) in content.lines().enumerate() {
                if *total >= MAX_SEARCH_RESULTS {
                    break;
                }
                if start_time.elapsed() > SEARCH_TIMEOUT {
                    break;
                }

                if let Some(col) = line.find(pattern) {
                    // Build context: a trimmed snippet around the match.
                    let context = build_context(line, col, pattern.len());

                    results.push(serde_json::json!({
                        "path": rel,
                        "line": line_idx + 1,
                        "column": col + 1,
                        "context": context,
                    }));
                    *total += 1;
                }
            }
        }
    }

    Ok(())
}

/// Build a context snippet around a match within a line.
fn build_context(line: &str, col: usize, match_len: usize) -> String {
    let line_len = line.len();
    let ctx_radius = 40;

    let start = col.saturating_sub(ctx_radius);
    let end = (col + match_len + ctx_radius).min(line_len);

    let mut snippet = String::new();
    if start > 0 {
        snippet.push_str("…");
    }
    snippet.push_str(&line[start..end]);
    if end < line_len {
        snippet.push_str("…");
    }
    snippet
}

// ── Tool implementations ───────────────────────────────────────────────────

/// Read the full contents of a text file.
///
/// Detects binary content (null byte), detects BOM, truncates at 1 MB.
/// Returns `{ ok: true, path, content, line_count, truncated, encoding }`.
pub fn tool_read_file(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let path_str = match args.get("path").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'path'"),
    };

    let resolved = match resolve_sandbox_path(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    let data = match fs::read(&resolved) {
        Ok(d) => d,
        Err(e) => return err_json("read_error", &format!("cannot read file: {e}")),
    };

    // Check for binary content.
    if is_binary(&data) {
        return serde_json::json!({
            "ok": true,
            "path": path_str,
            "binary": true,
            "size": data.len(),
        });
    }

    let truncated = data.len() > MAX_READ_BYTES as usize;
    let body = if truncated {
        &data[..MAX_READ_BYTES as usize]
    } else {
        &data[..]
    };

    let (encoding, _skip) = detect_bom(body);
    let content = decode_text(body);
    let line_count = content.lines().count();

    serde_json::json!({
        "ok": true,
        "path": path_str,
        "content": content,
        "line_count": line_count,
        "truncated": truncated,
        "encoding": encoding,
    })
}

/// Read a specific range of lines from a text file (1-indexed).
///
/// Returns `{ ok, path, lines: [{ line_number, content }], total_lines }`.
pub fn tool_read_file_lines(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let path_str = match args.get("path").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'path'"),
    };

    let resolved = match resolve_sandbox_path(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    let data = match fs::read(&resolved) {
        Ok(d) => d,
        Err(e) => return err_json("read_error", &format!("cannot read file: {e}")),
    };

    if is_binary(&data) {
        return serde_json::json!({
            "ok": true,
            "path": path_str,
            "binary": true,
            "size": data.len(),
        });
    }

    let content = decode_text(&data);
    let all_lines: Vec<&str> = content.lines().collect();
    let total_lines = all_lines.len();

    // Parse optional start_line / end_line (1-indexed).
    let start_line = args
        .get("start_line")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        .max(1) as usize;
    let end_line = args
        .get("end_line")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(total_lines);

    if start_line > total_lines {
        return serde_json::json!({
            "ok": true,
            "path": path_str,
            "lines": [],
            "total_lines": total_lines,
            "start_line": start_line,
            "end_line": end_line,
            "note": "start_line beyond file length",
        });
    }

    let end_line = end_line.min(total_lines);
    let selected: Vec<serde_json::Value> = all_lines[start_line - 1..end_line]
        .iter()
        .enumerate()
        .map(|(i, line)| {
            serde_json::json!({
                "line_number": start_line + i,
                "content": line,
            })
        })
        .collect();

    serde_json::json!({
        "ok": true,
        "path": path_str,
        "lines": selected,
        "total_lines": total_lines,
        "start_line": start_line,
        "end_line": end_line,
    })
}

/// Write content to a file. Atomic write via temp file + rename.
///
/// Creates parent directories as needed. Returns `{ ok, path, bytes_written }`.
pub fn tool_write_file(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let path_str = match args.get("path").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'path'"),
    };

    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'content'"),
    };

    let resolved = match resolve_sandbox_path_relaxed(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    // Create parent directories.
    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                return err_json("write_error", &format!("cannot create parent directories: {e}"));
            }
        }
    }

    // Atomic write: write to a temp file in the same directory, then rename.
    let dir = resolved.parent().unwrap_or(root);
    let bytes = content.as_bytes();

    let temp_result = (|| -> io::Result<()> {
        // Use tempfile crate for secure temp file creation.
        let mut tmp_file = tempfile::NamedTempFile::new_in(dir)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        tmp_file.write_all(bytes)?;
        tmp_file.flush()?;
        // Persist by renaming over the target.
        tmp_file.persist(&resolved)?;
        Ok(())
    })();

    match temp_result {
        Ok(_) => serde_json::json!({
            "ok": true,
            "path": path_str,
            "bytes_written": bytes.len(),
        }),
        Err(e) => err_json("write_error", &format!("cannot write file: {e}")),
    }
}

/// Find and replace occurrences of `old_text` with `new_text` in a file.
///
/// All occurrences are replaced (not just the first). Reports the affected
/// line numbers.
/// Returns `{ ok, path, replacements, affected_lines }`.
pub fn tool_edit_file(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let path_str = match args.get("path").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'path'"),
    };

    let old_text = match args.get("old_text").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'old_text'"),
    };

    let new_text = args
        .get("new_text")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if old_text.is_empty() {
        return err_json("invalid_param", "'old_text' must not be empty");
    }

    let resolved = match resolve_sandbox_path(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    let data = match fs::read(&resolved) {
        Ok(d) => d,
        Err(e) => return err_json("read_error", &format!("cannot read file: {e}")),
    };

    if is_binary(&data) {
        return err_json("binary_file", "cannot edit binary file");
    }

    let original = decode_text(&data);

    // Count occurrences and find affected lines before replacing.
    let mut replacements = 0usize;
    let mut affected_lines_set: Vec<usize> = Vec::new();

    // Scan for old_text to find affected lines.
    if !old_text.is_empty() {
        for (line_idx, line) in original.lines().enumerate() {
            if line.contains(old_text) {
                affected_lines_set.push(line_idx + 1);
            }
        }
    }

    // Perform the replacement (all occurrences).
    let modified = original.replace(old_text, new_text);

    // Count how many replacements were made.
    if old_text != new_text {
        let orig_len = original.len();
        let mod_len = modified.len();
        if old_text.len() > new_text.len() {
            // Text was removed — count by difference.
            let diff = orig_len - mod_len;
            if old_text.len() > new_text.len() {
                let per = old_text.len() - new_text.len();
                if per > 0 {
                    replacements = diff / per;
                }
            }
        } else if old_text.len() < new_text.len() {
            // Text was added — count by difference.
            let diff = mod_len - orig_len;
            let per = new_text.len() - old_text.len();
            if per > 0 {
                replacements = diff / per;
            }
        }
        // When lengths are equal, we approximate by looking at a simple count.
        // This is reliable for the common case.
        if replacements == 0 && old_text.len() == new_text.len() {
            // Count occurrences of old_text in the original.
            let mut count = 0;
            let mut pos = 0;
            while let Some(found) = original[pos..].find(old_text) {
                count += 1;
                pos += found + old_text.len();
                if pos > original.len() {
                    break;
                }
            }
            replacements = count;
        }
    }

    // Write modified content back atomically.
    let dir = resolved.parent().unwrap_or(root);
    let write_result = (|| -> io::Result<()> {
        let mut tmp_file = tempfile::NamedTempFile::new_in(dir)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        tmp_file.write_all(modified.as_bytes())?;
        tmp_file.flush()?;
        tmp_file.persist(&resolved)?;
        Ok(())
    })();

    match write_result {
        Ok(()) => {
            // Deduplicate and sort affected lines.
            affected_lines_set.sort_unstable();
            affected_lines_set.dedup();

            serde_json::json!({
                "ok": true,
                "path": path_str,
                "replacements": replacements,
                "affected_lines": affected_lines_set,
            })
        }
        Err(e) => err_json("write_error", &format!("cannot write edited file: {e}")),
    }
}

/// List files and directories at the given path, sorted by name.
///
/// Accepts optional `include_hidden` boolean (default false).
/// Returns `{ ok, path, entries: [{ name, type, size, modified }] }`.
pub fn tool_list_directory(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let path_str = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let include_hidden = args
        .get("include_hidden")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let resolved = match resolve_sandbox_path(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    if !resolved.is_dir() {
        return err_json("not_directory", "path is not a directory");
    }

    let entries = match fs::read_dir(&resolved) {
        Ok(e) => e,
        Err(e) => return err_json("read_error", &format!("cannot read directory: {e}")),
    };

    let mut result_entries: Vec<serde_json::Value> = Vec::new();

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let file_name = entry.file_name();
        let name = file_name.to_string_lossy().into_owned();

        // Filter hidden files (starting with `.`).
        if !include_hidden && name.starts_with('.') {
            continue;
        }

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let file_type = if metadata.is_dir() {
            "directory"
        } else if metadata.is_symlink() {
            "symlink"
        } else {
            "file"
        };

        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| d.as_secs() as i64)
            });

        result_entries.push(serde_json::json!({
            "name": name,
            "type": file_type,
            "size": metadata.len(),
            "modified": modified,
        }));
    }

    // Sort by name (case-insensitive).
    result_entries.sort_by(|a, b| {
        let a_name = a["name"].as_str().unwrap_or("");
        let b_name = b["name"].as_str().unwrap_or("");
        a_name.to_lowercase().cmp(&b_name.to_lowercase())
    });

    serde_json::json!({
        "ok": true,
        "path": path_str,
        "entries": result_entries,
    })
}

/// Recursively find files by extension.
///
/// Accepts optional `path` (subdirectory), `extension` filter, and `max_results`.
/// Returns `{ ok, files: [...] }` with sorted relative paths.
pub fn tool_glob_files(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let path_str = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let extension = args.get("extension").and_then(|v| v.as_str());

    // Normalize extension: strip leading dot if present.
    let ext = extension.map(|e| {
        if let Some(stripped) = e.strip_prefix('.') {
            stripped
        } else {
            e
        }
    });

    let max_results = args
        .get("max_results")
        .and_then(|v| v.as_i64())
        .map(|v| v as usize)
        .unwrap_or(10_000);

    let resolved = match resolve_sandbox_path(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    if !resolved.is_dir() {
        return err_json("not_directory", "path is not a directory");
    }

    let mut matches: Vec<String> = Vec::new();
    if let Err(e) = walk_files(root, &resolved, ext, &mut matches, max_results) {
        return err_json("walk_error", &e);
    }

    // Sort relative paths.
    matches.sort();

    serde_json::json!({
        "ok": true,
        "files": matches,
    })
}

/// Search for a substring in files within the sandbox.
///
/// Has a 30-second timeout and a 500-result cap.
/// Accepts optional `path` (subdirectory), `pattern` (required), and `extension` filter.
/// Returns `{ ok, results: [{ path, line, column, context }], truncated, timed_out }`.
pub fn tool_search_files(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'pattern'"),
    };

    if pattern.is_empty() {
        return err_json("invalid_param", "'pattern' must not be empty");
    }

    let path_str = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let extension = args.get("extension").and_then(|v| v.as_str());

    // Normalize extension.
    let ext = extension.map(|e| {
        if let Some(stripped) = e.strip_prefix('.') {
            stripped
        } else {
            e
        }
    });

    let resolved = match resolve_sandbox_path(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    if !resolved.is_dir() {
        return err_json("not_directory", "path is not a directory");
    }

    let mut results: Vec<serde_json::Value> = Vec::new();
    let mut total = 0usize;
    let start_time = Instant::now();

    if let Err(e) = search_in_files(
        root,
        &resolved,
        pattern,
        ext,
        &mut results,
        &mut total,
        &start_time,
    ) {
        return err_json("search_error", &e);
    }

    let timed_out = start_time.elapsed() > SEARCH_TIMEOUT;
    let truncated = total >= MAX_SEARCH_RESULTS;

    serde_json::json!({
        "ok": true,
        "results": results,
        "total_matches": total,
        "truncated": truncated,
        "timed_out": timed_out,
        "search_time_ms": start_time.elapsed().as_millis(),
    })
}

/// Get metadata about a file or directory.
///
/// Returns `{ ok, path, size, type, permissions, modified, created, accessed }`.
pub fn tool_file_info(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let path_str = match args.get("path").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return err_json("missing_param", "missing required parameter 'path'"),
    };

    let resolved = match resolve_sandbox_path(path_str, root) {
        Ok(p) => p,
        Err(e) => return err_json("resolution_error", &e),
    };

    let metadata = match fs::metadata(&resolved) {
        Ok(m) => m,
        Err(e) => return err_json("stat_error", &format!("cannot stat path: {e}")),
    };

    let file_type = if metadata.is_dir() {
        "directory"
    } else if metadata.is_symlink() {
        "symlink"
    } else {
        "file"
    };

    let duration_to_epoch = |t: std::time::SystemTime| -> Option<i64> {
        t.duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs() as i64)
    };

    let symlink_target = if metadata.is_symlink() {
        fs::read_link(&resolved)
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
    } else {
        None
    };

    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode();
        Some(mode)
    };

    #[cfg(not(unix))]
    let permissions: Option<u32> = None;

    serde_json::json!({
        "ok": true,
        "path": path_str,
        "size": metadata.len(),
        "type": file_type,
        "permissions": permissions,
        "modified": duration_to_epoch(metadata.modified().unwrap_or(std::time::UNIX_EPOCH)),
        "created": metadata.created().ok().and_then(duration_to_epoch),
        "accessed": metadata.accessed().ok().and_then(duration_to_epoch),
        "symlink_target": symlink_target,
        "readonly": metadata.permissions().readonly(),
    })
}
