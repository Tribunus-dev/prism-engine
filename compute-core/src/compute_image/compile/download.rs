//! HuggingFace source downloading — parse HF identifiers, download shards
//! and config files from the HF Hub.

use std::path::{Path, PathBuf};

// ═══════════════════════════════════════════════════════════════════════════
// HuggingFace source downloading
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a HuggingFace source string ("hf:org/model" or "hf:org/model@revision")
/// and return (hub_id, revision).
pub fn parse_hf_source(source: &str) -> Option<(&str, &str)> {
    let source = source.strip_prefix("hf:")?;
    let parts: Vec<&str> = source.splitn(2, '@').collect();
    let hub_id = parts[0];
    let revision = parts.get(1).copied().unwrap_or("main");
    Some((hub_id, revision))
}

/// Download a single file from HuggingFace Hub to a destination directory.
pub(crate) fn download_hf_file(
    hub_id: &str,
    filename: &str,
    revision: &str,
    dest_dir: &Path,
    hf_token: Option<&str>,
) -> crate::Result<PathBuf> {
    let dest = dest_dir.join(filename);

    // Ensure destination parent exists
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            crate::Error::from_reason(format!("create directory {}: {e}", parent.display()))
        })?;
    }

    // Build the HF API client
    let token: Option<String> = hf_token
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .or_else(|| std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()));
    let builder = hf_hub::api::sync::ApiBuilder::new();
    let api = builder
        .with_token(token)
        .build()
        .map_err(|e| crate::Error::from_reason(format!("HF API init: {e}")))?;

    // Download via hf-hub (uses ~/.cache/huggingface as backing store)
    let model = api.model(hub_id.to_string());
    let cached_path = model.get(filename).map_err(|e| {
        crate::Error::from_reason(format!("hf download {hub_id}/{filename}@{revision}: {e}"))
    })?;

    // Hardlink or copy from HF cache to our dest_dir
    std::fs::hard_link(&cached_path, &dest)
        .or_else(|_| std::fs::copy(&cached_path, &dest).map(|_| ()))
        .map_err(|e| {
            crate::Error::from_reason(format!(
                "link/copy {} -> {}: {e}",
                cached_path.display(),
                dest.display()
            ))
        })?;

    Ok(dest)
}

/// Parse the safetensors index to get the list of shard files.
pub(crate) fn fetch_shard_list(
    hub_id: &str,
    revision: &str,
    temp_dir: &Path,
    hf_token: Option<&str>,
) -> crate::Result<Vec<String>> {
    // Download the safetensors index file if not already present
    let index_filename = "model.safetensors.index.json";
    let index_path = temp_dir.join(index_filename);
    if !index_path.exists() {
        download_hf_file(hub_id, index_filename, revision, temp_dir, hf_token)?;
    }

    let index_text = std::fs::read_to_string(&index_path)
        .map_err(|e| crate::Error::from_reason(format!("read index: {e}")))?;
    let index: serde_json::Value = serde_json::from_str(&index_text)
        .map_err(|e| crate::Error::from_reason(format!("parse index: {e}")))?;

    // Collect unique shard filenames from weight_map
    use std::collections::BTreeSet;
    let shards: BTreeSet<String> = index["weight_map"]
        .as_object()
        .map(|m| {
            m.values()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    Ok(shards.into_iter().collect())
}

/// Download config.json, tokenizer files, and all safetensors shards
/// from HuggingFace Hub to the destination directory.
pub fn download_hf_model(
    hub_id: &str,
    revision: &str,
    dest_dir: &Path,
    hf_token: Option<&str>,
) -> crate::Result<()> {
    // 1. Download config.json first (required for architecture plan)
    download_hf_file(hub_id, "config.json", revision, dest_dir, hf_token)?;

    // 2. Download tokenizer files
    for name in &["tokenizer.json", "tokenizer_config.json"] {
        let _ = download_hf_file(hub_id, name, revision, dest_dir, hf_token);
    }

    // 3. Download auxiliary files
    for name in &[
        "generation_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ] {
        let _ = download_hf_file(hub_id, name, revision, dest_dir, hf_token);
    }

    // 4. Fetch the safetensors index to discover all shard filenames.
    let shard_list = match fetch_shard_list(hub_id, revision, dest_dir, hf_token) {
        Ok(shards) if !shards.is_empty() => shards,
        // No index — try downloading a single model.safetensors file
        _ => {
            let _ = download_hf_file(hub_id, "model.safetensors", revision, dest_dir, hf_token);
            return Ok(());
        }
    };

    // 5. Download each safetensors shard one at a time (streaming).
    for shard_name in &shard_list {
        download_hf_file(hub_id, shard_name, revision, dest_dir, hf_token)?;
    }

    Ok(())
}
