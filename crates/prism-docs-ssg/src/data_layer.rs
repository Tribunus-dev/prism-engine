//! `data_layer` — read and validate the v1 data layer against the
//! canonical JSON Schemas in `schemas/`.
//!
//! The data layer is the typed input the SSG reads in addition to
//! the content manifest. See `OBSERVATORY_V1_SPEC.md` §4.1 and
//! `docs/adr-033-observatory-schema-binding.md`.
//!
//! The module exposes:
//! - [`DataLayer`] — the validated, in-memory representation.
//! - [`DataLayer::load`] — read and validate the canonical
//!   twelve files from a given root directory.
//! - [`DataLayerError`] — typed errors with the failing file,
//!   schema, and the validation message.
//!
//! The data layer is the source of truth for status, evidence,
//! navigation, and configuration. The SSG composes its output
//! from both the content manifest (prose) and the data layer
//! (typed records).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The canonical twelve data files. Each entry maps a logical
/// name to (filename, schema filename).
const DATA_FILES: &[(&str, &str, &str)] = &[
    ("site", "site.json", "site.schema.json"),
    ("navigation", "navigation.json", "navigation.schema.json"),
    ("capabilities", "capabilities.json", "capability.schema.json"),
    ("capability_history", "capability-history.json", "capability-history.schema.json"),
    ("evidence_index", "evidence-index.json", "evidence.schema.json"),
    ("releases", "releases.json", "release.schema.json"),
    ("roadmap", "roadmap.json", "roadmap.schema.json"),
    ("models", "models.json", "models.schema.json"),
    ("docs_publication", "docs-publication.json", "docs-publication.schema.json"),
    ("search_index", "search-index.json", "search-index.schema.json"),
    ("observatory", "observatory.json", "observatory.schema.json"),
    ("architecture", "architecture.json", "architecture.schema.json"),
];

/// A validated data layer, ready to be projected onto the SSG's
/// output surfaces. Stored as `serde_json::Value` so the SSG can
/// read fields without taking a hard dependency on every shape;
/// the schema validation has already happened.
#[derive(Debug, Clone)]
pub struct DataLayer {
    /// The directory the data was loaded from.
    pub root: PathBuf,
    /// The schema directory the data was validated against.
    pub schema_dir: PathBuf,
    /// The data files, keyed by logical name (e.g., "capabilities").
    pub files: BTreeMap<String, DataFile>,
    /// The as_of_commit recorded in each top-level data file
    /// (every file in the v1 corpus carries this).
    pub as_of_commit: String,
    /// The recorded_at timestamp from `site.json` (used in the
    /// build identity).
    pub build_recorded_at: String,
}

/// A single data file, paired with its provenance.
#[derive(Debug, Clone)]
pub struct DataFile {
    pub logical_name: String,
    pub filename: String,
    pub schema_filename: String,
    pub path: PathBuf,
    pub value: serde_json::Value,
}

/// Errors that can occur while loading or validating the data
/// layer. All errors are typed; the SSG exits non-zero on any
/// error so CI fails loudly.
#[derive(Debug, Error)]
pub enum DataLayerError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("schema parse error at {path}: {source}")]
    SchemaParse {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("schema JSON parse error at {path}: {message}")]
    SchemaJsonParse { path: PathBuf, message: String },

    #[error("data parse error at {path}: {message}")]
    DataParse { path: PathBuf, message: String },

    #[error("validation failed for {data_file} against {schema_file}: {message}")]
    Validation {
        data_file: PathBuf,
        schema_file: PathBuf,
        message: String,
    },

    #[error("missing data file {filename} in {root}")]
    MissingDataFile { filename: String, root: PathBuf },

    #[error("missing schema file {filename} in {root}")]
    MissingSchemaFile { filename: String, root: PathBuf },
}

impl DataLayer {
    /// Load the canonical data layer from the given root. The
    /// root should contain a `data/` subdirectory with the
    /// twelve JSON files, and a sibling `schemas/` directory
    /// with the eight schema files. The function returns a
    /// fully-validated `DataLayer` or the first
    /// `DataLayerError` it encounters.
    pub fn load(data_root: &Path, schema_dir: &Path) -> Result<Self, DataLayerError> {
        let mut files = BTreeMap::new();
        let mut as_of_commit: Option<String> = None;
        let mut build_recorded_at: Option<String> = None;

        for (logical, filename, schema_filename) in DATA_FILES {
            let data_path = data_root.join(filename);
            let schema_path = schema_dir.join(schema_filename);

            if !data_path.exists() {
                return Err(DataLayerError::MissingDataFile {
                    filename: filename.to_string(),
                    root: data_root.to_path_buf(),
                });
            }
            if !schema_path.exists() {
                return Err(DataLayerError::MissingSchemaFile {
                    filename: schema_filename.to_string(),
                    root: schema_dir.to_path_buf(),
                });
            }

            // Read the data file as raw JSON.
            let data_text = fs::read_to_string(&data_path).map_err(|e| DataLayerError::Io {
                path: data_path.clone(),
                source: e,
            })?;
            let data_value: serde_json::Value =
                serde_json::from_str(&data_text).map_err(|e| DataLayerError::DataParse {
                    path: data_path.clone(),
                    message: e.to_string(),
                })?;

            // Read the schema file.
            let schema_text =
                fs::read_to_string(&schema_path).map_err(|e| DataLayerError::SchemaParse {
                    path: schema_path.clone(),
                    source: e,
                })?;
            let schema_value: serde_json::Value =
                serde_json::from_str(&schema_text).map_err(|e| {
                    DataLayerError::SchemaJsonParse {
                        path: schema_path.clone(),
                        message: e.to_string(),
                    }
                })?;

            // Validate. The schema is the canonical form for the
            // whole data file. Wrapper files (e.g., releases.json
            // wraps a "releases" array) carry a `properties` entry
            // for the wrapper key whose `items` field references
            // a `$defs`-level item shape. The schema validates
            // the whole value, including the items, in one pass.
            let validator = jsonschema::JSONSchema::compile(&schema_value).map_err(|e| {
                DataLayerError::Validation {
                    data_file: data_path.clone(),
                    schema_file: schema_path.clone(),
                    message: format!("schema is not a valid JSON Schema: {e}"),
                }
            })?;

            let data_for_validation = data_value.clone();
            let result = validator.validate(&data_for_validation);
            if let Err(errors) = result {
                let message = errors
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(DataLayerError::Validation {
                    data_file: data_path.clone(),
                    schema_file: schema_path.clone(),
                    message,
                });
            }

            // Pull as_of_commit and recorded_at for build identity.
            if let Some(s) = data_value.get("as_of_commit").and_then(|v| v.as_str()) {
                if logical == &"site" {
                    // site.json's as_of_commit is the inner build_identity.commit;
                    // prefer that.
                } else if as_of_commit.is_none() {
                    as_of_commit = Some(s.to_string());
                }
            }
            if logical == &"site" {
                if let Some(build) = data_value.get("build_identity") {
                    if let Some(c) = build.get("commit").and_then(|v| v.as_str()) {
                        as_of_commit = Some(c.to_string());
                    }
                    if let Some(t) = build.get("recorded_at").and_then(|v| v.as_str()) {
                        build_recorded_at = Some(t.to_string());
                    }
                }
            }

            files.insert(
                logical.to_string(),
                DataFile {
                    logical_name: logical.to_string(),
                    filename: filename.to_string(),
                    schema_filename: schema_filename.to_string(),
                    path: data_path,
                    value: data_value,
                },
            );
        }

        Ok(DataLayer {
            root: data_root.to_path_buf(),
            schema_dir: schema_dir.to_path_buf(),
            files,
            as_of_commit: as_of_commit.unwrap_or_default(),
            build_recorded_at: build_recorded_at.unwrap_or_default(),
        })
    }

    /// Get a data file by its logical name.
    pub fn get(&self, logical_name: &str) -> Option<&DataFile> {
        self.files.get(logical_name)
    }

    /// The list of logical names, in the canonical order.
    pub fn logical_names(&self) -> Vec<&'static str> {
        DATA_FILES.iter().map(|(n, _, _)| *n).collect()
    }
}

/// A small typed view of `site.json` for the build pipeline.
/// The full data layer is `serde_json::Value`; this struct is
/// the typed projection the build identity needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteSummary {
    #[serde(rename = "schema_version")]
    pub schema_version: String,
    #[serde(rename = "site_title")]
    pub site_title: String,
    #[serde(rename = "site_tagline")]
    pub site_tagline: String,
    #[serde(rename = "canonical_origin")]
    pub canonical_origin: String,
    #[serde(rename = "build_identity")]
    pub build_identity: BuildIdentity,
    #[serde(rename = "deployment_compatibility_window")]
    pub deployment_compatibility_window: DeploymentCompatibilityWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildIdentity {
    pub commit: String,
    #[serde(rename = "build_id")]
    pub build_id: String,
    #[serde(rename = "recorded_at")]
    pub recorded_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentCompatibilityWindow {
    #[serde(rename = "cutover_date")]
    pub cutover_date: String,
    pub reason: String,
}

impl DataFile {
    /// Parse this file's value as a `SiteSummary`. Returns
    /// `None` if the file is not `site.json`.
    pub fn as_site_summary(&self) -> Option<SiteSummary> {
        if self.logical_name != "site" {
            return None;
        }
        serde_json::from_value(self.value.clone()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn load_v1_data_layer_succeeds() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .to_path_buf();
        let data_root = base.join("docs/data");
        let schema_dir = base.join("schemas");

        let layer = DataLayer::load(&data_root, &schema_dir)
            .expect("v1 data layer should validate against the schemas");

        // The v1 corpus is small and well-defined.
        assert!(layer.get("capabilities").is_some());
        assert!(layer.get("evidence_index").is_some());
        assert!(layer.get("capability_history").is_some());
        assert!(layer.get("navigation").is_some());
        assert!(layer.get("site").is_some());
        assert!(layer.get("roadmap").is_some());
        assert!(layer.get("releases").is_some());
        assert!(layer.get("models").is_some());
        assert!(layer.get("architecture").is_some());
        assert!(layer.get("observatory").is_some());
        assert!(layer.get("docs_publication").is_some());
        assert!(layer.get("search_index").is_some());

        // The build identity was extracted.
        assert!(!layer.as_of_commit.is_empty());
    }

    #[test]
    fn site_summary_parses() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("repo root")
            .to_path_buf();
        let data_root = base.join("docs/data");
        let schema_dir = base.join("schemas");
        let layer = DataLayer::load(&data_root, &schema_dir).expect("load");
        let site = layer.get("site").expect("site.json");
        let summary = site.as_site_summary().expect("site summary");
        assert_eq!(summary.site_title, "Prism Engine");
        assert!(!summary.canonical_origin.is_empty());
    }

    #[test]
    fn missing_data_file_is_reported() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data_root = dir.path().join("data");
        std::fs::create_dir_all(&data_root).unwrap();
        let schema_dir = dir.path().join("schemas");
        std::fs::create_dir_all(&schema_dir).unwrap();
        let result = DataLayer::load(&data_root, &schema_dir);
        assert!(matches!(
            result,
            Err(DataLayerError::MissingDataFile { .. })
        ));
    }
}
