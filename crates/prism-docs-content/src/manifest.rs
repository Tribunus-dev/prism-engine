//! `ContentManifest` — the canonical registry of all entities on the
//! site.
//!
//! The manifest is the durable source of truth. It is loaded once at
//! SSG start, validated, then turned into a typed world. The world
//! owns the live state; the manifest is read-only after that.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::adr::Adr;
use crate::chapter::Chapter;
use crate::claim::Claim;
use crate::error::ContentError;
use crate::link::Link;
use crate::page::Page;

/// A stable, typed entity id.
///
/// Format: `<kind>:<slug>`. The kind prefix is one of `chapter`, `adr`,
/// `claim`, `page`. The slug is lowercase, kebab-case, and must be
/// unique within the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntityId(String);

impl EntityId {
    pub fn new(raw: impl Into<String>) -> Result<Self, ContentError> {
        let s: String = raw.into();
        if s.is_empty() {
            return Err(ContentError::InvalidEntityId {
                id: s,
                reason: "id must be non-empty".into(),
            });
        }
        if !s.contains(':') {
            return Err(ContentError::InvalidEntityId {
                id: s,
                reason: "id must be of the form `<kind>:<slug>`".into(),
            });
        }
        let (kind, slug) = s.split_once(':').unwrap_or(("", ""));
        if kind.is_empty() || slug.is_empty() {
            return Err(ContentError::InvalidEntityId {
                id: s,
                reason: "kind and slug must both be non-empty".into(),
            });
        }
        if !slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        {
            return Err(ContentError::InvalidEntityId {
                id: s,
                reason: "slug must be kebab-case (lowercase, digits, `-`, `_`)".into(),
            });
        }
        Ok(Self(s))
    }

    pub fn kind(&self) -> &str {
        self.0.split(':').next().unwrap_or("")
    }

    pub fn slug(&self) -> &str {
        self.0.split(':').nth(1).unwrap_or("")
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Discriminator for the entity kinds the manifest knows about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityKind {
    Chapter,
    Adr,
    Claim,
    Page,
    Link,
}

impl EntityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EntityKind::Chapter => "chapter",
            EntityKind::Adr => "adr",
            EntityKind::Claim => "claim",
            EntityKind::Page => "page",
            EntityKind::Link => "link",
        }
    }
}

/// One entry in the raw manifest, before resolution. The `kind` field
/// is required so the typed loader can dispatch to the right validator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEntry {
    pub id: EntityId,
    pub kind: EntityKind,
    /// Raw payload. The typed loader validates and converts this.
    #[serde(flatten)]
    pub payload: serde_json::Value,
}

/// The on-disk manifest, loaded directly from TOML.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawManifest {
    pub entity: Vec<EntityEntry>,
}

/// Result of loading and validating a manifest. This is the typed view
/// the runtime consumes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentManifest {
    pub chapters: Vec<Chapter>,
    pub adrs: Vec<Adr>,
    pub claims: Vec<Claim>,
    pub pages: Vec<Page>,
    pub links: Vec<Link>,
    /// Path of the content root, for resolving body paths.
    pub content_root: PathBuf,
}

impl ContentManifest {
    /// All known entity ids, for quick lookup during link resolution.
    pub fn known_ids(&self) -> impl Iterator<Item = &EntityId> {
        self.chapters
            .iter()
            .map(|c| &c.id)
            .chain(self.adrs.iter().map(|a| &a.id))
            .chain(self.claims.iter().map(|c| &c.id))
            .chain(self.pages.iter().map(|p| &p.id))
    }

    pub fn chapter(&self, id: &EntityId) -> Option<&Chapter> {
        self.chapters.iter().find(|c| &c.id == id)
    }

    pub fn adr(&self, id: &EntityId) -> Option<&Adr> {
        self.adrs.iter().find(|a| &a.id == id)
    }

    pub fn claim(&self, id: &EntityId) -> Option<&Claim> {
        self.claims.iter().find(|c| &c.id == id)
    }

    pub fn page(&self, id: &EntityId) -> Option<&Page> {
        self.pages.iter().find(|p| &p.id == id)
    }
}

/// A typed view of loading a manifest. Used by `prism-docs-runtime` to
/// surface whether the load was clean, what was loaded, and any
/// validation warnings (currently always fatal — the build refuses
/// partial manifests).
#[derive(Debug, Clone)]
pub struct ManifestLoad {
    pub manifest: ContentManifest,
    pub entity_count: usize,
    pub link_count: usize,
}

/// Load a manifest from a TOML file. The path of the file is the
/// content root for body-path resolution.
pub fn load_manifest(path: &Path) -> Result<ManifestLoad, ContentError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| ContentError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let raw: RawManifest = toml::from_str(&raw_text).map_err(|e| ContentError::ManifestParse {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let content_root = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    build_manifest(raw, &content_root)
}

/// Build a typed manifest from a raw one. Validates every entity and
/// every link. Returns the typed manifest or the first error.
pub fn build_manifest(
    raw: RawManifest,
    content_root: &Path,
) -> Result<ManifestLoad, ContentError> {
    let mut manifest = ContentManifest {
        content_root: content_root.to_path_buf(),
        ..Default::default()
    };

    // First pass: collect entities by id, fail on duplicates.
    let mut seen: BTreeMap<EntityId, EntityKind> = BTreeMap::new();
    for entry in &raw.entity {
        if seen.contains_key(&entry.id) {
            return Err(ContentError::DuplicateEntity { id: entry.id.clone() });
        }
        seen.insert(entry.id.clone(), entry.kind);
    }

    // Second pass: build typed entities.
    for entry in &raw.entity {
        match entry.kind {
            EntityKind::Chapter => {
                let mut payload = entry.payload.clone();
                inject_id(&mut payload, &entry.id);
                let chapter: Chapter = serde_json::from_value(payload).map_err(|e| {
                    ContentError::ManifestParse {
                        path: content_root.join("chapter"),
                        message: e.to_string(),
                    }
                })?;
                chapter.validate()?;
                manifest.chapters.push(chapter);
            }
            EntityKind::Adr => {
                let mut payload = entry.payload.clone();
                inject_id(&mut payload, &entry.id);
                let adr: Adr = serde_json::from_value(payload).map_err(|e| {
                    ContentError::ManifestParse {
                        path: content_root.join("adr"),
                        message: e.to_string(),
                    }
                })?;
                adr.validate()?;
                manifest.adrs.push(adr);
            }
            EntityKind::Claim => {
                let mut payload = entry.payload.clone();
                inject_id(&mut payload, &entry.id);
                let claim: Claim = serde_json::from_value(payload).map_err(|e| {
                    ContentError::ManifestParse {
                        path: content_root.join("claim"),
                        message: e.to_string(),
                    }
                })?;
                claim.validate()?;
                manifest.claims.push(claim);
            }
            EntityKind::Page => {
                let mut payload = entry.payload.clone();
                inject_id(&mut payload, &entry.id);
                let page: Page = serde_json::from_value(payload).map_err(|e| {
                    ContentError::ManifestParse {
                        path: content_root.join("page"),
                        message: e.to_string(),
                    }
                })?;
                page.validate()?;
                manifest.pages.push(page);
            }
            EntityKind::Link => {
                let mut payload = entry.payload.clone();
                inject_id(&mut payload, &entry.id);
                // Links also need a `from` injected by the resolver. The
                // manifest treats the link as a directed edge: the
                // entity entry's id IS the link id; the source of the
                // link must come from the payload's `from` field.
                if let serde_json::Value::Object(map) = &mut payload {
                    if !map.contains_key("from") {
                        // A link without an explicit `from` is
                        // self-anchored — useful for type-level
                        // references from the source entity to the
                        // target entity. The link's id is the source
                        // (entity ids in this case are
                        // `link:<from>-<to>`-shaped; we keep the
                        // existing `from` semantics by parsing it).
                        // We refuse to fabricate one: error out.
                        return Err(ContentError::InvalidValue {
                            id: entry.id.clone(),
                            component: "from".into(),
                            reason: "link entity must declare `from`".into(),
                        });
                    }
                }
                let link: Link = serde_json::from_value(payload).map_err(|e| {
                    ContentError::ManifestParse {
                        path: content_root.join("link"),
                        message: e.to_string(),
                    }
                })?;
                link.validate()?;
                manifest.links.push(link);
            }
        }
    }

    // Third pass: resolve all link targets.
    let known: BTreeMap<&EntityId, ()> =
        manifest.known_ids().map(|id| (id, ())).collect();
    for link in &manifest.links {
        if !known.contains_key(&link.to) {
            return Err(ContentError::BrokenLink {
                from: link.from.clone(),
                to: link.to.clone(),
            });
        }
    }

    // Fourth pass: detect cycles in `follows` links.
    detect_follow_cycles(&manifest)?;

    let entity_count = manifest.chapters.len()
        + manifest.adrs.len()
        + manifest.claims.len()
        + manifest.pages.len();
    let link_count = manifest.links.len();
    Ok(ManifestLoad {
        manifest,
        entity_count,
        link_count,
    })
}

fn detect_follow_cycles(manifest: &ContentManifest) -> Result<(), ContentError> {
    use crate::link::LinkKind;
    let follows: BTreeMap<&EntityId, &EntityId> = manifest
        .links
        .iter()
        .filter(|l| l.kind == LinkKind::Follows)
        .map(|l| (&l.from, &l.to))
        .collect();
    for start in follows.keys() {
        let mut chain = vec![(*start).clone()];
        let mut cursor: &EntityId = start;
        while let Some(next) = follows.get(cursor) {
            if chain.iter().any(|c| c == *next) {
                return Err(ContentError::CyclicLink {
                    start: (*start).clone(),
                    chain,
                });
            }
            chain.push((*next).clone());
            cursor = next;
        }
    }
    Ok(())
}

fn inject_id(payload: &mut serde_json::Value, id: &EntityId) {
    if let serde_json::Value::Object(map) = payload {
        map.insert("id".into(), serde_json::Value::String(id.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::ClaimClass;
    use crate::ontology::KnowledgeState;

    fn chapter_entry(id: &str, title: &str) -> EntityEntry {
        let payload = serde_json::json!({
            "slug": "test",
            "title": title,
            "order": 1,
            "intent": "an intent",
            "body_path": "chapters/test.md"
        });
        EntityEntry {
            id: EntityId::new(id).unwrap(),
            kind: EntityKind::Chapter,
            payload,
        }
    }

    fn claim_entry(id: &str, text: &str) -> EntityEntry {
        let payload = serde_json::json!({
            "text": text,
            "class": ClaimClass::Architectural,
            "state": KnowledgeState::Verified,
        });
        EntityEntry {
            id: EntityId::new(id).unwrap(),
            kind: EntityKind::Claim,
            payload,
        }
    }

    #[test]
    fn entity_id_validation() {
        assert!(EntityId::new("chapter:home-intent").is_ok());
        assert!(EntityId::new("adr:003-canonical-ecs-world").is_ok());
        assert!(EntityId::new("claim:inspectable").is_ok());
        assert!(EntityId::new("nope").is_err());
        assert!(EntityId::new(":nope").is_err());
        assert!(EntityId::new("nope:").is_err());
        assert!(EntityId::new("chapter:BadCase").is_err());
    }

    #[test]
    fn duplicate_entity_is_error() {
        let raw = RawManifest {
            entity: vec![
                chapter_entry("chapter:a", "A"),
                chapter_entry("chapter:a", "A again"),
            ],
        };
        let err = build_manifest(raw, Path::new(".")).unwrap_err();
        assert!(matches!(err, ContentError::DuplicateEntity { .. }));
    }

    #[test]
    fn known_ids_collects_all_kinds() {
        let raw = RawManifest {
            entity: vec![
                chapter_entry("chapter:a", "A"),
                claim_entry("claim:x", "x"),
            ],
        };
        let load = build_manifest(raw, Path::new(".")).unwrap();
        let ids: Vec<&EntityId> = load.manifest.known_ids().collect();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn broken_link_is_error() {
        let raw = RawManifest {
            entity: vec![
                chapter_entry("chapter:a", "A"),
                EntityEntry {
                    id: EntityId::new("link:a-to-b").unwrap(),
                    kind: EntityKind::Link,
                    payload: serde_json::json!({
                        "from": "chapter:a",
                        "to": "chapter:b",
                        "kind": "frames"
                    }),
                },
            ],
        };
        let err = build_manifest(raw, Path::new(".")).unwrap_err();
        assert!(matches!(err, ContentError::BrokenLink { .. }));
    }
}
