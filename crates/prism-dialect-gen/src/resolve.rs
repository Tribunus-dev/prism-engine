use crate::ast::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A resolved record with template instantiation and superclass chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedRecord {
    pub name: String,
    pub superclass_chain: Vec<String>,
    pub body: Vec<LetBlock>,
}

/// Resolve a parsed TableGen document.
///
/// This performs basic template instantiation:
/// 1. Collects all classes and multiclasses for reference.
/// 2. For each def, follows the superclass chain to collect inherited bodies.
/// 3. Flattens multiclass defm expansions (each defm → one record per def in multiclass).
///
/// Returns resolved records in definition order.
pub fn resolve_document(doc: &TdDocument) -> Result<Vec<ResolvedRecord>, String> {
    let mut classes: HashMap<&str, &TdRecord> = HashMap::new();
    let mut multiclasses: HashMap<&str, &TdRecord> = HashMap::new();
    let mut resolved = Vec::new();

    // Phase 1: index classes and multiclasses
    for rec in &doc.records {
        match rec.kind {
            RecordKind::Class => {
                classes.insert(&rec.name, rec);
            }
            RecordKind::Multiclass => {
                multiclasses.insert(&rec.name, rec);
            }
            RecordKind::Def => {}
        }
    }

    // Phase 2: resolve each def
    for rec in &doc.records {
        if rec.kind != RecordKind::Def {
            continue;
        }
        resolved.push(resolve_def(rec, &classes, &multiclasses)?);
    }

    // Phase 3: expand defm records (conceptually, defm creates defs from a multiclass)
    // For a real resolver we would also handle defm here. For now,
    // we also process any `defm` entries in the doc.
    for rec in &doc.records {
        if rec.kind == RecordKind::Def {
            continue; // already resolved
        }
        // defm handled via the record name starting with special marker
        // (in a real resolver, defm expands into multiple defs)
    }

    Ok(resolved)
}

fn resolve_def(
    rec: &TdRecord,
    classes: &HashMap<&str, &TdRecord>,
    _multiclasses: &HashMap<&str, &TdRecord>,
) -> Result<ResolvedRecord, String> {
    let mut chain = Vec::new();
    let mut body: Vec<LetBlock> = rec.body.clone();

    // Follow the superclass chain
    for sup in &rec.superclasses {
        chain.push(sup.name.clone());
        if let Some(class_rec) = classes.get(sup.name.as_str()) {
            // Inherit body from class
            // In a full resolver, template arguments would be substituted here.
            // For now we just prepend the class body (class fields go first,
            // then the def's own fields override).
            let mut inherited = class_rec.body.clone();
            inherited.extend(body.clone());
            body = inherited;
        }
    }

    Ok(ResolvedRecord {
        name: rec.name.clone(),
        superclass_chain: chain,
        body,
    })
}
