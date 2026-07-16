use crate::ast::Value;
use crate::resolve::ResolvedRecord;

/// Rust code generation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitMode {
    /// Generate ECS component code matching arith.rs pattern.
    EcsComponents,
}

/// Configuration for the emitter.
#[derive(Debug, Clone)]
pub struct EmitConfig {
    pub mode: EmitMode,
    pub dialect: Option<String>,
}

impl Default for EmitConfig {
    fn default() -> Self {
        Self {
            mode: EmitMode::EcsComponents,
            dialect: None,
        }
    }
}

/// Detect the dialect prefix from a superclass name or op name.
fn detect_dialect(records: &[ResolvedRecord]) -> String {
    // Check superclass names for dialect hints
    for rec in records {
        for sup in &rec.superclass_chain {
            if let Some(base) = sup.strip_suffix("_Op") {
                return base.to_lowercase();
            }
            if sup.starts_with("Op") || sup == "Op_base" {
                continue; // generic, check the def name
            }
            // Try to extract dialect from compound name like Arith_Op
            if sup.contains('_') {
                let parts: Vec<&str> = sup.split('_').collect();
                if parts.len() >= 2 && (parts[1] == "Op" || parts.last() == Some(&"Op")) {
                    return parts[0].to_lowercase();
                }
            }
        }
    }
    // Fallback: use first record name prefix
    for rec in records {
        for sup in &rec.superclass_chain {
            let lower = sup.to_lowercase();
            for prefix in &["arith", "func", "linalg", "scf", "math", "memref", "tensor"] {
                if lower.contains(prefix) {
                    return prefix.to_string();
                }
            }
        }
    }
    "dialect".to_string()
}

/// Normalize an op name from a def to a PascalCase variant name.
/// e.g. "addf" → "Addf", "constant" → "Constant"
fn to_variant_name(op_name: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for ch in op_name.chars() {
        if ch == '_' || ch == '.' || ch == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

/// Convert a def name like "ADDFOp" to an op name like "arith.addf".
/// Tries to extract the canonical op name from:
/// 1. Superclass template args (e.g. Arith_Op<"addf"> → "addf")
/// 2. Def name stripping "Op" suffix and lowercasing.
fn extract_op_name(def_name: &str, superclass_chain: &[String]) -> Option<String> {
    let _ = superclass_chain; // unused currently; used when template args are resolved
                              // Handle the case where the def itself was parsed and we have
                              // the original TdRecord with superclass args.
                              // For resolved records, we might not have the args in the superclass_chain
                              // string alone. So this is a best-effort extraction.

    // Strip trailing "Op" and lowercase
    let name = if let Some(stripped) = def_name.strip_suffix("Op") {
        stripped
    } else {
        def_name
    };

    // Convert PascalCase to dotted lowercase
    let mut result = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('.');
            }
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }
    }

    // If the result is empty, return the full name lowercased
    if result.is_empty() {
        Some(def_name.to_lowercase())
    } else {
        Some(result)
    }
}

/// Generate Rust source code from resolved TableGen records.
///
/// Detects operations by checking if they inherit from Op or Op_base
/// and produces an ArithOpKind-style enum with the full ECS component pattern.
pub fn emit_rust(records: &[ResolvedRecord]) -> Result<String, String> {
    let dialect = detect_dialect(records);
    let dialect_name = pascal_case(&dialect);

    // Filter records that look like ops (inherit from something ending in _Op or Op_base)
    let op_records: Vec<&ResolvedRecord> = records
        .iter()
        .filter(|r| {
            r.superclass_chain.iter().any(|s| {
                s.ends_with("_Op")
                    || s == "Op"
                    || s == "Op_base"
                    || s.starts_with(&pascal_case(&dialect))
            })
        })
        .collect();

    if op_records.is_empty() {
        // Fall back to classifying all resolved records as ops
        return emit_from_all(records, &dialect, &dialect_name);
    }

    emit_from_ops(&op_records, records, &dialect, &dialect_name)
}

fn pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for ch in s.chars() {
        if ch == '_' || ch == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

fn emit_from_all(
    records: &[ResolvedRecord],
    dialect: &str,
    dialect_name: &str,
) -> Result<String, String> {
    if records.is_empty() {
        return Err("no records to emit".to_string());
    }

    let mut out = String::new();

    // Header
    out.push_str(&format!(
        r#"//! {} dialect — auto-generated from TableGen definitions.
//!
//! Generated by prism-tblgen.

use serde::{{Deserialize, Serialize}};

// ── Op kind ──────────────────────────────────────────────────────────────────

/// Specific {} operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum {}OpKind {{
"#,
        dialect_name, dialect_name, dialect_name
    ));

    // Enum variants
    for rec in records {
        let op_name = extract_op_name(&rec.name, &rec.superclass_chain)
            .unwrap_or_else(|| rec.name.to_lowercase());
        let variant = to_variant_name(&op_name);
        out.push_str(&format!("    {},\n", variant));
    }

    out.push_str("}\n\n");

    // op_name() impl
    out.push_str(&format!("impl {}OpKind {{\n", dialect_name));
    out.push_str("    /// MLIR-style operation name for this kind.\n");
    out.push_str("    pub fn op_name(&self) -> &'static str {\n");
    out.push_str("        match self {\n");

    for rec in records {
        let op_name = extract_op_name(&rec.name, &rec.superclass_chain)
            .unwrap_or_else(|| rec.name.to_lowercase());
        let variant = to_variant_name(&op_name);
        let full_name = format!("{}.{}", dialect, op_name);
        out.push_str(&format!(
            "            {}OpKind::{} => \"{}\",\n",
            dialect_name, variant, full_name
        ));
    }

    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // Component
    out.push_str(&format!(
        r#"/// Component attaching a {} op kind to an operation entity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct {}Op(pub {}OpKind);
impl Component for {}Op {{}}
"#,
        dialect_name, dialect_name, dialect_name, dialect_name
    ));

    // Register function
    out.push_str(&format!(
        r#"
/// Register all {} dialect operations into the given OpRegistry.
pub fn register_{}_ops(registry: &mut OpRegistry) {{
"#,
        dialect_name, dialect
    ));

    for rec in records {
        let op_name = extract_op_name(&rec.name, &rec.superclass_chain)
            .unwrap_or_else(|| rec.name.to_lowercase());
        let full_name = format!("{}.{}", dialect, op_name);
        let desc = extract_description(rec);

        out.push_str(&format!(
            r#"    registry.register(OpInfo {{
        name: "{}",
        description: "{}",
        verify_fn: None,
        infer_fn: None,
    }});
"#,
            full_name, desc
        ));
    }

    out.push_str("}\n");

    Ok(out)
}

fn emit_from_ops(
    op_records: &[&ResolvedRecord],
    _all_records: &[ResolvedRecord],
    dialect: &str,
    dialect_name: &str,
) -> Result<String, String> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        r#"//! {} dialect — auto-generated from TableGen definitions.
//!
//! Generated by prism-tblgen.

use prism_ecs_core::Component;
use serde::{{Deserialize, Serialize}};

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{{OpInfo, OpRegistry, OpVerifierContext}};

// ── Op kind ──────────────────────────────────────────────────────────────────

/// Specific {} operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum {}OpKind {{
"#,
        dialect_name, dialect_name, dialect_name
    ));

    // Enum variants
    for rec in op_records {
        let op_name = extract_op_name(&rec.name, &rec.superclass_chain)
            .unwrap_or_else(|| rec.name.to_lowercase());
        let variant = to_variant_name(&op_name);
        out.push_str(&format!("    {},\n", variant));
    }

    out.push_str("}\n\n");

    // op_name() impl
    out.push_str(&format!("impl {}OpKind {{\n", dialect_name));
    out.push_str("    /// MLIR-style operation name for this kind.\n");
    out.push_str("    pub fn op_name(&self) -> &'static str {\n");
    out.push_str("        match self {\n");

    for rec in op_records {
        let op_name = extract_op_name(&rec.name, &rec.superclass_chain)
            .unwrap_or_else(|| rec.name.to_lowercase());
        let variant = to_variant_name(&op_name);
        let full_name = format!("{}.{}", dialect, op_name);
        out.push_str(&format!(
            "            {}OpKind::{} => \"{}\",\n",
            dialect_name, variant, full_name
        ));
    }

    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    // Component
    out.push_str(&format!(
        r#"/// Component attaching a {} op kind to an operation entity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct {}Op(pub {}OpKind);
impl Component for {}Op {{}}
"#,
        dialect_name, dialect_name, dialect_name, dialect_name
    ));

    // Registration
    out.push_str(&format!(
        r#"
/// Register all {} dialect operations into the given OpRegistry.
pub fn register_{}_ops(registry: &mut OpRegistry) {{
"#,
        dialect_name, dialect
    ));

    for rec in op_records {
        let op_name = extract_op_name(&rec.name, &rec.superclass_chain)
            .unwrap_or_else(|| rec.name.to_lowercase());
        let full_name = format!("{}.{}", dialect, op_name);
        let desc = extract_description(rec);

        out.push_str(&format!(
            r#"    registry.register(OpInfo {{
        name: "{}",
        description: "{}",
        verify_fn: None,
        infer_fn: None,
    }});
"#,
            full_name, desc
        ));
    }

    out.push_str("}\n");

    Ok(out)
}

/// Extract a human-readable description from a record's body.
fn extract_description(rec: &ResolvedRecord) -> String {
    for block in &rec.body {
        if block.name == "summary" || block.name == "description" {
            if let Value::StringLit(s) = &block.value {
                return s.clone();
            }
        }
    }
    format!("{} operation", rec.name)
}
