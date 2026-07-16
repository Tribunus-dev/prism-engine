//! Dialect registry — central catalog of dialect operations, types, and traits.
//!
//! Each dialect registers its ops, types, attributes, traits, and interfaces
//! with the [`DialectRegistry`]. The registry is then used during IR
//! construction to resolve op builders, verifiers, and type inference.
//!
//! # Design
//!
//! Mirrors MLIR's `DialectRegistry` (C++) and Melior's `DialectHandle` (Rust).
//! In our ECS-native IR, a dialect is a namespace prefix (e.g. `"arith"`,
//! `"func"`) that groups related operations, types, and traits.
//!
//! Each registration entry describes:
//! - Operations (name, builder, verifier, inferrer)
//! - Types (name, parser, printer)
//! - Attributes (name, parser, printer)
//! - Op traits (e.g. Commutative, Pure, Terminator)
//! - Op interfaces (e.g. InferType, LoopLike, Callable)

use std::collections::HashMap;

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{OpInfo, OpRegistry, OpVerifierContext};
use crate::traits::OpTraits;

// ── ComponentRegistration ───────────────────────────────────────────────────

/// What kinds of IR components a dialect can register.
#[derive(Debug, Clone)]
pub enum ComponentRegistration {
    /// An operation with its metadata.
    Operation(OpRegistration),
    /// A named type.
    Type(TypeRegistration),
    /// A named attribute.
    Attribute(AttributeRegistration),
    /// A trait declaration (e.g. Commutative).
    Trait(&'static str),
    /// An interface declaration (e.g. "InferType").
    Interface(&'static str),
}

/// Registration metadata for a single operation.
#[derive(Debug, Clone)]
pub struct OpRegistration {
    /// Fully qualified operation name (e.g. `"arith.addf"`).
    pub name: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// Traits associated with this op (bitflags).
    pub traits: OpTraits,
    /// Verification function, if any.
    pub verify_fn: Option<fn(&OpVerifierContext) -> Result<(), Vec<String>>>,
    /// Result type inference function, if any.
    pub infer_fn: Option<fn(&[Type], &[Attribute]) -> Option<Vec<Type>>>,
}

/// Registration metadata for a dialect type.
#[derive(Debug, Clone)]
pub struct TypeRegistration {
    /// Type name within the dialect (e.g. `"tensor"` → `"tensor<?xf32>"`).
    pub name: &'static str,
    /// Human-readable description.
    pub description: &'static str,
}

/// Registration metadata for a dialect attribute.
#[derive(Debug, Clone)]
pub struct AttributeRegistration {
    /// Attribute name (e.g. `"dense"`).
    pub name: &'static str,
    /// Human-readable description.
    pub description: &'static str,
}

// ── DialectRegistration trait ────────────────────────────────────────────────

/// A dialect that can register its components with the [`DialectRegistry`].
///
/// Implementations call `registry.register(...)` for each op, type,
/// attribute, trait, and interface the dialect provides.
pub trait DialectRegistration: Send + Sync {
    /// The dialect namespace (e.g. `"arith"`, `"func"`, `"scf"`).
    fn dialect_namespace(&self) -> &'static str;

    /// Register all components of this dialect into the registry.
    fn register(&self, registry: &mut DialectRegistry);
}

// ── DialectRegistry ─────────────────────────────────────────────────────────

/// Central registry mapping dialect namespaces to their registered components.
///
/// # Example
///
/// ```ignore
/// let mut registry = DialectRegistry::new();
/// registry.register_dialect(Box::new(ArithDialect));
/// registry.register_dialect(Box::new(FuncDialect));
/// let ops = registry.ops_for_dialect("arith");
/// ```
pub struct DialectRegistry {
    /// Map from dialect namespace → registered components.
    by_dialect: HashMap<&'static str, Vec<ComponentRegistration>>,
    /// Map from operation name → op info (aggregated across all dialects).
    op_registry: OpRegistry,
    /// Set of registered dialect namespaces.
    dialects: Vec<&'static str>,
}

impl DialectRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            by_dialect: HashMap::new(),
            op_registry: OpRegistry::new(),
            dialects: Vec::new(),
        }
    }

    /// Register a dialect's components via its [`DialectRegistration`].
    pub fn register_dialect(&mut self, dialect: Box<dyn DialectRegistration>) {
        let ns = dialect.dialect_namespace();
        dialect.register(self);
        if !self.dialects.contains(&ns) {
            self.dialects.push(ns);
        }
    }

    /// Register a single component under a dialect namespace.
    pub fn register(&mut self, namespace: &'static str, component: ComponentRegistration) {
        self.by_dialect
            .entry(namespace)
            .or_insert_with(Vec::new)
            .push(component.clone());

        // Also register ops into the flat op registry.
        if let ComponentRegistration::Operation(ref op_reg) = component {
            self.op_registry.register(OpInfo {
                name: op_reg.name,
                description: op_reg.description,
                verify_fn: op_reg.verify_fn,
                infer_fn: op_reg.infer_fn,
            });
        }
    }

    /// Get all registered component entries for a dialect.
    pub fn components_for_dialect(
        &self,
        namespace: &str,
    ) -> Option<&Vec<ComponentRegistration>> {
        self.by_dialect.get(namespace)
    }

    /// Get all op registrations for a given dialect.
    pub fn ops_for_dialect(&self, namespace: &str) -> Vec<&OpRegistration> {
        self.by_dialect
            .get(namespace)
            .map(|c| {
                c.iter()
                    .filter_map(|cr| {
                        if let ComponentRegistration::Operation(ref op) = cr {
                            Some(op)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all type registrations for a given dialect.
    pub fn types_for_dialect(&self, namespace: &str) -> Vec<&TypeRegistration> {
        self.by_dialect
            .get(namespace)
            .map(|c| {
                c.iter()
                    .filter_map(|cr| {
                        if let ComponentRegistration::Type(ref t) = cr {
                            Some(t)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all attribute registrations for a given dialect.
    pub fn attrs_for_dialect(&self, namespace: &str) -> Vec<&AttributeRegistration> {
        self.by_dialect
            .get(namespace)
            .map(|c| {
                c.iter()
                    .filter_map(|cr| {
                        if let ComponentRegistration::Attribute(ref a) = cr {
                            Some(a)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check whether a dialect is registered.
    pub fn has_dialect(&self, namespace: &str) -> bool {
        self.by_dialect.contains_key(namespace)
    }

    /// List all registered dialect namespaces.
    pub fn dialect_namespaces(&self) -> &[&'static str] {
        &self.dialects
    }

    /// Get the underlying op registry for verification and inference.
    pub fn op_registry(&self) -> &OpRegistry {
        &self.op_registry
    }

    /// Verify an operation using the op registry.
    pub fn verify_op(
        &self,
        name: &str,
        context: &OpVerifierContext,
    ) -> Result<(), Vec<String>> {
        self.op_registry.verify(name, context)
    }

    /// Infer result types for an operation using the op registry.
    pub fn infer_result_types(
        &self,
        name: &str,
        operand_types: &[Type],
        attributes: &[Attribute],
    ) -> Option<Vec<Type>> {
        self.op_registry.infer_result_types(name, operand_types, attributes)
    }
}

impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::OpTraits;

    struct ArithDialectReg;

    impl DialectRegistration for ArithDialectReg {
        fn dialect_namespace(&self) -> &'static str {
            "arith"
        }

        fn register(&self, registry: &mut DialectRegistry) {
            registry.register(
                "arith",
                ComponentRegistration::Operation(OpRegistration {
                    name: "arith.addf",
                    description: "Floating-point addition",
                    traits: OpTraits::COMMUTATIVE | OpTraits::PURE,
                    verify_fn: None,
                    infer_fn: None,
                }),
            );
            registry.register(
                "arith",
                ComponentRegistration::Operation(OpRegistration {
                    name: "arith.constant",
                    description: "Constant value",
                    traits: OpTraits::PURE,
                    verify_fn: None,
                    infer_fn: None,
                }),
            );
            registry.register(
                "arith",
                ComponentRegistration::Trait("Commutative"),
            );
        }
    }

    #[test]
    fn registry_create_and_query() {
        let mut registry = DialectRegistry::new();
        registry.register_dialect(Box::new(ArithDialectReg));

        assert!(registry.has_dialect("arith"));
        assert!(!registry.has_dialect("func"));

        let ops = registry.ops_for_dialect("arith");
        assert_eq!(ops.len(), 2);
        assert!(ops.iter().any(|o| o.name == "arith.addf"));
        assert!(ops.iter().any(|o| o.name == "arith.constant"));

        let namespaces = registry.dialect_namespaces();
        assert_eq!(namespaces, &["arith"]);
    }

    #[test]
    fn op_registry_integration() {
        let mut registry = DialectRegistry::new();
        registry.register_dialect(Box::new(ArithDialectReg));

        // Verify through the op registry.
        let ctx = OpVerifierContext::default();
        assert!(registry.verify_op("arith.addf", &ctx).is_ok());

        // Unknown op fails.
        assert!(registry.verify_op("unknown.op", &ctx).is_err());
    }

    #[test]
    fn multi_dialect_registration() {
        struct FuncDialectReg;

        impl DialectRegistration for FuncDialectReg {
            fn dialect_namespace(&self) -> &'static str {
                "func"
            }

            fn register(&self, registry: &mut DialectRegistry) {
                registry.register(
                    "func",
                    ComponentRegistration::Operation(OpRegistration {
                        name: "func.func",
                        description: "Function definition",
                        traits: OpTraits::PURE,
                        verify_fn: None,
                        infer_fn: None,
                    }),
                );
                registry.register(
                    "func",
                    ComponentRegistration::Operation(OpRegistration {
                        name: "func.return",
                        description: "Function return",
                        traits: OpTraits::TERMINATOR | OpTraits::PURE,
                        verify_fn: None,
                        infer_fn: None,
                    }),
                );
                registry.register(
                    "func",
                    ComponentRegistration::Type(TypeRegistration {
                        name: "function",
                        description: "Function type",
                    }),
                );
            }
        }

        let mut registry = DialectRegistry::new();
        registry.register_dialect(Box::new(ArithDialectReg));
        registry.register_dialect(Box::new(FuncDialectReg));

        assert!(registry.has_dialect("arith"));
        assert!(registry.has_dialect("func"));
        assert_eq!(registry.dialect_namespaces().len(), 2);

        assert_eq!(registry.ops_for_dialect("arith").len(), 2);
        assert_eq!(registry.ops_for_dialect("func").len(), 2);

        assert_eq!(registry.types_for_dialect("func").len(), 1);
        assert!(registry.types_for_dialect("arith").is_empty());
    }
}
