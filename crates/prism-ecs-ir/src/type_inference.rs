//! Type inference registry — maps op names to inference functions.
//!
//! Defines the type inference interface for operations. Each op can register
//! an inference function that computes result types from operand types and
//! attributes.

use std::collections::HashMap;

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;

/// Type inference registry — maps op names to their inference functions.
///
/// ```ignore
/// let mut registry = TypeInferenceRegistry::new();
/// registry.register("arith.addf", Box::new(|operand_types, _attrs| {
///     Some(vec![operand_types[0].clone()]) // result type == operand type
/// }));
/// let result = registry.infer("arith.addf", &[Type::f32(), Type::f32()], &[]);
/// ```
pub struct TypeInferenceRegistry {
    inferers: HashMap<&'static str, Box<dyn Fn(&[Type], &[Attribute]) -> Option<Vec<Type>>>>,
}

impl TypeInferenceRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            inferers: HashMap::new(),
        }
    }

    /// Register an inference function for an operation.
    pub fn register(
        &mut self,
        op_name: &'static str,
        inferer: Box<dyn Fn(&[Type], &[Attribute]) -> Option<Vec<Type>>>,
    ) {
        self.inferers.insert(op_name, inferer);
    }

    /// Infer result types for the given operation.
    ///
    /// Returns `None` if no inference function is registered or if inference
    /// fails (unsupported operand types, etc.).
    pub fn infer(
        &self,
        op_name: &str,
        operand_types: &[Type],
        attributes: &[Attribute],
    ) -> Option<Vec<Type>> {
        self.inferers
            .get(op_name)
            .and_then(|f| f(operand_types, attributes))
    }

    /// Check if an inference function is registered for the given op.
    pub fn has_inference(&self, op_name: &str) -> bool {
        self.inferers.contains_key(op_name)
    }

    /// Number of registered inference functions.
    pub fn len(&self) -> usize {
        self.inferers.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.inferers.is_empty()
    }
}

impl Default for TypeInferenceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_arith_addf() {
        let mut registry = TypeInferenceRegistry::new();

        // arith.addf: result type = operand type (f32 + f32 -> f32)
        registry.register(
            "arith.addf",
            Box::new(|operand_types, _attrs| {
                if operand_types.len() == 2 && operand_types[0] == operand_types[1] {
                    Some(vec![operand_types[0].clone()])
                } else {
                    None
                }
            }),
        );

        let result = registry.infer("arith.addf", &[Type::f32(), Type::f32()], &[]);
        assert_eq!(result, Some(vec![Type::f32()]));

        // Mismatched types should fail
        let result = registry.infer("arith.addf", &[Type::f32(), Type::i32()], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn unknown_op_returns_none() {
        let registry = TypeInferenceRegistry::new();
        let result = registry.infer("unknown.op", &[Type::i32()], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn register_multiple_ops() {
        let mut registry = TypeInferenceRegistry::new();
        registry.register("arith.addf", Box::new(|t, _| Some(t.to_vec())));
        registry.register("arith.mulf", Box::new(|t, _| Some(t.to_vec())));

        assert!(registry.has_inference("arith.addf"));
        assert!(registry.has_inference("arith.mulf"));
        assert!(!registry.has_inference("unknown.op"));
        assert_eq!(registry.len(), 2);
    }
}
