//! Linalg dialect — linear algebra operations as ECS components.
//!
//! Provides structured operations for matrix multiplication, batch
//! matrix multiplication, and tensor fill operations.
//!
//! All operations define a `LinalgOp` component on the entity alongside
//! the standard `OpMarker`, `OpName`, `Operands`, `Results`, and
//! `OpAttributes` components.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

use crate::op::{OpInfo, OpRegistry};

// ── Op kind ──────────────────────────────────────────────────────────────────

/// Specific linalg operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinalgOpKind {
    /// Structured matrix multiplication: C += A @ B.
    Matmul,
    /// Batch matrix multiplication: C[i] += A[i] @ B[i].
    BatchMatmul,
    /// Fill a tensor with a scalar value.
    Fill,
}

impl LinalgOpKind {
    /// MLIR-style operation name for this kind.
    pub fn op_name(&self) -> &'static str {
        match self {
            LinalgOpKind::Matmul => "linalg.matmul",
            LinalgOpKind::BatchMatmul => "linalg.batch_matmul",
            LinalgOpKind::Fill => "linalg.fill",
        }
    }

    /// Number of required operands for this kind.
    pub fn operand_count(&self) -> usize {
        match self {
            LinalgOpKind::Matmul => 3,
            LinalgOpKind::BatchMatmul => 4,
            LinalgOpKind::Fill => 2,
        }
    }
}

// ── Component ────────────────────────────────────────────────────────────────

/// Component attaching a linalg op kind to an operation entity.
///
/// Every entity representing a linalg operation carries this component
/// so dialects and passes can discriminate typed linalg operations from
/// other operations or from general `OpName`-only queries.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinalgOp(pub LinalgOpKind);
impl Component for LinalgOp {}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register all linalg dialect operations into the given OpRegistry.
pub fn register_linalg_ops(registry: &mut OpRegistry) {
    registry.register(OpInfo {
        name: "linalg.matmul",
        description: "Structured matrix multiplication: C += A @ B",
        verify_fn: None,
        infer_fn: None,
    });
    registry.register(OpInfo {
        name: "linalg.batch_matmul",
        description: "Batch matrix multiplication: C[i] += A[i] @ B[i]",
        verify_fn: None,
        infer_fn: None,
    });
    registry.register(OpInfo {
        name: "linalg.fill",
        description: "Fill a tensor with a scalar value",
        verify_fn: None,
        infer_fn: None,
    });
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::OpBuilder;
    use crate::ir_types::Type;
    use crate::op::op_name;
    use prism_ecs_core::{EntityKind, World};

    // ── Component tests ──────────────────────────────────────────────────────

    #[test]
    fn linalg_op_kind_op_name() {
        assert_eq!(LinalgOpKind::Matmul.op_name(), "linalg.matmul");
        assert_eq!(LinalgOpKind::BatchMatmul.op_name(), "linalg.batch_matmul");
        assert_eq!(LinalgOpKind::Fill.op_name(), "linalg.fill");
    }

    #[test]
    fn linalg_op_kind_operand_count() {
        assert_eq!(LinalgOpKind::Matmul.operand_count(), 3);
        assert_eq!(LinalgOpKind::BatchMatmul.operand_count(), 4);
        assert_eq!(LinalgOpKind::Fill.operand_count(), 2);
    }

    #[test]
    fn linalg_op_component_attached() {
        let mut world = World::new();
        let entity: prism_ecs_core::Entity = world
            .spawn(EntityKind::Node, Some("test_linalg".into()))
            .expect("spawn")
            .into();
        world
            .add_component(entity, LinalgOp(LinalgOpKind::Matmul))
            .expect("add LinalgOp");
        let retrieved = world.get_component::<LinalgOp>(entity);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, LinalgOpKind::Matmul);
    }

    #[test]
    fn linalg_op_serialization_roundtrip() {
        let op = LinalgOp(LinalgOpKind::Matmul);
        let json = serde_json::to_string(&op).unwrap();
        let back: LinalgOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op.0, back.0);

        let kind = LinalgOpKind::BatchMatmul;
        let json = serde_json::to_string(&kind).unwrap();
        let back: LinalgOpKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }

    // ── Builder integration tests ────────────────────────────────────────────

    #[test]
    fn create_matmul_via_builder() {
        let mut world = World::new();
        // Create operand values for A, B, C
        let a = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce_a", &[], &[], &[Type::f32()])
                .unwrap();
            crate::op::results(&world, op)[0]
        };
        let b_val = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce_b", &[], &[], &[Type::f32()])
                .unwrap();
            crate::op::results(&world, op)[0]
        };
        let c = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce_c", &[], &[], &[Type::f32()])
                .unwrap();
            crate::op::results(&world, op)[0]
        };
        let matmul = {
            let mut builder = OpBuilder::new(&mut world);
            builder
                .create_op("linalg.matmul", &[a, b_val, c], &[], &[Type::f32()])
                .unwrap()
        };

        assert_eq!(op_name(&world, matmul), Some("linalg.matmul".into()));
        assert_eq!(crate::op::operands(&world, matmul), vec![a, b_val, c]);

        let matmul_results = crate::op::results(&world, matmul);
        assert_eq!(matmul_results.len(), 1);

        // Attach LinalgOp component after construction
        world
            .add_component(matmul, LinalgOp(LinalgOpKind::Matmul))
            .expect("add LinalgOp");
        assert_eq!(
            world.get_component::<LinalgOp>(matmul).unwrap().0,
            LinalgOpKind::Matmul
        );
    }

    // ── Registry integration tests ───────────────────────────────────────────

    #[test]
    fn register_all_linalg_ops() {
        let mut registry = crate::op::OpRegistry::new();
        register_linalg_ops(&mut registry);

        let expected = ["linalg.matmul", "linalg.batch_matmul", "linalg.fill"];
        for name in &expected {
            assert!(
                registry.get(name).is_some(),
                "missing registration for {}",
                name
            );
        }
    }

    #[test]
    fn registry_verify_matmul_unknown() {
        let mut registry = crate::op::OpRegistry::new();
        register_linalg_ops(&mut registry);

        let ctx = crate::op::OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("linalg.matmul", &ctx).is_ok());
        assert!(registry.verify("unknown.op", &ctx).is_err());
    }
}
