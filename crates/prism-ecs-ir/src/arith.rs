//! Arith dialect — arithmetic and logical operations as ECS components.
//!
//! Provides typed verification rules and type inference for standard
//! arithmetic, comparison, shift, bitwise, and select operations.
//!
//! All operations define an `ArithOp` component on the entity alongside
//! the standard `OpMarker`, `OpName`, `Operands`, `Results`, and
//! `OpAttributes` components.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{OpInfo, OpRegistry, OpVerifierContext};

// ── Op kind ──────────────────────────────────────────────────────────────────

/// Specific arith operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithOpKind {
    // Integer arithmetic
    Addi,
    Subi,
    Muli,
    Divi,
    Remi,
    // Floating-point arithmetic
    Addf,
    Subf,
    Mulf,
    Divf,
    Remf,
    // Comparison
    Cmpi,
    Cmpf,
    // Constant
    Constant,
    // Unary
    Negf,
    Negi,
    // Shift
    Shli,
    Shrui,
    Shrsi,
    // Bitwise
    Andi,
    Ori,
    Xori,
    // Select
    Select,
}

impl ArithOpKind {
    /// MLIR-style operation name for this kind.
    pub fn op_name(&self) -> &'static str {
        match self {
            ArithOpKind::Addf => "arith.addf",
            ArithOpKind::Addi => "arith.addi",
            ArithOpKind::Subf => "arith.subf",
            ArithOpKind::Subi => "arith.subi",
            ArithOpKind::Mulf => "arith.mulf",
            ArithOpKind::Muli => "arith.muli",
            ArithOpKind::Divf => "arith.divf",
            ArithOpKind::Divi => "arith.divi",
            ArithOpKind::Remf => "arith.remf",
            ArithOpKind::Remi => "arith.remi",
            ArithOpKind::Cmpi => "arith.cmpi",
            ArithOpKind::Cmpf => "arith.cmpf",
            ArithOpKind::Constant => "arith.constant",
            ArithOpKind::Negf => "arith.negf",
            ArithOpKind::Negi => "arith.negi",
            ArithOpKind::Shli => "arith.shli",
            ArithOpKind::Shrui => "arith.shrui",
            ArithOpKind::Shrsi => "arith.shrsi",
            ArithOpKind::Andi => "arith.andi",
            ArithOpKind::Ori => "arith.ori",
            ArithOpKind::Xori => "arith.xori",
            ArithOpKind::Select => "arith.select",
        }
    }

    /// Number of required operands for this kind.
    pub fn operand_count(&self) -> usize {
        match self {
            ArithOpKind::Constant => 0,
            ArithOpKind::Negf | ArithOpKind::Negi => 1,
            ArithOpKind::Addi
            | ArithOpKind::Subi
            | ArithOpKind::Muli
            | ArithOpKind::Divi
            | ArithOpKind::Remi
            | ArithOpKind::Addf
            | ArithOpKind::Subf
            | ArithOpKind::Mulf
            | ArithOpKind::Divf
            | ArithOpKind::Remf
            | ArithOpKind::Cmpi
            | ArithOpKind::Cmpf
            | ArithOpKind::Shli
            | ArithOpKind::Shrui
            | ArithOpKind::Shrsi
            | ArithOpKind::Andi
            | ArithOpKind::Ori
            | ArithOpKind::Xori => 2,
            ArithOpKind::Select => 3,
        }
    }
}

// ── Component ────────────────────────────────────────────────────────────────

/// Component attaching an arith op kind to an operation entity.
///
/// Every entity representing an arith operation carries this component
/// so dialects and passes can discriminate typed arith operations from
/// other operations or from general `OpName`-only queries.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ArithOp(pub ArithOpKind);
impl Component for ArithOp {}

// ── Verifiers ────────────────────────────────────────────────────────────────

/// Verify a binary float arithmetic op: 2 operands, same float type.
pub fn verify_binary_float(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 2 {
        errors.push(format!(
            "expected 2 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if ctx.operand_types.len() >= 2 {
        let t0 = &ctx.operand_types[0];
        let t1 = &ctx.operand_types[1];
        if !matches!(t0, Type::Float(_)) {
            errors.push(format!("operand 0 is not a float type: {:?}", t0));
        }
        if t0 != t1 {
            errors.push(format!("operand types differ: {:?} vs {:?}", t0, t1));
        }
        if ctx.result_types.len() >= 1 && &ctx.result_types[0] != t0 {
            errors.push(format!(
                "result type {:?} does not match operand type {:?}",
                ctx.result_types[0], t0
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a binary integer arithmetic op: 2 operands, same integer type.
pub fn verify_binary_integer(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 2 {
        errors.push(format!(
            "expected 2 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if ctx.operand_types.len() >= 2 {
        let t0 = &ctx.operand_types[0];
        let t1 = &ctx.operand_types[1];
        if !matches!(t0, Type::Integer(_)) {
            errors.push(format!("operand 0 is not an integer type: {:?}", t0));
        }
        if t0 != t1 {
            errors.push(format!("operand types differ: {:?} vs {:?}", t0, t1));
        }
        if ctx.result_types.len() >= 1 && &ctx.result_types[0] != t0 {
            errors.push(format!(
                "result type {:?} does not match operand type {:?}",
                ctx.result_types[0], t0
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a comparison op: 2 operands of same type, result is index.
pub fn verify_compare(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 2 {
        errors.push(format!(
            "expected 2 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if ctx.operand_types.len() >= 2 {
        let t0 = &ctx.operand_types[0];
        let t1 = &ctx.operand_types[1];
        if t0 != t1 {
            errors.push(format!("operand types differ: {:?} vs {:?}", t0, t1));
        }
    }
    if ctx.result_types.len() >= 1 && ctx.result_types[0] != Type::Index {
        errors.push(format!(
            "compare result must be index, got {:?}",
            ctx.result_types[0]
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a unary float op: 1 operand, float type, result matches operand.
pub fn verify_unary_float(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 1 {
        errors.push(format!(
            "expected 1 operand, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if !ctx.operand_types.is_empty() {
        let t0 = &ctx.operand_types[0];
        if !matches!(t0, Type::Float(_)) {
            errors.push(format!("operand is not a float type: {:?}", t0));
        }
        if ctx.result_types.len() >= 1 && &ctx.result_types[0] != t0 {
            errors.push(format!(
                "result type {:?} does not match operand type {:?}",
                ctx.result_types[0], t0
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a unary integer op: 1 operand, integer type, result matches operand.
pub fn verify_unary_integer(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 1 {
        errors.push(format!(
            "expected 1 operand, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if !ctx.operand_types.is_empty() {
        let t0 = &ctx.operand_types[0];
        if !matches!(t0, Type::Integer(_)) {
            errors.push(format!("operand is not an integer type: {:?}", t0));
        }
        if ctx.result_types.len() >= 1 && &ctx.result_types[0] != t0 {
            errors.push(format!(
                "result type {:?} does not match operand type {:?}",
                ctx.result_types[0], t0
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a shift op: 2 operands, both integer types.
pub fn verify_shift(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 2 {
        errors.push(format!(
            "expected 2 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if ctx.operand_types.len() >= 2 {
        let t0 = &ctx.operand_types[0];
        let t1 = &ctx.operand_types[1];
        if !matches!(t0, Type::Integer(_)) {
            errors.push(format!("operand 0 is not an integer type: {:?}", t0));
        }
        if !matches!(t1, Type::Integer(_)) {
            errors.push(format!("operand 1 is not an integer type: {:?}", t1));
        }
        if ctx.result_types.len() >= 1 && &ctx.result_types[0] != t0 {
            errors.push(format!(
                "result type {:?} does not match operand 0 type {:?}",
                ctx.result_types[0], t0
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a bitwise op: 2 operands, same integer type.
pub fn verify_bitwise(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    // Same rules as binary integer
    verify_binary_integer(ctx)
}

/// Verify select: 3 operands, op0 is index, op1 == op2, result == op1.
pub fn verify_select(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 3 {
        errors.push(format!(
            "expected 3 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if ctx.operand_types.len() >= 3 {
        let cond = &ctx.operand_types[0];
        let t1 = &ctx.operand_types[1];
        let t2 = &ctx.operand_types[2];
        if *cond != Type::Index {
            errors.push(format!("select condition must be index, got {:?}", cond));
        }
        if t1 != t2 {
            errors.push(format!("select value types differ: {:?} vs {:?}", t1, t2));
        }
        if ctx.result_types.len() >= 1 && &ctx.result_types[0] != t1 {
            errors.push(format!(
                "result type {:?} does not match value type {:?}",
                ctx.result_types[0], t1
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify constant: 0 operands, 1 result, must have a value attribute.
pub fn verify_constant(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if !ctx.operand_types.is_empty() {
        errors.push(format!(
            "expected 0 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!("expected 1 result, got {}", ctx.result_types.len()));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ── Type inference ───────────────────────────────────────────────────────────

/// Infer result type for binary float ops: result type = operand 0 type.
pub fn infer_binary_float(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() == 2
        && matches!(&operand_types[0], Type::Float(_))
        && operand_types[0] == operand_types[1]
    {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result type for binary integer ops: result type = operand 0 type.
pub fn infer_binary_integer(
    operand_types: &[Type],
    _attributes: &[Attribute],
) -> Option<Vec<Type>> {
    if operand_types.len() == 2
        && matches!(&operand_types[0], Type::Integer(_))
        && operand_types[0] == operand_types[1]
    {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result type for compare ops: result type is index.
pub fn infer_compare(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() == 2 && operand_types[0] == operand_types[1] {
        Some(vec![Type::Index])
    } else {
        None
    }
}

/// Infer result type for unary float ops: result type = operand type.
pub fn infer_unary_float(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() == 1 && matches!(&operand_types[0], Type::Float(_)) {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result type for unary integer ops: result type = operand type.
pub fn infer_unary_integer(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() == 1 && matches!(&operand_types[0], Type::Integer(_)) {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result type for shift ops: result type = operand 0 type.
pub fn infer_shift(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() == 2
        && matches!(&operand_types[0], Type::Integer(_))
        && matches!(&operand_types[1], Type::Integer(_))
    {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result type for bitwise ops: result type = operand 0 type.
pub fn infer_bitwise(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    // Same as binary integer
    infer_binary_integer(operand_types, _attributes)
}

/// Infer result type for select: result type = operand 1 type.
pub fn infer_select(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() == 3
        && operand_types[0] == Type::Index
        && operand_types[1] == operand_types[2]
    {
        Some(vec![operand_types[1].clone()])
    } else {
        None
    }
}

/// Infer result type for constant: no inference from operands alone.
pub fn infer_constant(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    // Constant type comes from its value attribute, not from operands.
    None
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register all arith dialect operations into the given OpRegistry.
pub fn register_arith_ops(registry: &mut OpRegistry) {
    // Binary float ops
    for (name, kind_str, verify_fn, infer_fn) in &[
        (
            "arith.addf",
            "Floating-point addition",
            verify_binary_float as fn(&OpVerifierContext) -> Result<(), Vec<String>>,
            infer_binary_float as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>,
        ),
        (
            "arith.subf",
            "Floating-point subtraction",
            verify_binary_float,
            infer_binary_float,
        ),
        (
            "arith.mulf",
            "Floating-point multiplication",
            verify_binary_float,
            infer_binary_float,
        ),
        (
            "arith.divf",
            "Floating-point division",
            verify_binary_float,
            infer_binary_float,
        ),
        (
            "arith.remf",
            "Floating-point remainder",
            verify_binary_float,
            infer_binary_float,
        ),
    ] {
        registry.register(OpInfo {
            name,
            description: kind_str,
            verify_fn: Some(*verify_fn),
            infer_fn: Some(*infer_fn),
        });
    }

    // Binary integer ops
    for (name, kind_str, verify_fn, infer_fn) in &[
        (
            "arith.addi",
            "Integer addition",
            verify_binary_integer as fn(&OpVerifierContext) -> Result<(), Vec<String>>,
            infer_binary_integer as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>,
        ),
        (
            "arith.subi",
            "Integer subtraction",
            verify_binary_integer,
            infer_binary_integer,
        ),
        (
            "arith.muli",
            "Integer multiplication",
            verify_binary_integer,
            infer_binary_integer,
        ),
        (
            "arith.divi",
            "Integer division",
            verify_binary_integer,
            infer_binary_integer,
        ),
        (
            "arith.remi",
            "Integer remainder",
            verify_binary_integer,
            infer_binary_integer,
        ),
    ] {
        registry.register(OpInfo {
            name,
            description: kind_str,
            verify_fn: Some(*verify_fn),
            infer_fn: Some(*infer_fn),
        });
    }

    // Compare ops
    registry.register(OpInfo {
        name: "arith.cmpi",
        description: "Integer comparison",
        verify_fn: Some(verify_compare as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_compare as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "arith.cmpf",
        description: "Floating-point comparison",
        verify_fn: Some(verify_compare as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_compare as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });

    // Unary ops
    registry.register(OpInfo {
        name: "arith.negf",
        description: "Floating-point negation",
        verify_fn: Some(verify_unary_float as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_unary_float as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "arith.negi",
        description: "Integer negation",
        verify_fn: Some(verify_unary_integer as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_unary_integer as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });

    // Shift ops
    for (name, kind_str) in &[
        ("arith.shli", "Integer shift left"),
        ("arith.shrui", "Integer unsigned shift right"),
        ("arith.shrsi", "Integer signed shift right"),
    ] {
        registry.register(OpInfo {
            name,
            description: kind_str,
            verify_fn: Some(verify_shift as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
            infer_fn: Some(infer_shift as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
        });
    }

    // Bitwise ops
    for (name, kind_str) in &[
        ("arith.andi", "Integer bitwise and"),
        ("arith.ori", "Integer bitwise or"),
        ("arith.xori", "Integer bitwise xor"),
    ] {
        registry.register(OpInfo {
            name,
            description: kind_str,
            verify_fn: Some(verify_bitwise as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
            infer_fn: Some(infer_bitwise as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
        });
    }

    // Select
    registry.register(OpInfo {
        name: "arith.select",
        description: "Conditional select (index ? value1 : value2)",
        verify_fn: Some(verify_select as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_select as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });

    // Constant
    registry.register(OpInfo {
        name: "arith.constant",
        description: "Constant value",
        verify_fn: Some(verify_constant as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_constant as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
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
    fn arith_op_kind_op_name() {
        assert_eq!(ArithOpKind::Addf.op_name(), "arith.addf");
        assert_eq!(ArithOpKind::Addi.op_name(), "arith.addi");
        assert_eq!(ArithOpKind::Mulf.op_name(), "arith.mulf");
        assert_eq!(ArithOpKind::Constant.op_name(), "arith.constant");
        assert_eq!(ArithOpKind::Select.op_name(), "arith.select");
        assert_eq!(ArithOpKind::Negf.op_name(), "arith.negf");
        assert_eq!(ArithOpKind::Shli.op_name(), "arith.shli");
        assert_eq!(ArithOpKind::Andi.op_name(), "arith.andi");
        assert_eq!(ArithOpKind::Ori.op_name(), "arith.ori");
    }

    #[test]
    fn arith_op_kind_operand_count() {
        assert_eq!(ArithOpKind::Constant.operand_count(), 0);
        assert_eq!(ArithOpKind::Negf.operand_count(), 1);
        assert_eq!(ArithOpKind::Addf.operand_count(), 2);
        assert_eq!(ArithOpKind::Cmpi.operand_count(), 2);
        assert_eq!(ArithOpKind::Select.operand_count(), 3);
    }

    #[test]
    fn arith_op_component_attached() {
        let mut world = World::new();
        let entity: prism_ecs_core::Entity = world
            .spawn(EntityKind::Node, Some("test_arith".into()))
            .expect("spawn")
            .into();
        world
            .add_component(entity, ArithOp(ArithOpKind::Addf))
            .expect("add ArithOp");
        let retrieved = world.get_component::<ArithOp>(entity);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, ArithOpKind::Addf);
    }

    #[test]
    fn arith_op_serialization_roundtrip() {
        let op = ArithOp(ArithOpKind::Mulf);
        let json = serde_json::to_string(&op).unwrap();
        let back: ArithOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op.0, back.0);

        let kind = ArithOpKind::Addf;
        let json = serde_json::to_string(&kind).unwrap();
        let back: ArithOpKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }

    // ── Builder integration tests ────────────────────────────────────────────

    #[test]
    fn create_addf_via_builder() {
        let mut world = World::new();
        // Create operands using separate builders (drop between uses)
        let v1 = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::f32()])
                .unwrap();
            drop(b);
            crate::op::results(&world, op)[0]
        };
        let v2 = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::f32()])
                .unwrap();
            drop(b);
            crate::op::results(&world, op)[0]
        };
        let addf = {
            let mut builder = OpBuilder::new(&mut world);
            let addf = builder
                .create_op("arith.addf", &[v1, v2], &[], &[Type::f32()])
                .unwrap();
            drop(builder);
            addf
        };

        assert_eq!(op_name(&world, addf), Some("arith.addf".into()));
        assert_eq!(crate::op::operands(&world, addf), vec![v1, v2]);

        let addf_results = crate::op::results(&world, addf);
        assert_eq!(addf_results.len(), 1);
        let rtype = crate::value::value_type(&world, addf_results[0]);
        assert_eq!(rtype, Some(Type::f32()));

        // Attach ArithOp component after construction
        world
            .add_component(addf, ArithOp(ArithOpKind::Addf))
            .expect("add ArithOp");
        assert_eq!(
            world.get_component::<ArithOp>(addf).unwrap().0,
            ArithOpKind::Addf
        );
    }

    // ── Verification tests ───────────────────────────────────────────────────

    #[test]
    fn verify_binary_float_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_binary_float(&ctx).is_ok());
    }

    #[test]
    fn verify_binary_float_mismatched_types() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::bf16()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_binary_float(&ctx).is_err());
    }

    #[test]
    fn verify_binary_float_wrong_operand_count() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_binary_float(&ctx).is_err());
    }

    #[test]
    fn verify_binary_float_integer_operand() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32(), Type::i32()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_binary_float(&ctx).is_err());
    }

    #[test]
    fn verify_binary_integer_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32(), Type::i32()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_binary_integer(&ctx).is_ok());
    }

    #[test]
    fn verify_binary_integer_mismatched_types() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32(), Type::i64()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_binary_integer(&ctx).is_err());
    }

    #[test]
    fn verify_compare_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32()],
            result_types: vec![Type::Index],
            attributes: vec![],
        };
        assert!(verify_compare(&ctx).is_ok());
    }

    #[test]
    fn verify_compare_wrong_result_type() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32(), Type::i32()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_compare(&ctx).is_err());
    }

    #[test]
    fn verify_unary_float_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_unary_float(&ctx).is_ok());
    }

    #[test]
    fn verify_unary_integer_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_unary_integer(&ctx).is_ok());
    }

    #[test]
    fn verify_shift_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32(), Type::i32()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_shift(&ctx).is_ok());
    }

    #[test]
    fn verify_bitwise_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32(), Type::i32()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_bitwise(&ctx).is_ok());
    }

    #[test]
    fn verify_select_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index, Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_select(&ctx).is_ok());
    }

    #[test]
    fn verify_select_wrong_condition_type() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32(), Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_select(&ctx).is_err());
    }

    #[test]
    fn verify_select_wrong_operand_count() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index, Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_select(&ctx).is_err());
    }

    #[test]
    fn verify_constant_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_constant(&ctx).is_ok());
    }

    #[test]
    fn verify_constant_with_operands() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32()],
            result_types: vec![Type::i32()],
            attributes: vec![],
        };
        assert!(verify_constant(&ctx).is_err());
    }

    // ── Type inference tests ─────────────────────────────────────────────────

    #[test]
    fn infer_addf_to_f32() {
        // arith.addf(f32, f32) -> f32
        let result = infer_binary_float(&[Type::f32(), Type::f32()], &[]);
        assert_eq!(result, Some(vec![Type::f32()]));
    }

    #[test]
    fn infer_addf_mismatched_types() {
        let result = infer_binary_float(&[Type::f32(), Type::i32()], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn infer_addf_wrong_operand_count() {
        let result = infer_binary_float(&[Type::f32()], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn infer_addi_to_i32() {
        let result = infer_binary_integer(&[Type::i32(), Type::i32()], &[]);
        assert_eq!(result, Some(vec![Type::i32()]));
    }

    #[test]
    fn infer_addi_to_i64() {
        let result = infer_binary_integer(&[Type::i64(), Type::i64()], &[]);
        assert_eq!(result, Some(vec![Type::i64()]));
    }

    #[test]
    fn infer_compare_to_index() {
        let result = infer_compare(&[Type::f32(), Type::f32()], &[]);
        assert_eq!(result, Some(vec![Type::Index]));
    }

    #[test]
    fn infer_negf_to_f32() {
        let result = infer_unary_float(&[Type::f32()], &[]);
        assert_eq!(result, Some(vec![Type::f32()]));
    }

    #[test]
    fn infer_negi_to_i32() {
        let result = infer_unary_integer(&[Type::i32()], &[]);
        assert_eq!(result, Some(vec![Type::i32()]));
    }

    #[test]
    fn infer_shift_to_i32() {
        let result = infer_shift(&[Type::i32(), Type::i32()], &[]);
        assert_eq!(result, Some(vec![Type::i32()]));
    }

    #[test]
    fn infer_shift_mismatched_left_not_int() {
        let result = infer_shift(&[Type::f32(), Type::i32()], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn infer_select_ok() {
        let result = infer_select(&[Type::Index, Type::f32(), Type::f32()], &[]);
        assert_eq!(result, Some(vec![Type::f32()]));
    }

    #[test]
    fn infer_select_mismatched_value_types() {
        let result = infer_select(&[Type::Index, Type::f32(), Type::i32()], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn infer_constant_returns_none() {
        let result = infer_constant(&[], &[]);
        assert_eq!(result, None);
    }

    // ── Registry integration tests ───────────────────────────────────────────

    #[test]
    fn register_all_arith_ops() {
        let mut registry = crate::op::OpRegistry::new();
        register_arith_ops(&mut registry);

        // Spot-check a few
        assert!(registry.get("arith.addf").is_some());
        assert!(registry.get("arith.addi").is_some());
        assert!(registry.get("arith.select").is_some());
        assert!(registry.get("arith.constant").is_some());
        assert!(registry.get("arith.shli").is_some());
        assert!(registry.get("arith.cmpf").is_some());

        // All 22 arith ops should be registered
        let expected = [
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.remf",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.divi",
            "arith.remi",
            "arith.cmpi",
            "arith.cmpf",
            "arith.constant",
            "arith.negf",
            "arith.negi",
            "arith.shli",
            "arith.shrui",
            "arith.shrsi",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.select",
        ];
        for name in &expected {
            assert!(
                registry.get(name).is_some(),
                "missing registration for {}",
                name
            );
        }
    }

    #[test]
    fn registry_verify_addf() {
        let mut registry = crate::op::OpRegistry::new();
        register_arith_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("arith.addf", &ctx).is_ok());

        let bad_ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::i32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("arith.addf", &bad_ctx).is_err());
    }

    #[test]
    fn registry_infer_addf() {
        let mut registry = crate::op::OpRegistry::new();
        register_arith_ops(&mut registry);

        let result = registry.infer_result_types("arith.addf", &[Type::f32(), Type::f32()], &[]);
        assert_eq!(result, Some(vec![Type::f32()]));
    }

    #[test]
    fn registry_infer_addi() {
        let mut registry = crate::op::OpRegistry::new();
        register_arith_ops(&mut registry);

        let result = registry.infer_result_types("arith.addi", &[Type::i64(), Type::i64()], &[]);
        assert_eq!(result, Some(vec![Type::i64()]));
    }

    #[test]
    fn registry_infer_cmpf() {
        let mut registry = crate::op::OpRegistry::new();
        register_arith_ops(&mut registry);

        let result = registry.infer_result_types("arith.cmpf", &[Type::f32(), Type::f32()], &[]);
        assert_eq!(result, Some(vec![Type::Index]));
    }

    #[test]
    fn registry_infer_unknown_op() {
        let mut registry = crate::op::OpRegistry::new();
        register_arith_ops(&mut registry);

        let result = registry.infer_result_types("unknown.op", &[], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn registry_verify_unknown_op() {
        let mut registry = crate::op::OpRegistry::new();
        register_arith_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("unknown.op", &ctx).is_err());
    }
}
