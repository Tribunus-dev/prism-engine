# Wave 14: One dialect from upstream TableGen (arith)

**Status:** Draft (pre-implementation)
**Dependency:** Wave 13 (ECS-native IR kernel) complete
**Owner:** kernel

## 1. Scope

Define the `arith` dialect operations as concrete ECS component types within the `prism-ecs-ir` crate. Each arith operation (addf, addi, subf, subi, mulf, muli, etc.) gets a dedicated component struct implementing the `OpaqueOp` trait.

This is the first concrete dialect wired into the ECS-native IR. It proves that dialect-specific ops can be created, verified, serialized, and round-tripped through the Wave 13 framework.

**Updated approach:** We define the arith ops directly in Rust (not generated from TableGen). This matches the ADR's "start with 3 files" methodology and proves the pattern before scaling to code generation.

## 2. File map

| File | Contents | Agent |
|---|---|---|
| `src/arith.rs` | ArithOp enum + all arith dialect operations (addf, addi, subf, subi, etc.) + verification + serialization support | A1 |
| (within serde.rs) | Arith operation serialization tests — round-trip arith ops through JSON | A1 |

## 3. Design

```rust
/// Arith dialect operation kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithOpKind {
    Addi, Subi, Muli, Divi, Remi,
    Addf, Subf, Mulf, Divf, Remf,
    Cmpi, Cmpf,
    Constant,
    Negf, Negi,
    Shli, Shrui, Shrsi,
    Andi, Ori, Xori,
    Select,
    // ... extensible
}
```

Each arith operation entity will carry:
- `OpMarker`, `OpName` (e.g. "arith.addf"), `Operands`, `Results`, `OpAttributes` (from Wave 13)
- `ArithOpKind` component indicating the specific arith operation variant

```rust
#[derive(Debug, Clone, Copy, Component, Serialize, Deserialize)]
pub struct ArithOp(pub ArithOpKind);
impl Component for ArithOp {}
```

## 4. Verification rules

| Op | Operands | Result type inference | Constraints |
|---|---|---|---|
| arith.addf | {lhs: float, rhs: float} | float with same element type | lhs.type == rhs.type |
| arith.addi | {lhs: int, rhs: int} | int with same width/signedness | lhs.type == rhs.type |
| arith.constant | {} | declared on op | value attribute must match result type |
| arith.cmpi | {lhs: int, rhs: int} | i1 (boolean) | — |
| arith.select | {cond: i1, true_val: T, false_val: T} | T | true_val.type == false_val.type |

## 5. Type inference

Register inference functions in the `TypeInferenceRegistry`:

```rust
registry.register("arith.addf", Box::new(|operand_types, _attrs| {
    Some(vec![operand_types[0].clone()]) // result == operand type
}));
```

## 6. Serialization

ArithOpKind is serialized as an attribute on the OpAttributes component. The `arith.addf` op name is the primary discriminant; `ArithOpKind::Addf` is a helper for type-specific access.

## 7. Gate

- `cargo check -p prism-ecs-ir` passes with 0 errors
- An arith.addf op is created via OpBuilder, serialized to JSON, deserialized, and verified
- Verification: arith.addf with mismatched operand types fails
- Type inference: arith.addf(f32, f32) → f32 matches expected
