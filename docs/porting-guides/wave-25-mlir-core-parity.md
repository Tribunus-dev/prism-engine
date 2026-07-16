# Wave 25: MLIR Core Feature Parity — Locations, Attributes, Traits, Interfaces

**Status:** Draft (pre-implementation)
**Dependency:** Waves 13-24 (full IR core + dialects + codegen backends)
**Owner:** kernel

## 1. Scope

Deliver MLIR feature parity for the core IR system: locations, full attribute system, op traits, op interfaces, MemRef type, diagnostics, pass manager, SSA verifier.

## 2. Parallel work streams

### Stream A: Location tracking

`src/location.rs` — `Location` component attached to every operation entity.

```rust
pub struct Location(pub LocKind);

pub enum LocKind {
    Unknown,
    FileLineCol(String, u32, u32),  // file, line, col
    Name(String),                    // named location
    CallSite(Entity, Entity),        // caller, callee locations
    Fused(Vec<Location>),            // fused locations
}
```

OpBuilder sets `Location::Unknown` by default. Serialization includes location.

+ wire into OpBuilder.create_op(), serde, and `Diagnostic::location`

### Stream B: Full attribute system

Extend `ir_attrs.rs` from current 6 variants to 15+:

| Variant | Purpose |
|---|---|
| `Bool(bool)` | ✅ exists |
| `Integer(i64, Type)` | ✅ exists |
| `Float(f64, Type)` | ✅ exists |
| `String(String)` | ✅ exists |
| `Array(Vec<Attribute>)` | ✅ exists |
| `Dictionary(Vec<(String, Attribute)>)` | ✅ exists |
| `DenseElements(Vec<u8>, Type, Vec<u64>)` | raw tensor data |
| `SparseElements(DenseIndices, Vec<u8>, Type, Vec<u64>)` | sparse tensor |
| `UnitAttr` | presence-only flag |
| `FlatSymbolRef(String)` | symbol name reference |
| `FlatSymbolRefArray(Vec<String>)` | multiple symbol refs |
| `SymbolName(String)` | symbol definition |
| `StridedLayout(Option<i64>, Vec<i64>)` | memref stride |
| `AffineMap(AffineExpr)` | affine map value |

### Stream C: Op traits

`src/traits.rs` — bitflag-based trait system on operations.

```rust
bitflags! {
    pub struct OpTraits: u64 { ... }
}
```

### Stream D: Op interfaces

`src/interfaces.rs` — component-based interface system.

```rust
pub struct Interface(pub &'static str); // registered name
```

### Stream E: MemRef type + affine maps

Extend `ir_types.rs` with `Type::MemRef(MemRefType)` + `AffineMap` value type.

### Stream F: Diagnostic infrastructure

`src/diagnostic.rs` — `Diagnostic` resource, `emitError()`/`emitWarning()` on ops.

### Stream G: Pass manager

`src/pass_manager.rs` — `Pass` trait, `PassPipeline` resource, pass statistics.

### Stream H: SSA dominance verifier

ECS system that walks all ops and verifies every `Uses` entry satisfies dominance.
