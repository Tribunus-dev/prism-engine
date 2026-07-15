# Wave 13: ECS-Native IR Kernel — Porting Guide

**Status:** Draft (pre-implementation)
**Dependency:** Waves 1-12 complete (canonical World, crate split, IR benchmark)
**Owner:** kernel

## 1. Scope

Deliver a working ECS-native IR kernel that can represent structured ops, regions, SSA values, types, and attributes; serialize and deserialize deterministically; and produce the same output as upstream MLIR on a single `arith.addf` + `func.return` test.

**Non-goals (this wave):**
- Dialect definitions (Wave 14)
- MLIR pass lowering (Wave 15)
- Hardware execution (Wave 16)
- Evolutionary search integration (post-Waves 15-16)

## 2. Crate structure — `prism-ecs-ir`

New workspace member `crates/prism-ecs-ir/` with ECS-native IR types. No dependency on `compute-core`. Depends on:
- `prism-ecs-core` (World, Entity, Component, Resource)

### File map

| File | Contents | Agent |
|---|---|---|
| `Cargo.toml` | Workspace member manifest | (scaffold) |
| `src/lib.rs` | Crate root + re-exports | (scaffold) |
| `src/op.rs` | `OpaqueOp` trait, `Op` trait, `OpInfo` registry | A1 |
| `src/region.rs` | `Region` component, region kind, region entity | A1 |
| `src/block.rs` | `Block` entity, terminator, block args | A1 |
| `src/value.rs` | `Value` enum (OpResult, BlockArgument), SSA use-list | A1 |
| `src/ir_types.rs` | `Type` trait, `TypeKind` enum, builtin types (Integer, Float, Index, None, Function, Tensor, Vector, RankedTensor, UnrankedTensor, Complex) | A2 |
| `src/ir_attrs.rs` | `Attribute` trait, `AttributeKind` enum, builtin attrs (String, Integer, Float, Array, DenseElements, SparseElements, Dictionary, Bool) | A2 |
| `src/symbol_table.rs` | `SymbolTable` resource, symbol references | A2 |
| `src/serde.rs` | Deterministic binary + JSON serialization/deserialization, round-trip | A2 |
| `src/builder.rs` | `OpBuilder` — entity-scoped op construction, operand attachment | A3 |
| `src/rewrite_driver.rs` | `RewriteDriver`, `PatternRewriter` trait, pattern application | A3 |
| `src/type_inference.rs` | `TypeInference` trait, inference registry | A3 |
| `src/dominance.rs` | Dominance analysis (dominator tree, dominance frontier) | A3 |
| `src/evolution.rs` | Mutation operator scaffolding for per-tensor (format, operation) search (wired later) | A3 |

### Sub-wave 1 files (A1): op.rs, region.rs, block.rs, value.rs
### Sub-wave 2 files (A2): ir_types.rs, ir_attrs.rs, symbol_table.rs, serde.rs
### Sub-wave 3 files (A3): builder.rs, rewrite_driver.rs, type_inference.rs, dominance.rs, evolution.rs

## 3. Framework contracts — pattern mappings

### 3.1 OpaqueOp trait (`op.rs`)

Upstream MLIR concept: `mlir::Operation` — dynamically-typed operation with name, operands, results, attributes, regions.

ECS-native mapping:

```rust
/// Trait for all ECS-native operations. Every Op is also an Entity.
pub trait OpaqueOp: Component + NamedOp + std::fmt::Debug {
    /// The operation name (e.g. "arith.addf").
    fn op_name(&self) -> &'static str;

    /// Verify operation invariants. Returns Ok(()) or a list of verifier errors.
    fn verify(&self, _context: &OpVerifierContext) -> Result<(), Vec<String>> { Ok(()) }

    /// Infer result types (default: no inference).
    fn infer_result_types(&self, _operand_types: &[Type]) -> Option<Vec<Type>> { None }
}
```

**Entity-per-op model** (selected by Wave 12 benchmark if ≤3x compact memory overhead).

Each operation is an Entity with:
- A component implementing `OpaqueOp` (the operation data)
- A `Results` component (list of Value entities this op produces)
- Optionally a `Region` component for region-bearing ops (func, scf.for, etc.)

```rust
/// Component marking an entity as an operation.
#[derive(Component)]
pub struct OpMarker;

/// Operands: references to producing Value entities.
#[derive(Component)]
pub struct Operands(pub Vec<Entity>);

/// Results: Value entities produced by this operation.
#[derive(Component)]
pub struct Results(pub Vec<Entity>);

/// Operation name (string form, e.g. "arith.addf").
#[derive(Component)]
pub struct OpName(pub String);

/// Attributes attached to this operation.
#[derive(Component)]
pub struct OpAttributes(pub Vec<Attribute>);

/// Successor blocks (for terminator ops).
#[derive(Component)]
pub struct Successors(pub Vec<Entity>);
```

### 3.2 Region (`region.rs`)

Upstream: `mlir::Region` contains a list of Blocks.

ECS-native: Region is an Entity containing Block entities.

```rust
/// The kind of region (for verification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    Graph,     // SSA graph region (no block arguments)
    SSACFG,    // Structured control-flow region
}

/// A region entity marker.
#[derive(Component)]
pub struct RegionMarker;

/// Blocks contained in this region.
#[derive(Component)]
pub struct RegionBlocks(pub Vec<Entity>);
```

### 3.3 Block (`block.rs`)

Upstream: `mlir::Block` — list of operations with block arguments.

```rust
/// A block entity marker.
#[derive(Component)]
pub struct BlockMarker;

/// Block arguments (Value entities for this block's entry values).
#[derive(Component)]
pub struct BlockArguments(pub Vec<Entity>);
```

### 3.4 Value (`value.rs`)

Upstream: `mlir::Value` — either an OpResult or a BlockArgument.

```rust
/// A value in the IR graph. Each value is an Entity.
#[derive(Component)]
pub enum ValueKind {
    OpResult(Entity),        // producing operation
    BlockArgument(Entity),   // owning block, index
}

/// The type of this value.
#[derive(Component)]
pub struct ValueType(pub Type);

/// Use-list: entities (operations) that use this value as an operand.
#[derive(Component)]
pub struct Uses(pub Vec<Entity>);
```

### 3.5 Type system (`ir_types.rs`)

Upstream: `mlir::Type` — ordered, uniqued, polymorphic type hierarchy.

ECS-native: `Type` is an enum (not entity-per-type — types are value objects).

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    Integer(IntegerType),
    Float(FloatType),
    Index,
    NoneType,
    Function(FunctionType),
    Tensor(TensorType),
    Vector(VectorType),
    Complex(ComplexType),
    // ... extensible via dynamic dispatch for dialect types
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegerType {
    pub width: u32,           // 1, 8, 16, 32, 64, 128
    pub signedness: Signedness,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatType {
    pub kind: FloatKind,      // F16, BF16, F32, F64, F8E4M3, F8E5M2
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Vec<u64>,
    pub element_type: Box<Type>,
    pub encoding: Option<Attribute>,
}
```

### 3.6 Attribute system (`ir_attrs.rs`)

Upstream: `mlir::Attribute` — polymorphic, uniqued attribute objects.

ECS-native: `Attribute` is an enum.

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Bool(bool),
    Integer(i64, Type),       // value + type for signedness/width
    Float(f64, Type),
    String(String),
    Array(Vec<Attribute>),
    Dictionary(Vec<(String, Attribute)>),
    DenseElements(Vec<u8>, Type, Vec<u64>),  // raw bytes, element type, shape
    SparseElements(Vec<u64>, Vec<u8>, Type, Vec<u64>), // indices, values, element type, shape
    // extensible for dialect-specific attrs
}
```

### 3.7 SymbolTable (`symbol_table.rs`)

Upstream: `mlir::SymbolTable` — maps symbol names to symbol-defining operations.

ECS-native: Resource on the World.

```rust
/// Global symbol table resource.
pub struct SymbolTable {
    symbols: HashMap<String, Entity>,  // name → defining op entity
}

impl Resource for SymbolTable {}

impl SymbolTable {
    pub fn lookup(&self, name: &str) -> Option<Entity>;
    pub fn insert(&mut self, name: String, entity: Entity) -> Result<(), SymbolConflict>;
    pub fn erase(&mut self, name: &str) -> Option<Entity>;
}
```

### 3.8 Serialization (`serde.rs`)

Deterministic JSON and binary serialization for the ECS-native IR. The IR module exports a deterministic form: ops are topologically sorted, entities are referenced by stable IDs, types and attrs are inline value objects.

```rust
/// Serialized form of an IR module.
#[derive(Serialize, Deserialize)]
pub struct IrModuleSnapshot {
    pub ops: Vec<OpEntry>,           // topologically sorted
    pub regions: Vec<RegionEntry>,
    pub blocks: Vec<BlockEntry>,
    pub values: Vec<ValueEntry>,
}

/// Deterministic round-trip: serialize(deserialize(bytes)) == same bytes
pub fn to_json(module: Entity, world: &World) -> Result<String>;
pub fn from_json(json: &str, world: &mut World) -> Result<Entity>;
pub fn to_binary(module: Entity, world: &World) -> Result<Vec<u8>>;
pub fn from_binary(bytes: &[u8], world: &mut World) -> Result<Entity>;
```

### 3.9 OpBuilder (`builder.rs`)

Upstream: `mlir::OpBuilder` — creates ops at insertion points, handles SSA construction.

ECS-native: Spawns entities within the World transaction.

```rust
pub struct OpBuilder<'a> {
    world: &'a mut World,
    insertion_block: Option<Entity>,
    txn: Option<&'a mut WorldTxn>,
}

impl OpBuilder {
    pub fn create_op(&mut self, name: &str, operands: &[Entity], 
                     attributes: &[Attribute], results: &[Type]) -> Result<Entity>;
    pub fn set_insertion_point(&mut self, block: Entity);
    pub fn create_block(&mut self, args: &[Type]) -> Result<Entity>;
    pub fn create_region(&mut self) -> Result<Entity>;
}
```

### 3.10 RewriteDriver (`rewrite_driver.rs`)

Upstream: `mlir::RewriterBase` / `mlir::PatternRewriter` — applies rewrite patterns with folding, erasure, and replacement.

```rust
/// Pattern rewriter — applies modifications to the IR.
pub trait PatternRewriter {
    fn replace_op(&mut self, op: Entity, new_ops: &[Entity]) -> Result<()>;
    fn erase_op(&mut self, op: Entity) -> Result<()>;
    fn replace_all_uses_with(&mut self, old: Entity, new: Entity) -> Result<()>;
    fn insert_op_before(&mut self, anchor: Entity, op: Entity) -> Result<()>;
    fn insert_op_after(&mut self, anchor: Entity, op: Entity) -> Result<()>;
}

/// A rewrite pattern: matches a specific op and rewrites it.
pub trait RewritePattern: Send + Sync {
    fn match_and_rewrite(&self, op: Entity, rewriter: &mut dyn PatternRewriter,
                         world: &mut World) -> Result<bool>;
}

/// Rewrite driver — applies a set of patterns until fixpoint.
pub struct RewriteDriver {
    patterns: Vec<Box<dyn RewritePattern>>,
}

impl RewriteDriver {
    pub fn add_pattern(&mut self, pattern: Box<dyn RewritePattern>);
    pub fn apply(&self, world: &mut World, root_op: Entity) -> Result<u64>; // number of rewrites
}
```

### 3.11 TypeInference (`type_inference.rs`)

Upstream: `mlir::InferTypeOpInterface` — ops implement type inference.

```rust
/// Type inference registry — maps op names to inference functions.
pub struct TypeInferenceRegistry {
    inferers: HashMap<&'static str, Box<dyn Fn(&[Type], &[Attribute]) -> Option<Vec<Type>>>>,
}

impl TypeInferenceRegistry {
    pub fn register(&mut self, op_name: &'static str, 
                    inferer: Box<dyn Fn(&[Type], &[Attribute]) -> Option<Vec<Type>>>);
    pub fn infer(&self, op_name: &str, operand_types: &[Type],
                 attributes: &[Attribute]) -> Option<Vec<Type>>;
}
```

### 3.12 Dominance analysis (`dominance.rs`)

Upstream: `mlir::DominanceInfo` — standard dominator tree construction.

```rust
pub struct DominanceAnalyzer;

impl DominanceAnalyzer {
    /// Compute dominator tree for a region.
    pub fn compute_dominators(&self, region: Entity, world: &World) -> DominatorTree;
    /// Compute dominance frontier for each block.
    pub fn compute_frontier(&self, region: Entity, world: &World) -> DominanceFrontier;
}

pub struct DominatorTree {
    pub immediate_dominators: HashMap<Entity, Entity>,   // block → idom(block)
}

pub struct DominanceFrontier {
    pub frontiers: HashMap<Entity, Vec<Entity>>,         // block → DF(block)
}
```

## 4. 3-file trial

Before scaling to the full module, prove the pattern with exactly 3 files:

1. `src/op.rs` — OpaqueOp trait, OpMarker, Operands, Results, OpName, OpAttributes
2. `src/value.rs` — ValueKind, ValueType, Uses
3. `src/ir_types.rs` — Type enum + builtin types

Trial passes when:
- `cargo check -p prism-ecs-ir` compiles
- Test: create an `arith.addf` op with two float operands, query results, verify type

## 5. Differential test strategy

For each framework contract:

| Contract | Test pattern |
|---|---|
| Op creation | Create op via OpBuilder, verify entity exists, has correct components |
| Region/Block nesting | Create region → create block → attach ops, verify traversal |
| SSA use-def | Create value → create consuming op → verify uses and def |
| Type equality | IntegerType(32, Signed) == IntegerType(32, Signed); != IntegerType(64, Signed) |
| Attribute round-trip | Serialize Attribute → deserialize → match original |
| Serialization round-trip | Build module → to_json → from_json → verify entity structure matches |
| Rewrite | Match arith.addf(x, 0) → erase, verify x has one fewer use |
| Dominance | Block A dominates Block B → verify dominance query |
| SymbolTable | Insert symbol → lookup → erase → verify absent |

## 6. Work isolation

Files within each sub-wave are independent except `lib.rs` (re-exports). Sub-waves are sequential: A2 depends on types from A1 (Type/Attribute/Value/Op/Region/Block), A3 depends on A1+A2.

No two agents edit the same file. Shared type contracts (Type enum, Op trait, Value enum) are frozen before sub-wave 2 fan-out.

## 7. Gate

1. `cargo check -p prism-ecs-ir` compiles with 0 errors
2. All framework contract tests pass
3. A test module is serialized `to_json`, deserialized `from_json`, and entity structure matches
4. `arith.addf` + `func.return` module round-trips deterministically
5. At least one rewrite pattern applies successfully (e.g., erase no-op)
