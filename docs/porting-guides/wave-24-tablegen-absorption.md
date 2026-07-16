# Wave 24: TableGen Absorption — Rust ECS Code Generator

**Status:** Draft (pre-implementation)
**Dependency:** Waves 13-23 (ECS-native IR kernel, dialects, codegen, assessments)
**Owner:** kernel

## 1. Scope

Deliver a Rust crate `prism-tblgen` that parses MLIR-style `.td` (TableGen) files and generates Rust ECS component definitions matching the pattern proven in `arith.rs` / `func.rs` / `scf.rs` / `linalg.rs`.

This replaces manual dialect definitions with a code generator. The same `.td` file that an upstream MLIR developer writes becomes the source of truth for our ECS dialect module.

## 2. Crate structure — `prism-tblgen`

New workspace member `crates/prism-tblgen/`. CLI tool + library.

### File map

| File | Contents | Sub-wave |
|---|---|---|
| `Cargo.toml` | depends on `serde`, `serde_json`, `logos` or `nom` for lexing | A1 |
| `src/lib.rs` | Crate root — re-exports parser, IR, generator | A1 |
| `src/lexer.rs` | Tokenizer for `.td` files: identifiers, strings, dags, integers, `def`, `class`, `let`, `foreach`, `multiclass`, `dag`, `list`, `bits`, `code` | A1 |
| `src/parser.rs` | Recursive descent parser → `TdDocument` AST | A1 |
| `src/ast.rs` | AST types: `TdDocument`, `Record`, `Class`, `Multiclass`, `Def`, `Defm`, `TemplateArg`, `LetBlock`, `DagArg`, `Value`, `TypeConstraint` | A1 |
| `src/resolve.rs` | Template instantiation, class inheritance resolution, multiclass expansion | A2 |
| `src/annotations.rs` | Prism-specific annotation extraction: `search_space`, `allowed_formats`, `allowed_operations`, `allowed_layouts`, `hardware_capability` | A2 |
| `src/emit.rs` | Rust code emission → `OpName`, `ArithOpKind`, `verify_*`, `infer_*`, `register_*` functions | A2 |
| `src/cli.rs` | Subcommand: `generate` (input `.td` → output `.rs`) | A3 |
| `tests/arith_td.rs` | Integration test: parse `arith_ops.td`, generate Rust, match expected output | A3 |

## 3. TableGen subset

The parser handles the majority of TableGen — enough to parse MLIR's dialect `.td` files and Prism's extended evolutionary-search annotations:

### Types (lex + parser)

| Type | Example |
|---|---|
| `int` | `42`, `-1` |
| `bit` | `0`, `1` |
| `string` | `"floating-point addition"` |
| `code` | `[{...}]` |
| `list<type>` | `list<int>`, `list<OpTrait>` |
| `bits<n>` | `bits<32>` |
| `dag` | `(ins FloatLikeType:$lhs, FloatLikeType:$rhs)` |
| `ClassType` | `Op`, `Arith_Op`, `Trait` |

### Statements

| Statement | Example |
|---|---|
| `def` | `def ADDFOp : Arith_Op<"addf"> { ... }` |
| `class` | `class Arith_Op<string mnemonic> : Op<...> { ... }` |
| `multiclass` | `multiclass ArithIntOp<string mnemonic, string kind> { ... }` |
| `defm` | `defm ADDI : ArithIntOp<"addi", "Add">;` |
| `foreach` | `foreach <int n> = [0, 1, 2] in { ... }` |
| `let` | `let hasVerifier = 1;` |
| `include` | `include "mlir/Interfaces/InferTypeOpInterface.td"` |

### Expressions

| Expression | Example |
|---|---|
| Identifier | `ADDFOp`, `FloatLikeType` |
| Dag | `(ins $lhs:$rhs)` |
| List | `[F32, F64]` |
| Braces | `{ let a = 1; let b = 2; }` |
| String interpolation | `"arith.${mnemonic}"` |
| Bang operators | `!cast<Op>(NAME)`, `!eq(a, b)`, `!cond(...)` |

### Prism extensions

Additional annotations carried in `.td` `let` blocks or as trait-like declarations:

```tablegen
// Evolutionary search space for a tensor
let search_space = (per_tensor<
  formats: [Fp16, Bf16, Int8, Int4, Nf4, Nf8, Ternary158, Binary1],
  operations: [Matmul, TernaryGemm, BinaryPopcountGemm],
  layouts: [Blocked<16, 32>, Mma<v2>, Mma<v3, fp8>]
>);

// Hardware capability requirements
let hardware_capability = (requires<
  min_compute_capability: 8.0,
  shared_memory_per_block: 49152,
  tensor_core: "fp16"
>);

// Per-tensor evolution policy
let evolution_policy = (mutate<
  format: cycle,           // cycle through allowed formats
  operation: co_mutate,    // mutate format triggers operation change
  layout: independent      // layout mutates independently
>);
```

## 4. Sub-waves

### Sub-wave A1: Lexer + Parser + AST (3 files)

The core parsing infrastructure. Tokenize `.td` content, parse into a `TdDocument` AST that represents the full structure: records, classes, multiclasses, defs, defms, lets, dags.

- Lexer: `logos` or `nom` — TableGen has a simple lexical structure (C-family identifiers, `#` comments, `/* */` block comments)
- Parser: recursive descent. TableGen is `LL(k)` — no operator precedence headaches because dag/expression syntax is bracket-delimited.
- AST: `enum TdNode { Record { name, base_class, body }, Class { name, template_args, superclasses, body }, ... }`

### Sub-wave A2: Resolution + Annotations + Emission

- Template instantiation: substitute template arguments through the class hierarchy
- Inheritance resolution: flatten class hierarchies into concrete records
- Annotation extraction: parse Prism-specific `let` blocks into structured metadata
- Rust emission: walk resolved records, emit `OpName`, `Kind`, `verify`, `infer`, `register` functions

### Sub-wave A3: CLI + Integration Tests

- `prism-tblgen generate input.td output.rs` — standalone CLI
- Integration test: take `ArithOps.td` (a manual copy of the upstream file), generate Rust, verify the generated code compiles and matches expected patterns
- `build.rs` integration: optional `build.rs` in `prism-ecs-ir` that invokes `prism-tblgen` on `.td` files in `templates/`

## 5. Gate

1. `prism-tblgen` parses a representative `.td` file (arith ops) with 0 errors
2. Generated Rust code compiles (`cargo check -p prism-ecs-ir`) and tests pass
3. Generated code for `arith.addf` is semantically equivalent to the hand-written `arith.rs`
4. Prism annotations (`search_space`, `evolution_policy`) survive the parse→emit round-trip
5. `defm` + `foreach` expand correctly (test with a multiclass that generates 4 ops)
