# Wave 15: func + arith + scf + linalg → vector MLIR path

**Status:** Draft (pre-implementation)
**Dependency:** Waves 13-14 complete (ECS-native IR kernel + arith dialect)
**Owner:** kernel

## 1. Scope

Implement func, scf, and linalg dialect operations as ECS components, then build a lowering pipeline from linalg.matmul → scf.for → vector operations.

This is the first proof that the ECS-native model can run a real compiler pipeline — not just single ops.

## 2. Dialects

| Dialect | Ops | Verification |
|---|---|---|
| **func** | func.func, func.return, func.call | func.func: region body with terminator. func.return: 0+ operands |
| **scf** | scf.for, scf.if, scf.yield, scf.while | scf.for: 4 operands (lb, ub, step, iter_args), region body. scf.if: condition, 2 regions |
| **linalg** | linalg.matmul, linalg.batch_matmul, linalg.fill, linalg.generic | linalg.matmul: 3 operands (A, B, C). Structured op with indexing maps |

## 3. File map

| File | Contents | Agent |
|---|---|---|
| `src/func.rs` | func dialect ops (FuncOp, ReturnOp, CallOp) as ECS components | A1 |
| `src/scf.rs` | scf dialect ops (ForOp, IfOp, YieldOp, WhileOp) | A2 |
| `src/linalg.rs` | linalg dialect ops (MatmulOp, BatchMatmulOp, FillOp) | A3 |
| `src/lowering.rs` | Lowering patterns: linalg.matmul → scf.for → vector ops | A4 |

All files follow the same pattern as arith.rs: `ArithOpKind` enum, `ArithOp` component, verification functions, type inference.

## 4. Lowering pipeline

linalg.matmul(A[M,K], B[K,N], C[M,N]) → scf.for loops:

```rust
fn lower_matmul(contract: &MlirExecutionContract) -> Result<Entity, String>
```

This uses the RewriteDriver and PatternRewriter from Wave 13.

## 5. Gate

- func.func with region body and func.return serializes and deserializes correctly
- scf.for with loop body round-trips
- linalg.matmul lowers to scf.for loops via RewriteDriver
- All 90 existing tests + new tests pass
