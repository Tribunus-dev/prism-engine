# Wave 17 Assessment: IREE Flow → Stream → HAL

**Status:** Assessment (no code changes)
**Finding:** IREE is NOT integrated into the workspace. No `iree` crate, no IREE build output, no C bindings.

## Existing MLIR infrastructure
- `melior` (Rust MLIR bindings) available behind `mlir-runtime` feature in compute-core
- `MlirExecutionContract` exists in compute-core/src/ecs/core/compile_pipeline.rs
- Works with melior for `arith.addf` → `func.return` lowering
- No IREE HAL or Stream dialect lowering exists

## Path forward
1. Add IREE HAL C API bindings via `iree-sys` or custom FFI
2. Build the `iree-compiler` library and link via CMake/native
3. Wire HAL dispatch into the ECS-native pipeline
4. This is a multi-wave effort, not a single wave

## Recommendation
Defer Wave 17 until the ECS-native lowerer produces full compiled programs. The Metal codegen in Wave 16 is the fast path — use it as the primary GPU backend and defer IREE HAL until Triton/IREE co-design is settled.
