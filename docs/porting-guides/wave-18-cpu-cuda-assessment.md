# Wave 18 Assessment: CPU differential gates + CUDA HAL backend

**Status:** Assessment (no code changes)
**Finding:** No CUDA infrastructure in workspace (`nvidia`, `cuda` grep returns empty).

## Existing CPU infrastructure
- `backend-cpu` feature flag exists in compute-core
- `candle-cpu` feature for candle-based CPU execution
- CPU differential testing infrastructure exists in `compute-core/tests/`

## Path forward
1. Add CUDA backend via existing `candle-cu` or custom Metal-level binding
2. Wire CPU differential gates into the ECS-native lowerer
3. This requires CUDA toolkit and NVCC — needs build environment setup

## Recommendation
Diff testing between ECS-native IR and existing CPU evaluator is feasible now via the `backend-cpu` feature. CUDA HAL is deferred until GPU CI runners are available.
