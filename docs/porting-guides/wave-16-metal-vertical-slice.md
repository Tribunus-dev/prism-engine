# Wave 16: Vertical slice through existing evaluators to Metal

**Status:** Draft (pre-implementation)
**Dependency:** Waves 13-15 complete (ECS-native IR, arith+, lowering)
**Owner:** kernel

## 1. Scope

Lower a `linalg.matmul` from ECS-native IR to Metal GPU kernel source, compile it via the existing Metal runtime, and dispatch on real Apple Silicon hardware.

This is the first end-to-end proof: ECS-native model → run on GPU.

## 2. Design

### 2.1 Metal codegen

Add `src/codegen_metal.rs` to `prism-ecs-ir` — emits Metal Shading Language source from ECS-native IR ops.

```
linalg.matmul(A[M,K], B[K,N], C[M,N]) →
  kernel void matmul_MxKxN(...) {
    for (uint i = tid.y; i < M; i += threads.y)
      for (uint j = tid.x; j < N; j += threads.x) {
        float acc = 0;
        for (uint k = 0; k < K; k++)
          acc += A[i*K+k] * B[k*N+j];
        C[i*N+j] = acc;
      }
  }
```

### 2.2 Pipeline

1. `metal_lower_from_ir(world, root_op)` → Metal source string
2. Call through to `prism-metal-runtime` to compile PSO
3. Dispatch with concrete tensors
4. Validate result matches CPU reference

## 3. File map

| File | Contents |
|---|---|
| `src/codegen_metal.rs` | Metal codegen from ECS-native IR ops |

## 4. Gate

- `linalg.matmul` on f32[4,4] × f32[4,4] lowers to valid Metal source
- The Metal source compiles via `prism-metal-runtime` PSO cache
- GPU result matches CPU reference on a small test case
