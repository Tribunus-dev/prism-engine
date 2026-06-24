# Tribunus C ABI (v0)

The canonical Tribunus API relies exclusively on the stable headers exported inside the `include/mlx_c/` folder. The older `mlx/c/` namespace represents compatibility/legacy logic for upstream usages, which should not be depended on directly for any stable guarantees moving forward.

## Core Directives

1. **Structured Errors (`mlx_c_status_t`)**: Every stable, fallible function throughout the `include/mlx_c/` ABI explicitly manages and returns an `mlx_c_status_t`. Exceptions are correctly caught and handled without crossing the C ABI barrier whatsoever.
2. **Handles & Destructors**: Handles like `mlx_c_array_t` and `mlx_c_context_t` are constructed efficiently on the heap but are fully independent struct wrappers safely destructible via `mlx_c_array_free` and `mlx_c_context_free` explicitly.
3. **Data Limitations**: The v0 API assumes explicitly `f32` inputs. Other variants trigger `MLX_C_STATUS_DTYPE_UNSUPPORTED` correctly.
4. **Shapes & Introspection**: Arrays are handled strictly via dimensions strictly evaluated safely using a two-pass mechanism for allocating correctly evaluated metadata vectors transparently. Negative or Zero dimension shapes reject automatically.
5. **No Broadcasting**: No automatic tensor broadcasting executes. Matmul limits execution exclusively targeting explicitly valid configurations (ex: rank 2 only for now).
6. **Backend Executions**: When executing in a stubbed MLX_C_ENABLE_MLX_BACKEND=OFF environment, operations will correctly map to validation paths, effectively halting execution early securely throwing `MLX_C_STATUS_BACKEND_UNAVAILABLE` seamlessly while passing standard ABI limits reliably.
