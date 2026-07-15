# Wave 19 Assessment: Portable targets (Metal, CUDA, ANE, Vulkan, ROCm)

**Status:** Assessment (no code changes)

## Supported targets
- **Metal**: ✅ Wave 16 codegen works. prism-metal-runtime supports PSO caching and dispatch
- **ANE**: ⚠️ AneExecutor is behind `ane-executor` feature in prism-backend
- **CUDA**: ❌ No CUDA deps. Requires CUDA toolkit
- **Vulkan**: ❌ No Vulkan deps in workspace
- **ROCm**: ⚠️ Feature flags exist (`rocm-probe`, `amd-rocm`) but no integration

## Recommendation
Primary target stays Metal (Apple Silicon). CUDA/Vulkan/ROCm are deferred duntil cross-platform CI is available. ANE integration through the existing executor path.
