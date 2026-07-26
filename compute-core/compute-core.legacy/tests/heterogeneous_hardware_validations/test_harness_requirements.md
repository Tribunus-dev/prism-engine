# Cross-Cutting: Test Harness and Instrumentation Requirements

All seven hardware validation tests share common infrastructure. This document
specifies the harness needed to run them.

## Common Infrastructure

### IOSurface Creation and Management

Every test needs IOSurface-backed FP16 tensor I/O. The harness provides:

```rust
pub struct TestSurface {
    surface: IOSurfaceRef,
    base_address: *mut c_void,
    byte_size: usize,
}

impl TestSurface {
    /// Create an IOSurface with the given byte size.
    /// NSIGraphicsContext and IOSurface are available on macOS only.
    pub fn new(byte_size: usize) -> Self;

    /// Read a half-precision element at the given byte offset.
    pub fn read_f16(&self, offset: usize) -> f16;

    /// Write a half-precision element.
    pub fn write_f16(&self, offset: usize, value: f16);

    /// Fill with a deterministic test pattern (for layout identity tests).
    pub fn fill_pattern(&self, hidden: usize, seq: usize);
}
```

### Core ML Model Lifecycle

```rust
pub struct TestMlModel {
    model: MlModel,
    compute_policy: ComputePolicy,
}

impl TestMlModel {
    /// Compile a MIL program to .mlmodelc, load with the given compute policy.
    pub fn compile_and_load(
        mil_text: &str,
        weights: &[WeightDescriptor],
        policy: ComputePolicy,
    ) -> Result<Self>;

    /// Run a prediction. input_surfaces and output_surfaces are vectors of
    /// (feature_name, TestSurface) pairs.
    pub fn predict(
        &self,
        inputs: &[(&str, &TestSurface)],
        outputs: &[(&str, &TestSurface)],
    ) -> Result<Duration>;
}
```

### Metal Kernel Dispatch

```rust
pub struct TestMetalKernel {
    pipeline: MTLComputePipelineState,
}

impl TestMetalKernel {
    /// Load a Metal library from source, create pipeline state.
    pub fn from_source(source: &str, function: &str) -> Result<Self>;

    /// Dispatch with buffers and textures bound to IOSurfaces.
    pub fn dispatch(
        &self,
        surfaces: &[(&IOSurfaceRef, MTLResourceUsage)],
        grid: (u32, u32, u32),
        threadgroup: (u32, u32, u32),
    ) -> Result<Duration>;
}
```

### MLComputePlan Operation Profiling

```rust
/// For a loaded .mlmodelc, query which device each MIL operation executes on.
pub fn profile_device_usage(model_path: &Path) -> Result<Vec<OperationDeviceAssignment>>;

pub struct OperationDeviceAssignment {
    pub operation_name: String,
    pub operation_type: String,
    pub device: ComputeDevice,  // CPU, GPU, NeuralEngine, or Unknown
}
```

### Memory Footprint Measurement

```rust
/// Measure current process resident set size in bytes.
pub fn resident_bytes() -> Result<u64>;

/// Measure the memory delta caused by loading a model.
pub fn footprint_delta<F, T>(f: F) -> Result<(T, u64)>
    where F: FnOnce() -> Result<T>;
```

## Test Execution Requirements

### Test Runner

A single binary `cargo test --test heterogeneous_hardware_validations` that:

1. Checks hardware prerequisites (Apple Silicon, macOS 14+, Core ML available)
2. Prints a test plan before execution
3. Runs each test A–G
4. Collects results into `test_results/` directory
5. Generates a summary JSON `test_results/summary.json`

### Result Schema

```json
{
  "test_id": "A",
  "timestamp": "2026-06-25T12:00:00Z",
  "hardware": {
    "soc": "M4 Max",
    "macos_version": "14.5",
    "coreml_version": "7.3.0",
    "metal_version": "3.1",
    "ane_count": 1
  },
  "assertion": "Core ML masked SDPA produces correct results on ANE",
  "null_hypothesis": "CPU attention == ANE attention for all seq",
  "results": [
    {
      "variant": "A_ane_opaque",
      "seq": 1024,
      "head_dim": 128,
      "n_heads": 32,
      "max_abs_error": 0.0002,
      "cosine_similarity": 0.9999,
      "match_rate": 0.998,
      "device": {"scaled_dot_product_attention": "CPU"},
      "mean_latency_us": 850
    }
  ],
  "conclusion": "Null hypothesis holds — Core ML handles masked SDPA correctly",
  "compiler_action": "Classify as CorrectnessQualifiedButPlacementUnknown"
}
```

### CI Integration

These tests are hardware-dependent and cannot run in CI without Apple Silicon
runners. The test harness should:

1. Detect whether actual hardware is available (ANE exists, Metal available)
2. If not: emit a structured "skipped — no Apple Silicon" report
3. If yes: run full test suite and archive results

Results should be stored as CI artifacts for offline analysis and regression
tracking across macOS and SoC generations.

## Prioritization

Not all 7 tests are equally important for the next compiler decision. Priority order:

| Priority | Test | Why |
|----------|------|-----|
| P0 | A: SDPA fidelity | Determines correctness — highest risk if wrong |
| P0 | C: Layout identity | Determines whether SharedFp16ActivationContract needs IndexMapping |
| P0 | F: Prepare vs steady-state | Determines cost model structure — affects every variant decision |
| P1 | B: SRAM cliff | Critical for cost model accuracy, but less urgent than P0 |
| P1 | E: IOSurface min size | Safety constraint, low implementation cost |
| P2 | D: Multi-output uniform | Likely handled by Core ML, low risk |
| P2 | G: Metal LUT dequant | Forward-looking; compiler should not depend on this yet |
