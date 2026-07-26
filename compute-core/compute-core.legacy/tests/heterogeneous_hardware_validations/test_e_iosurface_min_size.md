# Test E: IOSurface Minimum Size Floor

**Compiler decision**: Whether `validate_manifest` must enforce a minimum
IOSurface byte length, and what the threshold is.

## Null Hypothesis

Core ML's `cpuAndNeuralEngine` execution path accepts arbitrarily small
IOSurface allocations (down to 1 byte) without runtime failure.

## Experimental Design

### Step 1: Generate MIL programs with graduated surface sizes

Single matmul: `input[1, hidden, 1, 1] × weight[hidden, hidden] = output[1, hidden, 1, 1]`

Vary hidden to produce decreasing IOSurface byte sizes:
- hidden=4096 → 8192 bytes
- hidden=1024 → 2048 bytes
- hidden=256 → 512 bytes
- hidden=64 → 128 bytes
- hidden=32 → 64 bytes
- hidden=16 → 32 bytes
- hidden=8 → 16 bytes
- hidden=4 → 8 bytes
- hidden=1 → 2 bytes

### Step 2: Test each with Core ML

For each hidden size:
1. Compile .mlmodelc
2. Load with `computeUnits = .cpuAndNeuralEngine`
3. Run prediction
4. Record: success/failure, error message, latency (if successful)

### Step 3: Test with direct ANE path (Orion infrastructure)

If Orion is available, repeat Step 2 using Orion's direct `_ANECompiler` +
`_ANEClient` path to reproduce the original finding and compare.

### Step 4: Binary search for exact floor

Starting from the smallest size that succeeded in Step 2, binary-search
downward to find the exact minimum working IOSurface size for Core ML ANE
execution.

## Acceptance Criteria

| Result | Interpretation | Compiler Action |
|--------|----------------|-----------------|
| All sizes succeed with Core ML | Core ML IOSurface path has no minimum floor | No validation needed for Core ML path |
| Sizes below X fail with status=0x1d on Core ML path | Floor exists and is reproducible | `validate_manifest` rejects slots below X bytes |
| Only Orion direct path shows floor; Core ML succeeds | Floor is a private-API artifact, not a Core ML limitation | Document: floor is Orion-specific; Core ML safe at any size |
| Floor differs between SoC generations | Floor is hardware-dependent | Per-SoC floor table in compiler hardware model |

## Required Evidence

For each (hidden, compute_policy, route): success/failure, error code,
measured latency (successful runs only). Final binary search result:
minimum_working_bytes on each SoC tested.

## Hardware Requirements

- Apple Silicon Mac (ideally M1, M2, M3, M4 to check SoC dependence)
- macOS 14.0+

## Expected Outcomes (Hypothesis)

Core ML will succeed for all sizes. The ~49KB minimum is specific to Orion's
direct ANE path which bypasses Core ML's buffer management layer. Core ML
allocates its own minimum surface size internally regardless of the MIL
declared shape. The compiler does NOT need this constraint for the Core ML
path, but adding it as a belt-and-suspenders check (e.g., reject < 48KB)
is cheap and protects against future regression.
