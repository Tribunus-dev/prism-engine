# Heterogeneous Hardware Validation Tests

Architectural assertion tests for PRISM-CIMAGE-HETEROGENEOUS-COMPILATION-0001.
Each test targets a specific claim from the compiler design discussion and produces
decisive pass/fail or measurement evidence on real Apple Silicon hardware.

## Test Matrix

| ID | Assertion | Compiler Decision | Hardware |
|----|-----------|-------------------|----------|
| A | Core ML masked SDPA produces correct results on ANE | Whether compiler must decompose SDPA or trust Core ML | M1+ macOS |
| B | ANE SRAM spill causes ~30% throughput cliff | Whether cost model needs nonlinear memory term | M1+ macOS |
| C | Same IOSurface bytes produce identical math across Metal/ANE/CPU | Whether SharedFp16ActivationContract needs index-mapping proof | M1+ macOS |
| D | Multi-output ANE programs require uniform buffer sizes (Core ML vs Orion) | Whether slot padding is required at cimage level | M1+ macOS |
| E | IOSurface minimum allocation floor (Core ML vs Orion) | Whether slot validate must enforce min bytes | M1+ macOS |
| F | constexpr_lut_to_dense materialization cost breakdown | Whether VariantPrepareCost is justified | M1+ macOS |
| G | Metal LUT dequant shader can match ANE palette-decompress throughput | Whether WeightEncoding::MetalShaderLutDequant is viable | M1+ macOS |

Run: `cargo test --test heterogeneous_hardware_validations --features prism-backend -- --nocapture`

## Pass/Fail Interpretation

Each test file documents:
- **Null hypothesis**: the assumption being tested
- **Compiler consequence**: what the compiler should do if the null holds vs. if it does not
- **Required evidence**: specific measurement threshold that constitutes a pass
