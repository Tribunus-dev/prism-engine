# Wave 20 Assessment: TableGen absorption decision

**Status:** Assessment (no code changes)
**Finding:** No `mlir-tblgen` or `llvm-tblgen` build tool in the workspace

## Options for dialect definition
1. **Manual Rust components** ✅ (used in Waves 13-16 — proven pattern)
2. **TableGen → Rust codegen** — would require building `mlir-tblgen` from LLVM source
3. **melior MLIR bindings** — for round-trip testing with upstream MLIR

## Recommendation
Continue with manual Rust component definitions. The proven pattern from Waves 13-16 is maintainable and doesn't require a TableGen build. Revisit when `mlir-tblgen` is reliably available as a build tool.
