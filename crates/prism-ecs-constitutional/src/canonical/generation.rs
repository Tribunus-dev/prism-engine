//! `CimageGeneration` — one compiler invocation's resolved output.
//! Authority: one-generation-per-compilation.
//!
//! A `CimageGeneration` is the canonical output of one compiler
//! invocation: a resolved execution image with fully-specified
//! tensor, kernel, and engram bindings. Every binding carries
//! its own receipt chain. The data types live in
//! `prism_ecs_ir::cimage_types` (the source of truth for IR
//! primitives) and are re-exported here so the compiler pipeline
//! has a single, stable import path:
//! `prism_ecs_constitutional::canonical::CimageGeneration`.

pub use prism_ecs_ir::cimage_types::{
    CimageGeneration, EngramBinding, RegionId, RepresentationBinding,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_identity_re_exports_compile() {
        // Sanity check that the re-exported types are constructible
        // through the canonical surface. The body is intentionally
        // empty — the test is type-level, not value-level.
        let _region = RegionId(String::from("region-1"));
    }
}
