//! Canonical schema ID constants for compilation-domain components (31-39).
//!
//! **Single authority:** owns the schema namespace allocation and the
//! `SCHEMA_*` integer constants for compilation-domain components. The
//! schema IDs are the durable contract between the constitutional crate
//! and any consumer that materializes, persists, or replays these
//! components. Bumping a schema ID is a wire-format break.
//!
//! Allocation table:
//!
//! | ID | Constant                     | Component type             |
//! |----|------------------------------|----------------------------|
//! | 31 | `SCHEMA_COMPILATION_JOB`     | `CompilationJob`           |
//! | 32 | `SCHEMA_JOB_INPUT`           | `JobInput`                 |
//! | 33 | `SCHEMA_JOB_CONFIG`          | `JobConfig`                |
//! | 34 | `SCHEMA_JOB_OUTPUT`          | `JobOutput`                |
//! | 35 | `SCHEMA_JOB_LIFECYCLE`       | `JobLifecycle`             |
//! | 36 | `SCHEMA_VALIDATION_RECEIPT`  | `ValidationReceipt`        |
//! | 37 | `SCHEMA_QUANTIZATION_PLAN`   | `QuantizationPlan`         |
//! | 38 | `SCHEMA_CIMAGE_PROMOTION`    | `CimagePromotion`          |
//! | 39 | `SCHEMA_QUANTIZATION_RESULT` | `QuantizationResultComponent` |
//!
//! Note: `QuantizedTensorSelectionComponent` lives at schema ID 40;
//! its constant is co-located with its type in [`super::quantization`].

use crate::types::ComponentSchemaId;

/// `CompilationJob` component schema. See [module docs](self).
pub const SCHEMA_COMPILATION_JOB: u64 = 31;
/// `JobInput` component schema. See [module docs](self).
pub const SCHEMA_JOB_INPUT: u64 = 32;
/// `JobConfig` component schema. See [module docs](self).
pub const SCHEMA_JOB_CONFIG: u64 = 33;
/// `JobOutput` component schema. See [module docs](self).
pub const SCHEMA_JOB_OUTPUT: u64 = 34;
/// `JobLifecycle` component schema. See [module docs](self).
pub const SCHEMA_JOB_LIFECYCLE: u64 = 35;
/// `ValidationReceipt` component schema. See [module docs](self).
pub const SCHEMA_VALIDATION_RECEIPT: u64 = 36;
/// `QuantizationPlan` component schema. See [module docs](self).
pub const SCHEMA_QUANTIZATION_PLAN: u64 = 37;
/// `CimagePromotion` component schema. See [module docs](self).
pub const SCHEMA_CIMAGE_PROMOTION: u64 = 38;
/// `QuantizationResultComponent` component schema. See [module docs](self).
pub const SCHEMA_QUANTIZATION_RESULT: u64 = 39;

/// Construct a typed [`ComponentSchemaId`] from a raw schema constant.
///
/// This is the single canonical conversion point: every place in the
/// compilation sub-modules that needs a [`ComponentSchemaId`] should
/// call this helper rather than re-wrapping the constant. New code
/// must not use `ComponentSchemaId(SCHEMA_*)` directly.
#[inline]
#[must_use]
pub const fn schema_id(raw: u64) -> ComponentSchemaId {
    ComponentSchemaId(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_constants_match_namespace_allocation() {
        // Schema IDs are part of the durable wire contract. Bumping
        // any of these without a migration is a breaking change.
        assert_eq!(SCHEMA_COMPILATION_JOB, 31);
        assert_eq!(SCHEMA_JOB_INPUT, 32);
        assert_eq!(SCHEMA_JOB_CONFIG, 33);
        assert_eq!(SCHEMA_JOB_OUTPUT, 34);
        assert_eq!(SCHEMA_JOB_LIFECYCLE, 35);
        assert_eq!(SCHEMA_VALIDATION_RECEIPT, 36);
        assert_eq!(SCHEMA_QUANTIZATION_PLAN, 37);
        assert_eq!(SCHEMA_CIMAGE_PROMOTION, 38);
        assert_eq!(SCHEMA_QUANTIZATION_RESULT, 39);
    }

    #[test]
    fn schema_ids_are_unique() {
        let ids = [
            SCHEMA_COMPILATION_JOB,
            SCHEMA_JOB_INPUT,
            SCHEMA_JOB_CONFIG,
            SCHEMA_JOB_OUTPUT,
            SCHEMA_JOB_LIFECYCLE,
            SCHEMA_VALIDATION_RECEIPT,
            SCHEMA_QUANTIZATION_PLAN,
            SCHEMA_CIMAGE_PROMOTION,
            SCHEMA_QUANTIZATION_RESULT,
        ];
        let mut sorted: Vec<u64> = ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate schema IDs");
    }

    #[test]
    fn schema_id_helper_wraps_constant() {
        let s = schema_id(SCHEMA_COMPILATION_JOB);
        assert_eq!(s, ComponentSchemaId(31));
    }
}
