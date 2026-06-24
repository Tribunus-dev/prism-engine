//! Variants — compiled shape-specialized program definitions, selection,
//! compatibility, and coverage.
//!
//! A single model compile pipeline emits multiple program variants, each
//! specialized for a particular [`ExecutionShapeClass`] and target profile.
//! This module provides the variant definition schema
//! ([`shape_class`]), compatibility checking ([`compatibility`]),
//! selection logic ([`selection`]), and coverage analysis ([`coverage`]).

pub mod compatibility;
pub mod coverage;
pub mod selection;
pub mod shape_class;

pub use compatibility::*;
pub use coverage::*;
pub use selection::*;
pub use shape_class::*;
