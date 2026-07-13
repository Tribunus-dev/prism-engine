//! Engram runtime — lookup, insertion application, and receipts.
//!
//! An engram is a trained pattern that can be inserted into a tensor
//! computation at a specific region. This module provides the runtime
//! machinery for looking up engrams by query, applying them on CPU or
//! Metal, and producing lookup receipts.

pub mod application;
pub mod lookup;
pub mod receipt;
