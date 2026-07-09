//! State Store — schema, KV paged cache, epochs, access control, validation, receipts.
//!
//! Provides in-memory paged KV cache management with epoch-gated access control,
//! deterministic memory accounting, and validation gate checks.

mod access;
mod epochs;
mod kv;
mod pages;
mod receipts;
mod schema;
mod validate;

pub use access::*;
pub use epochs::*;
pub use kv::*;
pub use pages::*;
pub use receipts::*;
pub use schema::*;
pub use validate::*;

#[cfg(test)]
mod tests;
