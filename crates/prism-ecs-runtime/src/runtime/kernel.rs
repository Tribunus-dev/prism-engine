//! Re-export of the constitutional `RuntimeKernel` surface.
//!
//! The constitutional [`crate::kernel`] module is the canonical home for
//! the `RuntimeKernel` handle, command envelope, commit outcome, and
//! agent snapshot types. This module re-exports it under the
//! `runtime::kernel` path so the canonical "runtime" surface is
//! grep-able from a single import path.
//!
//! Migration map: engine callers of the legacy
//! `ecs::runtime::kernel::*` use the constitutional `runtime::kernel::*`
//! types directly.

pub use crate::kernel::*;
