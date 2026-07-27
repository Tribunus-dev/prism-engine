//! Canonical command envelope, typed command set, and the constitutional
//! submit/replay path through the world.
//!
//! Authority: this directory owns the canonical authority for routing a
//! `CommandEnvelope` into the world — admission, epoch fencing, lease
//! coordination, the world-locked transaction, replay application, and
//! the journal/store completion handshake. The data shapes (`Command`,
//! `CommandResult`, `CommitOutcome`, `CommandEnvelope`) are the typed
//! vocabulary of kernel ingress; the `submit` function is the only
//! canonical writer of world state from the kernel.
//!
//! ## Classification
//!
//! The data shapes and the world-locked transaction are **canonical**.
//! The submit path itself touches process-local state (world `RwLock`,
//! lease coordinator, command store, sequence `AtomicU64`,
//! `mpsc::Receiver` via the state stream) and therefore crosses
//! execution-boundary criterion 3. The boundary is documented here; the
//! engine implements the *effect-side* dispatch through the existing
//! `WorkDispatcher` / `HardwareDispatcher` port traits in
//! [`crate::ports`]. Future work may extract a focused
//! `CommandDispatcher` trait; for now the canonical submit path lives
//! in [`submit`] as a free function over a borrowed
//! [`envelope::CommandDispatchContext`].
//!
//! ## Sub-module map
//!
//! - [`envelope`] — the typed command surface (`Command`,
//!   `CommandResult`, `CommitOutcome`, `CommandEnvelope`,
//!   `CommandDispatchContext`) and the typed lifecycle command
//!   implementations (`execute_lifecycle` and the concrete
//!   per-lifecycle-command bodies).
//! - [`submit`] — the canonical submit path: admission → lease
//!   coordination → world-locked transaction → journal completion, and
//!   the typed infrastructure command implementations.
//! - [`replay`] — the canonical replay path: `apply_recovered_command`
//!   re-applies a committed command to the world during journal
//!   recovery, with entity-id verification.
//!
//! ## Engine counterpart
//!
//! `compute-core/src/ecs/core/executor.rs` and `executor_projection.rs`
//! are execution-boundary math code (MLX arrays, hardware calls) and
//! are not absorbed here. The `SinkState` they once carried is already
//! absorbed into [`crate::attention_sink`]. `kernel_catalog.rs` is
//! already ported in `e633567e`.

pub mod envelope;
pub mod replay;
pub mod submit;

// Re-exports for the kernel's public surface (matches the original
// `pub use` block in `kernel/mod.rs`).
pub use envelope::{Command, CommandDispatchContext, CommandEnvelope, CommandResult, CommitOutcome};
pub use replay::apply_recovered_command;
pub use submit::{capture_world_snapshot, submit};
