//! Scheduling evidence (E bucket) — admitted, durable, replay-respects.
//!
//! Evidence in this module is durable, projection-rebuildable, and
//! respects replay. A receipt is emitted only for a committed
//! transaction; the receipt is a function of the committed state,
//! not the in-flight one.
//!
//! # Migration status
//!
//! Per the inventory v2.1 (steps 51-52), the receipts files
//! (`receipt.rs` and `receipts.rs`) move here from the engine. The
//! `engine_receipts.rs` shape (already in `prism-ecs-runtime`) is
//! the parent type; scheduling receipts are specializations.
