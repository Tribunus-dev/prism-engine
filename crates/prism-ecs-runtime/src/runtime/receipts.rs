//! Re-export of the constitutional receipt types.
//!
//! This module is the migration target for the engine's legacy
//! `compute-core/src/ecs/runtime/ledger/` subsystem. The engine-coupled
//! `TransitionLedger`, `TransitionReceipt`, `ReceiptDigest`, and related
//! types live in the engine's `legacy_runtime::ledger` and depend on
//! engine-internal `World` / `Entity` types. The constitutional receipt
//! types (model load, request admission, phase / step timing, terminal
//! request outcomes, worker exit) live in [`crate::engine_receipts`] and
//! are re-exported here as the canonical "runtime receipts" surface.
//!
//! # Migration map
//!
//! | Engine (legacy)                                | Constitutional (canonical)        |
//! |------------------------------------------------|-----------------------------------|
//! | `ecs::runtime::ledger::TransitionLedger`       | (engine-coupled; see `legacy_runtime::ledger`) |
//! | `ecs::runtime::ledger::TransitionReceipt`      | (engine-coupled; see `legacy_runtime::ledger`) |
//! | `ecs::runtime::ledger::ReceiptDigest`          | (engine-coupled; see `legacy_runtime::ledger`) |
//! | `ecs::runtime::ledger::ComponentTypeRegistry`  | (engine-coupled; see `legacy_runtime::ledger`) |
//! | `ecs::runtime::ledger::TransitionLedgerResource` | (engine-coupled; see `legacy_runtime::ledger`) |
//! | `ecs::runtime::engine_receipts::*` (engine re-export) | `runtime::receipts::*` (this module) |
//! | `ecs::runtime::engine_receipts` (engine re-implementation) | `prism_ecs_runtime::engine_receipts` |

// Re-export the constitutional receipt surface from `crate::engine_receipts`.
// Engine-coupled transition ledger types stay in the engine's
// `legacy_runtime::ledger` because they depend on engine-internal
// `World` / `Entity` / `Stage` types that the constitutional crate
// does not import.
pub use crate::engine_receipts::{
    AdmissionDecision, CancellationMode, DiffusionStepReceipt, ExecutionPhase, ModelLoadReceipt,
    ModelLoadReceiptBuilder, PhaseReceipt, PhaseReceiptBuilder, ReceiptBuilder,
    RequestAdmissionReceipt, RequestAdmissionReceiptBuilder, RequestOutcome, StepReceipt,
    StepReceiptBuilder, TerminalRequestReceipt, TerminalRequestReceiptBuilder, Timeline,
    TimelineEvent, WorkerExitReceipt, WorkerExitReceiptBuilder,
};
pub use prism_ecs_constitutional::ReceiptId;
