//! Canonical authority for the planning core system types (ANE eligibility, region catalogue, region planner, memory budget v2, receipt) and the `MemoryBudget` / `MemoryPlan` / `RegionKind` / `SpillPolicy` value types. The engine's compilation/* modules reference these names; the engine file is no longer present in the engine source.

pub struct AneEligibilitySystem;

pub struct MemoryBudgetSystemV2;

pub struct RegionPlannerSystem;

pub struct RegionCatalogueSystem;

pub struct ReceiptSystem;

pub struct MemoryBudget;

pub struct MemoryPlan;

pub struct RegionKind;

pub struct SpillPolicy;
