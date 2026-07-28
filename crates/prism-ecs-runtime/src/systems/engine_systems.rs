//! Canonical authority for the engine lifecycle system types (init, install, load, unload, cancel, metrics, shutdown, generate) and the `CimageLoadRequest` value type. These were previously declared in the engine's `ecs::system::engine_systems` module (no longer present in the engine source).

pub struct EngineInitSystem;

pub struct ModelInstallSystem;

pub struct ModelLoadSystem;

pub struct CimageLoadSystem;

pub struct CimageLoadRequest;

pub struct HostInferenceInitSystem;

pub struct ModelUnloadSystem;

pub struct EngineMetricsSystem;

pub struct EngineShutdownSystem;

pub struct GenerationRequestSystem;

pub struct CimageGenerateSystem;

pub struct InferenceCycleSystem;

pub struct TokenBudgetInferenceSystem;

pub struct CancelSystem;

pub struct MemoryPressureSystem;
