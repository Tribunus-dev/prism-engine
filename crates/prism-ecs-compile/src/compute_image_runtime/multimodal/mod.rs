//! Multimodal bindings — pure data types and pure algorithms for
//! multimodal (text / image / audio) projection and binding.

pub mod adapter;
pub mod binding;
pub mod descriptor;

pub use adapter::{
    EmbeddedModality, Gemma4DirectAudioProjectionAdapter, Gemma4DirectImageProjectionAdapter,
    LegacyVisionEncoderProjectorAdapter, ModalityInput, PreparedModality, TokenEmbeddingAdapter,
};
pub use binding::SealedSegmentBinding;
pub use descriptor::{
    AudioProcessorContractV1, ImageProcessorContractV1, InputModality, ModalityError,
    MultimodalArtifactSummary, MultimodalAssemblyReceipt, MultimodalCapabilities,
    MultimodalInputDescriptorV1, ProjectionBackend, ProjectionPrecision, ProjectionRole,
    ProjectionTensorRecord,
};
