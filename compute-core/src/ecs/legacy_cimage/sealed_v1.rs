//! SealedCimageV1 — versioned sealed cimage artifact with canonical manifest,
//! independently digestible sections, deterministic serialization, and strict
//! validation.
//!
//! # Format
//!
//! The sealed format is a single binary file with the following layout:
//!
//! ```text
//! [SealedCimageHeader]           — 128 bytes fixed
//! [Section Index]                — canonical JSON, Vec<SectionEntry>
//! --- 64-byte aligned sections ---
//! [Canonical Manifest section]
//! [Generation section]           — bincode serialized
//! [Tensor segments]              — one section per segment
//! [Kernel artifacts]             — one per artifact
//! [Kernel ABIs]                  — one per ABI
//! [Tokenizer section]            — optional
//! [Receipt sections]             — one per receipt
//! [Replay manifest section]      — optional
//! [Hardware contract section]    — optional
//! ```
//!
//! Every section is independently digestible via its SHA-256 in the section
//! index. The root digest covers the canonical manifest bytes. A streaming
//! loader can verify only the sections it needs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use prism_ecs_constitutional::canonical::generation::CimageGeneration;
use prism_ecs_constitutional::canonical::kernel_abi::{CompiledKernelArtifact, KernelAbi};
use prism_ecs_compile::pipeline::deployment_compiler::ServingProfile;
use crate::ecs::plan::CodecFamily;
// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes for SealedCimageV1: "PRISM SE1" (8 bytes).
pub const SEALED_CIMAGE_MAGIC: [u8; 8] = *b"PRISMSE1";

/// Format version for this sealed cimage layout.
pub const SEALED_CIMAGE_VERSION: u32 = 1;

/// Fixed header size in bytes.
pub const SEALED_CIMAGE_HEADER_SIZE: usize = 128;

/// 64-byte alignment boundary.
const ALIGNMENT: u64 = 64;

/// Round `offset` up to the next 64-byte boundary.
fn align_64(offset: u64) -> u64 {
    (offset + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

/// Compute the SHA-256 digest of a byte slice.
fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

// ---------------------------------------------------------------------------
// Section types
// ---------------------------------------------------------------------------

/// Kinds of sections within a sealed cimage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SectionKind {
    /// Canonical manifest JSON.
    Manifest,
    /// CimageGeneration serialized via bincode.
    Generation,
    /// Packed tensor bytes.
    TensorSegment,
    /// Compiled kernel bytes.
    KernelArtifact,
    /// KernelAbi serialized via bincode.
    KernelAbi,
    /// Tokenizer bytes.
    Tokenizer,
    /// Receipt bytes.
    Receipt,
    /// General payload bytes.
    Payload,
    /// Forward-compatible optional section (preserved but not validated).
    Optional,
}

/// A section within the sealed cimage — always 64-byte aligned in the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionEntry {
    /// Canonical section identifier (e.g. "generation", "manifest", "tensor:<digest>").
    pub id: String,
    /// The kind of content in this section.
    pub kind: SectionKind,
    /// Absolute byte offset in the file (always 64-byte aligned).
    pub offset: u64,
    /// Exact byte length of the section content (before padding).
    pub byte_len: u64,
    /// SHA-256 digest of the section content bytes (before padding).
    pub digest: [u8; 32],
}

// ---------------------------------------------------------------------------
// Header — fixed 128 bytes
// ---------------------------------------------------------------------------

/// On-disk header for SealedCimageV1 — exactly 128 bytes.
///
/// Serialized via bincode for deterministic fixed-size layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedCimageHeader {
    /// Magic bytes: "PRISMSE1"
    pub magic: [u8; 8],
    /// Format version (currently 1).
    pub version: u32,
    /// Number of section entries in the section index.
    pub section_count: u32,
    /// Absolute byte offset to the section index (canonical JSON).
    pub section_index_offset: u64,
    /// Byte length of the section index.
    pub section_index_len: u64,
    /// Absolute byte offset to the canonical manifest section.
    pub manifest_offset: u64,
    /// Byte length of the canonical manifest section.
    pub manifest_len: u64,
    /// SHA-256 digest of the canonical manifest bytes.
    pub root_digest: [u8; 32],
    /// Reserved for future use — zero-filled.
    pub reserved: [u8; 32],
}

impl SealedCimageHeader {
    /// Create a new header with default magic and version.
    fn new() -> Self {
        Self {
            magic: SEALED_CIMAGE_MAGIC,
            version: SEALED_CIMAGE_VERSION,
            section_count: 0,
            section_index_offset: 0,
            section_index_len: 0,
            manifest_offset: 0,
            manifest_len: 0,
            root_digest: [0u8; 32],
            reserved: [0u8; 32],
        }
    }

    /// Serialize the header to bytes via bincode.
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes =
            bincode::serialize(self).expect("SealedCimageHeader serialization must not fail");
        // Pad to exactly SEALED_CIMAGE_HEADER_SIZE bytes.
        if bytes.len() < SEALED_CIMAGE_HEADER_SIZE {
            bytes.resize(SEALED_CIMAGE_HEADER_SIZE, 0u8);
        }
        bytes
    }

    /// Deserialize the header from bytes.
    fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < SEALED_CIMAGE_HEADER_SIZE {
            return Err(format!(
                "header too short: got {} bytes, need at least {}",
                data.len(),
                SEALED_CIMAGE_HEADER_SIZE
            ));
        }
        // Read up to SEALED_CIMAGE_HEADER_SIZE bytes; bincode reads what it
        // needs and ignores trailing padding zeros.
        bincode::deserialize(&data[..SEALED_CIMAGE_HEADER_SIZE])
            .map_err(|e| format!("failed to deserialize header: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Canonical manifest
// ---------------------------------------------------------------------------

/// Identity of a tensor segment referenced by the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentIdentity {
    /// Physical segment identifier.
    pub id: String,
    /// SHA-256 digest of the segment bytes.
    pub digest: [u8; 32],
    /// Exact byte length of the segment.
    pub byte_len: u64,
}

/// Identity of a compiled kernel artifact referenced by the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelArtifactIdentity {
    /// Implementation identifier.
    pub implementation_id: String,
    /// Semantic identifier.
    pub semantic_id: String,
    /// SHA-256 digest of the artifact bytes.
    pub digest: [u8; 32],
    /// Exact byte length of the artifact.
    pub byte_len: u64,
}

/// Identity of a kernel ABI referenced by the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbiIdentity {
    /// Implementation identifier this ABI belongs to.
    pub implementation_id: String,
    /// SHA-256 digest of the ABI serialized bytes.
    pub digest: [u8; 32],
}

/// Identity of a tokenizer referenced by the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerIdentity {
    /// Tokenizer identifier.
    pub identifier: String,
    /// SHA-256 digest of the tokenizer bytes.
    pub digest: [u8; 32],
    /// Exact byte length of the tokenizer data.
    pub byte_len: u64,
}

/// Canonical manifest containing every component identity referenced by the
/// sealed cimage. The root digest covers these bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalManifest {
    /// Format version this manifest was produced with.
    pub format_version: u32,
    /// Generation identity.
    pub generation_id: String,
    /// Parent generation identity, if any.
    pub parent_generation: Option<String>,
    /// Base model identity.
    pub base_model: String,
    /// Compiler identity (name + version).
    pub compiler: String,
    /// Target hardware profile.
    pub hardware_profile: String,
    /// ISO 8601 timestamp.
    pub created_at: String,

    /// All tensor segment identities referenced by the generation.
    pub tensor_segments: Vec<SegmentIdentity>,
    /// All compiled kernel artifacts referenced.
    pub kernel_artifacts: Vec<KernelArtifactIdentity>,
    /// All kernel ABIs referenced.
    pub kernel_abis: Vec<AbiIdentity>,
    /// Receipt identities in the bundle.
    pub receipt_entries: Vec<ReceiptManifestEntry>,
    /// Tokenizer identity, if present.
    pub tokenizer: Option<TokenizerIdentity>,
    /// Replay manifest identity (digest of the ReplayManifest section), if present.
    pub replay_manifest: Option<String>,
    /// Hardware contract identity (digest of hardware capability section), if present.
    pub hardware_contract: Option<String>,

    /// Required capabilities that the loader must support.
    pub required_capabilities: Vec<String>,
    /// Optional capabilities present.
    pub optional_capabilities: Vec<String>,
}

/// A receipt entry in the manifest with its digest and length for integrity
/// verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptManifestEntry {
    pub id: String,
    pub digest: [u8; 32],
    pub byte_len: u64,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing a SealedCimageV1 from canonical components.
///
/// Collects all components, validates referential integrity, and produces a
/// sealed artifact with deterministic serialization.
#[derive(Debug, Clone)]
pub struct SealedCimageBuilder {
    generation: Option<CimageGeneration>,
    tensor_segments: BTreeMap<String, Vec<u8>>,
    kernel_artifacts: BTreeMap<String, CompiledKernelArtifact>,
    kernel_abis: BTreeMap<String, KernelAbi>,
    receipts: BTreeMap<String, Vec<u8>>,
    tokenizer: Option<(String, Vec<u8>)>,
    replay_manifest: Option<Vec<u8>>,
    hardware_contract: Option<Vec<u8>>,
    serving_profile: Option<ServingProfile>,
    capabilities: Vec<String>,
}

impl SealedCimageBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            generation: None,
            tensor_segments: BTreeMap::new(),
            kernel_artifacts: BTreeMap::new(),
            kernel_abis: BTreeMap::new(),
            receipts: BTreeMap::new(),
            tokenizer: None,
            replay_manifest: None,
            hardware_contract: None,
            serving_profile: None,
            capabilities: Vec::new(),
        }
    }

    /// Set the generation.
    ///
    /// Also infers required capabilities from the generation's codec families
    /// found in its tensor bindings.
    pub fn with_generation(mut self, generation: CimageGeneration) -> Self {
        // Infer capabilities from codec families in the generation's bindings.
        for (_logical_id, binding) in &generation.tensor_bindings {
            let cap = codec_family_to_capability(&binding.codec);
            if !self.capabilities.contains(&cap) {
                self.capabilities.push(cap);
            }
        }
        self.generation = Some(generation);
        self
    }

    /// Add a tensor segment with its physical segment ID.
    ///
    /// Returns an error if a segment with the same ID already exists.
    pub fn add_tensor_segment(mut self, id: String, data: Vec<u8>) -> Result<Self, String> {
        if self.tensor_segments.contains_key(&id) {
            return Err(format!("duplicate tensor segment id: {id}"));
        }
        self.tensor_segments.insert(id, data);
        Ok(self)
    }

    /// Add a compiled kernel artifact keyed by its implementation ID.
    ///
    /// Returns an error if an artifact with the same implementation ID already exists.
    pub fn add_kernel_artifact(
        mut self,
        id: String,
        artifact: CompiledKernelArtifact,
    ) -> Result<Self, String> {
        if self.kernel_artifacts.contains_key(&id) {
            return Err(format!("duplicate kernel artifact id: {id}"));
        }
        self.kernel_artifacts.insert(id, artifact);
        Ok(self)
    }

    /// Add a kernel ABI keyed by its implementation ID.
    ///
    /// Returns an error if an ABI with the same implementation ID already exists.
    pub fn add_kernel_abi(mut self, impl_id: String, abi: KernelAbi) -> Result<Self, String> {
        if self.kernel_abis.contains_key(&impl_id) {
            return Err(format!("duplicate kernel ABI id: {impl_id}"));
        }
        self.kernel_abis.insert(impl_id, abi);
        Ok(self)
    }

    /// Add a receipt with its receipt ID.
    ///
    /// Returns an error if a receipt with the same ID already exists.
    pub fn add_receipt(mut self, id: String, data: Vec<u8>) -> Result<Self, String> {
        if self.receipts.contains_key(&id) {
            return Err(format!("duplicate receipt id: {id}"));
        }
        self.receipts.insert(id, data);
        Ok(self)
    }

    /// Set the tokenizer.
    pub fn with_tokenizer(mut self, identifier: String, data: Vec<u8>) -> Self {
        self.tokenizer = Some((identifier, data));
        self
    }

    /// Set the replay manifest bytes.
    pub fn with_replay_manifest(mut self, data: Vec<u8>) -> Self {
        self.replay_manifest = Some(data);
        self
    }

    /// Set the hardware contract bytes.
    pub fn with_hardware_contract(mut self, data: Vec<u8>) -> Self {
        self.hardware_contract = Some(data);
        self
    }

    /// Set the serving profile.
    pub fn with_serving_profile(mut self, profile: ServingProfile) -> Self {
        self.serving_profile = Some(profile);
        self
    }

    /// Add an extra capability string.
    pub fn with_capability(mut self, cap: String) -> Self {
        if !self.capabilities.contains(&cap) {
            self.capabilities.push(cap);
        }
        self
    }

    /// Validate all constraints and build the SealedCimageV1.
    ///
    /// Validation checks:
    /// 1. Required fields are set (generation, serving_profile).
    /// 2. No duplicate tensor segment IDs.
    /// 3. Every tensor binding in the generation references a segment that exists.
    /// 4. Every kernel binding references an artifact that exists.
    /// 5. Returns error for missing required sections.
    pub fn build(self) -> Result<SealedCimageV1, String> {
        let generation = self
            .generation
            .ok_or_else(|| "generation is required".to_string())?;
        let serving_profile = self
            .serving_profile
            .ok_or_else(|| "serving_profile is required".to_string())?;

        // Check that every tensor binding in the generation has a corresponding segment.
        for (_logical_id, binding) in &generation.tensor_bindings {
            let primary_id = &binding.primary_segment.0;
            if !self.tensor_segments.contains_key(primary_id) {
                return Err(format!(
                    "tensor binding references missing primary segment: {primary_id}"
                ));
            }
            for scale in &binding.scale_segments {
                if !self.tensor_segments.contains_key(&scale.0) {
                    return Err(format!(
                        "tensor binding references missing scale segment: {}",
                        scale.0
                    ));
                }
            }
            for residual in &binding.residual_segments {
                if !self.tensor_segments.contains_key(&residual.0) {
                    return Err(format!(
                        "tensor binding references missing residual segment: {}",
                        residual.0
                    ));
                }
            }
        }

        // Check that every kernel binding references an artifact.
        for (_semantic_id, impl_id) in &generation.kernel_bindings {
            if !self.kernel_artifacts.contains_key(&impl_id.0) {
                return Err(format!(
                    "kernel binding references missing artifact: {}",
                    impl_id.0
                ));
            }
        }

        let capabilities = self.capabilities;

        // Determine optional capabilities — anything we have that isn't a
        // required one inferred from the codec families. Since we seeded
        // `capabilities` from codecs, optional set starts empty; users can
        // add more with `with_capability`.
        let required_capabilities = capabilities.clone();
        let optional_capabilities: Vec<String> = Vec::new();

        let now = prism_ecs_constitutional::canonical::compile_plan::compile_timestamp();

        Ok(SealedCimageV1 {
            generation,
            tensor_segments: self.tensor_segments,
            kernel_artifacts: self.kernel_artifacts,
            kernel_abis: self.kernel_abis,
            receipts: self.receipts,
            tokenizer: self.tokenizer,
            replay_manifest: self.replay_manifest,
            hardware_contract: self.hardware_contract,
            serving_profile,
            stored_serving_profile: false,
            required_capabilities,
            optional_capabilities,
            created_at: now,
            root_digest: [0u8; 32],
            preserved_optional_sections: Vec::new(),
        })
    }
}

impl Default for SealedCimageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a CodecFamily to a capability string.
fn codec_family_to_capability(codec: &CodecFamily) -> String {
    match codec {
        CodecFamily::Nf4 => "nf4".to_string(),
        CodecFamily::Int8 => "int8".to_string(),
        CodecFamily::Fp16 => "fp16".to_string(),
        CodecFamily::RawF32 => "raw_f32".to_string(),
        CodecFamily::SymInt4 => "sym_int4".to_string(),
        CodecFamily::Ternary => "ternary".to_string(),
        CodecFamily::Ternary1_58 => "ternary_1_58".to_string(),
        CodecFamily::Mixed => "mixed".to_string(),
        CodecFamily::Q8_0 | CodecFamily::Q4_K => "q8_0".to_string(),
        CodecFamily::Q2_K => "q2_k".to_string(),
        CodecFamily::IQ2_XXS => "iq2_xxs".to_string(),
    }
}

// ---------------------------------------------------------------------------
// SealedCimageV1
// ---------------------------------------------------------------------------

/// A fully sealed, versioned cimage artifact.
///
/// Contains everything needed to load and execute a compiled model:
/// the generation specification, all tensor segments, kernel artifacts,
/// ABIs, receipts, tokenizer data, replay manifest, and serving profile.
///
/// Every component is independently digestible by its SHA-256 in the section
/// index. The root digest covers the canonical manifest, which lists every
/// component identity.
#[derive(Debug, Clone)]
pub struct SealedCimageV1 {
    /// The underlying generation with bindings.
    generation: CimageGeneration,
    /// Tensor segment bytes keyed by physical segment ID.
    tensor_segments: BTreeMap<String, Vec<u8>>,
    /// Compiled kernel artifacts keyed by implementation ID.
    kernel_artifacts: BTreeMap<String, CompiledKernelArtifact>,
    /// Kernel ABIs keyed by implementation ID.
    kernel_abis: BTreeMap<String, KernelAbi>,
    /// Receipt bytes keyed by receipt ID.
    receipts: BTreeMap<String, Vec<u8>>,
    /// Optional tokenizer (identifier, bytes).
    tokenizer: Option<(String, Vec<u8>)>,
    /// Optional replay manifest bytes.
    replay_manifest: Option<Vec<u8>>,
    /// Optional hardware contract bytes.
    hardware_contract: Option<Vec<u8>>,
    /// Serving profile with model metadata.
    serving_profile: ServingProfile,
    /// Whether the serving_profile was deserialized from a dedicated section
    /// (true) or reconstructed from manifest fields (false, legacy path).
    #[allow(dead_code)]
    stored_serving_profile: bool,
    /// Required capabilities that loaders must support.
    required_capabilities: Vec<String>,
    /// Optional capabilities present.
    optional_capabilities: Vec<String>,
    /// ISO 8601 creation timestamp.
    created_at: String,
    /// Root digest from the section index (set during deserialization,
    /// or zero for builder-constructed instances).
    root_digest: [u8; 32],
    /// Forward-compatible optional sections preserved through round-trip.
    preserved_optional_sections: Vec<(SectionEntry, Vec<u8>)>,
}

impl SealedCimageV1 {
    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get a reference to the generation.
    pub fn generation(&self) -> &CimageGeneration {
        &self.generation
    }

    /// Get a reference to the serving profile.
    pub fn serving_profile(&self) -> &ServingProfile {
        &self.serving_profile
    }

    /// Find a tensor segment by its physical segment ID.
    pub fn find_tensor_segment(&self, id: &str) -> Option<&[u8]> {
        self.tensor_segments.get(id).map(|v| v.as_slice())
    }

    /// Find a kernel artifact by its implementation ID.
    pub fn find_kernel_artifact(&self, id: &str) -> Option<&CompiledKernelArtifact> {
        self.kernel_artifacts.get(id)
    }

    /// Get all tensor segment IDs.
    pub fn tensor_segment_ids(&self) -> impl Iterator<Item = &str> {
        self.tensor_segments.keys().map(|s| s.as_str())
    }

    /// Get all kernel artifact implementation IDs.
    pub fn kernel_artifact_ids(&self) -> impl Iterator<Item = &str> {
        self.kernel_artifacts.keys().map(|s| s.as_str())
    }

    /// Get all receipt IDs.
    pub fn receipt_ids(&self) -> impl Iterator<Item = &str> {
        self.receipts.keys().map(|s| s.as_str())
    }

    /// Get the tokenizer identity, if present.
    pub fn tokenizer(&self) -> Option<(&str, &[u8])> {
        self.tokenizer
            .as_ref()
            .map(|(id, data)| (id.as_str(), data.as_slice()))
    }

    /// Get the replay manifest bytes, if present.
    pub fn replay_manifest(&self) -> Option<&[u8]> {
        self.replay_manifest.as_deref()
    }

    /// Get the hardware contract bytes, if present.
    pub fn hardware_contract(&self) -> Option<&[u8]> {
        self.hardware_contract.as_deref()
    }

    /// Get the required capabilities.
    pub fn required_capabilities(&self) -> &[String] {
        &self.required_capabilities
    }

    /// Get the optional capabilities.
    pub fn optional_capabilities(&self) -> &[String] {
        &self.optional_capabilities
    }

    /// Get the creation timestamp.
    pub fn created_at(&self) -> &str {
        &self.created_at
    }

    // -----------------------------------------------------------------------
    // Manifest building
    // -----------------------------------------------------------------------

    /// Build the canonical manifest for this sealed cimage.
    fn build_manifest(&self) -> Result<CanonicalManifest, String> {
        let generation_id = self.generation.generation_id.0.clone();
        let parent_generation = self
            .generation
            .parent_generation
            .as_ref()
            .map(|g| g.0.clone());
        let base_model = self.generation.base_model.0.clone();
        let compiler = format!(
            "{} {}",
            self.generation.compiler_identity.name, self.generation.compiler_identity.version,
        );
        let hardware_profile = self.generation.hardware_profile.0.clone();
        let created_at = self.created_at.clone();

        // Build segment identities.
        let tensor_segments: Vec<SegmentIdentity> = self
            .tensor_segments
            .iter()
            .map(|(id, data)| SegmentIdentity {
                id: id.clone(),
                digest: sha256_digest(data),
                byte_len: data.len() as u64,
            })
            .collect();

        // Build kernel artifact identities.
        let kernel_artifacts: Vec<KernelArtifactIdentity> = self
            .kernel_artifacts
            .iter()
            .map(|(_id, artifact)| {
                let digest = sha256_digest(&artifact.compiled_bytes);
                KernelArtifactIdentity {
                    implementation_id: artifact.implementation_id.0.clone(),
                    semantic_id: artifact.semantic_id.0.clone(),
                    digest,
                    byte_len: artifact.compiled_bytes.len() as u64,
                }
            })
            .collect();

        // Build kernel ABI identities.
        let kernel_abis: Vec<AbiIdentity> = self
            .kernel_abis
            .iter()
            .map(|(impl_id, abi)| {
                let abi_bytes = bincode::serialize(abi)
                    .unwrap_or_else(|e| panic!("failed to serialize KernelAbi for {impl_id}: {e}"));
                AbiIdentity {
                    implementation_id: impl_id.clone(),
                    digest: sha256_digest(&abi_bytes),
                }
            })
            .collect();

        // Receipt identities with digests and lengths.
        let receipt_entries: Vec<ReceiptManifestEntry> = self
            .receipts
            .iter()
            .map(|(id, data)| ReceiptManifestEntry {
                id: id.clone(),
                digest: sha256_digest(data),
                byte_len: data.len() as u64,
            })
            .collect();

        // Tokenizer identity.
        let tokenizer = self.tokenizer.as_ref().map(|(id, data)| TokenizerIdentity {
            identifier: id.clone(),
            digest: sha256_digest(data),
            byte_len: data.len() as u64,
        });

        // Replay manifest digest, if present.
        let replay_manifest = self.replay_manifest.as_ref().map(|data| {
            sha256_digest(data)
                .iter()
                .map(|b: &u8| format!("{b:02x}"))
                .collect::<String>()
        });

        // Hardware contract digest, if present.
        let hardware_contract = self.hardware_contract.as_ref().map(|data| {
            sha256_digest(data)
                .iter()
                .map(|b: &u8| format!("{b:02x}"))
                .collect::<String>()
        });

        Ok(CanonicalManifest {
            format_version: 1,
            generation_id,
            parent_generation,
            base_model,
            compiler,
            hardware_profile,
            created_at,
            tensor_segments,
            kernel_artifacts,
            kernel_abis,
            receipt_entries,
            tokenizer,
            replay_manifest,
            hardware_contract,
            required_capabilities: self.required_capabilities.clone(),
            optional_capabilities: self.optional_capabilities.clone(),
        })
    }

    // -----------------------------------------------------------------------
    // Section index building and root digest
    // -----------------------------------------------------------------------

    /// Build the section index for this sealed cimage.
    ///
    /// Produces a sorted `Vec<SectionEntry>` (sorted by entry id) with correct
    /// byte_len and digest for every section. Offsets are set to 0 since
    /// they depend on the serialization layout.
    pub fn build_section_index(&self) -> Vec<SectionEntry> {
        let mut sections: Vec<(String, SectionKind, Vec<u8>)> = Vec::new();

        // Manifest.
        let manifest = self
            .build_manifest()
            .unwrap_or_else(|e| panic!("failed to build manifest: {e}"));
        let manifest_bytes: Vec<u8> = serde_json_canonicalizer::to_string(&manifest)
            .unwrap_or_else(|e| panic!("failed to serialize manifest: {e}"))
            .into_bytes();
        sections.push((
            "manifest".to_string(),
            SectionKind::Manifest,
            manifest_bytes,
        ));

        // Generation (bincode).
        let gen_bytes = bincode::serialize(&self.generation)
            .unwrap_or_else(|e| panic!("failed to serialize generation: {e}"));
        sections.push(("generation".to_string(), SectionKind::Generation, gen_bytes));

        // Serving profile (bincode).
        let profile_bytes = bincode::serialize(&self.serving_profile)
            .unwrap_or_else(|e| panic!("failed to serialize serving profile: {e}"));
        sections.push((
            "serving_profile".to_string(),
            SectionKind::Payload,
            profile_bytes,
        ));

        // Tensor segments (sorted by ID from BTreeMap).
        for (id, data) in &self.tensor_segments {
            sections.push((
                format!("tensor:{id}"),
                SectionKind::TensorSegment,
                data.clone(),
            ));
        }

        // Kernel artifacts (sorted by ID).
        for (_id, artifact) in &self.kernel_artifacts {
            let bytes = bincode::serialize(artifact)
                .unwrap_or_else(|e| panic!("failed to serialize kernel artifact {_id}: {e}"));
            sections.push((
                format!("kernel:{}", artifact.implementation_id.0),
                SectionKind::KernelArtifact,
                bytes,
            ));
        }

        // Kernel ABIs (sorted by ID).
        for (impl_id, abi) in &self.kernel_abis {
            let bytes = bincode::serialize(abi)
                .unwrap_or_else(|e| panic!("failed to serialize kernel ABI {impl_id}: {e}"));
            sections.push((format!("abi:{impl_id}"), SectionKind::KernelAbi, bytes));
        }

        // Tokenizer (optional).
        if let Some((_id, data)) = &self.tokenizer {
            sections.push((
                "tokenizer".to_string(),
                SectionKind::Tokenizer,
                data.clone(),
            ));
        }

        // Receipts (sorted by ID).
        for (id, data) in &self.receipts {
            sections.push((format!("receipt:{id}"), SectionKind::Receipt, data.clone()));
        }

        // Replay manifest (optional).
        if let Some(data) = &self.replay_manifest {
            sections.push((
                "replay_manifest".to_string(),
                SectionKind::Payload,
                data.clone(),
            ));
        }

        // Hardware contract (optional).
        if let Some(data) = &self.hardware_contract {
            sections.push((
                "hardware_contract".to_string(),
                SectionKind::Payload,
                data.clone(),
            ));
        }

        // Preserved optional sections.
        for (entry, data) in &self.preserved_optional_sections {
            sections.push((entry.id.clone(), entry.kind.clone(), data.clone()));
        }

        // Build entries with digests.
        let mut entries: Vec<SectionEntry> = sections
            .iter()
            .map(|(id, kind, bytes)| SectionEntry {
                id: id.clone(),
                kind: kind.clone(),
                offset: 0,
                byte_len: bytes.len() as u64,
                digest: sha256_digest(bytes),
            })
            .collect();

        // Sort by entry id for canonical root digest computation.
        entries.sort_by(|a, b| a.id.cmp(&b.id));
        entries
    }

    /// Compute the root digest: SHA-256 of the canonical JSON of the sorted
    /// section index.
    pub fn root_digest(&self) -> [u8; 32] {
        let sections = self.build_section_index();
        let index_bytes = bincode::serialize(&sections)
            .unwrap_or_else(|e| panic!("section index serialization must not fail: {e}"));
        sha256_digest(&index_bytes)
    }

    // -----------------------------------------------------------------------
    // Integrity verification
    // -----------------------------------------------------------------------

    /// Verify the integrity of this sealed cimage.
    ///
    /// Checks:
    /// - Root digest matches the stored root_digest.
    /// - All sections referenced in the manifest exist.
    /// - No duplicate identities exist.
    ///
    /// Returns `Ok(())` or a list of errors.
    pub fn verify_integrity(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        // 1. Check root digest against build_section_index() recomputation.
        // Non-zero root_digest indicates this was deserialized from a sealed
        // artifact; builder-constructed instances have zero root_digest and
        // are verified by the builder's own validation.
        if self.root_digest != [0u8; 32] {
            let sections = self.build_section_index();
            let index_bytes = bincode::serialize(&sections).unwrap_or_else(|e| {
                errors.push(format!("failed to serialize section index: {e}"));
                Vec::new()
            });
            if !index_bytes.is_empty() {
                let expected_root = sha256_digest(&index_bytes);
                if expected_root != self.root_digest {
                    errors.push(format!(
                        "root digest mismatch: computed {:02x?}, expected {:02x?}",
                        expected_root, self.root_digest
                    ));
                }
            }
        }

        // 2. Check that every segment referenced in the generation's bindings exists.
        for (_logical_id, binding) in &self.generation.tensor_bindings {
            let primary_id = &binding.primary_segment.0;
            if !self.tensor_segments.contains_key(primary_id) {
                errors.push(format!(
                    "generation references missing primary segment: {primary_id}"
                ));
            }
            for scale in &binding.scale_segments {
                if !self.tensor_segments.contains_key(&scale.0) {
                    errors.push(format!(
                        "generation references missing scale segment: {}",
                        scale.0
                    ));
                }
            }
            for residual in &binding.residual_segments {
                if !self.tensor_segments.contains_key(&residual.0) {
                    errors.push(format!(
                        "generation references missing residual segment: {}",
                        residual.0
                    ));
                }
            }
        }

        // 3. Check that every kernel binding references an artifact.
        for (_semantic_id, impl_id) in &self.generation.kernel_bindings {
            if !self.kernel_artifacts.contains_key(&impl_id.0) {
                errors.push(format!(
                    "generation references missing kernel artifact: {}",
                    impl_id.0
                ));
            }
        }

        // 4. Check no duplicate tensor segment IDs (guaranteed by BTreeMap but verify for safety).
        if self.tensor_segments.len()
            != self
                .tensor_segments
                .keys()
                .collect::<std::collections::HashSet<_>>()
                .len()
        {
            errors.push("duplicate tensor segment IDs detected".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serialize the sealed cimage to deterministic bytes.
    ///
    /// Produces a complete binary artifact:
    /// Produces a complete binary artifact:
    /// 1. Fixed 128-byte header (bincode).
    /// 2. Section index (canonical JSON, insertion order).
    /// 3. 64-byte aligned sections.
    ///
    /// Root digest covers the sorted-by-id section index entries (offset=0).
    pub fn serialize(&self) -> Result<Vec<u8>, String> {
        // Step 1: Enumerate all sections in deterministic order.
        let mut sections: Vec<(String, SectionKind, Vec<u8>)> = Vec::new();

        // Manifest section.
        let manifest = self
            .build_manifest()
            .map_err(|e| format!("failed to build manifest: {e}"))?;
        let manifest_bytes: Vec<u8> = serde_json_canonicalizer::to_string(&manifest)
            .map_err(|e| format!("failed to serialize manifest: {e}"))?
            .into_bytes();
        sections.push((
            "manifest".to_string(),
            SectionKind::Manifest,
            manifest_bytes,
        ));

        // Generation (bincode).
        let gen_bytes = bincode::serialize(&self.generation)
            .map_err(|e| format!("failed to serialize generation: {e}"))?;
        sections.push(("generation".to_string(), SectionKind::Generation, gen_bytes));

        // Serving profile (bincode).
        let profile_bytes = bincode::serialize(&self.serving_profile)
            .map_err(|e| format!("failed to serialize serving profile: {e}"))?;
        sections.push((
            "serving_profile".to_string(),
            SectionKind::Payload,
            profile_bytes,
        ));

        // Tensor segments (sorted by ID from BTreeMap).
        for (id, data) in &self.tensor_segments {
            sections.push((
                format!("tensor:{id}"),
                SectionKind::TensorSegment,
                data.clone(),
            ));
        }

        // Kernel artifacts (sorted by ID).
        for (_id, artifact) in &self.kernel_artifacts {
            let bytes = bincode::serialize(artifact)
                .map_err(|e| format!("failed to serialize kernel artifact {_id}: {e}"))?;
            sections.push((
                format!("kernel:{}", artifact.implementation_id.0),
                SectionKind::KernelArtifact,
                bytes,
            ));
        }

        // Kernel ABIs (sorted by ID).
        for (impl_id, abi) in &self.kernel_abis {
            let bytes = bincode::serialize(abi)
                .map_err(|e| format!("failed to serialize kernel ABI {impl_id}: {e}"))?;
            sections.push((format!("abi:{impl_id}"), SectionKind::KernelAbi, bytes));
        }

        // Tokenizer (optional).
        if let Some((_id, data)) = &self.tokenizer {
            sections.push((
                "tokenizer".to_string(),
                SectionKind::Tokenizer,
                data.clone(),
            ));
        }

        // Receipts (sorted by ID).
        for (id, data) in &self.receipts {
            sections.push((format!("receipt:{id}"), SectionKind::Receipt, data.clone()));
        }

        // Replay manifest (optional).
        if let Some(data) = &self.replay_manifest {
            sections.push((
                "replay_manifest".to_string(),
                SectionKind::Payload,
                data.clone(),
            ));
        }

        // Hardware contract (optional).
        if let Some(data) = &self.hardware_contract {
            sections.push((
                "hardware_contract".to_string(),
                SectionKind::Payload,
                data.clone(),
            ));
        }

        // Preserved optional sections.
        for (entry, data) in &self.preserved_optional_sections {
            sections.push((entry.id.clone(), entry.kind.clone(), data.clone()));
        }

        // Step 2: Compute section entries with digests, sorted by id.
        let mut section_entries: Vec<SectionEntry> = sections
            .iter()
            .map(|(id, kind, bytes)| {
                let digest = sha256_digest(bytes);
                SectionEntry {
                    id: id.clone(),
                    kind: kind.clone(),
                    offset: 0,
                    byte_len: bytes.len() as u64,
                    digest,
                }
            })
            .collect();

        // Sort entries by id for canonical section index ordering.
        section_entries.sort_by(|a, b| a.id.cmp(&b.id));
        // Reorder sections to match sorted entries.
        let mut sections_sorted: Vec<_> = sections.iter().collect();
        sections_sorted.sort_by_key(|e| e.0.clone());
        let sections: Vec<(String, SectionKind, Vec<u8>)> = sections_sorted
            .into_iter()
            .map(|(id, kind, bytes)| (id.clone(), kind.clone(), bytes.clone()))
            .collect();

        // Step 3: Compute root digest from sorted-by-id entries (bincode).
        let mut root_entries = section_entries.clone();
        for e in &mut root_entries {
            e.offset = 0;
        }
        let root_bytes = bincode::serialize(&root_entries)
            .map_err(|e| format!("failed to serialize root digest entries: {e}"))?;
        let root_digest = sha256_digest(&root_bytes);

        // Step 4: Build the layout.
        // Layout: header | section_index (bincode, sorted by id, real offsets) | sections
        let header_bytes_len = SEALED_CIMAGE_HEADER_SIZE as u64;

        // Bincode has fixed-size encoding — section index size is deterministic
        // regardless of offset values. Serialize with zero offsets to get size.
        let zero_entries: Vec<SectionEntry> = section_entries
            .iter()
            .map(|e| SectionEntry {
                offset: 0,
                ..e.clone()
            })
            .collect();
        let si_size = bincode::serialized_size(&zero_entries)
            .map_err(|e| format!("failed to calculate section index size: {e}"))?;

        let sections_start = align_64(header_bytes_len + si_size);

        // Compute section offsets.
        let mut cursor = sections_start;
        let mut final_entries: Vec<SectionEntry> = Vec::new();
        for (i, (_id, _kind, bytes)) in sections.iter().enumerate() {
            let aligned_offset = align_64(cursor);
            let end = aligned_offset + bytes.len() as u64;
            final_entries.push(SectionEntry {
                offset: aligned_offset,
                byte_len: bytes.len() as u64,
                ..section_entries[i].clone()
            });
            cursor = align_64(end);
        }

        let section_index_final: Vec<u8> = bincode::serialize(&final_entries)
            .map_err(|e| format!("failed to serialize section index: {e}"))?;
        let si_len = section_index_final.len() as u64;

        // Find manifest offset/length for header convenience fields.
        let manifest_entry = final_entries
            .iter()
            .find(|e| e.id == "manifest")
            .ok_or_else(|| "manifest section not found in index".to_string())?;

        // The section index area starts at header_bytes_len and is padded to
        // the 64-byte aligned sections start.
        // Bincode encoding is fixed-size — si_size == si_len, no iteration needed.

        // Build the header.
        let mut header = SealedCimageHeader::new();
        header.section_count = final_entries.len() as u32;
        header.section_index_offset = header_bytes_len;
        header.section_index_len = si_len;
        header.manifest_offset = manifest_entry.offset;
        header.manifest_len = manifest_entry.byte_len;
        header.root_digest = root_digest;

        // Assemble the output.
        let total_size = final_entries
            .last()
            .map(|e| align_64(e.offset + e.byte_len))
            .unwrap_or(sections_start);
        let mut output = Vec::with_capacity(total_size as usize);

        // Header.
        output.extend_from_slice(&header.to_bytes());
        // Pad to section_index_area_len (header + section index area).
        let header_padding = header_bytes_len as usize - output.len();
        output.extend(std::iter::repeat(0u8).take(header_padding));
        // Section index at known offset.
        output.extend_from_slice(&section_index_final);
        // Pad to sections boundary.
        let pre_sections = output.len();
        let gap = sections_start as usize - pre_sections;
        output.extend(std::iter::repeat(0u8).take(gap));

        // Write sections.
        for (i, entry) in final_entries.iter().enumerate() {
            let (_id, _kind, bytes) = &sections[i];
            let current = output.len() as u64;
            if current < entry.offset {
                let gap = entry.offset - current;
                output.extend(std::iter::repeat(0u8).take(gap as usize));
            }
            output.extend_from_slice(bytes);
            let end = output.len() as u64;
            let padding = align_64(end) - end;
            output.extend(std::iter::repeat(0u8).take(padding as usize));
        }

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // Deserialization
    // -----------------------------------------------------------------------

    /// Deserialize a sealed cimage from bytes.
    ///
    /// Validates magic, version, root digest, and section integrity.
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        // Step 1: Read the header.
        if data.len() < SEALED_CIMAGE_HEADER_SIZE {
            return Err(format!(
                "data too short: got {} bytes, need at least {}",
                data.len(),
                SEALED_CIMAGE_HEADER_SIZE
            ));
        }

        let header = SealedCimageHeader::from_bytes(data)?;

        // Validate magic.
        if header.magic != SEALED_CIMAGE_MAGIC {
            return Err(format!(
                "invalid magic: got {:?}, expected {:?}",
                header.magic, SEALED_CIMAGE_MAGIC
            ));
        }

        // Validate version.
        if header.version > SEALED_CIMAGE_VERSION {
            return Err(format!(
                "unsupported version: got {}, maximum supported: {}",
                header.version, SEALED_CIMAGE_VERSION
            ));
        }

        if header.version != SEALED_CIMAGE_VERSION {
            return Err(format!(
                "version mismatch: got {}, expected {}",
                header.version, SEALED_CIMAGE_VERSION
            ));
        }

        // Step 2: Read the section index.
        let si_start = header.section_index_offset as usize;
        let si_end = si_start + header.section_index_len as usize;
        if si_end > data.len() {
            return Err(format!(
                "section index extends past data end: {si_end} > {}",
                data.len()
            ));
        }
        let section_index_bytes = &data[si_start..si_end];
        let section_entries: Vec<SectionEntry> = bincode::deserialize(section_index_bytes)
            .map_err(|e| format!("failed to parse section index: {e}"))?;

        if section_entries.len() != header.section_count as usize {
            return Err(format!(
                "section count mismatch: header says {}, index has {}",
                header.section_count,
                section_entries.len()
            ));
        }

        // Step 3: Verify root digest against the section index (offset=0).
        // Step 3: Verify root digest against the section index (offset=0).
        let mut zero_offset_entries = section_entries.clone();
        for e in &mut zero_offset_entries {
            e.offset = 0;
        }
        let index_bytes = bincode::serialize(&zero_offset_entries)
            .map_err(|e| format!("failed to serialize section index for verification: {e}"))?;
        let computed_root = sha256_digest(&index_bytes);
        if computed_root != header.root_digest {
            return Err(format!(
                "root digest mismatch: computed {:02x?}, expected {:02x?}",
                computed_root, header.root_digest
            ));
        }

        // Step 4: Read the manifest section.
        let manifest_entry = section_entries
            .iter()
            .find(|e| e.id == "manifest")
            .ok_or_else(|| "manifest section not found in index".to_string())?;

        let manifest_start = manifest_entry.offset as usize;
        let manifest_end = manifest_start + manifest_entry.byte_len as usize;
        if manifest_end > data.len() {
            return Err(format!(
                "manifest section extends past data end: {manifest_end} > {}",
                data.len()
            ));
        }
        let manifest_bytes = &data[manifest_start..manifest_end];

        // Parse manifest.
        let manifest: CanonicalManifest = serde_json::from_slice(manifest_bytes)
            .map_err(|e| format!("failed to parse canonical manifest: {e}"))?;

        // Step 4: Read all sections and build the struct.
        let mut tensor_segments: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut kernel_artifacts: BTreeMap<String, CompiledKernelArtifact> = BTreeMap::new();
        let mut kernel_abis: BTreeMap<String, KernelAbi> = BTreeMap::new();
        let mut receipts: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut tokenizer: Option<(String, Vec<u8>)> = None;
        let mut replay_manifest: Option<Vec<u8>> = None;
        let mut hardware_contract: Option<Vec<u8>> = None;
        let mut generation: Option<CimageGeneration> = None;
        let mut serving_profile: Option<ServingProfile> = None;
        let mut stored_serving_profile = false;
        let mut preserved_optional_sections: Vec<(SectionEntry, Vec<u8>)> = Vec::new();

        for entry in &section_entries {
            let section_start = entry.offset as usize;
            let section_end = section_start + entry.byte_len as usize;
            if section_end > data.len() {
                return Err(format!(
                    "section '{}' extends past data end: {section_end} > {}",
                    entry.id,
                    data.len()
                ));
            }
            let section_bytes = &data[section_start..section_end];

            // Verify section digest.
            let actual_digest = sha256_digest(section_bytes);
            if actual_digest != entry.digest {
                return Err(format!(
                    "digest mismatch for section '{}': computed {:02x?}, expected {:02x?}",
                    entry.id, actual_digest, entry.digest
                ));
            }

            match entry.kind {
                SectionKind::Manifest => {
                    // Already parsed above.
                }
                SectionKind::Generation => {
                    let gen: CimageGeneration = bincode::deserialize(section_bytes)
                        .map_err(|e| format!("failed to deserialize generation: {e}"))?;
                    generation = Some(gen);
                }
                SectionKind::TensorSegment => {
                    // ID format: "tensor:{segment_id}"
                    let seg_id = entry
                        .id
                        .strip_prefix("tensor:")
                        .unwrap_or(&entry.id)
                        .to_string();
                    tensor_segments.insert(seg_id, section_bytes.to_vec());
                }
                SectionKind::KernelArtifact => {
                    let artifact: CompiledKernelArtifact = bincode::deserialize(section_bytes)
                        .map_err(|e| format!("failed to deserialize kernel artifact: {e}"))?;
                    let impl_id = artifact.implementation_id.0.clone();
                    kernel_artifacts.insert(impl_id, artifact);
                }
                SectionKind::KernelAbi => {
                    let abi: KernelAbi = bincode::deserialize(section_bytes)
                        .map_err(|e| format!("failed to deserialize kernel ABI: {e}"))?;
                    // ID format: "abi:{impl_id}"
                    let abi_id = entry
                        .id
                        .strip_prefix("abi:")
                        .unwrap_or(&entry.id)
                        .to_string();
                    kernel_abis.insert(abi_id, abi);
                }
                SectionKind::Tokenizer => {
                    // Tokenizer: identifier comes from manifest.
                    let id = manifest
                        .tokenizer
                        .as_ref()
                        .map(|t| t.identifier.clone())
                        .unwrap_or_default();
                    tokenizer = Some((id, section_bytes.to_vec()));
                }
                SectionKind::Receipt => {
                    let receipt_id = entry
                        .id
                        .strip_prefix("receipt:")
                        .unwrap_or(&entry.id)
                        .to_string();
                    receipts.insert(receipt_id, section_bytes.to_vec());
                }
                SectionKind::Payload => {
                    if entry.id == "replay_manifest" {
                        replay_manifest = Some(section_bytes.to_vec());
                    } else if entry.id == "hardware_contract" {
                        hardware_contract = Some(section_bytes.to_vec());
                    } else if entry.id == "serving_profile" {
                        let profile: ServingProfile = bincode::deserialize(section_bytes)
                            .map_err(|e| format!("failed to deserialize serving profile: {e}"))?;
                        serving_profile = Some(profile);
                        stored_serving_profile = true;
                    }
                }
                SectionKind::Optional => {
                    // Preserve forward-compatible optional section through round-trip.
                    preserved_optional_sections.push((entry.clone(), section_bytes.to_vec()));
                }
            }
        }

        let generation = generation.ok_or_else(|| "generation section is required".to_string())?;

        // Serving profile must come from a dedicated section in version 1 format.
        let serving_profile = serving_profile
            .ok_or_else(|| "serving_profile section not found — required section".to_string())?;

        // Check required capabilities.
        const SUPPORTED_CAPABILITIES: &[&str] = &[
            "nf4",
            "int8",
            "fp16",
            "raw_f32",
            "sym_int4",
            "ternary",
            "ternary_1_58",
            "mixed",
            "q8_0",
            "mtp",
            "serving_profile",
        ];
        for cap in &manifest.required_capabilities {
            if !SUPPORTED_CAPABILITIES.contains(&cap.as_str()) {
                return Err(format!(
                    "unsupported required capability: '{cap}'. Known: {:?}",
                    SUPPORTED_CAPABILITIES
                ));
            }
        }

        Ok(SealedCimageV1 {
            generation,
            tensor_segments,
            kernel_artifacts,
            kernel_abis,
            receipts,
            tokenizer,
            replay_manifest,
            hardware_contract,
            serving_profile,
            stored_serving_profile,
            required_capabilities: manifest.required_capabilities,
            optional_capabilities: manifest.optional_capabilities,
            created_at: manifest.created_at,
            root_digest: header.root_digest,
            preserved_optional_sections,
        })
    }
}

/// A SealedCimageV1 that has passed full integrity validation.
/// Can only be obtained via `ValidatedSealedCimage::validate()`.
#[derive(Debug, Clone)]
pub struct ValidatedSealedCimage(SealedCimageV1);

impl ValidatedSealedCimage {
    /// Validate a raw SealedCimageV1 and return a validated wrapper.
    /// Returns all errors found — never short-circuits.
    pub fn validate(raw: SealedCimageV1) -> Result<Self, Vec<String>> {
        let errors = raw.verify_integrity();
        if errors.is_ok() {
            Ok(ValidatedSealedCimage(raw))
        } else {
            Err(errors.unwrap_err())
        }
    }

    /// Get a reference to the inner SealedCimageV1.
    pub fn inner(&self) -> &SealedCimageV1 {
        &self.0
    }

    /// Consume the wrapper and return the inner SealedCimageV1.
    pub fn into_inner(self) -> SealedCimageV1 {
        self.0
    }

    /// Get the root digest.
    pub fn root_digest(&self) -> &[u8; 32] {
        &self.0.root_digest
    }

    /// Get the generation.
    pub fn generation(&self) -> &CimageGeneration {
        self.0.generation()
    }

    /// Get the serving profile.
    pub fn serving_profile(&self) -> &ServingProfile {
        self.0.serving_profile()
    }

    /// Find a tensor segment by its physical segment ID.
    pub fn find_tensor_segment(&self, id: &str) -> Option<&[u8]> {
        self.0.find_tensor_segment(id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_constitutional::canonical::execution_graph::{
        ExecutionGraph, ExecutionLane, ExecutionRegion, FusionConstraints, MemoryPlan, RegionId,
        RuntimeStatePlan,
    };
    use prism_ecs_constitutional::canonical::generation::RepresentationBinding;
    use prism_ecs_constitutional::canonical::identity::{
        CompilerIdentity, GenerationId, HardwareProfileId, LogicalTensorId, ModelSourceId,
        PhysicalSegmentId, ReceiptId, RepresentationId, Timestamp,
    };
    use prism_ecs_constitutional::canonical::kernel_abi::{
        DispatchGeometryPolicy, KernelAbi, KernelImplementationId, KernelSemanticId,
    };
    use crate::ecs::execution_profile::PhysicalTileLayout;
    use std::collections::BTreeMap;

    /// Build a minimal CimageGeneration for testing.
    fn minimal_generation() -> CimageGeneration {
        CimageGeneration {
            generation_id: GenerationId("test-gen-1".to_string()),
            parent_generation: None,
            base_model: ModelSourceId("test-model".to_string()),
            compiler_identity: CompilerIdentity {
                name: "test-compiler".to_string(),
                version: "1.0.0".to_string(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("apple-m1".to_string()),
            tensor_bindings: {
                let mut m = BTreeMap::new();
                m.insert(
                    LogicalTensorId("test-tensor-0".to_string()),
                    RepresentationBinding {
                        representation_id: RepresentationId("test-repr-0".to_string()),
                        codec: CodecFamily::Nf4,
                        layout: PhysicalTileLayout::default(),
                        primary_segment: PhysicalSegmentId("seg-0".to_string()),
                        scale_segments: vec![PhysicalSegmentId("seg-scale-0".to_string())],
                        residual_segments: vec![],
                        source_representation: None,
                        acceptance_receipt: ReceiptId("receipt-accept-0".to_string()),
                    },
                );
                m
            },
            kernel_bindings: {
                let mut m = BTreeMap::new();
                m.insert(
                    KernelSemanticId("prism.gemma4.prefill.v1".to_string()),
                    KernelImplementationId("test-impl-0".to_string()),
                );
                m
            },
            engram_bindings: BTreeMap::new(),
            execution_graph: ExecutionGraph {
                regions: vec![ExecutionRegion {
                    id: RegionId(0),
                    name: "test-region".to_string(),
                    operations: vec![],
                    target_lane: ExecutionLane::MetalGpu,
                    fusion_constraints: FusionConstraints {
                        max_fused_ops: Some(1),
                        force_fused: false,
                        force_unfused: false,
                    },
                    inputs: vec![],
                    outputs: vec![],
                }],
                edges: vec![],
                state: RuntimeStatePlan {
                    max_context_tokens: 8192,
                    kv_cache_bytes_per_token: 2048,
                    total_kv_cache_bytes: 16_777_216,
                },
                memory: MemoryPlan {
                    total_activation_bytes: 1_048_576,
                    total_weight_bytes: 4_194_304,
                    arena_region_count: 1,
                },
            },
            receipt_root: ReceiptId("root-receipt".to_string()),
            created_at: Timestamp("2026-07-14T00:00:00Z".to_string()),
        }
    }

    /// Build a minimal ServingProfile for testing.
    fn minimal_serving_profile() -> ServingProfile {
        ServingProfile {
            model_name: "test-model".to_string(),
            model_tag: "v1".to_string(),
            architecture: "gemma4".to_string(),
            context_length: 8192,
            precision: "nf4".to_string(),
            mtp_enabled: false,
        }
    }

    /// Build a minimal compiled kernel artifact.
    fn minimal_kernel_artifact() -> CompiledKernelArtifact {
        CompiledKernelArtifact {
            implementation_id: KernelImplementationId("test-impl-0".to_string()),
            semantic_id: KernelSemanticId("prism.gemma4.prefill.v1".to_string()),
            compiled_bytes: vec![0xde, 0xad, 0xbe, 0xef],
            sha256: "test-sha256".to_string(),
            entry_point: "gemma4_prefill".to_string(),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
                threads_per_threadgroup: (32, 1, 1),
            },
        }
    }

    fn minimal_kernel_abi() -> KernelAbi {
        KernelAbi {
            version: 1,
            buffers: vec![],
            constants: vec![],
            threadgroup_memory: vec![],
            dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
            threads_per_threadgroup: (32, 1, 1),
        }
    }

    #[test]
    fn test_deterministic_roundtrip() {
        let gen = minimal_generation();
        let profile = minimal_serving_profile();
        let artifact = minimal_kernel_artifact();
        let abi = minimal_kernel_abi();

        let sealed = SealedCimageBuilder::new()
            .with_generation(gen)
            .with_serving_profile(profile)
            .add_tensor_segment("seg-0".to_string(), vec![0x01, 0x02, 0x03, 0x04])
            .unwrap()
            .add_tensor_segment("seg-scale-0".to_string(), vec![0x10, 0x20])
            .unwrap()
            .add_kernel_artifact("test-impl-0".to_string(), artifact)
            .unwrap()
            .add_kernel_abi("test-impl-0".to_string(), abi)
            .unwrap()
            .build()
            .expect("builder should succeed");

        // Compute root digest.
        let root_digest = sealed.root_digest();

        // Serialize.
        let bytes = sealed.serialize().expect("serialization should succeed");

        // Deserialize.
        let deserialized =
            SealedCimageV1::deserialize(&bytes).expect("deserialization should succeed");

        // Verify root digest matches.
        assert_eq!(deserialized.root_digest(), root_digest);

        // Verify key data survived (serving profile now round-trips exactly).
        assert_eq!(deserialized.generation().generation_id.0, "test-gen-1");
        assert_eq!(deserialized.serving_profile().model_name, "test-model");
        assert_eq!(deserialized.serving_profile().architecture, "gemma4");
        assert_eq!(deserialized.serving_profile().context_length, 8192);
        assert_eq!(
            deserialized.find_tensor_segment("seg-0"),
            Some(&[0x01, 0x02, 0x03, 0x04][..])
        );
        assert_eq!(
            deserialized.find_tensor_segment("seg-scale-0"),
            Some(&[0x10, 0x20][..])
        );
        assert!(deserialized.find_kernel_artifact("test-impl-0").is_some());

        // Re-serialize and compare bytes identically.
        let bytes2 = deserialized
            .serialize()
            .expect("second serialization should succeed");
        assert_eq!(bytes, bytes2, "serialization must be deterministic");
    }

    #[test]
    fn test_builder_rejects_duplicate_segments() {
        let gen = minimal_generation();
        let profile = minimal_serving_profile();

        let result = SealedCimageBuilder::new()
            .with_generation(gen)
            .with_serving_profile(profile)
            .add_tensor_segment("seg-0".to_string(), vec![0x01, 0x02, 0x03, 0x04])
            .unwrap()
            .add_tensor_segment("seg-0".to_string(), vec![0x05, 0x06]);

        assert!(result.is_err(), "duplicate segment should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("duplicate tensor segment id"),
            "error should mention duplicate: {err}"
        );
    }

    #[test]
    fn test_builder_rejects_missing_referenced_segment() {
        // Create a generation with a tensor binding that references "seg-missing".
        let mut gen = minimal_generation();
        gen.tensor_bindings.insert(
            LogicalTensorId("orphan-tensor".to_string()),
            RepresentationBinding {
                representation_id: RepresentationId("orphan-repr".to_string()),
                codec: CodecFamily::Int8,
                layout: PhysicalTileLayout::default(),
                primary_segment: PhysicalSegmentId("seg-missing".to_string()),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("orphan-receipt".to_string()),
            },
        );

        let profile = minimal_serving_profile();

        let result = SealedCimageBuilder::new()
            .with_generation(gen)
            .with_serving_profile(profile)
            .add_tensor_segment("seg-0".to_string(), vec![0x01])
            .unwrap()
            .build();

        assert!(result.is_err(), "missing segment should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("seg-missing"),
            "error should mention missing segment id: {err}"
        );
    }

    #[test]
    fn test_verify_integrity_detects_corruption() {
        let gen = minimal_generation();
        let profile = minimal_serving_profile();
        let artifact = minimal_kernel_artifact();
        let abi = minimal_kernel_abi();

        let sealed = SealedCimageBuilder::new()
            .with_generation(gen)
            .with_serving_profile(profile)
            .add_tensor_segment("seg-0".to_string(), vec![0x01, 0x02, 0x03, 0x04])
            .unwrap()
            .add_tensor_segment("seg-scale-0".to_string(), vec![0x10, 0x20])
            .unwrap()
            .add_kernel_artifact("test-impl-0".to_string(), artifact)
            .unwrap()
            .add_kernel_abi("test-impl-0".to_string(), abi)
            .unwrap()
            .build()
            .expect("builder should succeed");

        let bytes = sealed.serialize().expect("serialization should succeed");

        // Corrupt one byte in the data.
        let mut corrupted = bytes.clone();

        // Find the manifest section and corrupt a manifest byte.
        // We corrupt the first tensor segment instead (offset determined by layout).
        // The simplest approach: corrupt a known offset.
        // After header (128) + section index + padding, the first section is the manifest.
        // Let's find "tensor:seg-0" in the bytes.
        let corrupt_offset = if let Some(pos) = corrupted
            .windows(4)
            .position(|w| w == [0x01, 0x02, 0x03, 0x04])
        {
            pos
        } else {
            // Fallback: corrupt offset 512 (likely inside a section)
            512usize.min(corrupted.len() - 1)
        };
        corrupted[corrupt_offset] ^= 0xff;

        let result = SealedCimageV1::deserialize(&corrupted);
        assert!(
            result.is_err(),
            "deserialization should detect corruption, got Ok"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("digest mismatch") || err.contains("root digest mismatch"),
            "error should mention digest mismatch: {err}"
        );
    }

    #[test]
    fn test_deserialize_rejects_wrong_magic() {
        let bad_bytes = vec![0x00u8; 512];
        let result = SealedCimageV1::deserialize(&bad_bytes);
        assert!(result.is_err(), "wrong magic should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("invalid magic") || err.contains("magic"),
            "error should mention invalid magic: {err}"
        );
    }

    #[test]
    fn test_deserialize_rejects_unsupported_version() {
        // Build a valid header with version = 999.
        let mut header = SealedCimageHeader::new();
        header.version = 999;
        header.magic = SEALED_CIMAGE_MAGIC;
        let header_bytes = header.to_bytes();
        let mut data = Vec::new();
        data.extend_from_slice(&header_bytes);
        // Pad to have enough data for a minimum attempt.
        data.resize(512, 0u8);

        let result = SealedCimageV1::deserialize(&data);
        assert!(result.is_err(), "unsupported version should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("unsupported version") || err.contains("version"),
            "error should mention version: {err}"
        );
    }

    #[test]
    fn test_builder_missing_required_fields() {
        // No generation, no serving profile.
        let result = SealedCimageBuilder::new().build();
        assert!(
            result.is_err(),
            "missing required fields should be rejected"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("generation") || err.contains("serving_profile"),
            "error should mention missing fields: {err}"
        );
    }

    #[test]
    fn test_sealed_sections_independently_addressable() {
        let gen = minimal_generation();
        let profile = minimal_serving_profile();
        let artifact = minimal_kernel_artifact();
        let abi = minimal_kernel_abi();

        let sealed = SealedCimageBuilder::new()
            .with_generation(gen)
            .with_serving_profile(profile)
            .add_tensor_segment("seg-0".to_string(), vec![0x01, 0x02, 0x03, 0x04])
            .unwrap()
            .add_tensor_segment("seg-scale-0".to_string(), vec![0x10, 0x20])
            .unwrap()
            .add_kernel_artifact("test-impl-0".to_string(), artifact)
            .unwrap()
            .add_kernel_abi("test-impl-0".to_string(), abi)
            .unwrap()
            .build()
            .expect("builder should succeed");

        let bytes = sealed.serialize().expect("serialization should succeed");

        // Parse the header and section index to verify all sections are independently readable.
        let header = SealedCimageHeader::from_bytes(&bytes).expect("header should be readable");
        assert_eq!(header.magic, SEALED_CIMAGE_MAGIC);
        assert_eq!(header.version, SEALED_CIMAGE_VERSION);

        let si_start = header.section_index_offset as usize;
        let si_end = si_start + header.section_index_len as usize;
        let si_bytes = &bytes[si_start..si_end];
        let entries: Vec<SectionEntry> =
            bincode::deserialize(si_bytes).expect("section index should parse");

        // Verify at least manifest, generation, and the two tensor segments exist.
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"manifest"), "manifest section must exist");
        assert!(ids.contains(&"generation"), "generation section must exist");
        assert!(ids.contains(&"tensor:seg-0"), "tensor:seg-0 must exist");
        assert!(
            ids.contains(&"tensor:seg-scale-0"),
            "tensor:seg-scale-0 must exist"
        );

        // Verify each section is 64-byte aligned.
        for entry in &entries {
            assert_eq!(
                entry.offset % 64,
                0,
                "section '{}' offset {} is not 64-byte aligned",
                entry.id,
                entry.offset
            );
        }

        // Verify each section digest independently.
        for entry in &entries {
            let section_start = entry.offset as usize;
            let section_end = section_start + entry.byte_len as usize;
            let section_bytes = &bytes[section_start..section_end];
            let expected_digest = sha256_digest(section_bytes);
            assert_eq!(
                expected_digest, entry.digest,
                "digest mismatch for section '{}'",
                entry.id
            );
        }
    }
}
