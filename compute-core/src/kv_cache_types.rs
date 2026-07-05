//! Pure-data types for KV cache, extracted from `kv_cache_coordinator.rs`
//! for hermetic access under `prism-backend`.
//!
//! These types have no mlx dependency: they operate on `Vec<u8>` byte buffers,
//! tier/backing enums, and generic trait-based policy dispatch.

use std::time::{Duration, Instant};

use crate::quantization::turboquant_kv::QjlCorrection;

// ── CompressedKvSlot ───────────────────────────────────────────────────────

/// Backing store for one compressed KV slot.
///
/// Unlike the raw [`KvCache`] which stores FP16 tensors in MLX arrays,
/// this stores the compressed byte buffers and optional QJL correction bits.
#[derive(Debug, Clone)]
pub struct CompressedKvSlot {
    /// Compressed keys + quantization scale/indices.
    pub compressed_keys: Vec<u8>,
    /// Compressed values + quantization scale/indices.
    pub compressed_values: Vec<u8>,
    /// QJL residual correction bits (separate for fast access).
    pub qjl_correction: Option<QjlCorrection>,
    /// Offset of this slot's first token in the global KV sequence.
    pub kv_offset: u32,
    /// Number of tokens stored in this slot.
    pub num_tokens: usize,
}

impl CompressedKvSlot {
    /// Create a new empty slot at the given KV offset.
    pub fn new(kv_offset: u32) -> Self {
        Self {
            compressed_keys: Vec::new(),
            compressed_values: Vec::new(),
            qjl_correction: None,
            kv_offset,
            num_tokens: 0,
        }
    }

    /// Serialize this slot into a single page-data blob for distributed
    /// KV cache transport (RDMA).
    ///
    /// Format (little-endian):
    ///   [key_len: u32][keys_data][val_len: u32][values_data]
    ///
    /// The 2-bit compressed data is transferred as-is; the receiving
    /// node uses [`from_page_data`](Self::from_page_data) to reconstruct.
    pub fn to_page_data(&self) -> Vec<u8> {
        let key_len = self.compressed_keys.len() as u32;
        let val_len = self.compressed_values.len() as u32;
        let mut buf = Vec::with_capacity(8 + key_len as usize + val_len as usize);
        buf.extend_from_slice(&key_len.to_le_bytes());
        buf.extend_from_slice(&self.compressed_keys);
        buf.extend_from_slice(&val_len.to_le_bytes());
        buf.extend_from_slice(&self.compressed_values);
        buf
    }

    /// Reconstruct a slot from its page-data blob (produced by
    /// [`to_page_data`](Self::to_page_data)).
    ///
    /// The slot will have `kv_offset = 0` and `num_tokens = 1`; the
    /// caller should adjust these after insertion into a
    /// [`CompressedKvCache`].
    pub fn from_page_data(data: &[u8]) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("page data too short: need at least 8 bytes for headers".into());
        }
        let key_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        let val_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if 8 + key_len + val_len != data.len() {
            return Err(format!(
                "page data length mismatch: expected {} bytes, got {}",
                8 + key_len + val_len,
                data.len()
            ));
        }
        let keys_end = 8 + key_len;
        let compressed_keys = data[8..keys_end].to_vec();
        let compressed_values = data[keys_end..].to_vec();
        Ok(Self {
            compressed_keys,
            compressed_values,
            qjl_correction: None,
            kv_offset: 0,
            num_tokens: 1,
        })
    }

    /// Returns `true` if this slot contains no compressed data.
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Allocated bytes for this slot (compressed buffers + correction + fixed overhead).
    pub fn allocated_bytes(&self) -> u64 {
        let data = self.compressed_keys.len()
            + self.compressed_values.len()
            + self
                .qjl_correction
                .as_ref()
                .map_or(0, |c| c.residual_bits.len());
        // Fixed overhead: three Vec's (24 bytes each on 64-bit) + Option overhead + fields
        let overhead: u64 = 80;
        data as u64 + overhead
    }
}

// ── 3-Tier KV Cache with ANE-Managed Page Migration ───────────────────────

/// Tier identifier for KV cache page location.
///
/// Pages migrate between tiers based on access frequency.
/// Tiers are platform-agnostic: the lowest-numbered tier is the fastest
/// device-local memory (dGPU VRAM, ANE SRAM, etc.). Platform-specific
/// mapping is configured at runtime via MemoryDomain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheTier {
    /// Fastest device-local memory: dGPU VRAM, ANE SRAM, etc.
    L0Device,
    /// Shared/unified memory accessible by CPU and GPU (IOSurface, CUDA Managed, etc.)
    L1Shared,
    /// Host system DRAM
    L2System,
    /// Disk-backed cold storage
    L3Disk,
}

impl KVCacheTier {
    /// Human-readable label for this tier.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::L0Device => "L0-Device",
            Self::L1Shared => "L1-Shared",
            Self::L2System => "L2-System",
            Self::L3Disk => "L3-Disk",
        }
    }

    /// Human-readable description of what this tier represents.
    pub const fn description(&self) -> &'static str {
        match self {
            Self::L0Device => "Fastest device-local memory (dGPU VRAM, ANE SRAM)",
            Self::L1Shared => "Shared/unified memory (IOSurface, CUDA Managed)",
            Self::L2System => "Host system DRAM",
            Self::L3Disk => "Disk-backed cold storage",
        }
    }
}

// ── PageBacking ────────────────────────────────────────────────────────────

/// Platform-agnostic backing handle for a KV cache page.
///
/// Each variant corresponds to a different physical memory type.
/// The tier enum describes the *semantic* level; this enum carries
/// the platform-specific handle for the actual storage.
#[derive(Debug, Clone)]
pub enum PageBacking {
    /// No backing storage (page has no resident data).
    None,
    /// Device-local memory buffer (dGPU VRAM, ANE SRAM).
    DeviceBuffer { handle: u64, byte_size: u64 },
    /// Shared/unified memory accessible by CPU and GPU.
    SharedBuffer {
        handle: u64,
        byte_size: u64,
        domain_tag: u32,
    },
    /// Host system DRAM (resident in process heap).
    SystemHeap { byte_size: u64 },
    /// Disk-backed (no resident data, token_start encodes the path).
    Disk,
}

impl PageBacking {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DeviceBuffer { .. } => "device-buffer",
            Self::SharedBuffer { .. } => "shared-buffer",
            Self::SystemHeap { .. } => "system-heap",
            Self::Disk => "disk",
        }
    }
}

// ── Disk-backed KV cache page eviction helpers ─────────────────────────────

/// Disk-backed KV cache page storage directory.
const KV_CACHE_DISK_DIR: &str = "/tmp/tribunus-kv-cache";

/// Write a cold page to disk. The file is an mmap'd segment containing
/// the 2-bit compressed page data.
pub fn evict_page_to_disk(page: &TiersPage) -> Result<String, String> {
    let _ = std::fs::create_dir_all(KV_CACHE_DISK_DIR);

    let filename = format!("{}/page_{:x}.kvp", KV_CACHE_DISK_DIR, page.token_start);

    let compressed = page
        .data
        .as_ref()
        .ok_or_else(|| "evict_page_to_disk: page has no data".to_string())?;

    std::fs::write(&filename, compressed).map_err(|e| format!("write KV page: {}", e))?;

    Ok(filename)
}

/// Load a page from disk into memory (as L3 data).
/// Returns the raw bytes read from the disk file.
pub fn load_page_from_disk(filename: &str) -> Result<Vec<u8>, String> {
    let data = std::fs::read(filename).map_err(|e| format!("read KV page: {}", e))?;
    let _ = std::fs::remove_file(filename);
    Ok(data)
}

// ── TiersPage ──────────────────────────────────────────────────────────────

/// A KV page tracked across tiers, holding data at up to one tier at a time.
///
/// Each page covers a contiguous range of token IDs (`token_start`..`token_end`).
/// The `current_tier` field indicates which tier the data is resident in.
/// The `data` field holds the opaque byte payload for tiers that keep data
/// in the process heap. The `backing` field holds a platform-specific handle
/// for tiers that use device or shared memory.
/// Promotion moves data from higher-numbered to lower-numbered tiers
/// (e.g. L2System → L1Shared → L0Device). Demotion moves the opposite direction.
pub struct TiersPage {
    /// Opaque page data payload (compressed bytes, raw FP16, etc.).
    /// The format is determined by the current tier and the compressor.
    pub data: Option<Vec<u8>>,
    /// Platform-specific backing handle (device buffer, shared buffer, etc.).
    pub backing: PageBacking,
    /// Current tier (the fastest one with data).
    pub current_tier: KVCacheTier,
    /// Last access time for promotion/demotion decisions.
    pub last_access: Instant,
    /// Token ID range this page covers.
    pub token_start: u32,
    /// End token (exclusive) for this page's range.
    pub token_end: u32,
}

impl TiersPage {
    /// Create a new page at the given tier with the provided token range.
    ///
    /// All data fields are set to `None` except the one matching `initial_tier`.
    /// The caller must fill in the appropriate data after creation.
    pub fn new(token_start: u32, token_end: u32, initial_tier: KVCacheTier) -> Self {
        let (data, backing) = match initial_tier {
            KVCacheTier::L2System => (Some(Vec::new()), PageBacking::SystemHeap { byte_size: 0 }),
            KVCacheTier::L3Disk => (None, PageBacking::Disk),
            // L0Device and L1Shared require platform-specific handles,
            // so they start with no heap data and no backing.
            _ => (None, PageBacking::None),
        };
        Self {
            data,
            backing,
            current_tier: initial_tier,
            last_access: Instant::now(),
            token_start,
            token_end,
        }
    }

    /// Record an access to this page, updating the last-access timestamp.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// Returns `true` if this page covers the given token position.
    pub fn contains_token(&self, token_id: u32) -> bool {
        token_id >= self.token_start && token_id < self.token_end
    }

    /// Returns the number of tokens this page covers.
    pub fn token_count(&self) -> u32 {
        self.token_end.saturating_sub(self.token_start)
    }

    /// Returns an estimate of the allocated bytes for this page at its current tier.
    pub fn allocated_bytes(&self) -> u64 {
        let overhead = std::mem::size_of::<Self>() as u64;
        let data_bytes = self.data.as_ref().map_or(0, |d| d.len() as u64);
        let backing_bytes = match &self.backing {
            PageBacking::None | PageBacking::Disk => 0,
            PageBacking::DeviceBuffer { byte_size, .. }
            | PageBacking::SharedBuffer { byte_size, .. }
            | PageBacking::SystemHeap { byte_size, .. } => *byte_size,
        };
        data_bytes + backing_bytes + overhead
    }
}

// ── PageMigrationPolicy ────────────────────────────────────────────────────

/// Platform-agnostic policy for KV cache page migration decisions.
///
/// Implementations provide platform-specific promotion, demotion, and
/// access-pattern evaluation logic. The [`PageMigrationService`] calls
/// this trait on every `tick()`.
pub trait PageMigrationPolicy: Send + Sync {
    /// Evaluate a single page and decide what action to take.
    ///
    /// Called for each tracked page during [`PageMigrationService::tick()`].
    /// The implementation may modify the page (promote, demote, keep) and
    /// returns `Ok(())` on success.
    fn evaluate_tick(&self, page: &mut TiersPage, now: Instant) -> Result<(), String>;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &'static str;
}

// ── PageMigrationService ──────────────────────────────────────────────────

/// Generic KV cache page migration service.
///
/// Manages a collection of [`TiersPage`] instances and delegates tier-specific
/// promotion/demotion decisions to a [`PageMigrationPolicy`].
/// Platform-agnostic logic (page registration, access tracking, disk eviction,
/// prefetch, EvolKV budget search) lives here.
pub struct PageMigrationService {
    /// All pages tracked across tiers.
    pub pages: Vec<TiersPage>,
    /// Platform-specific page migration policy.
    pub policy: Box<dyn PageMigrationPolicy>,
    /// Total cache budget in bytes for EvolKV optimization.
    pub total_cache_budget: usize,
    /// Best EvolKV budget found during search (None until first search).
    /// Only available under `mlx-backend`; the `learn_evolk_budgets` method
    /// lives in `kv_cache_coordinator.rs`.
    #[cfg(feature = "mlx-backend")]
    pub evolvk_budget: Option<crate::cache::evolkv::LayerBudget>,
}

impl PageMigrationService {
    /// Create a new page migration service.
    ///
    /// `policy` — the platform-specific migration policy that handles
    /// compress/decompress and tier promotion/demotion.
    /// `total_cache_budget` — total cache budget for EvolKV optimization.
    pub fn new(policy: Box<dyn PageMigrationPolicy>, total_cache_budget: usize) -> Self {
        Self {
            pages: Vec::new(),
            policy,
            total_cache_budget,
            #[cfg(feature = "mlx-backend")]
            evolvk_budget: None,
        }
    }

    /// Register a new page in the tiered cache.
    ///
    /// The page starts at `initial_tier` and is assigned the given token range.
    pub fn add_page(&mut self, token_start: u32, token_end: u32, initial_tier: KVCacheTier) {
        self.pages
            .push(TiersPage::new(token_start, token_end, initial_tier));
    }

    /// Record an access to the page covering `token_id`, updating its
    /// last-access timestamp.
    pub fn touch_token(&mut self, token_id: u32) {
        for page in &mut self.pages {
            if page.contains_token(token_id) {
                page.touch();
                return;
            }
        }
    }

    /// Called periodically (e.g. after every decode step) to examine all
    /// pages and promote/demote based on access time.
    ///
    /// Delegates per-page evaluation to the [`PageMigrationPolicy`].
    /// A failed promotion leaves the page at its current tier.
    pub fn tick(&mut self) -> Result<(), String> {
        let now = Instant::now();
        let pages = std::mem::take(&mut self.pages);
        for mut page in pages {
            self.policy.evaluate_tick(&mut page, now)?;
            self.pages.push(page);
        }
        Ok(())
    }

    /// Return the total allocated bytes across all tracked pages.
    pub fn allocated_bytes(&self) -> u64 {
        let pages: u64 = self.pages.iter().map(|p| p.allocated_bytes()).sum();
        let struct_overhead = std::mem::size_of::<Self>() as u64;
        pages + struct_overhead
    }

    /// Return counts of pages at each tier.
    pub fn tier_counts(&self) -> (usize, usize, usize, usize) {
        let mut l0 = 0usize;
        let mut l1 = 0usize;
        let mut l2 = 0usize;
        let mut l3 = 0usize;
        for page in &self.pages {
            match page.current_tier {
                KVCacheTier::L0Device => l0 += 1,
                KVCacheTier::L1Shared => l1 += 1,
                KVCacheTier::L2System => l2 += 1,
                KVCacheTier::L3Disk => l3 += 1,
            }
        }
        (l0, l1, l2, l3)
    }

    /// Check KV cache pressure, evict cold pages to disk.
    /// Pages not accessed for >cold_threshold are candidates for disk eviction.
    /// Returns the number of pages evicted to disk.
    pub fn check_and_evict(&mut self, cold_threshold: Duration) -> Result<usize, String> {
        let now = Instant::now();
        let mut evicted = 0usize;
        let mut to_evict: Vec<usize> = Vec::new();

        for (i, page) in self.pages.iter().enumerate() {
            let age = now.duration_since(page.last_access);
            // Pages already at L2System that are cold enough for disk eviction
            if age > cold_threshold && page.current_tier == KVCacheTier::L2System {
                to_evict.push(i);
            }
        }

        // Evict in reverse order so indices stay valid
        for &idx in to_evict.iter().rev() {
            let _filename = evict_page_to_disk(&self.pages[idx])?;
            self.pages[idx].data = None;
            self.pages[idx].backing = PageBacking::Disk;
            self.pages[idx].current_tier = KVCacheTier::L3Disk;
            evicted += 1;
        }

        Ok(evicted)
    }

    /// Prefetch KV pages predicted to be needed next.
    /// Loads disk pages back to L2System based on predicted access.
    /// Currently uses a simple heuristic: pages adjacent to recently accessed
    /// tokens are prefetched.
    pub fn prefetch_predicted(&mut self, hot_threshold: Duration) -> Result<usize, String> {
        let now = Instant::now();
        let mut prefetched = 0usize;
        let hot_tokens: Vec<u32> = self
            .pages
            .iter()
            .filter(|p| {
                let age = now.duration_since(p.last_access);
                age < hot_threshold && p.current_tier != KVCacheTier::L3Disk
            })
            .map(|p| p.token_start)
            .collect();

        // For each hot token range, prefetch adjacent disk pages
        for &hot_start in &hot_tokens {
            for adj in [hot_start.saturating_sub(256), hot_start + 256] {
                if let Some(page) = self
                    .pages
                    .iter_mut()
                    .find(|p| p.current_tier == KVCacheTier::L3Disk && p.contains_token(adj))
                {
                    let filename = format!("{}/page_{:x}.kvp", KV_CACHE_DISK_DIR, page.token_start);
                    let raw_data = load_page_from_disk(&filename)?;
                    page.data = Some(raw_data);
                    page.backing = PageBacking::SystemHeap { byte_size: 0 };
                    page.current_tier = KVCacheTier::L2System;
                    page.touch();
                    prefetched += 1;
                }
            }
        }

        Ok(prefetched)
    }
}
