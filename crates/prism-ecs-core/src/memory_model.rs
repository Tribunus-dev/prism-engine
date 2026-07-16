//! RPCS3 Cell/B.E. memory model patterns for the Prism ECS runtime.
//!
//! The Cell Broadband Engine splits physical memory into 256 MiB XDR (main
//! processor) and 256 MiB GDDR3 (RSX/video).  RPCS3 models this with:
//!
//! - **`block_t`** — page-aligned memory regions with type flags identifying
//!   both memory type and RSX context association.
//! - **`g_pages`** — a page table carrying reservation stamps for coherence
//!   tracking across the two memory domains.
//! - **Reservation slots** — timestamp-based conflict detection that guards
//!   atomic read-modify-write sequences (PPU `lwarx`/`stwcx.` and similar).
//!
//! The types below translate these idioms into Prism ECS components and a
//! coherence system.

use crate::component::Component;
use crate::scheduling::component_id::{ComponentId, SchedulableComponent, SchedulableResource};
use crate::scheduling::metadata::{Stage, SystemId, SystemSpec};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants — memory class bits, domain flags, and page permissions
// ---------------------------------------------------------------------------

/// Bit position for the "main memory" (XDR) class flag.
pub const MEM_CLASS_MAIN: u64 = 1 << 0;
/// Bit position for the "local/video memory" (GDDR3) class flag.
pub const MEM_CLASS_LOCAL: u64 = 1 << 1;

/// Bit position: page is readable.
pub const PAGE_READABLE: u64 = 1 << 0;
/// Bit position: page is writable.
pub const PAGE_WRITABLE: u64 = 1 << 1;
/// Bit position: page is executable.
pub const PAGE_EXECUTABLE: u64 = 1 << 2;
/// Bit position: page has been access-faulted in (demand paging).
pub const PAGE_PRESENT: u64 = 1 << 3;

// ---------------------------------------------------------------------------
// Newtype identifiers
// ---------------------------------------------------------------------------

/// Stable identifier for a memory domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryDomainId(pub u16);

/// Stable identifier for an execution context that may hold reservation slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionContextId(pub u64);

// ---------------------------------------------------------------------------
// MemoryDomain component
// ---------------------------------------------------------------------------

/// A page-aligned memory region — analogue of RPCS3's `block_t`.
///
/// Each domain represents one contiguous physical address space (XDR main,
/// GDDR3 RSX video, MMIO-mapped I/O, etc.).  The scheduler uses `memory_class`
/// to decide transfer policy between unified and discrete domains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryDomain {
    /// Human-readable label (e.g. `"xdr_main"`, `"gddr3_video"`, `"mmio"`).
    pub name: String,
    /// Base physical address of this domain.
    pub base_addr: u64,
    /// Size of the domain in bytes.
    pub size_bytes: u64,
    /// Memory class flags (MEM_CLASS_MAIN | MEM_CLASS_LOCAL, etc.).
    ///
    /// Consumers inspect individual bit positions rather than comparing
    /// against an enum discriminant so that a domain may carry multiple
    /// classifications simultaneously.
    pub memory_class: u64,
    /// Page size for this domain in bytes (typically 4096 or 65536).
    pub page_size: u64,
}

impl Component for MemoryDomain {}

impl MemoryDomain {
    /// Create a new memory domain.
    pub fn new(
        name: impl Into<String>,
        base_addr: u64,
        size_bytes: u64,
        memory_class: u64,
        page_size: u64,
    ) -> Self {
        Self {
            name: name.into(),
            base_addr,
            size_bytes,
            memory_class,
            page_size,
        }
    }

    /// Number of pages in this domain, rounded up.
    pub fn page_count(&self) -> u64 {
        self.size_bytes.div_ceil(self.page_size)
    }

    /// Translate a domain-relative offset to an absolute physical address.
    pub fn to_phys(&self, offset: u64) -> Option<u64> {
        if offset < self.size_bytes {
            Some(self.base_addr.wrapping_add(offset))
        } else {
            None
        }
    }

    /// Returns `true` if `addr` falls within this domain's range.
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base_addr && addr < self.base_addr.wrapping_add(self.size_bytes)
    }
}

// ---------------------------------------------------------------------------
// PageEntry
// ---------------------------------------------------------------------------

/// Single page table entry.
///
/// Mirrors RPCS3's page table entries: a physical address (or zero for
/// unmapped), an owning domain, and permission/state flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageEntry {
    /// Physical address the page maps to (0 = unmapped/fault).
    pub phys_addr: u64,
    /// Domain that owns this physical page.
    pub domain_id: MemoryDomainId,
    /// Permission and state flags (PAGE_READABLE | PAGE_WRITABLE | …).
    pub flags: u64,
}

impl PageEntry {
    /// Create a new page table entry.
    pub fn new(phys_addr: u64, domain_id: MemoryDomainId, flags: u64) -> Self {
        Self {
            phys_addr,
            domain_id,
            flags,
        }
    }

    /// Returns `true` if the page is present (mapped) in memory.
    pub fn is_present(&self) -> bool {
        self.flags & PAGE_PRESENT != 0 && self.phys_addr != 0
    }

    /// Returns `true` if the page is readable.
    pub fn is_readable(&self) -> bool {
        self.flags & PAGE_READABLE != 0
    }

    /// Returns `true` if the page is writable.
    pub fn is_writable(&self) -> bool {
        self.flags & PAGE_WRITABLE != 0
    }

    /// Returns `true` if the page is executable.
    pub fn is_executable(&self) -> bool {
        self.flags & PAGE_EXECUTABLE != 0
    }
}

// ---------------------------------------------------------------------------
// PageTable component
// ---------------------------------------------------------------------------

/// A per-entity (or per-context) page table.
///
/// Analogous to RPCS3's `g_pages`: a dense vector indexed by virtual page
/// number, each entry holding the physical address, owning domain, and
/// permission/coherence flags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageTable {
    /// Entries indexed by virtual page number.
    pub entries: Vec<PageEntry>,
    /// Number of significant bits in the virtual address (page-aligned).
    pub address_bits: u8,
}

impl Component for PageTable {}

impl PageTable {
    /// Construct a page table with `page_count` entries all initially
    /// unmapped.  `address_bits` is the virtual address width (e.g. 40
    /// for a 1 TiB address space).
    pub fn new(page_count: usize, address_bits: u8) -> Self {
        Self {
            entries: vec![PageEntry::new(0, MemoryDomainId(0), 0); page_count],
            address_bits,
        }
    }

    /// Map virtual page `vpn` to the given physical address and flags.
    ///
    /// # Panics
    ///
    /// Panics if `vpn` is out of bounds.
    pub fn map(&mut self, vpn: usize, phys_addr: u64, domain_id: MemoryDomainId, flags: u64) {
        self.entries[vpn] = PageEntry::new(phys_addr, domain_id, flags | PAGE_PRESENT);
    }

    /// Unmap virtual page `vpn`, returning the previous entry.
    ///
    /// # Panics
    ///
    /// Panics if `vpn` is out of bounds.
    pub fn unmap(&mut self, vpn: usize) -> PageEntry {
        let old = self.entries[vpn];
        self.entries[vpn] = PageEntry::new(0, MemoryDomainId(0), 0);
        old
    }

    /// Look up the page table entry for virtual page `vpn`.
    pub fn entry(&self, vpn: usize) -> Option<&PageEntry> {
        self.entries.get(vpn)
    }

    /// Mutable lookup for the page table entry at `vpn`.
    pub fn entry_mut(&mut self, vpn: usize) -> Option<&mut PageEntry> {
        self.entries.get_mut(vpn)
    }
}

// ---------------------------------------------------------------------------
// ReservationSlot component
// ---------------------------------------------------------------------------

/// A reservation slot tracking an in-progress atomic sequence on a cache line.
///
/// RPCS3's Cell PPU uses `lwarx` (load word and reserve indexed) /
/// `stwcx.` (store word conditional indexed) for atomic memory operations.
/// A reservation slot records the address and a timestamp so the coherence
/// system can detect whether another context wrote to the cache line between
/// the reserve and the conditional store, forcing a rollback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservationSlot {
    /// Aligned address of the reserved cache line (reservation granule).
    pub address: u64,
    /// Monotonically increasing timestamp from the issuing context's local
    /// counter — used to order and timeout reservations.
    pub timestamp: u64,
    /// The execution context that currently holds (or last held) this
    /// reservation.
    pub acquired_by: ExecutionContextId,
}

impl Component for ReservationSlot {}

impl ReservationSlot {
    /// Create a new reservation slot.
    pub fn new(address: u64, timestamp: u64, acquired_by: ExecutionContextId) -> Self {
        Self {
            address,
            timestamp,
            acquired_by,
        }
    }
}

// ---------------------------------------------------------------------------
// ReservationSystem — coherence checking
// ---------------------------------------------------------------------------

/// Result of a coherence check over a set of reservation slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoherenceResult {
    /// All reservations are still valid — no conflicting writes detected.
    Clean,
    /// One or more reservations were invalidated by a conflicting write.
    Conflict {
        /// The slot addresses that triggered the conflict.
        conflicting_addresses: Vec<u64>,
    },
}

impl CoherenceResult {
    /// Returns `true` if this is a conflict.
    pub fn is_conflict(&self) -> bool {
        matches!(self, CoherenceResult::Conflict { .. })
    }
}

/// Reservation coherence system — stateless bundle of checking functions.
///
/// In RPCS3, reservation coherence guards `lwarx`/`stwcx.` atomic sequences:
/// a store to a reserved address between reserve and conditional-store
/// invalidates the reservation, forcing the caller to retry.
///
/// This system wraps those checks so the schedule compiler can declare
/// read/write access on slot components and schedule it in the appropriate
/// coherence stage.
pub struct ReservationSystem;

impl ReservationSystem {
    /// System ID assigned to the reservation-checking system.
    pub const SYSTEM_ID: SystemId = SystemId(1801);
    /// System name for diagnostics and schedule manifests.
    pub const SYSTEM_NAME: &'static str = "ReservationSystem";

    /// Check a single reservation slot for coherence.
    ///
    /// A reservation is valid if no store to `slot.address` has occurred
    /// since `slot.timestamp`.  The caller provides `current_epoch` (a
    /// global store counter); if `current_epoch > slot.timestamp`, the
    /// reservation is stale and a conflict is reported.
    ///
    /// Returns `CoherenceResult::Clean` when the reservation is still valid.
    pub fn check_reservation(slot: &ReservationSlot, current_epoch: u64) -> CoherenceResult {
        if current_epoch > slot.timestamp {
            CoherenceResult::Conflict {
                conflicting_addresses: vec![slot.address],
            }
        } else {
            CoherenceResult::Clean
        }
    }

    /// Batch check a slice of reservation slots.
    ///
    /// Every slot whose `timestamp < current_epoch` is collected into the
    /// conflict list.  Returns `CoherenceResult::Clean` when every slot is
    /// still valid.
    pub fn check_all(slots: &[ReservationSlot], current_epoch: u64) -> CoherenceResult {
        let conflicting: Vec<u64> = slots
            .iter()
            .filter(|s| current_epoch > s.timestamp)
            .map(|s| s.address)
            .collect();

        if conflicting.is_empty() {
            CoherenceResult::Clean
        } else {
            CoherenceResult::Conflict {
                conflicting_addresses: conflicting,
            }
        }
    }

    /// Compute the epoch threshold that a context must beat to acquire a
    /// reservation on `address`.  If another context already holds a
    /// reservation at the same address, returns `Err` with that slot.
    pub fn acquire_slot(
        slots: &mut [ReservationSlot],
        address: u64,
        timestamp: u64,
        ctx: ExecutionContextId,
    ) -> Result<(), ReservationSlot> {
        // If an existing slot covers the same address, the caller must
        // first release it or wait for timeout.
        if let Some(existing) = slots.iter().find(|s| s.address == address) {
            return Err(existing.clone());
        }

        // Find an empty slot or evict the oldest.
        if let Some(free) = slots.iter_mut().find(|s| s.timestamp == 0) {
            *free = ReservationSlot::new(address, timestamp, ctx);
            Ok(())
        } else {
            // Overwrite the oldest reservation.
            let oldest_idx = slots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.timestamp)
                .map(|(i, _)| i)
                .unwrap_or(0);
            slots[oldest_idx] = ReservationSlot::new(address, timestamp, ctx);
            Ok(())
        }
    }

    /// Release the reservation at `address` held by `ctx`.
    ///
    /// Returns `true` if a matching slot was released, `false` if no
    /// reservation for this combination was found.
    pub fn release_slot(
        slots: &mut [ReservationSlot],
        address: u64,
        ctx: ExecutionContextId,
    ) -> bool {
        if let Some(slot) = slots
            .iter_mut()
            .find(|s| s.address == address && s.acquired_by == ctx)
        {
            slot.timestamp = 0;
            slot.address = 0;
            true
        } else {
            false
        }
    }
}

impl SystemSpec for ReservationSystem {
    type Reads = ();
    type Writes = ();
    type ReadsResources = ();
    type WritesResources = ();

    fn system_id() -> SystemId {
        Self::SYSTEM_ID
    }

    fn system_name() -> &'static str {
        Self::SYSTEM_NAME
    }

    /// Coherence checks run before any compute stage so that conflicting
    /// stores are detected before layer execution.
    fn stage() -> Stage {
        Stage::Intake
    }

    fn order() -> u16 {
        0
    }
}

// ---------------------------------------------------------------------------
// SchedulableComponent / SchedulableResource implementations
// ---------------------------------------------------------------------------

impl SchedulableComponent for MemoryDomain {
    const COMPONENT_ID: ComponentId = 100;
    const NAME: &'static str = "MemoryDomain";
}

impl SchedulableComponent for PageTable {
    const COMPONENT_ID: ComponentId = 101;
    const NAME: &'static str = "PageTable";
}

impl SchedulableComponent for ReservationSlot {
    const COMPONENT_ID: ComponentId = 102;
    const NAME: &'static str = "ReservationSlot";
}

impl SchedulableResource for ReservationSystem {
    const RESOURCE_ID: crate::scheduling::component_id::ResourceId = 80;
    const NAME: &'static str = "ReservationSystem";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_domain_basics() {
        let domain = MemoryDomain::new("xdr_main", 0x0000_0000, 0x1000_0000, MEM_CLASS_MAIN, 4096);
        assert_eq!(domain.name, "xdr_main");
        assert_eq!(domain.base_addr, 0x0000_0000);
        assert_eq!(domain.size_bytes, 0x1000_0000);
        assert_eq!(domain.page_count(), 0x10000);
        assert!(domain.contains(0x0000_0000));
        assert!(domain.contains(0x0FFF_FFFF));
        assert!(!domain.contains(0x1000_0000));
        assert_eq!(domain.to_phys(0), Some(0x0000_0000));
        assert_eq!(domain.to_phys(0x100), Some(0x0000_0100));
        assert_eq!(domain.to_phys(0x1000_0000), None);
    }

    #[test]
    fn page_table_map_unmap() {
        let mut pt = PageTable::new(1024, 40);
        pt.map(
            42,
            0x8000_0000,
            MemoryDomainId(1),
            PAGE_READABLE | PAGE_WRITABLE,
        );
        let e = pt.entry(42).unwrap();
        assert!(e.is_present());
        assert!(e.is_readable());
        assert!(e.is_writable());
        assert!(!e.is_executable());

        let unmapped = pt.unmap(42);
        assert_eq!(unmapped.phys_addr, 0x8000_0000);
        assert!(!pt.entry(42).unwrap().is_present());
    }

    #[test]
    fn reservation_slot_acquire_release() {
        let ctx_a = ExecutionContextId(1);
        let ctx_b = ExecutionContextId(2);
        let mut slots = vec![ReservationSlot::new(0, 0, ExecutionContextId(0)); 4];

        // Acquire a slot for ctx_a at address 0x1000.
        assert!(ReservationSystem::acquire_slot(&mut slots, 0x1000, 42, ctx_a).is_ok());
        assert_eq!(slots[0].address, 0x1000);
        assert_eq!(slots[0].timestamp, 42);
        assert_eq!(slots[0].acquired_by, ctx_a);

        // Acquire at the same address by ctx_b should fail.
        assert!(ReservationSystem::acquire_slot(&mut slots, 0x1000, 43, ctx_b).is_err());

        // Release by non-owner should return false.
        assert!(!ReservationSystem::release_slot(&mut slots, 0x1000, ctx_b));

        // Release by owner should succeed.
        assert!(ReservationSystem::release_slot(&mut slots, 0x1000, ctx_a));
        assert_eq!(slots[0].timestamp, 0);
    }

    #[test]
    fn reservation_coherence_check() {
        let slots = vec![
            ReservationSlot::new(0x1000, 50, ExecutionContextId(1)),
            ReservationSlot::new(0x2000, 60, ExecutionContextId(2)),
        ];

        // Both slots are before epoch 100 — conflict.
        let result = ReservationSystem::check_all(&slots, 100);
        assert!(result.is_conflict());
        match result {
            CoherenceResult::Conflict {
                conflicting_addresses,
            } => {
                assert_eq!(conflicting_addresses.len(), 2);
            }
            _ => panic!("expected conflict"),
        }

        // Epoch 55 invalidates only slot 0.
        let result = ReservationSystem::check_all(&slots, 55);
        assert!(result.is_conflict());
        match result {
            CoherenceResult::Conflict {
                conflicting_addresses,
            } => {
                assert_eq!(conflicting_addresses, vec![0x1000]);
            }
            _ => panic!("expected conflict"),
        }

        // Epoch 50 — at the boundary, slot 0 timestamp == epoch, so it's valid.
        let result = ReservationSystem::check_all(&slots, 50);
        assert!(!result.is_conflict());

        // Epoch 40 — both valid.
        let result = ReservationSystem::check_all(&slots, 40);
        assert!(!result.is_conflict());
    }
}
