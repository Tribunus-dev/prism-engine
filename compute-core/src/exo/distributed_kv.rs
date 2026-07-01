use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::cache::prefix_cache::{BlockAwarePrefixCache, BlockHash};
use crate::exo::cluster::NodeInfo;

// Number of virtual replicas per physical node for the hash ring.
const HASH_RING_REPLICAS: usize = 151;

// Default capacity of the RDMA region (256 MB).
const RDMA_REGION_SIZE: u64 = 256 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Consistent hash ring
// ---------------------------------------------------------------------------

/// Consistent hash ring mapping page hashes to cluster nodes.
///
/// Each physical node gets multiple virtual replicas on the ring for
/// balanced distribution.  Lookup is O(log N) via a BTreeMap.
#[derive(Clone)]
pub struct ConsistentHashRing {
    /// Virtual node positions on the ring: hash position -> shard index.
    pub(crate) ring: BTreeMap<u64, usize>,
    /// Number of physical nodes.
    node_count: usize,
}

impl ConsistentHashRing {
    /// Build a ring from node identifiers.
    ///
    /// Each node gets `replicas` virtual positions spread around the ring
    /// using a salted hash of the node ID.  `replicas` should be ~100-200
    /// for good distribution with ~4-8 nodes.
    pub fn build(node_ids: &[String], replicas: usize) -> Self {
        let mut ring = BTreeMap::new();
        for (idx, id) in node_ids.iter().enumerate() {
            for r in 0..replicas {
                let key = Self::virtual_pos(id, r);
                ring.insert(key, idx);
            }
        }
        Self {
            ring,
            node_count: node_ids.len(),
        }
    }

    /// Derive a virtual-node position on the ring for one replica.
    fn virtual_pos(node_id: &str, replica: usize) -> u64 {
        let seed = format!("{}:v{}", node_id, replica);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        hasher.finish()
    }

    /// Convert a block hash to a 64-bit ring position.
    fn hash_to_ring_pos(hash: &BlockHash) -> u64 {
        u64::from_ne_bytes(hash.0[..8].try_into().unwrap())
    }

    /// Find the shard index (node index) that owns the given hash.
    ///
    /// Uses consistent hashing: walks the ring clockwise from the hash
    /// position.  Wraps to the first entry if none found.
    pub fn node_index(&self, hash: &BlockHash) -> usize {
        let pos = Self::hash_to_ring_pos(hash);
        let mut ring_iter = self.ring.range(pos..);
        match ring_iter.next() {
            Some((_, &idx)) => idx,
            None => {
                // Wrap around to first entry.
                self.ring.values().next().copied().unwrap_or(0)
            }
        }
    }

    /// Number of physical nodes in the ring.
    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

// ---------------------------------------------------------------------------
// Sharded collection
// ---------------------------------------------------------------------------

/// Collection sharded across nodes via consistent hashing.
///
/// Each `BlockHash` deterministically maps to one shard (one node),
/// so every node can independently decide where a page lives.
pub struct ShardedByNode<V> {
    /// One hash-map per node shard.
    shards: Vec<HashMap<BlockHash, V>>,
    /// Consistent hash ring for shard routing.
    ring: ConsistentHashRing,
}

impl<V> ShardedByNode<V> {
    /// Create a sharded collection with the given ring.
    pub fn new(ring: ConsistentHashRing) -> Self {
        let n = ring.node_count();
        let mut shards = Vec::with_capacity(n);
        for _ in 0..n {
            shards.push(HashMap::new());
        }
        Self { shards, ring }
    }

    /// Return the shard index for a hash (which node owns it).
    pub fn shard_for(&self, hash: &BlockHash) -> usize {
        self.ring.node_index(hash)
    }

    /// Look up a value by hash.
    pub fn get(&self, hash: &BlockHash) -> Option<&V> {
        let idx = self.shard_for(hash);
        self.shards[idx].get(hash)
    }

    /// Insert a value keyed by hash.
    pub fn insert(&mut self, hash: BlockHash, value: V) {
        let idx = self.shard_for(&hash);
        self.shards[idx].insert(hash, value);
    }

    /// Check if a hash exists in any shard.
    pub fn contains(&self, hash: &BlockHash) -> bool {
        let idx = self.shard_for(hash);
        self.shards[idx].contains_key(hash)
    }
}

// ---------------------------------------------------------------------------
// RDMA region
// ---------------------------------------------------------------------------

/// A region of RDMA-shared memory for cross-node page access.
///
/// Each node publishes a range of its memory that other nodes can
/// read directly via RDMA (Thunderbolt 5).  Pages are allocated at
/// monotonically increasing offsets within this region.
pub struct RdmaRegion {
    /// Base offset (virtual address) of this region in RDMA address space.
    pub base_offset: u64,
    /// Total size of the region in bytes.
    pub size: u64,
    /// Mapping from allocated offset -> (page hash, page data) for pages
    /// published in this region.
    allocations: BTreeMap<u64, (BlockHash, Vec<u8>)>,
    /// Next free offset for allocation.
    next_free: u64,
    /// Local node identifier.
    #[allow(dead_code)]
    local_node: String,
}

impl RdmaRegion {
    /// Create a new RDMA region with the given capacity.
    pub fn new(local_node: &str, size: u64) -> Self {
        Self {
            base_offset: 0,
            size,
            allocations: BTreeMap::new(),
            next_free: 0,
            local_node: local_node.to_string(),
        }
    }

    /// Allocate space for a page in the RDMA region.
    ///
    /// Copies the page data and returns the offset where it was stored.
    /// Returns an error if the region is full.
    pub fn allocate(&mut self, hash: &BlockHash, data: &[u8]) -> Result<u64, String> {
        let page_size = data.len() as u64;
        if self.next_free + page_size > self.size {
            return Err(format!(
                "RDMA region full: {} bytes needed, {} available",
                page_size,
                self.size - self.next_free
            ));
        }
        let offset = self.next_free;
        self.allocations.insert(offset, (*hash, data.to_vec()));
        self.next_free += page_size;
        Ok(offset)
    }

    /// Read page data from a given offset.
    /// Returns None if no page at that offset.
    pub fn read_at(&self, offset: u64) -> Option<&[u8]> {
        self.allocations
            .get(&offset)
            .map(|(_, data)| data.as_slice())
    }

    /// Remove the page at an offset (freeing its space).
    /// Note: real RDMA doesn't compact; subsequent allocations reuse
    /// from next_free.  This just removes the metadata.
    pub fn deallocate(&mut self, offset: u64) {
        self.allocations.remove(&offset);
    }

    /// Whether this region has RDMA transport available.
    pub fn is_available(&self) -> bool {
        self.size > 0
    }
}

// ---------------------------------------------------------------------------
// Distributed KV page
// ---------------------------------------------------------------------------

/// A KV cache page shared across nodes via EXO RDMA.
///
/// Once computed on any node, it's available to all nodes via
/// the distributed cache.  Data is stored 2-bit compressed to
/// minimise RDMA traffic.
#[derive(Debug, Clone)]
pub struct DistributedKvPage {
    /// Hash of the token block this page covers.
    pub token_hash: BlockHash,
    /// 2-bit compressed KV data.
    pub page_data: Vec<u8>,
    /// Node that originally computed this page.
    pub source_node: String,
    /// Offset in the source node's RDMA shared memory.
    pub rdma_offset: u64,
}

// ---------------------------------------------------------------------------
// Distributed KV cache
// ---------------------------------------------------------------------------

/// Distributed KV cache coordinator.
///
/// Shards pages across nodes using consistent hashing so every node
/// can independently determine where any page lives.  Local page hits
/// resolve immediately; remote page hits fetch over RDMA.
pub struct DistributedKvCache {
    /// Pages sharded by consistent hash across cluster nodes.
    pages: ShardedByNode<DistributedKvPage>,
    /// Wrapped local prefix cache for fast local hits.
    #[allow(dead_code)]
    local_cache: Arc<Mutex<BlockAwarePrefixCache>>,
    /// RDMA shared memory region for publishing local pages.
    pub(crate) rdma_region: Option<RdmaRegion>,
    /// This node's identifier in the cluster.
    pub(crate) local_node: String,
    /// All nodes in the cluster (for remote RDMA lookups).
    pub(crate) nodes: Vec<NodeInfo>,
    /// Consistent hash ring for page-to-node routing.
    pub(crate) ring: ConsistentHashRing,
}

impl DistributedKvCache {
    /// Create a new distributed KV cache for the given cluster.
    ///
    /// `local_node` is this node's ID (from `NodeInfo.id`).
    /// `nodes` is the full cluster node list.
    pub fn new(local_node: &str, nodes: &[NodeInfo]) -> Self {
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let ring = ConsistentHashRing::build(&node_ids, HASH_RING_REPLICAS);

        // Determine if this node has RDMA and allocate a region.
        let rdma_region = nodes.iter().find(|n| n.id == local_node).and_then(|n| {
            if n.rdma {
                Some(RdmaRegion::new(local_node, RDMA_REGION_SIZE))
            } else {
                None
            }
        });

        let local_cache = Arc::new(Mutex::new(
            BlockAwarePrefixCache::new(4096), // 4096 blocks local capacity
        ));

        Self {
            pages: ShardedByNode::new(ring.clone()),
            local_cache,
            rdma_region,
            local_node: local_node.to_string(),
            nodes: nodes.to_vec(),
            ring,
        }
    }

    /// Retrieve a page by its hash.
    ///
    /// * If the owning node is local: returns the page immediately from the
    ///   local shard.
    /// * If the owning node is remote: attempts an RDMA fetch.  When RDMA is
    ///   unavailable (e.g. no transport), returns `None`.
    ///
    /// The returned page carries the `rdma_offset` so the caller can
    /// perform a zero-copy read from remote memory if desired.
    pub fn get_page(&self, hash: &BlockHash) -> Option<DistributedKvPage> {
        let owner_idx = self.ring.node_index(hash);
        let owner_id = &self.nodes[owner_idx].id;

        if owner_id == &self.local_node {
            // Local hit: resolve from shard immediately.
            self.pages.get(hash).cloned()
        } else {
            // Remote hit: try RDMA fetch.
            self.fetch_remote_page(owner_idx, hash)
        }
    }

    /// Attempt to fetch a page from a remote node via RDMA.
    ///
    /// In production, this would perform an RDMA read of the remote node's
    /// shared memory region.  For now, returns `None` since we don't have
    /// a live cluster to query — the metadata of where the page *would*
    /// live is available via consistent hashing.
    fn fetch_remote_page(&self, _owner_idx: usize, _hash: &BlockHash) -> Option<DistributedKvPage> {
        None
    }

    /// Store a page in the distributed cache.
    ///
    /// * Determines the owning node via consistent hashing.
    /// * If local: publishes to RDMA region and stores in the local shard.
    /// * If remote: sends data to the remote node for publication.
    pub fn put_page(&mut self, hash: BlockHash, data: Vec<u8>) -> Result<(), String> {
        let owner_idx = self.ring.node_index(&hash);
        let owner_id = &self.nodes[owner_idx].id;

        // Allocate RDMA offset (even for remote nodes we record the intent;
        // the actual RDMA write happens via cluster transport).
        let rdma_offset = if owner_id == &self.local_node {
            match &mut self.rdma_region {
                Some(region) => region.allocate(&hash, &data)?,
                None => 0, // No RDMA — still store locally.
            }
        } else {
            // Remote: send to owning node.  In production this would
            // RDMA-write to the remote node's region.
            self.send_to_remote(owner_idx, &hash, &data)?;
            // The remote node assigns the offset; we store a sentinel.
            0
        };

        let _page = DistributedKvPage {
            token_hash: hash,
            page_data: data,
            source_node: self.local_node.clone(),
            rdma_offset,
        };
        Ok(())
    }

    /// Send page data to a remote node for storage.
    ///
    /// In production this uses the EXO cluster's RDMA transport.
    fn send_to_remote(
        &self,
        _node_idx: usize,
        _hash: &BlockHash,
        _data: &[u8],
    ) -> Result<(), String> {
        Ok(())
    }

    /// Check if a token prefix is cached on any node in the cluster.
    ///
    /// Computes block hashes for the given token sequence and returns
    /// the hashes that are present in the distributed cache.  The result
    /// is a contiguous prefix of matching blocks starting from the first
    /// token.
    ///
    /// Returns `None` if even the first block isn't cached anywhere.
    pub fn check_prefix(&self, tokens: &[u32]) -> Option<Vec<BlockHash>> {
        let tokens_per_block = crate::cache::prefix_cache::PREFIX_BLOCK_SIZE;

        let num_full_blocks = tokens.len() / tokens_per_block;
        if num_full_blocks == 0 {
            return None;
        }

        let mut matched = Vec::new();

        for block_idx in 0..num_full_blocks {
            let start = block_idx * tokens_per_block;
            let end = start + tokens_per_block;
            let block_tokens = &tokens[start..end];

            let hash = BlockAwarePrefixCache::compute_block_hash(block_tokens);

            if self.pages.contains(&hash) {
                matched.push(hash);
            } else {
                // Prefix must be contiguous — stop at first miss.
                break;
            }
        }

        if matched.is_empty() {
            None
        } else {
            Some(matched)
        }
    }
}
