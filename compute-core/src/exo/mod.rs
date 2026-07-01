//! EXO distributed inference integration.
//!
//! Each Tribunus node joins an EXO cluster as an inference backend.
//! EXO handles model sharding, device discovery, and RDMA transport
//! (Thunderbolt 5).  Each node provides its local Tribunus runtime
//! as the inference backend — the full pipeline (IOSurface, ANE,
//! TurboQuant, memory plan) runs locally on each node's layer shard.
//!
//! # Architecture
//!
//! ```text
//! EXO Orchestrator
//!   ├─ Node 0: Tribunus (layers 0..11) ← Mac Studio M3 Ultra
//!   ├─ Node 1: Tribunus (layers 12..23) ← Mac mini M4 Pro
//!   ├─ Node 2: Tribunus (layers 24..35) ← Mac Studio M1 Max
//!   └─ RDMA Thunderbolt 5 mesh between nodes
//! ```
//!
//! The communication pattern is pipeline-parallel: while Node N
//! processes its layer range, Node N+1 is already receiving the
//! previous hidden state over RDMA.

pub mod autoscaler;
pub mod cluster;
pub mod distributed_kv;
pub mod hardware;

pub use autoscaler::*;
pub use cluster::*;
pub use distributed_kv::*;
pub use hardware::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::prefix_cache::{BlockAwarePrefixCache, PREFIX_BLOCK_SIZE};
    use crate::gpu_memory;

    #[test]
    fn test_detect_chip() {
        let chip = crate::exo::hardware::detect_chip();
        assert!(!chip.is_empty(), "chip name should not be empty");
        eprintln!("detected chip: {}", chip);
    }

    #[test]
    fn test_format_chip_name() {
        assert_eq!(format_chip_name("Apple M3 Ultra"), "Mac Studio M3 Ultra");
        assert_eq!(format_chip_name("Apple M1 Max"), "MacBook Pro M1 Max");
        assert_eq!(format_chip_name("Apple M2"), "Mac M2");
        assert_eq!(format_chip_name("unknown chip"), "unknown chip");
    }

    #[test]
    fn test_detect_ram() {
        let ram_mb = gpu_memory::total_physical_ram_mb();
        assert!(ram_mb > 0, "RAM should be > 0");
    }

    #[test]
    fn test_detected_rdma_is_bool() {
        // Just ensure it doesn't panic.
        let rdma = crate::exo::hardware::detect_rdma();
        eprintln!("RDMA detected: {}", rdma);
    }

    #[test]
    fn test_node_info_serialization() {
        let info = NodeInfo {
            id: "test-node".to_string(),
            address: "192.168.1.100:52415".to_string(),
            ram_gb: 64,
            model_layer_range: Some((0, 11)),
            rdma: true,
        };
        let json = serde_json::to_string_pretty(&info).unwrap();
        assert!(json.contains("test-node"));
        assert!(json.contains("52415"));
        eprintln!("NodeInfo JSON:\n{}", json);
    }

    #[test]
    fn test_cluster_info_serialization() {
        let info = ClusterInfo {
            nodes: vec![NodeInfo {
                id: "node0".to_string(),
                address: "192.168.1.100:52415".to_string(),
                ram_gb: 64,
                model_layer_range: Some((0, 11)),
                rdma: true,
            }],
            model: Some("gemma-4-27b".to_string()),
            model_shard: "node:64".to_string(),
            total_ram_gb: 64,
            cluster_ram_gb: 64,
        };
        let json = serde_json::to_string_pretty(&info).unwrap();
        assert!(json.contains("gemma-4-27b"));
        assert!(json.contains("cluster_ram_gb"));
        eprintln!("ClusterInfo JSON:\n{}", json);
    }

    #[test]
    fn test_hardware_detection() {
        let hw = detect_hardware();
        assert!(hw.ram_gb > 0, "RAM should be > 0 GB");
        assert!(!hw.chip.is_empty(), "chip should be non-empty");
        eprintln!(
            "Hardware: {} ({} GB, {} ANE cores, RDMA: {})",
            hw.chip, hw.ram_gb, hw.ane_cores, hw.rdma_available
        );
    }

    #[test]
    fn test_local_node_info() {
        let cache = crate::model_cache::ModelCache::new(2048);
        let node = ExoNode {
            model_cache: std::sync::Arc::new(std::sync::Mutex::new(cache)),
            exo_process: None,
            listen_addr: "0.0.0.0:52415".to_string(),
            rdma_enabled: true,
            chip: "Mac Studio M3 Ultra".to_string(),
            ram_gb: 512,
            autoscaler: None,
        };
        let info = node.local_node_info();
        assert!(info.id.contains("unknown") || !info.id.is_empty());
        assert_eq!(info.ram_gb, 512);
        assert!(info.rdma);
    }

    // ── Distributed KV cache tests ────────────────────────────────────

    #[test]
    fn test_consistent_hash_ring_build() {
        let ids = vec!["node0".into(), "node1".into(), "node2".into()];
        let ring = ConsistentHashRing::build(&ids, 151);
        assert_eq!(ring.node_count(), 3);
        // At least one virtual entry per node should be present.
        assert!(ring.ring.len() >= 3);
        assert_eq!(ring.ring.len(), 453); // 151 * 3
    }

    #[test]
    fn test_consistent_hash_ring_distribution() {
        let ids = vec![
            "node0".into(),
            "node1".into(),
            "node2".into(),
            "node3".into(),
        ];
        let ring = ConsistentHashRing::build(&ids, 151);

        let mut assignments = vec![0usize; 4];
        for i in 0..4096 {
            let tokens = vec![i as u32];
            let hash = BlockAwarePrefixCache::compute_block_hash(&tokens);
            let idx = ring.node_index(&hash);
            assignments[idx] += 1;
        }

        // Each node should get roughly 1024 items (4096/4).
        // Allow 30% deviation for the ring's pseudo-random distribution.
        for &count in &assignments {
            assert!(count > 600, "Node assigned too few items: {}", count);
            assert!(count < 1500, "Node assigned too many items: {}", count);
        }
        eprintln!("Hash distribution: {:?}", assignments);
    }

    #[test]
    fn test_sharded_by_node_insert_get() {
        let ids = vec!["node0".into(), "node1".into()];
        let ring = ConsistentHashRing::build(&ids, 151);
        let mut sharded = ShardedByNode::<DistributedKvPage>::new(ring);

        let tokens = vec![42u32, 99, 7, 13];
        let hash = BlockAwarePrefixCache::compute_block_hash(&tokens);

        let page = DistributedKvPage {
            token_hash: hash,
            page_data: vec![1, 2, 3, 4],
            source_node: "node0".into(),
            rdma_offset: 0,
        };

        sharded.insert(hash, page);

        // Re-compute hash and verify retrieval.
        let lookup_hash = BlockAwarePrefixCache::compute_block_hash(&tokens);
        let retrieved = sharded.get(&lookup_hash);
        assert!(retrieved.is_some(), "Should retrieve the page we inserted");
        assert_eq!(retrieved.unwrap().page_data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sharded_by_node_contains() {
        let ids = vec!["a".into(), "b".into()];
        let ring = ConsistentHashRing::build(&ids, 151);
        let mut sharded = ShardedByNode::<DistributedKvPage>::new(ring);

        let h1 = BlockAwarePrefixCache::compute_block_hash(&[1, 2, 3]);
        let h2 = BlockAwarePrefixCache::compute_block_hash(&[4, 5, 6]);

        sharded.insert(
            h1,
            DistributedKvPage {
                token_hash: h1,
                page_data: vec![10],
                source_node: "a".into(),
                rdma_offset: 0,
            },
        );

        assert!(sharded.contains(&h1), "h1 should be present");
        assert!(!sharded.contains(&h2), "h2 should not be present");
    }

    #[test]
    fn test_rdma_region_allocate_read() {
        let mut region = RdmaRegion::new("node0", 1024 * 1024);
        let hash = BlockAwarePrefixCache::compute_block_hash(&[1, 2, 3]);

        let data = vec![0xAB; 256];
        let offset = region.allocate(&hash, &data).unwrap();
        assert_eq!(offset, 0, "First allocation should be at offset 0");

        // Read back.
        let read = region.read_at(offset);
        assert!(read.is_some());
        assert_eq!(read.unwrap(), &data[..]);
    }

    #[test]
    fn test_rdma_region_full() {
        let mut region = RdmaRegion::new("node0", 100);
        let hash = BlockAwarePrefixCache::compute_block_hash(&[1]);
        let data = vec![0u8; 200];
        let result = region.allocate(&hash, &data);
        assert!(
            result.is_err(),
            "Allocation exceeding region size should fail"
        );
        assert!(result.unwrap_err().contains("full"));
    }

    #[test]
    fn test_distributed_kv_cache_new() {
        let nodes = vec![
            NodeInfo {
                id: "node0".into(),
                address: "192.168.1.1:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: true,
            },
            NodeInfo {
                id: "node1".into(),
                address: "192.168.1.2:52415".into(),
                ram_gb: 32,
                model_layer_range: None,
                rdma: true,
            },
        ];
        let cache = DistributedKvCache::new("node0", &nodes);

        // Should have an RDMA region since node0 has rdma: true.
        assert!(cache.rdma_region.is_some());
        assert_eq!(cache.local_node, "node0");
        assert_eq!(cache.nodes.len(), 2);
    }

    #[test]
    fn test_distributed_kv_cache_put_get_local() {
        let nodes = vec![
            NodeInfo {
                id: "node0".into(),
                address: "10.0.0.1:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: true,
            },
            NodeInfo {
                id: "node1".into(),
                address: "10.0.0.2:52415".into(),
                ram_gb: 32,
                model_layer_range: None,
                rdma: true,
            },
        ];
        let mut cache = DistributedKvCache::new("node0", &nodes);

        let tokens = vec![10u32, 20, 30, 40];
        let hash = BlockAwarePrefixCache::compute_block_hash(&tokens);

        // Determine if this hash lands on node0 or node1.
        let owner_idx = cache.ring.node_index(&hash);
        let owner_id = cache.nodes[owner_idx].id.clone();
        eprintln!("Hash {:02x}... owned by {}", hash.0[0], owner_id);

        let page_data = vec![0xAA; 128];
        cache.put_page(hash, page_data.clone()).unwrap();

        let lookup_hash = BlockAwarePrefixCache::compute_block_hash(&tokens);
        let retrieved = cache.get_page(&lookup_hash);

        if owner_id == "node0" {
            // Local page: should be immediately available.
            assert!(retrieved.is_some(), "Local page should be retrievable");
            if let Some(p) = retrieved {
                assert_eq!(p.page_data, page_data);
                assert_eq!(p.source_node, "node0");
            }
        } else {
            // Remote page: get_page returns None (no live RDMA cluster).
            assert!(
                retrieved.is_none(),
                "Remote page should return None without RDMA cluster"
            );
        }
    }

    #[test]
    fn test_distributed_kv_cache_check_prefix() {
        let nodes = vec![
            NodeInfo {
                id: "n0".into(),
                address: "10.0.0.1:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: false,
            },
            NodeInfo {
                id: "n1".into(),
                address: "10.0.0.2:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: false,
            },
        ];
        let mut cache = DistributedKvCache::new("n0", &nodes);

        // Build a token sequence long enough for multiple blocks.
        let block_size = PREFIX_BLOCK_SIZE;
        let tokens: Vec<u32> = (0..(block_size * 3) as u32).collect();

        // Insert first block.
        let block0_tokens: Vec<u32> = tokens[0..block_size].to_vec();
        let h0 = BlockAwarePrefixCache::compute_block_hash(&block0_tokens);
        cache.put_page(h0, vec![0u8; 64]).unwrap();

        // Insert second block.
        let block1_tokens: Vec<u32> = tokens[block_size..block_size * 2].to_vec();
        let h1 = BlockAwarePrefixCache::compute_block_hash(&block1_tokens);
        cache.put_page(h1, vec![1u8; 64]).unwrap();

        // check_prefix with all 3 blocks: should match first 2 (contiguous prefix).
        let result = cache.check_prefix(&tokens);
        assert!(result.is_some(), "Should match at least one block");
        let matched = result.unwrap();
        assert_eq!(matched.len(), 2, "Should match first 2 contiguous blocks");
        assert_eq!(matched[0], h0);
        assert_eq!(matched[1], h1);
    }

    #[test]
    fn test_distributed_kv_cache_check_prefix_no_match() {
        let nodes = vec![NodeInfo {
            id: "n0".into(),
            address: "10.0.0.1:52415".into(),
            ram_gb: 64,
            model_layer_range: None,
            rdma: false,
        }];
        let cache = DistributedKvCache::new("n0", &nodes);

        // No pages inserted — prefix check should return None.
        let tokens: Vec<u32> = (0..64).collect();
        let result = cache.check_prefix(&tokens);
        assert!(result.is_none(), "Empty cache should return None");
    }

    #[test]
    fn test_distributed_kv_cache_check_prefix_too_short() {
        let nodes = vec![NodeInfo {
            id: "n0".into(),
            address: "10.0.0.1:52415".into(),
            ram_gb: 64,
            model_layer_range: None,
            rdma: false,
        }];
        let cache = DistributedKvCache::new("n0", &nodes);

        // Fewer tokens than PREFIX_BLOCK_SIZE: should return None.
        let tokens = vec![1u32, 2, 3];
        let result = cache.check_prefix(&tokens);
        assert!(result.is_none(), "Too few tokens should return None");
    }
}
