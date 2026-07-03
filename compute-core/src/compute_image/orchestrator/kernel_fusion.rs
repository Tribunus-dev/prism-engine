//! Graph-driven kernel fusion analyzer.
//!
//! Analyzes the execution graph to identify adjacent decoder-layer pairs
//! that can be merged into a single Metal kernel dispatch. Fusion reduces
//! dispatch overhead and improves GPU utilization on memory-bandwidth-bound
//! decode steps.
//!
//! # Fusion Rules
//!
//! - Only decoder layers with matching `attention_kind` **and** `head_dim` fuse.
//! - Compaction epoch boundaries prevent fusion (the epoch triggers between
//!   layers, so the kernel boundary is required for correct KV cache state).
//! - Maximum fusion depth: 4 layers (constrained by Metal kernel register
//!   pressure limits on Apple Silicon).
//! - SWA (Sliding Window Attention) and Full attention never fuse together.

use crate::compute_image::compile::execution_graph::{
    ExecutionGraphDescriptor, NodeKind,
};

/// A fused kernel group: N consecutive layers that can be merged into
/// a single Metal kernel dispatch.
#[derive(Clone, Debug)]
pub struct FusedLayerGroup {
    /// Index of the first layer in this group.
    pub start_layer: usize,
    /// Number of fused layers.
    pub count: u32,
    /// Attention kind: all layers in this group share the same kind.
    pub attention_kind: u8,
    /// Head dimension: all layers in this group share the same dim.
    pub head_dim: u16,
}

/// Analyze an execution graph and produce fused kernel dispatch groups.
///
/// Fusion rules:
/// - Adjacent decoder layers with same `attention_kind` and `head_dim` fuse.
/// - Compaction epoch boundaries prevent fusion (epoch must trigger between
///   layers).
/// - Maximum fusion depth: 4 layers (kernel register pressure limit).
/// - SWA and Full attention never fuse together.
pub fn analyze_graph(graph: &ExecutionGraphDescriptor) -> Vec<FusedLayerGroup> {
    let mut groups = Vec::new();
    let mut i = 0;

    while i < graph.layers.len() {
        let current = &graph.layers[i];

        // Only fuse decoder layers.
        if current.node_kind != NodeKind::DecoderLayer as u8 {
            // Non-decoder node: single-element group.
            groups.push(FusedLayerGroup {
                start_layer: i,
                count: 1,
                attention_kind: current.attention_kind,
                head_dim: current.head_dim,
            });
            i += 1;
            continue;
        }

        // If this layer has a compaction epoch, it must stand alone —
        // compaction triggers after this layer, so no fusion across it.
        if current.compaction_epoch != 0xFF {
            groups.push(FusedLayerGroup {
                start_layer: i,
                count: 1,
                attention_kind: current.attention_kind,
                head_dim: current.head_dim,
            });
            i += 1;
            continue;
        }

        // Try to fuse consecutive same-kind decoder layers.
        let mut count = 1u32;
        while i + (count as usize) < graph.layers.len() && count < 4 {
            let next = &graph.layers[i + (count as usize)];

            // Stop if next node is not a decoder layer.
            if next.node_kind != NodeKind::DecoderLayer as u8 {
                break;
            }
            // Stop if attention kind changes (SWA vs Full).
            if next.attention_kind != current.attention_kind {
                break;
            }
            // Stop if head dimension changes.
            if next.head_dim != current.head_dim {
                break;
            }
            // Stop if a compaction epoch triggers between these layers
            // (0xFF = no epoch; any other value means a compaction boundary).
            if next.compaction_epoch != 0xFF {
                break;
            }

            count += 1;
        }

        groups.push(FusedLayerGroup {
            start_layer: i,
            count,
            attention_kind: current.attention_kind,
            head_dim: current.head_dim,
        });
        i += count as usize;
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::compile::execution_graph::{
        AttentionKind, LayerExecutionNode,
    };

    fn make_decoder(attention_kind: u8, head_dim: u16, compaction_epoch: u8) -> LayerExecutionNode {
        LayerExecutionNode {
            node_kind: NodeKind::DecoderLayer as u8,
            attention_kind,
            device_capability: 1, // Gpu
            compaction_epoch,
            layer_index: 0,
            head_dim,
            num_heads: 16,
            hidden_dim: 3840,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0; 8],
        }
    }

    fn make_non_decoder() -> LayerExecutionNode {
        LayerExecutionNode {
            node_kind: NodeKind::VisionPatchEmbed as u8,
            attention_kind: 0,
            device_capability: 1,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: 256,
            num_heads: 16,
            hidden_dim: 3840,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0; 8],
        }
    }

    fn build_graph(layers: Vec<LayerExecutionNode>) -> ExecutionGraphDescriptor {
        let num_layers = layers.len() as u16;
        ExecutionGraphDescriptor {
            magic: *b"PRMEXEC1",
            version: 1,
            num_layers,
            num_draft_layers: 0,
            num_compaction_epochs: 0,
            node_count: num_layers as u32,
            _pad: [0; 2],
            layers,
            compaction_epochs: vec![],
            draft_sub_graph: None,
        }
    }

    #[test]
    fn empty_graph() {
        let graph = build_graph(vec![]);
        let groups = analyze_graph(&graph);
        assert!(groups.is_empty());
    }

    #[test]
    fn single_decoder_layer() {
        let graph = build_graph(vec![make_decoder(
            AttentionKind::FullAttention as u8,
            256,
            0xFF,
        )]);
        let groups = analyze_graph(&graph);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].start_layer, 0);
        assert_eq!(groups[0].count, 1);
        assert_eq!(groups[0].attention_kind, AttentionKind::FullAttention as u8);
        assert_eq!(groups[0].head_dim, 256);
    }

    #[test]
    fn fuses_max_four_layers() {
        let layers = (0..8)
            .map(|_| {
                make_decoder(AttentionKind::FullAttention as u8, 256, 0xFF)
            })
            .collect();
        let graph = build_graph(layers);
        let groups = analyze_graph(&graph);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].start_layer, 0);
        assert_eq!(groups[0].count, 4);
        assert_eq!(groups[1].start_layer, 4);
        assert_eq!(groups[1].count, 4);
    }

    #[test]
    fn compaction_epoch_breaks_fusion() {
        let layers = vec![
            make_decoder(AttentionKind::FullAttention as u8, 256, 0xFF),
            make_decoder(AttentionKind::FullAttention as u8, 256, 0),  // epoch index 0
            make_decoder(AttentionKind::FullAttention as u8, 256, 0xFF),
        ];
        let graph = build_graph(layers);
        let groups = analyze_graph(&graph);
        // Layer 0: alone (no fusion past epoch boundary)
        // Layer 1: alone (epoch boundary prevents backward fusion)
        // Layer 2: alone
        assert_eq!(groups.len(), 3);
        for g in &groups {
            assert_eq!(g.count, 1);
        }
    }

    #[test]
    fn different_head_dim_breaks_fusion() {
        let layers = vec![
            make_decoder(0, 256, 0xFF),
            make_decoder(0, 512, 0xFF), // different head_dim
            make_decoder(0, 256, 0xFF),
        ];
        let graph = build_graph(layers);
        let groups = analyze_graph(&graph);
        assert_eq!(groups.len(), 3);
        for g in &groups {
            assert_eq!(g.count, 1);
        }
    }

    #[test]
    fn different_attention_kind_breaks_fusion() {
        let layers = vec![
            make_decoder(AttentionKind::SlidingWindow as u8, 256, 0xFF),
            make_decoder(AttentionKind::FullAttention as u8, 256, 0xFF),
            make_decoder(AttentionKind::SlidingWindow as u8, 256, 0xFF),
        ];
        let graph = build_graph(layers);
        let groups = analyze_graph(&graph);
        assert_eq!(groups.len(), 3);
        for g in &groups {
            assert_eq!(g.count, 1);
        }
    }

    #[test]
    fn non_decoder_are_single_element() {
        let layers = vec![
            make_decoder(0, 256, 0xFF),
            make_non_decoder(),
            make_decoder(0, 256, 0xFF),
        ];
        let graph = build_graph(layers);
        let groups = analyze_graph(&graph);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].start_layer, 0);
        assert_eq!(groups[0].count, 1);
        assert_eq!(groups[1].start_layer, 1);
        assert_eq!(groups[1].count, 1);
        assert_eq!(groups[2].start_layer, 2);
        assert_eq!(groups[2].count, 1);
    }

    #[test]
    fn gemma4_swa_pattern() {
        // Gemma 4: SWA at layers [5,11,17,23,29,35,41,47], full at others.
        // Simulate layers 0-7: full, full, full, full, full, SWA, full, full.
        let layers = (0..8)
            .map(|i| {
                let kind = if i == 5 {
                    AttentionKind::SlidingWindow as u8
                } else {
                    AttentionKind::FullAttention as u8
                };
                make_decoder(kind, 256, 0xFF)
            })
            .collect();
        let graph = build_graph(layers);
        let groups = analyze_graph(&graph);

        // Expected groups:
        // [0-3] Full (4, max fusion depth)
        // [4]   Full (1, SWA at 5 blocks fusion)
        // [5]   SWA  (1)
        // [6-7] Full (2)
        assert_eq!(groups.len(), 4);
        assert_eq!(groups[0].start_layer, 0);
        assert_eq!(groups[0].count, 4);
        assert_eq!(groups[1].start_layer, 4);
        assert_eq!(groups[1].count, 1);
        assert_eq!(groups[2].start_layer, 5);
        assert_eq!(groups[2].count, 1);
        assert_eq!(groups[3].start_layer, 6);
        assert_eq!(groups[3].count, 2);

        assert_eq!(
            groups[2].attention_kind,
            AttentionKind::SlidingWindow as u8
        );
        for g in &[&groups[0], &groups[1], &groups[3]] {
            assert_eq!(g.attention_kind, AttentionKind::FullAttention as u8);
        }
    }
}
