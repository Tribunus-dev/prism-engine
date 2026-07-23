//! Native XDNA target planning.
//!
//! This is deliberately a Prism lowering, rather than an adapter to an
//! external compiler. It turns the spatial graph into the explicit tile,
//! FIFO, DMA, worker, barrier, and runtime-command form consumed by the
//! native XDNA runtime.

use crate::cost::CostEstimate;
use crate::graph::{ComputeKind, MemoryKind, SpatialGraph, SpatialNode};
use crate::legalize::{LegalizationError, LegalizedGraph};
use crate::target::{LoweringError, SpatialTarget, TargetCapabilities};
use crate::xdna::*;
use prism_ecs_ir::cimage_types::TensorShape;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XdnaTarget {
    pub topology: XdnaTopology,
    pub default_element_type: XdnaElementType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XdnaMatmulTile {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XdnaRowPartition {
    pub tile: crate::xdna::TileCoord,
    pub row_offset: usize,
    pub rows: usize,
    pub byte_offset: u64,
    pub bytes: u64,
}

impl XdnaTarget {
    fn supports_compute(kind: &ComputeKind) -> bool {
        matches!(
            kind,
            ComputeKind::MatMul
                | ComputeKind::Elementwise
                | ComputeKind::Normalization
                | ComputeKind::Softmax
                | ComputeKind::Attention
        )
    }

    fn validate_compute_shape(
        id: crate::graph::SpatialNodeId,
        kind: &ComputeKind,
        shape: &crate::graph::ShapeContract,
    ) -> Vec<LegalizationError> {
        let mut errors = Vec::new();
        if !Self::supports_compute(kind) {
            errors.push(LegalizationError::TargetConstraintViolation {
                detail: format!(
                    "XDNA native codegen supports MatMul, Elementwise, Normalization, Softmax, and Attention; node {} is {:?}",
                    id.0, kind
                ),
            });
            return errors;
        }
        match kind {
            ComputeKind::Elementwise => {
                if shape.in_shapes.is_empty() || shape.out_shapes.len() != 1 {
                    errors.push(LegalizationError::TargetConstraintViolation {
                        detail: format!(
                            "XDNA elementwise node {} needs inputs and one output",
                            id.0
                        ),
                    });
                } else if shape
                    .in_shapes
                    .iter()
                    .any(|input| input != &shape.out_shapes[0])
                {
                    errors.push(LegalizationError::TargetConstraintViolation {
                        detail: format!(
                            "XDNA elementwise node {} requires shape-preserving inputs",
                            id.0
                        ),
                    });
                }
            }
            ComputeKind::Normalization => {
                if shape.in_shapes.len() != 1
                    || shape.out_shapes.len() != 1
                    || shape.in_shapes[0] != shape.out_shapes[0]
                {
                    errors.push(LegalizationError::TargetConstraintViolation {
                        detail: format!(
                            "XDNA normalization node {} requires one shape-preserving input",
                            id.0
                        ),
                    });
                }
            }
            ComputeKind::Softmax => {
                if shape.in_shapes.len() != 1
                    || shape.out_shapes.len() != 1
                    || shape.in_shapes[0] != shape.out_shapes[0]
                {
                    errors.push(LegalizationError::TargetConstraintViolation {
                        detail: format!(
                            "XDNA softmax node {} requires one shape-preserving input",
                            id.0
                        ),
                    });
                }
            }
            ComputeKind::Attention => {
                let valid_rank = shape
                    .out_shapes
                    .first()
                    .map(|output| matches!(output.dims.len(), 2 | 3) && !output.dims.is_empty())
                    .unwrap_or(false);
                if shape.in_shapes.len() != 3
                    || shape.out_shapes.len() != 1
                    || !valid_rank
                    || shape.in_shapes.iter().any(|input| {
                        shape
                            .out_shapes
                            .first()
                            .map(|output| input != output)
                            .unwrap_or(true)
                    })
                {
                    errors.push(LegalizationError::TargetConstraintViolation {
                        detail: format!("XDNA attention node {} requires three rank-2/3 shape-preserving inputs", id.0),
                    });
                }
            }
            _ => {}
        }
        errors
    }

    pub fn xdna1() -> Self {
        Self {
            topology: XdnaTopology::xdna1(),
            default_element_type: XdnaElementType::Int8,
        }
    }
    pub fn xdna2() -> Self {
        Self {
            topology: XdnaTopology::xdna2(),
            default_element_type: XdnaElementType::Int8,
        }
    }

    /// Select the largest balanced matmul tile that fits A, B, and C in one
    /// compute tile's local memory. This is a planning primitive for graph
    /// decomposition; it never pretends an oversized whole-matrix buffer is
    /// legal.
    pub fn matmul_tile(
        &self,
        m: usize,
        n: usize,
        k: usize,
        element_type: XdnaElementType,
    ) -> Result<XdnaMatmulTile, String> {
        if m == 0 || n == 0 || k == 0 {
            return Err("XDNA matmul dimensions must be nonzero".into());
        }
        let capacity = self.topology.tile_memory_bytes as u64;
        let bytes = element_type.bytes() as u64;
        let mut side = ((capacity / bytes / 3) as f64).sqrt() as usize;
        side = side.max(1);
        let mut tile = XdnaMatmulTile {
            m: m.min(side),
            n: n.min(side),
            k: k.min(side),
        };
        while (tile.m as u64 * tile.k as u64
            + tile.k as u64 * tile.n as u64
            + tile.m as u64 * tile.n as u64)
            .saturating_mul(bytes)
            > capacity
        {
            if tile.m >= tile.n && tile.m >= tile.k && tile.m > 1 {
                tile.m -= 1;
            } else if tile.n >= tile.k && tile.n > 1 {
                tile.n -= 1;
            } else if tile.k > 1 {
                tile.k -= 1;
            } else {
                return Err("XDNA tile memory cannot hold a matmul tile".into());
            }
        }
        Ok(tile)
    }

    /// Partition a contiguous tensor along its leading dimension so each
    /// shard, including input and output staging, fits one compute tile.
    /// Offsets are expressed in both rows and bytes to make the result usable
    /// directly by DMA lowering.
    pub fn partition_rows(
        &self,
        shape: &TensorShape,
        element_type: XdnaElementType,
    ) -> Result<Vec<XdnaRowPartition>, String> {
        self.partition_rows_for_buffers(shape, element_type, 1)
    }

    pub fn partition_rows_for_buffers(
        &self,
        shape: &TensorShape,
        element_type: XdnaElementType,
        buffer_count: usize,
    ) -> Result<Vec<XdnaRowPartition>, String> {
        if buffer_count == 0 {
            return Err("XDNA row partition requires at least one buffer".into());
        }
        let rows = *shape
            .dims
            .first()
            .ok_or_else(|| "XDNA row partition requires a non-empty shape".to_string())?;
        if rows == 0 || self.topology.compute_tiles.is_empty() {
            return Err("XDNA row partition requires rows and compute tiles".into());
        }
        let row_elements = shape.dims[1..]
            .iter()
            .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim as u64))
            .ok_or_else(|| "XDNA row element-count overflow".to_string())?;
        let row_bytes = row_elements
            .checked_mul(element_type.bytes() as u64)
            .ok_or_else(|| "XDNA row byte-size overflow".to_string())?;
        let working_row_bytes = row_bytes
            .checked_mul(buffer_count as u64)
            .ok_or_else(|| "XDNA working row byte-size overflow".to_string())?;
        if working_row_bytes == 0 || working_row_bytes > self.topology.tile_memory_bytes as u64 {
            return Err(format!("one XDNA row working set requires {working_row_bytes} bytes across {buffer_count} buffers, tile local memory is {}", self.topology.tile_memory_bytes));
        }
        let rows_per_tile = (self.topology.tile_memory_bytes as u64 / working_row_bytes) as usize;
        let mut partitions = Vec::new();
        let mut offset = 0;
        while offset < rows {
            let count = (rows - offset).min(rows_per_tile);
            let tile =
                self.topology.compute_tiles[partitions.len() % self.topology.compute_tiles.len()];
            partitions.push(XdnaRowPartition {
                tile,
                row_offset: offset,
                rows: count,
                byte_offset: offset as u64 * row_bytes,
                bytes: count as u64 * row_bytes,
            });
            offset += count;
        }
        Ok(partitions)
    }

    pub fn legalize(&self, graph: SpatialGraph) -> Result<LegalizedGraph, Vec<LegalizationError>> {
        let mut hard_errors = Vec::new();
        let element_bytes = self.default_element_type.bytes() as usize;
        for node in graph.nodes() {
            if let SpatialNode::Compute {
                id, kind, shape, ..
            } = node
            {
                hard_errors.extend(Self::validate_compute_shape(*id, kind, shape));
                for tensor in shape.in_shapes.iter().chain(shape.out_shapes.iter()) {
                    let bytes = tensor_bytes(tensor, element_bytes);
                    if bytes > self.topology.tile_memory_bytes as u64 {
                        hard_errors.push(LegalizationError::TargetConstraintViolation {
                            detail: format!("XDNA compute node {} tensor requires {} bytes, tile local memory is {} bytes", id.0, bytes, self.topology.tile_memory_bytes),
                        });
                    }
                }
            }
        }
        if !hard_errors.is_empty() {
            return Err(hard_errors);
        }
        let topology = self.topology.clone();
        let element_bytes = self.default_element_type.bytes() as usize;
        crate::legalize::legalize(graph, move |node| {
            let mut errors = Vec::new();
            if let SpatialNode::Compute {
                id, kind, shape, ..
            } = node
            {
                errors.extend(Self::validate_compute_shape(*id, kind, shape));
                for tensor in shape.in_shapes.iter().chain(shape.out_shapes.iter()) {
                    let bytes = tensor_bytes(tensor, element_bytes);
                    if bytes > topology.tile_memory_bytes as u64 {
                        errors.push(LegalizationError::TargetConstraintViolation { detail: format!("XDNA compute node {} tensor requires {} bytes, tile local memory is {} bytes", id.0, bytes, topology.tile_memory_bytes) });
                    }
                }
            }
            if errors.is_empty() {
                Ok(())
            } else {
                Err(errors)
            }
        })
    }

    pub fn lower(&self, graph: &LegalizedGraph) -> Result<XdnaProgram, Vec<String>> {
        self.lower_graph(graph.graph())
    }

    /// Compile a graph for execution, including target legality before any
    /// XDNA buffers, workers, or runtime sequence are emitted.
    pub fn lower_executable_graph(&self, graph: SpatialGraph) -> Result<XdnaProgram, Vec<String>> {
        let legalized = self.legalize(graph).map_err(|errors| {
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
        })?;
        self.lower(&legalized)
    }

    pub fn lower_graph(&self, graph: &SpatialGraph) -> Result<XdnaProgram, Vec<String>> {
        let order = graph
            .topological_sort()
            .ok_or_else(|| vec!["graph contains a cycle".into()])?;
        let compute_tiles = self.topology.compute_tiles.clone();
        if compute_tiles.is_empty() {
            return Err(vec!["XDNA topology has no compute tiles".into()]);
        }
        let mut program = XdnaProgram {
            topology: self.topology.clone(),
            buffers: Vec::new(),
            fifos: Vec::new(),
            transfers: Vec::new(),
            workers: Vec::new(),
            barriers: Vec::new(),
            sequence: Vec::new(),
        };
        let mut node_tiles = HashMap::new();
        let mut buffer_ids = HashMap::new();
        let mut dma_prefix = Vec::new();
        let mut dma_suffix = Vec::new();
        for node in graph.nodes() {
            let id = node.id();
            match node {
                SpatialNode::Compute { .. } => {
                    node_tiles.insert(id, compute_tiles[node_tiles.len() % compute_tiles.len()]);
                }
                SpatialNode::Memory { kind, region, .. } => {
                    let bytes = tensor_bytes(&region.shape, region.element_size);
                    let persistent =
                        matches!(kind, MemoryKind::WeightStorage | MemoryKind::KVCache);
                    let memory = if persistent {
                        self.topology
                            .memory_tiles
                            .first()
                            .copied()
                            .map(XdnaMemory::MemoryTile)
                            .unwrap_or(XdnaMemory::Shared)
                    } else {
                        XdnaMemory::Host
                    };
                    buffer_ids.insert(id, format!("buffer_{}", id.0));
                    program.buffers.push(XdnaBuffer {
                        id: format!("buffer_{}", id.0),
                        bytes: bytes.min(u32::MAX as u64) as u32,
                        element_type: self.default_element_type,
                        shape: shape_u32(&region.shape),
                        memory,
                        persistent,
                    });
                }
                _ => {}
            }
        }
        for edge in graph.edges() {
            let source = node_tiles.get(&edge.source).copied();
            let sink = node_tiles.get(&edge.sink).copied();
            if let (Some(producer), Some(consumer)) = (source, sink) {
                let bytes = edge.shape.as_ref().map(|s| tensor_bytes(s, 1)).unwrap_or(0);
                let bid = format!("edge_{}", edge.id.0);
                program.buffers.push(XdnaBuffer {
                    id: bid.clone(),
                    bytes: bytes.min(u32::MAX as u64) as u32,
                    element_type: self.default_element_type,
                    shape: edge.shape.as_ref().map(shape_u32).unwrap_or_default(),
                    memory: XdnaMemory::TileLocal(producer),
                    persistent: false,
                });
                program.fifos.push(ObjectFifo {
                    id: format!("fifo_{}", edge.id.0),
                    element_bytes: self.default_element_type.bytes(),
                    capacity: 2,
                    producer,
                    consumer,
                    buffer: bid,
                });
            }
            if let Some(SpatialNode::Memory { kind, region, .. }) = graph.get_node(edge.source) {
                if matches!(kind, MemoryKind::InputBuffer) {
                    if let Some(tile) = node_tiles.get(&edge.sink).copied() {
                        let source_id = format!("buffer_{}", edge.source.0);
                        let staging_id = format!("staging_in_{}", edge.id.0);
                        let bytes = tensor_bytes(&region.shape, region.element_size)
                            .min(u32::MAX as u64) as u32;
                        program.buffers.push(XdnaBuffer {
                            id: staging_id.clone(),
                            bytes,
                            element_type: self.default_element_type,
                            shape: shape_u32(&region.shape),
                            memory: XdnaMemory::TileLocal(tile),
                            persistent: false,
                        });
                        let transfer_id = format!("fill_memory_{}", edge.id.0);
                        program.transfers.push(DmaTransfer {
                            id: transfer_id.clone(),
                            source: source_id,
                            destination: staging_id,
                            bytes,
                            source_offset: 0,
                            destination_offset: 0,
                            rows: 1,
                            source_stride_bytes: 0,
                            destination_stride_bytes: 0,
                            channel: (program.transfers.len() as u16)
                                % self.topology.shim_dma_channels.max(1),
                            asynchronous: true,
                            waits_on: Vec::new(),
                        });
                        dma_prefix.push(RuntimeCommand::Fill { transfer_id });
                    }
                }
            }
            if let Some(SpatialNode::Memory {
                kind: MemoryKind::OutputBuffer,
                region,
                ..
            }) = graph.get_node(edge.sink)
            {
                if let Some(tile) = node_tiles.get(&edge.source).copied() {
                    let source_id = format!("staging_out_{}", edge.id.0);
                    let destination_id = format!("buffer_{}", edge.sink.0);
                    let bytes = tensor_bytes(&region.shape, region.element_size)
                        .min(u32::MAX as u64) as u32;
                    program.buffers.push(XdnaBuffer {
                        id: source_id.clone(),
                        bytes,
                        element_type: self.default_element_type,
                        shape: shape_u32(&region.shape),
                        memory: XdnaMemory::TileLocal(tile),
                        persistent: false,
                    });
                    let transfer_id = format!("drain_memory_{}", edge.id.0);
                    program.transfers.push(DmaTransfer {
                        id: transfer_id.clone(),
                        source: source_id,
                        destination: destination_id,
                        bytes,
                        source_offset: 0,
                        destination_offset: 0,
                        rows: 1,
                        source_stride_bytes: 0,
                        destination_stride_bytes: 0,
                        channel: (program.transfers.len() as u16)
                            % self.topology.shim_dma_channels.max(1),
                        asynchronous: true,
                        waits_on: Vec::new(),
                    });
                    dma_suffix.push(RuntimeCommand::Drain { transfer_id });
                }
            }
        }
        for edge in graph.edges() {
            if node_tiles.contains_key(&edge.source) && node_tiles.contains_key(&edge.sink) {
                let barrier_id = format!("barrier_{}_{}", edge.source.0, edge.sink.0);
                let producer = format!("worker_{}", edge.source.0);
                program.barriers.push(XdnaBarrier {
                    id: barrier_id.clone(),
                    waits_on: vec![producer],
                });
            }
        }
        for id in order {
            if let Some(SpatialNode::Compute { kind, .. }) = graph.get_node(id) {
                let tile = node_tiles[&id];
                let worker_id = format!("worker_{}", id.0);
                let inputs = graph
                    .incoming_edges(id)
                    .iter()
                    .filter_map(|edge| {
                        if node_tiles.contains_key(&edge.source) {
                            Some(format!("fifo_{}", edge.id.0))
                        } else if matches!(
                            graph.get_node(edge.source),
                            Some(SpatialNode::Memory {
                                kind: MemoryKind::InputBuffer,
                                ..
                            })
                        ) {
                            Some(format!("staging_in_{}", edge.id.0))
                        } else if matches!(
                            graph.get_node(edge.source),
                            Some(SpatialNode::Memory {
                                kind: MemoryKind::WeightStorage | MemoryKind::KVCache,
                                ..
                            })
                        ) {
                            Some(format!("buffer_{}", edge.source.0))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let outputs = graph
                    .outgoing_edges(id)
                    .iter()
                    .filter_map(|edge| {
                        if node_tiles.contains_key(&edge.sink) {
                            Some(format!("fifo_{}", edge.id.0))
                        } else if matches!(
                            graph.get_node(edge.sink),
                            Some(SpatialNode::Memory {
                                kind: MemoryKind::OutputBuffer,
                                ..
                            })
                        ) {
                            Some(format!("staging_out_{}", edge.id.0))
                        } else if matches!(
                            graph.get_node(edge.sink),
                            Some(SpatialNode::Memory {
                                kind: MemoryKind::KVCache,
                                ..
                            })
                        ) {
                            Some(format!("buffer_{}", edge.sink.0))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let waits_on = graph
                    .incoming_edges(id)
                    .iter()
                    .filter_map(|edge| {
                        if node_tiles.contains_key(&edge.source) {
                            Some(format!("barrier_{}_{}", edge.source.0, id.0))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                program.workers.push(XdnaWorker {
                    id: worker_id.clone(),
                    tile,
                    kernel: format!("prism.xdna.{}", compute_name(kind)),
                    inputs,
                    outputs,
                    waits_on: waits_on.clone(),
                    input_offsets: vec![],
                    output_offsets: vec![],
                });
                for barrier_id in waits_on {
                    program
                        .sequence
                        .push(RuntimeCommand::Barrier { barrier_id });
                }
                program.sequence.push(RuntimeCommand::Run { worker_id });
            }
        }
        dma_prefix.extend(program.sequence);
        dma_prefix.extend(dma_suffix);
        program.sequence = dma_prefix;
        program
            .validate()
            .map_err(|e| e.into_iter().collect::<Vec<_>>())?;
        Ok(program)
    }
}

impl SpatialTarget for XdnaTarget {
    type Calibration = ();
    type Artifact = XdnaProgram;

    fn legalize(&self, graph: &SpatialGraph) -> Result<LegalizedGraph, Vec<LegalizationError>> {
        XdnaTarget::legalize(self, graph.clone())
    }

    fn estimate(&self, graph: &LegalizedGraph) -> CostEstimate {
        let inner = graph.graph();
        let peak_memory = inner
            .nodes()
            .iter()
            .filter_map(|node| {
                if let SpatialNode::Memory { region, .. } = node {
                    Some(tensor_bytes(&region.shape, region.element_size))
                } else {
                    None
                }
            })
            .sum();
        let compute_nodes = inner
            .nodes()
            .iter()
            .filter(|node| matches!(node, SpatialNode::Compute { .. }))
            .count() as u64;
        CostEstimate::new(
            Duration::from_micros(compute_nodes.max(1) * 10),
            peak_memory,
            inner
                .edges()
                .iter()
                .filter_map(|edge| edge.shape.as_ref())
                .map(|shape| tensor_bytes(shape, 1))
                .sum(),
            inner.edges().len() as u32,
            compute_nodes as f64 * 0.001,
            0.25,
        )
    }

    fn lower(&self, graph: &LegalizedGraph) -> Result<Self::Artifact, LoweringError> {
        self.lower_graph(graph.graph())
            .map_err(|errors| LoweringError::BackendError(errors.join("; ")))
    }

    fn capabilities(&self) -> TargetCapabilities {
        TargetCapabilities {
            sequential_schedules: false,
            cross_domain_concurrency: true,
            gpu_ane_overlap: false,
            pipeline_overlap: true,
            max_concurrent_regions: match self.topology.generation {
                // amdxdna resource-solver limits for client NPUs: Phoenix /
                // Hawk Point support six workload contexts; Strix supports
                // sixteen. This is distinct from physical tile count.
                XdnaGeneration::Aie2 => 6,
                XdnaGeneration::Aie2p => 16,
            },
            max_weight_memory_bytes: if self.topology.l2_memory_bytes > 0 {
                u64::from(self.topology.l2_memory_bytes)
            } else {
                u64::from(self.topology.tile_memory_bytes)
                    * self.topology.compute_tiles.len() as u64
            },
            max_scratch_memory_bytes: u64::from(self.topology.tile_memory_bytes)
                * self.topology.compute_tiles.len() as u64,
            supports_compressed_kv_cache: true,
            supports_multi_gpu: false,
        }
    }
}

fn tensor_bytes(shape: &TensorShape, element_size: usize) -> u64 {
    shape
        .dims
        .iter()
        .try_fold(element_size as u64, |a, d| a.checked_mul(*d as u64))
        .unwrap_or(u64::MAX)
}
fn shape_u32(shape: &TensorShape) -> Vec<u32> {
    shape
        .dims
        .iter()
        .map(|d| (*d).min(u32::MAX as usize) as u32)
        .collect()
}
fn compute_name(kind: &ComputeKind) -> &'static str {
    match kind {
        ComputeKind::MatMul => "matmul",
        ComputeKind::Convolution => "convolution",
        ComputeKind::Elementwise => "elementwise",
        ComputeKind::Normalization => "normalization",
        ComputeKind::Softmax => "softmax",
        ComputeKind::Attention => "attention",
        ComputeKind::RoPE => "rope",
        ComputeKind::SSM => "ssm",
        ComputeKind::Reshape => "reshape",
        ComputeKind::Gather => "gather",
        ComputeKind::Custom(_) => "custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        ComputeIntensity, EdgeDirection, ShapeContract, SpatialEdge, SpatialEdgeId, SpatialNodeId,
    };
    use prism_ecs_ir::cimage_types::TensorShape;

    #[test]
    fn matmul_tile_respects_compute_tile_memory() {
        let target = XdnaTarget::xdna2();
        let tile = target
            .matmul_tile(4096, 4096, 4096, XdnaElementType::F16)
            .unwrap();
        let bytes = (tile.m * tile.k + tile.k * tile.n + tile.m * tile.n) * 2;
        assert!(bytes <= target.topology.tile_memory_bytes as usize);
        assert!(tile.m < 4096 && tile.n < 4096 && tile.k < 4096);
    }

    #[test]
    fn matmul_tile_rejects_zero_dimensions() {
        assert!(XdnaTarget::xdna1()
            .matmul_tile(0, 16, 16, XdnaElementType::Int8)
            .is_err());
    }

    #[test]
    fn row_partition_covers_tensor_with_dma_offsets() {
        let target = XdnaTarget::xdna2();
        let partitions = target
            .partition_rows(
                &TensorShape {
                    dims: vec![5000, 256],
                },
                XdnaElementType::Int8,
            )
            .unwrap();
        assert!(partitions.len() > 1);
        assert_eq!(partitions.first().unwrap().row_offset, 0);
        assert_eq!(partitions.first().unwrap().byte_offset, 0);
        let last = partitions.last().unwrap();
        assert_eq!(last.row_offset + last.rows, 5000);
        for partition in &partitions {
            assert!(partition.bytes <= target.topology.tile_memory_bytes as u64);
            assert_eq!(partition.byte_offset, partition.row_offset as u64 * 256);
        }
    }

    #[test]
    fn row_partition_rejects_row_larger_than_tile_memory() {
        let target = XdnaTarget::xdna1();
        assert!(target
            .partition_rows(
                &TensorShape {
                    dims: vec![2, 2_000_000]
                },
                XdnaElementType::Int8,
            )
            .is_err());
    }

    #[test]
    fn multi_buffer_row_partition_accounts_for_all_tile_residents() {
        let target = XdnaTarget::xdna2();
        let shape = TensorShape {
            dims: vec![2, 30_000],
        };
        assert!(target.partition_rows(&shape, XdnaElementType::Int8).is_ok());
        let error = target
            .partition_rows_for_buffers(&shape, XdnaElementType::Int8, 4)
            .unwrap_err();
        assert!(error.contains("across 4 buffers"));
    }

    #[test]
    fn weight_capacity_uses_memory_tile_l2() {
        let target = XdnaTarget::xdna2();
        let capabilities = <XdnaTarget as SpatialTarget>::capabilities(&target);
        assert_eq!(
            capabilities.max_weight_memory_bytes,
            u64::from(target.topology.l2_memory_bytes)
        );
        assert_ne!(
            capabilities.max_weight_memory_bytes,
            u64::from(target.topology.tile_memory_bytes)
                * target.topology.compute_tiles.len() as u64
        );
    }

    #[test]
    fn concurrent_context_capacity_matches_client_generation() {
        assert_eq!(
            <XdnaTarget as SpatialTarget>::capabilities(&XdnaTarget::xdna1())
                .max_concurrent_regions,
            6
        );
        assert_eq!(
            <XdnaTarget as SpatialTarget>::capabilities(&XdnaTarget::xdna2())
                .max_concurrent_regions,
            16
        );
    }

    #[test]
    fn executable_lowering_runs_legality_before_emission() {
        let mut graph = SpatialGraph::new();
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::Custom("unsupported_xdna_op".into()),
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![8, 8] }],
                vec![TensorShape { dims: vec![8, 8] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        let error = XdnaTarget::xdna2()
            .lower_executable_graph(graph)
            .unwrap_err()
            .join("; ");
        assert!(error.contains("MatMul, Elementwise, Normalization, Softmax, and Attention"));
    }

    #[test]
    fn executable_lowering_accepts_shape_preserving_softmax() {
        let mut graph = SpatialGraph::new();
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::Softmax,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![8, 8] }],
                vec![TensorShape { dims: vec![8, 8] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        let program = XdnaTarget::xdna2()
            .lower_executable_graph(graph)
            .expect("shape-preserving softmax is a native XDNA kernel");
        assert_eq!(program.workers[0].kernel, "prism.xdna.softmax");
    }

    #[test]
    fn executable_lowering_accepts_rank_three_attention() {
        let shape = TensorShape {
            dims: vec![2, 8, 16],
        };
        let mut graph = SpatialGraph::new();
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::Attention,
            shape: ShapeContract::new(
                vec![shape.clone(), shape.clone(), shape.clone()],
                vec![shape],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        let program = XdnaTarget::xdna2()
            .lower_executable_graph(graph)
            .expect("rank-three attention is a native XDNA kernel");
        assert_eq!(program.workers[0].kernel, "prism.xdna.attention");
    }

    #[test]
    fn lowering_materializes_fifo_and_dependency_barrier() {
        let mut graph = SpatialGraph::new();
        let shape = || {
            ShapeContract::new(
                vec![TensorShape { dims: vec![8, 8] }],
                vec![TensorShape { dims: vec![8, 8] }],
            )
        };
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: shape(),
            intensity: ComputeIntensity::ComputeBound,
        });
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(2),
            kind: ComputeKind::Elementwise,
            shape: shape(),
            intensity: ComputeIntensity::MemoryBound,
        });
        graph.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: SpatialNodeId(1),
            sink: SpatialNodeId(2),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![8, 8] }),
        });
        let program = XdnaTarget::xdna2().lower_graph(&graph).unwrap();
        assert_eq!(program.fifos.len(), 1);
        assert_eq!(program.barriers.len(), 1);
        assert!(program.workers[1].inputs.contains(&"fifo_1".to_string()));
        assert!(program
            .sequence
            .iter()
            .any(|command| matches!(command, RuntimeCommand::Barrier { .. })));
    }

    #[test]
    fn lowering_materializes_host_tile_dma_for_io_memory() {
        let mut graph = SpatialGraph::new();
        let region = crate::graph::MemoryRegion {
            shape: TensorShape { dims: vec![8, 8] },
            element_size: 1,
            strides: vec![],
        };
        graph.add_node(SpatialNode::Memory {
            id: SpatialNodeId(1),
            kind: MemoryKind::InputBuffer,
            region: region.clone(),
        });
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(2),
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![8, 8] }],
                vec![TensorShape { dims: vec![8, 8] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        graph.add_node(SpatialNode::Memory {
            id: SpatialNodeId(3),
            kind: MemoryKind::OutputBuffer,
            region,
        });
        graph.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: SpatialNodeId(1),
            sink: SpatialNodeId(2),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![8, 8] }),
        });
        graph.add_edge(SpatialEdge {
            id: SpatialEdgeId(2),
            source: SpatialNodeId(2),
            sink: SpatialNodeId(3),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![8, 8] }),
        });
        let program = XdnaTarget::xdna2().lower_graph(&graph).unwrap();
        assert_eq!(program.transfers.len(), 2);
        assert!(matches!(
            program.sequence.first(),
            Some(RuntimeCommand::Fill { .. })
        ));
        assert!(matches!(
            program.sequence.last(),
            Some(RuntimeCommand::Drain { .. })
        ));
    }

    #[test]
    fn persistent_weight_and_kv_edges_bind_memory_tile_buffers_directly() {
        let mut graph = SpatialGraph::new();
        let region = crate::graph::MemoryRegion {
            shape: TensorShape { dims: vec![8, 8] },
            element_size: 1,
            strides: vec![],
        };
        graph.add_node(SpatialNode::Memory {
            id: SpatialNodeId(1),
            kind: MemoryKind::WeightStorage,
            region: region.clone(),
        });
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(2),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![8, 8] }],
                vec![TensorShape { dims: vec![8, 8] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        graph.add_node(SpatialNode::Memory {
            id: SpatialNodeId(3),
            kind: MemoryKind::KVCache,
            region,
        });
        graph.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: SpatialNodeId(1),
            sink: SpatialNodeId(2),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });
        graph.add_edge(SpatialEdge {
            id: SpatialEdgeId(2),
            source: SpatialNodeId(2),
            sink: SpatialNodeId(3),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });
        let program = XdnaTarget::xdna2().lower_graph(&graph).unwrap();
        assert!(program.transfers.is_empty());
        assert!(program
            .buffers
            .iter()
            .filter(|buffer| buffer.persistent)
            .all(|buffer| matches!(buffer.memory, XdnaMemory::MemoryTile(_))));
        assert!(program.workers[0].inputs.contains(&"buffer_1".to_string()));
        assert!(program.workers[0].outputs.contains(&"buffer_3".to_string()));
        assert!(
            program
                .buffers
                .iter()
                .filter(|buffer| buffer.persistent)
                .count()
                >= 2
        );
    }

    #[test]
    fn target_legality_hard_rejects_oversized_tile_tensor() {
        let mut graph = SpatialGraph::new();
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![1024, 1024],
                }],
                vec![TensorShape {
                    dims: vec![1024, 1024],
                }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        let errors = XdnaTarget::xdna1()
            .legalize(graph)
            .expect_err("oversized tile tensor must be illegal");
        assert!(errors
            .iter()
            .any(|error| error.to_string().contains("tile local memory")));
    }

    #[test]
    fn target_legality_accounts_for_f16_element_width() {
        let mut graph = SpatialGraph::new();
        graph.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![1, 20_000],
                }],
                vec![TensorShape {
                    dims: vec![1, 20_000],
                }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        let target = XdnaTarget {
            topology: XdnaTopology::xdna2(),
            default_element_type: XdnaElementType::F16,
        };
        assert!(target
            .legalize(graph)
            .expect_err("F16 tensor must exceed tile memory")
            .iter()
            .any(|error| error.to_string().contains("tile local memory")));
    }
}
