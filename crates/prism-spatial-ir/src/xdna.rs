//! Native AMD XDNA/XDNA2 spatial resources and data-movement IR.
//!
//! XDNA is represented as a spatial program rather than a GPU-style kernel
//! launch. The IR intentionally stops at device-neutral AIE concepts; target
//! code generation can later lower these records to native device binaries.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XdnaGeneration {
    Aie2,
    Aie2p,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XdnaElementType {
    Int8,
    UInt8,
    Int16,
    F16,
    BF16,
    F32,
}

impl XdnaElementType {
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::F16 => 2,
            Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileCoord {
    pub col: u16,
    pub row: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XdnaTopology {
    pub generation: XdnaGeneration,
    pub columns: u16,
    pub rows: u16,
    pub compute_tiles: Vec<TileCoord>,
    pub memory_tiles: Vec<TileCoord>,
    pub shim_dma_channels: u16,
    pub tile_memory_bytes: u32,
    /// Capacity of the shared memory-tile/L2 tier across the array.
    #[serde(default)]
    pub l2_memory_bytes: u32,
    pub max_fifo_depth: u16,
}

impl XdnaTopology {
    pub fn xdna1() -> Self {
        // Phoenix/Hawk Point: four compute rows across five columns, with a
        // memory-tile row beneath the compute array.
        Self::client(XdnaGeneration::Aie2, 5, 2, 16 * 1024, 8)
    }
    pub fn xdna2() -> Self {
        Self::client(XdnaGeneration::Aie2p, 8, 4, 32 * 1024, 16)
    }

    /// Ryzen AI client topology: four compute rows followed by one memory
    /// row. The memory row is part of the array geometry but is not eligible
    /// for worker placement; it represents the shared on-chip L2/DMA tier.
    fn client(
        generation: XdnaGeneration,
        columns: u16,
        dma: u16,
        tile_memory_bytes: u32,
        max_fifo_depth: u16,
    ) -> Self {
        let compute_rows = 4;
        let memory_row = compute_rows;
        Self {
            generation,
            columns,
            rows: memory_row + 1,
            compute_tiles: (0..columns)
                .flat_map(|col| (0..compute_rows).map(move |row| TileCoord { col, row }))
                .collect(),
            memory_tiles: (0..columns)
                .map(|col| TileCoord {
                    col,
                    row: memory_row,
                })
                .collect(),
            shim_dma_channels: dma,
            tile_memory_bytes,
            l2_memory_bytes: if generation == XdnaGeneration::Aie2p {
                4 * 1024 * 1024
            } else {
                2560 * 1024
            },
            max_fifo_depth,
        }
    }
    pub fn new(
        generation: XdnaGeneration,
        columns: u16,
        rows: u16,
        dma: u16,
        tile_memory_bytes: u32,
        max_fifo_depth: u16,
    ) -> Self {
        let compute_tiles = (0..columns)
            .flat_map(|col| (0..rows).map(move |row| TileCoord { col, row }))
            .collect();
        Self {
            generation,
            columns,
            rows,
            compute_tiles,
            memory_tiles: Vec::new(),
            shim_dma_channels: dma,
            tile_memory_bytes,
            l2_memory_bytes: 0,
            max_fifo_depth,
        }
    }
    pub fn has_tile(&self, tile: TileCoord) -> bool {
        self.compute_tiles.contains(&tile) || self.memory_tiles.contains(&tile)
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.columns == 0 || self.rows == 0 {
            errors.push("XDNA topology dimensions must be nonzero".into());
        }
        if self.shim_dma_channels == 0 {
            errors.push("XDNA topology must expose at least one DMA channel".into());
        }
        if self.tile_memory_bytes == 0 {
            errors.push("XDNA tile memory must be nonzero".into());
        }
        if !self.memory_tiles.is_empty() && self.l2_memory_bytes == 0 {
            errors.push("XDNA memory tiles require nonzero shared L2 capacity".into());
        }
        if self.max_fifo_depth == 0 {
            errors.push("XDNA FIFO depth must be nonzero".into());
        }
        for tile in self.compute_tiles.iter().chain(self.memory_tiles.iter()) {
            if tile.col >= self.columns || tile.row >= self.rows {
                errors.push(format!(
                    "tile {:?} lies outside {}x{} topology",
                    tile, self.columns, self.rows
                ));
            }
        }
        let mut unique = std::collections::HashSet::new();
        for tile in self.compute_tiles.iter().chain(self.memory_tiles.iter()) {
            if !unique.insert(*tile) {
                errors.push(format!(
                    "tile {:?} is assigned to multiple XDNA roles",
                    tile
                ));
            }
        }
        if self.compute_tiles.is_empty() {
            errors.push("XDNA topology has no compute tiles".into());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum XdnaMemory {
    Host,
    Shared,
    TileLocal(TileCoord),
    MemoryTile(TileCoord),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XdnaBuffer {
    pub id: String,
    pub bytes: u32,
    pub element_type: XdnaElementType,
    pub shape: Vec<u32>,
    pub memory: XdnaMemory,
    pub persistent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectFifo {
    pub id: String,
    pub element_bytes: u32,
    pub capacity: u16,
    pub producer: TileCoord,
    pub consumer: TileCoord,
    pub buffer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DmaTransfer {
    pub id: String,
    pub source: String,
    pub destination: String,
    pub bytes: u32,
    /// Byte window within the source tensor. This is nonzero for tiled
    /// transfers and allows a command sequence to stream subtiles.
    #[serde(default)]
    pub source_offset: u64,
    /// Byte window within the destination tensor.
    #[serde(default)]
    pub destination_offset: u64,
    /// Optional 2-D DMA shape. When `rows` is greater than one, each row is
    /// copied with the declared stride instead of assuming one contiguous
    /// linear window.
    #[serde(default = "one_dma_row")]
    pub rows: u32,
    #[serde(default)]
    pub source_stride_bytes: u64,
    #[serde(default)]
    pub destination_stride_bytes: u64,
    pub channel: u16,
    pub asynchronous: bool,
    pub waits_on: Vec<String>,
}

const fn one_dma_row() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XdnaWorker {
    pub id: String,
    pub tile: TileCoord,
    pub kernel: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub waits_on: Vec<String>,
    /// Byte offsets for tensor operands when this worker consumes a streamed
    /// subtile. Empty means the complete buffer (legacy single-tile form).
    #[serde(default)]
    pub input_offsets: Vec<u64>,
    #[serde(default)]
    pub output_offsets: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XdnaBarrier {
    pub id: String,
    pub waits_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RuntimeCommand {
    Fill { transfer_id: String },
    Drain { transfer_id: String },
    Run { worker_id: String },
    Wait { event_id: String },
    Signal { event_id: String },
    Barrier { barrier_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XdnaProgram {
    pub topology: XdnaTopology,
    pub buffers: Vec<XdnaBuffer>,
    pub fifos: Vec<ObjectFifo>,
    pub transfers: Vec<DmaTransfer>,
    pub workers: Vec<XdnaWorker>,
    pub barriers: Vec<XdnaBarrier>,
    pub sequence: Vec<RuntimeCommand>,
}

impl XdnaProgram {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        use std::collections::{HashMap, HashSet};
        let mut errors = Vec::new();
        if let Err(topology_errors) = self.topology.validate() {
            errors.extend(topology_errors);
        }
        let buffer_ids: HashSet<_> = self.buffers.iter().map(|b| b.id.as_str()).collect();
        let worker_ids: std::collections::HashSet<_> =
            self.workers.iter().map(|w| w.id.as_str()).collect();
        let barrier_ids: std::collections::HashSet<_> =
            self.barriers.iter().map(|b| b.id.as_str()).collect();
        let transfer_ids: std::collections::HashSet<_> =
            self.transfers.iter().map(|t| t.id.as_str()).collect();
        let fifo_ids: HashSet<_> = self.fifos.iter().map(|f| f.id.as_str()).collect();
        if buffer_ids.len() != self.buffers.len() {
            errors.push("duplicate XDNA buffer identifier".into());
        }
        if worker_ids.len() != self.workers.len() {
            errors.push("duplicate XDNA worker identifier".into());
        }
        if barrier_ids.len() != self.barriers.len() {
            errors.push("duplicate XDNA barrier identifier".into());
        }
        if transfer_ids.len() != self.transfers.len() {
            errors.push("duplicate XDNA transfer identifier".into());
        }
        let mut tile_bytes: HashMap<TileCoord, u64> = HashMap::new();
        let mut l2_bytes = 0_u64;
        for buffer in &self.buffers {
            let expected = buffer
                .shape
                .iter()
                .fold(buffer.element_type.bytes(), |acc, dim| {
                    acc.saturating_mul(*dim)
                });
            if expected > buffer.bytes {
                errors.push(format!(
                    "buffer {} declares {} bytes but shape requires {}",
                    buffer.id, buffer.bytes, expected
                ));
            }
            if let XdnaMemory::TileLocal(tile) = buffer.memory {
                if !self.topology.has_tile(tile) {
                    errors.push(format!("buffer {} uses unknown tile {:?}", buffer.id, tile));
                }
                let total = tile_bytes.entry(tile).or_default();
                *total = total.saturating_add(buffer.bytes as u64);
                if *total > self.topology.tile_memory_bytes as u64 {
                    errors.push(format!(
                        "tile {:?} local memory {} exceeds {} bytes",
                        tile, total, self.topology.tile_memory_bytes
                    ));
                }
            }
            if let XdnaMemory::MemoryTile(tile) = buffer.memory {
                if !self.topology.memory_tiles.contains(&tile) {
                    errors.push(format!(
                        "buffer {} uses unknown memory tile {:?}",
                        buffer.id, tile
                    ));
                }
                l2_bytes = l2_bytes.saturating_add(buffer.bytes as u64);
            }
        }
        if l2_bytes > self.topology.l2_memory_bytes as u64 {
            errors.push(format!(
                "memory-tile storage {} exceeds shared L2 capacity {}",
                l2_bytes, self.topology.l2_memory_bytes
            ));
        }
        for fifo in &self.fifos {
            if fifo.capacity == 0 || fifo.capacity > self.topology.max_fifo_depth {
                errors.push(format!(
                    "fifo {} capacity {} is outside 1..={}",
                    fifo.id, fifo.capacity, self.topology.max_fifo_depth
                ));
            }
            if !buffer_ids.contains(fifo.buffer.as_str()) {
                errors.push(format!(
                    "fifo {} references unknown buffer {}",
                    fifo.id, fifo.buffer
                ));
            }
            if let Some(buffer) = self.buffers.iter().find(|buffer| buffer.id == fifo.buffer) {
                if fifo.element_bytes != buffer.element_type.bytes() {
                    errors.push(format!(
                        "fifo {} element width {} disagrees with buffer {} width {}",
                        fifo.id,
                        fifo.element_bytes,
                        fifo.buffer,
                        buffer.element_type.bytes()
                    ));
                }
                if !matches!(
                    buffer.memory,
                    XdnaMemory::TileLocal(_) | XdnaMemory::MemoryTile(_)
                ) {
                    errors.push(format!(
                        "fifo {} buffer {} must be tile-local or memory-tile resident",
                        fifo.id, fifo.buffer
                    ));
                }
            }
            if !self.topology.compute_tiles.contains(&fifo.producer)
                || !self.topology.compute_tiles.contains(&fifo.consumer)
            {
                errors.push(format!(
                    "fifo {} producer and consumer must be compute tiles",
                    fifo.id
                ));
            }
        }
        for transfer in &self.transfers {
            if transfer.rows == 0 {
                errors.push(format!("transfer {} has zero DMA rows", transfer.id));
                continue;
            }
            if transfer.channel >= self.topology.shim_dma_channels {
                errors.push(format!(
                    "transfer {} uses DMA channel {} but topology has {}",
                    transfer.id, transfer.channel, self.topology.shim_dma_channels
                ));
            }
            if !buffer_ids.contains(transfer.source.as_str())
                || !buffer_ids.contains(transfer.destination.as_str())
            {
                errors.push(format!(
                    "transfer {} references unknown buffer",
                    transfer.id
                ));
            }
            if let (Some(source), Some(destination)) = (
                self.buffers.iter().find(|b| b.id == transfer.source),
                self.buffers.iter().find(|b| b.id == transfer.destination),
            ) {
                let row_bytes = (transfer.bytes as u64).checked_div(transfer.rows as u64);
                let valid_shape = row_bytes.is_some_and(|width| {
                    width > 0 && (transfer.bytes as u64).is_multiple_of(transfer.rows as u64)
                });
                let source_stride = if transfer.source_stride_bytes == 0 {
                    row_bytes.unwrap_or(0)
                } else {
                    transfer.source_stride_bytes
                };
                let destination_stride = if transfer.destination_stride_bytes == 0 {
                    row_bytes.unwrap_or(0)
                } else {
                    transfer.destination_stride_bytes
                };
                let source_end = row_bytes
                    .map(|width| {
                        transfer
                            .source_offset
                            .saturating_add(
                                source_stride
                                    .saturating_mul(transfer.rows.saturating_sub(1) as u64),
                            )
                            .saturating_add(width)
                    })
                    .unwrap_or(u64::MAX);
                let destination_end = row_bytes
                    .map(|width| {
                        transfer
                            .destination_offset
                            .saturating_add(
                                destination_stride
                                    .saturating_mul(transfer.rows.saturating_sub(1) as u64),
                            )
                            .saturating_add(width)
                    })
                    .unwrap_or(u64::MAX);
                if !valid_shape
                    || source_end > source.bytes as u64
                    || destination_end > destination.bytes as u64
                {
                    errors.push(format!(
                        "transfer {} moves {} bytes at offsets {}:{} beyond source/destination capacity",
                        transfer.id,
                        transfer.bytes,
                        transfer.source_offset,
                        transfer.destination_offset
                    ));
                }
            }
            for dependency in &transfer.waits_on {
                if !transfer_ids.contains(dependency.as_str()) {
                    errors.push(format!(
                        "transfer {} waits on unknown transfer {}",
                        transfer.id, dependency
                    ));
                }
            }
        }
        for worker in &self.workers {
            if !self.topology.compute_tiles.contains(&worker.tile) {
                errors.push(format!(
                    "worker {} is not placed on a compute tile",
                    worker.id
                ));
            }
            for barrier in &worker.waits_on {
                if !barrier_ids.contains(barrier.as_str()) {
                    errors.push(format!(
                        "worker {} references unknown barrier {}",
                        worker.id, barrier
                    ));
                }
            }
            for binding in worker.inputs.iter().chain(worker.outputs.iter()) {
                if !fifo_ids.contains(binding.as_str()) && !buffer_ids.contains(binding.as_str()) {
                    errors.push(format!(
                        "worker {} references unknown FIFO or buffer {}",
                        worker.id, binding
                    ));
                }
            }
            for binding in &worker.inputs {
                if let Some(fifo) = self.fifos.iter().find(|fifo| fifo.id == *binding) {
                    if fifo.consumer != worker.tile {
                        errors.push(format!(
                            "worker {} consumes FIFO {} from tile {:?}, expected {:?}",
                            worker.id, fifo.id, worker.tile, fifo.consumer
                        ));
                    }
                }
            }
            for binding in &worker.outputs {
                if let Some(fifo) = self.fifos.iter().find(|fifo| fifo.id == *binding) {
                    if fifo.producer != worker.tile {
                        errors.push(format!(
                            "worker {} produces FIFO {} from tile {:?}, expected {:?}",
                            worker.id, fifo.id, worker.tile, fifo.producer
                        ));
                    }
                }
            }
        }
        for barrier in &self.barriers {
            for dependency in &barrier.waits_on {
                if !worker_ids.contains(dependency.as_str()) {
                    errors.push(format!(
                        "barrier {} references unknown worker {}",
                        barrier.id, dependency
                    ));
                }
            }
        }
        let mut completed_workers = HashSet::new();
        for command in &self.sequence {
            match command {
                RuntimeCommand::Fill { transfer_id } | RuntimeCommand::Drain { transfer_id } => {
                    if !transfer_ids.contains(transfer_id.as_str()) {
                        errors.push(format!(
                            "sequence references unknown transfer {}",
                            transfer_id
                        ));
                    }
                }
                RuntimeCommand::Run { worker_id } => {
                    if !worker_ids.contains(worker_id.as_str()) {
                        errors.push(format!("sequence references unknown worker {}", worker_id));
                    } else {
                        completed_workers.insert(worker_id.as_str());
                    }
                }
                RuntimeCommand::Barrier { barrier_id } => {
                    if !barrier_ids.contains(barrier_id.as_str()) {
                        errors.push(format!(
                            "sequence references unknown barrier {}",
                            barrier_id
                        ));
                    } else if let Some(barrier) = self
                        .barriers
                        .iter()
                        .find(|barrier| barrier.id == *barrier_id)
                    {
                        for dependency in &barrier.waits_on {
                            if !completed_workers.contains(dependency.as_str()) {
                                errors.push(format!(
                                    "barrier {} executes before worker {}",
                                    barrier.id, dependency
                                ));
                            }
                        }
                    }
                }
                RuntimeCommand::Wait { event_id } | RuntimeCommand::Signal { event_id } => {
                    if event_id.is_empty() {
                        errors.push("sequence contains an empty event identifier".into());
                    }
                }
            }
        }
        for transfer in &self.transfers {
            let command_index = |transfer_id: &str| {
                self.sequence.iter().position(|command| matches!(command, RuntimeCommand::Fill { transfer_id: id } | RuntimeCommand::Drain { transfer_id: id } if id == transfer_id))
            };
            if let Some(current) = command_index(&transfer.id) {
                for dependency in &transfer.waits_on {
                    if let Some(required) = command_index(dependency) {
                        if required >= current {
                            errors.push(format!(
                                "transfer {} is scheduled before dependency {}",
                                transfer.id, dependency
                            ));
                        }
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xdna2_has_strix_like_topology() {
        let topology = XdnaTopology::xdna2();
        assert_eq!(topology.columns, 8);
        assert_eq!(topology.rows, 5);
        assert_eq!(topology.compute_tiles.len(), 8 * 4);
        assert_eq!(topology.memory_tiles.len(), 8);
        assert_eq!(topology.l2_memory_bytes, 4 * 1024 * 1024);
    }

    #[test]
    fn xdna1_has_phoenix_like_topology() {
        let topology = XdnaTopology::xdna1();
        assert_eq!(topology.columns, 5);
        assert_eq!(topology.rows, 5);
        assert_eq!(topology.compute_tiles.len(), 5 * 4);
        assert_eq!(topology.memory_tiles.len(), 5);
        assert_eq!(topology.l2_memory_bytes, 2560 * 1024);
    }

    #[test]
    fn program_rejects_invalid_dma_and_fifo_references() {
        let topology = XdnaTopology::xdna1();
        let program = XdnaProgram {
            topology,
            buffers: vec![],
            fifos: vec![ObjectFifo {
                id: "f".into(),
                element_bytes: 2,
                capacity: 99,
                producer: TileCoord { col: 99, row: 0 },
                consumer: TileCoord { col: 0, row: 0 },
                buffer: "missing".into(),
            }],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        assert!(program.validate().is_err());
    }

    #[test]
    fn program_rejects_host_resident_fifo_storage() {
        let topology = XdnaTopology::xdna1();
        let tile = TileCoord { col: 0, row: 0 };
        let program = XdnaProgram {
            topology,
            buffers: vec![XdnaBuffer {
                id: "host_buffer".into(),
                bytes: 16,
                element_type: XdnaElementType::F16,
                shape: vec![8],
                memory: XdnaMemory::Host,
                persistent: false,
            }],
            fifos: vec![ObjectFifo {
                id: "fifo".into(),
                element_bytes: 2,
                capacity: 1,
                producer: tile,
                consumer: tile,
                buffer: "host_buffer".into(),
            }],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let error = program.validate().unwrap_err().join("; ");
        assert!(error.contains("must be tile-local or memory-tile resident"));
    }

    #[test]
    fn program_rejects_memory_tile_storage_over_shared_l2_capacity() {
        let topology = XdnaTopology::xdna2();
        let memory_tile = topology.memory_tiles[0];
        let bytes = topology.l2_memory_bytes.saturating_add(1);
        let program = XdnaProgram {
            topology,
            buffers: vec![XdnaBuffer {
                id: "kv".into(),
                bytes,
                element_type: XdnaElementType::Int8,
                shape: vec![bytes],
                memory: XdnaMemory::MemoryTile(memory_tile),
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        assert!(program
            .validate()
            .unwrap_err()
            .iter()
            .any(|error| error.contains("shared L2 capacity")));
    }
}
