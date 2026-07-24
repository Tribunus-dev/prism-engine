//! AMD XDNA NPU code generation for ECS-native IR operations.
//!
//! Lowers high-level IR operations (e.g., `linalg.matmul`) into XDNA graph
//! pseudo-code describing a graph-level program suitable for the AMD Ryzen AI
//! NPU (XDNA architecture with AIE2/AIE2P engines).
//!
//! The XDNA NPU uses a graph-of-nodes programming model where each node maps
//! to a hardware-scheduled AI Engine (AIE) tile operation. Unlike GPU-style
//! grid dispatch, the NPU compiler accepts a graph-level description and
//! handles tiling, routing, and resource allocation internally.

use prism_ecs_core::{Entity, World};

use prism_ecs_ir::ir_types::FloatType;
use prism_ecs_ir::ir_types::{FloatKind, TensorType, Type};
use prism_ecs_ir::op::{op_name, operands, Results};
use prism_ecs_ir::value::ValueType;
use prism_spatial_ir::xdna::{
    DmaTransfer, ObjectFifo, RuntimeCommand, XdnaBarrier, XdnaBuffer, XdnaElementType, XdnaMemory,
    XdnaProgram, XdnaWorker,
};
use prism_spatial_ir::xdna_target::XdnaTarget;

/// Error type for AMD XDNA NPU lowering failures.
#[derive(Debug)]
pub enum AmdNpuLowerError {
    /// The operation is not one that can be lowered to XDNA NPU IR.
    UnsupportedOp(String),
    /// A required operand or result is missing.
    MissingOperand(String),
    /// An operand or result is missing a type annotation.
    MissingType(String),
}

// ── Element type mapping ─────────────────────────────────────────────────────

/// Map an IR element type to its XDNA data type name.
fn element_type_to_xdna(ty: &Type) -> &'static str {
    match ty {
        Type::Float(FloatType { kind }) => match kind {
            FloatKind::F16 => "float16",
            FloatKind::BF16 => "bfloat16",
            FloatKind::F32 => "float32",
            FloatKind::F64 => "float64",
            FloatKind::F8E4M3 | FloatKind::F8E5M2 => "float8",
        },
        Type::Integer(_) => "int32",
        Type::Index
        | Type::NoneType
        | Type::Function(_)
        | Type::Tensor(_)
        | Type::Vector(_)
        | Type::Complex(_) => "float16",
    }
}

fn tensor_element_type(ty: &Type) -> Result<XdnaElementType, AmdNpuLowerError> {
    match ty {
        Type::Float(FloatType {
            kind: FloatKind::F16,
        }) => Ok(XdnaElementType::F16),
        Type::Float(FloatType {
            kind: FloatKind::BF16,
        }) => Ok(XdnaElementType::BF16),
        Type::Float(FloatType {
            kind: FloatKind::F32,
        }) => Ok(XdnaElementType::F32),
        Type::Integer(integer) if integer.width <= 8 => Ok(XdnaElementType::Int8),
        Type::Integer(integer) if integer.width <= 16 => Ok(XdnaElementType::Int16),
        other => Err(AmdNpuLowerError::UnsupportedOp(format!(
            "XDNA matmul element type is unsupported: {other:?}"
        ))),
    }
}

/// Extract a `TensorType` from a value entity's `ValueType` component.
fn require_tensor(
    world: &World,
    entity: Entity,
    label: &str,
) -> Result<TensorType, AmdNpuLowerError> {
    let value_ty = world
        .get_component::<ValueType>(entity)
        .ok_or_else(|| AmdNpuLowerError::MissingType(format!("{label} is missing ValueType")))?;

    match &value_ty.0 {
        Type::Tensor(t) => Ok(t.clone()),
        other => Err(AmdNpuLowerError::MissingType(format!(
            "{label} has non-tensor type {other:?}"
        ))),
    }
}

// ── Emitters ─────────────────────────────────────────────────────────────────

/// Emit an XDNA NPU graph description for a matmul operation.
#[rustfmt::skip]
fn emit_matmul_xdna(m: u64, n: u64, k: u64, xdna_type: &str) -> String {
    format!(
        r#"// AMD XDNA NPU Graph
// matmul {m}x{k}x{n} ({xdna_type})
NODE @0: matmul(A[M,K], B[K,N]) -> C[M,N]
  ENGINE: AIE2
  DATA_TYPE: {xdna_type}
  TILING: {{M: 64, K: 64, N: 64}}
"#,
        m = m, n = n, k = k, xdna_type = xdna_type,
    )
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Lower a `linalg.matmul` to an AMD XDNA NPU graph description.
///
/// Given a `linalg.matmul` op consuming operands `A`, `B`, `C` where the
/// semantics are `C += A @ B`, this function emits a textual graph-level
/// pseudo-code description targeting the AMD XDNA NPU architecture.
///
/// Each operand **must** carry a `tensor<...>` type with a 2-D shape so that
/// the dimensions `M`, `K`, `N` can be extracted.
pub fn lower_matmul_to_amd_npu(
    world: &World,
    matmul_op: Entity,
) -> Result<String, AmdNpuLowerError> {
    // 1. Verify the op is a matmul
    let name = op_name(world, matmul_op).unwrap_or_default();
    if name != "linalg.matmul" {
        return Err(AmdNpuLowerError::UnsupportedOp(format!(
            "expected 'linalg.matmul', got '{name}'"
        )));
    }

    // 2. Read operands
    let op_operands = operands(world, matmul_op);
    if op_operands.len() < 3 {
        return Err(AmdNpuLowerError::MissingOperand(format!(
            "matmul requires 3 operands (A, B, C), got {}",
            op_operands.len()
        )));
    }

    let a = op_operands[0];
    let b = op_operands[1];
    let c = op_operands[2];

    // 3. Extract tensor shapes
    let a_tensor = require_tensor(world, a, "operand A")?;
    let b_tensor = require_tensor(world, b, "operand B")?;
    let c_tensor = require_tensor(world, c, "operand C")?;

    // 4. Validate shapes: A[M,K] x B[K,N] = C[M,N]
    let shape_ok =
        a_tensor.shape.len() == 2 && b_tensor.shape.len() == 2 && c_tensor.shape.len() == 2;

    if !shape_ok {
        return Err(AmdNpuLowerError::MissingType(
            "all matmul operands must be 2-D tensors".into(),
        ));
    }

    let m = a_tensor.shape[0];
    let k_a = a_tensor.shape[1];
    let k_b = b_tensor.shape[0];
    let n = b_tensor.shape[1];
    let c_m = c_tensor.shape[0];
    let c_n = c_tensor.shape[1];

    if k_a != k_b {
        return Err(AmdNpuLowerError::MissingType(format!(
            "matmul dimension mismatch: A[1] = {k_a} ≠ B[0] = {k_b}"
        )));
    }
    if m != c_m || n != c_n {
        return Err(AmdNpuLowerError::MissingType(format!(
            "matmul result shape mismatch: expected [{m}, {n}], got [{c_m}, {c_n}]"
        )));
    }

    // 5. Emit XDNA graph description
    let xdna_type = element_type_to_xdna(&a_tensor.element_type);
    Ok(emit_matmul_xdna(m, n, k_a, xdna_type))
}

/// Build Prism's native XDNA program representation for a matmul. The
/// returned value is the compiler IR, not an external tool's source format.
pub fn lower_matmul_to_native_xdna(
    world: &World,
    matmul_op: Entity,
) -> Result<XdnaProgram, AmdNpuLowerError> {
    lower_matmul_to_native_xdna_with_target(world, matmul_op, &XdnaTarget::xdna2())
}

/// Lower a matmul against an explicitly discovered or configured XDNA target.
pub fn lower_matmul_to_native_xdna_with_target(
    world: &World,
    matmul_op: Entity,
    target: &XdnaTarget,
) -> Result<XdnaProgram, AmdNpuLowerError> {
    let name = op_name(world, matmul_op).unwrap_or_default();
    if name != "linalg.matmul" && name != "linalg.batch_matmul" {
        return Err(AmdNpuLowerError::UnsupportedOp(name));
    }
    let ops = operands(world, matmul_op);
    if ops.len() < 3 {
        return Err(AmdNpuLowerError::MissingOperand(
            "matmul requires A, B, C".into(),
        ));
    }
    let a = require_tensor(world, ops[0], "operand A")?;
    let b = require_tensor(world, ops[1], "operand B")?;
    let c = require_tensor(world, ops[2], "operand C")?;
    if a.element_type != b.element_type || a.element_type != c.element_type {
        return Err(AmdNpuLowerError::MissingType(
            "matmul operands must use one element type".into(),
        ));
    }
    let batch = name == "linalg.batch_matmul";
    let valid_rank = if batch {
        a.shape.len() == 3 && b.shape.len() == 3 && c.shape.len() == 3
    } else {
        a.shape.len() == 2 && b.shape.len() == 2 && c.shape.len() == 2
    };
    let rank_offset = if batch { 1 } else { 0 };
    if !valid_rank
        || a.shape[rank_offset + 1] != b.shape[rank_offset]
        || a.shape[rank_offset] != c.shape[rank_offset]
        || b.shape[rank_offset + 1] != c.shape[rank_offset + 1]
    {
        return Err(AmdNpuLowerError::MissingType(
            "invalid matmul shapes".into(),
        ));
    }
    let compute_tiles = target.topology.compute_tiles.clone();
    if compute_tiles.is_empty() {
        return Err(AmdNpuLowerError::UnsupportedOp(
            "XDNA target has no compute tiles".into(),
        ));
    }
    let mut program = XdnaProgram {
        topology: target.topology.clone(),
        buffers: Vec::new(),
        fifos: Vec::new(),
        transfers: Vec::new(),
        workers: vec![],
        barriers: vec![],
        sequence: vec![],
    };
    for (id, shape, persistent) in [
        ("A", &a.shape, false),
        ("B", &b.shape, true),
        ("C", &c.shape, false),
    ] {
        let xdna_type = tensor_element_type(&a.element_type)?;
        let bytes = shape.iter().product::<u64>() * xdna_type.bytes() as u64;
        program.buffers.push(XdnaBuffer {
            id: id.into(),
            bytes: bytes as u32,
            element_type: xdna_type,
            shape: shape.iter().map(|d| *d as u32).collect(),
            memory: if persistent {
                XdnaMemory::Shared
            } else {
                XdnaMemory::Host
            },
            persistent,
        });
    }
    let element_type = tensor_element_type(&a.element_type)?;
    let element_bytes = element_type.bytes() as u64;
    let dims = target
        .matmul_tile(
            a.shape[rank_offset] as usize,
            b.shape[rank_offset + 1] as usize,
            a.shape[rank_offset + 1] as usize,
            element_type,
        )
        .map_err(AmdNpuLowerError::UnsupportedOp)?;
    let tile_shape_a = if batch {
        vec![a.shape[0] as u32, dims.m as u32, dims.k as u32]
    } else {
        vec![dims.m as u32, dims.k as u32]
    };
    let tile_shape_b = if batch {
        vec![b.shape[0] as u32, dims.k as u32, dims.n as u32]
    } else {
        vec![dims.k as u32, dims.n as u32]
    };
    let tile_shape_c = if batch {
        vec![c.shape[0] as u32, dims.m as u32, dims.n as u32]
    } else {
        vec![dims.m as u32, dims.n as u32]
    };
    for tile in &compute_tiles {
        for (prefix, shape) in [
            ("A", tile_shape_a.clone()),
            ("B", tile_shape_b.clone()),
            ("C", tile_shape_c.clone()),
        ] {
            let buffer = format!("{prefix}_tile_{}_{}", tile.col, tile.row);
            let bytes = shape.iter().map(|d| *d as u64).product::<u64>() * element_bytes;
            program.buffers.push(XdnaBuffer {
                id: buffer.clone(),
                bytes: bytes as u32,
                element_type,
                shape,
                memory: XdnaMemory::TileLocal(*tile),
                persistent: false,
            });
        }
        for prefix in ["A", "B", "C"] {
            let buffer = format!("{prefix}_tile_{}_{}", tile.col, tile.row);
            program.fifos.push(ObjectFifo {
                id: format!("fifo_{prefix}_{}_{}", tile.col, tile.row),
                element_bytes: element_type.bytes(),
                capacity: 2,
                producer: *tile,
                consumer: *tile,
                buffer,
            });
        }
    }
    let m = a.shape[rank_offset] as usize;
    let n = b.shape[rank_offset + 1] as usize;
    let k = a.shape[rank_offset + 1] as usize;
    // XDNA1 commonly exposes two shim DMA channels while XDNA2 profiles may
    // expose four or more. Keep the three logical streams (A, B, C) legal on
    // both targets; dependencies serialize streams that share a channel.
    let dma_channels = target.topology.shim_dma_channels.max(1);
    let channel_a = 0;
    let channel_b = if dma_channels >= 3 { 2 } else { 0 };
    let channel_c = if dma_channels >= 2 { 1 } else { 0 };
    let mut previous_drain = None;
    for m0 in (0..m).step_by(dims.m) {
        for n0 in (0..n).step_by(dims.n) {
            let tm = (m - m0).min(dims.m);
            let tn = (n - n0).min(dims.n);
            for k0 in (0..k).step_by(dims.k) {
                let tk = (k - k0).min(dims.k);
                let tile_index = (m0 / dims.m + n0 / dims.n + k0 / dims.k) % compute_tiles.len();
                let worker_tile = compute_tiles[tile_index];
                let a_tile = format!("A_tile_{}_{}", worker_tile.col, worker_tile.row);
                let b_tile = format!("B_tile_{}_{}", worker_tile.col, worker_tile.row);
                let c_tile = format!("C_tile_{}_{}", worker_tile.col, worker_tile.row);
                let fifo_a = format!("fifo_A_{}_{}", worker_tile.col, worker_tile.row);
                let fifo_b = format!("fifo_B_{}_{}", worker_tile.col, worker_tile.row);
                let fifo_c = format!("fifo_C_{}_{}", worker_tile.col, worker_tile.row);
                let tag = format!("m{m0}_n{n0}_k{k0}");
                let fill_a = format!("fill_A_{tag}");
                let fill_b = format!("fill_B_{tag}");
                let worker_id = format!("worker_{tag}");
                let drain = format!("drain_C_{tag}");
                let barrier_id = format!("barrier_{tag}");
                let a_offset = ((m0 * k) + k0) as u64 * element_bytes;
                let b_offset = ((k0 * n) + n0) as u64 * element_bytes;
                let c_offset = ((m0 * n) + n0) as u64 * element_bytes;
                let a_bytes = (tm * tk) as u64 * element_bytes;
                let b_bytes = (tk * tn) as u64 * element_bytes;
                let c_bytes = (tm * tn) as u64 * element_bytes;
                let waits: Vec<String> = previous_drain.iter().cloned().collect();
                program.transfers.push(DmaTransfer {
                    id: fill_a.clone(),
                    source: "A".into(),
                    destination: a_tile.clone(),
                    bytes: a_bytes as u32,
                    source_offset: a_offset,
                    destination_offset: 0,
                    rows: tm as u32,
                    source_stride_bytes: k as u64 * element_bytes,
                    destination_stride_bytes: tk as u64 * element_bytes,
                    channel: channel_a,
                    asynchronous: true,
                    waits_on: waits.clone(),
                });
                program.transfers.push(DmaTransfer {
                    id: fill_b.clone(),
                    source: "B".into(),
                    destination: b_tile.clone(),
                    bytes: b_bytes as u32,
                    source_offset: b_offset,
                    destination_offset: 0,
                    rows: tk as u32,
                    source_stride_bytes: n as u64 * element_bytes,
                    destination_stride_bytes: tn as u64 * element_bytes,
                    channel: channel_b,
                    asynchronous: true,
                    waits_on: waits,
                });
                program.transfers.push(DmaTransfer {
                    id: drain.clone(),
                    source: c_tile,
                    destination: "C".into(),
                    bytes: c_bytes as u32,
                    source_offset: 0,
                    destination_offset: c_offset,
                    rows: tm as u32,
                    source_stride_bytes: tn as u64 * element_bytes,
                    destination_stride_bytes: n as u64 * element_bytes,
                    channel: channel_c,
                    asynchronous: true,
                    waits_on: vec![fill_a.clone(), fill_b.clone()],
                });
                program.workers.push(XdnaWorker {
                    id: worker_id.clone(),
                    tile: worker_tile,
                    kernel: if k0 == 0 {
                        "prism.xdna.matmul".into()
                    } else {
                        "prism.xdna.matmul_accumulate".into()
                    },
                    inputs: vec![fifo_a, fifo_b],
                    outputs: vec![fifo_c],
                    waits_on: vec![],
                    input_offsets: vec![0, 0],
                    output_offsets: vec![0],
                });
                program.barriers.push(XdnaBarrier {
                    id: barrier_id.clone(),
                    waits_on: vec![worker_id.clone()],
                });
                program.sequence.extend([
                    RuntimeCommand::Fill {
                        transfer_id: fill_a,
                    },
                    RuntimeCommand::Fill {
                        transfer_id: fill_b,
                    },
                    RuntimeCommand::Run { worker_id },
                    RuntimeCommand::Barrier { barrier_id },
                    RuntimeCommand::Drain {
                        transfer_id: drain.clone(),
                    },
                ]);
                previous_drain = Some(drain);
            }
        }
    }
    program
        .validate()
        .map_err(|e| AmdNpuLowerError::UnsupportedOp(e.join("; ")))?;
    Ok(program)
}

/// Lower a shape-preserving ECS unary operation into the same native XDNA
/// program form used by spatial compilation. This keeps the ECS entry point
/// from silently falling back to textual pseudo-code for common transformer
/// kernels.
pub fn lower_unary_to_native_xdna_with_target(
    world: &World,
    op: Entity,
    target: &XdnaTarget,
) -> Result<XdnaProgram, AmdNpuLowerError> {
    let kernel = match op_name(world, op).as_deref() {
        Some("linalg.elementwise" | "linalg.normalization" | "linalg.softmax") => {
            op_name(world, op)
                .unwrap()
                .replace("linalg.", "prism.xdna.")
        }
        Some(name) => return Err(AmdNpuLowerError::UnsupportedOp(name.into())),
        None => {
            return Err(AmdNpuLowerError::UnsupportedOp(
                "operation has no name".into(),
            ))
        }
    };
    let ops = operands(world, op);
    if ops.len() != 1 {
        return Err(AmdNpuLowerError::MissingOperand(
            "unary XDNA operation requires one input".into(),
        ));
    }
    let input = require_tensor(world, ops[0], "unary input")?;
    let result = world
        .get_component::<Results>(op)
        .and_then(|results| results.0.first().copied())
        .ok_or_else(|| AmdNpuLowerError::MissingOperand("unary operation has no result".into()))?;
    let output = require_tensor(world, result, "unary output")?;
    if input.shape != output.shape || input.element_type != output.element_type {
        return Err(AmdNpuLowerError::MissingType(
            "unary XDNA input/output must have matching shape and element type".into(),
        ));
    }
    let tile = *target.topology.compute_tiles.first().ok_or_else(|| {
        AmdNpuLowerError::UnsupportedOp("XDNA target has no compute tiles".into())
    })?;
    let element_type = tensor_element_type(&input.element_type)?;
    let bytes = input
        .shape
        .iter()
        .try_fold(element_type.bytes() as u64, |acc, dim| {
            acc.checked_mul(*dim)
        })
        .ok_or_else(|| AmdNpuLowerError::UnsupportedOp("unary tensor size overflow".into()))?;
    let shape_usize = input
        .shape
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                AmdNpuLowerError::UnsupportedOp("unary tensor dimension does not fit usize".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let partitions = target
        .partition_rows_for_buffers(
            &prism_ecs_ir::cimage_types::TensorShape {
                dims: shape_usize.clone(),
            },
            element_type,
            2,
        )
        .map_err(AmdNpuLowerError::UnsupportedOp)?;
    if partitions.len() > 1 {
        return lower_partitioned_unary_program(
            target,
            &kernel,
            &shape_usize,
            element_type,
            bytes,
            &partitions,
        );
    }
    if bytes > target.topology.tile_memory_bytes as u64 {
        return Err(AmdNpuLowerError::UnsupportedOp(
            "unary tensor does not fit one XDNA compute tile".into(),
        ));
    }
    let shape = input
        .shape
        .iter()
        .map(|dim| *dim as u32)
        .collect::<Vec<_>>();
    let program = XdnaProgram {
        topology: target.topology.clone(),
        buffers: vec![
            XdnaBuffer {
                id: "input".into(),
                bytes: bytes as u32,
                element_type,
                shape: shape.clone(),
                memory: XdnaMemory::Host,
                persistent: false,
            },
            XdnaBuffer {
                id: "output".into(),
                bytes: bytes as u32,
                element_type,
                shape: shape.clone(),
                memory: XdnaMemory::Host,
                persistent: false,
            },
            XdnaBuffer {
                id: "input_tile".into(),
                bytes: bytes as u32,
                element_type,
                shape: shape.clone(),
                memory: XdnaMemory::TileLocal(tile),
                persistent: false,
            },
            XdnaBuffer {
                id: "output_tile".into(),
                bytes: bytes as u32,
                element_type,
                shape,
                memory: XdnaMemory::TileLocal(tile),
                persistent: false,
            },
        ],
        fifos: vec![
            ObjectFifo {
                id: "input_fifo".into(),
                element_bytes: element_type.bytes(),
                capacity: 1,
                producer: tile,
                consumer: tile,
                buffer: "input_tile".into(),
            },
            ObjectFifo {
                id: "output_fifo".into(),
                element_bytes: element_type.bytes(),
                capacity: 1,
                producer: tile,
                consumer: tile,
                buffer: "output_tile".into(),
            },
        ],
        transfers: vec![
            DmaTransfer {
                id: "fill_input".into(),
                source: "input".into(),
                destination: "input_tile".into(),
                bytes: bytes as u32,
                source_offset: 0,
                destination_offset: 0,
                rows: 1,
                source_stride_bytes: 0,
                destination_stride_bytes: 0,
                channel: 0,
                asynchronous: true,
                waits_on: vec![],
            },
            DmaTransfer {
                id: "drain_output".into(),
                source: "output_tile".into(),
                destination: "output".into(),
                bytes: bytes as u32,
                source_offset: 0,
                destination_offset: 0,
                rows: 1,
                source_stride_bytes: 0,
                destination_stride_bytes: 0,
                channel: 1 % target.topology.shim_dma_channels.max(1),
                asynchronous: true,
                waits_on: vec!["fill_input".into()],
            },
        ],
        workers: vec![XdnaWorker {
            id: "worker_unary".into(),
            tile,
            kernel,
            inputs: vec!["input_fifo".into()],
            outputs: vec!["output_fifo".into()],
            waits_on: vec![],
            input_offsets: vec![0],
            output_offsets: vec![0],
        }],
        barriers: vec![],
        sequence: vec![
            RuntimeCommand::Fill {
                transfer_id: "fill_input".into(),
            },
            RuntimeCommand::Run {
                worker_id: "worker_unary".into(),
            },
            RuntimeCommand::Drain {
                transfer_id: "drain_output".into(),
            },
        ],
    };
    program
        .validate()
        .map_err(|errors| AmdNpuLowerError::UnsupportedOp(errors.join("; ")))?;
    Ok(program)
}

fn lower_partitioned_unary_program(
    target: &XdnaTarget,
    kernel: &str,
    shape: &[usize],
    element_type: XdnaElementType,
    total_bytes: u64,
    partitions: &[prism_spatial_ir::xdna_target::XdnaRowPartition],
) -> Result<XdnaProgram, AmdNpuLowerError> {
    let full_shape = shape.iter().map(|dim| *dim as u32).collect::<Vec<_>>();
    let mut program = XdnaProgram {
        topology: target.topology.clone(),
        buffers: vec![
            XdnaBuffer {
                id: "input".into(),
                bytes: total_bytes as u32,
                element_type,
                shape: full_shape.clone(),
                memory: XdnaMemory::Host,
                persistent: false,
            },
            XdnaBuffer {
                id: "output".into(),
                bytes: total_bytes as u32,
                element_type,
                shape: full_shape,
                memory: XdnaMemory::Host,
                persistent: false,
            },
        ],
        fifos: vec![],
        transfers: vec![],
        workers: vec![],
        barriers: vec![],
        sequence: vec![],
    };
    for (index, partition) in partitions.iter().enumerate() {
        let tag = format!("unary_{index}");
        let mut tile_shape = shape.to_vec();
        tile_shape[0] = partition.rows;
        let tile_shape = tile_shape.iter().map(|dim| *dim as u32).collect::<Vec<_>>();
        let tile_tag = format!("tile_{}_{}", partition.tile.col, partition.tile.row);
        let input_tile = format!("{tile_tag}_input_tile");
        let output_tile = format!("{tile_tag}_output_tile");
        let input_fifo = format!("{tag}_input_fifo");
        let output_fifo = format!("{tag}_output_fifo");
        let fill = format!("{tag}_fill");
        let drain = format!("{tag}_drain");
        if !program.buffers.iter().any(|buffer| buffer.id == input_tile) {
            program.buffers.extend([
                XdnaBuffer {
                    id: input_tile.clone(),
                    bytes: partition.bytes as u32,
                    element_type,
                    shape: tile_shape.clone(),
                    memory: XdnaMemory::TileLocal(partition.tile),
                    persistent: false,
                },
                XdnaBuffer {
                    id: output_tile.clone(),
                    bytes: partition.bytes as u32,
                    element_type,
                    shape: tile_shape,
                    memory: XdnaMemory::TileLocal(partition.tile),
                    persistent: false,
                },
            ]);
        }
        program.fifos.extend([
            ObjectFifo {
                id: input_fifo.clone(),
                element_bytes: element_type.bytes(),
                capacity: 1,
                producer: partition.tile,
                consumer: partition.tile,
                buffer: input_tile.clone(),
            },
            ObjectFifo {
                id: output_fifo.clone(),
                element_bytes: element_type.bytes(),
                capacity: 1,
                producer: partition.tile,
                consumer: partition.tile,
                buffer: output_tile.clone(),
            },
        ]);
        program.transfers.extend([
            DmaTransfer {
                id: fill.clone(),
                source: "input".into(),
                destination: input_tile,
                bytes: partition.bytes as u32,
                source_offset: partition.byte_offset,
                destination_offset: 0,
                rows: partition.rows as u32,
                source_stride_bytes: partition.bytes / partition.rows as u64,
                destination_stride_bytes: partition.bytes / partition.rows as u64,
                channel: (index as u16) % target.topology.shim_dma_channels.max(1),
                asynchronous: true,
                waits_on: vec![],
            },
            DmaTransfer {
                id: drain.clone(),
                source: output_tile,
                destination: "output".into(),
                bytes: partition.bytes as u32,
                source_offset: 0,
                destination_offset: partition.byte_offset,
                rows: partition.rows as u32,
                source_stride_bytes: partition.bytes / partition.rows as u64,
                destination_stride_bytes: partition.bytes / partition.rows as u64,
                channel: ((index + 1) as u16) % target.topology.shim_dma_channels.max(1),
                asynchronous: true,
                waits_on: vec![fill.clone()],
            },
        ]);
        let worker = format!("{tag}_worker");
        program.workers.push(XdnaWorker {
            id: worker.clone(),
            tile: partition.tile,
            kernel: kernel.into(),
            inputs: vec![input_fifo],
            outputs: vec![output_fifo],
            waits_on: vec![],
            input_offsets: vec![0],
            output_offsets: vec![0],
        });
        program.sequence.extend([
            RuntimeCommand::Fill { transfer_id: fill },
            RuntimeCommand::Run { worker_id: worker },
            RuntimeCommand::Drain { transfer_id: drain },
        ]);
    }
    program
        .validate()
        .map_err(|errors| AmdNpuLowerError::UnsupportedOp(errors.join("; ")))?;
    Ok(program)
}

pub fn lower_unary_to_native_xdna(
    world: &World,
    op: Entity,
) -> Result<XdnaProgram, AmdNpuLowerError> {
    lower_unary_to_native_xdna_with_target(world, op, &XdnaTarget::xdna2())
}

/// Lower an ECS attention operation with the canonical Q/K/V contract into a
/// native XDNA dataflow program.
pub fn lower_attention_to_native_xdna_with_target(
    world: &World,
    op: Entity,
    target: &XdnaTarget,
) -> Result<XdnaProgram, AmdNpuLowerError> {
    if op_name(world, op).as_deref() != Some("linalg.attention") {
        return Err(AmdNpuLowerError::UnsupportedOp(
            op_name(world, op).unwrap_or_default(),
        ));
    }
    let ops = operands(world, op);
    if ops.len() != 3 {
        return Err(AmdNpuLowerError::MissingOperand(
            "attention requires Q, K, and V".into(),
        ));
    }
    let q = require_tensor(world, ops[0], "attention Q")?;
    let k = require_tensor(world, ops[1], "attention K")?;
    let v = require_tensor(world, ops[2], "attention V")?;
    let result = world
        .get_component::<Results>(op)
        .and_then(|results| results.0.first().copied())
        .ok_or_else(|| AmdNpuLowerError::MissingOperand("attention has no result".into()))?;
    let output = require_tensor(world, result, "attention output")?;
    if q.shape.len() != 2 && q.shape.len() != 3
        || q.shape != k.shape
        || q.shape != v.shape
        || q.shape != output.shape
        || q.element_type != k.element_type
        || q.element_type != v.element_type
        || q.element_type != output.element_type
    {
        return Err(AmdNpuLowerError::MissingType(
            "attention requires matching rank-2/3 Q, K, V, and output tensors".into(),
        ));
    }
    let tile = *target.topology.compute_tiles.first().ok_or_else(|| {
        AmdNpuLowerError::UnsupportedOp("XDNA target has no compute tiles".into())
    })?;
    let element_type = tensor_element_type(&q.element_type)?;
    let bytes = q
        .shape
        .iter()
        .try_fold(element_type.bytes() as u64, |acc, dim| {
            acc.checked_mul(*dim)
        })
        .ok_or_else(|| AmdNpuLowerError::UnsupportedOp("attention tensor size overflow".into()))?;
    let shape_usize = q.shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
    let partitions = target
        .partition_rows_for_buffers(
            &prism_ecs_ir::cimage_types::TensorShape {
                dims: shape_usize.clone(),
            },
            element_type,
            4,
        )
        .map_err(AmdNpuLowerError::UnsupportedOp)?;
    if partitions.len() > 1 {
        return lower_partitioned_attention_program(
            target,
            &shape_usize,
            element_type,
            bytes,
            &partitions,
        );
    }
    if bytes.saturating_mul(4) > target.topology.tile_memory_bytes as u64 {
        return Err(AmdNpuLowerError::UnsupportedOp(
            "attention Q/K/V/output do not fit one XDNA compute tile".into(),
        ));
    }
    let shape = q.shape.iter().map(|dim| *dim as u32).collect::<Vec<_>>();
    let mut buffers = Vec::new();
    let mut fifos = Vec::new();
    let mut transfers = Vec::new();
    let mut sequence = Vec::new();
    for (name, channel) in [("q", 0_u16), ("k", 1_u16), ("v", 2_u16)] {
        let tile_name = format!("{name}_tile");
        buffers.push(XdnaBuffer {
            id: name.into(),
            bytes: bytes as u32,
            element_type,
            shape: shape.clone(),
            memory: XdnaMemory::Host,
            persistent: false,
        });
        buffers.push(XdnaBuffer {
            id: tile_name.clone(),
            bytes: bytes as u32,
            element_type,
            shape: shape.clone(),
            memory: XdnaMemory::TileLocal(tile),
            persistent: false,
        });
        fifos.push(ObjectFifo {
            id: format!("{name}_fifo"),
            element_bytes: element_type.bytes(),
            capacity: 1,
            producer: tile,
            consumer: tile,
            buffer: tile_name.clone(),
        });
        let fill = format!("fill_{name}");
        transfers.push(DmaTransfer {
            id: fill.clone(),
            source: name.into(),
            destination: tile_name,
            bytes: bytes as u32,
            source_offset: 0,
            destination_offset: 0,
            rows: 1,
            source_stride_bytes: 0,
            destination_stride_bytes: 0,
            channel: channel % target.topology.shim_dma_channels.max(1),
            asynchronous: true,
            waits_on: vec![],
        });
        sequence.push(RuntimeCommand::Fill { transfer_id: fill });
    }
    buffers.push(XdnaBuffer {
        id: "output".into(),
        bytes: bytes as u32,
        element_type,
        shape: shape.clone(),
        memory: XdnaMemory::Host,
        persistent: false,
    });
    buffers.push(XdnaBuffer {
        id: "output_tile".into(),
        bytes: bytes as u32,
        element_type,
        shape,
        memory: XdnaMemory::TileLocal(tile),
        persistent: false,
    });
    fifos.push(ObjectFifo {
        id: "output_fifo".into(),
        element_bytes: element_type.bytes(),
        capacity: 1,
        producer: tile,
        consumer: tile,
        buffer: "output_tile".into(),
    });
    transfers.push(DmaTransfer {
        id: "drain_output".into(),
        source: "output_tile".into(),
        destination: "output".into(),
        bytes: bytes as u32,
        source_offset: 0,
        destination_offset: 0,
        rows: 1,
        source_stride_bytes: 0,
        destination_stride_bytes: 0,
        channel: 3 % target.topology.shim_dma_channels.max(1),
        asynchronous: true,
        waits_on: vec!["fill_q".into(), "fill_k".into(), "fill_v".into()],
    });
    sequence.extend([
        RuntimeCommand::Run {
            worker_id: "worker_attention".into(),
        },
        RuntimeCommand::Drain {
            transfer_id: "drain_output".into(),
        },
    ]);
    let program = XdnaProgram {
        topology: target.topology.clone(),
        buffers,
        fifos,
        transfers,
        workers: vec![XdnaWorker {
            id: "worker_attention".into(),
            tile,
            kernel: "prism.xdna.attention".into(),
            inputs: vec!["q_fifo".into(), "k_fifo".into(), "v_fifo".into()],
            outputs: vec!["output_fifo".into()],
            waits_on: vec![],
            input_offsets: vec![0, 0, 0],
            output_offsets: vec![0],
        }],
        barriers: vec![],
        sequence,
    };
    program
        .validate()
        .map_err(|errors| AmdNpuLowerError::UnsupportedOp(errors.join("; ")))?;
    Ok(program)
}

pub fn lower_attention_to_native_xdna(
    world: &World,
    op: Entity,
) -> Result<XdnaProgram, AmdNpuLowerError> {
    lower_attention_to_native_xdna_with_target(world, op, &XdnaTarget::xdna2())
}

fn lower_partitioned_attention_program(
    target: &XdnaTarget,
    shape: &[usize],
    element_type: XdnaElementType,
    total_bytes: u64,
    partitions: &[prism_spatial_ir::xdna_target::XdnaRowPartition],
) -> Result<XdnaProgram, AmdNpuLowerError> {
    let full_shape = shape.iter().map(|dim| *dim as u32).collect::<Vec<_>>();
    let mut program = XdnaProgram {
        topology: target.topology.clone(),
        buffers: vec![],
        fifos: vec![],
        transfers: vec![],
        workers: vec![],
        barriers: vec![],
        sequence: vec![],
    };
    for name in ["q", "k", "v", "output"] {
        program.buffers.push(XdnaBuffer {
            id: name.into(),
            bytes: total_bytes as u32,
            element_type,
            shape: full_shape.clone(),
            memory: XdnaMemory::Host,
            persistent: false,
        });
    }
    for (index, partition) in partitions.iter().enumerate() {
        let tag = format!("attention_{index}");
        let tile_tag = format!("tile_{}_{}", partition.tile.col, partition.tile.row);
        let mut tile_shape = shape.to_vec();
        tile_shape[0] = partition.rows;
        let tile_shape = tile_shape.iter().map(|dim| *dim as u32).collect::<Vec<_>>();
        let mut inputs = Vec::new();
        let mut fills = Vec::new();
        for (name, channel) in [("q", 0_u16), ("k", 1), ("v", 2)] {
            let tile_buffer = format!("{tile_tag}_{name}_tile");
            let fifo = format!("{tag}_{name}_fifo");
            let fill = format!("{tag}_{name}_fill");
            if !program
                .buffers
                .iter()
                .any(|buffer| buffer.id == tile_buffer)
            {
                program.buffers.push(XdnaBuffer {
                    id: tile_buffer.clone(),
                    bytes: partition.bytes as u32,
                    element_type,
                    shape: tile_shape.clone(),
                    memory: XdnaMemory::TileLocal(partition.tile),
                    persistent: false,
                });
            }
            program.fifos.push(ObjectFifo {
                id: fifo.clone(),
                element_bytes: element_type.bytes(),
                capacity: 1,
                producer: partition.tile,
                consumer: partition.tile,
                buffer: tile_buffer.clone(),
            });
            program.transfers.push(DmaTransfer {
                id: fill.clone(),
                source: name.into(),
                destination: tile_buffer,
                bytes: partition.bytes as u32,
                source_offset: partition.byte_offset,
                destination_offset: 0,
                rows: partition.rows as u32,
                source_stride_bytes: partition.bytes / partition.rows as u64,
                destination_stride_bytes: partition.bytes / partition.rows as u64,
                channel: channel % target.topology.shim_dma_channels.max(1),
                asynchronous: true,
                waits_on: vec![],
            });
            inputs.push(fifo);
            fills.push(fill);
        }
        let output_tile = format!("{tile_tag}_output_tile");
        let output_fifo = format!("{tag}_output_fifo");
        let drain = format!("{tag}_drain");
        if !program
            .buffers
            .iter()
            .any(|buffer| buffer.id == output_tile)
        {
            program.buffers.push(XdnaBuffer {
                id: output_tile.clone(),
                bytes: partition.bytes as u32,
                element_type,
                shape: tile_shape,
                memory: XdnaMemory::TileLocal(partition.tile),
                persistent: false,
            });
        }
        program.fifos.push(ObjectFifo {
            id: output_fifo.clone(),
            element_bytes: element_type.bytes(),
            capacity: 1,
            producer: partition.tile,
            consumer: partition.tile,
            buffer: output_tile.clone(),
        });
        program.transfers.push(DmaTransfer {
            id: drain.clone(),
            source: output_tile,
            destination: "output".into(),
            bytes: partition.bytes as u32,
            source_offset: 0,
            destination_offset: partition.byte_offset,
            rows: partition.rows as u32,
            source_stride_bytes: partition.bytes / partition.rows as u64,
            destination_stride_bytes: partition.bytes / partition.rows as u64,
            channel: 3 % target.topology.shim_dma_channels.max(1),
            asynchronous: true,
            waits_on: fills.clone(),
        });
        let worker = format!("{tag}_worker");
        program.workers.push(XdnaWorker {
            id: worker.clone(),
            tile: partition.tile,
            kernel: "prism.xdna.attention".into(),
            inputs,
            outputs: vec![output_fifo],
            waits_on: vec![],
            input_offsets: vec![0, 0, 0],
            output_offsets: vec![0],
        });
        for fill in fills {
            program
                .sequence
                .push(RuntimeCommand::Fill { transfer_id: fill });
        }
        program.sequence.extend([
            RuntimeCommand::Run { worker_id: worker },
            RuntimeCommand::Drain { transfer_id: drain },
        ]);
    }
    program
        .validate()
        .map_err(|errors| AmdNpuLowerError::UnsupportedOp(errors.join("; ")))?;
    Ok(program)
}

/// Unified native ECS-to-XDNA lowering entry point. Unlike
/// [`lower_to_amd_npu`], this returns the executable Prism spatial program
/// rather than a textual description.
pub fn lower_operation_to_native_xdna(
    world: &World,
    op: Entity,
    target: &XdnaTarget,
) -> Result<XdnaProgram, AmdNpuLowerError> {
    match op_name(world, op).as_deref() {
        Some("linalg.matmul" | "linalg.batch_matmul") => {
            lower_matmul_to_native_xdna_with_target(world, op, target)
        }
        Some("linalg.attention") => lower_attention_to_native_xdna_with_target(world, op, target),
        Some("linalg.elementwise" | "linalg.normalization" | "linalg.softmax") => {
            lower_unary_to_native_xdna_with_target(world, op, target)
        }
        Some(name) => Err(AmdNpuLowerError::UnsupportedOp(format!(
            "no native XDNA lowering available for '{name}'"
        ))),
        None => Err(AmdNpuLowerError::UnsupportedOp(
            "operation has no name".into(),
        )),
    }
}

/// Lower any supported root IR operation to AMD XDNA NPU graph IR.
///
/// Dispatches to the appropriate lowering function based on the operation name.
pub fn lower_to_amd_npu(world: &World, root_op: Entity) -> Result<String, AmdNpuLowerError> {
    match op_name(world, root_op).as_deref() {
        Some("linalg.matmul" | "linalg.batch_matmul") => lower_matmul_to_amd_npu(world, root_op),
        Some(name) => Err(AmdNpuLowerError::UnsupportedOp(format!(
            "no AMD XDNA NPU lowering available for '{name}'"
        ))),
        None => Err(AmdNpuLowerError::UnsupportedOp(
            "operation has no name".into(),
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};
    use prism_ecs_ir::ir_types::{FloatKind, TensorType, Type};
    use prism_ecs_ir::op::{OpMarker, OpName, Operands, Results};
    use prism_ecs_ir::value::{Uses, ValueType};

    // ── helpers ───────────────────────────────────────────────────────────

    fn create_matmul_op(world: &mut World, a_ty: Type, b_ty: Type, c_ty: Type) -> Entity {
        let a: Entity = world
            .spawn(EntityKind::Node, Some("A".into()))
            .unwrap()
            .into();
        world.add_component(a, ValueType(a_ty)).unwrap();
        world.add_component(a, Uses(vec![])).unwrap();

        let b: Entity = world
            .spawn(EntityKind::Node, Some("B".into()))
            .unwrap()
            .into();
        world.add_component(b, ValueType(b_ty)).unwrap();
        world.add_component(b, Uses(vec![])).unwrap();

        let c: Entity = world
            .spawn(EntityKind::Node, Some("C".into()))
            .unwrap()
            .into();
        world.add_component(c, ValueType(c_ty.clone())).unwrap();
        world.add_component(c, Uses(vec![])).unwrap();

        let result: Entity = world
            .spawn(EntityKind::Node, Some("result".into()))
            .unwrap()
            .into();
        world
            .add_component(result, ValueType(c_ty.clone()))
            .unwrap();
        world.add_component(result, Uses(vec![])).unwrap();

        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![a, b, c])).unwrap();
        world.add_component(op, Results(vec![result])).unwrap();
        op
    }

    #[test]
    fn lower_unary_softmax_to_native_xdna_program() {
        let mut world = World::new();
        let ty = Type::Tensor(TensorType::new(vec![8, 16], Type::float(FloatKind::F16)));
        let input: Entity = world
            .spawn(EntityKind::Node, Some("input".into()))
            .unwrap()
            .into();
        let output: Entity = world
            .spawn(EntityKind::Node, Some("output".into()))
            .unwrap()
            .into();
        world.add_component(input, ValueType(ty.clone())).unwrap();
        world.add_component(output, ValueType(ty)).unwrap();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("softmax".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.softmax".into()))
            .unwrap();
        world.add_component(op, Operands(vec![input])).unwrap();
        world.add_component(op, Results(vec![output])).unwrap();
        let program = lower_operation_to_native_xdna(&world, op, &XdnaTarget::xdna2())
            .expect("native operation dispatch failed");
        assert_eq!(program.workers[0].kernel, "prism.xdna.softmax");
        assert_eq!(program.transfers.len(), 2);
    }

    #[test]
    fn lower_large_unary_uses_multiple_xdna_tiles() {
        let mut world = World::new();
        let ty = Type::Tensor(TensorType::new(
            vec![5000, 256],
            Type::float(FloatKind::F16),
        ));
        let input: Entity = world
            .spawn(EntityKind::Node, Some("input".into()))
            .unwrap()
            .into();
        let output: Entity = world
            .spawn(EntityKind::Node, Some("output".into()))
            .unwrap()
            .into();
        world.add_component(input, ValueType(ty.clone())).unwrap();
        world.add_component(output, ValueType(ty)).unwrap();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("softmax".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.softmax".into()))
            .unwrap();
        world.add_component(op, Operands(vec![input])).unwrap();
        world.add_component(op, Results(vec![output])).unwrap();
        let program = lower_operation_to_native_xdna(&world, op, &XdnaTarget::xdna2()).unwrap();
        assert!(program.workers.len() > 1);
        assert_eq!(program.workers.len(), program.transfers.len() / 2);
        assert!(program
            .sequence
            .iter()
            .any(|command| matches!(command, RuntimeCommand::Run { .. })));
    }

    #[test]
    fn lower_large_attention_uses_four_buffer_tile_partitions() {
        let mut world = World::new();
        let ty = Type::Tensor(TensorType::new(
            vec![5000, 256],
            Type::float(FloatKind::F16),
        ));
        let mut inputs = Vec::new();
        for name in ["q", "k", "v"] {
            let value: Entity = world
                .spawn(EntityKind::Node, Some(name.into()))
                .unwrap()
                .into();
            world.add_component(value, ValueType(ty.clone())).unwrap();
            inputs.push(value);
        }
        let output: Entity = world
            .spawn(EntityKind::Node, Some("output".into()))
            .unwrap()
            .into();
        world.add_component(output, ValueType(ty)).unwrap();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("attention".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.attention".into()))
            .unwrap();
        world.add_component(op, Operands(inputs)).unwrap();
        world.add_component(op, Results(vec![output])).unwrap();
        let program = lower_operation_to_native_xdna(&world, op, &XdnaTarget::xdna2()).unwrap();
        assert!(program.workers.len() > 1);
        assert_eq!(program.transfers.len(), program.workers.len() * 4);
        assert!(program
            .workers
            .iter()
            .all(|worker| worker.kernel == "prism.xdna.attention"));
    }

    // ── lower_matmul_to_amd_npu ───────────────────────────────────────────

    #[test]
    fn lower_matmul_produces_xdna_graph() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let xdna = lower_matmul_to_amd_npu(&world, op).expect("AMD XDNA lowering failed");

        assert!(xdna.contains("XDNA"), "expected 'XDNA', got:\n{xdna}");
        assert!(xdna.contains("AIE2"), "expected 'AIE2', got:\n{xdna}");
        assert!(xdna.contains("matmul"), "expected 'matmul', got:\n{xdna}");
        assert!(
            xdna.contains("matmul 4x8x16"),
            "expected '4x8x16', got:\n{xdna}"
        );
        assert!(xdna.contains("float16"), "expected 'float16', got:\n{xdna}");
        assert!(xdna.contains("TILING"), "expected 'TILING', got:\n{xdna}");
        assert!(
            xdna.contains("NODE @0: matmul"),
            "expected 'NODE @0: matmul', got:\n{xdna}"
        );
    }

    #[test]
    fn native_lowering_decomposes_oversized_matmul_with_accumulation() {
        let mut world = World::new();
        let f16 = Type::float(FloatKind::F16);
        let op = create_matmul_op(
            &mut world,
            Type::Tensor(TensorType::new(vec![160, 192], f16.clone())),
            Type::Tensor(TensorType::new(vec![192, 224], f16.clone())),
            Type::Tensor(TensorType::new(vec![160, 224], f16)),
        );
        let program = lower_matmul_to_native_xdna(&world, op).expect("tiled lowering failed");
        let workers = program.workers.len();
        assert!(workers > 1, "oversized matmul must decompose");
        assert!(program
            .workers
            .iter()
            .any(|worker| worker.kernel.ends_with("_accumulate")));
        assert!(program
            .buffers
            .iter()
            .filter(|buffer| matches!(buffer.memory, XdnaMemory::TileLocal(_)))
            .all(|buffer| buffer.bytes <= program.topology.tile_memory_bytes));
        assert_eq!(
            program.fifos.len(),
            program.topology.compute_tiles.len() * 3
        );
        assert_eq!(program.barriers.len(), workers);
        assert_eq!(program.sequence.len(), workers * 5);
        assert!(program.workers.iter().all(|worker| {
            worker.inputs.len() == 2
                && worker.outputs.len() == 1
                && worker.inputs[0].starts_with("fifo_A_")
                && worker.inputs[1].starts_with("fifo_B_")
                && worker.outputs[0].starts_with("fifo_C_")
        }));
        assert!(
            program
                .workers
                .iter()
                .map(|worker| worker.tile)
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 1
        );
        assert!(program.validate().is_ok());
    }

    #[test]
    fn native_lowering_allocates_dma_channels_for_xdna1() {
        let mut world = World::new();
        let f16 = Type::float(FloatKind::F16);
        let op = create_matmul_op(
            &mut world,
            Type::Tensor(TensorType::new(vec![32, 32], f16.clone())),
            Type::Tensor(TensorType::new(vec![32, 32], f16.clone())),
            Type::Tensor(TensorType::new(vec![32, 32], f16)),
        );
        let target = XdnaTarget::xdna1();
        let program = lower_matmul_to_native_xdna_with_target(&world, op, &target)
            .expect("XDNA1 lowering failed");
        assert!(program
            .transfers
            .iter()
            .all(|transfer| transfer.channel < target.topology.shim_dma_channels));
        assert!(program.validate().is_ok());
    }

    #[test]
    fn native_lowering_preserves_int8_element_width() {
        let mut world = World::new();
        let i8 = Type::integer(8, prism_ecs_ir::ir_types::Signedness::Signed);
        let op = create_matmul_op(
            &mut world,
            Type::Tensor(TensorType::new(vec![8, 16], i8.clone())),
            Type::Tensor(TensorType::new(vec![16, 8], i8.clone())),
            Type::Tensor(TensorType::new(vec![8, 8], i8)),
        );
        let program = lower_matmul_to_native_xdna(&world, op).expect("int8 lowering failed");
        assert!(program
            .buffers
            .iter()
            .all(|buffer| buffer.element_type == XdnaElementType::Int8));
        assert_eq!(program.buffers[0].bytes, 8 * 16);
        assert!(program.validate().is_ok());
    }

    #[test]
    fn lower_matmul_rejects_non_matmul() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("not_matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.fill".into()))
            .unwrap();

        let err = lower_matmul_to_amd_npu(&world, op).expect_err("should have failed");
        match err {
            AmdNpuLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("linalg.fill"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_rejects_missing_operands() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![])).unwrap();

        let err = lower_matmul_to_amd_npu(&world, op).expect_err("should have failed");
        match err {
            AmdNpuLowerError::MissingOperand(_) => {}
            other => panic!("expected MissingOperand, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_rejects_dimension_mismatch() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        // A[4,8] x B[16,16] — K doesn't match
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![16, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let err = lower_matmul_to_amd_npu(&world, op).expect_err("should fail");
        match err {
            AmdNpuLowerError::MissingType(msg) => {
                assert!(msg.contains("dimension mismatch"));
            }
            other => panic!("expected MissingType, got {other:?}"),
        }
    }

    // ── lower_to_amd_npu ──────────────────────────────────────────────────

    #[test]
    fn lower_to_amd_npu_dispatches_matmul() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let xdna = lower_to_amd_npu(&world, op).expect("AMD XDNA lowering failed");

        assert!(
            xdna.contains("XDNA"),
            "expected 'XDNA' in output, got:\n{xdna}"
        );
        assert!(
            xdna.contains("AIE2"),
            "expected 'AIE2' in output, got:\n{xdna}"
        );
    }

    #[test]
    fn lower_to_amd_npu_rejects_unknown_op() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("bogus".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("unknown.dance".into()))
            .unwrap();

        let err = lower_to_amd_npu(&world, op).expect_err("should fail");
        match err {
            AmdNpuLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("unknown.dance"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }
}
