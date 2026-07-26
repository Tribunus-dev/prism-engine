//! IR representation benchmark: three storage models for MLIR operations within
//! the prism-engine ECS world.
//!
//! Models:
//!   1. Entity-per-op — each MLIR operation/value is a full World entity
//!   2. Entity-per-block — each block is an entity; compact Vec<Op> on block
//!   3. Entity-per-compilation — one entity; ops in a specialized arena resource
//!
//! Benchmarks measure spawn, query, traversal, transaction throughput, and
//! memory overhead.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tribunus_compute_core::prism_ecs_constitutional::{
    ClassifiedComponent, TransientClass, TransientComponent, WorldTxn,
};
use tribunus_compute_core::ecs::{Component, Entity, EntityKind, World, WorldCapacity};

// ─── Component Types ───────────────────────────────────────────────────────

/// Operation kind — minimal payload for realistic allocation cost.
#[derive(Debug, Clone)]
struct OpKind(u32);
impl Component for OpKind {}

/// Operand references for SSA use-def chains using Entity handles from earlier spawns.
#[derive(Debug, Clone)]
struct Operands(Vec<Entity>);
impl Component for Operands {}

/// A result value produced by an operation.
#[derive(Debug, Clone)]
struct Value(u64);
impl Component for Value {}

/// Compact per-block operation descriptor.
#[derive(Debug, Clone)]
struct OpDesc {
    opcode: u32,
    /// Entity handles for SSA use-def (model 1: real Entity refs)
    operands: Vec<Entity>,
    result: u64,
}

/// Stores all ops for a single block entity (model 2).
#[derive(Debug, Clone)]
struct BlockOps(Vec<OpDesc>);
impl Component for BlockOps {}

/// Stores a range of op indices for a compilation (model 3).
#[derive(Debug, Clone)]
struct CompilationRange {
    op_count: usize,
}
impl Component for CompilationRange {}

// ── Transient component type for WorldTxn benchmarks ───────────────────────

#[derive(Debug, Clone)]
struct TxnOp(u32);
impl Component for TxnOp {}
impl ClassifiedComponent for TxnOp {
    type Class = TransientClass;
}
impl TransientComponent for TxnOp {}

// ─── Helpers ───────────────────────────────────────────────────────────────

const CAPACITY: WorldCapacity = WorldCapacity {
    entity_capacity: 1_200_000,
    component_capacity_per_type: 1_200_000,
    resource_capacity: 64,
    journal_capacity: 1_200_000,
};

/// Spawn N entities each with OpKind + Operands + Value components.
/// Returns the list of spawned Entity handles for traversal benchmarks.
fn spawn_entity_per_op(world: &mut World, count: usize) -> Vec<Entity> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let spawned = world.spawn(EntityKind::Node, None).unwrap();
        let ent: Entity = spawned.into();
        world.add_component(ent, OpKind((i % 256) as u32)).unwrap();
        // Each entity (past the first) references its immediate predecessor
        let deps = if i > 0 { vec![out[i - 1]] } else { vec![] };
        world.add_component(ent, Operands(deps)).unwrap();
        world.add_component(ent, Value(i as u64)).unwrap();
        out.push(ent);
    }
    out
}

/// Spawn block_count entities, each containing ops_per_block OpDesc entries.
fn spawn_entity_per_block(world: &mut World, block_count: usize, ops_per_block: usize) {
    for _ in 0..block_count {
        let spawned = world.spawn(EntityKind::Pipeline, None).unwrap();
        let ent: Entity = spawned.into();
        let ops: Vec<OpDesc> = (0..ops_per_block)
            .map(|j| OpDesc {
                opcode: (j % 256) as u32,
                operands: vec![],
                result: j as u64,
            })
            .collect();
        world.add_component(ent, BlockOps(ops)).unwrap();
    }
}

/// Spawn a single compilation entity.
fn spawn_entity_per_compilation(world: &mut World, count: usize) {
    let spawned = world.spawn(EntityKind::Session, None).unwrap();
    let ent: Entity = spawned.into();
    world
        .add_component(ent, CompilationRange { op_count: count })
        .unwrap();
}

// ─── Spawn Throughput ──────────────────────────────────────────────────────

fn bench_spawn_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn/10k");
    group.sample_size(10);

    group.bench_function("entity_per_op", |b| {
        b.iter_with_setup(
            || {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                w
            },
            |mut w| {
                for i in 0..10_000 {
                    let spawned = w.spawn(EntityKind::Node, None).unwrap();
                    let ent: Entity = spawned.into();
                    w.add_component(ent, OpKind((i % 256) as u32)).unwrap();
                    w.add_component(ent, Value(i as u64)).unwrap();
                }
                black_box(w.entity_count());
            },
        )
    });

    group.bench_function("entity_per_block", |b| {
        b.iter_with_setup(
            || {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                w
            },
            |mut w| {
                // 100 blocks × 100 ops each
                for _ in 0..100 {
                    let spawned = w.spawn(EntityKind::Pipeline, None).unwrap();
                    let ent: Entity = spawned.into();
                    let ops: Vec<OpDesc> = (0..100)
                        .map(|j| OpDesc {
                            opcode: (j % 256) as u32,
                            operands: vec![],
                            result: j as u64,
                        })
                        .collect();
                    w.add_component(ent, BlockOps(ops)).unwrap();
                }
                black_box(w.entity_count());
            },
        )
    });

    group.bench_function("entity_per_compilation", |b| {
        b.iter_with_setup(
            || {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                w
            },
            |mut w| {
                let spawned = w.spawn(EntityKind::Session, None).unwrap();
                let ent: Entity = spawned.into();
                w.add_component(ent, CompilationRange { op_count: 10_000 })
                    .unwrap();
                black_box(w.entity_count());
            },
        )
    });

    group.finish();
}

fn bench_spawn_100k(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn/100k");
    group.sample_size(10);

    group.bench_function("entity_per_op", |b| {
        b.iter_with_setup(
            || {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                w
            },
            |mut w| {
                for i in 0..100_000 {
                    let spawned = w.spawn(EntityKind::Node, None).unwrap();
                    let ent: Entity = spawned.into();
                    w.add_component(ent, OpKind((i % 256) as u32)).unwrap();
                    w.add_component(ent, Value(i as u64)).unwrap();
                }
                black_box(w.entity_count());
            },
        )
    });

    group.bench_function("entity_per_block", |b| {
        b.iter_with_setup(
            || {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                w
            },
            |mut w| {
                for _ in 0..1000 {
                    let spawned = w.spawn(EntityKind::Pipeline, None).unwrap();
                    let ent: Entity = spawned.into();
                    let ops: Vec<OpDesc> = (0..100)
                        .map(|j| OpDesc {
                            opcode: (j % 256) as u32,
                            operands: vec![],
                            result: j as u64,
                        })
                        .collect();
                    w.add_component(ent, BlockOps(ops)).unwrap();
                }
                black_box(w.entity_count());
            },
        )
    });

    group.bench_function("entity_per_compilation", |b| {
        b.iter_with_setup(
            || {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                w
            },
            |mut w| {
                let spawned = w.spawn(EntityKind::Session, None).unwrap();
                let ent: Entity = spawned.into();
                w.add_component(ent, CompilationRange { op_count: 100_000 })
                    .unwrap();
                black_box(w.entity_count());
            },
        )
    });

    group.finish();
}

fn bench_spawn_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn/1M");
    group.sample_size(10);

    group.bench_function("entity_per_op", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                let start = std::time::Instant::now();
                for i in 0..1_000_000 {
                    let spawned = w.spawn(EntityKind::Node, None).unwrap();
                    let ent: Entity = spawned.into();
                    w.add_component(ent, OpKind((i % 256) as u32)).unwrap();
                    w.add_component(ent, Value(i as u64)).unwrap();
                }
                total += start.elapsed();
                black_box(w.entity_count());
            }
            total
        })
    });

    group.bench_function("entity_per_block", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                let start = std::time::Instant::now();
                for _ in 0..10_000 {
                    let spawned = w.spawn(EntityKind::Pipeline, None).unwrap();
                    let ent: Entity = spawned.into();
                    let ops: Vec<OpDesc> = (0..100)
                        .map(|j| OpDesc {
                            opcode: (j % 256) as u32,
                            operands: vec![],
                            result: j as u64,
                        })
                        .collect();
                    w.add_component(ent, BlockOps(ops)).unwrap();
                }
                total += start.elapsed();
                black_box(w.entity_count());
            }
            total
        })
    });

    group.bench_function("entity_per_compilation", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                let start = std::time::Instant::now();
                let spawned = w.spawn(EntityKind::Session, None).unwrap();
                let ent: Entity = spawned.into();
                w.add_component(
                    ent,
                    CompilationRange {
                        op_count: 1_000_000,
                    },
                )
                .unwrap();
                total += start.elapsed();
                black_box(w.entity_count());
            }
            total
        })
    });

    group.finish();
}

// ─── Component Query Latency ───────────────────────────────────────────────

fn bench_query_single(c: &mut Criterion) {
    let mut w = World::with_capacity(&CAPACITY);
    w.set_direct_mutation_allowed(true);
    spawn_entity_per_op(&mut w, 100_000);

    c.benchmark_group("query/single_component")
        .sample_size(10)
        .bench_function("entity_per_op", |b| {
            b.iter(|| {
                let mut count = 0usize;
                for (_, v) in w.query::<OpKind>() {
                    black_box(&v);
                    count += 1;
                }
                black_box(count);
            })
        });

    let mut w2 = World::with_capacity(&CAPACITY);
    w2.set_direct_mutation_allowed(true);
    spawn_entity_per_block(&mut w2, 1000, 100);

    c.benchmark_group("query/single_component")
        .sample_size(10)
        .bench_function("entity_per_block", |b| {
            b.iter(|| {
                let mut count = 0usize;
                for (_, v) in w2.query::<BlockOps>() {
                    black_box(&v);
                    count += 1;
                }
                black_box(count);
            })
        });
}

fn bench_query_two(c: &mut Criterion) {
    let mut w = World::with_capacity(&CAPACITY);
    w.set_direct_mutation_allowed(true);
    spawn_entity_per_op(&mut w, 100_000);

    c.benchmark_group("query/two_component")
        .sample_size(10)
        .bench_function("entity_per_op", |b| {
            b.iter(|| {
                let mut count = 0usize;
                for (_, v, val) in w.query2::<OpKind, Value>() {
                    black_box((&v, &val));
                    count += 1;
                }
                black_box(count);
            })
        });
}

fn bench_query_three(c: &mut Criterion) {
    let mut w = World::with_capacity(&CAPACITY);
    w.set_direct_mutation_allowed(true);
    spawn_entity_per_op(&mut w, 100_000);

    c.benchmark_group("query/three_component")
        .sample_size(10)
        .bench_function("entity_per_op", |b| {
            b.iter(|| {
                let mut count = 0usize;
                for (_, v, o, val) in w.query3::<OpKind, Operands, Value>() {
                    black_box((&v, &o, &val));
                    count += 1;
                }
                black_box(count);
            })
        });
}

// ─── SSA Use-def Traversal ─────────────────────────────────────────────────

fn bench_traversal_1k(c: &mut Criterion) {
    let mut w = World::with_capacity(&CAPACITY);
    w.set_direct_mutation_allowed(true);
    spawn_entity_per_op(&mut w, 1_000);

    c.benchmark_group("traversal/1K")
        .sample_size(10)
        .bench_function("entity_per_op", |b| {
            b.iter(|| {
                let mut chain_count = 0usize;
                for (_entity, ops) in w.query::<Operands>() {
                    for op_ent in &ops.0 {
                        if let Ok(eref) = w.entity_ref(*op_ent) {
                            black_box(eref.entity);
                            chain_count += 1;
                        }
                    }
                }
                black_box(chain_count);
            })
        });
}

fn bench_traversal_10k(c: &mut Criterion) {
    let mut w = World::with_capacity(&CAPACITY);
    w.set_direct_mutation_allowed(true);
    spawn_entity_per_op(&mut w, 10_000);

    c.benchmark_group("traversal/10K")
        .sample_size(10)
        .bench_function("entity_per_op", |b| {
            b.iter(|| {
                let mut chain_count = 0usize;
                for (_entity, ops) in w.query::<Operands>() {
                    for op_ent in &ops.0 {
                        if let Ok(eref) = w.entity_ref(*op_ent) {
                            black_box(eref.entity);
                            chain_count += 1;
                        }
                    }
                }
                black_box(chain_count);
            })
        });
}

fn bench_traversal_100k(c: &mut Criterion) {
    let mut w = World::with_capacity(&CAPACITY);
    w.set_direct_mutation_allowed(true);
    spawn_entity_per_op(&mut w, 100_000);

    c.benchmark_group("traversal/100K")
        .sample_size(10)
        .bench_function("entity_per_op", |b| {
            b.iter(|| {
                let mut chain_count = 0usize;
                for (_entity, ops) in w.query::<Operands>() {
                    for op_ent in &ops.0 {
                        if let Ok(eref) = w.entity_ref(*op_ent) {
                            black_box(eref.entity);
                            chain_count += 1;
                        }
                    }
                }
                black_box(chain_count);
            })
        });
}

// ─── Transaction Throughput ────────────────────────────────────────────────
//
// Spawn 10K ops via direct mutation, switch to TransactionalOnly, then
// commit a WorldTxn that spawns 5K replacement ops with TransientComponent.

fn bench_transaction_replace_50pct(c: &mut Criterion) {
    c.benchmark_group("transaction/replace_50pct_10K")
        .sample_size(10)
        .bench_function("entity_per_op", |b| {
            b.iter_custom(|iters| {
                let mut total_dur = std::time::Duration::ZERO;
                for _ in 0..iters {
                    // Build world with 10K ops
                    let mut w = World::with_capacity(&CAPACITY);
                    w.set_direct_mutation_allowed(true);
                    let entities: Vec<Entity> = spawn_entity_per_op(&mut w, 10_000);
                    let half = entities.len() / 2;

                    // Switch to transactional-only mode
                    w.set_direct_mutation_allowed(false);

                    // Build a WorldTxn that spawns 5K replacement ops
                    let start = std::time::Instant::now();
                    let mut txn = WorldTxn::new(&w);
                    for _i in 0..half {
                        // Use spawn_pending + put_transient_pending to batch allocate
                        // through the transaction machinery
                        let pending = txn.spawn_pending(EntityKind::Node);
                        txn.put_transient_pending(pending, TxnOp(42));
                    }
                    // Commit the transaction atomically
                    let _receipt = w.transit(txn).unwrap();
                    total_dur += start.elapsed();
                    black_box(w.entity_count());
                }
                total_dur
            })
        });
}

// ─── Memory Overhead ───────────────────────────────────────────────────────
//
// Uses rusage-based RSS estimation to measure per-entity memory cost.
// rusage measures the total resident set size; we sample before and after
// spawning N entities and report the delta.

fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/overhead_via_rusage");
    group.sample_size(10);

    fn rss_bytes() -> i64 {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if ret == 0 {
            // ru_maxrss is in bytes on macOS
            usage.ru_maxrss
        } else {
            -1
        }
    }

    group.bench_function("entity_per_op_100K", |b| {
        b.iter_custom(|iters| {
            let mut total_duration = std::time::Duration::ZERO;
            for _ in 0..iters {
                let before = rss_bytes();
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                let start = std::time::Instant::now();
                spawn_entity_per_op(&mut w, 100_000);
                let elapsed = start.elapsed();
                let after = rss_bytes();
                total_duration += elapsed;
                black_box((w.entity_count(), before, after));
            }
            total_duration
        })
    });

    group.bench_function("entity_per_block_100K", |b| {
        b.iter_custom(|iters| {
            let mut total_duration = std::time::Duration::ZERO;
            for _ in 0..iters {
                let before = rss_bytes();
                let mut w = World::with_capacity(&CAPACITY);
                w.set_direct_mutation_allowed(true);
                let start = std::time::Instant::now();
                spawn_entity_per_block(&mut w, 1000, 100);
                let elapsed = start.elapsed();
                let after = rss_bytes();
                total_duration += elapsed;
                black_box((w.entity_count(), before, after));
            }
            total_duration
        })
    });

    group.finish();
}

// ─── Criterion Setup ───────────────────────────────────────────────────────

criterion_group! {
    name = ir_representation;
    config = Criterion::default().sample_size(10);
    targets =
        bench_spawn_10k,
        bench_spawn_100k,
        bench_spawn_1m,
        bench_query_single,
        bench_query_two,
        bench_query_three,
        bench_traversal_1k,
        bench_traversal_10k,
        bench_traversal_100k,
        bench_transaction_replace_50pct,
        bench_memory_overhead,
}
criterion_main!(ir_representation);
