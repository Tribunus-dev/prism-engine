// Acceptance tests for the deterministic schedule compiler.
//
// Each gate from the Slice 1 specification.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::runtime::scheduling::component_id::{
    ComponentId, ComponentMask, ComponentRegistry, ResourceId, ResourceMask,
    SchedulableComponent, SchedulableResource,
};
use crate::runtime::scheduling::error::MaskError;
use crate::runtime::scheduling::error::{RegistryError, ScheduleError};
use crate::runtime::scheduling::manifest::MANIFEST_SCHEMA_VERSION;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult,
};
use crate::runtime::scheduling::schedule::Schedule;
use crate::runtime::world::World;
use crate::runtime::scheduling::command::CommandWriter;

use lazy_static::lazy_static;

// -----------------------------------------------------------------------
//  Test component types for mask/registry tests
// -----------------------------------------------------------------------

struct TestCompA;
impl SchedulableComponent for TestCompA {
    const COMPONENT_ID: ComponentId = 0;
    const NAME: &'static str = "TestCompA";
}

#[allow(dead_code)]
struct TestCompB;
impl SchedulableComponent for TestCompB {
    const COMPONENT_ID: ComponentId = 1;
    const NAME: &'static str = "TestCompB";
}

#[allow(dead_code)]
struct SharedReader;
impl SchedulableComponent for SharedReader {
    const COMPONENT_ID: ComponentId = 3;
    const NAME: &'static str = "SharedReader";
}

#[allow(dead_code)]
struct TestRes;
impl SchedulableResource for TestRes {
    const RESOURCE_ID: ResourceId = 0;
    const NAME: &'static str = "TestRes";
}

// -----------------------------------------------------------------------
//  SysWrapper — simple ErasedSystem for test metadata
// -----------------------------------------------------------------------

struct SysWrapper(&'static SystemMetadata, AtomicU32);

impl ErasedSystem for SysWrapper {
    fn metadata(&self) -> &SystemMetadata {
        self.0
    }
    fn run(
        &mut self,
        _world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        self.1.fetch_add(1, Ordering::SeqCst);
        SystemResult::ok()
    }
}

// -----------------------------------------------------------------------
//  Metadata registry (leaked for 'static lifetime in tests)
// -----------------------------------------------------------------------

lazy_static! {
    static ref TEST_META_REGISTRY: std::sync::Mutex<Vec<&'static SystemMetadata>> =
        std::sync::Mutex::new(Vec::new());
}

fn register_meta(meta: SystemMetadata) -> &'static SystemMetadata {
    let leaked: &'static SystemMetadata = Box::leak(Box::new(meta));
    TEST_META_REGISTRY.lock().unwrap().push(leaked);
    leaked
}

fn make_sys(meta: &'static SystemMetadata) -> Box<dyn ErasedSystem> {
    Box::new(SysWrapper(meta, AtomicU32::new(0)))
}

// ===================================================================
//  1. Component registry rejects duplicate IDs and out-of-range IDs
// ===================================================================

#[test]
fn gate01_registry_rejects_duplicate_id() {
    let mut reg = ComponentRegistry::new();
    reg.register::<TestCompA>().unwrap();
    let err = reg.register::<TestCompA>().unwrap_err();
    assert!(matches!(err, RegistryError::ComponentIdCollision(0, _, _)));
}

#[test]
fn gate01_registry_rejects_out_of_range() {
    let mut m = ComponentMask::empty();
    let err = m.insert(300).unwrap_err();
    assert!(matches!(err, MaskError::OutOfRange { id: 300, .. }));
}

// ===================================================================
//  2. Duplicate SystemId and name rejected
// ===================================================================

#[test]
fn gate02_duplicate_system_id_rejected() {
    let a = register_meta(SystemMetadata {
        id: SystemId(1), name: "sys_a", stage: Stage::Intake, order: 0,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Reject,
    });
    let b = register_meta(SystemMetadata {
        id: SystemId(1), name: "sys_b", stage: Stage::Intake, order: 1,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Reject,
    });
    let err = Schedule::compile(vec![make_sys(a), make_sys(b)]).unwrap_err();
    assert!(matches!(err, ScheduleError::SystemIdCollision(SystemId(1))));
}

#[test]
fn gate02_duplicate_name_rejected() {
    let a = register_meta(SystemMetadata {
        id: SystemId(1), name: "same", stage: Stage::Intake, order: 0,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Reject,
    });
    let b = register_meta(SystemMetadata {
        id: SystemId(2), name: "same", stage: Stage::Intake, order: 1,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Reject,
    });
    let err = Schedule::compile(vec![make_sys(a), make_sys(b)]).unwrap_err();
    assert!(matches!(err, ScheduleError::SystemNameCollision("same")));
}

// ===================================================================
//  3. Stage-inverting explicit edges rejected
// ===================================================================

#[test]
fn gate03_stage_inversion_rejected() {
    let decode = register_meta(SystemMetadata {
        id: SystemId(10), name: "decode", stage: Stage::Decode, order: 0,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[SystemId(11)], // decode after receipt — inversion!
        before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    });
    let receipt = register_meta(SystemMetadata {
        id: SystemId(11), name: "receipt", stage: Stage::Receipt, order: 0,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    });
    let err = Schedule::compile(vec![make_sys(decode), make_sys(receipt)]);
    assert!(err.is_err());
}

// ===================================================================
//  4. Cycle detection with path diagnostics
// ===================================================================

#[test]
fn gate04_cycle_detected() {
    let a = register_meta(SystemMetadata {
        id: SystemId(20), name: "cycle_a", stage: Stage::Intake, order: 0,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[SystemId(22)], before: &[],
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    });
    let b = register_meta(SystemMetadata {
        id: SystemId(21), name: "cycle_b", stage: Stage::Intake, order: 0,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[SystemId(20)], before: &[],
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    });
    let c = register_meta(SystemMetadata {
        id: SystemId(22), name: "cycle_c", stage: Stage::Intake, order: 0,
        reads: ComponentMask::empty(), writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[SystemId(21)], before: &[],
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    });
    let err = Schedule::compile(vec![make_sys(a), make_sys(b), make_sys(c)]);
    assert!(err.is_err());
    match err.unwrap_err() {
        ScheduleError::CycleDetected(path) => {
            assert!(!path.is_empty(), "cycle path must not be empty");
        }
        other => panic!("expected CycleDetected, got: {other}"),
    }
}

// ===================================================================
//  5. Undeclared write/write overlap rejected
// ===================================================================

#[test]
fn gate05_undeclared_write_overlap_rejected() {
    let mut w = ComponentMask::empty();
    w.insert(0).unwrap();
    let a = register_meta(SystemMetadata {
        id: SystemId(30), name: "writer_a", stage: Stage::Intake, order: 0,
        reads: ComponentMask::empty(), writes: w,
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Reject,
    });
    let b = register_meta(SystemMetadata {
        id: SystemId(31), name: "writer_b", stage: Stage::Intake, order: 0,
        reads: ComponentMask::empty(), writes: w,
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Reject,
    });
    let err = Schedule::compile(vec![make_sys(a), make_sys(b)]);
    assert!(err.is_err());
    assert!(matches!(err.unwrap_err(), ScheduleError::IllegalHazard { .. }));
}

// ===================================================================
//  6. Shared reads never create ordering edges
// ===================================================================

#[test]
fn gate06_shared_reads_no_edges() {
    let mut r = ComponentMask::empty();
    r.insert(3).unwrap();
    let a = register_meta(SystemMetadata {
        id: SystemId(40), name: "reader_a", stage: Stage::Intake, order: 0,
        reads: r, writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Commutative,
    });
    let b = register_meta(SystemMetadata {
        id: SystemId(41), name: "reader_b", stage: Stage::Intake, order: 0,
        reads: r, writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::Commutative,
    });
    let sched = Schedule::compile(vec![make_sys(a), make_sys(b)]);
    assert!(sched.is_ok(), "shared reads should compile");
    let schedule = sched.unwrap();
    let manifest = schedule.manifest();
    assert_eq!(manifest.system_count, 2);
    assert_eq!(manifest.schema_version, MANIFEST_SCHEMA_VERSION);
}

// ===================================================================
//  7. Declared producer-consumer dependencies
// ===================================================================

#[test]
fn gate07_producer_consumer_respected() {
    let mut w = ComponentMask::empty();
    w.insert(0).unwrap();
    let mut r = ComponentMask::empty();
    r.insert(0).unwrap();
    let prod = register_meta(SystemMetadata {
        id: SystemId(50), name: "producer", stage: Stage::Prefill, order: 0,
        reads: ComponentMask::empty(), writes: w,
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[], before: &[], execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    });
    let cons = register_meta(SystemMetadata {
        id: SystemId(51), name: "consumer", stage: Stage::Prefill, order: 0,
        reads: r, writes: ComponentMask::empty(),
        reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
        after: &[SystemId(50)], before: &[], // explicit after producer
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    });
    let sched = Schedule::compile(vec![make_sys(prod), make_sys(cons)]).unwrap();
    let order = &sched.manifest().execution_order;
    let p = order.iter().position(|id| *id == SystemId(50)).unwrap();
    let c = order.iter().position(|id| *id == SystemId(51)).unwrap();
    assert!(p < c, "producer must run before consumer");
}

// ===================================================================
//  8. Manifest byte-identical across repeated compilations
// ===================================================================

#[test]
fn gate08_manifest_reproducible() {
    let sys = |id: u32, name: &'static str, stage: Stage| {
        register_meta(SystemMetadata {
            id: SystemId(id), name, stage, order: 0,
            reads: ComponentMask::empty(), writes: ComponentMask::empty(),
            reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
            after: &[], before: &[], execution_class: ExecutionClass::Serial,
            serialization: SerializationPolicy::Reject,
        })
    };
    let meta = vec![
        sys(100, "a", Stage::Intake),
        sys(101, "b", Stage::Decode),
        sys(102, "c", Stage::Maintenance),
        sys(103, "d", Stage::Receipt),
    ];
    let systems_a: Vec<_> = meta.iter().map(|m| make_sys(m)).collect();
    let systems_b: Vec<_> = meta.iter().map(|m| make_sys(m)).collect();
    let sched_a = Schedule::compile(systems_a).unwrap();
    let sched_b = Schedule::compile(systems_b).unwrap();
    assert_eq!(sched_a.manifest().digest, sched_b.manifest().digest);
}

// ===================================================================
//  9. Schedule::run does not panic with multiple stages
// ===================================================================

#[test]
fn gate09_schedule_run_no_panic() {
    let sys = |id: u32, name: &'static str, stage: Stage| {
        register_meta(SystemMetadata {
            id: SystemId(id), name, stage, order: 0,
            reads: ComponentMask::empty(), writes: ComponentMask::empty(),
            reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
            after: &[], before: &[], execution_class: ExecutionClass::Serial,
            serialization: SerializationPolicy::Commutative,
        })
    };
    let systems = vec![
        make_sys(sys(200, "intake", Stage::Intake)),
        make_sys(sys(201, "prefill", Stage::Prefill)),
        make_sys(sys(202, "decode", Stage::Decode)),
        make_sys(sys(203, "maintenance", Stage::Maintenance)),
        make_sys(sys(204, "receipt", Stage::Receipt)),
    ];
    let mut schedule = Schedule::compile(systems).unwrap();
    let mut world = World::default();
    let results = schedule.run(&mut world);
    assert_eq!(results.len(), 5);
    for (id, r) in &results {
        assert!(matches!(r, SystemResult::Ok), "system {id:?} failed");
    }
}

// ===================================================================
//  10. Stage ordering enforced
// ===================================================================

#[test]
fn gate10_stage_ordering_enforced() {
    let sys = |id: u32, name: &'static str, stage: Stage| {
        register_meta(SystemMetadata {
            id: SystemId(id), name, stage, order: 0,
            reads: ComponentMask::empty(), writes: ComponentMask::empty(),
            reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
            after: &[], before: &[], execution_class: ExecutionClass::Serial,
            serialization: SerializationPolicy::Commutative,
        })
    };
    let systems = vec![
        make_sys(sys(300, "receipt", Stage::Receipt)),
        make_sys(sys(301, "decode", Stage::Decode)),
        make_sys(sys(302, "intake", Stage::Intake)),
        make_sys(sys(303, "prefill", Stage::Prefill)),
    ];
    let schedule = Schedule::compile(systems).unwrap();
    let order = &schedule.manifest().execution_order;
    let i = order.iter().position(|id| *id == SystemId(302)).unwrap();
    let p = order.iter().position(|id| *id == SystemId(303)).unwrap();
    let d = order.iter().position(|id| *id == SystemId(301)).unwrap();
    let r = order.iter().position(|id| *id == SystemId(300)).unwrap();
    assert!(i < p, "intake before prefill");
    assert!(p < d, "prefill before decode");
    assert!(d < r, "decode before receipt");
}

// ===================================================================
//  StableOrder hazard resolution deterministic
// ===================================================================

#[test]
fn stable_order_hazard_deterministic() {
    let mut w = ComponentMask::empty();
    w.insert(0).unwrap();
    let make = |id: u32, name: &'static str| {
        register_meta(SystemMetadata {
            id: SystemId(id), name, stage: Stage::Intake, order: 0,
            reads: ComponentMask::empty(), writes: w,
            reads_resources: ResourceMask::empty(), writes_resources: ResourceMask::empty(),
            after: &[], before: &[], execution_class: ExecutionClass::Serial,
            serialization: SerializationPolicy::StableOrder,
        })
    };
    let a = make(400, "stable_a");
    let b = make(401, "stable_b");
    let s1 = Schedule::compile(vec![make_sys(a), make_sys(b)]).unwrap();
    let s2 = Schedule::compile(vec![make_sys(a), make_sys(b)]).unwrap();
    assert_eq!(s1.manifest().execution_order, s2.manifest().execution_order);
}
