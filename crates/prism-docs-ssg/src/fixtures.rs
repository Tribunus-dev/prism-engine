//! Hardcoded fixtures for the interactive pages.
//!
//! The capabilities page, demo page, and projection-repro page
//! carry example data that drives the filterable grid, the
//! workflow stages, and the SVG canvas. The data lives here
//! because it is illustrative, not authoritative content.
//!
//! When a future change introduces a typed entity kind for
//! these (e.g., `CapabilityCard`), this module disappears and
//! the manifest becomes the source. For now, the fixtures
//! populate the world at SSG bootstrap so the renderers have
//! data to project.

use prism_docs_runtime::components::capability::{
    CapabilityBody, CapabilityClass, CapabilityDomain, CapabilityId, CapabilityLimitation,
    CapabilitySourcePath, CapabilityState, CapabilityTitle,
};
use prism_docs_runtime::components::demo::{
    DemoBandBody, DemoBandStatus, DemoBandTitle, DemoGateBody, DemoGateId, DemoGateNum,
    DemoGateOrder, DemoGateTitle,
};
use prism_docs_runtime::components::projection::{
    ProjectionLayer, ProjectionLayers, ProjectionStageId, ProjectionStageLabel,
    ProjectionStageOrder, ProjectionSubjectId, ProjectionSubjectKind, ProjectionSubjectName,
};
use prism_ecs_core::{Entity, EntityKind, World};

/// Insert the capability cards into the world. Returns the
/// number of cards inserted.
pub fn insert_capability_cards(world: &mut World) -> usize {
    let cards: Vec<(&str, &str, &str, &str, &str, &str, &str, &str)> = vec![
        (
            "cap:replay",
            "Replay",
            "runtime",
            "Capture, run, minimize, import, export, and compare replay bundles.",
            "verified",
            "architectural",
            "crates/prism-mcp-replay/src/lib.rs",
            "Restart recovery proven for in-memory and file-backed stores.",
        ),
        (
            "cap:provenance",
            "Provenance",
            "evidence",
            "Receipts carry commit, build, model, target, inputs, route, conformance gates.",
            "verified",
            "architectural",
            "evidence-schema/",
            "Schema version 1.0; canonical keys typed as newtypes.",
        ),
        (
            "cap:experiments",
            "Experiments",
            "evidence",
            "Create, run, cancel, resume, compare, and promote experiments.",
            "verified",
            "architectural",
            "crates/prism-mcp-lab/src/lib.rs",
            "Promotion is a typed operation; comparison preserves provenance.",
        ),
        (
            "cap:regression",
            "Regression",
            "evidence",
            "Plan benchmarks, compare evidence, detect regressions, promote baselines.",
            "compile-verified",
            "architectural",
            "crates/prism-mcp-bench/src/lib.rs",
            "Compile-verified path; hardware-specific baselines pending.",
        ),
        (
            "cap:metal-runtime",
            "Apple Silicon / Metal",
            "runtime",
            "Metal dispatch on Apple Silicon; ANE prefill; CoreAudio streams.",
            "measured",
            "compile-verified",
            "crates/prism-metal-runtime/src/",
            "Tok/s measurements pending a hardware-bound run.",
        ),
        (
            "cap:rocm-runtime",
            "MI300X / ROCm-HIP",
            "runtime",
            "ROCm/HIP dispatch on AMD MI300X; gfx942-oriented validation.",
            "compile-verified",
            "compile-verified",
            "crates/prism-rocm-runtime/src/",
            "Compile-verified; device-evidence on the silicon pending.",
        ),
        (
            "cap:xdna-runtime",
            "XDNA / XDNA2",
            "runtime",
            "Tile, FIFO, DMA, barrier, and resource legalization for spatial plans.",
            "compile-verified",
            "compile-verified",
            "crates/prism-amd-npu-runtime/src/",
            "Compile-verified planning; hardware execution requires XDNA-capable system.",
        ),
        (
            "cap:ane-runtime",
            "Apple Neural Engine",
            "runtime",
            "ANE prefill for the Apple path.",
            "compile-verified",
            "compile-verified",
            "crates/prism-ane/src/",
            "Planned; not yet validated end-to-end.",
        ),
        (
            "cap:ternary",
            "Ternary distillation",
            "compiler",
            "Progressive ternarization with quality and admission gates.",
            "compile-verified",
            "compile-verified",
            "crates/prism-ecs-compile/src/",
            "Compile-verified; resumable; constitutional admission applies.",
        ),
        (
            "cap:mixed-precision",
            "Mixed precision",
            "compiler",
            "Mixed-precision candidates searched against the quality contract.",
            "verified",
            "architectural",
            "crates/prism-ecs-quantization/src/",
            "Implemented; quality contract is constitutional.",
        ),
        (
            "cap:cimage",
            "ComputeImage",
            "artifact",
            "Six-strata deployment artifact with typed identity and binding ABI.",
            "verified",
            "architectural",
            "crates/prism-ecs-artifact/src/",
            "Implemented; ABI v1 documented in docs/cimage-layout-abi-v1.md.",
        ),
        (
            "cap:kv-policy",
            "KV-cache policy",
            "compiler",
            "KV-cache search, compression, and ownership as a constitutional concern.",
            "compile-verified",
            "illustrative",
            "crates/prism-kv-cache/src/",
            "Compile-verified; full runtime integration hardening in progress.",
        ),
    ];

    let mut count = 0;
    for (id, title, domain, body, state, class, source, limitation) in &cards {
        let entity: Entity = world
            .spawn(EntityKind::Node, None)
            .map(|s| s.entity.into())
            .expect("spawn capability");
        world
            .add_component(entity, CapabilityId(id.to_string()))
            .expect("add CapabilityId");
        world
            .add_component(entity, CapabilityTitle(title.to_string()))
            .expect("add CapabilityTitle");
        world
            .add_component(entity, CapabilityDomain(domain.to_string()))
            .expect("add CapabilityDomain");
        world
            .add_component(entity, CapabilityBody(body.to_string()))
            .expect("add CapabilityBody");
        world
            .add_component(entity, CapabilityState(state.to_string()))
            .expect("add CapabilityState");
        world
            .add_component(entity, CapabilityClass(class.to_string()))
            .expect("add CapabilityClass");
        world
            .add_component(entity, CapabilitySourcePath(source.to_string()))
            .expect("add CapabilitySourcePath");
        world
            .add_component(entity, CapabilityLimitation(limitation.to_string()))
            .expect("add CapabilityLimitation");
        count += 1;
    }
    count
}

/// Insert the demo workflow gates and milestone bands.
pub fn insert_demo_data(world: &mut World) -> usize {
    let gates: Vec<(&str, &str, &str, &str, u32)> = vec![
        ("gate:ingest", "01", "Ingest", "GGUF or SafeTensors, identity + digest", 1),
        ("gate:compile", "02", "Compile", "PrismIR, representation search", 2),
        ("gate:realize", "03", "Realize", "ComputeImage, .cimage artifact", 3),
        ("gate:prove", "04", "Prove", "Metal run, receipt + replay fields", 4),
    ];

    let mut count = 0;
    for (id, num, title, body, order) in &gates {
        let entity: Entity = world
            .spawn(EntityKind::Node, None)
            .map(|s| s.entity.into())
            .expect("spawn demo gate");
        world
            .add_component(entity, DemoGateId(id.to_string()))
            .expect("add DemoGateId");
        world
            .add_component(entity, DemoGateNum(num.to_string()))
            .expect("add DemoGateNum");
        world
            .add_component(entity, DemoGateTitle(title.to_string()))
            .expect("add DemoGateTitle");
        world
            .add_component(entity, DemoGateBody(body.to_string()))
            .expect("add DemoGateBody");
        world
            .add_component(entity, DemoGateOrder(*order))
            .expect("add DemoGateOrder");
        count += 1;
    }

    let bands: Vec<(&str, &str, &str)> = vec![
        ("ready", "Self-contained launch", "The application installs and starts without asking the reader to recreate the development environment."),
        ("active", "Observable compilation", "The model source, ComputeImage fields, target route, and validation state are visible as one connected workflow."),
        ("gated", "Reproducible evidence", "A recorded run can be repeated on the supported machine and produces versioned receipts rather than illustrative dashboard values."),
    ];

    for (status, title, body) in &bands {
        let entity: Entity = world
            .spawn(EntityKind::Node, None)
            .map(|s| s.entity.into())
            .expect("spawn demo band");
        world
            .add_component(entity, DemoBandTitle(title.to_string()))
            .expect("add DemoBandTitle");
        world
            .add_component(entity, DemoBandBody(body.to_string()))
            .expect("add DemoBandBody");
        world
            .add_component(entity, DemoBandStatus(status.to_string()))
            .expect("add DemoBandStatus");
        count += 1;
    }

    count
}

/// Insert the canonical subject, layers, and stages.
pub fn insert_projection_subject(world: &mut World) -> usize {
    let mut count = 0;

    // Subject.
    let subject: Entity = world
        .spawn(EntityKind::Node, None)
        .map(|s| s.entity.into())
        .expect("spawn subject");
    world
        .add_component(
            subject,
            ProjectionSubjectId("computational-subject:prism-model".to_string()),
        )
        .expect("add ProjectionSubjectId");
    world
        .add_component(
            subject,
            ProjectionSubjectName("Computational Subject — Prism Model".to_string()),
        )
        .expect("add ProjectionSubjectName");
    world
        .add_component(
            subject,
            ProjectionSubjectKind("ComputeImage".to_string()),
        )
        .expect("add ProjectionSubjectKind");

    // Layers.
    let layers: Vec<(&str, &str, u8, &str)> = vec![
        ("metadata", "Metadata", 0, "#6ad4ff"),
        ("logical", "Logical tensors", 1, "#c56ad4"),
        ("physical", "Physical layouts", 2, "#ffb86c"),
        ("execution", "Execution views", 3, "#8effa3"),
        ("plan", "Plan + receipts", 4, "#e8e8ee"),
    ];
    let layer_structs: Vec<ProjectionLayer> = layers
        .iter()
        .map(|(id, name, depth, color)| ProjectionLayer {
            id: id.to_string(),
            name: name.to_string(),
            depth: *depth,
            color: color.to_string(),
        })
        .collect();
    let subject_for_layers: Entity = subject;
    world
        .add_component(subject_for_layers, ProjectionLayers(layer_structs))
        .expect("add ProjectionLayers");
    count += 1;

    // Stages.
    let stages: Vec<(&str, &str, u32)> = vec![
        ("replay", "replay", 1),
        ("project", "project", 2),
        ("reconcile", "reconcile", 3),
    ];
    for (id, label, order) in &stages {
        let entity: Entity = world
            .spawn(EntityKind::Node, None)
            .map(|s| s.entity.into())
            .expect("spawn stage");
        world
            .add_component(entity, ProjectionStageId(id.to_string()))
            .expect("add ProjectionStageId");
        world
            .add_component(entity, ProjectionStageLabel(label.to_string()))
            .expect("add ProjectionStageLabel");
        world
            .add_component(entity, ProjectionStageOrder(*order))
            .expect("add ProjectionStageOrder");
        count += 1;
    }

    count
}
