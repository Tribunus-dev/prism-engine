# Semantic Region IR

Prism now has an initial persistent semantic sub-tensor abstraction.

A semantic region identifies **why a stable subset of a logical tensor exists**. It remains independent from the physical tiles, buffers, codecs, lanes, and residency choices that a target backend may later select.

The initial vertical slice provides:

- deterministic semantic-region identities;
- explicit graph, architecture, sensitivity, user, and hybrid provenance classes;
- fail-closed partition verification;
- canonical partition and plan digests;
- bounded region-level representation assignments;
- a versioned explicit JSON specification;
- a real SafeTensors-backed demonstration that verifies tensor identity and shape;
- a compile-verified receipt that marks numerical quality as unproven and execution performance as unmeasured.

The central claim is deliberately narrow:

> Prism can preserve semantic sub-tensor identity separately from the physical layout that a backend will eventually select.

## Demo

```bash
cargo run -p prism-ecs-compile \
  --example semantic_region_plan -- \
  --model-dir "$MODEL_DIR" \
  --tensor "model.layers.0.self_attn.qkv_proj.weight" \
  --spec examples/semantic-regions/qkv-gqa.example.json \
  --assign query_projection=fp16 \
  --assign key_projection=int8 \
  --assign value_projection=int8 \
  --json-out /tmp/semantic-region-plan.json
```

The checked-in example dimensions are illustrative and must match the mapped tensor exactly. Use a model-specific verified specification for a real model.

## Evidence classes

| Field | Initial status |
|---|---|
| Tensor identity and shape | Repository-backed mapped checkpoint |
| Region boundaries | Explicit architecture contract |
| Partition legality | Compile-verified |
| Representation assignment | Compile-verified plan |
| Numerical quality | Unproven |
| Latency and throughput | Unmeasured |

The first slice does not alter backend kernels, runtime scheduling, or the stable ComputeImage payload ABI. Future work will add static model discovery, region probes, hierarchical search, physical realization, optional ComputeImage manifests, measured backend execution, and coalesced scheduler/residency integration.
