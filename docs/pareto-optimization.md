# Pareto deployment optimization

Prism treats model deployment as a multi-objective search rather than a single weighted score. A search genome describes representation, packing, geometry, fusion, memory, Engram, and runtime choices. The compiler evaluates those choices, but a deployment candidate is not admitted to the durable archive until it has an explicit execution gate and evidence.

The deployment boundary is implemented in `prism_ecs_ir::evolution::pareto`. `DeploymentIdentity` binds a candidate to the source model, tokenizer, Engram generation, target, and workload. `DeploymentMeasurements` carries quality, latency, throughput, memory, KV, power, and transfer observations. `HardGate` records whether the candidate passed the required backend and quality/resource checks. `DeploymentEvidence` records the compiler/backend provenance and receipt references.

`ParetoArchive` retains non-dominated admitted candidates. It never admits a candidate without at least one passed gate, and it removes candidates that are dominated across the declared objective vector. `DeploymentPolicy` selects a point from the archive using hard deployment constraints first and weighted priorities second. This keeps policy selection separate from search: the same archive can serve a laptop memory policy, a low-latency server policy, or a high-concurrency policy.

The compiler search promotes its evaluated candidate records into this archive after joint backend evaluation. Surrogate scores can guide exploration, but they do not satisfy the deployment evidence gate. A production search still requires a measured evaluator and jointly feasible backend profile.

During receipt build, admitted candidates are closed over the emitted CImage digest and compilation receipt identifier. The ECS session retains the archive and selects a deployment digest using `DeploymentPolicy::quality_first`; later runtime or serving adapters can replace that policy without rerunning compilation.

The intended evidence progression is simulated planning, compile verification, differential correctness, hardware measurement, and production qualification. These are distinct claims; a legal CImage or a non-dominated simulated candidate is not by itself proof of hardware performance.

Engram-native models use the same candidate boundary but must bind the Engram artifact generation as part of identity. The table, hash/token mapping, gate, fusion projections, and compatibility contract should be versioned together. Placement and residency can then participate in the deployment search without pretending that an Engram table is an independently editable cache.
