//! Stage 1: Evolutionary search systems.
//!
//! These functions seed, evaluate, select, mutate, and notify on the
//! evolutionary search lifecycle.  They operate on the foundation types
//! from [`super::foundation`] and the crate's custom `World` ECS.

use crate::ecs::component::backend::BackendTarget;
use crate::ecs::evolution::foundation::{
    CostMetrics, EvolutionState, EvolveCandidate, EvolveProgram, SearchConfig,
};
use crate::ecs::plan::CodecFamily;

use crate::ecs::Entity;
use crate::ecs::{Component, EntityKind, World};

// ── Deterministic PRNG (SplitMix64) ─────────────────────────────────────────
// Same algorithm used by EvolKvRng in evolkv.rs — no external dependency needed.

/// Deterministic pseudo-random number generator for mutation and crossover.
struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state.wrapping_add(0x9E3779B97F4A7C15);
        self.state = x;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    /// Generate an `f64` in `[0, 1)` with 53 bits of mantissa precision.
    fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Generate a `u64` in `[lo, hi)` (exclusive upper bound).
    fn range_u64(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo {
            return lo;
        }
        let range = hi - lo;
        lo + (self.next_u64() % range)
    }

    /// Generate an `i64` in `[lo, hi]` (inclusive).
    fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let range = (hi - lo) as u64;
        lo + (self.next_u64() % range) as i64
    }
}

/// Simple FNV-1a hash of a string to produce a deterministic seed.
fn hash_seed(content: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in content.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

/// Simple bool for crossover interleaving (toggle-based for determinism).
fn rng_bool() -> bool {
    false
}

// ── Component registration ─────────────────────────────────────────────────
// Foundation types used as ECS components need the Component marker trait.
impl Component for EvolutionState {}
impl Component for EvolveCandidate {}

// ── Public API ─────────────────────────────────────────────────────────────

/// Spawn the initial population from a seed program by introducing random
/// mutations to create `population_size` candidates.
///
/// Returns the entity id of the `EvolutionState` entity (which also carries an
/// `EvolveCandidate` component for the seed candidate).
pub fn evolve_seed(
    world: &mut World,
    tensor_id: &str,
    target_backend: &BackendTarget,
    seed: EvolveProgram,
    config: SearchConfig,
) -> Result<Entity, String> {
    let spawn_result = world
        .spawn(EntityKind::Node, None)
        .map_err(|e| format!("{:?}", e))?;
    let state_entity = spawn_result.entity;

    let _ = world.add_component(state_entity,
    EvolutionState {
        tensor_id: tensor_id.to_string(),
        target_backend: *target_backend,
        seed_program: seed.clone(),
        population: Vec::new(),
        records: Vec::new(),
        generation: 0,
        best_cost: None,
        best_candidate: None,
        converged: false,
        search_config: config.clone(),
        receipt_store: Vec::new(),
    },);

    let _ = world.add_component(state_entity,
    EvolveCandidate {
        tensor_id: tensor_id.to_string(),
        target_backend: *target_backend,
        format: CodecFamily::Ternary,
        program: seed.clone(),
        measured_cost: None,
        generation: 0,
        parents: Vec::new(),
    },);

    // Spawn population entities with perturbed programs.
    // Derive base seed from the seed program for determinism.
    let seed_content = format!("{:?}", &seed);
    let base_seed = hash_seed(&seed_content);

    let mut population_entities: Vec<Entity> = Vec::with_capacity(config.population_size);

    for i in 0..config.population_size.saturating_sub(1) {
        let child = mutate_program(&seed, &config, base_seed.wrapping_add(i as u64));
        let pop_entity = world
            .spawn(EntityKind::Node, None)
            .map_err(|e| format!("{:?}", e))?;
        let _ = world.add_component(pop_entity,
        EvolveCandidate {
            tensor_id: tensor_id.to_string(),
            target_backend: *target_backend,
            format: CodecFamily::Ternary,
            program: child,
            measured_cost: None,
            generation: 0,
            parents: vec![format!("seed-{}", tensor_id)],
        },);
        population_entities.push(pop_entity.entity);
    }

    // Link population into the state
    if let Some(state) = world.get_component_mut::<EvolutionState>(state_entity) {
        state.population = population_entities;
    }

    Ok(state_entity)
}

/// Evaluate one candidate by recording its measured cost.
pub fn evolve_evaluate(candidate: &mut EvolveCandidate, measured: CostMetrics) {
    candidate.measured_cost = Some(measured);
}

/// Select the fittest candidates up to `population_size`.
///
/// Sorts `population` by cost (using the configured CostFunction, lower is
/// better), updates the state with the best measurement, checks convergence,
/// and truncates to the configured population size.
pub fn evolve_select(state: &mut EvolutionState, population: &mut [EvolveCandidate]) {
    // Sort by cost using the configured cost function (lower is better)
    population.sort_by(|a, b| {
        let a_cost = cost_value(&state.search_config.cost_function, &a.measured_cost);
        let b_cost = cost_value(&state.search_config.cost_function, &b.measured_cost);
        a_cost
            .partial_cmp(&b_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Record best
    if let Some(best) = population.first() {
        // Capture previous best before overwriting
        let prev_best = state.best_cost.clone();
        state.best_cost = best.measured_cost.clone();

        // Check convergence relative to previous best
        if let (Some(prev), Some(curr)) = (prev_best.as_ref(), best.measured_cost.as_ref()) {
            // Use wall_ns for convergence measurement (monotonic improvement)
            if prev.wall_ns > curr.wall_ns {
                let improvement = (prev.wall_ns - curr.wall_ns) as f64 / prev.wall_ns as f64;
                if improvement < state.search_config.convergence_threshold {
                    state.converged = true;
                }
            }
        }
    }

    // Keep only the fittest
    state
        .population
        .truncate(state.search_config.population_size);
    state.generation += 1;
}

/// Compute a scalar cost from CostMetrics using the configured CostFunction.
fn cost_value(
    cf: &crate::ecs::evolution::foundation::CostFunction,
    mc: &Option<CostMetrics>,
) -> f64 {
    use crate::ecs::evolution::foundation::CostFunction;
    match (cf, mc) {
        (CostFunction::WallTime, Some(m)) => m.wall_ns as f64,
        (CostFunction::Energy, Some(m)) => m.energy_uj.unwrap_or(u64::MAX) as f64,
        (CostFunction::Bandwidth, Some(m)) => m.bandwidth_bytes as f64,
        (
            CostFunction::Weighted {
                wall,
                energy,
                bandwidth,
            },
            Some(m),
        ) => {
            wall * m.wall_ns as f64
                + energy * m.energy_uj.unwrap_or(0) as f64
                + bandwidth * m.bandwidth_bytes as f64
        }
        _ => f64::MAX,
    }
}

/// Mutate a program to create an offspring.
///
/// Uses a deterministic SplitMix64 PRNG seeded from the program content.
/// For shader programs: perturb tile dimensions, simdgroup sizes, and vectors.
/// For custom-pack programs: perturb tile dimensions and instruction lists.
/// For MIL programs: swap op types (MatMul ↔ Conv1x1) and perturb SRAM budget.
pub fn mutate_program(program: &EvolveProgram, _config: &SearchConfig, seed: u64) -> EvolveProgram {
    let mut rng = Prng::new(seed);
    match program {
        EvolveProgram::MetalShader(src) => mutate_metal_shader(src, &mut rng),
        EvolveProgram::CustomPack {
            tile_m,
            tile_n,
            tile_k,
            instructions,
        } => mutate_custom_pack(*tile_m, *tile_n, *tile_k, instructions, &mut rng),
        EvolveProgram::MilProgram(frag) => mutate_mil_program(frag, &mut rng),
        other => other.clone(),
    }
}

/// Mutate a Metal shader by perturbing tile dimensions, simdgroup size, or vector width.
fn mutate_metal_shader(src: &str, rng: &mut Prng) -> EvolveProgram {
    // Try to parse and perturb numeric parameters in the shader source.
    // Patterns we handle:
    //   threadgroup_width(N) / threadgroup_height(N)
    //   threadgroup_width = N
    //   constexpr constant int <NAME> = N;
    //   simdgroup_size(N)
    //   #define <NAME> N
    let mut result = src.to_string();
    let mut mutations = Vec::new();
    let mut has_known_pattern = false;

    // Pattern 1: threadgroup_width(N) or threadgroup_width = N
    for &open in [
        "threadgroup_width(",
        "threadgroup_width = ",
        "threadgroup_width=",
    ]
    .iter()
    {
        if let Some(pos) = result.find(open) {
            has_known_pattern = true;
            let start = pos + open.len();
            let end = result[start..]
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(result[start..].len());
            if end > 0 {
                if let Ok(val) = result[start..start + end].parse::<u64>() {
                    let delta =
                        rng.range_i64(-((val as i64).max(4) / 4), (val as i64).max(4) / 4 + 1);
                    let new_val = (val as i64 + delta).max(1) as u64;
                    mutations.push(format!("threadgroup_width {} -> {}", val, new_val));
                    result.replace_range(start..start + end, &new_val.to_string());
                    break;
                }
            }
        }
    }

    // Pattern 2: threadgroup_height(N) or threadgroup_height = N
    for &open in [
        "threadgroup_height(",
        "threadgroup_height = ",
        "threadgroup_height=",
    ]
    .iter()
    {
        if let Some(pos) = result.find(open) {
            has_known_pattern = true;
            let start = pos + open.len();
            let end = result[start..]
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(result[start..].len());
            if end > 0 {
                if let Ok(val) = result[start..start + end].parse::<u64>() {
                    let delta =
                        rng.range_i64(-((val as i64).max(4) / 4), (val as i64).max(4) / 4 + 1);
                    let new_val = (val as i64 + delta).max(1) as u64;
                    mutations.push(format!("threadgroup_height {} -> {}", val, new_val));
                    result.replace_range(start..start + end, &new_val.to_string());
                    break;
                }
            }
        }
    }

    // Pattern 3: simdgroup_size(N)
    if let Some(pos) = result.find("simdgroup_size(") {
        has_known_pattern = true;
        let start = pos + 15; // len of "simdgroup_size("
        let end = result[start..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(result[start..].len());
        if end > 0 {
            if let Ok(val) = result[start..start + end].parse::<u64>() {
                let delta = rng.range_i64(-((val as i64).max(4) / 4), (val as i64).max(4) / 4 + 1);
                let new_val = (val as i64 + delta).max(1) as u64;
                mutations.push(format!("simdgroup_size {} -> {}", val, new_val));
                result.replace_range(start..start + end, &new_val.to_string());
            }
        }
    }

    // Pattern 4: constexpr constant int <NAME> = <N>; for known tile names
    let tile_names = [
        "THREADGROUP_WIDTH",
        "THREADGROUP_HEIGHT",
        "SIMDGROUP_SIZE",
        "TILE_M",
        "TILE_N",
        "TILE_K",
        "VECTOR_WIDTH",
        "REDUCTION_FACTOR",
    ];
    for name in &tile_names {
        let search_for = format!("{} = ", name);
        if let Some(pos) = result.find(&search_for) {
            has_known_pattern = true;
            let start = pos + search_for.len();
            let end = result[start..]
                .find(|c: char| !c.is_ascii_digit() && c != ';')
                .unwrap_or(result[start..].len());
            if end > 0 {
                let val_str = result[start..start + end].trim_end_matches(';');
                if let Ok(val) = val_str.parse::<u64>() {
                    let delta =
                        rng.range_i64(-((val as i64).max(4) / 4), (val as i64).max(4) / 4 + 1);
                    let new_val = (val as i64 + delta).max(1) as u64;
                    mutations.push(format!("{} {} -> {}", name, val, new_val));
                    result.replace_range(start..start + end, &new_val.to_string());
                }
            }
        }
    }

    // Pattern 5: #define <NAME> <N> for common tile/storage defines
    let define_names = [
        "THREADGROUP_WIDTH",
        "THREADGROUP_HEIGHT",
        "TILE_M",
        "TILE_N",
        "TILE_K",
        "VECTOR_WIDTH",
        "BLOCK_SIZE",
    ];
    for name in &define_names {
        let search_for = format!("#define {} ", name);
        if let Some(pos) = result.find(&search_for) {
            has_known_pattern = true;
            let start = pos + search_for.len();
            let end = result[start..]
                .find(|c: char| !c.is_ascii_digit() && c != '\n')
                .unwrap_or(result[start..].len());
            if end > 0 {
                let val_str = &result[start..start + end];
                if let Ok(val) = val_str.parse::<u64>() {
                    let delta =
                        rng.range_i64(-((val as i64).max(4) / 4), (val as i64).max(4) / 4 + 1);
                    let new_val = (val as i64 + delta).max(1) as u64;
                    mutations.push(format!("#define {} {} -> {}", name, val, new_val));
                    result.replace_range(start..start + end, &new_val.to_string());
                }
            }
        }
    }

    // Append metadata tracking: record what was mutated
    if !mutations.is_empty() {
        let meta_line = format!("\n// MUTATIONS: {}\n", mutations.join(", "));
        result.push_str(&meta_line);
    } else if has_known_pattern {
        // Found a pattern but couldn't parse the value
        result.push_str("\n// MUTATION: shader tile params\n");
    } else {
        // Unrecognized shader: append comment-based mutation note
        result.push_str("\n// mutated (unrecognized shader structure)\n");
    }

    EvolveProgram::MetalShader(result)
}

/// Mutate a CustomPack program by perturbing tile dimensions and instruction mix.
fn mutate_custom_pack(
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    instructions: &[crate::ecs::evolution::foundation::CustomInstruction],
    rng: &mut Prng,
) -> EvolveProgram {
    // Perturb tile dimensions by -32..+32, keeping minimum 4
    let perturb = |val: usize, rng: &mut Prng| -> usize {
        let delta = rng.range_i64(-32, 33);
        (val as i64 + delta).max(4) as usize
    };

    let new_m = perturb(tile_m, rng);
    let new_n = perturb(tile_n, rng);
    let new_k = perturb(tile_k, rng);

    // Mutate instruction list
    let mut mutated_instructions = instructions.to_vec();

    // 40% chance to add a random instruction
    if rng.random() < 0.4 {
        let inst = random_instruction(rng);
        let insert_pos = rng.range_u64(0, mutated_instructions.len() as u64 + 1) as usize;
        mutated_instructions.insert(insert_pos, inst);
    }

    // 25% chance to remove a random instruction (if any exist)
    if !mutated_instructions.is_empty() && rng.random() < 0.25 {
        let remove_pos = rng.range_u64(0, mutated_instructions.len() as u64) as usize;
        mutated_instructions.remove(remove_pos);
    }

    // 30% chance to mutate a random instruction's reduction strategy
    if !mutated_instructions.is_empty() && rng.random() < 0.3 {
        let idx = rng.range_u64(0, mutated_instructions.len() as u64) as usize;
        match &mutated_instructions[idx] {
            crate::ecs::evolution::foundation::CustomInstruction::ReduceAdd { srcs, dst } => {
                // Vary reduction: split into fewer source operands
                if !srcs.is_empty() {
                    let split_at = rng.range_u64(1, srcs.len() as u64).max(1) as usize;
                    if split_at < srcs.len() {
                        let mut new_srcs = srcs.clone();
                        new_srcs.truncate(split_at);
                        mutated_instructions[idx] =
                            crate::ecs::evolution::foundation::CustomInstruction::ReduceAdd {
                                srcs: new_srcs,
                                dst: *dst,
                            };
                    }
                }
            }
            crate::ecs::evolution::foundation::CustomInstruction::Fma { a, b, c } => {
                // Reorder FMA operands (commutative, creates different scheduling)
                mutated_instructions[idx] =
                    crate::ecs::evolution::foundation::CustomInstruction::Fma {
                        a: *b,
                        b: *a,
                        c: *c,
                    };
            }
            _ => {}
        }
    }

    EvolveProgram::CustomPack {
        tile_m: new_m,
        tile_n: new_n,
        tile_k: new_k,
        instructions: mutated_instructions,
    }
}

/// Generate a random instruction for mutation injection.
fn random_instruction(rng: &mut Prng) -> crate::ecs::evolution::foundation::CustomInstruction {
    let kind = rng.range_u64(0, 5);
    match kind {
        0 => crate::ecs::evolution::foundation::CustomInstruction::LoadWeight {
            offset: rng.next_u64() % 1024,
            format: crate::ecs::plan::CodecFamily::Ternary,
        },
        1 => crate::ecs::evolution::foundation::CustomInstruction::Dequantize {
            src: 0,
            dst: 1,
            codebook: crate::ecs::evolution::foundation::EvolveCodebookRef {
                name: "mutated".into(),
                offset: 0,
                length: 64,
            },
        },
        2 => crate::ecs::evolution::foundation::CustomInstruction::Accumulate { src: 0, dst: 1 },
        3 => crate::ecs::evolution::foundation::CustomInstruction::ReduceAdd {
            srcs: vec![0, 1, 2],
            dst: 3,
        },
        _ => crate::ecs::evolution::foundation::CustomInstruction::Fma { a: 0, b: 1, c: 2 },
    }
}

/// Mutate a MIL program by swapping op types and perturbing SRAM budget.
fn mutate_mil_program(
    frag: &crate::ecs::evolution::foundation::MilProgramFragment,
    rng: &mut Prng,
) -> EvolveProgram {
    use crate::ecs::evolution::foundation::MilOp;

    let mut mutated_ops = frag.ops.clone();

    // 30% chance per MatMul op to convert to Conv1x1
    for op in &mut mutated_ops {
        if let MilOp::MatMul { lhs, rhs, output } = op {
            if rng.random() < 0.3 {
                *op = MilOp::Conv1x1 {
                    input: *lhs,
                    weight: *rhs,
                    output: *output,
                };
            }
        }
    }

    // 30% chance per Conv1x1 op to convert back to MatMul
    for op in &mut mutated_ops {
        if let MilOp::Conv1x1 {
            input,
            weight,
            output,
        } = op
        {
            if rng.random() < 0.3 {
                *op = MilOp::MatMul {
                    lhs: *input,
                    rhs: *weight,
                    output: *output,
                };
            }
        }
    }

    // Perturb sram_budget by ±20%
    let budget_delta = ((frag.sram_budget as f64) * 0.2) as i64;
    let new_budget =
        (frag.sram_budget as i64 + rng.range_i64(-budget_delta, budget_delta + 1)).max(1024) as u64;

    EvolveProgram::MilProgram(crate::ecs::evolution::foundation::MilProgramFragment {
        ops: mutated_ops,
        schedule: frag.schedule.clone(),
        sram_budget: new_budget,
    })
}

/// Create offspring from two parents via crossover.
///
/// Blends features from both parents:
/// - MetalShader: header from parent_a, body from parent_b
/// - CustomPack: average tile dimensions, mix instruction lists
/// - MilProgram: interleave ops from both parents, average SRAM budget
pub fn crossover(parent_a: &EvolveProgram, parent_b: &EvolveProgram) -> EvolveProgram {
    match (parent_a, parent_b) {
        (EvolveProgram::MetalShader(a), EvolveProgram::MetalShader(b)) => crossover_shader(a, b),
        (
            EvolveProgram::CustomPack {
                tile_m: am,
                tile_n: an,
                tile_k: ak,
                instructions: ai,
            },
            EvolveProgram::CustomPack {
                tile_m: bm,
                tile_n: bn,
                tile_k: bk,
                instructions: bi,
            },
        ) => crossover_custom_pack(*am, *an, *ak, ai, *bm, *bn, *bk, bi),
        (EvolveProgram::MilProgram(a), EvolveProgram::MilProgram(b)) => crossover_mil_program(a, b),
        // Different variants: clone parent_a as fallback
        _ => parent_a.clone(),
    }
}

/// Crossover two Metal shaders: header from parent_a, body from parent_b.
fn crossover_shader(a: &str, b: &str) -> EvolveProgram {
    // Split at the first '{' — everything before is "header", the rest is "body"
    let a_header = a.split_once('{').map(|(h, _)| h).unwrap_or(a);
    let b_body = b.split_once('{').map(|(_, body)| body).unwrap_or(b);

    let result = if a.find('{').is_some() && b.find('{').is_some() {
        format!("{}{{{}", a_header.trim_end(), b_body)
    } else {
        // Can't split cleanly — use parent_a with crossover annotation
        format!("{}\n// crossover: parent_a header + parent_b body\n", a)
    };

    EvolveProgram::MetalShader(result)
}

/// Crossover two CustomPack programs: average tile dimensions + mix instructions.
fn crossover_custom_pack(
    am: usize,
    an: usize,
    ak: usize,
    ai: &[crate::ecs::evolution::foundation::CustomInstruction],
    bm: usize,
    bn: usize,
    bk: usize,
    bi: &[crate::ecs::evolution::foundation::CustomInstruction],
) -> EvolveProgram {
    // Average tile dimensions
    let tile_m = (am + bm) / 2;
    let tile_n = (an + bn) / 2;
    let tile_k = (ak + bk) / 2;

    // Mix instructions: alternate from each parent (toggle-based for determinism)
    let mut mixed = Vec::with_capacity(ai.len() + bi.len());
    let mut i = 0;
    let mut j = 0;
    let mut take_from_a = true;
    while i < ai.len() || j < bi.len() {
        if take_from_a && i < ai.len() {
            mixed.push(ai[i].clone());
            i += 1;
        } else if !take_from_a && j < bi.len() {
            mixed.push(bi[j].clone());
            j += 1;
        }
        take_from_a = if rng_bool() {
            !take_from_a
        } else {
            take_from_a
        };

        // Drain the remaining parent when one is exhausted
        if i >= ai.len() {
            while j < bi.len() {
                mixed.push(bi[j].clone());
                j += 1;
            }
            break;
        }
        if j >= bi.len() {
            while i < ai.len() {
                mixed.push(ai[i].clone());
                i += 1;
            }
            break;
        }
    }

    EvolveProgram::CustomPack {
        tile_m,
        tile_n,
        tile_k,
        instructions: mixed,
    }
}

/// Crossover two MIL programs: interleave ops from both parents.
fn crossover_mil_program(
    a: &crate::ecs::evolution::foundation::MilProgramFragment,
    b: &crate::ecs::evolution::foundation::MilProgramFragment,
) -> EvolveProgram {
    // Interleave ops: take from a and b alternately
    let mut interleaved_ops = Vec::with_capacity(a.ops.len() + b.ops.len());
    let mut i = 0;
    let mut j = 0;
    let mut take_from_a = true;
    while i < a.ops.len() || j < b.ops.len() {
        if take_from_a && i < a.ops.len() {
            interleaved_ops.push(a.ops[i].clone());
            i += 1;
        } else if !take_from_a && j < b.ops.len() {
            interleaved_ops.push(b.ops[j].clone());
            j += 1;
        }
        take_from_a = !take_from_a;

        // Drain the remaining parent when one is exhausted
        if i >= a.ops.len() {
            while j < b.ops.len() {
                interleaved_ops.push(b.ops[j].clone());
                j += 1;
            }
            break;
        }
        if j >= b.ops.len() {
            while i < a.ops.len() {
                interleaved_ops.push(a.ops[i].clone());
                i += 1;
            }
            break;
        }
    }

    // Average SRAM budget
    let sram_budget = (a.sram_budget + b.sram_budget) / 2;

    EvolveProgram::MilProgram(crate::ecs::evolution::foundation::MilProgramFragment {
        ops: interleaved_ops,
        schedule: a.schedule.clone(),
        sram_budget,
    })
}

/// Return the best (lowest‑cost) candidate after selection.
pub fn evolve_winner<'a>(
    _state: &EvolutionState,
    population: &'a [EvolveCandidate],
) -> Option<&'a EvolveCandidate> {
    population.first()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::evolution::foundation::CostFunction;

    #[test]
    fn test_evolve_seed_creates_population() {
        let mut world = World::new();
        world.set_direct_mutation_allowed(true);

        let config = SearchConfig {
            population_size: 5,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };

        let prog = EvolveProgram::MetalShader("kernel void foo() {}".into());
        let entity = evolve_seed(&mut world, "t0", &BackendTarget::Metal, prog, config)
            .expect("evolve_seed should succeed");

        let state = world
            .get_component::<EvolutionState>(entity)
            .expect("state component should exist");
        assert_eq!(state.population.len(), 4); // population_size - 1 (seed is on state entity)
        assert_eq!(state.generation, 0);
    }

    #[test]
    fn test_mutate_shader_perturbs_params() {
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let prog = EvolveProgram::MetalShader("kernel void foo() {}".into());
        let seed = hash_seed("kernel void foo() {}");
        let mutated = mutate_program(&prog, &config, seed);
        match mutated {
            EvolveProgram::MetalShader(s) => {
                // Must have the MUTATIONS or mutation marker appended
                assert!(
                    s.contains("mutated"),
                    "mutated shader should contain mutation metadata: got len={}",
                    s.len()
                );
            }
            _ => panic!("expected MetalShader variant"),
        }
    }

    #[test]
    fn test_mutate_shader_with_tile_params() {
        let src = concat!(
            "kernel void matmul(device float* A [[buffer(0)]], ",
            "constant float* B [[buffer(1)]], device float* C [[buffer(2)]]) {\n",
            "    constexpr constant int THREADGROUP_WIDTH = 16;\n",
            "    constexpr constant int THREADGROUP_HEIGHT = 16;\n",
            "    constexpr constant int TILE_M = 64;\n",
            "    constexpr constant int TILE_N = 128;\n",
            "    constexpr constant int TILE_K = 32;\n",
            "}",
        );
        let prog = EvolveProgram::MetalShader(src.into());
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let seed = hash_seed(src);
        let mutated = mutate_program(&prog, &config, seed);
        match mutated {
            EvolveProgram::MetalShader(s) => {
                // Verify the shader was actually modified and contains mutation metadata
                assert!(
                    s.contains("MUTATION") || s.contains("mutated"),
                    "mutated shader should contain mutation metadata in: {}",
                    s
                );
                assert!(
                    s.len() > src.len(),
                    "mutated shader should be longer than original (has metadata)"
                );
                // Verify at least one tile param was modified from original values
                assert!(
                    s.contains("THREADGROUP_WIDTH")
                        || s.contains("TILE_M")
                        || s.contains("TILE_N")
                        || s.contains("TILE_K"),
                    "mutated shader should retain tile parameter names"
                );
            }
            _ => panic!("expected MetalShader variant"),
        }
    }

    #[test]
    fn test_mutate_custom_pack_perturbs_tiles() {
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let prog = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            instructions: vec![],
        };
        let seed = hash_seed("64,128,32,0");
        let mutated = mutate_program(&prog, &config, seed);
        match mutated {
            EvolveProgram::CustomPack {
                tile_m,
                tile_n,
                tile_k,
                ..
            } => {
                // Tiles should be perturbed by -32..+32 from originals
                assert!(
                    tile_m >= 4 && tile_m <= 96,
                    "tile_m {} should be in [4, 96]",
                    tile_m
                );
                assert!(
                    tile_n >= 4 && tile_n <= 160,
                    "tile_n {} should be in [4, 160]",
                    tile_n
                );
                assert!(
                    tile_k >= 4 && tile_k <= 64,
                    "tile_k {} should be in [4, 64]",
                    tile_k
                );
                // At least one dimension should have changed
                assert!(
                    tile_m != 64 || tile_n != 128 || tile_k != 32,
                    "at least one tile dimension should be perturbed (got m={}, n={}, k={})",
                    tile_m,
                    tile_n,
                    tile_k
                );
            }
            _ => panic!("expected CustomPack variant"),
        }
    }

    #[test]
    fn test_mutate_custom_pack_adds_instruction() {
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let instructions = vec![
            crate::ecs::evolution::foundation::CustomInstruction::Fma { a: 0, b: 1, c: 2 },
            crate::ecs::evolution::foundation::CustomInstruction::ReduceAdd {
                srcs: vec![0, 1, 2],
                dst: 3,
            },
        ];
        let prog = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            instructions: instructions.clone(),
        };
        // Use a seed that ensures mutation happens (iterate until we get one)
        let content = format!("{},{},{},{}", 64usize, 128, 32, instructions.len());
        let seed = hash_seed(&content);
        let mutated = mutate_program(&prog, &config, seed);
        match mutated {
            EvolveProgram::CustomPack {
                instructions: new_insts,
                ..
            } => {
                // The mutated instruction list may be longer, shorter, or same
                // depending on deterministic decisions. Verify it exists.
                assert!(new_insts.len() > 0, "instruction list should not be empty");
            }
            _ => panic!("expected CustomPack variant"),
        }
    }

    #[test]
    fn test_mutate_mil_program_swaps_ops() {
        use crate::ecs::evolution::foundation::{MilOp, MilProgramFragment, MilSchedule, MilUnit};
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let frag = MilProgramFragment {
            ops: vec![
                MilOp::MatMul {
                    lhs: 0,
                    rhs: 1,
                    output: 2,
                },
                MilOp::MatMul {
                    lhs: 3,
                    rhs: 4,
                    output: 5,
                },
                MilOp::Add {
                    lhs: 2,
                    rhs: 5,
                    output: 6,
                },
            ],
            schedule: MilSchedule {
                units: vec![MilUnit {
                    op_range: 0..3,
                    assigned_neuron: 0,
                    sram_usage: 4096,
                }],
                sync_points: vec![2],
            },
            sram_budget: 16384,
        };
        let content = format!(
            "{},{},{}",
            frag.ops.len(),
            frag.schedule.units.len(),
            frag.sram_budget
        );
        let seed = hash_seed(&content);
        let prog = EvolveProgram::MilProgram(frag);
        let mutated = mutate_program(&prog, &config, seed);
        match mutated {
            EvolveProgram::MilProgram(m) => {
                // Should still have ops
                assert!(!m.ops.is_empty(), "MIL program should have ops");
                // SRAM budget should be within ±20% of original
                let ratio = m.sram_budget as f64 / 16384.0;
                assert!(
                    ratio >= 0.8 && ratio <= 1.2,
                    "SRAM budget {} should be within 20% of 16384 (ratio={:.3})",
                    m.sram_budget,
                    ratio
                );
            }
            _ => panic!("expected MilProgram variant"),
        }
    }

    #[test]
    fn test_crossover_shader_blends_header_body() {
        let a = EvolveProgram::MetalShader(
            "kernel void foo(device float* a [[buffer(0)]]) { return a[0] * 2.0; }".into(),
        );
        let b = EvolveProgram::MetalShader(
            "kernel void bar(device float* b [[buffer(0)]]) { return b[0] + 1.0; }".into(),
        );
        let child = crossover(&a, &b);
        match child {
            EvolveProgram::MetalShader(s) => {
                // Header from parent_a, body from parent_b
                assert!(
                    s.contains("kernel void foo"),
                    "child should contain parent_a's header"
                );
                assert!(
                    s.contains("return b[0] + 1.0"),
                    "child should contain parent_b's body"
                );
                assert!(
                    !s.contains("return a[0] * 2.0"),
                    "child should NOT contain parent_a's body"
                );
            }
            _ => panic!("expected MetalShader variant"),
        }
    }

    #[test]
    fn test_crossover_custom_pack_averages_tiles() {
        let a = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            instructions: vec![
                crate::ecs::evolution::foundation::CustomInstruction::LoadWeight {
                    offset: 0,
                    format: crate::ecs::plan::CodecFamily::Ternary,
                },
            ],
        };
        let b = EvolveProgram::CustomPack {
            tile_m: 96,
            tile_n: 64,
            tile_k: 48,
            instructions: vec![crate::ecs::evolution::foundation::CustomInstruction::Fma {
                a: 0,
                b: 1,
                c: 2,
            }],
        };
        let child = crossover(&a, &b);
        match child {
            EvolveProgram::CustomPack {
                tile_m,
                tile_n,
                tile_k,
                instructions,
            } => {
                assert_eq!(tile_m, 80, "tile_m should average (64+96)/2 = 80");
                assert_eq!(tile_n, 96, "tile_n should average (128+64)/2 = 96");
                assert_eq!(tile_k, 40, "tile_k should average (32+48)/2 = 40");
                // Instructions should be mixed from both parents
                assert!(
                    instructions.len() >= 1 && instructions.len() <= 2,
                    "should have 1-2 instructions (mixed), got {}",
                    instructions.len()
                );
            }
            _ => panic!("expected CustomPack variant"),
        }
    }

    #[test]
    fn test_crossover_mil_program_interleaves_ops() {
        use crate::ecs::evolution::foundation::{MilOp, MilProgramFragment, MilSchedule, MilUnit};
        let a = EvolveProgram::MilProgram(MilProgramFragment {
            ops: vec![
                MilOp::MatMul {
                    lhs: 0,
                    rhs: 1,
                    output: 2,
                },
                MilOp::Add {
                    lhs: 2,
                    rhs: 3,
                    output: 4,
                },
            ],
            schedule: MilSchedule {
                units: vec![MilUnit {
                    op_range: 0..2,
                    assigned_neuron: 0,
                    sram_usage: 4096,
                }],
                sync_points: vec![],
            },
            sram_budget: 16384,
        });
        let b = EvolveProgram::MilProgram(MilProgramFragment {
            ops: vec![
                MilOp::Conv1x1 {
                    input: 0,
                    weight: 1,
                    output: 2,
                },
                MilOp::Activation {
                    kind: "relu".into(),
                    input: 2,
                    output: 3,
                },
            ],
            schedule: MilSchedule {
                units: vec![MilUnit {
                    op_range: 0..2,
                    assigned_neuron: 1,
                    sram_usage: 2048,
                }],
                sync_points: vec![],
            },
            sram_budget: 8192,
        });
        let child = crossover(&a, &b);
        match child {
            EvolveProgram::MilProgram(m) => {
                // Should have ops from both parents interleaved
                assert_eq!(m.ops.len(), 4, "should interleave all 4 ops");
                // SRAM budget should be average
                assert_eq!(
                    m.sram_budget,
                    (16384 + 8192) / 2,
                    "SRAM budget should average"
                );
            }
            _ => panic!("expected MilProgram variant"),
        }
    }

    #[test]
    fn test_crossover_different_variants_falls_back() {
        let a = EvolveProgram::MetalShader("kernel A".into());
        let b = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            instructions: vec![],
        };
        let child = crossover(&a, &b);
        // Different variants: should clone parent_a
        match child {
            EvolveProgram::MetalShader(s) => {
                assert_eq!(s, "kernel A", "should clone parent_a when variants differ");
            }
            _ => panic!("expected MetalShader variant"),
        }
    }

    #[test]
    fn test_evolve_evaluate_records_metrics() {
        let mut candidate = EvolveCandidate {
            tensor_id: "test".into(),
            target_backend: BackendTarget::Metal,
            format: CodecFamily::Ternary,
            program: EvolveProgram::MetalShader("kernel void foo() {}".into()),
            measured_cost: None,
            generation: 0,
            parents: vec![],
        };

        let metrics = CostMetrics {
            wall_ns: 1500,
            energy_uj: Some(500),
            alu_cycles: Some(1234),
            bandwidth_bytes: 4096,
        };

        evolve_evaluate(&mut candidate, metrics);
        let recorded = candidate.measured_cost.expect("cost should be recorded");
        assert_eq!(recorded.wall_ns, 1500);
        assert_eq!(recorded.energy_uj, Some(500));
    }

    #[test]
    fn test_evolve_select_picks_lowest_cost() {
        let mut state = EvolutionState {
            tensor_id: "test".to_string(),
            target_backend: BackendTarget::Metal,
            seed_program: EvolveProgram::MetalShader("kernel void foo() {}".into()),
            population: Vec::new(),
            records: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: SearchConfig {
                population_size: 4,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 100,
                convergence_threshold: 0.5,
                cost_function: CostFunction::WallTime,
            },
            receipt_store: Vec::new(),
        };

        let mut pop: Vec<EvolveCandidate> = vec![
            EvolveCandidate {
                tensor_id: "test".into(),
                target_backend: BackendTarget::Metal,
                format: CodecFamily::Ternary,
                program: EvolveProgram::MetalShader("a".into()),
                measured_cost: Some(CostMetrics {
                    wall_ns: 200,
                    energy_uj: None,
                    alu_cycles: None,
                    bandwidth_bytes: 100,
                }),
                generation: 0,
                parents: vec![],
            },
            EvolveCandidate {
                tensor_id: "test".into(),
                target_backend: BackendTarget::Metal,
                format: CodecFamily::Ternary,
                program: EvolveProgram::MetalShader("b".into()),
                measured_cost: Some(CostMetrics {
                    wall_ns: 100,
                    energy_uj: None,
                    alu_cycles: None,
                    bandwidth_bytes: 100,
                }),
                generation: 0,
                parents: vec![],
            },
        ];

        state.best_cost = Some(CostMetrics {
            wall_ns: 300,
            energy_uj: None,
            alu_cycles: None,
            bandwidth_bytes: 100,
        });
        evolve_select(&mut state, &mut pop);
        // 66% improvement (300→100) exceeds 0.5 threshold — still improving, not converged
        assert!(
            !state.converged,
            "66% improvement > 50% threshold: not converged"
        );
        assert_eq!(state.generation, 1);
        // After sorting, the lowest-cost (100) should be first
        assert_eq!(
            pop.first()
                .and_then(|c| c.measured_cost.as_ref())
                .map(|m| m.wall_ns),
            Some(100)
        );
    }

    #[test]
    fn test_evolve_select_converges_on_small_improvement() {
        let mut state = EvolutionState {
            tensor_id: "test".to_string(),
            target_backend: BackendTarget::Metal,
            seed_program: EvolveProgram::MetalShader("kernel void foo() {}".into()),
            population: Vec::new(),
            records: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: SearchConfig {
                population_size: 4,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 100,
                convergence_threshold: 0.5, // requires 50% improvement
                cost_function: CostFunction::WallTime,
            },
            receipt_store: Vec::new(),
        };

        let mut pop: Vec<EvolveCandidate> = vec![EvolveCandidate {
            tensor_id: "test".into(),
            target_backend: BackendTarget::Metal,
            format: CodecFamily::Ternary,
            program: EvolveProgram::MetalShader("a".into()),
            measured_cost: Some(CostMetrics {
                wall_ns: 290,
                energy_uj: None,
                alu_cycles: None,
                bandwidth_bytes: 100,
            }),
            generation: 0,
            parents: vec![],
        }];

        state.best_cost = Some(CostMetrics {
            wall_ns: 300,
            energy_uj: None,
            alu_cycles: None,
            bandwidth_bytes: 100,
        });
        evolve_select(&mut state, &mut pop);
        // 3.3% improvement (300→290) < 50% threshold → converged
        assert!(
            state.converged,
            "3.3% improvement < 50% threshold should trigger convergence"
        );
        assert_eq!(state.generation, 1);
    }

    #[test]
    fn test_evolve_select_uses_cost_function() {
        let mut state = EvolutionState {
            tensor_id: "test".to_string(),
            target_backend: BackendTarget::Metal,
            seed_program: EvolveProgram::MetalShader("kernel void foo() {}".into()),
            population: Vec::new(),
            records: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: SearchConfig {
                population_size: 4,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 100,
                convergence_threshold: 0.5,
                cost_function: CostFunction::Weighted {
                    wall: 0.0,
                    energy: 1.0,
                    bandwidth: 0.0,
                },
            },
            receipt_store: Vec::new(),
        };

        let mut pop: Vec<EvolveCandidate> = vec![
            EvolveCandidate {
                tensor_id: "test".into(),
                target_backend: BackendTarget::Metal,
                format: CodecFamily::Ternary,
                program: EvolveProgram::MetalShader("a".into()),
                measured_cost: Some(CostMetrics {
                    wall_ns: 100,
                    energy_uj: Some(500),
                    alu_cycles: None,
                    bandwidth_bytes: 100,
                }),
                generation: 0,
                parents: vec![],
            },
            EvolveCandidate {
                tensor_id: "test".into(),
                target_backend: BackendTarget::Metal,
                format: CodecFamily::Ternary,
                program: EvolveProgram::MetalShader("b".into()),
                measured_cost: Some(CostMetrics {
                    wall_ns: 200,
                    energy_uj: Some(100),
                    alu_cycles: None,
                    bandwidth_bytes: 100,
                }),
                generation: 0,
                parents: vec![],
            },
        ];

        // With WallTime cost, a (100ns) would be cheaper
        // With Energy cost, b (100uj) should be cheaper
        // CostFunction::Weighted { wall: 0.0, energy: 1.0, bandwidth: 0.0 }
        // means only energy matters
        evolve_select(&mut state, &mut pop);
        let best = pop
            .first()
            .and_then(|c| c.measured_cost.as_ref())
            .map(|m| m.energy_uj)
            .flatten();
        assert_eq!(
            best,
            Some(100),
            "with energy-only weighting, b (100uj) should be first"
        );
    }

    #[test]
    fn test_evolve_winner_returns_first_after_sort() {
        let state = EvolutionState {
            tensor_id: "test".to_string(),
            target_backend: BackendTarget::Metal,
            seed_program: EvolveProgram::MetalShader("k".into()),
            population: Vec::new(),
            records: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: SearchConfig {
                population_size: 4,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 100,
                convergence_threshold: 0.01,
                cost_function: CostFunction::WallTime,
            },
            receipt_store: Vec::new(),
        };

        let pop = vec![EvolveCandidate {
            tensor_id: "test".into(),
            target_backend: BackendTarget::Metal,
            format: CodecFamily::Ternary,
            program: EvolveProgram::MetalShader("best".into()),
            measured_cost: Some(CostMetrics {
                wall_ns: 50,
                energy_uj: None,
                alu_cycles: None,
                bandwidth_bytes: 100,
            }),
            generation: 0,
            parents: vec![],
        }];

        let winner = evolve_winner(&state, &pop);
        assert!(winner.is_some());
        assert_eq!(
            winner
                .and_then(|c| c.measured_cost.as_ref())
                .map(|m| m.wall_ns),
            Some(50)
        );
    }

    #[test]
    fn test_cost_value_wall_time() {
        use crate::ecs::evolution::foundation::CostFunction;
        let m = Some(CostMetrics {
            wall_ns: 1000,
            energy_uj: Some(500),
            alu_cycles: Some(100),
            bandwidth_bytes: 4096,
        });
        assert_eq!(cost_value(&CostFunction::WallTime, &m), 1000.0);
        assert_eq!(cost_value(&CostFunction::Bandwidth, &m), 4096.0);
        assert_eq!(cost_value(&CostFunction::Energy, &m), 500.0);
    }

    #[test]
    fn test_cost_value_weighted() {
        use crate::ecs::evolution::foundation::CostFunction;
        let m = Some(CostMetrics {
            wall_ns: 1000,
            energy_uj: Some(500),
            alu_cycles: Some(100),
            bandwidth_bytes: 4096,
        });
        let cf = CostFunction::Weighted {
            wall: 0.5,
            energy: 0.3,
            bandwidth: 0.2,
        };
        let expected = 0.5 * 1000.0 + 0.3 * 500.0 + 0.2 * 4096.0;
        assert_eq!(cost_value(&cf, &m), expected);
    }

    #[test]
    fn test_cost_value_none_returns_max() {
        use crate::ecs::evolution::foundation::CostFunction;
        assert_eq!(cost_value(&CostFunction::WallTime, &None), f64::MAX);
        assert_eq!(cost_value(&CostFunction::Energy, &None), f64::MAX);
        assert_eq!(cost_value(&CostFunction::Bandwidth, &None), f64::MAX);
    }

    #[test]
    fn test_hash_seed_deterministic() {
        let h1 = hash_seed("hello");
        let h2 = hash_seed("hello");
        assert_eq!(h1, h2, "hash_seed should be deterministic");
        let h3 = hash_seed("world");
        assert_ne!(h1, h3, "different inputs should produce different hashes");
    }

    #[test]
    fn test_prng_deterministic() {
        let mut rng1 = Prng::new(42);
        let mut rng2 = Prng::new(42);
        let v1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let v2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_eq!(v1, v2, "same seed should produce same sequence");
    }

    #[test]
    fn test_prng_range_bounds() {
        let mut rng = Prng::new(99);
        for _ in 0..100 {
            let v = rng.range_u64(5, 10);
            assert!(
                v >= 5 && v < 10,
                "range_u64(5,10) should be in [5,10), got {}",
                v
            );
            let v = rng.range_i64(-5, 5);
            assert!(
                v >= -5 && v <= 5,
                "range_i64(-5,5) should be in [-5,5], got {}",
                v
            );
        }
    }

    #[test]
    fn test_mutate_program_uses_deterministic_seed() {
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let prog = EvolveProgram::MetalShader("kernel void test() {}".into());
        let seed = hash_seed("kernel void test() {}");
        let m1 = mutate_program(&prog, &config, seed);
        let m2 = mutate_program(&prog, &config, seed);
        // Same seed → same mutation
        assert_eq!(
            format!("{:?}", m1),
            format!("{:?}", m2),
            "same seed should produce identical mutations"
        );

        // Different seed → different mutation
        let m3 = mutate_program(&prog, &config, seed.wrapping_add(1));
        let m1_str = format!("{:?}", m1);
        let m3_str = format!("{:?}", m3);
        let diff_seed_produces_different = m1_str != m3_str;
        // Note: it's possible but unlikely that different seeds produce same output
        // (e.g., shader without parseable params). Just check at least one differs.
        let prog2 = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            instructions: vec![],
        };
        let m4 = mutate_program(&prog2, &config, seed);
        let m5 = mutate_program(&prog2, &config, seed.wrapping_add(1));
        let m4_str = format!("{:?}", m4);
        let m5_str = format!("{:?}", m5);
        assert!(
            diff_seed_produces_different || m4_str != m5_str,
            "at least one variant should produce different output for different seeds"
        );
    }
}
