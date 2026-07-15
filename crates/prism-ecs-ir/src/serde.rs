//! Deterministic serialization for the ECS-native IR.
//!
//! Snapshots an IR module (rooted at a top-level operation entity) to
//! deterministic JSON and back. The snapshot is a flat list of entity entries
//! and value entries, ordered so that operands appear before their uses.
//!
//! The in-memory IR module uses this entity hierarchy:
//!
//! ```text
//! Op (root)
//!   └── RegionRef → [Region]          // component on ops (defined here)
//!         └── RegionBlocks → [Block]  // component on region entities
//!               └── BlockOps → [Op]   // component on block entities
//! ```
//!
//! Each Op carries Result values. Each value carries a ValueDef (provenance),
//! ValueType, and Uses (consuming ops).
//!
//! ## Known limitations
//!
//! - `OpName` is not preserved across serialization. The snapshot rebuilds
//!   structural connectivity (operands, results, region/block hierarchy) but
//!   operation names default to `"builtin.unregistered"` on deserialization.
//!   Extending `EntityEntry` with an optional `name` field would fix this.

use prism_ecs_core::{Entity, EntityKind, World};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::block::{BlockArguments, BlockMarker, BlockOps};
use crate::ir_types::Type;
use crate::op::{is_op, OpMarker, OpName, Operands, RegionRef, Results};
use crate::region::{region_blocks, RegionBlocks, RegionKind, RegionKindComp, RegionMarker};
use crate::value::{value_users, Uses, ValueDef, ValueKind, ValueType};

// ── Snapshot types ──────────────────────────────────────────────────────────

/// A complete, deterministic snapshot of an IR module.
///
/// All entities (ops, regions, blocks) and values are flattened into
/// ordered lists. The order guarantees definition-before-use:
///
/// - The root op appears first.
/// - Regions, blocks, and inner ops appear in nesting order.
/// - Values appear after their defining entity and before any use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrModuleSnapshot {
    /// All entities except values, in topological order.
    pub entities: Vec<EntityEntry>,
    /// All SSA values, in definition order.
    pub values: Vec<ValueEntry>,
}

/// A flattened entity entry in the snapshot.
///
/// The `kind` field is a string tag: `"op"`, `"region"`, or `"block"`.
/// Values are stored separately in [`IrModuleSnapshot::values`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityEntry {
    /// The entity's unique ID (the u64 component of `Entity(u64, u32)`).
    pub id: u64,
    /// The entity kind tag.
    pub kind: String, // "op", "region", "block", "value"
}

/// A flattened SSA value entry in the snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueEntry {
    /// The value entity's unique ID.
    pub id: u64,
    /// The value's IR type, serialized as JSON (from `ir_types::Type`).
    pub type_json: serde_json::Value,
    /// The defining entity ID (the op or block that produces this value).
    pub defining_entity: u64,
    /// The index of this value in the defining entity's result/argument list.
    pub def_index: u32,
    /// IDs of entities (ops) that consume this value.
    pub uses: Vec<u64>,
}

// ── Error type ──────────────────────────────────────────────────────────────

/// Errors that can occur during IR serialization/deserialization.
#[derive(Debug)]
pub enum SerdeError {
    /// The root operation entity is not alive in the world.
    RootEntityNotAlive(Entity),
    /// The root entity is not an operation (no OpMarker component).
    RootNotAnOp(Entity),
    /// JSON serialization failed.
    JsonSerialize(serde_json::Error),
    /// JSON deserialization failed.
    JsonDeserialize(serde_json::Error),
    /// A referenced entity was not found in the snapshot.
    EntityNotFound(u64),
    /// Expected value type missing on a value entity.
    MissingValueType(u64),
    /// Expected value provenance missing on a value entity.
    MissingValueDef(u64),
}

impl fmt::Display for SerdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerdeError::RootEntityNotAlive(e) => {
                write!(f, "root entity {e:?} is not alive in the world")
            }
            SerdeError::RootNotAnOp(e) => {
                write!(f, "root entity {e:?} is not an operation")
            }
            SerdeError::JsonSerialize(e) => write!(f, "JSON serialization error: {e}"),
            SerdeError::JsonDeserialize(e) => write!(f, "JSON deserialization error: {e}"),
            SerdeError::EntityNotFound(id) => {
                write!(f, "entity with ID {id} not found in snapshot")
            }
            SerdeError::MissingValueType(id) => {
                write!(f, "value entity {id} has no ValueType component")
            }
            SerdeError::MissingValueDef(id) => {
                write!(f, "value entity {id} has no ValueDef component")
            }
        }
    }
}

impl std::error::Error for SerdeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SerdeError::JsonSerialize(e) => Some(e),
            SerdeError::JsonDeserialize(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for SerdeError {
    fn from(e: serde_json::Error) -> Self {
        SerdeError::JsonSerialize(e)
    }
}

// ── Topological traversal ──────────────────────────────────────────────────

/// Walk the IR module from `root_op`, collecting all entities and values in
/// topological order (definition before use, block order within regions).
fn collect_snapshot(
    root_op: Entity,
    world: &World,
) -> Result<(Vec<EntityEntry>, Vec<ValueEntry>), SerdeError> {
    let mut entities: Vec<EntityEntry> = Vec::new();
    let mut values: Vec<ValueEntry> = Vec::new();
    let mut visited_entities: HashSet<u64> = HashSet::new();
    let mut visited_values: HashSet<u64> = HashSet::new();

    // ── Helper: record a value and its transitive dependencies ──────────
    fn record_value(
        val_entity: Entity,
        world: &World,
        entities: &mut Vec<EntityEntry>,
        values: &mut Vec<ValueEntry>,
        visited_entities: &mut HashSet<u64>,
        visited_values: &mut HashSet<u64>,
    ) -> Result<(), SerdeError> {
        let vid = val_entity.id();
        if !visited_values.insert(vid) {
            return Ok(());
        }

        // Record the value entity itself.
        if visited_entities.insert(vid) {
            entities.push(EntityEntry {
                id: vid,
                kind: "value".to_string(),
            });
        }

        // Extract ValueDef.
        let def = world
            .get_component::<ValueDef>(val_entity)
            .ok_or(SerdeError::MissingValueDef(vid))?;

        // Extract ValueType.
        let ty = world
            .get_component::<ValueType>(val_entity)
            .ok_or(SerdeError::MissingValueType(vid))?;

        // Extract uses.
        let uses: Vec<u64> = value_users(world, val_entity)
            .iter()
            .map(|e| e.id())
            .collect();

        // Serialize type to JSON.
        let type_json = serde_json::to_value(&ty.0).map_err(SerdeError::JsonSerialize)?;

        values.push(ValueEntry {
            id: vid,
            type_json,
            defining_entity: def.defining_entity.id(),
            def_index: def.index,
            uses,
        });

        Ok(())
    }

    // ── Walk a single operation ──────────────────────────────────────────
    fn walk_op(
        op_entity: Entity,
        world: &World,
        entities: &mut Vec<EntityEntry>,
        values: &mut Vec<ValueEntry>,
        visited_entities: &mut HashSet<u64>,
        visited_values: &mut HashSet<u64>,
    ) -> Result<(), SerdeError> {
        let oid = op_entity.id();
        if !visited_entities.insert(oid) {
            return Ok(());
        }

        entities.push(EntityEntry {
            id: oid,
            kind: "op".to_string(),
        });

        // Record result values produced by this op.
        if let Some(res) = world.get_component::<Results>(op_entity) {
            for &val_entity in &res.0 {
                record_value(
                    val_entity,
                    world,
                    entities,
                    values,
                    visited_entities,
                    visited_values,
                )?;
            }
        }

        // Walk nested regions (via RegionRef component).
        if let Some(region_ref) = world.get_component::<RegionRef>(op_entity) {
            for &region_entity in &region_ref.0 {
                walk_region(
                    region_entity,
                    world,
                    entities,
                    values,
                    visited_entities,
                    visited_values,
                )?;
            }
        }

        Ok(())
    }

    // ── Walk a region ─────────────────────────────────────────────────────
    fn walk_region(
        region_entity: Entity,
        world: &World,
        entities: &mut Vec<EntityEntry>,
        values: &mut Vec<ValueEntry>,
        visited_entities: &mut HashSet<u64>,
        visited_values: &mut HashSet<u64>,
    ) -> Result<(), SerdeError> {
        let rid = region_entity.id();
        if !visited_entities.insert(rid) {
            return Ok(());
        }

        entities.push(EntityEntry {
            id: rid,
            kind: "region".to_string(),
        });

        // Walk blocks in order (via RegionBlocks component).
        let blocks = region_blocks(world, region_entity);
        for &block_entity in &blocks {
            walk_block(
                block_entity,
                world,
                entities,
                values,
                visited_entities,
                visited_values,
            )?;
        }

        Ok(())
    }

    // ── Walk a block ──────────────────────────────────────────────────────
    fn walk_block(
        block_entity: Entity,
        world: &World,
        entities: &mut Vec<EntityEntry>,
        values: &mut Vec<ValueEntry>,
        visited_entities: &mut HashSet<u64>,
        visited_values: &mut HashSet<u64>,
    ) -> Result<(), SerdeError> {
        let bid = block_entity.id();
        if !visited_entities.insert(bid) {
            return Ok(());
        }

        entities.push(EntityEntry {
            id: bid,
            kind: "block".to_string(),
        });

        // Record block argument values (values defined by this block).
        if let Some(args) = world.get_component::<BlockArguments>(block_entity) {
            for &arg_entity in &args.0 {
                record_value(
                    arg_entity,
                    world,
                    entities,
                    values,
                    visited_entities,
                    visited_values,
                )?;
            }
        }

        // Walk ops in block order (via BlockOps component).
        if let Some(bops) = world.get_component::<BlockOps>(block_entity) {
            for &op_entity in &bops.0 {
                walk_op(
                    op_entity,
                    world,
                    entities,
                    values,
                    visited_entities,
                    visited_values,
                )?;
            }
        }

        Ok(())
    }

    // ── Start the walk from the root op ───────────────────────────────────
    walk_op(
        root_op,
        world,
        &mut entities,
        &mut values,
        &mut visited_entities,
        &mut visited_values,
    )?;

    Ok((entities, values))
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Serialize an IR module rooted at `root_op` to a deterministic JSON string.
///
/// The snapshot is ordered topologically:
/// - Ops appear in block order.
/// - Regions and blocks appear in nesting order.
/// - Values appear after their defining entity and before first use.
///
/// # Errors
///
/// Returns [`SerdeError::RootEntityNotAlive`] if `root_op` is stale or dead.
/// Returns [`SerdeError::RootNotAnOp`] if `root_op` is not an operation.
pub fn to_json(root_op: Entity, world: &World) -> Result<String, SerdeError> {
    if !world.is_alive(root_op) {
        return Err(SerdeError::RootEntityNotAlive(root_op));
    }
    if !is_op(world, root_op) {
        return Err(SerdeError::RootNotAnOp(root_op));
    }

    let (entity_entries, value_entries) = collect_snapshot(root_op, world)?;

    let snapshot = IrModuleSnapshot {
        entities: entity_entries,
        values: value_entries,
    };

    serde_json::to_string_pretty(&snapshot).map_err(SerdeError::JsonSerialize)
}

/// Deserialize a JSON snapshot back into the `world`.
///
/// All entities are recreated at their original IDs using
/// [`World::spawn_entity_with_id`]. The returned `Entity` handle is the
/// reconstructed root operation entity.
///
/// # Note
///
/// `OpName` is not preserved in the snapshot. After deserialization,
/// operation entities carry `OpMarker`, `Operands`, and `Results` but
/// use a default name (`"builtin.unregistered"`). Callers that need
/// operation names should set them after deserialization or extend the
/// `EntityEntry` struct with a `name` field.
///
/// # Errors
///
/// Returns [`SerdeError::JsonDeserialize`] on parse failure.
/// Returns [`SerdeError::EntityNotFound`] if the snapshot is empty.
pub fn from_json(json: &str, world: &mut World) -> Result<Entity, SerdeError> {
    let snapshot: IrModuleSnapshot =
        serde_json::from_str(json).map_err(SerdeError::JsonDeserialize)?;

    if snapshot.entities.is_empty() {
        return Err(SerdeError::EntityNotFound(0));
    }

    // Build a lookup: entity ID → kind tag.
    let mut entity_kinds: HashMap<u64, &str> = HashMap::new();
    for entry in &snapshot.entities {
        entity_kinds.insert(entry.id, &entry.kind);
    }

    let root_id = snapshot.entities[0].id;

    // ── Phase 1: Spawn all non-value entities ────────────────────────────
    let mut entity_map: HashMap<u64, Entity> = HashMap::new();
    for entry in &snapshot.entities {
        if entry.kind == "value" {
            continue;
        }
        let e = world.spawn_entity_with_id(entry.id, EntityKind::Node);
        entity_map.insert(entry.id, e);
    }

    // ── Phase 2: Spawn value entities and add ValueDef/ValueType/Uses ───
    for entry in &snapshot.values {
        let val_entity = world.spawn_entity_with_id(entry.id, EntityKind::Node);
        // Add to entity_map so later phases can reference this value.
        entity_map.insert(entry.id, val_entity);

        // Determine ValueKind from the defining entity's type.
        let kind = match entity_kinds.get(&entry.defining_entity).copied() {
            Some("block") => ValueKind::BlockArgument,
            _ => ValueKind::OpResult,
        };

        let value_def = ValueDef {
            kind,
            defining_entity: *entity_map
                .get(&entry.defining_entity)
                .unwrap_or(&Entity(entry.defining_entity, 0)),
            index: entry.def_index,
        };
        world
            .add_component(val_entity, value_def)
            .expect("add ValueDef");

        // Reconstruct ValueType.
        let ty: Type =
            serde_json::from_value(entry.type_json.clone()).map_err(SerdeError::JsonDeserialize)?;
        world
            .add_component(val_entity, ValueType(ty))
            .expect("add ValueType");

        // Reconstruct Uses.
        let use_entities: Vec<Entity> = entry
            .uses
            .iter()
            .map(|&id| *entity_map.get(&id).unwrap_or(&Entity(id, 0)))
            .collect();
        world
            .add_component(val_entity, Uses(use_entities))
            .expect("add Uses");
    }

    // ── Phase 3: Add op components ───────────────────────────────────────
    for entry in &snapshot.entities {
        if entry.kind != "op" {
            continue;
        }
        let &op_entity = entity_map
            .get(&entry.id)
            .expect("op entity should exist in entity_map");

        // OpMarker
        world
            .add_component(op_entity, OpMarker)
            .expect("add OpMarker");

        // OpName — not preserved in the snapshot; use a placeholder.
        world
            .add_component(op_entity, OpName("builtin.unregistered".into()))
            .expect("add OpName");

        // Operands: values whose use-list includes this op.
        let mut operands: Vec<Entity> = Vec::new();
        for v in &snapshot.values {
            if v.uses.contains(&entry.id) {
                operands.push(*entity_map.get(&v.id).unwrap_or(&Entity(v.id, 0)));
            }
        }
        world
            .add_component(op_entity, Operands(operands))
            .expect("add Operands");

        // Results: values whose defining_entity is this op.
        let mut results: Vec<Entity> = Vec::new();
        for v in &snapshot.values {
            if v.defining_entity == entry.id {
                results.push(*entity_map.get(&v.id).unwrap_or(&Entity(v.id, 0)));
            }
        }
        // Sort results by def_index for consistent ordering.
        results.sort_by_key(|e| {
            snapshot
                .values
                .iter()
                .find(|v| v.id == e.id())
                .map(|v| v.def_index)
                .unwrap_or(0)
        });
        world
            .add_component(op_entity, Results(results))
            .expect("add Results");
    }

    // ── Phase 4: Reconstruct region/block hierarchy ─────────────────────
    rebuild_region_block_hierarchy(&snapshot, world, &entity_map);

    // ── Phase 5: Add marker/kind components for region and block entities ──
    use crate::block::{BlockArguments, BlockMarker, TerminatorOp};
    use crate::region::{RegionKind, RegionKindComp, RegionMarker};
    for entry in &snapshot.entities {
        let &ent = entity_map.get(&entry.id).expect("entity in snapshot");
        match entry.kind.as_str() {
            "region" => {
                world.add_component(ent, RegionMarker).ok();
                world
                    .add_component(ent, RegionKindComp(RegionKind::SSACFG))
                    .ok();
            }
            "block" => {
                world.add_component(ent, BlockMarker).ok();
                world.add_component(ent, BlockArguments(vec![])).ok();
                world.add_component(ent, TerminatorOp(None)).ok();
            }
            _ => {}
        }
    }

    Ok(*entity_map.get(&root_id).expect("root entity should exist"))
}

/// Reconstruct RegionRef, RegionBlocks, and BlockOps components from the
/// the topological order in the snapshot.
///
/// The snapshot's entity list is ordered: an op is followed by its regions,
/// each region by its blocks, each block by its ops. We walk this linearized
/// tree to infer parent-child edges.
fn rebuild_region_block_hierarchy(
    snapshot: &IrModuleSnapshot,
    world: &mut World,
    entity_map: &HashMap<u64, Entity>,
) {
    // Accumulate edges: (parent_id, [child_ids])
    let mut op_regions = Vec::<(u64, Vec<u64>)>::new();
    let mut region_blocks = Vec::<(u64, Vec<u64>)>::new();
    let mut block_ops = Vec::<(u64, Vec<u64>)>::new();

    // Track the current container at each level as we walk.
    let mut current_op: Option<u64> = None;
    let mut current_region: Option<u64> = None;
    let mut current_block: Option<u64> = None;

    // ── Restore marker components from the snapshot entity kind tags ───────
    for entry in &snapshot.entities {
        let entity = *entity_map.get(&entry.id).expect("entity should be in map");
        match entry.kind.as_str() {
            "region" => {
                world.add_component(entity, RegionMarker).ok();
                world
                    .add_component(entity, RegionKindComp(RegionKind::SSACFG))
                    .ok();
            }
            "block" => {
                world.add_component(entity, BlockMarker).ok();
                world.add_component(entity, BlockArguments(vec![])).ok();
            }
            _ => {}
        }
    }

    for entry in &snapshot.entities {
        match entry.kind.as_str() {
            "op" => {
                if let Some(bid) = current_block {
                    block_ops.push((bid, vec![entry.id]));
                }
                current_op = Some(entry.id);
                current_region = None;
            }
            "region" => {
                if let Some(oid) = current_op {
                    op_regions.push((oid, vec![entry.id]));
                }
                current_region = Some(entry.id);
                current_block = None;
            }
            "block" => {
                if let Some(rid) = current_region {
                    region_blocks.push((rid, vec![entry.id]));
                }
                current_block = Some(entry.id);
            }
            "value" => {
                // Values don't affect hierarchy.
            }
            _ => {}
        }
    }

    // ── Merge consecutive edges for the same parent ──────────────────────

    // RegionRef on ops.
    let mut merged_op_regions: HashMap<u64, Vec<u64>> = HashMap::new();
    for (op_id, reg_ids) in op_regions {
        merged_op_regions.entry(op_id).or_default().extend(reg_ids);
    }
    for (op_id, reg_ids) in merged_op_regions {
        let region_entities: Vec<Entity> = reg_ids
            .iter()
            .map(|&id| *entity_map.get(&id).unwrap_or(&Entity(id, 0)))
            .collect();
        world
            .add_component(
                *entity_map.get(&op_id).unwrap_or(&Entity(op_id, 0)),
                RegionRef(region_entities),
            )
            .ok();
    }

    // RegionBlocks on regions.
    let mut merged_region_blocks: HashMap<u64, Vec<u64>> = HashMap::new();
    for (rid, block_ids) in region_blocks {
        merged_region_blocks
            .entry(rid)
            .or_default()
            .extend(block_ids);
    }
    for (rid, block_ids) in merged_region_blocks {
        let block_entities: Vec<Entity> = block_ids
            .iter()
            .map(|&id| *entity_map.get(&id).unwrap_or(&Entity(id, 0)))
            .collect();
        world
            .add_component(
                *entity_map.get(&rid).unwrap_or(&Entity(rid, 0)),
                RegionBlocks(block_entities),
            )
            .ok();
    }

    // BlockOps on blocks.
    let mut merged_block_ops: HashMap<u64, Vec<u64>> = HashMap::new();
    for (bid, op_ids) in block_ops {
        merged_block_ops.entry(bid).or_default().extend(op_ids);
    }
    for (bid, op_ids) in merged_block_ops {
        let op_entities: Vec<Entity> = op_ids
            .iter()
            .map(|&id| *entity_map.get(&id).unwrap_or(&Entity(id, 0)))
            .collect();
        world
            .add_component(
                *entity_map.get(&bid).unwrap_or(&Entity(bid, 0)),
                BlockOps(op_entities),
            )
            .ok();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::TerminatorOp;
    use crate::ir_types::Type;
    use crate::region::{RegionKind, RegionKindComp};

    /// Build a minimal module using the actual region/block types:
    ///
    /// ```text
    /// "module" (root op)
    ///   └── RegionRef → [region]
    ///         └── RegionBlocks → [block]
    ///               └── BlockOps → [op1, op2]
    ///                     ├── "arith.constant" → v1 (f32)
    ///                     └── "arith.addf" v1 → v2 (f32)
    /// ```
    fn build_test_module(world: &mut World) -> Entity {
        // ── Root op ──────────────────────────────────────────────────────
        let root_op: Entity = world
            .spawn(EntityKind::Node, Some("module".into()))
            .expect("spawn")
            .into();
        world.add_component(root_op, OpMarker).unwrap();
        world
            .add_component(root_op, OpName("module".into()))
            .unwrap();
        world.add_component(root_op, Operands(vec![])).unwrap();
        world.add_component(root_op, Results(vec![])).unwrap();

        // ── Region ───────────────────────────────────────────────────────
        let region: Entity = world
            .spawn(EntityKind::Node, Some("body".into()))
            .expect("spawn")
            .into();
        world.add_component(region, RegionMarker).unwrap();
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .unwrap();
        world.add_component(region, RegionBlocks(vec![])).unwrap();

        // ── Block ────────────────────────────────────────────────────────
        let block: Entity = world
            .spawn(EntityKind::Node, Some("entry".into()))
            .expect("spawn")
            .into();
        world.add_component(block, BlockMarker).unwrap();
        world.add_component(block, BlockArguments(vec![])).unwrap();
        world.add_component(block, BlockOps(vec![])).unwrap();
        world.add_component(block, TerminatorOp(None)).unwrap();

        // Link: region → block.
        world
            .add_component(region, RegionBlocks(vec![block]))
            .unwrap();

        // Link: root op → region (via RegionRef serialization component).
        world
            .add_component(root_op, RegionRef(vec![region]))
            .unwrap();

        // ── Op 1: arith.constant producing v1 ────────────────────────────
        let op1: Entity = world
            .spawn(EntityKind::Node, Some("op_arith.constant".into()))
            .expect("spawn")
            .into();
        world.add_component(op1, OpMarker).unwrap();
        world
            .add_component(op1, OpName("arith.constant".into()))
            .unwrap();
        world.add_component(op1, Operands(vec![])).unwrap();
        world.add_component(op1, Results(vec![])).unwrap();

        let v1: Entity = world
            .spawn(EntityKind::Node, Some("constant.r0".into()))
            .expect("spawn")
            .into();
        world
            .add_component(v1, ValueDef::op_result(op1, 0))
            .unwrap();
        world.add_component(v1, ValueType(Type::f32())).unwrap();
        world.add_component(v1, Uses(vec![])).unwrap();
        world.add_component(op1, Results(vec![v1])).unwrap();

        // Op1 → block.
        world.add_component(block, BlockOps(vec![op1])).unwrap();

        // ── Op 2: arith.addf v1 → v2 ────────────────────────────────────
        let op2: Entity = world
            .spawn(EntityKind::Node, Some("op_arith.addf".into()))
            .expect("spawn")
            .into();
        world.add_component(op2, OpMarker).unwrap();
        world
            .add_component(op2, OpName("arith.addf".into()))
            .unwrap();
        world.add_component(op2, Operands(vec![v1])).unwrap();
        world.add_component(op2, Results(vec![])).unwrap();

        let v2: Entity = world
            .spawn(EntityKind::Node, Some("addf.r0".into()))
            .expect("spawn")
            .into();
        world
            .add_component(v2, ValueDef::op_result(op2, 0))
            .unwrap();
        world.add_component(v2, ValueType(Type::f32())).unwrap();
        world.add_component(v2, Uses(vec![])).unwrap();
        world.add_component(op2, Results(vec![v2])).unwrap();

        // Update v1 uses to include op2.
        world.add_component(v1, Uses(vec![op2])).unwrap();

        // Add op2 to block (update block ops to include both).
        world
            .add_component(block, BlockOps(vec![op1, op2]))
            .unwrap();

        root_op
    }

    #[test]
    fn roundtrip_small_module() {
        let mut world = World::new();
        let root_op = build_test_module(&mut world);

        // Serialize.
        let json = to_json(root_op, &world).expect("to_json failed");

        // Deserialize into a fresh world.
        let mut world2 = World::new();
        let restored = from_json(&json, &mut world2).expect("from_json failed");

        // Verify the restored entity is alive and is an op.
        assert!(world2.is_alive(restored));
        assert!(is_op(&world2, restored));

        // Verify region structure.
        let region_ref = world2
            .get_component::<RegionRef>(restored)
            .expect("root op should have RegionRef");
        assert_eq!(region_ref.0.len(), 1);

        let region_entity = region_ref.0[0];
        assert!(
            world2
                .get_component::<RegionMarker>(region_entity)
                .is_some(),
            "region should have RegionMarker"
        );

        let region_blocks_comp = world2
            .get_component::<RegionBlocks>(region_entity)
            .expect("region should have RegionBlocks");
        assert_eq!(region_blocks_comp.0.len(), 1);

        let block_entity = region_blocks_comp.0[0];
        assert!(
            world2.get_component::<BlockMarker>(block_entity).is_some(),
            "block should have BlockMarker"
        );

        let block_ops_comp = world2
            .get_component::<BlockOps>(block_entity)
            .expect("block should have BlockOps");
        assert_eq!(block_ops_comp.0.len(), 2);

        // Verify ops.
        let ops_in_block = &block_ops_comp.0;
        let names: Vec<String> = ops_in_block
            .iter()
            .map(|&op| {
                world2
                    .get_component::<OpName>(op)
                    .map(|n| n.0.clone())
                    .unwrap_or_default()
            })
            .collect();
        // Names are not preserved across serialization, so we check
        // the default "builtin.unregistered" placeholder.
        assert_eq!(
            names,
            vec![
                "builtin.unregistered".to_string(),
                "builtin.unregistered".to_string()
            ]
        );

        // Verify value structure.
        let const_op = ops_in_block[0];
        let addf_op = ops_in_block[1];

        let const_results = world2
            .get_component::<Results>(const_op)
            .map(|r| r.0.clone())
            .unwrap_or_default();
        assert_eq!(const_results.len(), 1);
        let v1 = const_results[0];
        assert!(world2.get_component::<ValueDef>(v1).is_some());
        assert_eq!(
            world2.get_component::<ValueType>(v1).map(|t| t.0.clone()),
            Some(Type::f32())
        );

        let addf_results = world2
            .get_component::<Results>(addf_op)
            .map(|r| r.0.clone())
            .unwrap_or_default();
        assert_eq!(addf_results.len(), 1);
        let v2 = addf_results[0];
        assert!(world2.get_component::<ValueDef>(v2).is_some());
        assert_eq!(
            world2.get_component::<ValueType>(v2).map(|t| t.0.clone()),
            Some(Type::f32())
        );

        // Verify operand chain: addf consumes v1.
        let addf_operands = world2
            .get_component::<Operands>(addf_op)
            .map(|o| o.0.clone())
            .unwrap_or_default();
        assert_eq!(addf_operands, vec![v1]);

        // Verify use chain: v1 → addf, v2 → (none).
        assert_eq!(value_users(&world2, v1), vec![addf_op]);
        assert!(value_users(&world2, v2).is_empty());
    }

    #[test]
    fn to_json_errors_on_bad_root() {
        let world = World::new();
        let dead_entity = Entity(999, 1);

        let err = to_json(dead_entity, &world).unwrap_err();
        assert!(matches!(err, SerdeError::RootEntityNotAlive(_)));

        // Entity that is alive but not an op.
        let mut world2 = World::new();
        let non_op = world2.spawn_entity(EntityKind::Node);
        let err = to_json(non_op, &world2).unwrap_err();
        assert!(matches!(err, SerdeError::RootNotAnOp(_)));
    }

    #[test]
    fn from_json_errors_on_bad_input() {
        let mut world = World::new();

        // Malformed JSON.
        let err = from_json("not json", &mut world).unwrap_err();
        assert!(matches!(err, SerdeError::JsonDeserialize(_)));

        // Technically valid JSON but missing required fields.
        let err = from_json("{}", &mut world).unwrap_err();
        assert!(matches!(err, SerdeError::JsonDeserialize(_)));

        // Empty entities array.
        let err = from_json(r#"{"entities":[],"values":[]}"#, &mut world).unwrap_err();
        assert!(matches!(err, SerdeError::EntityNotFound(0)));
    }

    #[test]
    fn deterministic_snapshot_order() {
        let mut world = World::new();
        let root_op = build_test_module(&mut world);

        let json1 = to_json(root_op, &world).expect("first serialization");
        let json2 = to_json(root_op, &world).expect("second serialization");

        // Deterministic = byte-identical JSON.
        assert_eq!(json1, json2, "snapshots must be deterministic");

        // Verify key structural elements in the JSON.
        assert!(json1.contains("\"kind\": \"op\""));
        assert!(json1.contains("\"kind\": \"region\""));
        assert!(json1.contains("\"kind\": \"block\""));
        assert!(json1.contains("\"kind\": \"value\""));
    }

    #[test]
    fn block_argument_roundtrip() {
        let mut world = World::new();

        // Build: root op → region → block (with block argument v1) → op uses v1.
        let root_op: Entity = world
            .spawn(EntityKind::Node, Some("module".into()))
            .expect("spawn")
            .into();
        world.add_component(root_op, OpMarker).unwrap();
        world
            .add_component(root_op, OpName("module".into()))
            .unwrap();
        world.add_component(root_op, Operands(vec![])).unwrap();
        world.add_component(root_op, Results(vec![])).unwrap();

        let region: Entity = world
            .spawn(EntityKind::Node, Some("body".into()))
            .expect("spawn")
            .into();
        world.add_component(region, RegionMarker).unwrap();
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .unwrap();
        world.add_component(region, RegionBlocks(vec![])).unwrap();

        let block: Entity = world
            .spawn(EntityKind::Node, Some("entry".into()))
            .expect("spawn")
            .into();
        world.add_component(block, BlockMarker).unwrap();
        world.add_component(block, BlockArguments(vec![])).unwrap();
        world.add_component(block, BlockOps(vec![])).unwrap();
        world.add_component(block, TerminatorOp(None)).unwrap();

        // Block argument v1.
        let v1: Entity = world
            .spawn(EntityKind::Node, Some("arg0".into()))
            .expect("spawn")
            .into();
        world
            .add_component(v1, ValueDef::block_argument(block, 0))
            .unwrap();
        world.add_component(v1, ValueType(Type::f32())).unwrap();
        world.add_component(v1, Uses(vec![])).unwrap();

        // Update block arguments.
        world
            .add_component(block, BlockArguments(vec![v1]))
            .unwrap();

        // Op that uses v1.
        let op: Entity = world
            .spawn(EntityKind::Node, Some("op_arith.addf".into()))
            .expect("spawn")
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("arith.addf".into()))
            .unwrap();
        world.add_component(op, Operands(vec![v1])).unwrap();
        world.add_component(op, Results(vec![])).unwrap();
        world.add_component(block, BlockOps(vec![op])).unwrap();
        world.add_component(v1, Uses(vec![op])).unwrap();

        world
            .add_component(region, RegionBlocks(vec![block]))
            .unwrap();
        world
            .add_component(root_op, RegionRef(vec![region]))
            .unwrap();

        let json = to_json(root_op, &world).expect("to_json");

        let mut world2 = World::new();
        let restored = from_json(&json, &mut world2).expect("from_json");

        // Verify block argument survived round-trip.
        let restored_region = world2.get_component::<RegionRef>(restored).unwrap().0[0];
        let restored_region_blocks = world2
            .get_component::<RegionBlocks>(restored_region)
            .unwrap();
        let restored_block = restored_region_blocks.0[0];

        // Check the op in the block has a block-argument operand.
        let restored_block_ops = world2.get_component::<BlockOps>(restored_block).unwrap();
        let restored_op = restored_block_ops.0[0];
        let restored_operands = world2
            .get_component::<Operands>(restored_op)
            .map(|o| o.0.clone())
            .unwrap_or_default();
        assert_eq!(restored_operands.len(), 1);

        let restored_v1 = restored_operands[0];
        let def = world2
            .get_component::<ValueDef>(restored_v1)
            .expect("v1 should have ValueDef");
        assert_eq!(def.kind, ValueKind::BlockArgument);
        assert_eq!(def.defining_entity, restored_block);
        assert_eq!(def.index, 0);
    }

    #[test]
    fn json_roundtrip_maintains_all_value_properties() {
        let mut world = World::new();
        let root_op = build_test_module(&mut world);

        let json = to_json(root_op, &world).expect("to_json");

        // Parse the JSON and verify all expected fields.
        let snapshot: IrModuleSnapshot = serde_json::from_str(&json).expect("valid snapshot JSON");

        // Should have entities: root_op, region, block, op1, op2, v1, v2
        assert!(snapshot.entities.len() >= 5, "expected at least 5 entities");

        // Verify value entries.
        assert_eq!(snapshot.values.len(), 2, "expected 2 value entries");

        // Both values should have valid type_json.
        for v in &snapshot.values {
            assert!(
                !v.type_json.is_null(),
                "value {} should have non-null type",
                v.id
            );
        }

        // Operands: find the addf op and verify its use in value entries.
        let op_entries: Vec<&EntityEntry> = snapshot
            .entities
            .iter()
            .filter(|e| e.kind == "op")
            .collect();
        assert!(op_entries.len() >= 3, "expected at least 3 ops");
    }
}
