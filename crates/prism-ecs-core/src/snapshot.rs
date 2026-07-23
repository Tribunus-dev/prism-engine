use crate::World;
/// Stable allocator fingerprint used by runtime snapshots.
pub fn export_allocator_snapshot(world: &World) -> Vec<u8> {
    world.all_entities().iter().flat_map(|e| e.id().to_le_bytes()).collect()
}
