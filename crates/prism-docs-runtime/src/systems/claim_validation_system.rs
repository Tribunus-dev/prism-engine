//! `claim_validation_system` — validates every claim entity.
//!
//! Reads every `Claim` entity, re-applies the constitutional rule
//! (Measured claims must carry at least one source ref), and emits
//! a typed event per validation failure. Failures are fatal: the
//! SSG refuses to emit a page with a broken claim.

use prism_ecs_core::World;

use crate::components::claim::{ClaimClassComponent, ClaimSourceRefs, ClaimText};
use crate::components::identity::SiteEntityId;
use crate::error::RuntimeError;

pub fn run(world: &mut World) -> Result<(), RuntimeError> {
    // Walk every entity that has all three components, then check
    // its SiteEntityId-derived SiteEntityKind. The query is
    // efficient because the storage is columnar.
    for (entity, _id, class, text) in world
        .query3::<SiteEntityId, ClaimClassComponent, ClaimText>()
    {
        // We use the SiteEntityId presence as a proxy for "is a
        // claim-like entity" — the claim text + class
        // combination is the discriminating pair. A future change
        // will introduce a `Claim` marker component for a sharper
        // query.
        if class.0 == "measured" {
            let source_refs = world.get_component::<ClaimSourceRefs>(entity);
            let is_empty = source_refs.map(|s| s.0.is_empty()).unwrap_or(true);
            if is_empty {
                return Err(RuntimeError::invalid_value(
                    entity,
                    "claim",
                    format!(
                        "Measured claim `{}` must include at least one source_ref",
                        text.0
                    ),
                ));
            }
        }
    }
    Ok(())
}
