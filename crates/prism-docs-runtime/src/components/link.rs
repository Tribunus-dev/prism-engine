//! Link components — typed edges between entities in the world.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

use crate::components::identity::SiteEntityId;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinkFrom(pub SiteEntityId);
impl Component for LinkFrom {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinkTo(pub SiteEntityId);
impl Component for LinkTo {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LinkKindComponent {
    Frames,
    Follows,
    Depends,
    Constrained,
    Supersedes,
    Composes,
}

impl LinkKindComponent {
    pub fn as_str(self) -> &'static str {
        match self {
            LinkKindComponent::Frames => "frames",
            LinkKindComponent::Follows => "follows",
            LinkKindComponent::Depends => "depends",
            LinkKindComponent::Constrained => "constrained",
            LinkKindComponent::Supersedes => "supersedes",
            LinkKindComponent::Composes => "composes",
        }
    }
}

impl Component for LinkKindComponent {}
