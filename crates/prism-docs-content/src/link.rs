//! Typed links between entities.
//!
//! A `Link` declares a relationship from one entity to another. The
//! manifest's link resolution step validates every link target.

use serde::{Deserialize, Serialize};

use crate::error::ContentError;
use crate::manifest::EntityId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LinkKind {
    /// A chapter that frames a claim.
    Frames,
    /// A page that comes before another in the narrative.
    Follows,
    /// A claim that depends on another claim.
    Depends,
    /// A chapter that depends on an ADR.
    Constrained,
    /// An ADR that supersedes another.
    Supersedes,
    /// A page that links to a chapter.
    Composes,
}

impl LinkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LinkKind::Frames => "frames",
            LinkKind::Follows => "follows",
            LinkKind::Depends => "depends",
            LinkKind::Constrained => "constrained",
            LinkKind::Supersedes => "supersedes",
            LinkKind::Composes => "composes",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub from: EntityId,
    pub to: EntityId,
    pub kind: LinkKind,
}

impl Link {
    pub fn validate(&self) -> Result<(), ContentError> {
        if self.from == self.to {
            return Err(ContentError::SelfLink { id: self.from.clone() });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link() -> Link {
        Link {
            from: EntityId::new("chapter:home-intent").unwrap(),
            to: EntityId::new("claim:inspectable").unwrap(),
            kind: LinkKind::Frames,
        }
    }

    #[test]
    fn validate_ok() {
        assert!(link().validate().is_ok());
    }

    #[test]
    fn self_link_rejected() {
        let id = EntityId::new("chapter:home-intent").unwrap();
        let l = Link {
            from: id.clone(),
            to: id,
            kind: LinkKind::Frames,
        };
        let err = l.validate().unwrap_err();
        assert!(matches!(err, ContentError::SelfLink { .. }));
    }
}
