//! `Claim` — a single statement about the architecture with a typed
//! confidence and class.
//!
//! Claims are the smallest unit of declarative content. The whole site
//! is a graph of claims, chapters, and ADRs.

use serde::{Deserialize, Serialize};

use crate::error::ContentError;
use crate::manifest::EntityId;
use crate::ontology::{ClaimClass, KnowledgeState};
use crate::source_ref::SourceRef;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub id: EntityId,
    pub text: String,
    pub class: ClaimClass,
    pub state: KnowledgeState,
    /// Source references. `Measured` claims require at least one.
    #[serde(default)]
    pub source_refs: Vec<SourceRef>,
    /// Optional link to the chapter that frames this claim.
    #[serde(default)]
    pub framed_by: Option<EntityId>,
}

impl Claim {
    /// Validate the claim's invariants.
    ///
    /// The constitutional rule: a `Measured` claim must carry at least
    /// one source reference and a measurement constraint. We surface
    /// this as a typed error so the build fails loudly.
    pub fn validate(&self) -> Result<(), ContentError> {
        if self.text.trim().is_empty() {
            return Err(ContentError::InvalidValue {
                id: self.id.clone(),
                component: "text".into(),
                reason: "claim text must be non-empty".into(),
            });
        }
        if self.class.requires_source() && self.source_refs.is_empty() {
            return Err(ContentError::ClaimInvalid {
                id: self.id.clone(),
                class: self.class.as_str().into(),
                state: self.state.as_str().into(),
                reason: "Measured claim must include at least one source_ref".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_ref::SourceRef;

    fn claim() -> Claim {
        Claim {
            id: EntityId::new("claim:inspectable").unwrap(),
            text: "ComputeImages are inspectable artifacts.".into(),
            class: ClaimClass::Architectural,
            state: KnowledgeState::Verified,
            source_refs: vec![],
            framed_by: None,
        }
    }

    #[test]
    fn validate_ok() {
        assert!(claim().validate().is_ok());
    }

    #[test]
    fn measured_claim_requires_source() {
        let mut c = claim();
        c.class = ClaimClass::Measured;
        c.text = "ANE prefill runs at 28.4 tok/s on M2.".into();
        let err = c.validate().unwrap_err();
        assert!(matches!(err, ContentError::ClaimInvalid { .. }));
    }

    #[test]
    fn measured_claim_with_source_is_ok() {
        let mut c = claim();
        c.class = ClaimClass::Measured;
        c.text = "ANE prefill runs at 28.4 tok/s on M2.".into();
        c.source_refs = vec![SourceRef::new("bench/ane-m2.json")];
        assert!(c.validate().is_ok());
    }
}
