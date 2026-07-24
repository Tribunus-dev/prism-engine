# Ontology of Prism

This vocabulary is canonical. A new interface, document, or implementation should use these nouns rather than inventing synonyms.

| Noun | Definition | Relationships |
| --- | --- | --- |
| Intent | What the user or model means. | Becomes a Representation. |
| Representation | The semantic expression of Intent. | Is explored by a Plan. |
| Plan | Legal transformations under exploration. | Commits to a ComputeImage. |
| Persistent Computational Subject | The identity that remains invariant across every phase. | Owns Intent, Representations, Plans, Execution, and Receipts. |
| Candidate | A possible Plan under comparison. | Has Knowledge and an Existence State. |
| ComputeImage | The sealed executable embodiment of an admitted Plan. | Is realized by Execution and validated by Receipts. |
| Execution | Realization of a ComputeImage on a Provider. | Produces observable outcomes. |
| Execution Packet | Ordered work and capability handoffs derived from a ComputeImage. | Is consumed by Execution. |
| Receipt | Observable evidence about a decision or realization. | Records provenance and claim scope. |
| Capability Surface | The explicit capabilities a target exposes. | Constrains Plans and Execution. |
| Provider | A hardware or runtime boundary that realizes Execution. | Owns capability surfaces. |
| Knowledge Source | How a statement is known. | Measured Observation, Compile Verification, Repository Evidence, Architectural Derivation, Illustrative Example, Research Direction, or Speculation. |
| Knowledge State | What is visible, inferred, hidden, or not yet observable. | Determines progressive disclosure. |
| Existence State | The lifecycle status of an object. | Possible, Planned, Hypothetical, Active, Archived, Unknown, Partial, Deferred, Rejected, or Experimental. |
| Evidence | The bounded material that supports a claim. | Is carried by a Receipt. |

## Invariant

The Persistent Computational Subject is the protagonist. Intent, Representation, Plan, ComputeImage, Execution, and Receipt are phases or observations of that subject, not replacement identities.
