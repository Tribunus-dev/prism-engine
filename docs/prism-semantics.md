# Prism Semantics

Status: frozen for this sprint (2026-07-24)

This document is the canonical source for semantic state transitions, belief updates, conservation, and observation meaning. It replaces Meaning Runtime semantics as a separate architecture layer. Do not split semantics into separate runtime layers while this freeze holds.

Compatibility notes
This document is intentionally narrow and should be consumed with `prism-runtime.md` and `prism-observation-protocol.md` as the execution and surface projections.

Semantics answers how the nouns in the Ontology become meaningful. It contains knowledge, belief, evidence, claims, conservation, and mental-model transitions.

## Meaning state

Every observable object carries `knowledge`, `belief`, `existence`, `evidence`, `history`, and `relationships`. Knowledge describes how the object is known. Belief describes confidence progression. Existence describes lifecycle. Evidence describes the bounded support for a claim.

## Mental model transition

Every observation has `priorKnowledge`, `question`, `observation`, `conflict`, `transformation`, `evidence`, `confidence`, and `remainingUnknowns`. The visitor is part of the semantic system: an observation changes both the subject’s visible state and the visitor’s mental model.

## Conservation

Intent cannot disappear. Identity cannot split without provenance. Evidence cannot increase without observation. Claims cannot strengthen without new evidence.

## Knowledge sources

Measured Observation, Compile Verification, Repository Evidence, Architectural Derivation, Illustrative Example, Research Direction, and Speculation remain distinct. Rendering cannot strengthen a source.
