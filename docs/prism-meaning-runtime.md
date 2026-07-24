# Prism Meaning Runtime

Status: frozen for this sprint (2026-07-24)

This document is frozen. The authoritative operational definitions are now in:
- [`prism-runtime.md`](prism-runtime.md)
- [`prism-semantics.md`](prism-semantics.md)

The Meaning Runtime is the prior naming for the canonical semantic layers already embodied by the above sources.

Do not introduce new Meaning Runtime abstractions while this freeze is in place; evolve meaning semantics only in the canonical sources.

The Meaning Runtime determines what an object means as it moves through Prism. It sits between the canonical ontology and any renderer.

## Belief state

Every object carries a belief state independent from existence and knowledge: `unknown → hypothesized → derived → observed → verified → measured → historical`.

Belief can only strengthen through a recorded observation or a stronger evidence source. A visual transition never upgrades belief by itself.

## Conservation laws

Intent cannot disappear.

Identity cannot split without provenance.

Evidence cannot increase without observation.

Claims cannot strengthen without new evidence.

These laws are runtime assertions, not decorative language.

## Continuity

The Observatory may remember the last observed instrument, subject stage, and knowledge depth. This is explicit semantic continuity, not hidden profiling or behavioral analytics. The visitor can reset it.

## Living objects

Every canonical noun is inspectable through the same object protocol. An Intent, Representation, Plan, ComputeImage, Receipt, Execution Packet, Capability Surface, Provider, and Evidence object exposes identity, knowledge, belief, existence, relationships, and history.

## Self-description

Every meaningful interaction can expose the active physics rules: identity preserved, observation boundary crossed, evidence changed or unchanged, and optical rule satisfied.
