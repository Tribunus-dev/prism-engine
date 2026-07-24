# Prism Interaction Runtime v1

The Prism Observatory is the first client of a reusable interaction runtime. Website, desktop inspector, debugger, ComputeImage explorer, and Fabric console should consume the same semantic state and choose different renderers.

## Kernel modules

The kernel exposes Subject Runtime, Observation Runtime, Knowledge Runtime, Receipt Runtime, Optical Runtime, Interaction Runtime, and Rendering Runtime boundaries.

## Observation entity

An Observation is an ECS-like entity with `subject`, `instrument`, `knowledgeState`, `evidenceState`, `opticalState`, and `transitionState`. Scenes are views over observations; they cannot create replacement subjects.

## Observer modes

Observer, Builder, Researcher, Compiler Engineer, Runtime Engineer, and Infrastructure Engineer change perspective and depth without changing the subject or its claims.

## Interaction events

Every meaningful interaction records what remained invariant, what transformed, what became visible, what became hidden, and what evidence increased. These events form semantic history, not behavioral analytics.

## Visual state machine

The optical runtime moves through `observation → neutral → focus → dispersion → exploration → commit → evidence`. Visual treatments derive from this state machine and never invent independent state.
