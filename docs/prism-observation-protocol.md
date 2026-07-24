# Prism Observation Interface

Every observable Prism noun implements the same inspection surface:

`identity`, `state`, `knowledge`, `belief`, `evidence`, `history`, `relationships`, and `capabilities`.

The protocol applies to Intent, Representation, Plan, ComputeImage, Receipt, Execution Packet, Capability Surface, Provider, Execution, and Evidence.

An inspection is read-only. It cannot mutate identity, strengthen a claim, or create a replacement subject. Mutations must arrive through a recorded observation event with provenance.

The runtime kernel owns the inspection surface, and inspection is wired through the shared runtime context in `runtimeContext().kernel` during execution. “Observation Protocol” remains the compatibility name for existing documentation.
