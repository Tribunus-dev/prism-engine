# Prism Experience Architecture

Prism is an instrument for observing, preserving, and executing computation as one continuous semantic object.

The website is governed by one educational standard: every interaction must leave the visitor with a more accurate mental model of computation than they had before entering it.

This document is the constitutional layer for the website. It is not a component catalog. A future interaction, scene, or visual treatment should be rejected when it contradicts these principles.

## First principles

Computations possess semantic intent independent of any particular execution target.

Intent should be transformed, not reconstructed.

Observation must be distinguishable from interpretation.

Execution is the realization of preserved intent.

Evidence is part of execution, not an afterthought.

Everything else in the experience is derived from these statements.

## Immutable principles

### Computational intent

Intent is never discarded. Every transformation preserves provenance. The interface must never imply that intent was reconstructed when it was only transformed.

### Progressive revelation

Information is revealed, never dumped. Every interaction answers one question while creating exactly one new question.

### Evidence before assertion

Nothing is claimed without an explicit evidence class. Illustration and measurement must never share visual language.

### Persistent identity

A computational object never ceases to exist. It changes representation, not identity.

### One world

The visitor never feels that they left one page and entered another. Navigation is movement through one instrument.

## Five orthogonal systems

The narrative system controls scenes, questions, discovery, pacing, and acts. The semantic system controls objects, relationships, knowledge, receipts, and identity. The cognitive system controls misconceptions, visitor takeaways, attention, density, and the next question. The visual system controls optics, typography, motion, geometry, spacing, and atmosphere. The technical system controls rendering, accessibility, build, performance, and progressive enhancement.

These systems communicate through explicit scene state. None of them should hide policy inside an unrelated component.

## Recursive interaction grammar

Every scene follows the same sequence:

`intent → reveal → explore → commit → observe → persist`

The compiler, scheduler, architecture graph, and website navigation all use this grammar. Intent identifies the question, reveal exposes structure, explore presents alternatives, commit records a decision, observe distinguishes what happened from what was inferred, and persist carries the object forward. The visitor should recognize that the website behaves like the system it describes.

## Progressive disclosure

Discover provides emotional understanding with almost no text. Understand provides the normal narrative and architectural model. Inspect exposes technical diagrams, interactions, and contracts. Reference exposes repository links, evidence classes, and implementation boundaries.

Changing density changes emphasis and detail, never the underlying claim.

## Mythology and engineering

Every scene separates metaphor, architecture, implementation, and evidence. A beam entering a prism is metaphor. Semantic domains separating is architecture. GGUF parsing and ECS systems are implementation. A receipt or repository reference is evidence. The metaphor may invite curiosity; it must never become a capability claim.

## Medium: the Prism Observatory

Prism is an observatory, not a collection of pages. Each route is an instrument for observing the same computation from a different angle: Origin observes intent, Representation observes semantic structure, Compiler observes transformation, ComputeImage observes embodiment, Scheduler observes realization, Evidence observes truth, and Fabric observes scale. Orientation communicates the active instrument and the object under observation; it does not merely provide movement between documents.

## Canonical ontology

These nouns are stable across the website and implementation:

| Noun | Meaning |
| --- | --- |
| Intent | What the user or model means. |
| Representation | The semantic expression of intent. |
| Plan | The legal transformations under exploration. |
| ComputeImage | The sealed executable embodiment. |
| Execution | The realization on hardware. |
| Receipt | Observable evidence about that realization. |

The persistent subject that survives all six phases is named the **Semantic Continuum**. Its identity is stable even as it moves from Intent to Receipt.

## Persistent computational subject

The website follows one semantic object through multiple representations:

`model artifact → semantic representation → candidate plan → ComputeImage → execution packet → receipt → Fabric artifact`

The object id and provenance chain remain stable while the visible representation changes. The website calls this subject the Semantic Continuum so the protagonist has one architectural name rather than a rotating set of synonyms.

## State ontology

Every canonical object has an explicit existence state: `possible`, `planned`, `hypothetical`, `active`, `archived`, `unknown`, `partial`, `deferred`, or `rejected`. Existence state is not a claim of quality or execution; it describes where the object is in its lifecycle.

## Knowledge sources

Every important statement belongs to a knowledge source: measured observation, compile verification, repository evidence, architectural derivation, illustrative example, research direction, or speculation. The source determines wording, evidence, and visual treatment.

## Failure and uncertainty

The experience must represent uncertainty honestly. `unknown`, `partial`, `deferred`, `rejected`, and `experimental` are first-class outcomes. A failure state must explain what was learned, what remains unproven, and what the next valid action is. Nothing should visually imply that every path succeeds.

## Canonical objects

The shared vocabulary is constitutional: Persistent Computational Subject, ComputeImage, Receipt, Execution Packet, Capability Surface, and Provider. ComputeImage is a canonical object, not merely a file format. Every scene should refer to these nouns consistently and preserve their identity as their representations change.

## Visitor outcomes

Every scene declares the misconception a visitor should leave behind and the more accurate model they should carry forward:

| Scene | Misconception | Visitor takeaway |
| --- | --- | --- |
| Origin | A model is just weights. | A model already contains semantic intent. |
| Representation | Compilation is only lowering. | Compilation is semantic transformation. |
| ComputeImage | A ComputeImage is another model format. | It is an executable semantic artifact. |
| Scheduler | Scheduling is only hardware selection. | Scheduling preserves intent across capability boundaries. |
| Evidence | Benchmarks prove correctness by themselves. | Receipts expose the scope of claims. |

## Review standard

New work must pass `docs/design-review-checklist.md` before it becomes part of the shared experience system.
