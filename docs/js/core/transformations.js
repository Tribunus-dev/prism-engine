import { CLAIM_CLASSES, OBSERVATION_KINDS } from './ontology.js';

const record = (id, from, to, knowledgeGained, sourceRefs = []) => Object.freeze({ id, from, to, preconditions: ['identity exists', 'input is inspectable'], invariants: ['identity preserved', 'intent preserved'], postconditions: [`${to} is inspectable`], knowledgeGained, evidenceGained: ['bounded architectural observation'], evidenceLost: [], capabilitiesChanged: [], relationshipsChanged: ['subject history extended'], deterministic: true, identityPreserved: true, intentPreserved: true, sourceRefs, claimClass: CLAIM_CLASSES.ARCHITECTURAL });
export const TRANSFORMATIONS = Object.freeze({
  sourceToSemantic: record('source-artifact-to-semantic-representation', 'SourceArtifact', 'SemanticRepresentation', ['semantic structure revealed'], ['docs/prism-semantics.md']),
  semanticToPlan: record('semantic-representation-to-candidate-plan', 'SemanticRepresentation', 'CandidatePlan', ['candidate space made explicit'], ['docs/prism-runtime.md']),
  planToImage: record('candidate-plan-to-computeimage', 'CandidatePlan', 'ComputeImage', ['deployment contract sealed'], ['docs/cimage-layout-abi-v1.md']),
  imageToExecution: record('computeimage-to-execution', 'ComputeImage', 'Execution', ['provider boundary observed'], ['docs/prism-runtime.md']),
  executionToReceipt: record('execution-to-receipt', 'Execution', 'Receipt', ['execution evidence bounded'], ['docs/prism-observation-protocol.md']),
  imageToFabric: record('computeimage-to-fabric-placement', 'ComputeImage', 'FabricPlacement', ['placement domain revealed'], ['docs/prism-runtime.md'])
});
export const validateTransformations = registry => Object.values(registry).flatMap(item => [
  !item.identityPreserved && `${item.id} must account for identity`,
  !item.intentPreserved && `${item.id} must account for intent`,
  !item.deterministic && `${item.id} must declare determinism`,
  !item.from || !item.to ? `${item.id} must declare endpoints` : ''
]).filter(Boolean);
