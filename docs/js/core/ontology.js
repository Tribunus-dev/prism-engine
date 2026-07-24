export const CLAIM_CLASSES = Object.freeze({ ILLUSTRATIVE: 'illustrative', ARCHITECTURAL: 'architectural', REPOSITORY: 'repository', COMPILE: 'compile-verified', MEASURED: 'measured' });
export const KNOWLEDGE_STATES = Object.freeze({ UNKNOWN: 'unknown', HYPOTHESIZED: 'hypothesized', DERIVED: 'derived', OBSERVED: 'observed', VERIFIED: 'verified', MEASURED: 'measured' });
export const EXISTENCE_STATES = Object.freeze({ POSSIBLE: 'possible', ACTIVE: 'active', SEALED: 'sealed', EXECUTING: 'executing', COMPLETE: 'complete' });
export const OBSERVATION_KINDS = Object.freeze({ ORIGIN: 'origin', REPRESENTATION: 'representation', COMPILER: 'compiler', EXECUTION: 'execution', EVIDENCE: 'evidence', FABRIC: 'fabric', PARTICIPATION: 'participation' });
export const validateClaim = claim => {
  if (!claim || !claim.id) return 'claim is missing an id';
  if (!Object.values(CLAIM_CLASSES).includes(claim.class)) return `${claim.id} has an invalid claim class`;
  if (claim.class === CLAIM_CLASSES.MEASURED && (!claim.sourceRefs?.length || !claim.constraints)) return `${claim.id} measured claims require sourceRefs and constraints`;
  return '';
};
