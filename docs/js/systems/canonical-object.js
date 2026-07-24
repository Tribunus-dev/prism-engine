
import { CANONICAL_OBJECT_STAGES, CANONICAL_PHASE_STAGES } from '../core/canonical-contract.js';

export const createCanonicalObjectSystem = () => {
  const start = (context) => {
    const domRuntime = context.domRuntime;
    const kernel = context.kernel;
    const owner = 'canonical-object';
    const getSubject = () => context?.runtime?.getCanonicalSubject?.() || null;
    const specimen = document.querySelector('[data-computeimage-life] .computeimage-specimen, [data-computeimage-renderer]');
    if (!specimen) return { stop() {} };
    const computation = getSubject();
    if (!computation) return { stop() {} };
    specimen.dataset.subjectId = computation.id;
    specimen.dataset.canonicalObject = 'true';
    specimen.tabIndex = 0;
    specimen.setAttribute('aria-label', 'Canonical ComputeImage object. Focus to hear its current observation.');
    const narrative = document.createElement('p');
    narrative.className = 'visually-hidden';
    narrative.id = 'canonical-object-narrative';
    narrative.setAttribute('aria-live', 'polite');
    specimen.after(narrative);
    domRuntime?.claimNode?.(owner, narrative);
    specimen.setAttribute('aria-describedby', narrative.id);
    const renderer = context.computeImageRenderer;
    const mounted = renderer?.mount?.(specimen, computation);
    const modes = CANONICAL_OBJECT_STAGES;
    const phaseStages = CANONICAL_PHASE_STAGES;
    const update = event => {
      const stage = event.detail?.stage || phaseStages[event?.phase] || document.body.dataset.canonicalStage || 'source';
      const currentComputation = getSubject();
      if (!currentComputation) return;
      mounted?.setMode?.(modes[stage] || 'silhouette');
      specimen.dataset.canonicalStage = stage;
      specimen.dataset.subjectId = currentComputation?.id || '';
      document.body.dataset.canonicalObjectId = currentComputation?.id || '';
      const claims = currentComputation?.claims || [];
      const sourceRefs = Array.isArray(claims) ? claims.flatMap(claim => claim.sourceRefs || []) : [];
      const evidenceBoundary = currentComputation?.evidenceBoundary || kernel?.state?.repositoryState?.evidenceBoundary || '';
      const receipt = kernel?.state.receipts.at(-1);
      if (receipt) mounted?.attachReceipt?.(receipt.id);
      const observation = context.kernel?.state.currentObservation || 'observation pending';
      narrative.textContent = `Canonical ComputeImage ${stage}. Observation ${observation}. ${receipt ? `Evidence class ${receipt.claimClass}; receipt ${receipt.id}.` : 'No receipt is attached yet.'} Source references ${sourceRefs.length}; evidence boundary: ${evidenceBoundary || 'repository-state'}. Identity and intent remain preserved; remaining unknowns are disclosed in the observatory. `;
    };
    addEventListener('prism:canonical-stage', update);
    kernel?.on('observation', update);
    kernel?.on('repository-ready', update);
    update({ detail: { stage: document.body.dataset.canonicalStage || 'source' } });
    return {
      stop: () => {
        removeEventListener('prism:canonical-stage', update);
        kernel?.off?.('observation', update);
        kernel?.off?.('repository-ready', update);
      },
    };
  };

  return { start };
};
