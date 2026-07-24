
export const createCanonicalObjectSystem = () => {
  const start = (context) => {
    const domRuntime = context.domRuntime;
    const kernel = context.kernel;
    const repository = context.repository;
    const owner = 'canonical-object';
    const claims = repository?.claims || context?.client?.claims || [];
    const specimen = document.querySelector('[data-computeimage-life] .computeimage-specimen, [data-computeimage-renderer]');
    if (!specimen) return { stop() {} };
    const repositoryState = repository?.state;
    const computation = kernel?.ensureComputeImageSubject?.({
      claims,
      sourceRefs: claims.flatMap(claim => claim.sourceRefs || []),
      evidenceBoundary: repositoryState?.evidenceBoundary || 'repository evidence pending',
      provenance: {
        source: repositoryState?.evidenceBoundary ? 'repository-state' : 'runtime-fallback',
        boundary: repositoryState?.evidenceBoundary || 'repository evidence pending',
      },
    }) || kernel?.subject?.computeImage;
    if (!computation) return { stop() {} };
    if (kernel?.subject) kernel.subject.computeImage = computation;
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
    const modes = { source: 'silhouette', representation: 'semantic', plan: 'physical', computeimage: 'identity', execution: 'execution', receipt: 'evidence', fabric: 'fabric' };
    const phaseStages = { intent: 'source', representation: 'representation', plan: 'plan', computeimage: 'computeimage', execution: 'execution', receipt: 'receipt', fabric: 'fabric' };
    const update = event => {
      const stage = event.detail?.stage || phaseStages[event?.phase] || document.body.dataset.canonicalStage || 'source';
      mounted?.setMode?.(modes[stage] || 'silhouette');
      specimen.dataset.canonicalStage = stage;
      document.body.dataset.canonicalObjectId = computation.id;
      computation.claims = repository?.claims || computation.claims;
      computation.sourceRefs = computation.claims.flatMap(claim => claim.sourceRefs || []);
      computation.evidenceBoundary = repository?.state?.evidenceBoundary || computation.evidenceBoundary;
      const receipt = kernel?.state.receipts.at(-1);
      if (receipt) mounted?.attachReceipt?.(receipt.id);
      const observation = context.kernel?.state.currentObservation || 'observation pending';
      narrative.textContent = `Canonical ComputeImage ${stage}. Observation ${observation}. ${receipt ? `Evidence class ${receipt.claimClass}; receipt ${receipt.id}.` : 'No receipt is attached yet.'} Source references ${computation.sourceRefs.length}; evidence boundary: ${computation.evidenceBoundary}. Identity and intent remain preserved; remaining unknowns are disclosed in the observatory. `;
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
