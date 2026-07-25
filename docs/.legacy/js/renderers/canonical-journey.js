
import { CANONICAL_JOURNEY_STAGES } from '../core/canonical-contract.js';

export const createCanonicalJourneyRenderer = () => {
  const start = (context) => {
    const kernel = context?.kernel;
    const journey = document.querySelector('[data-canonical-journey]');
    if (!journey) return { stop() {} };
    const stages = [...journey.querySelectorAll('[data-canonical-stage]')];
    if (!stages.length) return { stop() {} };

    let active = -1;
    const update = () => {
      const focus = innerHeight * 0.62;
      let best = 0;
      let distance = Infinity;
      stages.forEach((stage, index) => {
        const next = Math.abs(stage.getBoundingClientRect().top - focus);
        if (next < distance) {
          best = index;
          distance = next;
        }
      });
      if (best === active) return;
      active = best;
      stages.forEach((stage, index) => stage.toggleAttribute('data-canonical-active', index === active));
      const stage = stages[active].dataset.canonicalStage;
      document.body.dataset.canonicalJourneyStage = stage;
      const data = CANONICAL_JOURNEY_STAGES[stage];
      if (data) {
        try {
          context?.client?.transform({
            from: data[0],
            to: data[1],
            preconditions: data[2],
            invariants: ['identity preserved', 'intent preserved'],
            postconditions: [`${data[1]} observed`],
            evidenceGained: stage === 'receipt' ? ['bounded receipt'] : [],
            evidenceLost: [],
            capabilitiesChanged: [],
            relationshipsChanged: [stage],
            deterministic: true,
            identityPreserved: true,
            intentPreserved: true,
          });
        } catch {}
      }
    };
    const scrollHandler = () => update();
    kernel?.on('scroll', scrollHandler);
    const resizeHandler = () => update();
    addEventListener('resize', resizeHandler);
    update();
    return {
      stop: () => {
        kernel?.off?.('scroll', scrollHandler);
        removeEventListener('resize', resizeHandler);
      }
    };
  };

  return { start };
};
