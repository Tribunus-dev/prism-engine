import { runtimeContext } from '../runtime/runtime-context.js';

export const createEffectBudgetSystem = () => {
  const start = (context = runtimeContext()) => {
    const kernel = context?.kernel;
    const budget = Object.freeze({
      source: ['beam', 'origin', 'haze'],
      representation: ['dispersion', 'layer', 'haze'],
      plan: ['frontier', 'candidate', 'haze'],
      computeimage: ['seal', 'artifact', 'haze'],
      execution: ['pulse', 'packet', 'haze'],
      receipt: ['confirmation', 'receipt', 'still'],
      fabric: ['placement', 'route', 'haze'],
    });
    const apply = stage => {
      const [primary, foreground, ambient] = budget[stage] || budget.source;
      document.body.dataset.opticalPrimary = primary;
      document.body.dataset.opticalForeground = foreground;
      document.body.dataset.opticalAmbient = ambient;
      document.body.dataset.opticalBudget = '1:1:1';
      document.body.dataset.motionMode = matchMedia('(prefers-reduced-motion: reduce)').matches ? 'discrete' : 'continuous';
    };
    const sync = () => apply(document.body.dataset.canonicalStage || 'source');
    const observationHandler = () => sync();
    const scrollHandler = () => sync();
    if (kernel) {
      kernel.on('observation', observationHandler);
      kernel.on('scroll', scrollHandler);
    }
    const stageHandler = () => sync();
    addEventListener('prism:canonical-stage', stageHandler);
    sync();
    return {
      stop: () => {
        kernel?.off?.('observation', observationHandler);
        kernel?.off?.('scroll', scrollHandler);
        removeEventListener('prism:canonical-stage', stageHandler);
      },
    };
  };

  return { start };
};
