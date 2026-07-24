import { runtimeContext } from '../runtime/runtime-context.js';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const makeMotionPreferenceHandler = kernel => () => applyMotionPreference(kernel);

const applyMotionPreference = (kernel) => {
  document.documentElement.dataset.prismReducedMotion = String(reducedMotion.matches);
  document.body.dataset.prismMotion = reducedMotion.matches ? 'discrete' : 'animated';
  kernel?.record({
    type: 'motion-preference',
    visible: reducedMotion.matches ? 'discrete observation states' : 'animated observation states',
    transformed: 'renderer preference applied',
    hidden: 'decorative motion only'
  });
};

const labelInteractiveSurfaces = () => {
  document.querySelectorAll('[role="button"]:not(button):not(a)').forEach(element => {
    if (!element.hasAttribute('tabindex')) element.tabIndex = 0;
    if (!element.hasAttribute('aria-label') && !element.hasAttribute('aria-labelledby') && !element.textContent.trim()) element.setAttribute('aria-label', 'Interactive Prism observation');
    if (element.dataset.prismKeyboardReady) return;
    element.dataset.prismKeyboardReady = 'true';
    element.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      element.click();
    });
  });
};

const announceCanonicalState = observation => {
  const narrative = document.querySelector('#canonical-object-narrative');
  if (!narrative || !observation) return;
  narrative.textContent = `Current observation: ${observation.instrument || observation.phase || 'Prism computation'}. Evidence state: ${observation.evidenceState || 'bounded'}. Identity remains preserved.`;
};

export const createAccessibilitySystem = () => {
  const start = (context = runtimeContext()) => {
    const kernel = context?.kernel;
    if (!kernel) return { stop() {} };
    const motionHandler = makeMotionPreferenceHandler(kernel);
    applyMotionPreference(kernel);
    if (reducedMotion.addEventListener) {
      reducedMotion.addEventListener('change', motionHandler);
    } else if (reducedMotion.addListener) {
      reducedMotion.addListener(motionHandler);
    }
    labelInteractiveSurfaces();
    const handlers = [
      ['observation', announceCanonicalState],
      ['repository-ready', labelInteractiveSurfaces],
    ];
    handlers.forEach(([event, handler]) => kernel.on(event, handler));
    return {
      stop: () => {
        if (reducedMotion.removeEventListener) {
          reducedMotion.removeEventListener('change', motionHandler);
        } else if (reducedMotion.removeListener) {
          reducedMotion.removeListener(motionHandler);
        }
        handlers.forEach(([event, handler]) => kernel.off?.(event, handler));
      },
    };
  };
  return { start };
};
