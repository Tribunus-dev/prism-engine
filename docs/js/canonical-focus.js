import { runtimeContext } from './runtime/runtime-context.js';

export const createCanonicalFocusSystem = () => {
  const start = (context = runtimeContext()) => {
    const kernel = context?.kernel;
    const domRuntime = context?.domRuntime;
    const owner = 'canonical-focus';
    const journey = document.querySelector('.tensor-journey');
    if (!journey) return { stop() {} };
    const toolbar = journey.querySelector('.journey-toolbar');
    if (!toolbar) return { stop() {} };
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'canonical-focus-toggle';
    button.setAttribute('aria-pressed', 'false');
    button.textContent = 'Focus on one computation';
    toolbar.append(button);
    domRuntime?.claimNode?.(owner, button);
    domRuntime?.claim('canonical-focus', '.canonical-focus-toggle');
    domRuntime?.assertOwnership('canonical-focus', button);
    const setFocus = enabled => {
      document.body.classList.toggle('canonical-focus', enabled);
      button.setAttribute('aria-pressed', String(enabled));
      button.textContent = enabled ? 'Exit focused observation' : 'Focus on one computation';
      kernel?.record({
        type: enabled ? 'focus-entered' : 'focus-exited',
        phase: 'reflection',
        visible: enabled ? 'one ComputeImage journey' : 'full Observatory',
        transformed: 'attention scope changed',
        hidden: enabled ? 'supporting instruments' : 'none',
      });
      kernel?.remember({ focus: enabled });
    };
    const toggleHandler = () => setFocus(!document.body.classList.contains('canonical-focus'));
    button.addEventListener('click', toggleHandler);
    const escape = event => {
      if (event.key === 'Escape' && document.body.classList.contains('canonical-focus')) setFocus(false);
    };
    document.addEventListener('keydown', escape);
    return {
      stop: () => {
        button.removeEventListener('click', toggleHandler);
        document.removeEventListener('keydown', escape);
      },
    };
  };

  return { start };
};
