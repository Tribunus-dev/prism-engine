import { runtimeContext } from '../runtime/runtime-context.js';

export const createScrollObserverSystem = () => {
  const start = (context = runtimeContext()) => {
  const kernel = context?.kernel;
  document.body.dataset.prismScrollOwner = 'kernel';
  let lastY = window.scrollY;
  let lastDirection = 'stationary';
  let lastBucket = -1;
  let frame = 0;
  let scrollingTimer;

  const observe = () => {
    frame = 0;
    const max = Math.max(document.documentElement.scrollHeight - innerHeight, 0);
    const y = window.scrollY;
    const direction = y === lastY ? 'stationary' : y > lastY ? 'forward' : 'backward';
    const progress = max ? y / max : 0;
    lastY = y;
    document.body.dataset.prismScrollDirection = direction;
    document.body.style.setProperty('--prism-scroll-progress', `${(progress * 100).toFixed(2)}%`);
    const bucket = Math.floor(progress * 20);
    if (direction !== lastDirection || bucket !== lastBucket) {
      kernel?.record({
        type: 'scroll-observed',
        visible: direction === 'stationary' ? 'reading position' : `reading movement / ${direction}`,
        transformed: 'observation position changed',
        hidden: 'unselected surfaces',
        progress
      });
      lastDirection = direction;
      lastBucket = bucket;
    }
    kernel?.emit('scroll', { direction, progress, y });
  };

  const request = () => {
    document.body.dataset.prismScrolling = 'true';
    clearTimeout(scrollingTimer);
    scrollingTimer = setTimeout(() => delete document.body.dataset.prismScrolling, 260);
    if (!frame) frame = requestAnimationFrame(observe);
  };

  addEventListener('scroll', request, { passive: true });
  addEventListener('resize', request, { passive: true });
  observe();
  return { stop() { removeEventListener('scroll', request); removeEventListener('resize', request); } };
  };
  return { start };
};
