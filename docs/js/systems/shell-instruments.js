import { runtimeContext } from '../runtime/runtime-context.js';

export const createShellInstrumentsSystem = () => {
  const start = (context = runtimeContext()) => {
    const domRuntime = context?.domRuntime;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const owner = 'shell-instruments';
    const initPlanningInstruments = () => {
      const groups = [...document.querySelectorAll('.phase-flow,.target-grid,.validation-grid,.gate-line')]
        .map((group) => ({ group, nodes: [...group.children].filter((node) => node.matches('div,article')) }))
        .filter(({ nodes }) => nodes.length > 1);
      groups.forEach(({ group, nodes }) => {
        const readout = document.createElement('div');
        readout.className = 'plan-readout';
        readout.setAttribute('aria-live', 'polite');
        readout.innerHTML = '<span>ACTIVE PLAN</span><strong></strong>';
        group.after(readout);
        domRuntime?.claimNode?.(owner, readout);
        nodes.forEach((node, index) => {
          node.setAttribute('role', 'button');
          node.setAttribute('aria-pressed', 'false');
          node.tabIndex = 0;
          const activate = () => {
            nodes.forEach((item, itemIndex) => {
              const active = itemIndex === index;
              item.toggleAttribute('data-plan-active', active);
              item.setAttribute('aria-pressed', String(active));
            });
            const label = node.querySelector('h3,b')?.textContent || node.querySelector('span')?.textContent || `Stage ${index + 1}`;
            readout.querySelector('strong').textContent = label.trim();
            group.dataset.planIndex = String(index);
            document.body.dataset.prismPlan = group.className;
            document.body.dataset.prismPlanStep = String(index + 1).padStart(2, '0');
          };
          node.addEventListener('click', activate);
          node.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(); }
          });
        });
        if (nodes[0]) nodes[0].click();
      });

      if (reducedMotion.matches) return;
      let frame = 0;
      const update = () => {
        frame = 0;
        groups.forEach(({ group, nodes }) => {
          const rect = group.getBoundingClientRect();
          if (rect.bottom < innerHeight * 0.18 || rect.top > innerHeight * 0.82) return;
          const travel = Math.max(rect.height - innerHeight * 0.25, 1);
          const amount = Math.max(0, Math.min(1, (innerHeight * 0.64 - rect.top) / travel));
          const index = Math.min(nodes.length - 1, Math.floor(amount * nodes.length));
          if (!nodes[index].hasAttribute('data-plan-active')) nodes[index].click();
        });
      };
      const request = () => {
        if (!frame) frame = requestAnimationFrame(update);
      };
      const kernel = context?.kernel;
      kernel?.on('scroll', request);
      addEventListener('resize', request);
      request();
      return { request, kernel, update };
    };

    const initMythologySignatures = () => {
      const origin = document.querySelector('.prism-origin[data-origin-reveal]');
      if (origin && !reducedMotion.matches) {
        const update = () => {
          const rect = origin.getBoundingClientRect();
          const amount = Math.max(0, Math.min(1, (innerHeight * 0.72 - rect.top) / Math.max(rect.height * 0.72, 1)));
          origin.style.setProperty('--origin-progress', amount.toFixed(3));
          origin.dataset.originPhase = amount < 0.34 ? 'beam' : amount < 0.68 ? 'split' : 'spectrum';
        };
        context?.kernel?.on('scroll', update);
        addEventListener('resize', update);
        update();
      }
      const guide = document.querySelector('.guide-flow');
      if (!guide) return;
      const nodes = [...guide.children].filter((node) => node.matches('div'));
      if (nodes.length < 2) return;
      guide.dataset.readingFocus = 'true';
      const activate = (index) => {
        nodes.forEach((node, nodeIndex) => node.toggleAttribute('data-reading-active', nodeIndex === index));
        document.body.dataset.prismReadingStep = String(index + 1).padStart(2, '0');
      };
      const update = () => {
        const rect = guide.getBoundingClientRect();
        const amount = Math.max(0, Math.min(1, (innerHeight * 0.7 - rect.top) / Math.max(rect.height * 0.7, 1)));
        activate(Math.min(nodes.length - 1, Math.floor(amount * nodes.length)));
      };
      const kernel = context?.kernel;
      kernel?.on('scroll', update);
      addEventListener('resize', update);
      nodes.forEach((node, index) => {
        node.tabIndex = 0;
        node.addEventListener('mouseenter', () => activate(index));
        node.addEventListener('focus', () => activate(index));
      });
      activate(0);
      update();
      return { kernel };
    };

    initPlanningInstruments();
    initMythologySignatures();
    if (domRuntime) domRuntime.claim('shell-instruments', '.plan-readout');
    return { stop() {} };
  };

  return { start };
};
