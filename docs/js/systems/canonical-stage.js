
export const createCanonicalStageSystem = () => {
  const start = (context) => {
    const kernel = context?.kernel;
    const stages = [...document.querySelectorAll('[data-canonical-stage]')];
    if (!stages.length || !kernel) return { stop() {} };
    let active = '';
    const update = () => {
      const focus = innerHeight * 0.42;
      const current = stages.reduce(
        (best, section) =>
          Math.abs(section.getBoundingClientRect().top - focus) < Math.abs(best.getBoundingClientRect().top - focus)
            ? section
            : best,
        stages[0],
      );
      const stage = current.dataset.canonicalStage;
      if (stage === active) return;
      active = stage;
      document.body.dataset.canonicalStage = stage;
      stages.forEach(section => section.toggleAttribute('data-canonical-active', section === current));
      dispatchEvent(new CustomEvent('prism:canonical-stage', { detail: { stage } }));
      kernel?.record({ type: 'canonical-stage', observation: stage, visible: `canonical journey / ${stage}`, transformed: 'same computation disclosed at a new boundary', hidden: 'later stages remain undisclosed' });
    };
    const updateListener = () => update();
    kernel.on('scroll', updateListener);
    const resizeListener = () => update();
    addEventListener('resize', resizeListener);
    update();
    return {
      stop: () => {
        kernel.off?.('scroll', updateListener);
        removeEventListener('resize', resizeListener);
      },
    };
  };

  return { start };
};
