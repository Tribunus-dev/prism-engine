
export const createStateProjectionSystem = () => {
  const start = (context) => {
    const kernel = context?.kernel;
    if (!kernel) return;
    const project = () => {
      const canonicalSubject = context?.runtime?.getCanonicalSubject?.() || context?.runtime?.stateSubject;
      if (!canonicalSubject) return;
      document.body.dataset.computationalSubject = canonicalSubject.id;
      document.body.dataset.observerMode = kernel.state.observerMode;
      document.body.dataset.opticalState = kernel.state.opticalState;
      document.body.dataset.continuity = 'explicit';
    };
    const listeners = [['observer-mode', project], ['optical-state', project], ['continuity', project]];
    listeners.forEach(([event, handler]) => kernel.on(event, handler));
    project();
    return {
      stop: () => listeners.forEach(([event, handler]) => kernel.off?.(event, handler))
    };
  };
  return { start };
};
