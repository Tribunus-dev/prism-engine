
export const createComputeImageRenderer = (context) => {
  const modes = ['silhouette', 'identity', 'semantic', 'physical', 'execution', 'history', 'evidence', 'fabric'];
  const instances = new WeakMap();
  const resolveComputation = (computation) => {
    if (computation) return computation;
    return context?.kernel?.ensureComputeImageSubject?.() || context?.kernel?.subject?.computeImage || null;
  };
  const mount = (element, computation) => {
    if (!element) return null;
    const resolved = resolveComputation(computation);
    if (!resolved) return null;
    const kernel = context?.kernel;
    const instance = { element, computation: resolved };
    element.dataset.computeimageRenderer = 'shared';
    element.dataset.subjectId = resolved.id;
    const setMode = mode => {
      if (!modes.includes(mode)) return false;
      resolved.mode = mode;
      element.dataset.computeimageMode = mode;
      kernel?.record({
        type: 'computeimage-mode',
        observation: mode,
        visible: `ComputeImage ${mode} mode`,
        transformed: 'same geometry, new disclosure',
        hidden: 'undisclosed layers',
      });
      return true;
    };
    instance.setMode = setMode;
    instance.selectLayer = layer => {
      resolved.layer = layer;
      element.dataset.computeimageLayer = layer;
      return layer;
    };
    instance.attachReceipt = receiptId => {
      resolved.receiptId = receiptId;
      element.dataset.computeimageReceipt = receiptId;
      return receiptId;
    };
    instance.destroy = () => {
      delete element.dataset.computeimageRenderer;
      delete element.dataset.computeimageMode;
      instances.delete(element);
    };
    instances.set(element, instance);
    setMode('silhouette');
    return instance;
  };

  const get = element => instances.get(element);
  const renderer = Object.freeze({ mount, get, modes: Object.freeze([...modes]) });
  return {
    renderer,
    mountAll: () => {
      document.querySelectorAll('[data-computeimage-life], [data-computeimage-renderer]').forEach(element => {
        if (!instances.has(element)) mount(element);
      });
    },
  };
};

export const createComputeImageControls = () => {
  const start = (context) => {
    const kernel = context.kernel;
    const domRuntime = context.domRuntime;
    const owner = 'computeimage-controls';
    const root = document.querySelector('[data-computeimage-life]');
    if (!root) return { stop() {} };
    const views = {
      identity: ['Identity preserved', 'The subject remains the same while its representations accumulate.', 'identity · provenance · subject', 'SEALED / PLANNED'],
      topology: ['Topology exposed', 'Logical tensors, execution views, and residency become inspectable relationships.', 'views · relationships · residency', 'MAPPED / ILLUSTRATIVE'],
      execution: ['Execution observed', 'The artifact offers capabilities to a provider without becoming the provider.', 'capability · provider · packet', 'READY / BOUNDARY'],
      history: ['History attached', 'Evolution records what changed and what remained invariant across generations.', 'diff · provenance · evolution', 'EVOLVING / TRACKED'],
      receipts: ['Receipts attached', 'Evidence names the observation and its limits instead of claiming universal correctness.', 'receipt · constraints · scope', 'EVIDENCE / BOUNDED'],
    };
    const buttons = [...root.querySelectorAll('[data-life-view]')];
    const prediction = document.createElement('div');
    prediction.className = 'computeimage-prediction';
    prediction.innerHTML = '<span class="tiny-label">MAKE A PREDICTION</span><strong>When the instrument changes, what remains?</strong><div><button type="button" data-prediction="changes">The identity changes</button><button type="button" data-prediction="persists">The identity persists</button></div><p hidden data-prediction-result></p>';
    root.querySelector('.section-heading').after(prediction);
    domRuntime?.claimNode?.(owner, prediction);
    domRuntime?.claim('computeimage-controls', '.computeimage-prediction');
    domRuntime?.assertOwnership('computeimage-controls', prediction);
    prediction.querySelectorAll('[data-prediction]').forEach(button =>
      button.addEventListener('click', () => {
        const correct = button.dataset.prediction === 'persists';
        const result = prediction.querySelector('[data-prediction-result]');
        result.hidden = false;
        result.textContent = correct
          ? 'Correct. The observation changes; the Semantic Continuum remains invariant.'
          : 'The observation changes, but the Semantic Continuum keeps its identity. Inspect the provenance.';
        prediction.dataset.predictionState = correct ? 'resolved' : 'corrected';
        kernel?.record({
          type: 'prediction',
          visible: result.textContent,
          transformed: 'mental model tested',
          hidden: 'future observations',
          evidenceIncreased: 'identity invariant observed',
        });
      }),
    );

    const apply = key => {
      const data = views[key];
      if (!data) return;
      buttons.forEach(button => button.classList.toggle('is-active', button.dataset.lifeView === key));
      root.querySelector('[data-life-title]').textContent = data[0];
      root.querySelector('[data-life-copy]').textContent = data[1];
      root.querySelector('[data-life-proof]').textContent = data[2];
      root.querySelector('[data-life-status]').textContent = data[3];
      document.body.dataset.computeimageObservation = key;
      const transformation = {
        id: `computeimage-observation-${key}`,
        from: 'ComputeImage',
        to: key,
        preconditions: ['subject identity exists'],
        invariants: ['identity preserved', 'provenance remains attached'],
        postconditions: [data[0]],
        evidenceGained: key === 'receipts' ? ['bounded receipt surface'] : [],
        evidenceLost: [],
        identityPreserved: true,
        intentPreserved: true,
        deterministic: true,
      };
      try {
        context?.client?.transform(transformation);
      } catch (error) {
        context?.client?.observe({ type: 'transformation-rejected', visible: error.message, transformed: 'none', hidden: key });
      }
    };
    buttons.forEach(button => button.addEventListener('click', () => apply(button.dataset.lifeView)));
    apply('identity');
    return { stop() {} };
  };

  return { start };
};
