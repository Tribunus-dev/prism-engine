
export const createObservatoryShellSystem = () => {
  const start = (context) => {
    const kernel = context.kernel;
    const domRuntime = context.domRuntime;
    const owner = 'observatory-shell';
    const labels = { identity: 'Identity', structure: 'Structure', transformation: 'Transformation', embodiment: 'Embodiment', execution: 'Execution', evidence: 'Evidence', scale: 'Scale' };
    const layers = ['intuition', 'architecture', 'implementation', 'evidence', 'repository'];
    const scene = {
      observation: document.body.dataset.sceneObservation,
      question: document.body.dataset.sceneQuestion,
      cause: document.body.dataset.sceneCause,
      effect: document.body.dataset.sceneEffect,
      knowledge: document.body.dataset.sceneKnowledge,
      existence: document.body.dataset.sceneExistence,
    };
    const header = document.querySelector('.component-header');
    if (!header || document.querySelector('.observatory-shell')) return { stop() {} };
    const observatoryPortalRoot = document.querySelector('#prism-portal-root') || (() => {
      const root = document.createElement('div');
      root.id = 'prism-portal-root';
      root.setAttribute('aria-hidden', 'true');
      const fallbackHost = document.querySelector('[data-observatory-shell], .observatory-shell, .component-header, header.nav, main') || document.body;
      fallbackHost.append(root);
      return root;
    })();
    domRuntime?.claim('observatory-shell-root', '#prism-portal-root');
    domRuntime?.assertOwnership('observatory-shell-root', observatoryPortalRoot);

    document.body.dataset.observatory = 'true';
    document.body.dataset.opticalState = 'observation';
    const shell = document.createElement('section');
    shell.className = 'observatory-shell';
    shell.setAttribute('aria-label', 'Current observation');
    shell.innerHTML = `<div class="observatory-context">
      <strong>Observation</strong>
      <p data-observatory-question>Tracking the current observation.</p>
    </div>
    <div class="observatory-causality">
      <span>intent</span><span>phase</span><span><b>knowledge source</b></span>
    </div>
    <div class="observatory-state" data-observatory-state>active</div>
    <select data-knowledge-layer class="observatory-knowledge-layer">
      ${layers.map((layer) => `<option value="${layer}">${layer}</option>`).join('')}
    </select>
    <label class="observatory-mode-label">Mode
      <select data-observer-mode>
        ${kernel?.modes?.map((mode) => `<option value="${mode}">${mode}</option>`).join('')}
      </select>
    </label>`;
    header.after(shell);
    domRuntime?.claim('observatory-shell', '.observatory-shell');
    domRuntime?.assertOwnership('observatory-shell', shell);
    const renderObservation = observation => {
      if (!observation) return;
      shell.querySelector('.observatory-context strong').textContent = observation.instrument || labels[scene.observation] || 'Observation';
      shell.querySelector('[data-observatory-question]').textContent = observation.question || `${observation.phase || 'observation'} is now in view.`;
      shell.querySelector('.observatory-causality span:nth-of-type(3) b').textContent = observation.knowledgeState || scene.knowledge || 'knowledge source';
      shell.querySelector('.observatory-state').textContent = observation.existence || scene.existence || 'active';
      shell.dataset.observationId = observation.id || '';
    };
    kernel?.on('observation', renderObservation);
    renderObservation(kernel?.state.observations.at(-1));

    const questions = document.createElement('details');
    questions.className = 'observatory-questions';
    questions.innerHTML = `<summary>Questions now available</summary><div>${(kernel?.questions() || [scene.next || 'What should be observed next?']).map(question => `<button type="button" data-question="${question}">→ ${question}</button>`).join('')}</div>`;
    shell.append(questions);
    domRuntime?.claimNode?.(owner, questions);
    domRuntime?.assertOwnership?.(owner, questions);
    domRuntime?.claim('observatory-shell', '.observatory-questions');
    domRuntime?.assertOwnership('observatory-shell', questions);
    questions.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => kernel?.record({ type: 'question-selected', visible: button.dataset.question, transformed: 'curiosity directed', hidden: 'unasked questions' })));

    const select = shell.querySelector('[data-knowledge-layer]');
    const setLayer = layer => {
      const index = layers.indexOf(layer);
      if (index < 0) return;
      document.body.dataset.knowledgeLayer = layer;
      document.body.style.setProperty('--knowledge-depth', index);
      document.querySelectorAll('[data-knowledge-level]').forEach(node => node.toggleAttribute('hidden', Number(node.dataset.knowledgeLevel) > index));
    };
    if (select) {
      select.value = layers.includes(document.body.dataset.knowledgeLayer) ? document.body.dataset.knowledgeLayer : 'intuition';
      select.addEventListener('change', () => setLayer(select.value));
      setLayer(select.value);
    }
    const mode = shell.querySelector('[data-observer-mode]');
    if (mode) {
      if (!mode.value) mode.value = kernel?.state?.observerMode || kernel?.modes?.[0] || 'observer';
      mode.value = kernel?.state?.observerMode || mode.value;
      mode.addEventListener('change', () => {
        kernel?.setMode(mode.value);
        kernel?.record({ type: 'observer-mode-changed', transformed: mode.value, visible: 'new perspective', hidden: 'unchanged subject' });
      });
    }

    const receipt = document.createElement('div');
    receipt.className = 'observatory-receipt';
    receipt.setAttribute('aria-live', 'polite');
    receipt.innerHTML = '<span>OBSERVATION RECEIPT</span><strong>Instrument ready</strong><small>subject identity preserved</small>';
    const selfDescription = document.createElement('aside');
    selfDescription.className = 'physics-inspector';
    selfDescription.innerHTML = '<button type="button" data-physics-toggle aria-expanded="false">Physics active</button><div hidden><span>IDENTITY PRESERVED</span><span>OBSERVATION BOUNDARY CROSSED</span><span>EVIDENCE BOUNDED</span><span>OPTICAL RULE SATISFIED</span><button type="button" data-reset-continuity>Reset continuity</button></div>';

    observatoryPortalRoot.append(receipt, selfDescription);
    domRuntime?.claim('observatory-shell', '.observatory-receipt, .physics-inspector');
    domRuntime?.assertOwnership('observatory-shell', receipt);
    domRuntime?.assertOwnership('observatory-shell', selfDescription);

    const physicsToggle = selfDescription.querySelector('[data-physics-toggle]');
    const physicsPanel = selfDescription.querySelector('div');
    physicsToggle.addEventListener('click', () => {
      const open = physicsPanel.hidden;
      physicsPanel.hidden = !open;
      physicsToggle.setAttribute('aria-expanded', String(open));
      kernel?.record({ type: 'physics-inspection', visible: 'active conservation laws', evidenceIncreased: 'none' });
    });
    selfDescription.querySelector('[data-reset-continuity]').addEventListener('click', () => {
      kernel?.resetContinuity();
      receipt.querySelector('small').textContent = 'continuity reset by visitor';
    });

    const record = (action, target) => {
      const label = target?.textContent?.trim().replace(/\s+/g, ' ').slice(0, 72) || 'instrument surface';
      receipt.querySelector('strong').textContent = `${action} observed`;
      receipt.querySelector('small').textContent = `${label} · ${scene.knowledge || 'illustrative-example'}`;
      kernel?.setOpticalState(action === 'hover' ? 'focus' : action === 'interaction' ? 'exploration' : 'observation');
      kernel?.record({ type: action, visible: label, transformed: 'focus changed', hidden: 'unselected surfaces', evidenceIncreased: scene.knowledge || 'none' });
      kernel?.receipt({ observation: label, knowledgeSource: scene.knowledge || 'illustrative-example', outcome: action });
    };
    const documentClickHandler = event => {
      const target = event.target.closest('button, a, summary, [role="button"]');
      if (target) record('interaction', target);
    };
    const documentPointerHandler = event => {
      const target = event.target.closest('[data-living-node], [data-atlas-node], [data-surface], [data-object-step]');
      if (target) record('hover', target);
    };
    document.addEventListener('click', documentClickHandler);
    document.addEventListener('pointerover', documentPointerHandler, { passive: true });

    domRuntime?.claim('observatory-shell', '.observatory-shell, .physics-inspector, .observatory-receipt');
    return {
      stop: () => {
        kernel?.off?.('observation', renderObservation);
        document.removeEventListener('click', documentClickHandler);
        document.removeEventListener('pointerover', documentPointerHandler);
        shell.remove();
        receipt.remove();
        selfDescription.remove();
      },
    };
  };

  return { start };
};
