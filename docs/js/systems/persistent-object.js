import { runtimeContext } from '../runtime/runtime-context.js';

export const createPersistentObjectSystem = () => {
  const start = (context = runtimeContext()) => {
    const kernel = context.kernel;
    const domRuntime = context.domRuntime;
    const stages = [
      ['intent', 'Intent', 'What computation entered the field?'],
      ['reveal', 'Reveal', 'Which structure is already present?'],
      ['search', 'Search', 'Which representations can satisfy the contract?'],
      ['decision', 'Decision', 'Which candidate was admitted?'],
      ['evidence', 'Evidence', 'What was actually observed?'],
      ['persistence', 'Persistence', 'What remains attached to the object?'],
    ];
    const objectViews = ['silhouette', 'cross-section', 'topology', 'ABI', 'execution', 'receipts', 'history', 'Fabric', 'capabilities', 'relationships'];
    const objectId = kernel?.subject?.id || 'computational-subject:prism-model';
    const key = 'prism-experience-object';
    const owner = 'persistent-object';

    const read = () => {
      try {
        return JSON.parse(sessionStorage.getItem(key) || '{}');
      } catch {
        return {};
      }
    };
    const saved = read();
    const current = Math.max(0, Math.min(stages.length - 1, Number(saved.stage) || 0));
    const state = {
      objectId,
      stage: current,
      scene: document.body.dataset.scene || 'origin',
      claim: document.body.dataset.sceneClaim || 'illustrative',
      knowledge: document.body.dataset.sceneKnowledge || 'illustrative-example',
      existence: document.body.dataset.sceneExistence || 'active',
      misconception: document.body.dataset.sceneMisconception || '',
      takeaway: document.body.dataset.sceneTakeaway || '',
    };

    const mount = document.querySelector('.component-header');
    if (!mount || document.querySelector('.computational-object')) return { stop() {} };

    const object = document.createElement('aside');
    object.className = 'computational-object';
    object.setAttribute('aria-label', 'Persistent computational object');
    object.innerHTML = `<button type="button" class="object-toggle" aria-expanded="false"><span class="object-sigil">◈</span><span><small>SEMANTIC CONTINUUM</small><strong>One computation</strong></span><em data-object-stage></em></button><div class="object-panel" hidden><div class="object-panel-heading"><span>ONE SUBJECT / MANY INSTRUMENTS</span><code>${objectId}</code></div><p data-object-intent></p><div class="object-cognition"><span><small>MISCONCEPTION</small><b data-object-misconception></b></span><span><small>TAKEAWAY</small><b data-object-takeaway></b></span></div><div class="object-view-label">COMPUTEIMAGE OBSERVATION</div><div class="object-views">${objectViews.map(view => `<button type="button" data-object-view="${view}">${view}</button>`).join('')}</div><div class="object-stages" role="list" aria-label="Intent to persistence"><span class="object-line" aria-hidden="true"></span>${stages.map(([id, label], index) => `<button type="button" role="listitem" data-object-step="${id}" aria-label="${label}" title="${label}"><i>${String(index + 1).padStart(2, '0')}</i><b>${label}</b></button>`).join('')}</div><div class="object-footer"><span data-object-claim></span><span data-object-existence></span><button type="button" data-object-advance>Advance →</button></div></div>`;
    mount.append(object);
    domRuntime?.claimNode?.(owner, object);
    domRuntime?.assertOwnership('persistent-object', object);

    const toggle = object.querySelector('.object-toggle');
    const panel = object.querySelector('.object-panel');
    const persist = () => {
      try {
        sessionStorage.setItem(key, JSON.stringify({ stage: state.stage, scene: state.scene, claim: state.claim }));
      } catch {}
    };

    const apply = (stage = state.stage) => {
      state.stage = Math.max(0, Math.min(stages.length - 1, stage));
      const [id, label, question] = stages[state.stage];
      document.body.dataset.objectStage = id;
      document.body.dataset.computationalSubject = objectId;
      object.querySelector('[data-object-stage]').textContent = `${String(state.stage + 1).padStart(2, '0')} / ${label}`;
      object.querySelector('[data-object-intent]').textContent = question;
      object.querySelector('[data-object-misconception]').textContent = state.misconception;
      object.querySelector('[data-object-takeaway]').textContent = state.takeaway;
      object.querySelector('[data-object-claim]').textContent = `${state.claim.replaceAll('-', ' ')} · ${id}`;
      object.querySelector('[data-object-existence]').textContent = `${state.existence} · ${state.knowledge.replaceAll('-', ' ')}`;
      object.querySelectorAll('[data-object-step]').forEach((step, index) => {
        step.toggleAttribute('aria-current', index === state.stage);
        step.toggleAttribute('data-complete', index < state.stage);
      });
      persist();
      kernel?.record({
        type: 'persistent-object-stage',
        visible: label,
        transformed: 'object persistence surfaced',
        hidden: 'ephemeral navigation state',
      });
    };

    const clickHandler = () => {
      const open = panel.hidden;
      panel.hidden = !open;
      toggle.setAttribute('aria-expanded', String(open));
    };
    const stepHandlers = [];
    const viewHandlers = [];
    const advanceHandler = () => apply(state.stage + 1);

    toggle.addEventListener('click', clickHandler);
    object.querySelectorAll('[data-object-step]').forEach((step, index) => {
      const handler = () => apply(index);
      stepHandlers.push([step, handler]);
      step.addEventListener('click', handler);
    });
    object.querySelectorAll('[data-object-view]').forEach((view) => {
      const handler = () => {
        document.body.dataset.objectView = view.dataset.objectView;
        object.querySelectorAll('[data-object-view]').forEach(item => item.toggleAttribute('aria-current', item === view));
      };
      viewHandlers.push([view, handler]);
      view.addEventListener('click', handler);
    });
    object.querySelector('[data-object-advance]').addEventListener('click', advanceHandler);

    apply(current);

    kernel?.record({
      type: 'persistent-object-initialized',
      visible: 'subject continuity',
      transformed: 'object state restored',
      hidden: 'one-shot DOM snapshots',
    });

    return {
      stop: () => {
        toggle.removeEventListener('click', clickHandler);
        stepHandlers.forEach(([button, handler]) => button.removeEventListener('click', handler));
        viewHandlers.forEach(([button, handler]) => button.removeEventListener('click', handler));
        object.querySelector('[data-object-advance]')?.removeEventListener('click', advanceHandler);
        object.remove();
      },
    };
  };

  return { start };
};
