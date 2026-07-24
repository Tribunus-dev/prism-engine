
export const createVisitorStateSystem = () => {
  const start = (context) => {
    const intents = {
      explore: { label: 'Explore', emphasis: 'ComputeImage first' },
      understand: { label: 'Understand', emphasis: 'Compiler and representation' },
      inspect: { label: 'Inspect', emphasis: 'Implementation and relationships' },
      validate: { label: 'Validate', emphasis: 'Capabilities and evidence' },
      contribute: { label: 'Contribute', emphasis: 'Research and repository' },
    };
    const densities = ['discover', 'understand', 'inspect', 'reference'];
    const state = { intent: 'explore', density: 'discover' };
    const domRuntime = context?.domRuntime;
    const shell = document.querySelector('.component-header');
    if (!shell) return { stop() {} };
    const owner = 'visitor-state';

    const controls = document.createElement('div');
    controls.className = 'visitor-controls';
    controls.innerHTML = `<button type="button" class="visitor-toggle" aria-expanded="false">Journey</button><div class="visitor-panel" hidden><span class="visitor-label">INTENT</span><div class="visitor-profiles">${Object.entries(intents).map(([key, value]) => `<button type="button" data-visitor-intent="${key}">${value.label}</button>`).join('')}</div><span class="visitor-label">DENSITY</span><div class="visitor-profiles">${densities.map(key => `<button type="button" data-density="${key}">${key}</button>`).join('')}</div></div>`;
    shell.append(controls);
    domRuntime?.claim('visitor-state', '.visitor-controls, .visitor-panel');
    domRuntime?.claimNode?.(owner, controls);
    domRuntime?.assertOwnership?.(owner, controls);

    const toggle = controls.querySelector('.visitor-toggle');
    const panel = controls.querySelector('.visitor-panel');
    const apply = () => {
      document.body.dataset.visitorIntent = state.intent;
      document.body.dataset.density = state.density;
      controls.querySelectorAll('[data-visitor-intent]').forEach(button => {
        button.toggleAttribute('aria-current', button.dataset.visitorIntent === state.intent);
      });
      controls.querySelectorAll('[data-density]').forEach(button => {
        button.toggleAttribute('aria-current', button.dataset.density === state.density);
      });
      context?.kernel?.record({
        type: 'visitor-intent',
        visible: state.intent,
        transformed: 'perspective changed',
        hidden: 'subject unchanged',
      });
    };

    const togglePanel = () => {
      const open = panel.hidden;
      panel.hidden = !open;
      toggle.setAttribute('aria-expanded', String(open));
      domRuntime?.mark('visitor-state-panel-toggled', { open, section: state.intent });
    };
    const intentHandlers = [];
    const densityHandlers = [];

    const intentButtons = controls.querySelectorAll('[data-visitor-intent]');
    intentButtons.forEach(button => {
      const handler = () => {
        state.intent = button.dataset.visitorIntent;
        apply();
      };
      intentHandlers.push([button, handler]);
      button.addEventListener('click', handler);
    });
    const densityButtons = controls.querySelectorAll('[data-density]');
    densityButtons.forEach(button => {
      const handler = () => {
        state.density = button.dataset.density;
        apply();
      };
      densityHandlers.push([button, handler]);
      button.addEventListener('click', handler);
    });

    toggle.addEventListener('click', togglePanel);
    apply();

    return {
      stop: () => {
        toggle.removeEventListener('click', togglePanel);
        intentHandlers.forEach(([button, handler]) => button.removeEventListener('click', handler));
        densityHandlers.forEach(([button, handler]) => button.removeEventListener('click', handler));
        controls.remove();
      },
    };
  };

  return { start };
};
