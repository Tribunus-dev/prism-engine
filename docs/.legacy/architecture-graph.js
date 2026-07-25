(() => {
  const nodes = [...document.querySelectorAll('.system-node')];
  if (!nodes.length) return;
  const data = {
    graph: ['MODEL / IR', 'Model graph', 'Owns the semantic model: tensor identity, operator relationships, shape contracts, and graph boundaries.', 'Weights, tokenizer, metadata', 'Graph IR and tensor contracts', 'Evolution search · Physical layout', 'If the graph changes, candidate generation and every downstream execution view are re-evaluated.'],
    search: ['COMPILER / SEARCH', 'Evolution search', 'Explores precision, codec, fusion, tile, and KV candidates against the reference and resource budget.', 'Graph IR and quality contract', 'Admitted candidate frontier', 'Model graph · Physical layout · Evidence', 'A better candidate changes layout pressure, kernel choice, residency, and the proof that reaches the runtime.'],
    layout: ['COMPILER / ABI', 'Physical layout', 'Turns logical tensors into tiled, aligned, codec-aware storage and execution views.', 'Candidate frontier and target caps', 'Tile layouts and execution views', 'Evolution search · Residency · Runtime scheduler', 'Changing a tile or codec propagates into buffer sizes, kernel signatures, DMA plans, and residency.'],
    runtime: ['RUNTIME / CONTROL', 'Runtime scheduler', 'Executes the sealed plan: chooses views, orders work, dispatches backends, and records completion.', 'CImage, request, device capabilities', 'Tokens, metrics, execution receipt', 'Physical layout · Residency · Evidence', 'The scheduler does not invent policy; it realizes the contracts produced upstream.'],
    residency: ['RUNTIME / MEMORY', 'Residency', 'Owns where weights, activations, KV state, and scratch live across CPU, GPU, NPU, and shared memory.', 'Execution views and device memory', 'Ownership, buffers, transfer edges', 'Physical layout · Runtime scheduler · Evidence', 'A residency change can remove a PCIe transfer, alter tile capacity, or force a different device plan.'],
    evidence: ['GOVERNANCE / PROOF', 'Evidence', 'Captures quality, legality, cost, receipts, and replay facts so decisions remain inspectable after compilation.', 'Search scores and runtime outcomes', 'Proof, receipts, projections', 'Evolution search · Runtime scheduler · Residency', 'Evidence feeds the next search and makes every hardware-specific change comparable.']
  };
  const map = document.querySelector('.system-map');
  const set = key => {
    const value = data[key];
    if (!value) return;
    nodes.forEach(node => {
      const active = node.dataset.system === key;
      node.classList.toggle('active', active);
      node.setAttribute('aria-pressed', String(active));
    });
    document.querySelector('#system-kind').textContent = value[0];
    document.querySelector('#system-name').textContent = value[1];
    document.querySelector('#system-description').textContent = value[2];
    document.querySelector('#system-input').textContent = value[3];
    document.querySelector('#system-output').textContent = value[4];
    document.querySelector('#system-neighbors').textContent = value[5];
    document.querySelector('#system-note').textContent = value[6];
  };
  nodes.forEach(node => node.addEventListener('click', () => set(node.dataset.system)));
  set('graph');
  if (!map || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const wires = map.querySelector('.system-wires');
  const reveal = () => {
    const rect = map.getBoundingClientRect();
    const amount = Math.max(0, Math.min(1, (innerHeight * .72 - rect.top) / Math.max(rect.height * .9, 1)));
    nodes.forEach((node, index) => node.toggleAttribute('data-assembled', index < Math.max(1, Math.ceil(amount * nodes.length))));
    wires?.style.setProperty('--assembly-progress', String(amount));
    map.dataset.assemblyPhase = amount < .28 ? 'model' : amount < .58 ? 'compiler' : amount < .82 ? 'runtime' : 'evidence';
  };
  addEventListener('scroll', reveal, { passive: true });
  addEventListener('resize', reveal);
  reveal();
})();
(() => {
  const map = document.querySelector('.system-map');
  if (!map) return;
  const byPhase = { model: 'graph', compiler: 'search', runtime: 'runtime', evidence: 'evidence' };
  let phase = '';
  const sync = () => {
    const next = byPhase[map.dataset.assemblyPhase];
    if (next && next !== phase) { phase = next; map.querySelector(`[data-system="${next}"]`)?.click(); }
  };
  new MutationObserver(sync).observe(map, { attributes: true, attributeFilter: ['data-assembly-phase'] });
  sync();
})();
