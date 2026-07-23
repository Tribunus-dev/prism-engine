(() => {
  const track = document.querySelector('.signature-figure .signature-track');
  if (!track) return;

  const views = [
    ['01 / SEMANTIC VIEW', 'LogicalTensor', 'What the tensor means', 'name: attention.q_proj.weight · shape: [4096, 4096] · dtype: BF16', 'Illustrative fields — representative only; values vary by model.'],
    ['02 / STORAGE VIEW', 'PhysicalTileLayout', 'How its bits are stored', 'codec: mixed-precision · tile: 32×32 · residency: GPU local memory', 'Illustrative fields — representative only; values vary by target.'],
    ['03 / EXECUTION VIEW', 'ExecutionView', 'How a lane consumes it', 'lane: ROCm/HIP GPU · kernel: matmul · handoff: activation boundary', 'Illustrative fields — representative only; values vary by backend.'],
    ['04 / SEALED PROOF', 'ComputeImage', 'What execution is admitted', 'artifact: model.cimage · manifest: sealed · evidence: receipt attached', 'Illustrative fields — representative only; sealing and proof depend on the build.']
  ];

  const nodes = [...track.children].filter((node) => node.tagName === 'DIV');
  const inspector = document.createElement('aside');
  inspector.className = 'signature-inspector';
  inspector.setAttribute('aria-live', 'polite');
  inspector.innerHTML = '<span class="signature-inspector-kicker"></span><h3></h3><p class="signature-inspector-question"></p><code class="signature-inspector-fields"></code><small class="signature-inspector-disclaimer"></small>';
  track.parentElement.append(inspector);

  const set = (index) => {
    const [kicker, name, question, fields, disclaimer] = views[index];
    nodes.forEach((node, nodeIndex) => {
      const selected = nodeIndex === index;
      node.classList.toggle('signature-selected', selected);
      node.setAttribute('role', 'button');
      node.setAttribute('tabindex', '0');
      node.setAttribute('aria-pressed', String(selected));
    });
    inspector.classList.toggle('signature-proof', index === views.length - 1);
    inspector.querySelector('.signature-inspector-kicker').textContent = kicker;
    inspector.querySelector('h3').textContent = name;
    inspector.querySelector('.signature-inspector-question').textContent = question;
    inspector.querySelector('.signature-inspector-fields').textContent = fields;
    inspector.querySelector('.signature-inspector-disclaimer').textContent = disclaimer;
  };

  nodes.forEach((node, index) => {
    node.addEventListener('click', () => set(index));
    node.addEventListener('focus', () => set(index));
    node.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); set(index); }
    });
  });
  set(0);
})();
