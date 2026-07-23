(() => {
  const chapter = document.querySelector('.computeimage-anatomy');
  if (!chapter) return;
  const layers = [...chapter.querySelectorAll('[data-anatomy-layer]')];
  const progress = chapter.querySelector('[data-anatomy-progress]');
  const label = chapter.querySelector('[data-anatomy-label]');
  const description = chapter.querySelector('[data-anatomy-description]');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const copy = [
    ['Metadata', 'Identity and target contract are admitted first.'],
    ['Logical tensors', 'Tensor meaning stays separate from physical storage.'],
    ['Execution views', 'Kernels, queues, and residency describe how work is consumed.'],
    ['Receipts', 'Evidence records the quality and legality boundary; values are not implied here.'],
    ['Payload', 'Tiles and packed bytes complete the illustrative sealed object.']
  ];
  let frame = 0;
  const clamp = value => Math.max(0, Math.min(1, value));
  const update = () => {
    frame = 0;
    if (reduced.matches) return;
    const rect = chapter.getBoundingClientRect();
    const chapterProgress = Number.parseFloat(chapter.style.getPropertyValue('--chapter-progress'));
    const travel = Math.max(chapter.offsetHeight - window.innerHeight, 1);
    const amount = Number.isFinite(chapterProgress) ? clamp(chapterProgress) : clamp(-rect.top / travel);
    const stage = Math.min(copy.length - 1, Math.floor(amount * copy.length));
    layers.forEach((layer, index) => layer.classList.toggle('is-revealed', index <= stage));
    if (progress) progress.style.width = `${amount * 100}%`;
    if (label) label.textContent = copy[stage][0];
    if (description) description.textContent = copy[stage][1];
  };
  const requestUpdate = () => { if (!frame) frame = requestAnimationFrame(update); };
  if (reduced.matches) {
    layers.forEach(layer => layer.classList.add('is-revealed'));
    if (progress) progress.style.width = '100%';
    if (label) label.textContent = copy[copy.length - 1][0];
    if (description) description.textContent = copy[copy.length - 1][1];
  } else {
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestUpdate, { passive: true });
    reduced.addEventListener?.('change', () => window.location.reload());
    requestUpdate();
  }
})();
