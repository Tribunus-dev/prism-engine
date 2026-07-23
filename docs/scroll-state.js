(() => {
  const stages = [
    ['01', 'SOURCE', 'Model identity enters the compiler.'],
    ['02', 'COMPILE', 'Graph semantics and lowered work become explicit.'],
    ['03', 'SEARCH', 'Representations evolve against the BF16 canary.'],
    ['04', 'REALIZE', 'The target hardware shapes execution views.'],
    ['05', 'PROVE', 'Quality, legality, and receipts gate admission.']
  ];
  const sources = ['.hero', '#working-path', '#compiler', '#architecture', '#status'];
  const sections = sources.map(selector => document.querySelector(selector)).filter(Boolean);
  if (!sections.length) return;

  const rail = document.createElement('aside');
  rail.className = 'scroll-state-rail';
  rail.setAttribute('aria-label', 'Compiler state progress');
  rail.innerHTML = `<div class="scroll-state-title">PRISM / COMPILER STATE</div><div class="scroll-state-items">${stages.map((s, i) => `<button type="button" data-scroll-stage="${i}" aria-label="${s[1]}: ${s[2]}"><b>${s[0]}</b><span>${s[1]}</span></button>`).join('')}</div><p class="scroll-state-description"></p>`;
  document.body.append(rail);
  const items = [...rail.querySelectorAll('[data-scroll-stage]')];
  const description = rail.querySelector('.scroll-state-description');
  sections.forEach((section, index) => { section.classList.add('scroll-chapter'); section.style.setProperty('--chapter-index', index); });

  let boundaries = [];
  const measure = () => { boundaries = sections.map(section => section.getBoundingClientRect().top + window.scrollY); update(); };
  const update = () => {
    const probe = window.scrollY + Math.max(1, window.innerHeight * 0.48);
    let index = 0;
    boundaries.forEach((boundary, i) => { if (boundary <= probe) index = i; });
    const safe = Math.max(0, Math.min(index, stages.length - 1));
    document.documentElement.dataset.compilerState = stages[safe][1].toLowerCase();
    sections.forEach((section, i) => section.classList.toggle('is-chapter-active', i === safe));
    items.forEach((item, i) => { item.classList.toggle('is-active', i === safe); item.classList.toggle('is-complete', i < safe); item.setAttribute('aria-current', i === safe ? 'step' : 'false'); });
    description.textContent = stages[safe][2];
  };

  items.forEach((item, index) => item.addEventListener('click', () => sections[index] && sections[index].scrollIntoView({ behavior: 'smooth', block: 'start' })));
  let ticking = false;
  window.addEventListener('scroll', () => { if (!ticking) { window.requestAnimationFrame(() => { ticking = false; update(); }); ticking = true; } }, { passive: true });
  window.addEventListener('resize', measure, { passive: true });
  measure();
})();
