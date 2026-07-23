(() => {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const chapterSelectors = [
    '.hero', '#working-path', '#compiler', '.tensor-journey',
    '.signature-figure', '#status', '#next-route', '.closing'
  ];
  const chapters = chapterSelectors.map(selector => document.querySelector(selector)).filter(Boolean);
  if (!chapters.length) return;

  const chapterElements = chapter => {
    const groups = [
      '.section-heading', '.hero-copy', '.hero-visual',
      '.working-path-grid > *', '.working-path-note',
      '.lab-window > *', '.lab-window .lab-step', '.lab-window .metrics > div',
      '.journey-frame > *', '.signature-track > *', '.figure-link',
      '.reality-bands > article', '.journey-guide-card', '.closing > *'
    ];
    return [...new Set(groups.flatMap(selector => [...chapter.querySelectorAll(selector)]))];
  };

  chapters.forEach((chapter, chapterIndex) => {
    chapter.dataset.scrollChapter = '';
    chapter.style.setProperty('--chapter-index', chapterIndex);
    const elements = chapterElements(chapter);
    elements.forEach((element, index) => {
      const connector = element.matches('.working-path-grid > i, .signature-track > i');
      const node = element.matches('.working-path-grid > .path-stage, .signature-track > div, .reality-bands > article, .journey-guide-card');
      element.dataset.scrollReveal = connector ? 'connector' : node ? 'node' : '';
      element.style.setProperty('--scroll-stagger', `${Math.min(index, 8) * 0.055}`);
    });
  });

  const clamp = value => Math.max(0, Math.min(1, value));
  let frame = 0;
  let lastScrollY = window.scrollY;

  const update = () => {
    frame = 0;
    const viewport = window.innerHeight || 1;
    const scrollDirection = window.scrollY >= lastScrollY ? 1 : -1;
    lastScrollY = window.scrollY;

    chapters.forEach((chapter, chapterIndex) => {
      const rect = chapter.getBoundingClientRect();
      const progress = clamp((viewport * 0.82 - rect.top) / Math.max(rect.height * 0.68, viewport * 0.42));
      const state = Math.min(4, Math.floor(progress * 5));
      chapter.style.setProperty('--chapter-progress', progress.toFixed(3));
      chapter.style.setProperty('--chapter-state', state);
      chapter.dataset.scrollState = String(state);
      chapter.dataset.scrollDirection = scrollDirection > 0 ? 'forward' : 'backward';

      const elements = chapter.querySelectorAll('[data-scroll-reveal]');
      elements.forEach(element => {
        const stagger = Number(element.style.getPropertyValue('--scroll-stagger')) || 0;
        const elementProgress = clamp((progress - stagger) / 0.28);
        element.style.setProperty('--element-progress', elementProgress.toFixed(3));
        element.classList.toggle('is-scroll-active', elementProgress > 0.72);
      });
    });

    const rail = document.querySelector('.scroll-state-rail');
    if (rail) rail.style.setProperty('--rail-progress', clamp((window.scrollY + viewport * 0.35) / Math.max(document.body.scrollHeight - viewport, 1)).toFixed(3));
  };

  const requestUpdate = () => { if (!frame) frame = requestAnimationFrame(update); };
  if (reduceMotion.matches) {
    chapters.forEach(chapter => {
      chapter.dataset.scrollState = '4';
      chapter.querySelectorAll('[data-scroll-reveal]').forEach(element => element.classList.add('is-scroll-active'));
    });
  } else {
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestUpdate, { passive: true });
    reduceMotion.addEventListener?.('change', () => window.location.reload());
    requestUpdate();
  }
})();
