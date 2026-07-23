(() => {
  const selectors = [
    '.hero-copy > *', '.hero-visual > *', '.signal-inner > *',
    '.working-path .section-heading', '.working-path-grid > *', '.working-path-note',
    '#compiler .section-heading', '#compiler .lab-window', '#compiler .caption-row',
    '.tensor-journey .section-heading', '.tensor-journey .journey-frame',
    '.signature-figure .section-heading', '.signature-track > *', '.figure-link',
    '#status .section-heading', '.reality-bands article',
    '#next-route .section-heading', '.journey-guide-card', '.closing > *'
  ];
  const nodes = [...new Set(selectors.flatMap(selector => [...document.querySelectorAll(selector)]))];
  nodes.forEach((node, index) => { node.dataset.scrollReveal = ''; node.style.setProperty('--reveal-delay', `${Math.min(index % 7, 6) * 75}ms`); });
  const rail = document.querySelector('.scroll-state-rail');
  const reveal = node => node.classList.add('is-visible');
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window)) {
    nodes.forEach(reveal); if (rail) rail.classList.add('is-visible'); return;
  }
  const observer = new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) { reveal(entry.target); observer.unobserve(entry.target); } }), { rootMargin: '0px 0px -10% 0px', threshold: .12 });
  nodes.forEach(node => observer.observe(node));
  if (rail) setTimeout(() => rail.classList.add('is-visible'), 250);
})();
