(() => {
  const story = document.querySelector('.apple-story');
  if (!story) return;
  const chapters = [...story.querySelectorAll('.scroll-chapter')];
  if (!chapters.length) return;

  let boundaries = [];
  let ticking = false;
  const clamp = value => Math.max(0, Math.min(1, value));
  const ease = value => value * value * (3 - 2 * value);
  const measure = () => {
    boundaries = chapters.map(chapter => chapter.getBoundingClientRect().top + window.scrollY);
    update();
  };
  const update = () => {
    ticking = false;
    const viewport = Math.max(1, window.innerHeight);
    const scroll = window.scrollY;
    chapters.forEach((chapter, index) => {
      const progress = clamp((scroll - boundaries[index]) / viewport);
      const focus = clamp(1 - Math.abs(progress - 0.5) * 2);
      const eased = ease(progress);
      chapter.style.setProperty('--chapter-progress', progress.toFixed(4));
      chapter.style.setProperty('--chapter-focus', focus.toFixed(4));
      chapter.style.setProperty('--chapter-enter', eased.toFixed(4));
      chapter.style.setProperty('--chapter-heading-y', `${((0.5 - focus) * 24).toFixed(2)}px`);
      chapter.style.setProperty('--chapter-content-y', `${((0.5 - focus) * 14).toFixed(2)}px`);
      chapter.style.setProperty('--chapter-content-scale', (0.985 + focus * 0.015).toFixed(4));
      chapter.style.setProperty('--chapter-opacity', Math.max(0.42, 0.42 + focus * 0.58).toFixed(4));
      chapter.style.setProperty('--chapter-blur', `${((1 - focus) * 1.5).toFixed(2)}px`);
      chapter.style.setProperty('--chapter-index-progress', index);
    });
    const hero = story.querySelector('.hero');
    if (hero) {
      const progress = clamp((scroll - boundaries[0]) / viewport);
      hero.style.setProperty('--hero-scale', (1 + progress * 0.035).toFixed(4));
      hero.style.setProperty('--hero-copy-y', `${(-progress * 18).toFixed(2)}px`);
    }
  };
  const requestUpdate = () => {
    if (!ticking) {
      ticking = true;
      window.requestAnimationFrame(update);
    }
  };
  window.addEventListener('scroll', requestUpdate, { passive: true });
  window.addEventListener('resize', measure, { passive: true });
  measure();
})();
