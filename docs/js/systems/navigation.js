
export const chapterMap = [
  ['Introduction', 'index.html', 'compiler'],
  ['Start Here', 'field-guide.html', 'compiler'],
  ['Architecture', 'architecture.html', 'compiler'],
  ['Compilation', 'demo.html#compiler', 'compiler'],
  ['Runtime', 'heterogeneous.html', 'runtime'],
  ['ComputeImages', 'general-compute.html', 'hardware'],
  ['Research', 'prism-ml.html', 'research'],
  ['Roadmap', 'roadmap.html', 'proof'],
  ['Work With Prism', 'work-with-prism.html', 'research'],
];

export const currentChapter = (context = {}) => {
  const file = context?.runtime?.getProjection?.()?.route
    || context?.runtime?.currentRoute
    || context?.currentFile
    || context?.location
    || location.pathname.split('/').pop()
    || 'index.html';
  return Math.max(0, chapterMap.findIndex(chapter => chapter[1].split('#')[0] === file));
};

export const createNavigationSystem = () => {
  const start = (context) => {
    const kernel = context?.kernel;
    const domRuntime = context?.domRuntime;
    const owner = 'navigation';
    const createInScope = () => {
      const header = document.querySelector('.component-header');
      const shell = header || document.body;
      const nodes = [];
      const current = currentChapter(context);
      const [name, , accent] = chapterMap[current];
      const primary = [0, 2, 3, 4, 6];
      const primaryIndex = primary.indexOf(current);
      document.body.style.setProperty('--chapter-accent', `var(--${accent})`);
      document.querySelectorAll('[data-reading-chapter]').forEach(element => { element.textContent = name; });
      document.querySelectorAll('[data-reading-count]').forEach(element => { element.textContent = primaryIndex >= 0 ? `Primary journey · ${primaryIndex + 1} of ${primary.length}` : 'Reference surface'; });
      document.querySelectorAll('.reading-meter i').forEach((element, index) => element.classList.toggle('is-filled', primaryIndex >= 0 && index <= primaryIndex));

      const progress = document.createElement('div');
      progress.className = 'chapter-progress';
      shell.prepend(progress);
      domRuntime?.claimNode?.(owner, progress);
      const rail = document.createElement('nav');
      rail.className = 'chapter-rail';
      rail.setAttribute('aria-label', 'Observatory orientation');
      rail.innerHTML = `<span class="chapter-rail-group">Primary journey</span>${primary.map((index, position) => { const item = chapterMap[index]; return `<a href="${item[1]}" ${index === current ? 'aria-current="page"' : ''}>${String(position + 1).padStart(2, '0')} ${item[0]}</a>`; }).join('')}<span class="chapter-rail-group">Reference</span>${chapterMap.map(([label, href], index) => primary.includes(index) ? '' : `<a href="${href}" ${index === current ? 'aria-current="page"' : ''}>${label}</a>`).join('')}`;
      shell.append(rail);
      domRuntime?.claimNode?.(owner, rail);
      nodes.push(progress, rail);

      const footer = document.querySelector('footer.site-footer');
      if (footer) {
        const nav = document.createElement('nav');
        nav.className = 'chapter-nav';
        nav.setAttribute('aria-label', 'Continue observing');
        const previous = primaryIndex > 0 ? chapterMap[primary[primaryIndex - 1]] : null;
        const next = primaryIndex >= 0 && primaryIndex < primary.length - 1 ? chapterMap[primary[primaryIndex + 1]] : null;
        nav.innerHTML = `${previous ? `<a href="${previous[1]}"><small>Previous observation</small><strong>← ${previous[0]}</strong></a>` : '<span></span>'}${next ? `<a href="${next[1]}"><small>Continue observing</small><strong>${next[0]} →</strong></a>` : '<span></span>'}`;
        footer.before(nav);
        domRuntime?.claimNode?.(owner, nav);
        nodes.push(nav);
      }

      const update = ({ progress: amount } = {}) => {
        const max = document.documentElement.scrollHeight - innerHeight;
        const value = Number.isFinite(amount) ? amount : (max ? scrollY / max : 0);
        progress.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
      };
      const scrollHandler = event => update(event);
      kernel?.on('scroll', scrollHandler);
      addEventListener('resize', update);
      update();
      return {
        stop: () => {
          kernel?.off?.('scroll', scrollHandler);
          removeEventListener('resize', update);
          nodes.forEach((node) => node?.remove?.());
        }
      };
    };

    const initMenu = () => {
      const button = document.querySelector('.menu-toggle');
      const navigation = document.querySelector('#primary-navigation');
      if (!button || !navigation) return;
      const close = (focus = false) => {
        navigation.classList.remove('is-open');
        button.setAttribute('aria-expanded', 'false');
        if (focus) button.focus();
      };
      button.addEventListener('click', () => {
        const open = navigation.classList.toggle('is-open');
        button.setAttribute('aria-expanded', String(open));
        if (open) navigation.querySelector('a')?.focus();
      });
      navigation.querySelectorAll('a').forEach(link => link.addEventListener('click', () => close()));
      const escape = event => {
        if (event.key === 'Escape' && navigation.classList.contains('is-open')) {
          event.preventDefault();
          close(true);
        }
      };
      document.addEventListener('keydown', escape);
      return {
        stop: () => {
          document.removeEventListener('keydown', escape);
        },
      };
    };

    initMenu();
    const chapterScope = createInScope();
    domRuntime?.claim('navigation', '.chapter-progress, .chapter-rail, .chapter-nav');
    return chapterScope;
  };

  return { start };
};
