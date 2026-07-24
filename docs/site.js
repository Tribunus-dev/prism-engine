// Single explicit ES-module entrypoint. Dependencies and initialization order live in app.js.

function ensureFavicon(linkHref) {
  const existing = document.querySelector('link[rel="icon"]');
  if (existing) return;
  const icon = document.createElement('link');
  icon.rel = 'icon';
  icon.type = 'image/svg+xml';
  icon.href = linkHref;
  document.head.appendChild(icon);
}

function canonicalPathFromLocation() {
  const path = window.location.pathname;
  if (!path || path === '/' || path === '/docs/' || path === '/docs') {
    return '/';
  }
  if (path.endsWith('/index.html')) {
    return '/';
  }
  return path.replace(/\.html$/, '');
}

function ensureHeadMeta() {
  const baseUrl = new URL('https://prism-engine.tribunus.dev');
  const canonicalHref = new URL(canonicalPathFromLocation(), baseUrl).toString();
  const title = document.title || 'Prism Engine Observatory';
  const description = document.querySelector('meta[name="description"]')?.content
    || 'Prism Engine Observatory: runtime-first documentation and experiments in compute-image orchestration.';

  const canonical = document.createElement('link');
  canonical.rel = 'canonical';
  canonical.href = canonicalHref;
  document.head.appendChild(canonical);

  const hasOgTitle = document.querySelector('meta[property="og:title"]');
  if (!hasOgTitle) {
    const ogTitle = document.createElement('meta');
    ogTitle.setAttribute('property', 'og:title');
    ogTitle.content = title;
    document.head.appendChild(ogTitle);
  }

  const hasOgDescription = document.querySelector('meta[property="og:description"]');
  if (!hasOgDescription) {
    const ogDescription = document.createElement('meta');
    ogDescription.setAttribute('property', 'og:description');
    ogDescription.content = description;
    document.head.appendChild(ogDescription);
  }

  const hasOgUrl = document.querySelector('meta[property="og:url"]');
  if (!hasOgUrl) {
    const ogUrl = document.createElement('meta');
    ogUrl.setAttribute('property', 'og:url');
    ogUrl.content = canonicalHref;
    document.head.appendChild(ogUrl);
  }
}

if (new URLSearchParams(location.search).get('prismRuntime') !== 'off') {
  ensureFavicon('/docs/favicon.svg');
  ensureHeadMeta();
  window.addEventListener('error', event => { console.error('[prism]', event.error || event.message || 'runtime error'); });
  window.addEventListener('unhandledrejection', event => { console.error('[prism]', event.reason || 'runtime rejection'); });
  import('/js/app.js').catch(error => { console.error('[prism] module load failed', error); });
}
