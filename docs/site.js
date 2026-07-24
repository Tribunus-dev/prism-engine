// Single explicit ES-module entrypoint. Dependencies and initialization order live in app.js.
if (new URLSearchParams(location.search).get('prismRuntime') !== 'off') {
  window.addEventListener('error', event => { console.error('[prism]', event.error || event.message || 'runtime error'); });
  window.addEventListener('unhandledrejection', event => { console.error('[prism]', event.reason || 'runtime rejection'); });
  import('./js/app.js').catch(error => { console.error('[prism] module load failed', error); });
}
