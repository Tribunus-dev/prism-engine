const levelOrder = ['released', 'validated', 'qualifying', 'implemented', 'planned'];

const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
}[character]));

const renderLegend = (root, levels) => {
  if (!root) return;
  root.innerHTML = levelOrder
    .filter((level) => levels[level])
    .map((level) => `<article data-level="${level}"><span>${level}</span><p>${escapeHtml(levels[level])}</p></article>`)
    .join('');
};

const renderCapabilities = (root, capabilities, filter = 'all') => {
  if (!root) return;
  const visible = filter === 'all' ? capabilities : capabilities.filter((entry) => entry.domain === filter);
  root.innerHTML = visible.map((entry) => {
    const paths = (entry.sourcePaths || []).map((path) => `<code>${escapeHtml(path)}</code>`).join('');
    const limitations = (entry.limitations || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('');
    const features = (entry.buildFeatures || []).map((item) => `<code>${escapeHtml(item)}</code>`).join('');
    return `<article class="capability-card" data-domain="${escapeHtml(entry.domain)}" data-level="${escapeHtml(entry.level)}">
      <div class="capability-card-head"><span>${escapeHtml(entry.domain)}</span><b>${escapeHtml(entry.level)}</b></div>
      <h3>${escapeHtml(entry.label)}</h3><p>${escapeHtml(entry.summary)}</p>
      <dl><div><dt>Source</dt><dd>${paths || '<span>Repository surface</span>'}</dd></div>${features ? `<div><dt>Features</dt><dd>${features}</dd></div>` : ''}</dl>
      <div class="capability-limit"><span>BOUNDARY</span><ul>${limitations}</ul></div>
    </article>`;
  }).join('') || '<p class="capability-empty">No capabilities match this filter.</p>';
};

const start = async () => {
  const response = await fetch('./data/capabilities.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Capability registry request failed: ${response.status}`);
  const registry = await response.json();
  document.querySelector('[data-capability-commit]').textContent = registry.generatedFromCommit.slice(0, 12);
  document.querySelector('[data-capability-date]').textContent = registry.verifiedAt;
  renderLegend(document.querySelector('[data-capability-legend]'), registry.levels || {});
  const grid = document.querySelector('[data-capability-grid]');
  renderCapabilities(grid, registry.capabilities || []);
  document.querySelectorAll('[data-capability-filter]').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('[data-capability-filter]').forEach((candidate) => candidate.classList.toggle('is-active', candidate === button));
      renderCapabilities(grid, registry.capabilities || [], button.dataset.capabilityFilter || 'all');
    });
  });
};

start().catch((error) => {
  console.error('[prism] capability registry failed', error);
  const root = document.querySelector('[data-capability-grid]');
  if (root) root.innerHTML = '<p class="capability-empty">The generated capability registry could not be loaded. Open <a href="data/capabilities.json">the source data</a> directly.</p>';
});
