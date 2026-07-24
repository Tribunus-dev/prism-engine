const levelOrder = ['released', 'validated', 'qualifying', 'implemented', 'planned'];

const inferDomainFromId = (id = '') => {
  const value = String(id || '').toLowerCase();
  if (value.includes('computeimage') || value.includes('artifact')) return 'artifact';
  if (value.includes('metal') || value.includes('rocm') || value.includes('cuda') || value.includes('xdna') || value.includes('ane') || value.includes('cpu') || value.includes('backend')) return 'runtime';
  if (value.includes('compiler')) return 'compiler';
  if (value.includes('replay') || value.includes('evidence')) return 'evidence';
  if (value.includes('constit') || value.includes('authority')) return 'authority';
  return 'model';
};

const inferLevel = (status = '') => {
  const normalized = String(status || '').toLowerCase().trim();
  if (normalized === 'repository-evidence' || normalized === 'measured' || normalized === 'validated') return 'validated';
  if (normalized === 'compile-verified' || normalized === 'illustrative' || normalized === 'architectural-derivation') return 'implemented';
  if (normalized === 'research-direction') return 'planned';
  return levelOrder.includes(normalized) ? normalized : 'implemented';
};

const fallbackCapabilitiesState = () => ({
  source: 'repository-state',
  commit: 'local-worktree',
  verifiedAt: 'unverified',
  levels: {
    released: 'Versioned, reproducible distribution with explicit support boundaries.',
    validated: 'Measured on a defined build and hardware configuration with evidence.',
    qualifying: 'Implemented and tested or compile-verified, with target evidence still being gathered.',
    implemented: 'Code path, data model, command, or provider boundary exists.',
    planned: 'Architecture or accepted design exists; end-to-end implementation is incomplete.',
  },
});

const normalizeRepositoryCapabilities = (repositoryState = {}) => {
  const capabilities = Array.isArray(repositoryState.capabilities) ? repositoryState.capabilities : [];
  return {
    generatedFromCommit: repositoryState.commit?.slice?.(0, 12) || 'local-worktree',
    verifiedAt: repositoryState.generatedAt || repositoryState.generated_at || 'local-worktree',
    levels: fallbackCapabilitiesState().levels,
    capabilities: capabilities.map((capability = {}, index) => ({
      id: capability.id || `capability-${index + 1}`,
      domain: capability.domain || inferDomainFromId(capability.id),
      label: capability.label || 'Capability',
      level: inferLevel(capability.status),
      summary: capability.summary || 'Capability summary is staged from repository state.',
      sourcePaths: capability.sourceRefs || [],
      buildFeatures: capability.tags || [],
      limitations: capability.limitations || ['Evidence boundary and support limits remain explicit.'],
    })),
  };
};

const loadCapabilityRegistry = async () => {
  const response = await fetch('./repository-state.json', { cache: 'no-store' });
  if (response.ok) {
    const repositoryState = await response.json();
    return {
      source: 'repository-state',
      ...normalizeRepositoryCapabilities(repositoryState),
    };
  }
  const fallback = await fetch('./data/capabilities.json', { cache: 'no-store' });
  if (!fallback.ok) throw new Error(`Capability registry request failed: ${response.status} / ${fallback.status}`);
  return {
    source: 'capability-registry',
    ...(await fallback.json()),
  };
};

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
  const registry = await loadCapabilityRegistry();
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
  if (root) root.innerHTML = '<p class="capability-empty">The generated capability registry could not be loaded. Open <a href="repository-state.json">the repository source</a> directly.</p>';
});
