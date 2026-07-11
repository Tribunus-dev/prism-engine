const PRISM_PAGES = [
  {
    type: 'cover',
    eyebrow: 'Field Guide 001 · July 2026',
    title: 'Prism Engine',
    subtitle: 'Governed native AI compute',
    body: `<img src="assets/tribunus-mark.svg" alt="Tribunus" class="cover-mark">
      <p class="cover-line">Compile once.<br>Execute explicitly.<br>Keep the receipt.</p>`,
    footer: 'Tribunus · AGPL-3.0-only'
  },
  {
    eyebrow: 'The premise',
    title: 'The output is not the whole execution.',
    body: `<p class="lead">A model answer cannot tell you which artifact ran, where its tensors lived, which fallback fired, what state changed, or whether the result survived a restart.</p>
      <p>Prism Engine treats inference as governed systems work. It joins a local model runtime and compute-image compiler to a transactional kernel that can validate, receipt, replay, and reject outcomes.</p>
      <blockquote>A backend result is evidence. It becomes authority only through a validated transaction and durable receipt.</blockquote>
      <div class="rule"><span>01</span><p>No hidden materialization.</p></div>
      <div class="rule"><span>02</span><p>No acknowledgement before durable evidence.</p></div>
      <div class="rule"><span>03</span><p>No claim broader than the proving gate.</p></div>`,
    footer: 'Why Prism exists'
  },
  {
    eyebrow: 'The product surface',
    title: 'One engine, two connected systems.',
    body: `<div class="split-block"><h3>Local runtime</h3><p>The <code>prism</code> CLI pulls, compiles, lists, and runs supported models. It offers interactive chat and an OpenAI-compatible HTTP endpoint.</p></div>
      <div class="split-block"><h3>Compute kernel</h3><p>The Tribunus compute core governs artifacts, devices, residency, sessions, work, compilation, ingress, and execution evidence.</p></div>
      <div class="split-block"><h3>Integration</h3><p>Rust libraries, a C ABI, Swift bridge, Node-API bindings, CLI binaries, and server surfaces share the workspace.</p></div>
      <p class="note">The focused runtime is usable today. The wider constitutional kernel is an active migration with explicit canonical, shadow, and development states.</p>`,
    footer: 'Runtime + authority substrate'
  },
  {
    eyebrow: 'Compute images',
    title: 'The model becomes an inspectable artifact.',
    body: `<img src="assets/compute-image-pipeline.svg" alt="Compute image pipeline from model source through qualification" class="full-diagram">
      <p class="caption">The compiler turns source weights and model configuration into a versioned <code>.cimage</code> artifact. Qualification binds format, layouts, execution intent, and validation evidence without pretending every backend shares one memory model.</p>
      <div class="fact-grid"><div><b>Identity</b><span>Digest and provenance</span></div><div><b>Layout</b><span>Typed tensor storage</span></div><div><b>Plan</b><span>Phases and residency</span></div><div><b>Evidence</b><span>Admission and receipts</span></div></div>`,
    footer: 'Source → .cimage → qualification'
  },
  {
    eyebrow: 'System architecture',
    title: 'Authority and execution are separate planes.',
    body: `<img src="assets/authority-planes.svg" alt="Prism authority, execution, evidence, and projection planes" class="full-diagram tall-diagram">
      <p class="caption">Hardware handles, queues, caches, and backend results remain execution-plane facts. Validated ECS transactions produce durable events and receipts. Analytical views can always be rebuilt.</p>`,
    footer: 'The constitutional boundary'
  },
  {
    eyebrow: 'Transaction lifecycle',
    title: 'Every accepted change has a mechanical path.',
    body: `<img src="assets/transaction-lifecycle.svg" alt="Command validation transaction effects receipt and replay lifecycle" class="full-diagram tall-diagram">
      <div class="axiom"><b>Reject stale outcomes.</b><p>Completion must still match the entity generation, lease, resource claim, and expected effect.</p></div>
      <div class="axiom"><b>Commit durably before acknowledgement.</b><p>A queue terminal state or derived row cannot replace the authoritative receipt.</p></div>`,
    footer: 'Command → receipt → replay'
  },
  {
    eyebrow: 'Canonical state',
    title: 'One ECS is the authority spine.',
    body: `<table class="zine-table"><thead><tr><th>Domain</th><th>State</th></tr></thead><tbody>
      <tr><td>Artifact ingestion</td><td><span class="state done">Replay verified</span></td></tr>
      <tr><td>Device discovery</td><td><span class="state done">Canonical</span></td></tr>
      <tr><td>Model residency</td><td><span class="state shadow">Shadow</span></td></tr>
      <tr><td>Sessions and work</td><td><span class="state shadow">Shadow</span></td></tr>
      <tr><td>Execution leases</td><td><span class="state shadow">Shadow</span></td></tr>
      <tr><td>Compilation and pipelines</td><td><span class="state shadow">Shadow</span></td></tr>
      <tr><td>Distributed topology and ingress</td><td><span class="state shadow">Shadow</span></td></tr>
      <tr><td>Legacy authority purge</td><td><span class="state active">Active</span></td></tr>
      </tbody></table>
      <p class="note"><b>Shadow</b> means a transactional representation exists alongside legacy runtime authority. It does not mean cutover is complete.</p>`,
    footer: 'Current migration state'
  },
  {
    eyebrow: 'Storage truth',
    title: 'Durable truth, coordination, and analysis do different jobs.',
    body: `<div class="truth-stack"><div class="truth primary"><b>PGlite / PostgreSQL</b><span>Durable authority</span></div><div class="truth"><b>Valkey</b><span>Coordination visibility</span></div><div class="truth"><b>DuckDB</b><span>Analytical projection</span></div><div class="truth"><b>Tokio</b><span>Ephemeral local execution</span></div><div class="truth"><b>IOSurface / backend memory</b><span>Hardware execution facts</span></div></div>
      <blockquote>A Valkey acknowledgement is not truth. A DuckDB row is not truth. A backend result is not truth. Only a durable receipt can become authority.</blockquote>`,
    footer: 'Storage truth doctrine'
  },
  {
    eyebrow: 'Apple Silicon',
    title: 'The strongest path is native and explicit.',
    body: `<div class="hardware-map"><div class="chip">Unified memory fabric</div><div class="hardware-row"><div><b>CPU</b><span>Control, fallback, validation</span></div><div><b>GPU / Metal</b><span>Primary accelerated dispatch</span></div><div><b>ANE / Core ML</b><span>Integrated, still qualifying</span></div></div><div class="memory-bar">Residency · layouts · synchronization · completion evidence</div></div>
      <p>Prism is optimized first for Apple Silicon, where Metal is the primary accelerated route. Accelerate and Core ML integrations serve distinct execution cells rather than interchangeable labels.</p>
      <p class="note">A successful Metal build does not prove ANE execution, numerical admission, or every declared binary.</p>`,
    footer: 'Metal first, hardware honest'
  },
  {
    eyebrow: 'Portability',
    title: 'Hardware differences stay visible.',
    body: `<table class="zine-table backend-status"><thead><tr><th>Target</th><th>Current claim</th></tr></thead><tbody>
      <tr><td>Apple Metal</td><td>Primary accelerated path</td></tr>
      <tr><td>Core ML / ANE</td><td>Integrated; qualifying</td></tr>
      <tr><td>CPU / Linux</td><td>Build checked; runtime hardening</td></tr>
      <tr><td>AMD ROCm</td><td>Development surface</td></tr>
      <tr><td>Intel / Level Zero</td><td>Development surface</td></tr>
      <tr><td>CUDA / Tensix</td><td>Development surface</td></tr>
      </tbody></table>
      <div class="big-quote">No hidden copies.<br><span>Explicit materialization whenever required. Zero-copy only where the backend can prove it.</span></div>`,
    footer: 'Portable semantics, native lowering'
  },
  {
    eyebrow: 'Scheduling and residency',
    title: 'Placement is a governed decision.',
    body: `<img src="assets/scheduling-map.svg" alt="Scheduling and model residency decision flow" class="full-diagram tall-diagram">
      <p class="caption">A work item does not select a backend by convenience alone. Capability observations, resource claims, model format, memory limits, lease ownership, and health all constrain placement.</p>`,
    footer: 'Discover → admit → lease → execute'
  },
  {
    eyebrow: 'Recovery',
    title: 'Restart is part of the correctness model.',
    body: `<div class="replay-sequence"><div><b>01</b><span>Open durable event store</span></div><div><b>02</b><span>Verify event envelopes</span></div><div><b>03</b><span>Apply registered replay functions</span></div><div><b>04</b><span>Reconstruct canonical entities</span></div><div><b>05</b><span>Rebuild derived projections</span></div><div><b>06</b><span>Reject orphaned or stale execution outcomes</span></div></div>
      <blockquote>Recovery is not “load whatever the cache remembers.” It is deterministic reconstruction from admitted evidence.</blockquote>`,
    footer: 'Durable before ack'
  },
  {
    eyebrow: 'Evidence boundaries',
    title: 'Green means exactly what the gate ran.',
    body: `<div class="gate"><span>BUILD</span><p>Proves selected sources and features compile on that toolchain.</p></div><div class="gate"><span>TEST</span><p>Proves the exercised cases under their fixtures and environment.</p></div><div class="gate"><span>HARDWARE</span><p>Proves a route on the observed device, driver, shape, and model.</p></div><div class="gate"><span>RECEIPT</span><p>Records the admitted artifact, route, outcome, and authority transition.</p></div>
      <p class="note">Continuous integration checks an Apple <code>prism-backend</code> library path and a narrower Linux CPU build. It does not qualify every backend, model family, or binary in the workspace.</p>`,
    footer: 'Evidence, not implication'
  },
  {
    eyebrow: 'Integration map',
    title: 'Many callers, one governed core.',
    body: `<div class="integration-map"><div class="callers"><span>CLI</span><span>HTTP</span><span>Swift</span><span>Node</span><span>C ABI</span></div><div class="arrow-down">↓ commands / queries</div><div class="kernel-box">Tribunus compute core<br><small>ECS · compiler · scheduler · evidence · replay</small></div><div class="arrow-down">↓ execution contracts</div><div class="backends"><span>Metal</span><span>Core ML</span><span>CPU</span><span>Future native backends</span></div></div>
      <p class="caption">Bridges produce commands and consume projections. They do not become independent sources of domain truth.</p>`,
    footer: 'FFI and application boundaries'
  },
  {
    eyebrow: 'Work ahead',
    title: 'The next wins are narrow and executable.',
    body: `<div class="work-item"><b>Authority cutover</b><p>Move remaining model, cache, trust, routing, and server registries onto transactional ECS paths.</p></div><div class="work-item"><b>Backend qualification</b><p>Attach numerical and performance claims to reproducible hardware evidence.</p></div><div class="work-item"><b>Portable execution</b><p>Harden CPU execution and make non-Apple memory, launch, and completion contracts explicit.</p></div><div class="work-item"><b>Adversarial recovery</b><p>Stress replay determinism, stale rejection, failure atomicity, privacy-preserving receipts, and projection rebuilding.</p></div>
      <p class="collaboration">Challenge the architecture with executable counterexamples.</p>`,
    footer: 'Contribute with evidence'
  },
  {
    type: 'back',
    eyebrow: 'Prism Engine · Field Guide 001',
    title: 'Build from the evidence outward.',
    body: `<p class="back-statement">Prism Engine is pre-1.0 systems software. Its strongest path today is native Apple Silicon execution; its larger goal is portable compute whose state, placement, effects, and recovery are mechanically inspectable.</p>
      <div class="qr-grid"><div><img src="assets/qr-github.svg" alt="GitHub QR code"><b>Source</b><span>Tribunus-dev/prism-engine</span></div><div><img src="assets/qr-research.svg" alt="Research QR code"><b>Research</b><span>research.tribunus.dev</span></div></div>
      <p class="license">AGPL-3.0-only<br>Designed to print on four duplex US Letter sheets.</p>`,
    footer: 'github.com/Tribunus-dev/prism-engine'
  }
];

function pageMarkup(page, pageNumber) {
  const kind = page.type ? ` page-${page.type}` : '';
  return `<article class="page${kind}" data-page="${pageNumber}">
    <div class="page-inner">
      <header class="page-header"><span>${page.eyebrow}</span><span>${String(pageNumber).padStart(2, '0')} / 16</span></header>
      <main class="page-content"><h1>${page.title}</h1>${page.subtitle ? `<h2>${page.subtitle}</h2>` : ''}${page.body}</main>
      <footer class="page-footer"><span>${page.footer}</span><span>PRISM / ${String(pageNumber).padStart(2, '0')}</span></footer>
    </div>
  </article>`;
}

function renderReadingEdition(target) {
  target.innerHTML = PRISM_PAGES.map((page, index) => pageMarkup(page, index + 1)).join('');
}

function renderImposedEdition(target) {
  const sides = [
    ['Sheet 1 · front', 16, 1], ['Sheet 1 · back', 2, 15],
    ['Sheet 2 · front', 14, 3], ['Sheet 2 · back', 4, 13],
    ['Sheet 3 · front', 12, 5], ['Sheet 3 · back', 6, 11],
    ['Sheet 4 · front', 10, 7], ['Sheet 4 · back', 8, 9]
  ];
  target.innerHTML = sides.map(([label, left, right]) => `<section class="sheet-side">
    <div class="sheet-label screen-only">${label} · pages ${left} + ${right}</div>
    <div class="spread">${pageMarkup(PRISM_PAGES[left - 1], left)}${pageMarkup(PRISM_PAGES[right - 1], right)}<i class="fold-guide screen-only"></i></div>
  </section>`).join('');
}
