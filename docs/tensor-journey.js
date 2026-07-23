(() => {
  const svg = document.querySelector('.journey-svg');
  if (!svg) return;
  const token = svg.querySelector('.tensor-token');
  const stations = [...svg.querySelectorAll('.journey-station')];
  const stage = document.querySelector('#journey-stage');
  const caption = document.querySelector('#journey-caption');
  const play = document.querySelector('#journey-play');
  const reset = document.querySelector('#journey-reset');
  const toolbar = document.querySelector('.journey-toolbar');
  const stages = [
    ['01 / INGEST', 'Read the source tensor and preserve its identity in the ECS world.'],
    ['02 / GRAPH', 'Attach the tensor to semantic operators, shapes, and dependencies.'],
    ['03 / REPRESENT', 'Search progressive quantization, ternarization, and mixed-precision candidates.'],
    ['04 / LOWER', 'Turn the selected graph region into explicit uops and kernel work.'],
    ['05 / PLACE', 'Choose CPU, GPU, or NPU lanes, spatial regions, memory, and queues.'],
    ['06 / KV + VERIFY', 'Search KV-cache policy, check legality, quality, resources, and replay evidence.'],
    ['07 / CIMAGE', 'Seal the tensor view into the target-aware executable CImage.']
  ];
  const tensorStates = [
    ['Logical identity', 'attention.q_proj.weight', 'shape [4096, 4096] · BF16 · projection weight', 'Source bytes are named and attributed before any physical choice.', 'SOURCE / MODEL'],
    ['Graph semantics', 'attention.q_proj.weight', 'role projection · axes [out, in] · dependency attention.q', 'The tensor is attached to ECS graph edges and shape contracts.', 'ECS / GRAPH'],
    ['Representation frontier', 'attention.q_proj.weight', 'BF16 → INT8 → ternary candidates · loss gate pending', 'Candidate formats are compared against the reference; no choice is implied by the animation.', 'SEARCH / Q + T'],
    ['Lowered work', 'attention.q_proj.weight', 'matmul uops · tile-independent kernel contract', 'Logical work becomes explicit operations without choosing a vendor kernel yet.', 'LOWER / UOPS'],
    ['Target execution view', 'attention.q_proj.weight', 'MI300X ROCm/HIP · 32×32 tiles · GPU-local residency', 'The target profile adds tile, kernel, queue, and fallback requirements.', 'TARGET / MI300X'],
    ['KV + evidence gate', 'attention.q_proj.weight', 'KV policy: compressed candidate · numerical proof pending', 'State policy and validation gates are recorded before publication.', 'PROVE / KV + LOSS'],
    ['ComputeImage view', 'attention.q_proj.weight', 'model.cimage · execution view sealed · receipt required', 'The artifact can carry the selected view; execution proof still comes from a real run.', 'ARTIFACT / CIMAGE']
  ];
  const targets = {
    mi300x: ['MI300X / ROCm-HIP', 'gfx942 · GPU-local HBM · HIP kernels'],
    apple: ['APPLE SILICON / METAL', 'unified memory · Metal kernels · CPU fallback'],
    xdna: ['XDNA2 / SPATIAL PLAN', 'tiles · FIFOs · DMA · legality boundary']
  };
  const candidates = [
    ['BF16 reference', 'baseline', 'reference'],
    ['INT8 per-channel', 'lower memory', 'admitted'],
    ['Ternary + BF16 fallback', 'aggressive compression', 'frontier'],
    ['Mixed precision + target tiles', 'best fit', 'selected']
  ];
  let target = 'mi300x';
  let current = 0;
  let timer = null;
  const points = [[110, 235], [250, 100], [410, 370], [585, 100], [750, 370], [915, 100], [1090, 235]];
  const stationLabels = [['SOURCE', 'MODEL'], ['GRAPH', 'ECS'], ['REPRESENT', 'Q / T'], ['LOWER', 'UOP'], ['PLACE', 'CPU/GPU/NPU'], ['KV + PROVE', 'LOSS'], ['CIMAGE', 'READY']];
  stations.forEach((station, index) => {
    const labels = station.querySelectorAll('text:not(.station-index)');
    if (labels.length >= 2) { labels[0].textContent = stationLabels[index][0]; labels[1].textContent = stationLabels[index][1]; }
  });
  const frame = svg.closest('.journey-frame');
  const inspector = document.createElement('aside');
  inspector.className = 'tensor-inspector';
  inspector.setAttribute('aria-live', 'polite');
  inspector.innerHTML = '<div class="tensor-inspector-heading"><span class="tiny-label">TENSOR STATE / SELECTED STAGE</span><strong></strong></div><div class="tensor-inspector-grid"><div><span>FIELD</span><b class="tensor-field-name"></b></div><div><span>VALUE</span><b class="tensor-field-value"></b></div><div><span>HARDWARE MATCH</span><b class="tensor-field-match"></b></div></div><p class="tensor-inspector-explanation"></p><small class="tensor-inspector-disclaimer">Illustrative record. Values describe the compiler contract, not a measured receipt.</small>';
  frame.append(inspector);
  const explorer = document.createElement('div');
  explorer.className = 'tensor-search-explorer';
  explorer.innerHTML = '<div class="search-explorer-toolbar"><label>TARGET DEPLOYMENT <select aria-label="Target deployment"><option value="mi300x">MI300X / ROCm-HIP</option><option value="apple">Apple Silicon / Metal</option><option value="xdna">XDNA2 / spatial plan</option></select></label><span class="search-generation">GENERATION 01 / 04</span></div><div class="candidate-frontier" role="list" aria-label="Evolutionary representation candidates"></div><p class="search-explorer-note"></p>';
  frame.append(explorer);
  const frontier = explorer.querySelector('.candidate-frontier');
  const note = explorer.querySelector('.search-explorer-note');
  const targetSelect = explorer.querySelector('select');
  const renderFrontier = () => {
    frontier.innerHTML = candidates.map((candidate, index) => `<button type="button" class="candidate-chip ${index === Math.min(current, 3) ? 'candidate-selected' : ''}" role="listitem" data-candidate="${index}"><b>${candidate[0]}</b><span>${candidate[1]}</span><em>${candidate[2]}</em></button>`).join('');
    frontier.querySelectorAll('.candidate-chip').forEach(button => button.addEventListener('click', () => {
      frontier.querySelectorAll('.candidate-chip').forEach(item => item.classList.remove('candidate-selected'));
      button.classList.add('candidate-selected');
      note.textContent = `Compiler comparison: ${button.querySelector('b').textContent} evaluated against the BF16 reference for ${targets[target][0]}. The chip describes a search state, not a measured benchmark.`;
    }));
    note.textContent = `Generation ${String(Math.min(current + 1, 4)).padStart(2, '0')} compares representations against ${targets[target][0]} using ${targets[target][1]}.`;
  };
  targetSelect.addEventListener('change', () => { target = targetSelect.value; renderFrontier(); });
  const next = document.createElement('button');
  next.type = 'button'; next.className = 'journey-next'; next.textContent = 'Next compiler state →';
  toolbar.append(next);
  const journeyDescription = svg.querySelector('#journey-svg-desc');
  if (journeyDescription) journeyDescription.textContent = 'A tensor moves through source memory, ECS-native graph operations, progressive representation candidates, lowered kernels, CPU GPU or NPU spatial placement, KV policy, and a CImage.';
  stations.forEach((station, index) => {
    station.setAttribute('role', 'button');
    station.setAttribute('tabindex', '0');
    station.setAttribute('aria-label', stages[index][0]);
  });
  const render = () => {
    const [label, copy] = stages[current];
    stage.textContent = label;
    caption.textContent = copy;
    const state = tensorStates[current];
    inspector.querySelector('.tensor-inspector-heading strong').textContent = state[0];
    inspector.querySelector('.tensor-field-name').textContent = state[1];
    inspector.querySelector('.tensor-field-value').textContent = state[2];
    inspector.querySelector('.tensor-field-match').textContent = state[4];
    inspector.querySelector('.tensor-inspector-explanation').textContent = state[3];
    explorer.querySelector('.search-generation').textContent = `GENERATION ${String(Math.min(current + 1, 4)).padStart(2, '0')} / 04`;
    renderFrontier();
    token.style.transform = `translate(${points[current][0]}px,${points[current][1]}px)`;
    stations.forEach((station, index) => {
      station.classList.toggle('active', index === current);
      station.classList.toggle('visited', index < current);
      station.setAttribute('aria-current', index === current ? 'step' : 'false');
    });
  };
  const advance = () => { current = (current + 1) % stages.length; render(); };
  const start = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      play.textContent = 'Play';
      play.setAttribute('aria-pressed', 'false');
      return;
    }
    clearInterval(timer);
    timer = setInterval(advance, 5200);
    play.textContent = 'Pause';
    play.setAttribute('aria-pressed', 'true');
  };
  play.addEventListener('click', () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
      play.textContent = 'Play';
      play.setAttribute('aria-pressed', 'false');
    } else start();
  });
  reset.addEventListener('click', () => { current = 0; render(); start(); });
  next.addEventListener('click', () => { clearInterval(timer); timer = null; play.textContent = 'Play'; play.setAttribute('aria-pressed', 'false'); advance(); });
  stations.forEach((station, index) => {
    const select = () => { current = index; render(); };
    station.addEventListener('click', select);
    station.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
    });
  });
  render();
  renderFrontier();
  play.textContent = 'Play';
  play.setAttribute('aria-pressed', 'false');
})();
