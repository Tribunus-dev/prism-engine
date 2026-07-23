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
    ['03 / REPRESENT', 'Walk upward from ternary through INT4/NF4 and INT8 until the first effective BF16 match is admitted.'],
    ['04 / LOWER', 'Turn the selected graph region into explicit uops and kernel work.'],
    ['05 / PLACE', 'Choose CPU, GPU, or NPU lanes, spatial regions, memory, and queues.'],
    ['06 / KV + VERIFY', 'Canary-check ternary fitness, descend quantization levels when it fails, then validate KV policy.'],
    ['07 / CIMAGE', 'Seal the tensor view into the target-aware executable CImage.']
  ];
  const tensorStates = [
    ['Logical identity', 'attention.q_proj.weight', 'shape [4096, 4096] · BF16 · projection weight', 'Source bytes are named and attributed before any physical choice.', 'SOURCE / MODEL'],
    ['Graph semantics', 'attention.q_proj.weight', 'role projection · axes [out, in] · dependency attention.q', 'The tensor is attached to ECS graph edges and shape contracts.', 'ECS / GRAPH'],
    ['Representation frontier', 'attention.q_proj.weight', 'ternary → INT4/NF4 → INT8 → BF16 · first effective match', 'The search begins at the lowest-bit candidate and walks upward until the canary accepts the lowest representation that effectively matches BF16.', 'SEARCH / UPWARD ADMISSION'],
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
    ['Ternary candidate', 'lowest-bit first', 'canary gate'],
    ['INT4 / NF4', 'next upward level', 'if ternary fails'],
    ['INT8', 'next upward level', 'if INT4 fails'],
    ['BF16 canary', 'effective-match reference', 'admission bound']
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
  explorer.innerHTML = '<div class="search-explorer-toolbar"><label>TARGET DEPLOYMENT <select aria-label="Target deployment"><option value="mi300x">MI300X / ROCm-HIP</option><option value="apple">Apple Silicon / Metal</option><option value="xdna">XDNA2 / spatial plan</option></select></label><span class="search-generation">GENERATION 01 / 04</span></div><div class="compiler-graph" role="img" aria-label="Compiler graph walking upward from ternary through higher bit representations until the first effective BF16 match"><div class="graph-node graph-source"><b>BF16 CANARY</b><span>source reference</span></div><div class="graph-edge edge-source"></div><div class="graph-candidates"><div class="graph-node graph-candidate" data-candidate="0"><b>TERNARY</b><span>lowest-bit first</span></div><div class="graph-node graph-candidate" data-candidate="1"><b>INT4 / NF4</b><span>walk upward</span></div><div class="graph-node graph-candidate" data-candidate="2"><b>INT8</b><span>walk upward</span></div><div class="graph-node graph-candidate" data-candidate="3"><b>BF16</b><span>effective match</span></div></div><div class="graph-edge edge-target"></div><div class="graph-node graph-target"><b class="graph-target-name">MI300X</b><span class="graph-target-detail">ROCm/HIP · tiles</span></div><div class="graph-edge edge-proof"></div><div class="graph-node graph-proof"><b>CANARY PROOF</b><span>first effective match</span></div></div><div class="candidate-frontier" role="list" aria-label="Progressive quantization candidates"></div><p class="search-explorer-note"></p>';
  frame.append(explorer);
  const frontier = explorer.querySelector('.candidate-frontier');
  const note = explorer.querySelector('.search-explorer-note');
  const targetSelect = explorer.querySelector('select');
  const graphTargetName = explorer.querySelector('.graph-target-name');
  const graphTargetDetail = explorer.querySelector('.graph-target-detail');
  const renderFrontier = () => {
    frontier.innerHTML = candidates.map((candidate, index) => `<button type="button" class="candidate-chip ${index === Math.min(current, 3) ? 'candidate-selected' : ''}" role="listitem" data-candidate="${index}"><b>${candidate[0]}</b><span>${candidate[1]}</span><em>${candidate[2]}</em></button>`).join('');
    frontier.querySelectorAll('.candidate-chip').forEach(button => button.addEventListener('click', () => {
      frontier.querySelectorAll('.candidate-chip').forEach(item => item.classList.remove('candidate-selected'));
      button.classList.add('candidate-selected');
      explorer.querySelectorAll('.graph-candidate').forEach(node => node.classList.toggle('graph-active', node.dataset.candidate === button.dataset.candidate));
      note.textContent = `Compiler comparison: ${button.querySelector('b').textContent} evaluated against the BF16 reference for ${targets[target][0]}. The chip describes a search state, not a measured benchmark.`;
    }));
    explorer.querySelectorAll('.graph-candidate').forEach(node => node.classList.toggle('graph-active', Number(node.dataset.candidate) === Math.min(current, 3)));
    graphTargetName.textContent = targets[target][0].split(' / ')[0];
    graphTargetDetail.textContent = targets[target][1];
    note.textContent = `Generation ${String(Math.min(current + 1, 4)).padStart(2, '0')} walks upward from the lowest-bit candidate against the BF16 canary for ${targets[target][0]}; mixed precision keeps each tensor at its admitted level.`;
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
