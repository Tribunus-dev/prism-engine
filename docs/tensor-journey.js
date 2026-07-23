(() => {
  const svg = document.querySelector('.journey-svg');
  if (!svg) return;
  const token = svg.querySelector('.tensor-token');
  const stations = [...svg.querySelectorAll('.journey-station')];
  const stage = document.querySelector('#journey-stage');
  const caption = document.querySelector('#journey-caption');
  const play = document.querySelector('#journey-play');
  const reset = document.querySelector('#journey-reset');
  const stages = [
    ['01 / INGEST', 'Read the source tensor and preserve its identity in the ECS world.'],
    ['02 / GRAPH', 'Attach the tensor to semantic operators, shapes, and dependencies.'],
    ['03 / REPRESENT', 'Search progressive quantization, ternarization, and mixed-precision candidates.'],
    ['04 / LOWER', 'Turn the selected graph region into explicit uops and kernel work.'],
    ['05 / PLACE', 'Choose CPU, GPU, or NPU lanes, spatial regions, memory, and queues.'],
    ['06 / KV + VERIFY', 'Search KV-cache policy, check legality, quality, resources, and replay evidence.'],
    ['07 / CIMAGE', 'Seal the tensor view into the target-aware executable CImage.']
  ];
  let current = 0;
  let timer = null;
  const points = [[110, 235], [250, 100], [410, 370], [585, 100], [750, 370], [915, 100], [1090, 235]];
  const stationLabels = [['SOURCE', 'MODEL'], ['GRAPH', 'ECS'], ['REPRESENT', 'Q / T'], ['LOWER', 'UOP'], ['PLACE', 'CPU/GPU/NPU'], ['KV + PROVE', 'LOSS'], ['CIMAGE', 'READY']];
  stations.forEach((station, index) => {
    const labels = station.querySelectorAll('text:not(.station-index)');
    if (labels.length >= 2) { labels[0].textContent = stationLabels[index][0]; labels[1].textContent = stationLabels[index][1]; }
  });
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
    timer = setInterval(advance, 1800);
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
  stations.forEach((station, index) => {
    const select = () => { current = index; render(); };
    station.addEventListener('click', select);
    station.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
    });
  });
  render();
  start();
})();
