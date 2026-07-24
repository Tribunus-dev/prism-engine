export const CANONICAL_OBJECT_STAGES = {
  source: 'silhouette',
  representation: 'representation',
  plan: 'physical',
  computeimage: 'identity',
  execution: 'execution',
  receipt: 'evidence',
  fabric: 'fabric',
};

export const CANONICAL_PHASE_STAGES = {
  intent: 'source',
  representation: 'representation',
  plan: 'plan',
  computeimage: 'computeimage',
  execution: 'execution',
  receipt: 'receipt',
  fabric: 'fabric',
};

export const CANONICAL_JOURNEY_STAGES = {
  execution: ['ComputeImage', 'Execution', ['provider capability is compatible']],
  receipt: ['Execution', 'Receipt', ['observation occurred', 'provenance is available']],
  fabric: ['ComputeImage', 'Fabric', ['subject identity persists']],
};

export const CANONICAL_JOURNEY_ORDER = ['execution', 'receipt', 'fabric'];

export const COMPUTEIMAGE_RENDERER_MODES = [
  'silhouette',
  'identity',
  'representation',
  'physical',
  'execution',
  'history',
  'evidence',
  'fabric',
];
