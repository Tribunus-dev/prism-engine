import { pathToFileURL } from 'node:url';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chapterMap } from '../js/systems/navigation.js';

const root = process.cwd();
const docsRoot = resolve(root, 'docs');
const report = [];

const ontology = await import(pathToFileURL(resolve(docsRoot, 'js/core/ontology.js')).href);
const transformations = await import(pathToFileURL(resolve(docsRoot, 'js/core/transformations.js')).href);
const projections = await import(pathToFileURL(resolve(docsRoot, 'js/core/observation-projections.js')).href);
const observationGraph = await import(pathToFileURL(resolve(docsRoot, 'js/core/observation-graph.js')).href);
const canonicalContract = await import(pathToFileURL(resolve(docsRoot, 'js/core/canonical-contract.js')).href);

const claimClassValues = new Set(Object.values(ontology.CLAIM_CLASSES || {}));
const knowledgeStateValues = new Set(Object.values(ontology.KNOWLEDGE_STATES || {}));
const repositoryState = JSON.parse(await readFile(resolve(docsRoot, 'repository-state.json'), 'utf8'));
const capabilities = Array.isArray(repositoryState?.capabilities) ? repositoryState.capabilities : [];
const claims = Array.isArray(repositoryState?.claims) ? repositoryState.claims : [];

if (!Array.isArray(repositoryState?.capabilities) || repositoryState.capabilities.length === 0) {
  report.push('repository-state.json capabilities is empty or missing; canonical capability truth is missing.');
}
if (!Array.isArray(repositoryState?.claims) || repositoryState.claims.length === 0) {
  report.push('repository-state.json claims is empty or missing; canonical claim truth is missing.');
}

for (const capability of capabilities) {
  if (!capability?.id) {
    report.push(`capability missing id: ${JSON.stringify(capability)}`);
  }
}

for (const claim of claims) {
  if (!claim?.id) {
    report.push(`claim missing id: ${JSON.stringify(claim)}`);
    continue;
  }
  if (claim.class && !claimClassValues.has(claim.class)) {
    report.push(`claim ${claim.id} has invalid class ${claim.class}`);
  }
  if (claim.class === ontology.CLAIM_CLASSES?.MEASURED) {
    if (!Array.isArray(claim.sourceRefs) || claim.sourceRefs.length === 0) {
      report.push(`measured claim ${claim.id} missing sourceRefs`);
    }
    if (!claim.constraints) {
      report.push(`measured claim ${claim.id} missing constraints`);
    }
  }
}

for (const transformation of Object.values(transformations.TRANSFORMATIONS || {})) {
  const errors = transformations.validateTransformations({ [transformation.id]: transformation }) || [];
  if (!transformation?.id) report.push('transformation entry missing id');
  if (!transformation?.from || !transformation?.to) {
    report.push(`transformation ${transformation?.id || '<unknown>'} missing endpoint(s)`);
  }
  if (!Array.isArray(transformation?.preconditions) || transformation.preconditions.length === 0) {
    report.push(`transformation ${transformation?.id || '<unknown>'} missing preconditions`);
  }
  if (!Array.isArray(transformation?.postconditions) || transformation.postconditions.length === 0) {
    report.push(`transformation ${transformation?.id || '<unknown>'} missing postconditions`);
  }
  if (!Array.isArray(transformation?.sourceRefs) || transformation.sourceRefs.length === 0) {
    report.push(`transformation ${transformation?.id || '<unknown>'} missing sourceRefs`);
  }
  if (!transformation?.claimClass || !claimClassValues.has(transformation.claimClass)) {
    report.push(`transformation ${transformation?.id || '<unknown>'} missing/invalid claimClass`);
  }
  if (errors.length) report.push(...errors);
}

for (const route of Object.values(projections.PROJECTIONS || {})) {
  for (const claim of route?.claims || []) {
    if (!claim?.id || !claim?.class) {
      report.push(`projection ${route.route || '<unknown>'} has invalid claim metadata`);
      continue;
    }
    if (!claimClassValues.has(claim.class)) {
      report.push(`projection ${route.route || '<unknown>'} claim ${claim.id} has invalid class ${claim.class}`);
    }
  }
}

for (const [sceneName, scene] of Object.entries(observationGraph.SCENES || {})) {
  if (!scene?.phase) {
    report.push(`scene ${sceneName} missing phase`);
  } else if (!Object.values(canonicalContract.CANONICAL_PHASE_STAGES || {}).includes(scene.phase)) {
    report.push(`scene ${sceneName} has phase not in canonical phase map: ${scene.phase}`);
  }
  if (scene?.claim && !claimClassValues.has(scene.claim)) {
    report.push(`scene ${sceneName} uses non-ontology claim ${scene.claim}`);
  }
  if (scene?.knowledge && !knowledgeStateValues.has(scene.knowledge)) {
    report.push(`scene ${sceneName} uses non-ontology knowledge state ${scene.knowledge}`);
  }
  if (scene?.next && !Object.prototype.hasOwnProperty.call(observationGraph.SCENES || {}, scene.next)) {
    report.push(`scene ${sceneName} transitions to unknown scene ${scene.next}`);
  }
}

const scenes = new Set(Object.keys(observationGraph.SCENES || {}));
const pageScenes = new Map(Object.entries(observationGraph.PAGE_SCENES || {}));
const projectedRoutes = new Set(Object.values(projections.PROJECTIONS || {}).map(route => route.route));
const navigationRoutes = new Set((chapterMap || []).map(([_, route]) => String(route || '').split('#')[0]).filter(Boolean));
for (const route of navigationRoutes) {
  if (!projectedRoutes.has(route)) {
    report.push(`navigation route ${route} is not declared in PROJECTIONS`);
  }
}
for (const route of projectedRoutes) {
  if (!navigationRoutes.has(route)) {
    report.push(`projection route ${route} is not represented in chapterMap`);
  }
}

for (const [page, scene] of pageScenes) {
  if (!scenes.has(scene)) {
    report.push(`pageScene ${page} references unknown scene ${scene}`);
  }
}
const canonicalObjectSource = await readFile(resolve(docsRoot, 'js/systems/canonical-object.js'), 'utf8');
const canonicalStages = new Set();
for (const match of canonicalObjectSource.matchAll(/([a-zA-Z0-9-]+):\s*'([^']+)'/g)) {
  const value = match[2];
  if (value) canonicalStages.add(value);
}
for (const stage of Object.values(canonicalContract.CANONICAL_JOURNEY_STAGES || {})) {
  if (Array.isArray(stage) && stage[0] && stage[1]) {
    // valid journey definition; no action needed
  } else {
    report.push(`canonical journey stage token malformed: ${JSON.stringify(stage)}`);
  }
}
for (const [stageName] of Object.entries(canonicalContract.CANONICAL_JOURNEY_STAGES || {})) {
  canonicalStages.add(stageName);
}
const requiredStages = canonicalContract.CANONICAL_JOURNEY_ORDER || ['execution', 'receipt', 'fabric'];
for (const required of requiredStages) {
  if (!canonicalStages.has(required)) {
    report.push(`canonical journey runtime configuration missing required stage: ${required}`);
  }
}
for (const stage of canonicalStages) {
  if (!/^[a-z-]+$/.test(stage)) {
    report.push(`unexpected canonical journey stage token: ${stage}`);
  }
}

if (!scenes.has('compute-image') || !scenes.has('scheduler') || !scenes.has('fabric')) {
  report.push('observation-graph scenes are missing canonical journey milestones');
}
if (!scenes.has('origin')) report.push('observation graph missing required origin scene marker');
if (!pageScenes.size || !scenes.size) report.push('observation graph metadata is incomplete');

for (const route of Object.values(projections.PROJECTIONS || {})) {
  if (!route?.route) continue;
  if (!pageScenes.has(route.route)) {
    report.push(`pageScenes missing route ${route.route}`);
  }
}

if (projectionsPrimaryRouteCount(Object.values(projections.PROJECTIONS || {})) === -1) {
  report.push('projection ordering is missing stable route priority values');
}
for (const [id, projection] of Object.entries(projections.PROJECTIONS || {})) {
  if (!projection?.observation) {
    report.push(`projection ${id} missing observation`);
  }
  if (projection?.claims?.length === 0) {
    report.push(`projection ${projection.route || id} has no claims`);
  }
}
for (const [route, scene] of pageScenes) {
  const projection = Object.values(projections.PROJECTIONS || {}).find(item => item.route === route);
  if (!projection) continue;
  if (!observationGraph.PAGE_SCENES || !Object.prototype.hasOwnProperty.call(observationGraph.PAGE_SCENES, route)) {
    report.push(`pageScenes missing projected route ${route}`);
  }
  if (!scene) {
    report.push(`page ${route} resolves to unknown scene`);
  } else if (!observationGraph.SCENES || !Object.prototype.hasOwnProperty.call(observationGraph.SCENES, scene)) {
    report.push(`page ${route} maps to unknown scene ${scene}`);
  }
}

const canonicalRepositoryClaimIds = new Set((claims || []).map(claim => claim?.id).filter(Boolean));
for (const route of Object.values(projections.PROJECTIONS || {})) {
  for (const claim of route.claims || []) {
    if (claim?.id && !canonicalRepositoryClaimIds.has(claim.id)) {
      report.push(`projection ${route.route || '<unknown>'} references unknown claim id ${claim.id}`);
    }
  }
}

function projectionsPrimaryRouteCount(allProjections = []) {
  const positions = allProjections
    .map(projection => projection?.position)
    .filter((position) => Number.isFinite(position))
    .map((position) => Number(position));
  if (positions.some(pos => pos <= 0)) return -1;
  const sorted = [...positions].sort((a, b) => a - b);
  const unique = new Set(positions);
  if (unique.size !== positions.length) return -1;
  for (let index = 0; index < sorted.length; index += 1) {
    if (sorted[index] !== index + 1) return -1;
  }
  return positions.length;
}

if (report.length) {
  console.log('Observatory runtime validation FAILED');
  for (const issue of report) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Observatory runtime validation PASSED');
