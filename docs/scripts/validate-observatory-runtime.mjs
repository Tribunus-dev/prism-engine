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

const claimClassValues = new Set(Object.values(ontology.CLAIM_CLASSES || {}));
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

if (report.length) {
  console.log('Observatory runtime validation FAILED');
  for (const issue of report) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Observatory runtime validation PASSED');
