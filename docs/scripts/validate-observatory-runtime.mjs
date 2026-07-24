import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const docsRoot = resolve(root, 'docs');
const report = [];

const ontology = await import(pathToFileURL(resolve(docsRoot, 'js/core/ontology.js')).href);
const transformations = await import(pathToFileURL(resolve(docsRoot, 'js/core/transformations.js')).href);

const claimClassValues = new Set(Object.values(ontology.CLAIM_CLASSES || {}));
const repositoryState = JSON.parse(await readFile(resolve(docsRoot, 'repository-state.json'), 'utf8'));
const claimsGenerated = JSON.parse(await readFile(resolve(docsRoot, 'claims.generated.json'), 'utf8'));

const claims = Array.isArray(repositoryState?.claims)
  ? repositoryState.claims
  : Array.isArray(claimsGenerated?.claims)
    ? claimsGenerated.claims
    : [];

if (!Array.isArray(claims) || claims.length === 0) {
  report.push('claims list is empty; expected repository-backed claims to be available.');
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

const graphSource = await readFile(resolve(docsRoot, 'js/core/observation-graph.js'), 'utf8');

const extractObject = (name) => {
  const match = graphSource.match(new RegExp(`const\\s+${name}\\s*=\\s*\\{([\\s\\S]*?)\\n\\s*};`));
  return match ? `{${match[1]} }` : '';
};

const scenesBlock = extractObject('scenes');
const pageScenesBlock = extractObject('pageScenes');

if (!scenesBlock || !pageScenesBlock) {
  report.push('unable to parse scene/pageScenes blocks from observation-graph.js');
}

const scenes = new Set();
for (const match of scenesBlock.matchAll(/^\s{6}((?:'[^']+'|\"[^\"]+\"|[a-zA-Z0-9_-]+))\s*:\s*\{\s*$/gm)) {
  scenes.add(match[1].replace(/[\"']/g, ''));
}

const pageScenes = new Map();
for (const match of pageScenesBlock.matchAll(/^\s{6}['"]([^'"]+)['"]\s*:\s*['"]([^'"]+)['"]/gm)) {
  pageScenes.set(match[1], match[2]);
}

for (const [page, scene] of pageScenes) {
  if (!scenes.has(scene)) {
    report.push(`pageScene ${page} references unknown scene ${scene}`);
  }
}

if (!scenes.has('origin')) report.push('observation graph missing required origin scene marker');
if (!pageScenes.size || !scenes.size) report.push('observation graph metadata is incomplete');

if (report.length) {
  console.log('Observatory runtime validation FAILED');
  for (const issue of report) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Observatory runtime validation PASSED');
