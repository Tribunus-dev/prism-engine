import { readFile } from 'node:fs/promises';

const docsRoot = new URL('.', import.meta.url).pathname.replace(/\/$/, '');
const read = async relative => readFile(new URL(relative, `file://${docsRoot}/`), 'utf8');

const checks = [];

const runtimeSource = await read('../js/runtime/create-runtime.js');
const canonicalObjectSource = await read('../js/systems/canonical-object.js');
const computeImageSource = await read('../js/renderers/computeimage.js');
const observationGraphSource = await read('../js/core/observation-graph.js');
const stateProjectionSource = await read('../js/systems/state-projection.js');

if (!runtimeSource.includes('getCanonicalSubject:')) {
  checks.push('create-runtime.js is missing getCanonicalSubject');
}

if (!runtimeSource.includes('subjectFromRepository')) {
  checks.push('create-runtime.js is missing subjectFromRepository');
}

if (!runtimeSource.includes('runtime.stateSubject')) {
  checks.push('create-runtime.js is missing runtime.stateSubject updates');
}

if (!canonicalObjectSource.includes('context?.runtime?.getCanonicalSubject?.()')) {
  checks.push('canonical-object.js is not consuming runtime.getCanonicalSubject');
}

if (!/modes\s*=\s*\{[^}]*representation:\s*[\'\"]representation[\'\"]/.test(canonicalObjectSource)) {
  checks.push('canonical-object.js does not map representation to representation mode');
}

if (!computeImageSource.includes('context?.runtime?.getCanonicalSubject')) {
  checks.push('computeimage.js is not consuming runtime.getCanonicalSubject');
}
if (computeImageSource.includes('kernel?.subject?.computeImage') || computeImageSource.includes('kernel?.subject?.')) {
  checks.push('computeimage.js must not fall back to kernel-local subject construction');
}

if (!observationGraphSource.includes('context?.runtime?.getCanonicalSubject') && !observationGraphSource.includes('context?.runtime?.stateSubject')) {
  checks.push('observation-graph.js is not consuming canonical runtime subject');
}
if (observationGraphSource.includes('kernel?.subject?.computeImage') || observationGraphSource.includes('kernel?.ensureComputeImageSubject')) {
  checks.push('observation-graph.js must not fall back to kernel-local subject construction');
}

if (!stateProjectionSource.includes('runtime?.getCanonicalSubject') && !stateProjectionSource.includes('runtime?.stateSubject')) {
  checks.push('state-projection.js is not reading canonical subject through runtime state');
}
if (stateProjectionSource.includes('kernel?.subject?.computeImage') || stateProjectionSource.includes('kernel?.subject?.')) {
  checks.push('state-projection.js must not fall back to kernel-local subject construction');
}

if (canonicalObjectSource.includes('kernel?.ensureComputeImageSubject') || canonicalObjectSource.includes('kernel?.subject?.computeImage')) {
  checks.push('canonical-object.js must not fallback to kernel-local subject construction');
}

if (!computeImageSource.includes("'representation'") && !computeImageSource.includes('"representation"')) {
  checks.push('computeimage.js does not expose representation mode in renderer modes list');
}

if (checks.length) {
  console.log('Canonical subject validation FAILED');
  for (const issue of checks) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Canonical subject validation PASSED');
