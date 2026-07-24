import { readFile } from 'node:fs/promises';

const docsRoot = new URL('.', import.meta.url).pathname.replace(/\/$/, '');
const read = async relative => readFile(new URL(relative, `file://${docsRoot}/`), 'utf8');

const checks = [];

const runtimeSource = await read('../js/runtime/create-runtime.js');
const canonicalObjectSource = await read('../js/systems/canonical-object.js');
const computeImageSource = await read('../js/renderers/computeimage.js');

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

if (!computeImageSource.includes('context?.runtime?.getCanonicalSubject')) {
  checks.push('computeimage.js is not consuming runtime.getCanonicalSubject');
}

if (!computeImageSource.includes('kernel?.ensureComputeImageSubject?.()') && !computeImageSource.includes('kernel?.subject?.computeImage')) {
  checks.push('computeimage.js lost direct ComputeImage fallback coverage; keep at least one safe fallback');
}

if (checks.length) {
  console.log('Canonical subject validation FAILED');
  for (const issue of checks) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Canonical subject validation PASSED');
