import { readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const docsRoot = new URL('.', import.meta.url).pathname.replace(/\/$/, '');
const read = async relative => readFile(new URL(relative, `file://${docsRoot}/`), 'utf8');

const checks = [];

const runtimeSource = await read('../js/runtime/create-runtime.js');
const canonicalObjectSource = await read('../js/systems/canonical-object.js');
const computeImageSource = await read('../js/renderers/computeimage.js');
const observationGraphSource = await read('../js/core/observation-graph.js');
const stateProjectionSource = await read('../js/systems/state-projection.js');
const navigationSystemSource = await read('../js/systems/navigation.js');
const siteShellSource = await read('../js/site-shell.js');
const meaningRuntimeSource = await read('../prism-meaning-runtime.md');
const interactionRuntimeSource = await read('../prism-interaction-runtime.md');

if (!runtimeSource.includes('getCanonicalSubject:')) {
  checks.push('create-runtime.js is missing getCanonicalSubject');
}

if (!runtimeSource.includes('applyRepositorySnapshot(')) {
  checks.push('create-runtime.js is missing shared repository snapshot normalizer');
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

if (!/modes\s*=\s*\{[^}]*representation:\s*['\"]representation['\"]/ .test(canonicalObjectSource)) {
  checks.push('canonical-object.js is not mapping representation to representation mode');
}

if (!computeImageSource.includes('context?.runtime?.getCanonicalSubject')) {
  checks.push('computeimage.js is not consuming runtime.getCanonicalSubject');
}
if (computeImageSource.includes('kernel?.subject?.computeImage') || computeImageSource.includes('kernel?.subject?.')) {
  checks.push('computeimage.js must not fall back to kernel-local subject construction');
}

if (!observationGraphSource.includes('context?.runtime?.getCanonicalSubject?.()')) {
  checks.push('observation-graph.js is not consuming canonical runtime subject');
}
if (observationGraphSource.includes('kernel?.subject?.computeImage') || observationGraphSource.includes('kernel?.ensureComputeImageSubject')) {
  checks.push('observation-graph.js must not fall back to kernel-local subject construction');
}

if (!stateProjectionSource.includes('runtime?.getCanonicalSubject?.()')) {
  checks.push('state-projection.js is not reading canonical subject through runtime state');
}
if (stateProjectionSource.includes('kernel?.subject?.computeImage') || stateProjectionSource.includes('kernel?.subject?.')) {
  checks.push('state-projection.js must not use kernel-local subject construction');
}

if (canonicalObjectSource.includes('kernel?.ensureComputeImageSubject') || canonicalObjectSource.includes('kernel?.subject?.computeImage')) {
  checks.push('canonical-object.js must not fallback to kernel-local subject construction');
}

if (!computeImageSource.includes("'representation'") && !computeImageSource.includes('"representation"')) {
  checks.push('computeimage.js does not expose representation mode in renderer modes list');
}

if (navigationSystemSource.includes('|| context?.route')) {
  checks.push('navigation.js still has context.route fallback and should use canonical runtime projection only');
}
if (observationGraphSource.includes('|| context?.route')) {
  checks.push('observation-graph.js still has route fallback and should rely on runtime projection only');
}
if (navigationSystemSource.includes('context?.runtime?.currentRoute')) {
  checks.push('navigation.js should use runtime.getCurrentRoute and not context?.runtime?.currentRoute');
}
if (observationGraphSource.includes('context?.runtime?.currentRoute')) {
  checks.push('observation-graph.js should use runtime.getCurrentRoute and not context?.runtime?.currentRoute');
}
if (observationGraphSource.includes('context?.runtime?.getCurrentRoute?.()') === false) {
  checks.push('observation-graph.js should use runtime.getCurrentRoute for route resolution');
}
if (navigationSystemSource.includes('context?.runtime?.getCurrentRoute?.()') === false) {
  checks.push('navigation.js should use runtime.getCurrentRoute for route resolution');
}

if (siteShellSource.includes('currentChapter()')) {
  checks.push('site-shell.js should pass canonical route context into currentChapter calls');
}
if (/^##\s+Belief state|^##\s+Conservation laws|^##\s+Continuity|^##\s+Living objects|^##\s+Self-description/m.test(meaningRuntimeSource)) {
  checks.push('prism-meaning-runtime.md appears to re-define semantic sections that should live in prism-semantics.md');
}
if (/^##\s+Kernel observations|^##\s+Observation entity|^##\s+Observer modes|^##\s+Interaction events|^##\s+Visual state machine/m.test(interactionRuntimeSource)) {
  checks.push('prism-interaction-runtime.md appears to re-define canonical interaction sections that should live in prism-runtime.md');
}

if (!/##\s+Canonical source|Canonical source/i.test(meaningRuntimeSource)) {
  checks.push('prism-meaning-runtime.md should include a canonical source section');
}
if (!/##\s+Compatibility guidance|Compatibility guidance/i.test(meaningRuntimeSource)) {
  checks.push('prism-meaning-runtime.md should include compatibility guidance');
}
if (!/##\s+Canonical source|Canonical source/i.test(interactionRuntimeSource)) {
  checks.push('prism-interaction-runtime.md should include a canonical source section');
}
if (!/##\s+Compatibility guidance|Compatibility guidance/i.test(interactionRuntimeSource)) {
  checks.push('prism-interaction-runtime.md should include compatibility guidance');
}

const root = new URL('../../', `file://${docsRoot}/`).pathname;
const filePaths = [
  'js/systems/navigation.js',
  'js/core/observation-graph.js',
  'js/systems/canonical-object.js',
  'js/renderers/computeimage.js',
  'js/systems/state-projection.js',
  'js/renderers/canonical-journey.js',
  'js/systems/canonical-stage.js',
  'js/systems/accessibility.js',
  'js/site-shell.js',
];

for (const file of filePaths) {
  const source = await read(`../${file}`);
  const lower = source;
  const hasSubjectFallback = /context\?\.runtime\?\.subject|runtime\?\.state\?\.subject|kernel\?\.subject\?\.computeImage|kernel\?\.subject\?\./g.test(lower);
  const hasLegacyRoute = /context\?\.runtime\?\.currentRoute/g.test(lower);
  const staleCanonicalFallback = /getSubject\(\)\s*\|\|\s*computation/g.test(lower);
  if (hasSubjectFallback && file !== 'js/observatory-kernel.js') {
    checks.push(`${file} contains legacy kernel subject fallback; use runtime.getCanonicalSubject/stateSubject instead`);
  }
  if (staleCanonicalFallback && file !== 'js/runtime/create-runtime.js') {
    checks.push(`${file} contains stale canonical subject fallback; always use current getCanonicalSubject result`);
  }
  if (/(runtime\?\.stateSubject|stateSubject\s*\|\|)/g.test(lower) && file !== 'js/runtime/create-runtime.js') {
    checks.push(`${file} contains stateSubject fallback; consume runtime.getCanonicalSubject directly`);
  }
  if (hasLegacyRoute && file !== 'js/runtime/create-runtime.js') {
    checks.push(`${file} contains legacy runtime.currentRoute access`);
  }
}

if (checks.length) {
  console.log('Canonical subject validation FAILED');
  for (const issue of checks) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Canonical subject validation PASSED');
