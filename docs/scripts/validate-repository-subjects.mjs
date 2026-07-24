import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = process.cwd();
const repository = JSON.parse(await readFile(resolve(root, 'docs/repository-state.json'), 'utf8'));
const docs = {
  runtime: await readFile(resolve(root, 'docs/prism-runtime.md'), 'utf8'),
  semantics: await readFile(resolve(root, 'docs/prism-semantics.md'), 'utf8'),
  meaning: await readFile(resolve(root, 'docs/prism-meaning-runtime.md'), 'utf8'),
  interaction: await readFile(resolve(root, 'docs/prism-interaction-runtime.md'), 'utf8'),
};

const issues = [];
const canonicalSubjectIds = new Set();
const claims = Array.isArray(repository.claims) ? repository.claims : [];
for (const claim of claims) {
  if (!claim?.subjectId) {
    issues.push(`claim ${claim.id || '<unknown>'} has no subjectId`);
  } else {
    canonicalSubjectIds.add(claim.subjectId);
  }
}
if (canonicalSubjectIds.size > 1) {
  issues.push(`repository-state has multiple subjectIds: ${[...canonicalSubjectIds].join(', ')}`);
}

const expectedStatus = 'Status: frozen for this sprint';
if (!docs.runtime.includes(expectedStatus)) {
  issues.push(`prism-runtime.md missing "${expectedStatus}"`);
}
if (!docs.semantics.includes(expectedStatus)) {
  issues.push(`prism-semantics.md missing "${expectedStatus}"`);
}
if (!docs.meaning.includes(expectedStatus)) {
  issues.push(`prism-meaning-runtime.md missing "${expectedStatus}"`);
}
if (!docs.interaction.includes(expectedStatus)) {
  issues.push(`prism-interaction-runtime.md missing "${expectedStatus}"`);
}

if (!docs.runtime.includes('[`prism-semantics.md`]') && !docs.runtime.includes('prism-semantics.md')) {
  issues.push('prism-runtime.md should cross-link prism-semantics.md');
}
if (!docs.semantics.includes('`prism-runtime.md`') && !docs.semantics.includes('prism-runtime.md')) {
  issues.push('prism-semantics.md should reference prism-runtime.md');
}

if (!docs.meaning.includes('compatibility naming layer')) {
  issues.push('prism-meaning-runtime.md should be marked as a compatibility naming layer');
}
if (!docs.interaction.includes('compatibility naming layer')) {
  issues.push('prism-interaction-runtime.md should be marked as a compatibility naming layer');
}

if (issues.length) {
  console.log('Repository subject/canonicalization validation FAILED');
  for (const issue of issues) {
    console.log(`- ${issue}`);
  }
  process.exit(1);
}

console.log('Repository subject/canonicalization validation PASSED');
