import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = process.cwd();
const docs = [
  {
    path: 'docs/prism-runtime.md',
    mustContain: [
      'Status: frozen for this sprint',
      'Canonical mappings',
    ],
  },
  {
    path: 'docs/prism-meaning-runtime.md',
    mustContain: [
      'Status: frozen for this sprint',
      'archived as a compatibility naming layer only',
      '`prism-runtime.md`',
      '`prism-semantics.md`',
      'Do not introduce new Meaning Runtime abstractions',
    ],
  },
  {
    path: 'docs/prism-interaction-runtime.md',
    mustContain: [
      'Status: frozen for this sprint',
      'archived as a compatibility naming layer',
      '`prism-runtime.md`',
      '`prism-experience-architecture.md`',
      'Do not add new interaction runtime layers',
    ],
  },
  {
    path: 'docs/prism-semantics.md',
    mustContain: [
      'Status: frozen for this sprint',
      'This document is the canonical source for semantic',
      'Do not split semantics into separate runtime layers',
    ],
  },
];

const issues = [];

for (const entry of docs) {
  const content = await readFile(resolve(root, entry.path), 'utf8');
  for (const needle of entry.mustContain) {
    if (!content.includes(needle)) {
      issues.push(`${entry.path} missing required freeze contract: ${needle}`);
    }
  }
}

if (issues.length) {
  console.log('Doc freeze validation FAILED');
  for (const issue of issues) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Doc freeze validation PASSED');
