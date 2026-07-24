import { readFile, access } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { resolve } from 'node:path';
import { glob } from 'node:fs/promises';

const docsRoot = new URL('..', import.meta.url).pathname;
const docsPath = new URL('.', import.meta.url).pathname;

const parseCssFile = async (path) => {
  const raw = await readFile(path, 'utf8');
  const lines = raw.split(/\n/);
  const issues = [];
  let inBlockComment = false;
  let seenNonImport = false;
  lines.forEach((line, index) => {
    const normalized = line.trim();
    if (!normalized) return;
    let value = normalized;
    if (inBlockComment) {
      if (value.includes('*/')) {
        value = value.slice(value.indexOf('*/') + 2).trim();
        inBlockComment = false;
      } else {
        return;
      }
    }
    if (!inBlockComment && value.startsWith('/*')) {
      if (!value.includes('*/')) {
        inBlockComment = true;
        return;
      }
      const close = value.indexOf('*/');
      value = value.slice(close + 2).trim();
      if (!value) return;
    }
    if (value.startsWith('//')) return;
    if (/^@charset\b/i.test(value)) return;
    if (/^@import\b/i.test(value)) {
      if (seenNonImport) {
        issues.push(`Invalid @import ordering in ${path}: line ${index + 1}`);
      }
      return;
    }
    seenNonImport = true;
  });
  return issues;
};

const validateCssImports = async () => {
  const files = [];
  for await (const file of glob('*.css', { cwd: docsPath })) {
    files.push(file);
  }
  const issues = [];
  for (const file of files) {
    issues.push(...await parseCssFile(resolve(docsPath, file)));
  }
  if (issues.length) {
    console.log('CSS import order validation FAILED');
    issues.forEach(issue => console.log(`- ${issue}`));
    process.exit(1);
  }
  console.log('CSS import order validation PASSED');
};

const validateHtmlStyleLinks = async () => {
  const htmls = [];
  for await (const file of glob('*.html', { cwd: docsPath })) {
    htmls.push(file);
  }
  const issues = [];
  for (const file of htmls) {
    const content = await readFile(resolve(docsPath, file), 'utf8');
    const hrefs = [...content.matchAll(/<link\b[^>]*\brel=['\"]stylesheet['\"][^>]*\bhref=['\"]([^'\"]+)['\"]/g)]
      .map(([, href]) => href);
    for (const href of hrefs) {
      if (href.startsWith('http') || href.startsWith('//')) continue;
      const resolved = resolve(docsPath, href.split('?')[0].split('#')[0]);
      try {
        await access(resolved, fsConstants.F_OK);
      } catch {
        issues.push(`Missing stylesheet in ${file}: ${href}`);
      }
    }
  }
  if (issues.length) {
    console.log('CSS style link validation FAILED');
    issues.forEach(issue => console.log(`- ${issue}`));
    process.exit(1);
  }
  console.log('CSS style link validation PASSED');
};

await validateCssImports();
await validateHtmlStyleLinks();
