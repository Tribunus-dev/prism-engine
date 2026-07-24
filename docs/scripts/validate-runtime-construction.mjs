import { readFile, readdir } from 'node:fs/promises';
import { resolve, relative, extname } from 'node:path';

const root = process.cwd();
const allowedRuntimeContextFile = resolve(root, 'docs/js/runtime/runtime-context.js');
const jsRoot = resolve(root, 'docs/js');
const issues = [];

const walk = async (dir, files = []) => {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const path = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.startsWith('.')) continue;
      await walk(path, files);
    } else if (['.js', '.mjs'].includes(extname(entry.name))) {
      files.push(path);
    }
  }
  return files;
};

const jsFiles = await walk(jsRoot);

for (const file of jsFiles) {
  const content = await readFile(file, 'utf8');
  if (file === allowedRuntimeContextFile) continue;
  if (/import\s+\{?\s*runtimeContext\s*\}?\s+from\s+['"].+runtime-context\.js['"]/.test(content)) {
    issues.push(`runtimeContext import remains in ${relative(root, file)}`);
  }
  if (/\bruntimeContext\s*\(/.test(content)) {
    issues.push(`runtimeContext accessor used in ${relative(root, file)}`);
  }
}

if (issues.length) {
  console.log('Runtime construction validation FAILED');
  for (const issue of issues) console.log(`- ${issue}`);
  process.exit(1);
}

console.log('Runtime construction validation PASSED');
