/**
 * Scan test files and mark APIs as tested in _registry.json.
 * Usage: node tests/compatibility/mark_tested.mjs
 */

import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const registryPath = join(__dirname, '_registry.json');
const registry = JSON.parse(readFileSync(registryPath, 'utf8'));

// Scan all test files
const testDir = __dirname;
const allFiles = [];

function scanDir(dir) {
  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory() && !entry.name.startsWith('__')) {
      scanDir(fullPath);
    } else if (entry.name.endsWith('.test.ts')) {
      allFiles.push(fullPath);
    }
  }
}

scanDir(testDir);

// For each API in registry, check if any test file references it
for (const [apiName, apiInfo] of Object.entries(registry)) {
  // Extract the function/class name (e.g., 'torch.zeros' -> 'zeros', 'Tensor.prototype.add' -> 'add')
  const parts = apiName.split('.');
  const shortName = parts[parts.length - 1];

  // Also check for alternative names
  const altNames = [
    apiName,                  // torch.zeros
    shortName,                // zeros
    parts.slice(1).join('.'), // zeros (for torch.zeros)
  ];

  for (const file of allFiles) {
    const content = readFileSync(file, 'utf8');
    for (const name of altNames) {
      // Check if the API is referenced in test (describe('torch.zeros') or torch.zeros(...) or tensor.add(...))
      if (content.includes(`'${name}`) || content.includes(`.${name}(`) || content.includes(`.${name}`)) {
        // Get relative path from tests/compatibility/
        const relPath = file.replace(__dirname, '').replace(/\\/g, '/').replace(/^\//, '');
        registry[apiName].tested = true;
        registry[apiName].testFile = relPath;
        break;
      }
    }
  }
}

// Write updated registry
writeFileSync(registryPath, JSON.stringify(registry, null, 2) + '\n', 'utf8');

const total = Object.keys(registry).length;
const tested = Object.values(registry).filter(v => v.tested).length;
const pct = ((tested / total) * 100).toFixed(1);
console.log(`Registry updated: ${tested}/${total} APIs marked as tested (${pct}%)`);
