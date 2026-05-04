/**
 * List all untested APIs.
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const registry = JSON.parse(readFileSync(join(__dirname, '_registry.json'), 'utf8'));

const untested = Object.entries(registry)
  .filter(([_, v]) => !v.tested)
  .map(([name, v]) => ({ name, category: v.category }));

// Group by category
const byCat = {};
for (const { name, category } of untested) {
  if (!byCat[category]) byCat[category] = [];
  byCat[category].push(name);
}

for (const [cat, apis] of Object.entries(byCat)) {
  console.log(`\n📁 ${cat} (${apis.length} untested):`);
  for (const api of apis) {
    console.log(`  - ${api}`);
  }
}

console.log(`\n\nTotal untested: ${untested.length}`);
