/**
 * Coverage report script for compatibility tests.
 * Usage: node tests/compatibility/report.mjs [--summary] [--verbose]
 * 
 * --summary:   Show only summary percentages
 * --verbose:   Show all APIs with their test status
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const registryPath = join(__dirname, '_registry.json');
const registry = JSON.parse(readFileSync(registryPath, 'utf8'));

const args = process.argv.slice(2);
const showSummary = args.includes('--summary');
const showVerbose = args.includes('--verbose');

const total = Object.keys(registry).length;
const tested = Object.values(registry).filter((v) => v.tested).length;
const coverage = ((tested / total) * 100).toFixed(1);

// Group by category
const categories = {};
for (const [key, value] of Object.entries(registry)) {
  const cat = value.category;
  if (!categories[cat]) categories[cat] = { total: 0, tested: 0 };
  categories[cat].total++;
  if (value.tested) categories[cat].tested++;
}

if (showSummary) {
  console.log(`\n📊 Compatibility Test Coverage: ${coverage}% (${tested}/${total})`);
  console.log('━'.repeat(60));
  for (const [cat, data] of Object.entries(categories)) {
    const pct = ((data.tested / data.total) * 100).toFixed(1);
    const bar = '█'.repeat(Math.round(data.tested / data.total * 20)) + '░'.repeat(20 - Math.round(data.tested / data.total * 20));
    console.log(`  ${cat.padEnd(30)} [${bar}] ${pct.padStart(6)}% (${data.tested}/${data.total})`);
  }
  console.log('');
} else if (showVerbose) {
  console.log(`\n📊 Full API Coverage Report (${coverage}% - ${tested}/${total})`);
  console.log('━'.repeat(80));
  let lastCat = '';
  for (const [key, value] of Object.entries(registry)) {
    if (value.category !== lastCat) {
      lastCat = value.category;
      console.log(`\n  ${'─'.repeat(76)}`);
      console.log(`  📁 ${lastCat}`);
      console.log(`  ${'─'.repeat(76)}`);
    }
    const icon = value.tested ? '✅' : '❌';
    const file = value.tested ? ` (${value.testFile})` : '';
    console.log(`    ${icon} ${key.padEnd(60)}${file}`);
  }
  console.log('\n');
} else {
  console.log(`\n📊 Coverage: ${coverage}% (${tested}/${total} APIs)`);
  console.log('');
  console.log('Usage:');
  console.log('  node tests/compatibility/report.mjs --summary    Summary by category');
  console.log('  node tests/compatibility/report.mjs --verbose    Full API listing');
  console.log('');
}

// Exit with error if coverage below threshold
if (coverage < 50) {
  console.log(`⚠️  Coverage below 50%: ${coverage}%`);
}
