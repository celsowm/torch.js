import { defineConfig } from 'vitest/config';
import { readFileSync } from 'fs';

export default defineConfig({
  plugins: [
    {
      name: 'wgsl-loader',
      transform(code, id) {
        if (id.endsWith('.wgsl')) {
          const source = readFileSync(id, 'utf8');
          return {
            code: `export default ${JSON.stringify(source)};`,
            map: null,
          };
        }
      },
    },
  ],
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});