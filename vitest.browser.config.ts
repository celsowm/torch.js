import { defineConfig } from 'vitest/config';
import { readFileSync } from 'fs';
import { playwright } from '@vitest/browser-playwright';

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
    browser: {
      enabled: true,
      provider: playwright({
        launch: {
          args: ['--enable-unsafe-webgpu'],
        },
      }),
      headless: false,
      instances: [
        { browser: 'chromium' },
      ],
    },
    include: ['tests/**/*.test.ts'],
  },
});
