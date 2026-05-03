import { defineConfig } from 'tsup';
import { readFileSync } from 'fs';

const wgslPlugin = {
  name: 'wgsl-loader',
  setup(build: any) {
    build.onLoad({ filter: /\.wgsl$/ }, (args: any) => {
      const source = readFileSync(args.path, 'utf8');
      return {
        contents: `export default ${JSON.stringify(source)};`,
        loader: 'js',
      };
    });
  },
};

// Check if building in production mode (strips debug logging)
const isProduction = process.env.NODE_ENV === 'production';

const debugDefine = {
  __DEBUG__: isProduction ? 'false' : 'true',
};

// Production builds enable minification to strip dead code
const productionOptions = isProduction ? {
  minify: true,
  treeshake: true,
} : {};

export default defineConfig([
  // Main entries
  {
    entry: { 
      index: 'src/index.ts',
      core: 'src/core.ts',
      'serialization/common': 'src/serialization/common.ts'
    },
    format: ['esm'],
    dts: true,
    clean: true,
    outDir: 'dist',
    sourcemap: true,
    external: ['@torchjsorg/wgpu-native'],
    esbuildPlugins: [wgslPlugin],
    ...productionOptions,
    esbuildOptions(options) {
      options.define = {
        ...options.define,
        'process.env.IS_BROWSER': 'true',
        ...debugDefine,
      };
    },
  },
]);
