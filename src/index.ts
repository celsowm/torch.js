import { createTorch } from './create_torch';
import { save, load } from './serialization/browser';
import * as _core from './core';

const _torchValue = createTorch(save, load);

/**
 * Main torch object.
 */
const torch = {
  ..._core,
  ..._torchValue,
  webgpu: _core.webgpu,
  utils: _core.utils,
  profiler: _core.profiler,
};

// Ensure sub-modules are attached correctly if they weren't in the spread
(torch as any).utils = _core.utils;
(torch as any).profiler = _core.profiler;
(torch as any).webgpu = _core.webgpu;

export default torch;
// Also export torch as a named export so `import * as torch` works
export { torch };
export * from './core';
export {
  getDevice,
  getOrCreatePipeline,
  calculateWorkgroups,
  getCapabilities,
  BufferUsage,
  readBuffer
} from './backend';
export type { Tensor, DType, TensorOptions } from './core';
