/**
 * torch.js - WebGPU-powered tensor library with PyTorch-compatible API.
 * @status partial
 */

import * as nn from './nn';
import * as optim from './optim';
import * as linalg from './linalg';
import * as fft from './fft';
import * as special from './special';
import * as webgpu from './webgpu';
import * as profiler from './profiler';
import * as benchmark from './utils/benchmark';

// Re-export tensor class and types
export { Tensor } from './tensor';
export type { TensorOptions, GradFn } from './tensor';
export type { DType, TypedArray } from './dtype';

// Re-export all operations
export * from './ops';

// Re-export modules
export { nn, optim, linalg, fft, special, webgpu, profiler };
import * as data from './utils/data';
export const utils = { benchmark, data };

// Re-export backend utilities
export { initWebGPU as init, syncDevice } from './backend';

// Re-export serialization types
export type { SaveFunc, LoadFunc } from './serialization/types';

// Debug utilities (stripped in production builds)
export { DEBUG, DEBUG_ASYNC, debugLog, warn, error, assert } from './debug';

// Internal exports for debugging (not part of public API)
import { getDevice, getOrCreatePipeline, calculateWorkgroups, readBuffer } from './backend';
export const _internals = {
  getDevice,
  getOrCreatePipeline,
  calculateWorkgroups,
  readBuffer,
};

// Import createTorch function
export { createTorch } from './create_torch';