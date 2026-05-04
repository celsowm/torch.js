import * as ops from './ops';
import * as nn from './nn';
import * as optim from './optim';
import * as linalg from './linalg';
import * as webgpu from './webgpu';
import { initWebGPU, syncDevice as _syncDevice } from './backend';
import { no_grad, enable_grad, inference_mode, is_grad_enabled } from './grad_mode';
import type { SaveFunc, LoadFunc } from './serialization/types';

/**
 * Creates the torch object with specific serialization implementations.
 */
export function createTorch(save: SaveFunc, load: LoadFunc) {
  return {
    // Initialization
    init: initWebGPU,

    // Serialization
    save,
    load,

    // Operations (spread all ops)
    ...ops,

    // Linear Algebra module
    linalg,

    // Synchronization
    syncDevice: _syncDevice,

    // Neural network module
    nn,

    // Optimizer module
    optim,

    // WebGPU/Memory management
    webgpu,

    // Autograd context managers
    no_grad,
    enable_grad,
    inference_mode,
    is_grad_enabled,
  };
}