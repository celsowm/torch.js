import * as ops from './ops';
import * as nn from './nn';
import * as optim from './optim';
import * as linalg from './linalg';
import * as fft from './fft';
import { special } from './special';
import * as webgpu from './webgpu';
import * as autograd from './autograd';
import * as distributions from './distributions';
import * as sparse from './sparse';
import { conv2d, conv1d, max_pool2d, avg_pool2d } from './nn/functional';
import { initWebGPU, syncDevice as _syncDevice } from './backend';
import { no_grad, enable_grad, inference_mode, is_grad_enabled } from './grad_mode';
import type { SaveFunc, LoadFunc } from './serialization/types';
import { loadPyTorchZip } from './serialization/pytorch_loader';

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

    // Explicit PyTorch .pt loading
    loadPyTorch: loadPyTorchZip,

    // Operations (spread all ops)
    ...ops,

    // Linear Algebra module
    linalg,

    // Neural network module
    nn,

    // Optimizer module
    optim,

    // Autograd module
    autograd,

    // Distributions
    distributions,

    // Sparse tensors
    sparse,

    // WebGPU/Memory management
    webgpu,

    // Autograd context managers (also accessible as torch.no_grad etc.)
    no_grad,
    enable_grad,
    inference_mode,
    is_grad_enabled,

    // FFT module
    fft,

    // Special math functions
    special,

    // Convenience re-exports from nn.functional
    conv2d,
    conv1d,
    max_pool2d,
    avg_pool2d,
  };
}