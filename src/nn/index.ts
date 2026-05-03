/**
 * Neural network module exports.
 * @status partial
 * @pytorch torch.nn
 */

// Base classes
export { Module } from './module';
export { Parameter } from './parameter';

// Layers
export { Linear } from './linear';
export { Sequential } from './sequential';
export { Flatten } from './flatten';
export { Embedding } from './embedding';
export { LayerNorm } from './layernorm';

// Containers
export { ModuleList, ModuleDict } from './containers';

// Activations
export { ReLU, Sigmoid, Tanh, LogSoftmax, GELU, Softmax } from './activation';

// Regularization
export { Dropout, Dropout2d } from './dropout';

// Functional interface
import * as functional from './functional';
export { functional };
export { functional as F };

// Re-export common functional functions at top level for convenience
export { relu, gelu, softmax, log_softmax, nll_loss, cross_entropy, dropout } from './functional';
