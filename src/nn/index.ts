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
export { Conv1d, Conv2d } from './conv';
export { MaxPool2d, AvgPool2d } from './pooling';
export { BatchNorm1d, BatchNorm2d } from './batchnorm';
export { RNN, LSTM, GRU } from './rnn';

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
export {
  relu, gelu, softmax, log_softmax,
  nll_loss, cross_entropy, dropout,
  mse_loss, binary_cross_entropy, binary_cross_entropy_with_logits,
  smooth_l1_loss, l1_loss,
  conv2d, max_pool2d, avg_pool2d,
  sigmoid, tanh, leaky_relu, elu, selu,
} from './functional';
