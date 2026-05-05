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
export { Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d } from './conv';
export { MaxPool2d, AvgPool2d, AdaptiveMaxPool2d, FractionalMaxPool2d, LPPool2d, MaxUnpool2d } from './pooling';
export { MaxPool1d, AvgPool1d, MaxPool3d, AvgPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool3d } from './pool';
export { BatchNorm1d, BatchNorm2d } from './batchnorm';
export { InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, GroupNorm, RMSNorm, SyncBatchNorm1d, SyncBatchNorm2d, SyncBatchNorm3d, LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d } from './normalization';
export { RNN, LSTM, GRU } from './rnn';
export { Upsample, UpsamplingNearest2d, UpsamplingBilinear2d } from './upsampling';

// Transformer
export { 
  TransformerEncoder, 
  TransformerDecoder, 
  TransformerEncoderLayer, 
  TransformerDecoderLayer,
  type TransformerEncoderLayerParams,
  type TransformerDecoderLayerParams,
} from './transformer';

// Padding
export {
  ConstantPad1d,
  ConstantPad2d,
  ConstantPad3d,
  ZeroPad2d,
  ReflectionPad2d,
  ReplicationPad2d,
  Pad1d,
  Pad2d,
} from './padding';

// Utils
export { Unflatten, Fold, Unfold, PixelShuffle } from './utils';

// Containers
export { ModuleList, ModuleDict } from './containers';

// Activations
export { ReLU, Sigmoid, Tanh, LogSoftmax, GELU, Softmax, PReLU, CELU, RReLU, Hardtanh, Hardshrink, Softshrink, LogSigmoid, Softmin, LeakyReLU, ELU, SELU, Threshold, Softplus, Softsign, Tanhshrink, Mish, SiLU, Hardsigmoid, Hardswish, GLU, ReLU6, Softmax2d } from './activation';

// Attention
export { MultiheadAttention } from './attention';

// Regularization
export { Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout } from './dropout';

// Loss function classes
export {
  NLLLoss,
  CrossEntropyLoss,
  BCELoss,
  BCEWithLogitsLoss,
  HuberLoss,
  SmoothL1Loss,
  L1Loss,
  MSELoss,
  KLDivLoss,
  HingeEmbeddingLoss,
  CosineEmbeddingLoss,
  TripletMarginLoss,
  CTCLoss,
  MultiMarginLoss,
} from './losses';

// New loss functions not already in functional.ts
export {
  huber_loss,
  kl_div_loss,
  hinge_embedding_loss,
  cosine_embedding_loss,
  triplet_margin_loss,
  ctc_loss,
  multi_margin_loss,
} from './losses';

// Initialization
import * as init from './init';
export { init };

// Functional interface
import * as functional from './functional';
export { functional };
export { functional as F };

// Re-export common functional functions at top level for convenience
// Note: loss functions come from losses.ts, not functional.ts
export {
  relu, gelu, softmax, log_softmax,
  dropout,
  conv2d, max_pool2d, avg_pool2d,
  sigmoid, tanh, leaky_relu, elu, selu,
} from './functional';
